/*
 * train_janus_sonar.c — Janus Sonar SINGLE 3M on notorch, pure C.
 *
 * Architecture (v2 single — dual removed because α did not specialize
 * on 241KB corpus at 2.25M; single-longer beat dual-short on train loss):
 *   VOCAB 2048 BPE, CTX 128, DIM 160, HEADS 5, HEAD_DIM 32,
 *   LAYERS 6, HIDDEN 320, RoPE on QK, RMSNorm, SwiGLU FFN.
 *
 *   ~3.11M parameters — Karpathy sanity range for 231KB clean corpus.
 *
 * Triple attention per layer (equal 1/3 blend; learnable gate[H,3] next round):
 *   a) MH causal (Q K V)              — semantic
 *   b) RRPRAM positional (Wr · x, Vr) — structural
 *   c) Janus echo MH (echo·echo·echo) — introspective self-resonance, echo = Wj^T · x
 *
 * Dataset: ../dataset_clean.txt (231K after speaker-tag strip, 387 literary em-dashes kept).
 * Tokenizer: arianna_bpe_merges.txt (vocab 2048).
 *
 *   make train_janus_sonar
 *   ./train_janus_sonar [steps] [lr]              # from scratch
 *   ./train_janus_sonar --resume [steps] [lr]     # resume from checkpoint
 */
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define DIM       160
#define NLAYERS   6
#define NHEADS    5
#define HEAD_DIM  32              /* DIM / NHEADS */
#define HIDDEN    320
#define CTX       128
#define VOCAB     2048

#define LOG_EVERY   50
#define CKPT_EVERY  1000
#define EVAL_SEQS   16
#define CKPT_PREFIX "sonar_single_v2_ckpt"

typedef struct {
    nt_tensor *wte;                                      /* [VOCAB, DIM] */
    struct {
        nt_tensor *rms1;                                 /* [DIM] */
        nt_tensor *wq, *wk, *wv, *wvr, *wj, *wo;         /* [DIM, DIM] */
        nt_tensor *wr;                                   /* [NHEADS*DIM, CTX] — RRPRAM positional */
        nt_tensor *rms2;
        nt_tensor *w_gate, *w_up;                        /* [HIDDEN, DIM] */
        nt_tensor *w_down;                               /* [DIM, HIDDEN] */
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < NLAYERS; l++) {
        n += m->L[l].rms1->len + m->L[l].rms2->len + m->L[l].wr->len;
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].wv->len;
        n += m->L[l].wvr->len + m->L[l].wj->len + m->L[l].wo->len;
        n += m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    }
    return n;
}

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->wte, VOCAB, DIM);
    float rs = 0.02f / sqrtf(2.0f * NLAYERS);
    float out_scale_attn = rs / 0.1f;
    float out_scale_ffn  = rs / 0.1f;
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq   = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq,  DIM, DIM);
        m->L[l].wk   = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wk,  DIM, DIM);
        m->L[l].wv   = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wv,  DIM, DIM);
        m->L[l].wvr  = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wvr, DIM, DIM);
        m->L[l].wj   = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wj,  DIM, DIM);
        m->L[l].wo   = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo,  DIM, DIM);
        for (int i = 0; i < m->L[l].wo->len; i++) m->L[l].wo->data[i] *= out_scale_attn;
        m->L[l].wr   = nt_tensor_new2d(NHEADS * DIM, CTX);
        nt_tensor_xavier(m->L[l].wr, DIM, CTX);
        m->L[l].rms2 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_gate, DIM, HIDDEN);
        m->L[l].w_up   = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_up,   DIM, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN); nt_tensor_xavier(m->L[l].w_down, HIDDEN, DIM);
        for (int i = 0; i < m->L[l].w_down->len; i++) m->L[l].w_down->data[i] *= out_scale_ffn;
    }
    m->rms_f = nt_tensor_new(DIM); nt_tensor_fill(m->rms_f, 1.0f);
    m->head  = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->head, DIM, VOCAB);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq);   nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv);   nt_tensor_free(m->L[l].wvr);
        nt_tensor_free(m->L[l].wj);   nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].wr);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

/* 12 tensors per layer: rms1, 6 attn, wr, rms2, 3 ffn */
static int model_n_tensors(void) { return 1 + NLAYERS * 12 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++] = m->L[l].rms1;
        p[i++] = m->L[l].wq;   p[i++] = m->L[l].wk;   p[i++] = m->L[l].wv;
        p[i++] = m->L[l].wvr;  p[i++] = m->L[l].wj;   p[i++] = m->L[l].wo;
        p[i++] = m->L[l].wr;
        p[i++] = m->L[l].rms2;
        p[i++] = m->L[l].w_gate; p[i++] = m->L[l].w_up; p[i++] = m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static void save_model(Model* m, const char* prefix) {
    char path[256]; snprintf(path, sizeof(path), "%s.bin", prefix);
    nt_tensor** p = model_param_array(m);
    nt_save(path, p, model_n_tensors());
    free(p);
}

static void save_checkpoint(Model* m, int step, float best) {
    save_model(m, CKPT_PREFIX);
    char mp[256]; snprintf(mp, sizeof(mp), "%s.meta", CKPT_PREFIX);
    FILE* f = fopen(mp, "w");
    if (f) { fprintf(f, "%d\n%.6f\n", step, best); fclose(f); }
}

static int load_checkpoint(Model* m, float* best_loss) {
    char wp[256], mp[256];
    snprintf(wp, sizeof(wp), "%s.bin", CKPT_PREFIX);
    snprintf(mp, sizeof(mp), "%s.meta", CKPT_PREFIX);
    int n = 0;
    nt_tensor** loaded = nt_load(wp, &n);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n != expected) {
        for (int i = 0; i < n; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    nt_tensor** dst = model_param_array(m);
    for (int i = 0; i < expected; i++) {
        memcpy(dst[i]->data, loaded[i]->data, dst[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(dst);
    int step = 0; *best_loss = 99.0f;
    FILE* f = fopen(mp, "r");
    if (f) { fscanf(f, "%d\n%f\n", &step, best_loss); fclose(f); }
    return step;
}

/* ── Forward ── */

static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    struct {
        int rms1;
        int wq, wk, wv, wvr, wj, wo, wr;
        int rms2;
        int w_gate, w_up, w_down;
    } li[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1   = nt_tape_param(m->L[l].rms1);
        li[l].wq     = nt_tape_param(m->L[l].wq);
        li[l].wk     = nt_tape_param(m->L[l].wk);
        li[l].wv     = nt_tape_param(m->L[l].wv);
        li[l].wvr    = nt_tape_param(m->L[l].wvr);
        li[l].wj     = nt_tape_param(m->L[l].wj);
        li[l].wo     = nt_tape_param(m->L[l].wo);
        li[l].wr     = nt_tape_param(m->L[l].wr);
        li[l].rms2   = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = nt_tape_param(m->L[l].w_gate);
        li[l].w_up   = nt_tape_param(m->L[l].w_up);
        li[l].w_down = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);

    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, DIM);

        int q   = nt_seq_linear  (li[l].wq,  xn, CTX);
        int k   = nt_seq_linear  (li[l].wk,  xn, CTX);
        int v   = nt_seq_linear  (li[l].wv,  xn, CTX);
        int vr  = nt_seq_linear  (li[l].wvr, xn, CTX);
        int ech = nt_seq_linear_t(li[l].wj,  xn, CTX);

        q = nt_rope(q, CTX, HEAD_DIM);
        k = nt_rope(k, CTX, HEAD_DIM);

        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);

        int blend = nt_add(nt_add(a_qkv, a_rr), a_j);
        blend = nt_scale(blend, 1.0f / 3.0f);

        int proj = nt_seq_linear(li[l].wo, blend, CTX);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, DIM);
        int g = nt_silu(nt_seq_linear(li[l].w_gate, xn, CTX));
        int u =         nt_seq_linear(li[l].w_up,   xn, CTX);
        int d =         nt_seq_linear(li[l].w_down, nt_mul(g, u), CTX);
        h = nt_add(h, d);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, VOCAB);
}

/* ── Eval ── */

static float eval_loss(Model* m, int* encoded, int n_tokens) {
    float total = 0; int count = 0;
    int stride = n_tokens / EVAL_SEQS;
    if (stride < 1) stride = 1;
    for (int s = 0; s < EVAL_SEQS; s++) {
        int off = s * stride;
        if (off + CTX + 1 > n_tokens) break;
        nt_tape_start();
        nt_train_mode(0);
        int loss_idx = forward(m, encoded + off, encoded + off + 1);
        total += nt_tape_get()->entries[loss_idx].output->data[0];
        count++;
        nt_tape_clear();
        nt_train_mode(1);
    }
    return count > 0 ? total / count : 99.0f;
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; }

int main(int argc, char** argv) {
    int resume = 0, ao = 1;
    if (argc > 1 && strcmp(argv[1], "--resume") == 0) { resume = 1; ao = 2; }
    int steps = ao < argc ? atoi(argv[ao]) : 10000;
    float base_lr = (ao+1) < argc ? (float)atof(argv[ao+1]) : 3e-4f;

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch — JANUS SONAR SINGLE v2 training\n");
    printf("  DIM=%d L=%d H=%d HD=%d FFN=%d CTX=%d V=%d\n",
           DIM, NLAYERS, NHEADS, HEAD_DIM, HIDDEN, CTX, VOCAB);
    printf("  Triple attn: MHA + RRPRAM + Janus Echo (equal 1/3 blend)\n");
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("════════════════════════════════════════════════════════\n");

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, "arianna_bpe_merges.txt");
    if (nm < 0) { printf("cannot load arianna_bpe_merges.txt\n"); return 1; }
    printf("bpe: %d merges, vocab %d\n", bpe.n_merges, bpe.vocab_size);

    const char* path = "../dataset_clean.txt";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fsize + 1);
    fread(raw, 1, fsize, f); raw[fsize] = 0; fclose(f);

    int* encoded = (int*)malloc(fsize * sizeof(int));
    int n_tokens = nt_bpe_encode(&bpe, raw, (int)fsize, encoded, (int)fsize);
    free(raw);
    printf("corpus: %.1f KB → %d BPE tokens\n", fsize/1024.0, n_tokens);

    nt_seed(42);
    Model* model = model_new();
    long np = count_params(model);
    printf("model: %ld params (%.2f MB) — single\n", np, np*4.0f/1048576.0f);
    printf("karpathy: %.1f epochs over %d steps\n", (float)steps * CTX / n_tokens, steps);

    float best_loss = 99.0f;
    if (resume) {
        int loaded = load_checkpoint(model, &best_loss);
        if (loaded >= 0) printf("RESUMED from step %d (best=%.4f)\n", loaded, best_loss);
        else printf("resume requested but no checkpoint — starting fresh\n");
    }

    nt_schedule sched = nt_schedule_cosine(base_lr, steps/10, steps, base_lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\ntraining...\n");
    printf("─────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float first_loss = 0;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        int off = rand() % (n_tokens - CTX - 1);

        nt_tape_start();
        int loss_idx = forward(model, encoded + off, encoded + off + 1);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) first_loss = lv;
        if (lv < best_loss) best_loss = lv;

        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();

        if ((step+1) % LOG_EVERY == 0 || step == 0) {
            printf("  step %5d/%d | train %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step+1, steps, lv, best_loss, lr, (now_ms()-t0)/1000.0);
            fflush(stdout);
        }

        if ((step+1) % CKPT_EVERY == 0) {
            float val = eval_loss(model, encoded, n_tokens);
            printf("  ──── ckpt %d | val %.4f\n", step+1, val);
            save_checkpoint(model, step+1, best_loss);
            fflush(stdout);
        }
    }

    float final_val = eval_loss(model, encoded, n_tokens);
    double total_s = (now_ms()-t0)/1000.0;

    printf("─────────────────────────────────────────────────────\n");
    printf("  train: %.4f → best %.4f\n", first_loss, best_loss);
    printf("  val:   %.4f\n", final_val);
    printf("  time:  %.0fs (%.1f min) | %.2f steps/s\n", total_s, total_s/60.0, steps/total_s);
    printf("  nans:  %d\n", guard.total_nan_count);

    printf("\n── saving ──\n");
    save_model(model, "sonar_single_v2");
    printf("  sonar_single_v2.bin (%.2f MB)\n", np*4.0f/1048576.0f);
    save_checkpoint(model, steps, best_loss);

    model_free(model); free(encoded);
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Janus Sonar SINGLE v2 trained. %d steps. Pure C. No PyTorch.\n", steps);
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
