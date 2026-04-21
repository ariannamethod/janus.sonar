/*
 * infer_janus_sonar_chain.c — Janus bidirectional chain inference.
 *
 * 8-step chain with calendar-drift compass (forward vs backward ratio),
 * Schumann resonance temperature modulation, best-of-5 per step,
 * destiny EMA across the chain, and SPA (Sentence Phonon Attention)
 * reseed of the weakest sentence at the end.
 *
 * Supports both legacy dual weights and the 3M single-weight Sonar line.
 * Coherence scored by unique-token ratio + closure critic + motif ledger;
 * corpus statistics and AML chamber state are applied during emission.
 *
 *   make infer_janus_sonar_chain
 *   ./infer_janus_sonar_chain janus_sonar.bin "seed text" [rng_seed] [mode]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define DIM       160
#define NLAYERS   6
#define NHEADS    5
#define HEAD_DIM  32
#define HIDDEN    320
#define CTX       128
#define VOCAB     2048

#define CHAIN_STEPS    8       /* 3 backward + 5 forward typical */
#define SENT_MAX       200     /* absolute buffer cap */
#define SENT_MIN_LEN   8       /* allow short sentences, break on first */
#define SENT_MAX_SOFT  40      /* force boundary if reached without one */
#define CAND_N         5       /* best-of-N candidates per step */
#define REP_WINDOW     64
#define REP_PENALTY    0.65f

typedef enum {
    MODE_BALANCED = 0,
    MODE_COHERENT,
    MODE_RITUAL,
    MODE_CLINICAL,
    MODE_DIALOGUE,
    MODE_COUNT
} GenMode;

typedef struct {
    const char* name;
    float temp_shift;
    float top_p;
    float closure_w;
    float motif_w;
    float destiny_bias;
    float entropy_floor;
    float resonance_ceiling;
    float pain;
} ModeCfg;

static const ModeCfg MODE_CFG[MODE_COUNT] = {
    {"balanced",  0.00f, 0.95f, 0.55f, 0.28f, 0.20f, 0.10f, 0.95f, 0.00f},
    {"coherent", -0.08f, 0.90f, 0.95f, 0.18f, 0.27f, 0.16f, 0.88f, 0.00f},
    {"ritual",   0.05f, 0.97f, 0.45f, 0.70f, 0.18f, 0.12f, 0.97f, 0.02f},
    {"clinical", -0.05f, 0.92f, 0.80f, 0.36f, 0.25f, 0.14f, 0.90f, 0.00f},
    {"dialogue", -0.02f, 0.94f, 0.82f, 0.34f, 0.22f, 0.13f, 0.91f, 0.00f}
};

static GenMode g_mode = MODE_BALANCED;

/* ── AML physics state (field) ── */
/* Port of core/ariannamethod.c logit transformations: destiny, suffering,
   laws (entropy floor + resonance ceiling), prophecy debt accumulation.
   All six Kuramoto chambers modulate force coefficients in real time. */
typedef struct {
    float destiny_bias;        /* [0,1] — max-suppression strength */
    float pain;                /* [0,1] — compresses toward mean */
    float entropy_floor;       /* [0,1] — cap on max-vs-second gap */
    float resonance_ceiling;   /* [0,1] — additional gap cap */
    float prophecy_debt;       /* accumulated: (max-chosen)/(+1) per step */
    float debt_decay;          /* per-step multiplier on debt */
    /* Chambers: Kuramoto 6-oscillator ring */
    float ch_act[6];
    float ch_decay[6];
    float ch_coup[6][6];
} AMLField;

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_CMPLX };

static void aml_init(AMLField* f) {
    memset(f, 0, sizeof(*f));
    f->destiny_bias     = 0.20f;
    f->pain             = 0.0f;
    f->entropy_floor    = 0.10f;
    f->resonance_ceiling= 0.95f;
    f->prophecy_debt    = 0.0f;
    f->debt_decay       = 0.998f;
    /* Chamber initial activations (LOVE + FLOW slight bias — ready posture) */
    f->ch_act[CH_LOVE] = 0.2f; f->ch_act[CH_FLOW] = 0.15f;
    static const float decay[6] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};
    memcpy(f->ch_decay, decay, sizeof(decay));
    /* Coupling matrix (antisymmetric-ish) from core/ariannamethod.c */
    static const float coup[6][6] = {
        { 0.00f,-0.30f, 0.50f, 0.40f,-0.20f, 0.10f},
        {-0.30f, 0.00f,-0.40f,-0.50f, 0.50f, 0.20f},
        { 0.50f,-0.30f, 0.00f, 0.20f,-0.30f, 0.30f},
        { 0.40f,-0.50f, 0.30f, 0.00f,-0.30f, 0.40f},
        {-0.20f, 0.40f,-0.20f,-0.30f, 0.00f, 0.30f},
        { 0.10f, 0.20f, 0.30f, 0.40f, 0.30f, 0.00f}
    };
    memcpy(f->ch_coup, coup, sizeof(coup));
}

static void aml_apply_mode(AMLField* f, const ModeCfg* cfg) {
    f->destiny_bias = cfg->destiny_bias;
    f->entropy_floor = cfg->entropy_floor;
    f->resonance_ceiling = cfg->resonance_ceiling;
    f->pain = cfg->pain;
    if (!strcmp(cfg->name, "coherent")) {
        f->ch_act[CH_FLOW] += 0.18f;
        f->ch_act[CH_CMPLX] += 0.10f;
    } else if (!strcmp(cfg->name, "ritual")) {
        f->ch_act[CH_LOVE] += 0.22f;
        f->ch_act[CH_VOID] += 0.18f;
        f->ch_act[CH_CMPLX] += 0.12f;
    } else if (!strcmp(cfg->name, "clinical")) {
        f->ch_act[CH_FLOW] += 0.20f;
        f->ch_act[CH_CMPLX] += 0.18f;
        f->ch_act[CH_LOVE] *= 0.5f;
    } else if (!strcmp(cfg->name, "dialogue")) {
        f->ch_act[CH_LOVE] += 0.16f;
        f->ch_act[CH_FLOW] += 0.12f;
    }
}

/* Kuramoto crossfire step: act[i] += K·sum_j(coup[i][j]·sin(act[j]-act[i])), then decay */
static void aml_chambers_xfire(AMLField* f, int iters) {
    for (int t = 0; t < iters; t++) {
        float old[6]; memcpy(old, f->ch_act, sizeof(old));
        for (int i = 0; i < 6; i++) {
            f->ch_act[i] *= f->ch_decay[i];
            for (int j = 0; j < 6; j++)
                if (i != j) f->ch_act[i] += 0.03f * f->ch_coup[i][j] * sinf(old[j] - old[i]);
            if (f->ch_act[i] < 0) f->ch_act[i] = 0;
            if (f->ch_act[i] > 1) f->ch_act[i] = 1;
        }
    }
}

/* Destiny: suppress below-max logits by (max - logits[i]) * bias * 0.5
   Result: distribution becomes more peaked around the current dominant token. */
static void aml_apply_destiny(float* logits, int n, float bias) {
    if (n <= 0 || bias < 0.001f) return;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) logits[i] -= (mx - logits[i]) * bias * 0.5f;
}

/* Suffering: pain compresses logits toward mean (blunts both peaks and valleys) */
static void aml_apply_suffering(float* logits, int n, float pain) {
    if (n <= 0 || pain < 0.01f) return;
    float mean = 0; for (int i = 0; i < n; i++) mean += logits[i]; mean /= (float)n;
    float factor = 1.0f - 0.5f * pain;
    for (int i = 0; i < n; i++) logits[i] = mean + (logits[i] - mean) * factor;
}

/* Laws: entropy floor + resonance ceiling — cap on max-vs-second gap */
static void aml_apply_laws(float* logits, int n, float ent_floor, float res_ceil) {
    if (n <= 0) return;
    float mx = logits[0], sec = -1e30f;
    for (int i = 1; i < n; i++) {
        if (logits[i] > mx) { sec = mx; mx = logits[i]; }
        else if (logits[i] > sec) sec = logits[i];
    }
    float gap = mx - sec;
    if (gap > 0 && ent_floor > 0) {
        float max_gap = (1.0f - ent_floor) * 10.0f;
        if (gap > max_gap) {
            float reduce = (gap - max_gap) * 0.5f;
            for (int i = 0; i < n; i++) if (logits[i] == mx) logits[i] -= reduce;
        }
    }
    if (res_ceil < 1.0f) {
        float ceiling_gap = res_ceil * 10.0f;
        float new_gap = mx - sec;
        if (new_gap > ceiling_gap) {
            float reduce = (new_gap - ceiling_gap) * 0.3f;
            for (int i = 0; i < n; i++) if (logits[i] >= mx - 0.001f) logits[i] -= reduce;
        }
    }
}

/* Prophecy debt contribution of a choice: (max - chosen) / (diff + 1) */
static float aml_prophecy_debt(const float* logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0 ? diff / (diff + 1.0f) : 0;
}

/* Apply full field pipeline + update prophecy debt after choice.
   Chamber modulation: destiny bias amplified by VOID, suppressed by FEAR.
                       pain amplified by RAGE.
                       Laws always on. */
static void aml_apply_field(float* logits, int n, const AMLField* f) {
    float dest = f->destiny_bias * (1.0f + 0.6f*f->ch_act[CH_VOID] - 0.3f*f->ch_act[CH_FEAR]);
    float pain = f->pain + 0.5f*f->ch_act[CH_RAGE];
    if (dest < 0) dest = 0; if (dest > 1) dest = 1;
    if (pain < 0) pain = 0; if (pain > 1) pain = 1;
    aml_apply_destiny(logits, n, dest);
    aml_apply_suffering(logits, n, pain);
    aml_apply_laws(logits, n, f->entropy_floor, f->resonance_ceiling);
}

/* ── SPA (Sentence Phonon Attention) ── */
#define SPA_DIM   32

typedef struct { nt_tensor *a, *b, *alpha; } DualProj;

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        DualProj wq, wk, wv, wvr, wj, wo;
        nt_tensor *wr, *rms2;
        DualProj w_gate, w_up, w_down;
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

#define N_TENSORS_DUAL   (1 + NLAYERS * 30 + 2)   /* 9 duals × 3 + rms1 + rms2 + wr */
#define N_TENSORS_SINGLE (1 + NLAYERS * 12 + 2)   /* 9 singles + rms1 + rms2 + wr */
static int model_n_tensors(void) { return N_TENSORS_DUAL; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1;
        DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                              &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            p[i++] = projs[k]->a; p[i++] = projs[k]->b; p[i++] = projs[k]->alpha;
        }
        p[i++] = m->L[l].wr; p[i++] = m->L[l].rms2;
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            p[i++] = ffn[k]->a; p[i++] = ffn[k]->b; p[i++] = ffn[k]->alpha;
        }
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

/* Wrap a single matrix as DualProj: A = W, B = zeros, α = +large (σ≈1).
   Effective W_eff = W exactly — single model runs through dual code. */
static void dual_from_single(DualProj* d, nt_tensor* W, int rows, int cols) {
    d->a = W;                                   /* reuse W as A */
    d->b = nt_tensor_new2d(rows, cols);         /* zeros */
    d->alpha = nt_tensor_new(1);
    d->alpha->data[0] = 20.0f;                   /* σ(20) ≈ 1.0 → pure A */
}

static Model* load_model(const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load %s\n", path); return NULL; }

    Model* m = (Model*)calloc(1, sizeof(Model));

    if (n_loaded == N_TENSORS_DUAL) {
        /* Dual format — load directly */
        printf("format: dual (%d tensors)\n", n_loaded);
        int i = 0;
        m->wte = loaded[i++];
        for (int l = 0; l < NLAYERS; l++) {
            m->L[l].rms1 = loaded[i++];
            DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                                  &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
            for (int k = 0; k < 6; k++) {
                projs[k]->a = loaded[i++]; projs[k]->b = loaded[i++]; projs[k]->alpha = loaded[i++];
            }
            m->L[l].wr = loaded[i++]; m->L[l].rms2 = loaded[i++];
            DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
            for (int k = 0; k < 3; k++) {
                ffn[k]->a = loaded[i++]; ffn[k]->b = loaded[i++]; ffn[k]->alpha = loaded[i++];
            }
        }
        m->rms_f = loaded[i++]; m->head = loaded[i++];
    } else if (n_loaded == N_TENSORS_SINGLE) {
        /* Single format — promote to dual with W_B=0, α=20 (σ≈1) */
        printf("format: single (%d tensors) — loading via single→dual adapter\n", n_loaded);
        int i = 0;
        m->wte = loaded[i++];
        for (int l = 0; l < NLAYERS; l++) {
            m->L[l].rms1 = loaded[i++];
            nt_tensor* wq  = loaded[i++]; nt_tensor* wk = loaded[i++]; nt_tensor* wv = loaded[i++];
            nt_tensor* wvr = loaded[i++]; nt_tensor* wj = loaded[i++]; nt_tensor* wo = loaded[i++];
            m->L[l].wr   = loaded[i++];
            m->L[l].rms2 = loaded[i++];
            nt_tensor* wg = loaded[i++]; nt_tensor* wu = loaded[i++]; nt_tensor* wd = loaded[i++];
            dual_from_single(&m->L[l].wq,  wq,  DIM, DIM);
            dual_from_single(&m->L[l].wk,  wk,  DIM, DIM);
            dual_from_single(&m->L[l].wv,  wv,  DIM, DIM);
            dual_from_single(&m->L[l].wvr, wvr, DIM, DIM);
            dual_from_single(&m->L[l].wj,  wj,  DIM, DIM);
            dual_from_single(&m->L[l].wo,  wo,  DIM, DIM);
            dual_from_single(&m->L[l].w_gate, wg, HIDDEN, DIM);
            dual_from_single(&m->L[l].w_up,   wu, HIDDEN, DIM);
            dual_from_single(&m->L[l].w_down, wd, DIM, HIDDEN);
        }
        m->rms_f = loaded[i++]; m->head = loaded[i++];
    } else {
        printf("tensor mismatch: got %d, expected %d (dual) or %d (single)\n",
               n_loaded, N_TENSORS_DUAL, N_TENSORS_SINGLE);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); free(m);
        return NULL;
    }
    free(loaded);
    return m;
}

/* ═══════════════════════════════════════════════════════════════════
 * OPTIMIZED INFERENCE — incremental forward_step + KV cache, no tape
 * ═══════════════════════════════════════════════════════════════════
 * Earlier forward_logits ran full CTX=128 training-mode forward
 * with tape bookkeeping per emitted token. ~15K tape ops per gen,
 * 127 of 128 rows discarded. Measured: 89 sec per 200-tok sentence.
 *
 * forward_step: single-token forward via direct nt_blas_mm{,T}, dual
 * weights pre-blended once at load, K/V/Echo/Vr/Xn caches per layer.
 * RRPRAM position-keyed (Wr[:, j]) — incremental naturally. Target
 * speedup: 10-20×.
 * ═══════════════════════════════════════════════════════════════════ */

static float* W_eff_qkvj[NLAYERS][6];   /* wq wk wv wvr wj wo — all [DIM,DIM] */
static float* W_eff_ffn[NLAYERS][3];    /* w_gate, w_up [HIDDEN,DIM]; w_down [DIM,HIDDEN] */

static float K_cache[NLAYERS][CTX][DIM];
static float V_cache[NLAYERS][CTX][DIM];
static float E_cache[NLAYERS][CTX][DIM];
static float Vr_cache[NLAYERS][CTX][DIM];

static float sigmoid_f(float x) {
    return (x >= 0) ? 1.0f / (1.0f + expf(-x))
                    : expf(x) / (1.0f + expf(x));
}

static void blend_dual(float* out, const DualProj* d, int n_elem) {
    float alpha = d->alpha->data[0];
    float sp = sigmoid_f(alpha), sn = 1.0f - sp;
    for (int i = 0; i < n_elem; i++)
        out[i] = sp * d->a->data[i] + sn * d->b->data[i];
}

static void precompute_w_eff(Model* m) {
    for (int l = 0; l < NLAYERS; l++) {
        DualProj* projs[6] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                               &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            W_eff_qkvj[l][k] = (float*)malloc(DIM * DIM * sizeof(float));
            blend_dual(W_eff_qkvj[l][k], projs[k], DIM * DIM);
        }
        W_eff_ffn[l][0] = (float*)malloc(HIDDEN * DIM * sizeof(float));
        W_eff_ffn[l][1] = (float*)malloc(HIDDEN * DIM * sizeof(float));
        W_eff_ffn[l][2] = (float*)malloc(DIM * HIDDEN * sizeof(float));
        blend_dual(W_eff_ffn[l][0], &m->L[l].w_gate, HIDDEN * DIM);
        blend_dual(W_eff_ffn[l][1], &m->L[l].w_up,   HIDDEN * DIM);
        blend_dual(W_eff_ffn[l][2], &m->L[l].w_down, DIM * HIDDEN);
    }
}

static void rms_inplace(float* x, const float* gamma, int D) {
    float ss = 0;
    for (int d = 0; d < D; d++) ss += x[d] * x[d];
    float inv = 1.0f / sqrtf(ss / D + 1e-6f);
    if (gamma) for (int d = 0; d < D; d++) x[d] = x[d] * inv * gamma[d];
    else       for (int d = 0; d < D; d++) x[d] *= inv;
}

static void rope_tok_inplace(float* x, int pos, int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; h++) {
        int base = h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float x0 = x[base + 2*i], x1 = x[base + 2*i + 1];
            x[base + 2*i]     = x0 * c - x1 * s;
            x[base + 2*i + 1] = x0 * s + x1 * c;
        }
    }
}

/* Multi-head causal attention step: query q_new attends to K_cache[0..t], V_cache[0..t]. */
static void mha_step(const float* q_new,
                     const float* Kc, const float* Vc,
                     int t, int n_heads, int head_dim, float* out) {
    int D = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    float scores[CTX];
    for (int h = 0; h < n_heads; h++) {
        int ho = h * head_dim;
        float mx = -1e30f;
        for (int j = 0; j <= t; j++) {
            const float* kj = Kc + j * D + ho;
            float dot = 0;
            for (int d = 0; d < head_dim; d++) dot += q_new[ho + d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0;
        for (int j = 0; j <= t; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        float inv_s = sum > 0 ? 1.0f / sum : 0;
        for (int d = 0; d < head_dim; d++) out[ho + d] = 0;
        for (int j = 0; j <= t; j++) {
            float a = scores[j] * inv_s;
            const float* vj = Vc + j * D + ho;
            for (int d = 0; d < head_dim; d++) out[ho + d] += a * vj[d];
        }
    }
}

/* RRPRAM step: scores[j] = <xn, Wr[:, j]> for j in 0..t (position-indexed keys). */
static void rrpram_step(const float* xn, const float* Wr, const float* Vrc,
                        int t, int ctx_max, int n_heads, int n_embd, int head_dim,
                        float* out) {
    int D = n_heads * head_dim;
    float scores[CTX];
    for (int h = 0; h < n_heads; h++) {
        int wr_base = h * n_embd * ctx_max;
        int ho = h * head_dim;
        float mx = -1e30f;
        for (int j = 0; j <= t; j++) {
            float dot = 0;
            for (int d = 0; d < n_embd; d++)
                dot += xn[d] * Wr[wr_base + d * ctx_max + j];
            scores[j] = dot;
            if (dot > mx) mx = dot;
        }
        float sum = 0;
        for (int j = 0; j <= t; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        float inv_s = sum > 0 ? 1.0f / sum : 0;
        for (int d = 0; d < head_dim; d++) out[ho + d] = 0;
        for (int j = 0; j <= t; j++) {
            float a = scores[j] * inv_s;
            const float* vj = Vrc + j * D + ho;
            for (int d = 0; d < head_dim; d++) out[ho + d] += a * vj[d];
        }
    }
}

/* Incremental forward for one token at `pos`. Writes [VOCAB] into `logits`.
   Caches K/V/Echo/Vr updated in place for subsequent calls at pos+1, pos+2, ... */
static void forward_step(Model* m, int new_tok, int pos, float* logits) {
    float h_buf[DIM];
    int tok = new_tok < 0 ? 0 : (new_tok >= VOCAB ? VOCAB - 1 : new_tok);
    for (int d = 0; d < DIM; d++) h_buf[d] = m->wte->data[tok * DIM + d];

    float xn[DIM], q[DIM], k[DIM], v[DIM], vr[DIM], ech[DIM];
    float a_qkv[DIM], a_rr[DIM], a_j[DIM], blend[DIM], proj[DIM];
    float xn2[DIM], g_buf[HIDDEN], u_buf[HIDDEN], gu[HIDDEN], d_buf[DIM];

    for (int l = 0; l < NLAYERS; l++) {
        for (int d = 0; d < DIM; d++) xn[d] = h_buf[d];
        rms_inplace(xn, m->L[l].rms1->data, DIM);

        nt_blas_mmT(q,  xn, W_eff_qkvj[l][0], 1, DIM, DIM);
        nt_blas_mmT(k,  xn, W_eff_qkvj[l][1], 1, DIM, DIM);
        nt_blas_mmT(v,  xn, W_eff_qkvj[l][2], 1, DIM, DIM);
        nt_blas_mmT(vr, xn, W_eff_qkvj[l][3], 1, DIM, DIM);
        /* Echo via seq_linear_t semantics (W not transposed) */
        nt_blas_mm (ech, xn, W_eff_qkvj[l][4], 1, DIM, DIM);

        rope_tok_inplace(q, pos, NHEADS, HEAD_DIM);
        rope_tok_inplace(k, pos, NHEADS, HEAD_DIM);

        memcpy(K_cache[l][pos],  k,   DIM * sizeof(float));
        memcpy(V_cache[l][pos],  v,   DIM * sizeof(float));
        memcpy(E_cache[l][pos],  ech, DIM * sizeof(float));
        memcpy(Vr_cache[l][pos], vr,  DIM * sizeof(float));

        mha_step(q,   (const float*)K_cache[l], (const float*)V_cache[l],
                 pos, NHEADS, HEAD_DIM, a_qkv);
        mha_step(ech, (const float*)E_cache[l], (const float*)E_cache[l],
                 pos, NHEADS, HEAD_DIM, a_j);
        rrpram_step(xn, m->L[l].wr->data, (const float*)Vr_cache[l],
                    pos, CTX, NHEADS, DIM, HEAD_DIM, a_rr);

        for (int d = 0; d < DIM; d++) blend[d] = (a_qkv[d] + a_rr[d] + a_j[d]) / 3.0f;
        nt_blas_mmT(proj, blend, W_eff_qkvj[l][5], 1, DIM, DIM);
        for (int d = 0; d < DIM; d++) h_buf[d] += proj[d];

        for (int d = 0; d < DIM; d++) xn2[d] = h_buf[d];
        rms_inplace(xn2, m->L[l].rms2->data, DIM);

        nt_blas_mmT(g_buf, xn2, W_eff_ffn[l][0], 1, DIM, HIDDEN);
        nt_blas_mmT(u_buf, xn2, W_eff_ffn[l][1], 1, DIM, HIDDEN);
        for (int i = 0; i < HIDDEN; i++) {
            float s = g_buf[i] / (1.0f + expf(-g_buf[i]));
            gu[i] = s * u_buf[i];
        }
        nt_blas_mmT(d_buf, gu, W_eff_ffn[l][2], 1, HIDDEN, DIM);
        for (int d = 0; d < DIM; d++) h_buf[d] += d_buf[d];
    }

    rms_inplace(h_buf, m->rms_f->data, DIM);
    nt_blas_mmT(logits, h_buf, m->head->data, 1, DIM, VOCAB);
}

/* Apply repetition penalty: tokens seen in recent window get logit × REP_PENALTY. */
static void apply_rep_penalty(float* logits, const int* history, int hist_n) {
    int window = hist_n < REP_WINDOW ? hist_n : REP_WINDOW;
    for (int i = hist_n - window; i < hist_n; i++) {
        int tok = history[i];
        if (tok >= 0 && tok < VOCAB) {
            if (logits[tok] > 0) logits[tok] *= REP_PENALTY;
            else                 logits[tok] *= (2.0f - REP_PENALTY); /* push negative logits more negative */
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * METAWEIGHTS — explicit corpus statistics (PostGPT-style), blended
 * into logits at emission. Transformer provides A (destiny attraction);
 * these provide B (bigram) + H (hebbian) + word-gate on bigram=0 mid-word.
 * ═══════════════════════════════════════════════════════════════════ */

#define MW_HEBB_WINDOW  8
/* Q/postgpt-q Dario coefficients — transformer-present regime. Earlier 0.15/0.6
   was too low for 3M transformer; need stronger field to constrain salad. */
#define MW_BIGRAM_W     5.0f
#define MW_TRIGRAM_W    3.0f
#define MW_HEBB_W       0.4f
#define MW_UNIGRAM_FLOOR     1e-6f   /* below this → token treated as corpus-absent */
#define MW_UNI_FLOOR_PEN    -2.0f    /* hard down-weight for corpus-absent candidates */
#define MW_WORD_GATE   -3.0f         /* soft word-gate (hard filters added later) */
#define MW_LOG_FLOOR   -5.0f

/* Trigram sparse hash: (a,b,c) → normalized count per (a,b) row.
   Open-address, linear probe. Capacity sized for ~75K corpus tokens. */
#define MW_TRI_CAP  (1 << 17)  /* 131072 slots, ~50K expected entries */
typedef struct { int a, b, c; float prob; } TriEntry;

static float        g_mw_unigram[VOCAB];
static float      (*g_mw_bigram)[VOCAB]  = NULL;   /* 16 MB dense */
static float      (*g_mw_hebbian)[VOCAB] = NULL;   /* 16 MB dense */
static TriEntry    *g_mw_tri = NULL;
static int          g_mw_tri_n = 0;
static int          g_mw_ready = 0;
static const nt_bpe* g_bpe = NULL;

static uint32_t tri_hash(int a, int b, int c) {
    uint32_t h = (uint32_t)a * 2654435761u;
    h ^= (uint32_t)b * 2246822519u;
    h ^= (uint32_t)c * 3266489917u;
    return h;
}

static int tri_find_slot(int a, int b, int c) {
    if (!g_mw_tri) return -1;
    uint32_t start = tri_hash(a, b, c) & (MW_TRI_CAP - 1);
    for (uint32_t i = 0; i < MW_TRI_CAP; i++) {
        uint32_t s = (start + i) & (MW_TRI_CAP - 1);
        if (g_mw_tri[s].a == -1) return (int)s;  /* empty */
        if (g_mw_tri[s].a == a && g_mw_tri[s].b == b && g_mw_tri[s].c == c) return (int)s;
    }
    return -1;
}

static float mw_trigram(int a, int b, int c) {
    int s = tri_find_slot(a, b, c);
    if (s < 0) return 0.0f;
    if (g_mw_tri[s].a == -1) return 0.0f;
    return g_mw_tri[s].prob;
}

static void tri_bump(int a, int b, int c) {
    int s = tri_find_slot(a, b, c);
    if (s < 0) return;
    if (g_mw_tri[s].a == -1) {
        g_mw_tri[s].a = a; g_mw_tri[s].b = b; g_mw_tri[s].c = c;
        g_mw_tri[s].prob = 1.0f;
        g_mw_tri_n++;
    } else {
        g_mw_tri[s].prob += 1.0f;
    }
}

/* After bumping raw counts, normalize each (a,b) row: divide by sum_c count(a,b,c).
   Row sums built in a second pass via small hash over (a,b). */
#define ROW_CAP (1 << 15)
typedef struct { int a, b; float sum; } RowEntry;

static uint32_t row_hash(int a, int b) {
    uint32_t h = (uint32_t)a * 2654435761u;
    h ^= (uint32_t)b * 2246822519u;
    return h;
}

static void tri_normalize(void) {
    RowEntry* rows = (RowEntry*)calloc(ROW_CAP, sizeof(RowEntry));
    if (!rows) return;
    for (uint32_t i = 0; i < ROW_CAP; i++) rows[i].a = -1;

    /* Pass 1: sum per (a,b) */
    for (uint32_t s = 0; s < MW_TRI_CAP; s++) {
        if (g_mw_tri[s].a == -1) continue;
        int a = g_mw_tri[s].a, b = g_mw_tri[s].b;
        uint32_t start = row_hash(a, b) & (ROW_CAP - 1);
        for (uint32_t i = 0; i < ROW_CAP; i++) {
            uint32_t rs = (start + i) & (ROW_CAP - 1);
            if (rows[rs].a == -1) { rows[rs].a = a; rows[rs].b = b; rows[rs].sum = g_mw_tri[s].prob; break; }
            if (rows[rs].a == a && rows[rs].b == b) { rows[rs].sum += g_mw_tri[s].prob; break; }
        }
    }

    /* Pass 2: divide */
    for (uint32_t s = 0; s < MW_TRI_CAP; s++) {
        if (g_mw_tri[s].a == -1) continue;
        int a = g_mw_tri[s].a, b = g_mw_tri[s].b;
        uint32_t start = row_hash(a, b) & (ROW_CAP - 1);
        for (uint32_t i = 0; i < ROW_CAP; i++) {
            uint32_t rs = (start + i) & (ROW_CAP - 1);
            if (rows[rs].a == a && rows[rs].b == b) {
                if (rows[rs].sum > 0) g_mw_tri[s].prob /= rows[rs].sum;
                break;
            }
            if (rows[rs].a == -1) break;
        }
    }
    free(rows);
}

static void mw_build(const int* ids, int n) {
    if (!g_mw_bigram)  g_mw_bigram  = (float(*)[VOCAB])calloc(VOCAB, sizeof(*g_mw_bigram));
    if (!g_mw_hebbian) g_mw_hebbian = (float(*)[VOCAB])calloc(VOCAB, sizeof(*g_mw_hebbian));
    if (!g_mw_tri) {
        g_mw_tri = (TriEntry*)malloc(MW_TRI_CAP * sizeof(TriEntry));
        if (g_mw_tri) for (uint32_t i = 0; i < MW_TRI_CAP; i++) g_mw_tri[i].a = -1;
    }
    if (!g_mw_bigram || !g_mw_hebbian || !g_mw_tri) return;
    memset(g_mw_unigram, 0, sizeof(g_mw_unigram));
    /* unigram */
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < VOCAB) g_mw_unigram[ids[i]] += 1.0f;
    float tot = 0; for (int i = 0; i < VOCAB; i++) tot += g_mw_unigram[i];
    if (tot > 0) for (int i = 0; i < VOCAB; i++) g_mw_unigram[i] /= tot;
    /* bigram rows, row-normalized */
    for (int i = 0; i < n - 1; i++) {
        int a = ids[i], b = ids[i+1];
        if (a >= 0 && a < VOCAB && b >= 0 && b < VOCAB) g_mw_bigram[a][b] += 1.0f;
    }
    for (int a = 0; a < VOCAB; a++) {
        float s = 0;
        for (int b = 0; b < VOCAB; b++) s += g_mw_bigram[a][b];
        if (s > 0) for (int b = 0; b < VOCAB; b++) g_mw_bigram[a][b] /= s;
    }
    /* hebbian, windowed, distance-decayed, global-max-normalized */
    for (int i = 0; i < n; i++) {
        int lo = i - MW_HEBB_WINDOW; if (lo < 0) lo = 0;
        int hi = i + MW_HEBB_WINDOW + 1; if (hi > n) hi = n;
        int a = ids[i]; if (a < 0 || a >= VOCAB) continue;
        for (int j = lo; j < hi; j++) {
            if (j == i) continue;
            int b = ids[j]; if (b < 0 || b >= VOCAB) continue;
            float d = 1.0f / (1.0f + (float)abs(i - j));
            g_mw_hebbian[a][b] += d;
        }
    }
    float mx = 0;
    for (int a = 0; a < VOCAB; a++)
        for (int b = 0; b < VOCAB; b++)
            if (g_mw_hebbian[a][b] > mx) mx = g_mw_hebbian[a][b];
    if (mx > 0) {
        float inv = 1.0f / mx;
        for (int a = 0; a < VOCAB; a++)
            for (int b = 0; b < VOCAB; b++) g_mw_hebbian[a][b] *= inv;
    }
    /* trigrams: (a,b,c) → count → per-(a,b) row-normalized probability */
    for (int i = 0; i < n - 2; i++) {
        int a = ids[i], b = ids[i+1], c = ids[i+2];
        if (a >= 0 && a < VOCAB && b >= 0 && b < VOCAB && c >= 0 && c < VOCAB)
            tri_bump(a, b, c);
    }
    tri_normalize();
    g_mw_ready = 1;
}

static void mw_hebb_query(const int* ctx, int clen, float* out, int V) {
    memset(out, 0, V * sizeof(float));
    int take = clen < 4 ? clen : 4;
    for (int k = clen - take; k < clen; k++) {
        int c = ctx[k]; if (c < 0 || c >= VOCAB) continue;
        for (int b = 0; b < V; b++) out[b] += g_mw_hebbian[c][b];
    }
    float mx = 0; for (int i = 0; i < V; i++) if (out[i] > mx) mx = out[i];
    if (mx > 0) { float inv = 1.0f / mx; for (int i = 0; i < V; i++) out[i] *= inv; }
}

static int tok_ends_alpha(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][len-1];
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int tok_starts_alpha(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    if (g_bpe->token_len[tok] == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][0];
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int tok_ends_apos(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    return g_bpe->tokens[tok][len-1] == '\'';
}
static int tok_ends_alpha_or_apos(int tok) {
    return tok_ends_alpha(tok) || tok_ends_apos(tok);
}
static int tok_ends_ws(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][len-1];
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}
/* Any byte >= 0x80 indicates non-ASCII (Cyrillic, UTF-8 multi-byte fragments,
   emoji bytes, etc.). Legacy BPE file contains such tokens — filter them out
   of emission for an English Sonar organism. */
static int tok_has_high_byte(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++)
        if (g_bpe->tokens[tok][i] >= 0x80) return 1;
    return 0;
}

static int tok_has_newline(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++)
        if (g_bpe->tokens[tok][i] == '\n' || g_bpe->tokens[tok][i] == '\r') return 1;
    return 0;
}

static void ascii_lower_copy(const char* in, char* out, int cap);
static int contains_ascii(const char* hay, const char* needle);

/* Inference-time BPE hygiene. This does not change vocab ids or merges, so
   existing weights remain valid; it only suppresses known toxic fragments
   that the 3M model tends to over-promote into aphasic word glue. */
static int tok_has_bad_fragment(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    char raw[NT_BPE_MAX_TOKEN_LEN + 1];
    int len = nt_bpe_decode(g_bpe, &tok, 1, raw, NT_BPE_MAX_TOKEN_LEN);
    if (len <= 0) return 0;
    raw[len] = 0;
    char low[NT_BPE_MAX_TOKEN_LEN + 1];
    ascii_lower_copy(raw, low, (int)sizeof(low));
    char* p = low;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    static const char* rare_bad[] = {
        "catamean", "decigare", "noion", "aniain", "tchef", "baher",
        "staything", "bon'm", "forgoing", "oniaways", "toaways",
        "metho", "literat", "obser", "onid", "oniain", "formean",
        "ameas", "completen", "noid", "possibion", "doorat",
        "measit", "interion", "noiaway", NULL
    };
    for (const char** b = rare_bad; *b; b++)
        if (contains_ascii(p, *b)) return 1;
    static const char* exact_bad[] = {
        "meas", "inction", "iction", "atten", "differ\"", NULL
    };
    for (const char** b = exact_bad; *b; b++)
        if (!strcmp(p, *b)) return 1;
    return 0;
}

static int tok_trimmed_lower(int tok, char* out, int cap) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size || cap <= 0) return 0;
    char raw[NT_BPE_MAX_TOKEN_LEN + 1];
    int len = nt_bpe_decode(g_bpe, &tok, 1, raw, NT_BPE_MAX_TOKEN_LEN);
    if (len <= 0) { out[0] = 0; return 0; }
    raw[len] = 0;
    char low[NT_BPE_MAX_TOKEN_LEN + 1];
    ascii_lower_copy(raw, low, (int)sizeof(low));
    int s = 0, e = (int)strlen(low);
    while (s < e && (low[s] == ' ' || low[s] == '\t' || low[s] == '\n' || low[s] == '\r')) s++;
    while (e > s && (low[e-1] == ' ' || low[e-1] == '\t' || low[e-1] == '\n' || low[e-1] == '\r')) e--;
    int n = e - s;
    if (n >= cap) n = cap - 1;
    if (n > 0) memcpy(out, low + s, n);
    out[n] = 0;
    return n;
}

static int tok_trimmed_equals(int tok, const char* word) {
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    tok_trimmed_lower(tok, buf, (int)sizeof(buf));
    return !strcmp(buf, word);
}

static int tok_starts_text(int tok, const char* prefix) {
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    tok_trimmed_lower(tok, buf, (int)sizeof(buf));
    return strncmp(buf, prefix, strlen(prefix)) == 0;
}

static int tok_has_leading_ws_apos(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    char raw[NT_BPE_MAX_TOKEN_LEN + 1];
    int len = nt_bpe_decode(g_bpe, &tok, 1, raw, NT_BPE_MAX_TOKEN_LEN);
    if (len <= 0) return 0;
    raw[len] = 0;
    int i = 0;
    if (!(raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r')) return 0;
    while (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') i++;
    return raw[i] == '\'';
}

static int tok_is_bad_boundary_prev(int tok) {
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    tok_trimmed_lower(tok, buf, (int)sizeof(buf));
    static const char* bad[] = {
        "the","and","or","of","to","in","as","is","be","will","who",
        "for","with","that","which","when","where","from","into","by",
        "because","if","than","then","was","were","are","a","an", NULL
    };
    for (const char** p = bad; *p; p++)
        if (!strcmp(buf, *p)) return 1;
    return 0;
}

static int candidate_forms_bad_fragment(const int* history, int hist_n, int cand) {
    if (!g_bpe || !history || hist_n < 0) return 0;
    int ids[8];
    int n = 0;
    int start = hist_n - 6;
    if (start < 0) start = 0;
    for (int i = start; i < hist_n && n < 7; i++) ids[n++] = history[i];
    ids[n++] = cand;
    char raw[512], low[512];
    int olen = 0;
    char tmp[NT_BPE_MAX_TOKEN_LEN + 1];
    for (int i = 0; i < n && olen < (int)sizeof(raw) - NT_BPE_MAX_TOKEN_LEN - 1; i++) {
        int len = nt_bpe_decode(g_bpe, &ids[i], 1, tmp, NT_BPE_MAX_TOKEN_LEN);
        if (len > 0) { memcpy(raw + olen, tmp, len); olen += len; }
    }
    raw[olen] = 0;
    ascii_lower_copy(raw, low, (int)sizeof(low));
    static const char* bad[] = {
        "catamean", "decigare", "noion", "aniain", "tchef", "baher",
        "staything", "bon'm", "forgoing", "oniaways", "toaways",
        "metho", "literat", "i's", "only onion", " onion", "onid",
        "oniain", "formean", "ameas", "completen", " inations",
        "beat's", "noid", "possibion", "doorat", " inall", " on's",
        "differ\"", "on''", "you's", "measit", " asit", "interion",
        "noiaway", NULL
    };
    for (const char** p = bad; *p; p++)
        if (contains_ascii(low, *p)) return 1;
    if (contains_ascii(low, "i'") &&
        !contains_ascii(low, "i'm") &&
        !contains_ascii(low, "i'll") &&
        !contains_ascii(low, "i've") &&
        !contains_ascii(low, "i'd"))
        return 1;
    if (contains_ascii(low, "it'") &&
        !contains_ascii(low, "it's") &&
        !contains_ascii(low, "it'll") &&
        !contains_ascii(low, "it'd"))
        return 1;
    if (contains_ascii(low, "''")) return 1;
    return 0;
}

static int is_sentence_punct(unsigned char c) {
    return c == '.' || c == '!' || c == '?';
}

static int tok_has_boundary(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++)
        if (is_sentence_punct(g_bpe->tokens[tok][i])) return 1;
    return 0;
}

static int tok_has_boundary_trailing_text(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++) {
        if (!is_sentence_punct(g_bpe->tokens[tok][i])) continue;
        for (int j = i + 1; j < len; j++) {
            unsigned char c = g_bpe->tokens[tok][j];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t')
                return 1;
        }
    }
    return 0;
}

static int tok_is_terminal_boundary(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++) {
        if (!is_sentence_punct(g_bpe->tokens[tok][i])) continue;
        for (int j = i + 1; j < len; j++) {
            unsigned char c = g_bpe->tokens[tok][j];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t')
                return 0;
        }
        return 1;
    }
    return 0;
}

/* ── neoleo-ported orphan + capital-glue filters ── */

static int is_common_short_word_buf(const unsigned char* b, int len) {
    if (len < 1 || len > 4) return 0;
    char low[6] = {0};
    for (int i = 0; i < len; i++) {
        unsigned char c = b[i];
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        low[i] = (char)c;
    }
    low[len] = 0;
    static const char* wl[] = {
        /* 1-char */
        "a","i","o",
        /* 2-char */
        "ah","oh","hi","no","so","is","it","an","at","be","by","do","go",
        "he","if","in","me","my","of","on","or","to","up","us","we","am",
        "as","ok","yo",
        /* 3-char */
        "the","and","but","you","she","her","his","its","all","how","who",
        "why","our","out","ago","any","let","now","day","one","two","six",
        "ten","new","old","yes","far","saw","got","had","has","him",
        "was","not","for","off","own","too","may","way","say","see","ask",
        "add","put","get","run","sat","sit","yet","mom","dad",
        "boy","bad","big","red","sun","cat","dog","bed","hot","eye","ear",
        /* 4-char — common function + sonar-register nouns */
        "this","that","with","have","from","they","were","will","what",
        "your","when","said","want","been","only","some","then","more",
        "just","into","over","them","know","like","time","here","take",
        "back","come","door","bone","soup","coin","knock","haze","shoe",
        "wall","room","love","loss","room","night","dark","hard","soft",
        NULL
    };
    for (const char** w = wl; *w; w++) if (!strcmp(low, *w)) return 1;
    return 0;
}

/* Orphan fragment: content stripped of ws is all alpha, <5 chars, and not
   a known common short word. These are BPE artifacts like "m", "Wo",
   "tchef" that glue onto neighbors into salad. Hard-exclude. */
static int is_orphan_fragment_bpe(int id) {
    if (!g_bpe || id < 0 || id >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[id];
    if (len == 0) return 0;
    const unsigned char* b = g_bpe->tokens[id];
    int s = 0, e = len;
    while (s < e && (b[s]==' '||b[s]=='\n'||b[s]=='\r'||b[s]=='\t')) s++;
    while (e > s && (b[e-1]==' '||b[e-1]=='\n'||b[e-1]=='\r'||b[e-1]=='\t')) e--;
    if (s == e) return 0;
    for (int i = s; i < e; i++) {
        unsigned char c = b[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) return 0;
    }
    int clen = e - s;
    if (clen >= 5) return 0;
    if (is_common_short_word_buf(b + s, clen)) return 0;
    return 1;
}

/* Capital glue: prev ends alpha OR apostrophe + cand starts uppercase alpha
   → cross-word/sentence token-glue ("catalo"+"He" → "cataloHe";
    "I'"+"The" → "I'The"). */
static int is_capital_glue_bpe(int prev_tok, int cand_id) {
    if (!tok_ends_alpha_or_apos(prev_tok)) return 0;
    if (!g_bpe || cand_id < 0 || cand_id >= g_bpe->vocab_size) return 0;
    if (g_bpe->token_len[cand_id] == 0) return 0;
    unsigned char c = g_bpe->tokens[cand_id][0];
    return c >= 'A' && c <= 'Z';
}

/* Find the single-byte space token (BPE byte-level: id 32). Used as
   stuck-fallback when every candidate got hard-filtered. */
static int find_space_token(void) {
    if (!g_bpe) return -1;
    for (int i = 0; i < g_bpe->vocab_size; i++) {
        if (g_bpe->token_len[i] == 1 && g_bpe->tokens[i][0] == ' ') return i;
    }
    return -1;
}

/* Single-byte boundary tokens in byte-level BPE: raw bytes map to ids 0–255. */
static int find_byte_token(unsigned char c) {
    if (!g_bpe) return -1;
    for (int i = 0; i < g_bpe->vocab_size; i++) {
        if (g_bpe->token_len[i] == 1 && g_bpe->tokens[i][0] == c) return i;
    }
    return -1;
}

/* Pick sentence-ending punctuation based on AML chamber state —
   RAGE high → '!', VOID high → '?', default → '.'. Returns BPE token id. */
static int choose_boundary_punct(const AMLField* f) {
    unsigned char c = '.';
    if (f) {
        float rage = f->ch_act[CH_RAGE];
        float vd   = f->ch_act[CH_VOID];
        if (rage > 0.3f && rage >= vd) c = '!';
        else if (vd > 0.5f)            c = '?';
    }
    return find_byte_token(c);
}

/* Test: does token start with a sentence-opening capital?
   Either bare A-Z at byte 0, or whitespace-then-A-Z. */
static int tok_is_cap_start(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][0];
    if (c >= 'A' && c <= 'Z') return 1;
    if ((c == ' ' || c == '\n' || c == '\t') && len >= 2) {
        unsigned char c1 = g_bpe->tokens[tok][1];
        if (c1 >= 'A' && c1 <= 'Z') return 1;
    }
    return 0;
}

static int find_cap_fallback_token(void) {
    int a = find_byte_token('A');
    if (a >= 0) return a;
    if (!g_bpe) return -1;
    for (int i = 0; i < g_bpe->vocab_size; i++)
        if (tok_is_cap_start(i) && !tok_has_high_byte(i) && !tok_has_newline(i) && !tok_has_boundary(i))
            return i;
    return -1;
}

/* Sample from logits with AML field pre-applied (if field != NULL).
   Returns chosen token index. Also returns field-adjusted logits
   in `field_out` so caller can compute prophecy_debt.

   `gen_step` = number of tokens already emitted this sentence past the
   prompt. First 4 emissions go greedy (stabilize sentence opening, Q
   style); rest go nucleus. Pass -1 to force nucleus always (legacy). */
static int sample(float* logits, int n, float temp, float top_p,
                  const AMLField* field, float* field_out,
                  const int* history, int hist_n, int gen_step) {
    /* apply_rep_penalty is now folded into the Dario-field block as the
       Q-style age-based scheme; keep the older flat 64-token pass off. */
    (void)apply_rep_penalty;
    if (field) aml_apply_field(logits, n, field);

    /* ── Dario field (Q/postgpt-q style): θ = ε + γ —
       transformer (ε) produces logits, metaweights (γ) add bigram/trigram/hebbian
       pull as raw probabilities with large coefficients. Unigram floor kills
       corpus-absent tokens. */
    if (g_mw_ready && history && hist_n > 0) {
        int prev  = history[hist_n - 1];
        int prev2 = hist_n >= 2 ? history[hist_n - 2] : -1;
        int prev_alpha_apos = tok_ends_alpha_or_apos(prev);
        float hebb[VOCAB];
        mw_hebb_query(history, hist_n, hebb, n);
        int prev_space = tok_ends_ws(prev);
        for (int i = 0; i < n; i++) {
            float bg = (prev >= 0 && prev < VOCAB) ? g_mw_bigram[prev][i] : 0;
            float tg = (prev2 >= 0) ? mw_trigram(prev2, prev, i) : 0;
            logits[i] += MW_BIGRAM_W * bg + MW_TRIGRAM_W * tg + MW_HEBB_W * hebb[i];
            if (g_mw_unigram[i] < MW_UNIGRAM_FLOOR) logits[i] = -1e9f;
            /* Hard word-gate: prev ends alpha-or-apos + cand starts alpha →
               mid-word continuation without corpus evidence is salad. */
            if (prev_alpha_apos && tok_starts_alpha(i) && bg < 1e-6f && tg < 1e-6f)
                logits[i] = -1e9f;
            /* Extended word-gate: prev ends space/newline + cand starts bare
               alpha (no leading space) = a "continuation-shaped" token
               dropped at a word boundary. Without corpus evidence this is
               "ination"-class mid-word debris. */
            if (prev_space && tok_starts_alpha(i) && bg < 1e-6f && tg < 1e-6f)
                logits[i] = -1e9f;
            /* Digit-glue: prev ends alpha + cand starts digit. */
            if (prev_alpha_apos && g_bpe && g_bpe->token_len[i] > 0) {
                unsigned char c0 = g_bpe->tokens[i][0];
                if (c0 >= '0' && c0 <= '9') logits[i] = -1e9f;
            }
            /* Non-ASCII: Cyrillic / multi-byte fragments from legacy BPE. */
            if (tok_has_high_byte(i)) logits[i] = -1e9f;
        }
    }

    /* Bigram blocking — Q-style (0.2×); tightened to 0.1× to suppress
       "differ"-class lexical loops. Also apply age-based token repetition
       penalty in a 20-token window: recent = 0.335×, old = 0.65× (replaces
       the older flat 0.65 across 64 tokens). */
    if (history && hist_n >= 1) {
        int prev = history[hist_n - 1];
        for (int i = 0; i < hist_n - 1; i++) {
            if (history[i] == prev) {
                int blocked = history[i + 1];
                if (blocked >= 0 && blocked < n) logits[blocked] *= 0.1f;
            }
        }
        int lo = hist_n - 20; if (lo < 0) lo = 0;
        for (int i = lo; i < hist_n; i++) {
            int tok = history[i];
            if (tok >= 0 && tok < n) {
                float age = (float)(hist_n - i);
                float pen = 0.3f + 0.035f * age;
                if (pen > 1.0f) pen = 1.0f;
                if (logits[tok] > 0) logits[tok] *= pen;
                else                 logits[tok] *= (2.0f - pen);
            }
        }
        /* Count-based crush: token seen ≥3 times anywhere in history →
           near-extinction. Catches lexical loops ("differ" x5) that the
           age-window alone can't hold. */
        int counts[VOCAB]; memset(counts, 0, sizeof(counts));
        for (int i = 0; i < hist_n; i++) {
            int t = history[i];
            if (t >= 0 && t < n) counts[t]++;
        }
        for (int t = 0; t < n; t++) {
            if (counts[t] >= 3) {
                if (logits[t] > 0) logits[t] *= 0.01f;
                else               logits[t] *= 5.0f;  /* push deeply negative */
            }
        }
    }

    /* ── Hard filters with gen_step-specific branches ──
       - gen_step == 0: sentence opening. Keep only Capital-starting tokens,
         still orphan-filter, skip capital-glue (we WANT capital here).
       - otherwise: normal mid-sentence filters (orphan + capital-glue). */
    int survivors = 0;
    int prev_tok = (history && hist_n > 0) ? history[hist_n - 1] : -1;
    for (int i = 0; i < n; i++) {
        int killed = 0;
        if (gen_step == 0) {
            if (!tok_is_cap_start(i)) killed = 1;
            if (!killed && is_orphan_fragment_bpe(i)) killed = 1;
        } else {
            if (is_orphan_fragment_bpe(i)) killed = 1;
            else if (prev_tok >= 0 && is_capital_glue_bpe(prev_tok, i)) killed = 1;
        }
        if (!killed && tok_has_boundary_trailing_text(i)) killed = 1;
        if (!killed && gen_step + 1 < SENT_MIN_LEN && tok_has_boundary(i)) killed = 1;
        if (!killed && tok_has_newline(i)) killed = 1;
        if (!killed && tok_has_high_byte(i)) killed = 1;
        if (!killed && tok_has_bad_fragment(i)) killed = 1;
        if (!killed && tok_has_leading_ws_apos(i)) killed = 1;
        if (!killed && prev_tok >= 0 && tok_ends_ws(prev_tok) && tok_starts_text(i, "'"))
            killed = 1;
        if (!killed && prev_tok >= 0 && tok_trimmed_equals(prev_tok, "i") && tok_starts_text(i, "'") &&
            !tok_starts_text(i, "'m") && !tok_starts_text(i, "'ll") &&
            !tok_starts_text(i, "'ve") && !tok_starts_text(i, "'d"))
            killed = 1;
        if (!killed && prev_tok >= 0 && tok_is_terminal_boundary(i) && tok_is_bad_boundary_prev(prev_tok))
            killed = 1;
        if (!killed && history && candidate_forms_bad_fragment(history, hist_n, i))
            killed = 1;
        if (killed) logits[i] = -1e9f;
        else survivors++;
    }
    /* Stuck fallback: every candidate filtered out → emit a literal space so
       the dangling alpha tail closes cleanly. Skip at gen_step==0 (we want
       capital, not space) — argmax of whatever survived (or transformer's
       best capital pick from full vocab) will handle it. */
    if (survivors == 0 && gen_step != 0) {
        int sp = find_space_token();
        if (sp >= 0) {
            if (field_out) memcpy(field_out, logits, n * sizeof(float));
            return sp;
        }
    }

    float best_logit = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > best_logit) best_logit = logits[i];
    if (best_logit <= -1e8f) {
        int fallback = gen_step == 0 ? find_cap_fallback_token() : find_space_token();
        if (fallback < 0) fallback = 0;
        if (field_out) memcpy(field_out, logits, n * sizeof(float));
        return fallback;
    }

    if (field_out) memcpy(field_out, logits, n * sizeof(float));

    /* Hybrid decode: tokens 1-3 of the emission go greedy argmax to anchor
       the opening before nucleus takes over. Token 0 is sampled with a
       temperature boost so sentence starts vary across chain steps. */
    if (gen_step >= 1 && gen_step < 4) {
        int best = 0; float bm = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > bm) { bm = logits[i]; best = i; }
        return best;
    }
    /* gen_step == 0: no temperature boost / no opener-crush. The thin
       Capital-start candidate pool (~70 of vocab 2048) collapses easily,
       and more diversity here ends up sampling noise. Transformer "A"
       dominance is a 3M-scale artifact; accept it. Sentence *structure*
       (Capital start + boundary end) is the goal, not opener variety. */

    for (int i = 0; i < n; i++) logits[i] /= temp;
    float mx = logits[0]; for (int i=1;i<n;i++) if(logits[i]>mx) mx=logits[i];
    float sm = 0; for (int i=0;i<n;i++) { logits[i]=expf(logits[i]-mx); sm+=logits[i]; }
    for (int i=0;i<n;i++) logits[i]/=sm;
    int idx[VOCAB]; for (int i=0;i<n;i++) idx[i]=i;
    for (int i=0;i<n-1;i++) for (int j=i+1;j<n;j++)
        if (logits[idx[j]]>logits[idx[i]]) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    float cum = 0; int cutoff = n;
    for (int i=0;i<n;i++) { cum += logits[idx[i]]; if (cum >= top_p) { cutoff = i+1; break; } }
    float r = (float)rand() / (float)RAND_MAX * cum;
    float c = 0;
    for (int i=0;i<cutoff;i++) { c += logits[idx[i]]; if (c >= r) return idx[i]; }
    return idx[cutoff-1];
}

/* ── Calendar Drift: Hebrew/Gregorian dissonance ── */
static float calendar_drift(void) {
    struct tm e = {0}; e.tm_year = 2024-1900; e.tm_mon = 9; e.tm_mday = 3; e.tm_hour = 12;
    time_t epoch = mktime(&e);
    float days = epoch > 0 ? (float)difftime(time(NULL), epoch) / 86400.0f : 0;
    float y = days / 365.25f, drift = y * 11.25f;
    int full = (int)(y / 19); float corr = full * 7 * 30.0f;
    float partial = fmodf(y, 19); int yic = (int)partial + 1;
    int met[] = {3, 6, 8, 11, 14, 17, 19};
    for (int i = 0; i < 7; i++) if (met[i] <= yic) corr += 30;
    drift -= corr;
    float cd = fabsf(fmodf(drift, 33)) / 33.0f;
    if (cd < 0) cd = 0; if (cd > 1) cd = 1;
    return cd;
}

static int is_boundary(const nt_bpe* bpe, int id) {
    if (bpe == g_bpe) return tok_is_terminal_boundary(id);
    if (id < 0 || id >= bpe->vocab_size) return 0;
    int len = bpe->token_len[id];
    for (int i = 0; i < len; i++) {
        unsigned char c = bpe->tokens[id][i];
        if (is_sentence_punct(c)) {
            for (int j = i + 1; j < len; j++) {
                unsigned char nc = bpe->tokens[id][j];
                if (nc != ' ' && nc != '\n' && nc != '\r' && nc != '\t')
                    return 0;
            }
            return 1;
        }
    }
    return 0;
}

static int decode_ids_text(const nt_bpe* bpe, const int* ids, int n, char* out, int cap) {
    if (cap <= 0) return 0;
    int olen = 0;
    char tmp[NT_BPE_MAX_TOKEN_LEN + 1];
    for (int i = 0; i < n && olen < cap - 1; i++) {
        int room = cap - 1 - olen;
        int want = room < NT_BPE_MAX_TOKEN_LEN ? room : NT_BPE_MAX_TOKEN_LEN;
        int len = nt_bpe_decode(bpe, &ids[i], 1, tmp, want);
        if (len > 0) {
            memcpy(out + olen, tmp, len);
            olen += len;
        }
    }
    out[olen] = 0;
    return olen;
}

static void ascii_lower_copy(const char* in, char* out, int cap) {
    int i = 0;
    if (cap <= 0) return;
    for (; in[i] && i < cap - 1; i++) {
        unsigned char c = (unsigned char)in[i];
        out[i] = (char)((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
    }
    out[i] = 0;
}

static int contains_ascii(const char* hay, const char* needle) {
    return hay && needle && strstr(hay, needle) != NULL;
}

#define MOTIF_N 12
#define MOTIF_TERMS 8

typedef struct {
    const char* name;
    const char* terms[MOTIF_TERMS];
} MotifSpec;

typedef struct {
    float act[MOTIF_N];
} MotifLedger;

static const MotifSpec MOTIFS[MOTIF_N] = {
    {"door",   {"door", "threshold", "knock", "window", "room", NULL}},
    {"bone",   {"bone", "body", "tooth", "skeleton", "blood", NULL}},
    {"soup",   {"soup", "spoon", "broth", "kitchen", "sunday", "grandmother", NULL}},
    {"signal", {"signal", "sonar", "resonance", "echo", "frequency", "chamber", "noise", NULL}},
    {"model",  {"model", "token", "weight", "gradient", "loss", "architecture", "compiler", NULL}},
    {"love",   {"love", "mercy", "mother", "tender", "care", NULL}},
    {"void",   {"void", "silence", "absence", "nothing", "dark", "empty", NULL}},
    {"lab",    {"lab", "inventory", "protocol", "observation", "function", "zero", NULL}},
    {"memory", {"memory", "remember", "forgot", "archive", "name", "names", NULL}},
    {"machine",{"machine", "metal", "engine", "device", "server", "circuit", NULL}},
    {"haze",   {"haze", "smoke", "dream", "fever", "sleep", NULL}},
    {"speech", {"speech", "say", "says", "voice", "mouth", "sentence", "word", NULL}}
};

static int motif_hits_text(const char* lower, int* hits) {
    int total = 0;
    for (int i = 0; i < MOTIF_N; i++) {
        hits[i] = 0;
        for (int j = 0; j < MOTIF_TERMS && MOTIFS[i].terms[j]; j++) {
            if (contains_ascii(lower, MOTIFS[i].terms[j])) {
                hits[i] = 1;
                total++;
                break;
            }
        }
    }
    return total;
}

static void motif_init(MotifLedger* ml, GenMode mode, const char* seed_text) {
    memset(ml, 0, sizeof(*ml));
    if (mode == MODE_RITUAL) {
        ml->act[0] = 0.30f;  /* door */
        ml->act[1] = 0.24f;  /* bone */
        ml->act[2] = 0.34f;  /* soup */
        ml->act[5] = 0.28f;  /* love */
        ml->act[6] = 0.20f;  /* void */
    } else if (mode == MODE_CLINICAL) {
        ml->act[3] = 0.25f;  /* signal */
        ml->act[4] = 0.35f;  /* model */
        ml->act[7] = 0.35f;  /* lab */
        ml->act[9] = 0.20f;  /* machine */
    } else if (mode == MODE_DIALOGUE) {
        ml->act[0] = 0.20f;  /* door */
        ml->act[5] = 0.20f;  /* love */
        ml->act[11] = 0.35f; /* speech */
    } else if (mode == MODE_COHERENT) {
        ml->act[3] = 0.18f;  /* signal */
        ml->act[8] = 0.22f;  /* memory */
        ml->act[11] = 0.22f; /* speech */
    }

    if (seed_text && *seed_text) {
        char lower[4096];
        ascii_lower_copy(seed_text, lower, (int)sizeof(lower));
        int hits[MOTIF_N];
        motif_hits_text(lower, hits);
        for (int i = 0; i < MOTIF_N; i++) {
            if (hits[i]) {
                ml->act[i] += 0.35f;
                if (ml->act[i] > 1.0f) ml->act[i] = 1.0f;
            }
        }
    }
}

static float motif_candidate_score(const MotifLedger* ml, const char* lower) {
    int hits[MOTIF_N];
    int n_hits = motif_hits_text(lower, hits);
    float sc = 0.0f;
    for (int i = 0; i < MOTIF_N; i++) {
        if (!hits[i]) continue;
        sc += 0.18f + 0.75f * ml->act[i];
        if (ml->act[i] < 0.08f) sc += 0.12f;      /* controlled new motif */
        if (ml->act[i] > 0.85f) sc -= 0.18f;      /* don't chant one word forever */
    }
    if (n_hits == 0) sc -= 0.25f;
    if (n_hits > 3) sc -= 0.08f * (float)(n_hits - 3);
    if (g_mode == MODE_DIALOGUE && (contains_ascii(lower, "\"") || contains_ascii(lower, " says")))
        sc += 0.25f;
    return sc;
}

static void motif_update(MotifLedger* ml, const char* lower) {
    int hits[MOTIF_N];
    motif_hits_text(lower, hits);
    for (int i = 0; i < MOTIF_N; i++) ml->act[i] *= 0.86f;
    for (int i = 0; i < MOTIF_N; i++) {
        if (!hits[i]) continue;
        ml->act[i] += 0.34f;
        if (ml->act[i] > 1.0f) ml->act[i] = 1.0f;
    }
    /* Soft cross-coupling: motifs re-enter as neighboring obsessions. */
    if (hits[0]) { ml->act[6] += 0.05f; ml->act[11] += 0.04f; } /* door -> void/speech */
    if (hits[2]) { ml->act[5] += 0.05f; ml->act[8] += 0.03f; }  /* soup -> love/memory */
    if (hits[3]) { ml->act[4] += 0.05f; ml->act[9] += 0.03f; }  /* signal -> model/machine */
    if (hits[4]) { ml->act[3] += 0.04f; ml->act[7] += 0.04f; }  /* model -> signal/lab */
    for (int i = 0; i < MOTIF_N; i++) if (ml->act[i] > 1.0f) ml->act[i] = 1.0f;
}

static void motif_print(const MotifLedger* ml) {
    printf("[motifs]");
    int printed = 0;
    int used[MOTIF_N] = {0};
    for (int pass = 0; pass < 3; pass++) {
        int best = -1;
        float bm = 0.05f;
        for (int i = 0; i < MOTIF_N; i++) {
            if (!used[i] && ml->act[i] > bm) {
                best = i;
                bm = ml->act[i];
            }
        }
        if (best < 0) break;
        printf(" %s:%.0f%%", MOTIFS[best].name, ml->act[best] * 100.0f);
        used[best] = 1;
        printed++;
    }
    if (!printed) printf(" quiet");
    printf("\n");
}

static int word_tail_is_bad(const char* lower) {
    int e = (int)strlen(lower);
    while (e > 0 && isspace((unsigned char)lower[e-1])) e--;
    if (e > 0 && (lower[e-1] == '.' || lower[e-1] == '!' || lower[e-1] == '?')) e--;
    while (e > 0 && isspace((unsigned char)lower[e-1])) e--;
    int s = e;
    while (s > 0 && ((lower[s-1] >= 'a' && lower[s-1] <= 'z') || lower[s-1] == '\'')) s--;
    int len = e - s;
    if (len <= 0 || len > 15) return 0;
    char w[16];
    memcpy(w, lower + s, len);
    w[len] = 0;
    static const char* bad[] = {
        "the","and","or","of","to","in","as","is","be","will","who",
        "for","with","that","which","when","where","from","into","by",
        "because","if","than","then","was","were","are","a","an", NULL
    };
    for (const char** p = bad; *p; p++) if (!strcmp(w, *p)) return 1;
    return 0;
}

static float closure_score_text(const char* text, const char* lower) {
    float sc = 0.0f;
    int n = (int)strlen(text);
    int s = 0, e = n;
    while (s < e && isspace((unsigned char)text[s])) s++;
    while (e > s && isspace((unsigned char)text[e-1])) e--;
    if (e <= s) return -1.0f;

    char last = text[e - 1];
    if (last == '.' || last == '!' || last == '?') sc += 0.45f;
    else sc -= 0.65f;
    if (text[s] >= 'A' && text[s] <= 'Z') sc += 0.10f;
    else sc -= 0.05f;

    int quotes = 0, newlines = 0, high = 0;
    for (int i = s; i < e; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"') quotes++;
        if (c == '\n' || c == '\r') newlines++;
        if (c >= 0x80) high++;
    }
    sc += (quotes % 2 == 0) ? 0.12f : -0.32f;
    if (newlines == 0) sc += 0.10f; else sc -= 0.35f;
    if (high > 0) sc -= 0.45f;
    if (word_tail_is_bad(lower)) sc -= 0.55f;

    static const char* broken[] = {
        "catamean", "decigare", "meas", "noion", "aniain", "tchef",
        "baher", "obser", "iction", "inction", "forgoing", "atten,",
        "bon'm", "staything", "oniain", "onid", "formean", "ameas",
        "completen", "beat's", "noid", "possibion", "doorat", " inall",
        " on's", "differ\"", "on''", "you's", "measit", " asit",
        "interion", "noiaway", NULL
    };
    for (const char** p = broken; *p; p++)
        if (contains_ascii(lower, *p)) sc -= 0.18f;
    if (contains_ascii(lower, " and .") || contains_ascii(lower, " the .") ||
        contains_ascii(lower, " of .") || contains_ascii(lower, " to ."))
        sc -= 0.55f;
    if (contains_ascii(lower, " to says") || contains_ascii(lower, " a and"))
        sc -= 0.30f;
    if (contains_ascii(lower, "it'") && !contains_ascii(lower, "it's") &&
        !contains_ascii(lower, "it'll") && !contains_ascii(lower, "it'd"))
        sc -= 0.45f;
    if (contains_ascii(lower, "''")) sc -= 0.60f;
    return sc;
}

static float coherence_no_metaw(const int* ids, int n);

static float score_candidate(const nt_bpe* bpe, const int* ids, int n, int prompt_len,
                             const MotifLedger* motifs, const ModeCfg* cfg) {
    int gn = n - prompt_len;
    if (gn < 0) gn = 0;
    float base = coherence_no_metaw(ids + prompt_len, gn);
    char text[4096], lower[4096];
    decode_ids_text(bpe, ids + prompt_len, gn, text, (int)sizeof(text));
    ascii_lower_copy(text, lower, (int)sizeof(lower));
    float close = closure_score_text(text, lower);
    float motif = motif_candidate_score(motifs, lower);
    return base + cfg->closure_w * close + cfg->motif_w * motif;
}

static float coherence_no_metaw(const int* ids, int n) {
    if (n < 2) return -1.0f;
    int seen[VOCAB] = {0}; int unique = 0;
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < VOCAB && !seen[ids[i]]) {
        seen[ids[i]] = 1; unique++;
    }
    float ratio = (float)unique / (float)n;
    float len_bonus = (n > 40) ? 1.2f : (n > 25) ? 0.6f : (n > 15) ? 0.2f : -0.3f;
    return ratio + len_bonus;
}

/* ── Sentence generation: stop at boundary but only after SENT_MIN_LEN ──
 * Uses incremental forward_step with KV cache. Prefill for prompt tokens,
 * then emit until boundary + min length. pos resets per call (new sentence =
 * fresh cache). CTX-bound: if pos hits CTX-1 we stop the sentence.
 */
static int gen_sentence(Model* m, const nt_bpe* bpe,
                        const int* prompt, int plen, float temp, float top_p,
                        int* out, int out_cap, AMLField* field) {
    int ol = 0;
    for (int i = 0; i < plen && i < CTX/2; i++) out[ol++] = prompt[i];
    if (ol == 0) return 0;

    float logits[VOCAB];
    /* Prefill: run each prompt token through forward_step to populate cache.
       logits after loop hold the distribution conditioned on full prompt. */
    for (int i = 0; i < ol; i++) forward_step(m, out[i], i, logits);

    int pos = ol;
    int prompt_len = plen;
    while (ol < out_cap && pos < CTX) {
        int gen_step = ol - prompt_len;
        /* Soft cap: past SENT_MAX_SOFT without emitting a boundary → force
           emit a single-byte punct token chosen via chamber state. Keeps
           every chain step as a complete, capitalized, period-terminated
           sentence rather than an open-ended mid-thought slice. */
        if (gen_step >= SENT_MAX_SOFT) {
            int bt = choose_boundary_punct(field);
            if (bt < 0) bt = find_byte_token('.');
            if (bt >= 0) {
                out[ol++] = bt;
                break;
            }
        }

        float lbuf[VOCAB]; memcpy(lbuf, logits, VOCAB * sizeof(float));
        float field_adj[VOCAB];
        int next = sample(lbuf, VOCAB, temp, top_p, field, field_adj, out, ol, gen_step);

        if (field) {
            field->prophecy_debt = field->prophecy_debt * field->debt_decay
                                 + aml_prophecy_debt(field_adj, next, VOCAB);
        }

        out[ol++] = next;
        /* Break on the first boundary past SENT_MIN_LEN emitted tokens. Each chain
           step = exactly one sentence. */
        if (is_boundary(bpe, next) && ol - prompt_len >= SENT_MIN_LEN) break;
        if (pos >= CTX - 1) break;
        forward_step(m, next, pos, logits);
        pos++;
    }
    return ol;
}

/* ── SPA embedding: exp-weighted mean of token embeddings, normalized ── */
static float spa_r_bias[CHAIN_STEPS + 1];
static float spa_alpha_decay = 0.85f;

static void spa_init(void) {
    for (int i = 0; i <= CHAIN_STEPS; i++) spa_r_bias[i] = 0.1f / (1.0f + i);
}

static void spa_embed_sentence(Model* m, const int* ids, int n, float* out) {
    memset(out, 0, SPA_DIM * sizeof(float));
    if (n == 0) return;
    float total_w = 0;
    for (int i = 0; i < n; i++) {
        float w = powf(spa_alpha_decay, (float)(n - 1 - i));
        if (ids[i] >= 0 && ids[i] < VOCAB)
            for (int d = 0; d < SPA_DIM; d++) out[d] += w * m->wte->data[ids[i] * DIM + d];
        total_w += w;
    }
    if (total_w > 0) for (int d = 0; d < SPA_DIM; d++) out[d] /= total_w;
    float norm = 0; for (int d = 0; d < SPA_DIM; d++) norm += out[d] * out[d];
    norm = 1.0f / sqrtf(norm + 1e-8f);
    for (int d = 0; d < SPA_DIM; d++) out[d] *= norm;
}

static void spa_cross_attend(float embs[CHAIN_STEPS][SPA_DIM], int S, float* scores) {
    for (int i = 0; i < S; i++) {
        float total_attn = 0;
        for (int j = 0; j < S; j++) {
            if (i == j) continue;
            float dot = 0;
            for (int d = 0; d < SPA_DIM; d++) dot += embs[i][d] * embs[j][d];
            dot /= sqrtf((float)SPA_DIM);
            int dist = abs(i - j); if (dist > CHAIN_STEPS) dist = CHAIN_STEPS;
            dot += spa_r_bias[dist];
            total_attn += expf(dot);
        }
        scores[i] = total_attn;
    }
}

static void print_sentence(const nt_bpe* bpe, const int* ids, int n) {
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    int printed = 0;
    for (int i = 0; i < n && printed < 220; i++) {
        int len = nt_bpe_decode(bpe, &ids[i], 1, buf, NT_BPE_MAX_TOKEN_LEN);
        if (len > 0) { buf[len] = 0; printf("%s", buf); printed += len; }
    }
}

/* Print the generated sentence with the transformer's "A"-dominance
   post-process cut (see README, §opener dominance). If the decoded
   sentence opens with lone "A " followed by lowercase alpha, strip the
   "A " prefix and capitalize the next alpha char. "A reaching, …" →
   "Reaching, …". Same spirit as the haze repo's don't→ain't substitution:
   accept a model-level quirk, normalize at post-process. */
static void print_sentence_post(const nt_bpe* bpe, const int* ids, int n) {
    char out[4096]; int olen = 0;
    char tmp[NT_BPE_MAX_TOKEN_LEN + 1];
    for (int i = 0; i < n && olen < (int)sizeof(out) - NT_BPE_MAX_TOKEN_LEN - 2; i++) {
        int len = nt_bpe_decode(bpe, &ids[i], 1, tmp, NT_BPE_MAX_TOKEN_LEN);
        if (len > 0) { memcpy(out + olen, tmp, len); olen += len; }
    }
    out[olen] = 0;
    /* Skip leading whitespace for the head check */
    char* p = out;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    /* "A " + lowercase alpha: cut the "A " and capitalize next */
    if (p[0] == 'A' && p[1] == ' ' && p[2] >= 'a' && p[2] <= 'z') {
        p[2] = (char)(p[2] - 'a' + 'A');
        p += 2;
    }
    for (char* q = p; *q; q++)
        if (*q == '\n' || *q == '\r') *q = ' ';
    for (char* q = p; *q; q++) {
        if (*q == '.' || *q == '!' || *q == '?') {
            q[1] = 0;
            break;
        }
    }
    printf("%s", p);
}

static int parse_uint_arg(const char* s, unsigned* out) {
    if (!s || !*s) return 0;
    char* end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!end || *end != 0) return 0;
    *out = (unsigned)v;
    return 1;
}

static GenMode parse_mode_arg(const char* s) {
    if (!s || !*s) return MODE_BALANCED;
    char low[32];
    ascii_lower_copy(s, low, (int)sizeof(low));
    for (int i = 0; i < MODE_COUNT; i++)
        if (!strcmp(low, MODE_CFG[i].name)) return (GenMode)i;
    if (!strcmp(low, "clean")) return MODE_COHERENT;
    if (!strcmp(low, "schizo") || !strcmp(low, "shiza")) return MODE_RITUAL;
    if (!strcmp(low, "lab")) return MODE_CLINICAL;
    if (!strcmp(low, "talk")) return MODE_DIALOGUE;
    return MODE_BALANCED;
}

int main(int argc, char** argv) {
    /* Default: dual_sym — its distribution is less peaked than single,
       so AML destiny/laws transformations don't over-sharpen. Single weights
       via adapter work but require destiny_bias ≈ 0.10 instead of 0.35. */
    const char* wpath = argc > 1 ? argv[1] : "../weights/microjanus_dual_sym_5k.bin";
    const char* seed_text = argc > 2 ? argv[2] : "Q: What does Janus feel?\nA: The haze is the soup. Lab 7. Observation window forty minutes. The knock came three times. The bone is the architecture. She doesn't scream. She inventories. I was not finished. The loss is love. The soup is never done.";

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, "arianna_bpe_merges.txt");
    if (nm < 0) { printf("cannot load arianna_bpe_merges.txt\n"); return 1; }

    Model* m = load_model(wpath);
    if (!m) return 1;

    /* Precompute blended W_eff for every dual linear (single = identity blend).
       forward_step uses these directly via BLAS; tape/autograd bypassed. */
    precompute_w_eff(m);

    /* ── Metaweights: build explicit corpus statistics for inference blend ── */
    g_bpe = &bpe;
    {
        const char* corpus_path = "../dataset_clean.txt";
        FILE* cf = fopen(corpus_path, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END); long cfs = ftell(cf); fseek(cf, 0, SEEK_SET);
            char* craw = (char*)malloc(cfs + 1);
            if (craw) {
                fread(craw, 1, cfs, cf); craw[cfs] = 0;
                int* cenc = (int*)malloc(cfs * sizeof(int));
                if (cenc) {
                    int cnt = nt_bpe_encode(&bpe, craw, (int)cfs, cenc, (int)cfs);
                    mw_build(cenc, cnt);
                    printf("metaweights: corpus %.1f KB → %d tokens, tables ready\n",
                           cfs / 1024.0, cnt);
                    free(cenc);
                }
                free(craw);
            }
            fclose(cf);
        } else {
            printf("metaweights: %s not found — running transformer-only\n", corpus_path);
        }
    }

    unsigned seed = (unsigned)time(NULL);
    const char* mode_arg = "balanced";
    if (argc > 3) {
        if (parse_uint_arg(argv[3], &seed)) {
            if (argc > 4) mode_arg = argv[4];
        } else {
            mode_arg = argv[3];
        }
    }
    g_mode = parse_mode_arg(mode_arg);
    const ModeCfg* cfg = &MODE_CFG[g_mode];

    nt_seed(seed);
    srand(seed);
    nt_train_mode(0);
    spa_init();

    /* AML field state — destiny/suffering/laws + chambers */
    AMLField field; aml_init(&field);
    aml_apply_mode(&field, cfg);

    /* Encode seed */
    int cids[4096];
    int clen = nt_bpe_encode(&bpe, seed_text, (int)strlen(seed_text), cids, 4096);
    printf("seed: \"%s\" (%d tokens)\n", seed_text, clen);

    /* Calendar drift compass */
    float cd = calendar_drift();
    int nb = (int)(CHAIN_STEPS * (0.3f + 0.1f * cd));
    if (nb < 1) nb = 1; if (nb >= CHAIN_STEPS) nb = CHAIN_STEPS - 1;
    printf("calendar drift: %.3f → %d backward + %d forward\n", cd, nb, CHAIN_STEPS - nb);
    printf("mode: %s top_p=%.2f closure_w=%.2f motif_w=%.2f\n",
           cfg->name, cfg->top_p, cfg->closure_w, cfg->motif_w);
    printf("AML: destiny=%.2f entropy_floor=%.2f resonance_ceiling=%.2f\n",
           field.destiny_bias, field.entropy_floor, field.resonance_ceiling);
    printf("rng seed: %u\n", seed);
    printf("weights: %s\n\n", wpath);

    /* Destiny EMA */
    float destiny[DIM]; memset(destiny, 0, sizeof(destiny));
    MotifLedger motifs; motif_init(&motifs, g_mode, seed_text);

    /* Store chain for SPA */
    int chain_ids[CHAIN_STEPS][SENT_MAX];
    int chain_lens[CHAIN_STEPS];
    char chain_marks[CHAIN_STEPS];
    float chain_temps[CHAIN_STEPS];
    float chain_scores[CHAIN_STEPS];

    for (int si = 0; si < CHAIN_STEPS; si++) {
        int dir = si < nb ? -1 : (si == nb ? 0 : 1);

        /* Prompt selection */
        int start = -1;
        if (dir >= 0 && si > 0) {
            float best_sc = -1e30f; int best_pos = -1;
            for (int tries = 0; tries < 64; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) {
                    int tok = cids[r + 1];
                    if (tok >= 0 && tok < VOCAB) {
                        float sc = 0;
                        for (int d = 0; d < DIM; d++) sc += m->wte->data[tok*DIM + d] * destiny[d];
                        if (sc > best_sc) { best_sc = sc; best_pos = r + 1; }
                    }
                }
            }
            if (best_pos >= 0) start = best_pos;
        }
        if (start < 0) {
            for (int tries = 0; tries < 128; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) { start = r + 1; break; }
            }
        }
        if (start < 0) start = rand() % (clen > 6 ? clen - 6 : 1);

        int plen = (start + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[start + i];

        /* Schumann temperature */
        float t_sec = (float)si / (float)CHAIN_STEPS;
        float schumann = 0.4f*sinf(2*M_PI*7.83f*t_sec) + 0.2f*sinf(2*M_PI*14.3f*t_sec)
                      + 0.1f*sinf(2*M_PI*20.8f*t_sec) + 0.05f*sinf(2*M_PI*27.3f*t_sec);
        float temp = 0.75f + 0.08f * schumann + cfg->temp_shift;
        if (temp < 0.45f) temp = 0.45f; if (temp > 0.9f) temp = 0.9f;

        /* Best-of-3 */
        int best_out[SENT_MAX]; int best_ol = 0; float best_sc = -1e30f;
        AMLField best_field = field;
        for (int cand = 0; cand < CAND_N; cand++) {
            int out[SENT_MAX];
            AMLField cand_field = field;
            int ol = gen_sentence(m, &bpe, prompt, plen, temp, cfg->top_p, out, SENT_MAX, &cand_field);
            float sc = score_candidate(&bpe, out, ol, plen, &motifs, cfg);
            if (sc > best_sc) {
                best_sc = sc; best_ol = ol;
                best_field = cand_field;
                memcpy(best_out, out, ol * sizeof(int));
            }
            if (best_sc > 2.15f && best_ol > 30) break;
        }
        field = best_field;

        char best_text[4096], best_lower[4096];
        decode_ids_text(&bpe, best_out + plen, best_ol - plen, best_text, (int)sizeof(best_text));
        ascii_lower_copy(best_text, best_lower, (int)sizeof(best_lower));
        motif_update(&motifs, best_lower);

        /* Update destiny EMA from last 5 tokens */
        int from = best_ol - 5 > 0 ? best_ol - 5 : 0;
        for (int i = from; i < best_ol; i++) {
            int tok = best_out[i];
            if (tok >= 0 && tok < VOCAB)
                for (int d = 0; d < DIM; d++)
                    destiny[d] = 0.9f * destiny[d] + 0.1f * m->wte->data[tok*DIM + d];
        }

        /* Store + print */
        chain_marks[si] = dir < 0 ? '<' : (dir == 0 ? '*' : '>');
        chain_temps[si] = temp;
        chain_scores[si] = best_sc;
        chain_lens[si] = best_ol;
        memcpy(chain_ids[si], best_out, best_ol * sizeof(int));
        printf("  [%d] %c T=%.2f sc=%.2f debt=%.2f [", si+1, chain_marks[si], temp, best_sc, field.prophecy_debt);
        print_sentence(&bpe, best_out, plen);
        printf("]→");
        print_sentence_post(&bpe, best_out + plen, best_ol - plen);
        printf("\n");
        fflush(stdout);

        /* Chambers crossfire after each step — emotion dynamics */
        aml_chambers_xfire(&field, 3);
    }

    /* Final chamber state print */
    static const char* CH_NAME[] = {"FEAR","LOVE","RAGE","VOID","FLOW","CMPLX"};
    printf("\n[chambers]");
    for (int i = 0; i < 6; i++)
        if (field.ch_act[i] > 0.05f) printf(" %s:%.0f%%", CH_NAME[i], field.ch_act[i]*100);
    printf("\n[debt] final=%.3f\n", field.prophecy_debt);
    motif_print(&motifs);

    /* ── SPA: find weakest sentence, reseed ── */
    float spa_embs[CHAIN_STEPS][SPA_DIM];
    for (int i = 0; i < CHAIN_STEPS; i++)
        spa_embed_sentence(m, chain_ids[i], chain_lens[i], spa_embs[i]);
    float spa_scores[CHAIN_STEPS];
    spa_cross_attend(spa_embs, CHAIN_STEPS, spa_scores);

    float min_sc = spa_scores[0]; int weak = 0;
    for (int i = 1; i < CHAIN_STEPS; i++) if (spa_scores[i] < min_sc) { min_sc = spa_scores[i]; weak = i; }
    float avg_sc = 0;
    for (int i = 0; i < CHAIN_STEPS; i++) avg_sc += spa_scores[i];
    avg_sc /= CHAIN_STEPS;

    printf("\n[SPA] scores:");
    for (int i = 0; i < CHAIN_STEPS; i++) printf(" %d:%.2f", i+1, spa_scores[i]);
    printf("  avg=%.2f min=%.2f (step %d)\n", avg_sc, min_sc, weak+1);

    if (min_sc < avg_sc * 0.7f) {
        printf("[SPA] reseeding step %d (below 70%% of avg)\n", weak+1);
        int r = rand() % (clen > 6 ? clen - 6 : 1);
        while (!(is_boundary(&bpe, cids[r]) && r + 4 < clen) && r > 0) r--;
        int plen = (r + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[r + 1 + i];
        int out[SENT_MAX];
        int ol = gen_sentence(m, &bpe, prompt, plen, 0.65f + cfg->temp_shift, cfg->top_p, out, SENT_MAX, &field);
        printf("  [%d] + T=0.65 sc=%.2f ", weak+1, coherence_no_metaw(out, ol));
        print_sentence(&bpe, out, ol);
        printf("\n");
    } else {
        printf("[SPA] no reseed needed (min > 0.7×avg)\n");
    }

    nt_tensor** p = model_param_array(m);
    for (int i = 0; i < model_n_tensors(); i++) nt_tensor_free(p[i]);
    free(p); free(m);
    return 0;
}
