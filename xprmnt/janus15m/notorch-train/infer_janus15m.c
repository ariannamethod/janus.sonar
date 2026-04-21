/*
 * infer_janus15m.c — minimal baseline inference for janus 15M base weights.
 *
 * Intentionally spartan: forward_step with KV cache (no tape), nucleus
 * sampling on raw logits. No Dario field, no motif ledger, no hard filters,
 * no mode presets. This is the baseline — we look at its output to decide
 * which layers of the microjanus stack actually transfer to 30 MB FineWeb.
 *
 *   Build: make infer_janus15m
 *   Run:   ./infer_janus15m janus15m_v1.bin "The knock came three times"
 *          ./infer_janus15m janus15m_v1.bin "..." 128   # max emit tokens
 *          ./infer_janus15m janus15m_v1.bin "..." 128 0.75   # + temp
 *
 * Config must match training: DIM 320 / L 8 / H 5 / HD 64 / HIDDEN 1024 /
 * CTX 128 / VOCAB 4096 / RRPRAM R=64. 107 tensors per checkpoint.
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define DIM       320
#define NLAYERS   8
#define NHEADS    5
#define HEAD_DIM  64
#define HIDDEN    1024
#define CTX       128
#define VOCAB     4096
#define RRPRAM_R  64

#define MAX_EMIT_DEFAULT 80

/* ── Dario field constants (Q / postgpt-q magnitudes) ── */
#define MW_HEBB_WINDOW    8
#define MW_BIGRAM_W       5.0f
#define MW_TRIGRAM_W      3.0f
#define MW_HEBB_W         0.4f
#define MW_UNIGRAM_FLOOR  1e-6f

/* ── Sentence structure ── */
#define SENT_MIN_LEN   8
#define SENT_MAX_SOFT  40

/* ── Model ── */
typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wvr, *wj, *wo;
        nt_tensor *wr_a, *wr_b;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static int n_expected_tensors(void) { return 1 + NLAYERS * 13 + 2; }

static Model* load_model(const char* path) {
    int n = 0;
    nt_tensor** t = nt_load(path, &n);
    if (!t) { fprintf(stderr, "load failed: %s\n", path); return NULL; }
    if (n != n_expected_tensors()) {
        fprintf(stderr, "tensor mismatch: got %d, expected %d\n", n, n_expected_tensors());
        for (int i = 0; i < n; i++) nt_tensor_free(t[i]); free(t); return NULL;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    int i = 0;
    m->wte = t[i++];
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1   = t[i++];
        m->L[l].wq     = t[i++]; m->L[l].wk    = t[i++]; m->L[l].wv   = t[i++];
        m->L[l].wvr    = t[i++]; m->L[l].wj    = t[i++]; m->L[l].wo   = t[i++];
        m->L[l].wr_a   = t[i++]; m->L[l].wr_b  = t[i++];
        m->L[l].rms2   = t[i++];
        m->L[l].w_gate = t[i++]; m->L[l].w_up  = t[i++]; m->L[l].w_down = t[i++];
    }
    m->rms_f = t[i++]; m->head = t[i++];
    free(t);
    return m;
}

/* ── Pre-computed RRPRAM effective weight: Wr_eff = Wr_a @ Wr_b ──
   Shape [NHEADS*DIM, CTX]. Computed once at load; no tape, no backward. */
static float* Wr_eff[NLAYERS];   /* each malloc'd [NHEADS*DIM * CTX] */

static void precompute_wr_eff(Model* m) {
    for (int l = 0; l < NLAYERS; l++) {
        int rows = NHEADS * DIM;       /* 1600 */
        Wr_eff[l] = (float*)malloc(rows * CTX * sizeof(float));
        /* Wr_eff = Wr_a[rows, R] @ Wr_b[R, CTX] */
        nt_blas_mm(Wr_eff[l], m->L[l].wr_a->data, m->L[l].wr_b->data, rows, RRPRAM_R, CTX);
    }
}

/* ── KV cache ── */
static float K_cache[NLAYERS][CTX][DIM];
static float V_cache[NLAYERS][CTX][DIM];
static float E_cache[NLAYERS][CTX][DIM];
static float Vr_cache[NLAYERS][CTX][DIM];

/* ── Primitives (ported from microjanus forward_step) ── */
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

/* RRPRAM step: scores[j] = <xn, Wr_eff[:, j]> for j in 0..t; softmax; weighted V. */
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

/* forward_step: one emission at position `pos`, logits → [VOCAB]. */
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

        /* Q, K, V, Vr via seq_linear semantics (W row-major [out, in]) */
        nt_blas_mmT(q,  xn, m->L[l].wq->data,  1, DIM, DIM);
        nt_blas_mmT(k,  xn, m->L[l].wk->data,  1, DIM, DIM);
        nt_blas_mmT(v,  xn, m->L[l].wv->data,  1, DIM, DIM);
        nt_blas_mmT(vr, xn, m->L[l].wvr->data, 1, DIM, DIM);
        /* Echo via seq_linear_t (W not transposed) */
        nt_blas_mm (ech, xn, m->L[l].wj->data, 1, DIM, DIM);

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
        rrpram_step(xn, Wr_eff[l], (const float*)Vr_cache[l],
                    pos, CTX, NHEADS, DIM, HEAD_DIM, a_rr);

        for (int d = 0; d < DIM; d++) blend[d] = (a_qkv[d] + a_rr[d] + a_j[d]) / 3.0f;
        nt_blas_mmT(proj, blend, m->L[l].wo->data, 1, DIM, DIM);
        for (int d = 0; d < DIM; d++) h_buf[d] += proj[d];

        for (int d = 0; d < DIM; d++) xn2[d] = h_buf[d];
        rms_inplace(xn2, m->L[l].rms2->data, DIM);

        nt_blas_mmT(g_buf, xn2, m->L[l].w_gate->data, 1, DIM, HIDDEN);
        nt_blas_mmT(u_buf, xn2, m->L[l].w_up->data,   1, DIM, HIDDEN);
        for (int i = 0; i < HIDDEN; i++) {
            float s = g_buf[i] / (1.0f + expf(-g_buf[i]));
            gu[i] = s * u_buf[i];
        }
        nt_blas_mmT(d_buf, gu, m->L[l].w_down->data, 1, HIDDEN, DIM);
        for (int d = 0; d < DIM; d++) h_buf[d] += d_buf[d];
    }

    rms_inplace(h_buf, m->rms_f->data, DIM);
    nt_blas_mmT(logits, h_buf, m->head->data, 1, DIM, VOCAB);
}

/* ═══════════════════════════════════════════════════════════════════
 * Dario field (bigram + trigram + hebbian + unigram-floor) + hard filters
 * ═══════════════════════════════════════════════════════════════════ */

#define MW_TRI_CAP  (1 << 17)   /* 131072 slots */
typedef struct { int a, b, c; float prob; } TriEntry;

static float  g_uni[VOCAB];
static float (*g_bg)[VOCAB] = NULL;
static float (*g_hb)[VOCAB] = NULL;
static TriEntry* g_tri = NULL;
static const nt_bpe* g_bpe = NULL;
static int g_stats_ready = 0;

static uint32_t tri_hash(int a, int b, int c) {
    uint32_t h = (uint32_t)a * 2654435761u;
    h ^= (uint32_t)b * 2246822519u;
    h ^= (uint32_t)c * 3266489917u;
    return h;
}
static int tri_slot(int a, int b, int c) {
    if (!g_tri) return -1;
    uint32_t start = tri_hash(a,b,c) & (MW_TRI_CAP - 1);
    for (uint32_t i = 0; i < MW_TRI_CAP; i++) {
        uint32_t s = (start + i) & (MW_TRI_CAP - 1);
        if (g_tri[s].a == -1) return (int)s;
        if (g_tri[s].a == a && g_tri[s].b == b && g_tri[s].c == c) return (int)s;
    }
    return -1;
}
static float tri_get(int a, int b, int c) {
    int s = tri_slot(a,b,c);
    if (s < 0 || g_tri[s].a == -1) return 0.0f;
    return g_tri[s].prob;
}
static void tri_bump(int a, int b, int c) {
    int s = tri_slot(a,b,c);
    if (s < 0) return;
    if (g_tri[s].a == -1) { g_tri[s].a=a; g_tri[s].b=b; g_tri[s].c=c; g_tri[s].prob=1; }
    else g_tri[s].prob += 1.0f;
}

#define ROW_CAP (1 << 16)
typedef struct { int a, b; float sum; } RowEntry;
static void tri_normalize(void) {
    RowEntry* rows = (RowEntry*)calloc(ROW_CAP, sizeof(RowEntry));
    if (!rows) return;
    for (uint32_t i = 0; i < ROW_CAP; i++) rows[i].a = -1;
    for (uint32_t s = 0; s < MW_TRI_CAP; s++) {
        if (g_tri[s].a == -1) continue;
        uint32_t start = ((uint32_t)g_tri[s].a * 2654435761u) ^ ((uint32_t)g_tri[s].b * 2246822519u);
        start &= (ROW_CAP - 1);
        for (uint32_t i = 0; i < ROW_CAP; i++) {
            uint32_t rs = (start + i) & (ROW_CAP - 1);
            if (rows[rs].a == -1) { rows[rs].a=g_tri[s].a; rows[rs].b=g_tri[s].b; rows[rs].sum=g_tri[s].prob; break; }
            if (rows[rs].a == g_tri[s].a && rows[rs].b == g_tri[s].b) { rows[rs].sum += g_tri[s].prob; break; }
        }
    }
    for (uint32_t s = 0; s < MW_TRI_CAP; s++) {
        if (g_tri[s].a == -1) continue;
        uint32_t start = ((uint32_t)g_tri[s].a * 2654435761u) ^ ((uint32_t)g_tri[s].b * 2246822519u);
        start &= (ROW_CAP - 1);
        for (uint32_t i = 0; i < ROW_CAP; i++) {
            uint32_t rs = (start + i) & (ROW_CAP - 1);
            if (rows[rs].a == g_tri[s].a && rows[rs].b == g_tri[s].b) { if (rows[rs].sum > 0) g_tri[s].prob /= rows[rs].sum; break; }
            if (rows[rs].a == -1) break;
        }
    }
    free(rows);
}

/* Build stats from uint16 train.bin shard — BPE 4096 matches training. */
static void build_stats_from_shard(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "stats shard missing: %s\n", path); return; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    long n = bytes / 2;
    uint16_t* buf = (uint16_t*)malloc(bytes);
    fread(buf, 1, bytes, f); fclose(f);

    g_bg = (float(*)[VOCAB])calloc(VOCAB, sizeof(*g_bg));
    g_hb = (float(*)[VOCAB])calloc(VOCAB, sizeof(*g_hb));
    g_tri = (TriEntry*)malloc(MW_TRI_CAP * sizeof(TriEntry));
    for (uint32_t i = 0; i < MW_TRI_CAP; i++) g_tri[i].a = -1;
    memset(g_uni, 0, sizeof(g_uni));

    /* unigram */
    for (long i = 0; i < n; i++) if (buf[i] < VOCAB) g_uni[buf[i]] += 1.0f;
    float tot = 0; for (int i = 0; i < VOCAB; i++) tot += g_uni[i];
    if (tot > 0) for (int i = 0; i < VOCAB; i++) g_uni[i] /= tot;

    /* bigram, row-normalized */
    for (long i = 0; i < n - 1; i++) {
        int a = buf[i], b = buf[i+1];
        if (a < VOCAB && b < VOCAB) g_bg[a][b] += 1.0f;
    }
    for (int a = 0; a < VOCAB; a++) {
        float s = 0; for (int b = 0; b < VOCAB; b++) s += g_bg[a][b];
        if (s > 0) for (int b = 0; b < VOCAB; b++) g_bg[a][b] /= s;
    }

    /* hebbian, window ±8, distance-decayed, global-max-normalized */
    for (long i = 0; i < n; i++) {
        long lo = i - MW_HEBB_WINDOW; if (lo < 0) lo = 0;
        long hi = i + MW_HEBB_WINDOW + 1; if (hi > n) hi = n;
        int a = buf[i]; if (a >= VOCAB) continue;
        for (long j = lo; j < hi; j++) {
            if (j == i) continue;
            int b = buf[j]; if (b >= VOCAB) continue;
            float d = 1.0f / (1.0f + (float)labs(i - j));
            g_hb[a][b] += d;
        }
    }
    float mx = 0; for (int a = 0; a < VOCAB; a++) for (int b = 0; b < VOCAB; b++) if (g_hb[a][b] > mx) mx = g_hb[a][b];
    if (mx > 0) { float inv = 1.0f/mx; for (int a = 0; a < VOCAB; a++) for (int b = 0; b < VOCAB; b++) g_hb[a][b] *= inv; }

    /* trigram (a,b,c) → P(c|a,b) */
    for (long i = 0; i < n - 2; i++) {
        int a = buf[i], b = buf[i+1], c = buf[i+2];
        if (a < VOCAB && b < VOCAB && c < VOCAB) tri_bump(a, b, c);
    }
    tri_normalize();

    free(buf);
    g_stats_ready = 1;
    fprintf(stderr, "stats built from %ld tokens: bigram + trigram + hebbian + unigram\n", n);
}

static void mw_hebb_query(const int* ctx, int clen, float* out, int V) {
    memset(out, 0, V * sizeof(float));
    int take = clen < 4 ? clen : 4;
    for (int k = clen - take; k < clen; k++) {
        int c = ctx[k]; if (c < 0 || c >= VOCAB) continue;
        for (int b = 0; b < V; b++) out[b] += g_hb[c][b];
    }
    float mx = 0; for (int i = 0; i < V; i++) if (out[i] > mx) mx = out[i];
    if (mx > 0) { float inv = 1.0f/mx; for (int i = 0; i < V; i++) out[i] *= inv; }
}

/* ── token byte-shape helpers ── */
static int tok_ends_alpha(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][len-1];
    return (c>='a'&&c<='z')||(c>='A'&&c<='Z');
}
static int tok_starts_alpha(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    if (g_bpe->token_len[tok] == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][0];
    return (c>='a'&&c<='z')||(c>='A'&&c<='Z');
}
static int tok_ends_apos(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    return g_bpe->tokens[tok][len-1] == '\'';
}
static int tok_ends_alpha_or_apos(int tok) { return tok_ends_alpha(tok) || tok_ends_apos(tok); }
static int tok_ends_ws(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][len-1];
    return c==' '||c=='\n'||c=='\r'||c=='\t';
}
static int tok_has_high_byte(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok];
    for (int i = 0; i < len; i++) if (g_bpe->tokens[tok][i] >= 0x80) return 1;
    return 0;
}
static int tok_is_cap_start(int tok) {
    if (!g_bpe || tok < 0 || tok >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[tok]; if (len == 0) return 0;
    unsigned char c = g_bpe->tokens[tok][0];
    if (c >= 'A' && c <= 'Z') return 1;
    if ((c==' '||c=='\n'||c=='\t') && len >= 2) { unsigned char c1 = g_bpe->tokens[tok][1]; if (c1>='A'&&c1<='Z') return 1; }
    return 0;
}

static int is_common_short(const unsigned char* b, int len) {
    if (len < 1 || len > 4) return 0;
    char low[6] = {0};
    for (int i = 0; i < len; i++) { unsigned char c = b[i]; if (c>='A'&&c<='Z') c=(unsigned char)(c-'A'+'a'); low[i]=(char)c; }
    low[len] = 0;
    static const char* wl[] = {
        "a","i","o","ah","oh","hi","no","so","is","it","an","at","be","by","do","go",
        "he","if","in","me","my","of","on","or","to","up","us","we","am","as","ok","yo",
        "the","and","but","you","she","her","his","its","all","how","who","why","our",
        "out","ago","any","let","now","day","one","two","six","ten","new","old","yes",
        "far","saw","got","had","has","him","was","not","for","off","own","too","may",
        "way","say","see","ask","add","put","get","run","sat","sit","yet","mom","dad",
        "boy","bad","big","red","sun","cat","dog","bed","hot","eye","ear",
        "this","that","with","have","from","they","were","will","what","your","when",
        "said","want","been","only","some","then","more","just","into","over","them",
        "know","like","time","here","take","back","come",
        NULL };
    for (const char** w = wl; *w; w++) if (!strcmp(low, *w)) return 1;
    return 0;
}
static int is_orphan_fragment(int id) {
    if (!g_bpe || id < 0 || id >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[id]; if (len == 0) return 0;
    const unsigned char* b = g_bpe->tokens[id];
    int s = 0, e = len;
    while (s < e && (b[s]==' '||b[s]=='\n'||b[s]=='\r'||b[s]=='\t')) s++;
    while (e > s && (b[e-1]==' '||b[e-1]=='\n'||b[e-1]=='\r'||b[e-1]=='\t')) e--;
    if (s == e) return 0;
    for (int i = s; i < e; i++) { unsigned char c = b[i]; if (!((c>='a'&&c<='z')||(c>='A'&&c<='Z'))) return 0; }
    int clen = e - s;
    if (clen >= 5) return 0;
    if (is_common_short(b + s, clen)) return 0;
    return 1;
}
static int is_capital_glue(int prev, int cand) {
    if (!tok_ends_alpha_or_apos(prev)) return 0;
    if (!g_bpe || cand < 0 || cand >= g_bpe->vocab_size || g_bpe->token_len[cand] == 0) return 0;
    unsigned char c = g_bpe->tokens[cand][0];
    return c >= 'A' && c <= 'Z';
}
static int is_boundary(int id) {
    if (!g_bpe || id < 0 || id >= g_bpe->vocab_size) return 0;
    int len = g_bpe->token_len[id];
    for (int i = 0; i < len; i++) {
        unsigned char c = g_bpe->tokens[id][i];
        if (c=='.'||c=='!'||c=='?') {
            if (i == len-1) return 1;
            unsigned char n = g_bpe->tokens[id][i+1];
            if (n==' '||n=='\n'||n=='\r') return 1;
        }
    }
    return 0;
}
static int find_byte_tok(unsigned char c) {
    if (!g_bpe) return -1;
    for (int i = 0; i < g_bpe->vocab_size; i++) if (g_bpe->token_len[i]==1 && g_bpe->tokens[i][0]==c) return i;
    return -1;
}

/* ── Sampling: Dario field + filters + nucleus + hybrid decode ──
   `gen_step` = tokens already emitted past prompt in this sentence. */
static int sample(float* logits, int n, float temp, float top_p,
                  const int* history, int hist_n, int gen_step) {
    /* Dario field injection */
    if (g_stats_ready && history && hist_n > 0) {
        int prev  = history[hist_n - 1];
        int prev2 = hist_n >= 2 ? history[hist_n - 2] : -1;
        int prev_ap = tok_ends_alpha_or_apos(prev);
        int prev_sp = tok_ends_ws(prev);
        float hebb[VOCAB]; mw_hebb_query(history, hist_n, hebb, n);
        for (int i = 0; i < n; i++) {
            float bg = (prev>=0 && prev<VOCAB) ? g_bg[prev][i] : 0;
            float tg = (prev2>=0) ? tri_get(prev2, prev, i) : 0;
            logits[i] += MW_BIGRAM_W * bg + MW_TRIGRAM_W * tg + MW_HEBB_W * hebb[i];
            if (g_uni[i] < MW_UNIGRAM_FLOOR) logits[i] = -1e9f;
            if (prev_ap && tok_starts_alpha(i) && bg < 1e-6f && tg < 1e-6f) logits[i] = -1e9f;
            if (prev_sp && tok_starts_alpha(i) && bg < 1e-6f && tg < 1e-6f) logits[i] = -1e9f;
            if (prev_ap && g_bpe && g_bpe->token_len[i] > 0) {
                unsigned char c0 = g_bpe->tokens[i][0];
                if (c0 >= '0' && c0 <= '9') logits[i] = -1e9f;
            }
            if (tok_has_high_byte(i)) logits[i] = -1e9f;
        }
    }

    /* Repetition: bigram blocking + age-based penalty + count crush */
    if (history && hist_n >= 1) {
        int prev = history[hist_n - 1];
        for (int i = 0; i < hist_n - 1; i++) {
            if (history[i] == prev) { int blk = history[i+1]; if (blk>=0 && blk<n) logits[blk] *= 0.1f; }
        }
        int lo = hist_n - 20; if (lo < 0) lo = 0;
        for (int i = lo; i < hist_n; i++) {
            int t = history[i]; if (t<0||t>=n) continue;
            float age = (float)(hist_n - i);
            float pen = 0.3f + 0.035f * age; if (pen > 1.0f) pen = 1.0f;
            if (logits[t] > 0) logits[t] *= pen; else logits[t] *= (2.0f - pen);
        }
        int cnt[VOCAB] = {0};
        for (int i = 0; i < hist_n; i++) { int t = history[i]; if (t>=0 && t<n) cnt[t]++; }
        for (int t = 0; t < n; t++) if (cnt[t] >= 3) {
            if (logits[t] > 0) logits[t] *= 0.01f; else logits[t] *= 5.0f;
        }
    }

    /* Hard filters — gen_step-specific */
    int survivors = 0;
    int prev_tok = (history && hist_n > 0) ? history[hist_n - 1] : -1;
    for (int i = 0; i < n; i++) {
        int killed = 0;
        if (gen_step == 0) {
            if (!tok_is_cap_start(i)) killed = 1;
            if (!killed && is_orphan_fragment(i)) killed = 1;
        } else {
            if (is_orphan_fragment(i)) killed = 1;
            else if (prev_tok >= 0 && is_capital_glue(prev_tok, i)) killed = 1;
        }
        if (killed) logits[i] = -1e9f; else survivors++;
    }
    if (survivors == 0 && gen_step != 0) {
        int sp = find_byte_tok(' '); if (sp >= 0) return sp;
    }

    /* Hybrid decode: tokens 1-3 greedy argmax, token 0 + rest nucleus */
    if (gen_step >= 1 && gen_step < 4) {
        int best = 0; float bm = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > bm) { bm = logits[i]; best = i; }
        return best;
    }

    /* Nucleus */
    for (int i = 0; i < n; i++) logits[i] /= temp;
    float mx = logits[0]; for (int i=1;i<n;i++) if(logits[i]>mx) mx=logits[i];
    float sm = 0; for (int i=0;i<n;i++) { logits[i]=expf(logits[i]-mx); sm+=logits[i]; }
    for (int i=0;i<n;i++) logits[i]/=sm;
    int idx[VOCAB]; for (int i=0;i<n;i++) idx[i]=i;
    /* Partial sort top-K */
    int K = 128;
    for (int i = 0; i < K && i < n; i++) {
        int best = i;
        for (int j = i+1; j < n; j++) if (logits[idx[j]] > logits[idx[best]]) best = j;
        if (best != i) { int t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
    }
    float cum = 0; int cutoff = K;
    for (int i = 0; i < K; i++) { cum += logits[idx[i]]; if (cum >= top_p) { cutoff = i+1; break; } }
    float r = (float)rand() / (float)RAND_MAX * cum;
    float c = 0;
    for (int i = 0; i < cutoff; i++) { c += logits[idx[i]]; if (c >= r) return idx[i]; }
    return idx[cutoff - 1];
}

/* ── Main ── */
int main(int argc, char** argv) {
    const char* wpath = argc > 1 ? argv[1] : "janus15m_v1.bin";
    const char* seed  = argc > 2 ? argv[2] : "The knock came three times";
    int max_emit = argc > 3 ? atoi(argv[3]) : MAX_EMIT_DEFAULT;
    float temp   = argc > 4 ? (float)atof(argv[4]) : 0.75f;
    float top_p  = argc > 5 ? (float)atof(argv[5]) : 0.9f;

    nt_bpe bpe;
    if (nt_bpe_load(&bpe, "janus15m_bpe_merges.txt") < 0) {
        fprintf(stderr, "cannot load janus15m_bpe_merges.txt\n"); return 1;
    }
    g_bpe = &bpe;
    printf("bpe: %d merges, vocab %d\n", bpe.n_merges, bpe.vocab_size);

    Model* m = load_model(wpath);
    if (!m) return 1;
    precompute_wr_eff(m);
    printf("model: %s loaded, Wr_eff precomputed\n", wpath);

    build_stats_from_shard("train.bin");

    nt_seed((unsigned)time(NULL));
    srand((unsigned)time(NULL));

    int prompt[CTX];
    int plen = nt_bpe_encode(&bpe, seed, (int)strlen(seed), prompt, CTX/2);
    if (plen <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
    printf("prompt: \"%s\" (%d tokens)\n", seed, plen);
    printf("emit: temp=%.2f top_p=%.2f max=%d\n", temp, top_p, max_emit);
    printf("\n%s", seed);
    fflush(stdout);

    float logits[VOCAB];
    for (int i = 0; i < plen; i++) forward_step(m, prompt[i], i, logits);

    int history[CTX]; int hlen = 0;
    for (int i = 0; i < plen; i++) history[hlen++] = prompt[i];

    int pos = plen;
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    int emitted = 0;
    for (; emitted < max_emit && pos < CTX; emitted++) {
        int gen_step = emitted;
        /* force boundary at SENT_MAX_SOFT */
        if (gen_step >= SENT_MAX_SOFT) {
            int bt = find_byte_tok('.'); if (bt < 0) bt = find_byte_tok('!');
            if (bt >= 0) {
                int l = nt_bpe_decode(&bpe, &bt, 1, buf, NT_BPE_MAX_TOKEN_LEN);
                if (l > 0) { buf[l] = 0; printf("%s", buf); }
                break;
            }
        }
        float lbuf[VOCAB]; memcpy(lbuf, logits, VOCAB * sizeof(float));
        int next = sample(lbuf, VOCAB, temp, top_p, history, hlen, gen_step);
        if (hlen < CTX) history[hlen++] = next;
        int len = nt_bpe_decode(&bpe, &next, 1, buf, NT_BPE_MAX_TOKEN_LEN);
        if (len > 0) { buf[len] = 0; printf("%s", buf); fflush(stdout); }
        /* Break on first boundary past ≥ SENT_MIN_LEN emitted tokens */
        if (is_boundary(next) && emitted + 1 >= SENT_MIN_LEN) { emitted++; break; }
        if (pos >= CTX - 1) break;
        forward_step(m, next, pos, logits);
        pos++;
    }
    printf("\n");

    /* cleanup */
    nt_tensor** all = (nt_tensor**)malloc(n_expected_tensors() * sizeof(nt_tensor*));
    int i = 0;
    all[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        all[i++] = m->L[l].rms1;
        all[i++] = m->L[l].wq; all[i++] = m->L[l].wk; all[i++] = m->L[l].wv;
        all[i++] = m->L[l].wvr; all[i++] = m->L[l].wj; all[i++] = m->L[l].wo;
        all[i++] = m->L[l].wr_a; all[i++] = m->L[l].wr_b;
        all[i++] = m->L[l].rms2;
        all[i++] = m->L[l].w_gate; all[i++] = m->L[l].w_up; all[i++] = m->L[l].w_down;
        free(Wr_eff[l]);
    }
    all[i++] = m->rms_f; all[i++] = m->head;
    for (int j = 0; j < n_expected_tensors(); j++) nt_tensor_free(all[j]);
    free(all); free(m);
    return 0;
}
