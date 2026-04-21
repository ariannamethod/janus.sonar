/*
 * test_sonar_ops.c — finite-difference verification for SPA ops.
 *
 *   NT_OP_SEQ_ROW       — row pick from [T, D] sequence
 *   NT_OP_TRIPLET_LOSS  — contrastive loss with margin + hinge
 *
 *   make test_sonar_ops
 *   ./test_sonar_ops
 *
 * Passes quietly, fails loudly.
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPS_FD     1e-3f      /* single-precision finite-diff step */
#define TOL_FD     1e-2f      /* absolute tolerance on grad match */

static int g_failures = 0;

static void copy_grad_or_zero(int tape_idx, float* dst, size_t bytes) {
    nt_tape_entry* e = &nt_tape_get()->entries[tape_idx];
    if (e->grad) memcpy(dst, e->grad->data, bytes);
    else memset(dst, 0, bytes);
}

static void check_grad(const char* label, int idx, float expected, float actual) {
    float d = fabsf(expected - actual);
    float scale = fabsf(expected) > 1 ? fabsf(expected) : 1.0f;
    if (d > TOL_FD * scale && d > TOL_FD) {
        printf("  FAIL  %-30s idx=%2d  num=%+.5f  bwd=%+.5f  |d|=%.5f\n",
               label, idx, expected, actual, d);
        g_failures++;
    }
}

/* ───────────── seq_row test ─────────────
   x is a tape param [T*D] acting as a flat [T, D] sequence.
   loss = triplet_loss(seq_row(x, row=1), pos, neg, margin)
   We finite-diff every entry of x and check backward gradients.
*/
static void test_seq_row(void) {
    const int T = 3, D = 4, target_row = 1;
    const float margin = 0.5f;

    float x_data[T*D];
    for (int i = 0; i < T*D; i++) x_data[i] = 0.1f * (i + 1) - 0.3f;  /* varied */
    float p_data[D] = { 0.2f, -0.1f, 0.3f,  0.05f };
    float n_data[D] = { 0.1f,  0.2f, -0.2f, 0.15f };

    /* Backward pass: get analytical grads */
    float ana_gx[T*D];
    {
        nt_tape_start();
        nt_tensor* x = nt_tensor_new(T*D); memcpy(x->data, x_data, sizeof(x_data));
        nt_tensor* p = nt_tensor_new(D);    memcpy(p->data, p_data, sizeof(p_data));
        nt_tensor* n = nt_tensor_new(D);    memcpy(n->data, n_data, sizeof(n_data));
        int xi = nt_tape_param(x);
        int pi = nt_tape_param(p);
        int ni = nt_tape_param(n);
        int yi = nt_seq_row(xi, target_row, D);
        int li = nt_triplet_loss(yi, pi, ni, margin);
        nt_tape_backward(li);
        memcpy(ana_gx, nt_tape_get()->entries[xi].grad->data, sizeof(ana_gx));
        nt_tensor_free(x); nt_tensor_free(p); nt_tensor_free(n);
        nt_tape_clear();
    }

    /* Finite-diff each x[i] */
    for (int i = 0; i < T*D; i++) {
        float saved = x_data[i];
        float l_hi = 0, l_lo = 0;

        for (int side = 0; side < 2; side++) {
            x_data[i] = saved + (side == 0 ? EPS_FD : -EPS_FD);
            nt_tape_start();
            nt_tensor* x = nt_tensor_new(T*D); memcpy(x->data, x_data, sizeof(x_data));
            nt_tensor* p = nt_tensor_new(D);    memcpy(p->data, p_data, sizeof(p_data));
            nt_tensor* n = nt_tensor_new(D);    memcpy(n->data, n_data, sizeof(n_data));
            int xi = nt_tape_param(x);
            int pi = nt_tape_param(p);
            int ni = nt_tape_param(n);
            int yi = nt_seq_row(xi, target_row, D);
            int li = nt_triplet_loss(yi, pi, ni, margin);
            float v = nt_tape_get()->entries[li].output->data[0];
            if (side == 0) l_hi = v; else l_lo = v;
            nt_tensor_free(x); nt_tensor_free(p); nt_tensor_free(n);
            nt_tape_clear();
        }
        x_data[i] = saved;

        float num = (l_hi - l_lo) / (2 * EPS_FD);
        /* Expectation: only the target_row's positions should have non-zero grad. */
        int row = i / D;
        if (row != target_row) {
            check_grad("seq_row.grad_x[other]", i, 0.0f, ana_gx[i]);
            check_grad("seq_row.num[other]   ", i, 0.0f, num);
        } else {
            check_grad("seq_row.grad_x[target]", i, num, ana_gx[i]);
        }
    }
}

/* ───────────── triplet_loss test ─────────────
   Three independent [D] tensors as params. Verify grads under BOTH active
   and inactive hinge regimes.
*/
static void test_triplet(const char* label, float a_scale, float p_scale, float n_scale) {
    const int D = 5;
    const float margin = 0.3f;
    float a_data[D], p_data[D], n_data[D];
    for (int i = 0; i < D; i++) {
        a_data[i] = a_scale * (0.1f * (i + 1) - 0.2f);
        p_data[i] = p_scale * (0.2f * (i + 1) - 0.4f);
        n_data[i] = n_scale * (-0.1f * (i + 1) + 0.3f);
    }

    float ana_ga[D], ana_gp[D], ana_gn[D];
    float baseline_loss = 0;

    {
        nt_tape_start();
        nt_tensor* a = nt_tensor_new(D); memcpy(a->data, a_data, sizeof(a_data));
        nt_tensor* p = nt_tensor_new(D); memcpy(p->data, p_data, sizeof(p_data));
        nt_tensor* n = nt_tensor_new(D); memcpy(n->data, n_data, sizeof(n_data));
        int ai = nt_tape_param(a);
        int pi = nt_tape_param(p);
        int ni = nt_tape_param(n);
        int li = nt_triplet_loss(ai, pi, ni, margin);
        baseline_loss = nt_tape_get()->entries[li].output->data[0];
        nt_tape_backward(li);
        copy_grad_or_zero(ai, ana_ga, sizeof(ana_ga));
        copy_grad_or_zero(pi, ana_gp, sizeof(ana_gp));
        copy_grad_or_zero(ni, ana_gn, sizeof(ana_gn));
        nt_tensor_free(a); nt_tensor_free(p); nt_tensor_free(n);
        nt_tape_clear();
    }

    printf("  %s: baseline_loss=%+.5f (active=%s)\n",
           label, baseline_loss, baseline_loss > 0 ? "yes" : "no");

    /* Finite-diff all three tensors */
    float* probes[3]    = { a_data, p_data, n_data };
    float* ana_grads[3] = { ana_ga, ana_gp, ana_gn };
    const char* names[3] = { "a", "p", "n" };

    for (int which = 0; which < 3; which++) {
        for (int i = 0; i < D; i++) {
            float saved = probes[which][i];
            float l_hi = 0, l_lo = 0;
            for (int side = 0; side < 2; side++) {
                probes[which][i] = saved + (side == 0 ? EPS_FD : -EPS_FD);
                nt_tape_start();
                nt_tensor* a = nt_tensor_new(D); memcpy(a->data, a_data, sizeof(a_data));
                nt_tensor* p = nt_tensor_new(D); memcpy(p->data, p_data, sizeof(p_data));
                nt_tensor* n = nt_tensor_new(D); memcpy(n->data, n_data, sizeof(n_data));
                int ai = nt_tape_param(a);
                int pi = nt_tape_param(p);
                int ni = nt_tape_param(n);
                int li = nt_triplet_loss(ai, pi, ni, margin);
                float v = nt_tape_get()->entries[li].output->data[0];
                if (side == 0) l_hi = v; else l_lo = v;
                nt_tensor_free(a); nt_tensor_free(p); nt_tensor_free(n);
                nt_tape_clear();
            }
            probes[which][i] = saved;

            float num = (l_hi - l_lo) / (2 * EPS_FD);
            char tag[64];
            snprintf(tag, sizeof(tag), "triplet.%s.d/d%s[i]", label, names[which]);
            check_grad(tag, i, num, ana_grads[which][i]);
        }
    }
}

int main(void) {
    nt_seed(1);
    printf("== SPA ops finite-diff verification ==\n");

    printf("[seq_row]\n");
    test_seq_row();

    printf("[triplet_loss — ACTIVE regime (margin forces hinge > 0)]\n");
    test_triplet("active", 1.0f, -1.0f, 1.0f);

    printf("[triplet_loss — INACTIVE regime (pos much more aligned than neg)]\n");
    test_triplet("inactive", 1.0f, 4.0f, -1.0f);

    if (g_failures == 0) {
        printf("\n== OK: all gradients match within %.3f ==\n", TOL_FD);
        return 0;
    } else {
        printf("\n== FAIL: %d gradient mismatches ==\n", g_failures);
        return 1;
    }
}
