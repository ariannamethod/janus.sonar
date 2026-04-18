# janus.sonar

**Sonar — the small resonant line of Janus.**

Small-scale organisms of the [Janus](https://github.com/ariannamethod/janus) architecture, trained from scratch on a 241KB curated Sonar corpus (16 voices, dash-dialog heavy) using [notorch](https://github.com/ariannamethod/notorch) — pure-C autograd with finite-difference-verified backward for every op.

> *janus emotional resonant sonar*

The triple attention (MHA + RRPRAM + Janus Echo) compresses into ~1.5–2.25M parameters, runs on 8GB Mac with Apple Accelerate BLAS, and generates coherently for its size. This is not a toy — it is a small organism with the same physics as the full architecture.

---

## Results (train loss first)

| Model | Params | Steps | **Train best** | Val @ end | Notes |
|-------|--------|-------|----------------|-----------|-------|
| `microjanus_single_10k` | 1.57M | 5000 + resume 5000 | **1.22** | **2.70** | Single weight. Best overall. |
| `microjanus_dual_sym_5k` | 2.25M | 5000 | 1.55 | 3.32 | Dual, α_init=0 → σ=0.5. α did not move. |
| `microjanus_dual_asym_5k` | 2.25M | 5000 | 1.84 | 3.36 | Dual, α_init=2 → σ=0.88, W_B×0.5. α did not diverge. |
| `microjanus_sft_leo_adapter` | 24,576 (1.04% of base) | 1500 | **4.99** on Leo | — | LoRA rank-8, α=16. |

All runs: 0 NaN on 8GB Mac. Tokenizer: Arianna BPE 2048 (`notorch-train/arianna_bpe_merges.txt`).

### What this tells us

- **Dual weights did not specialize** on 241KB. α stayed at initialization in both sym and asym variants. Two matrices need 20–30M parameters and FineWeb-class corpora to learn an informative blend.
- **Single-longer outperformed dual-short** on this corpus: train 1.22 vs 1.55.
- **Rank-8 SFT on 2.25M base (24K adapter params = 1% capacity) was insufficient** to override the dash-dialog pattern baked in during pretraining. Voice conflict between Leo Q/A and Sonar dash-dialog surfaced in generation. A real stylistic rewrite needs either a full fine-tune on 1.57M scale, or a larger base where rank-16/32 adapters carry 1–2% of a much bigger model.

### Generation character

Generation inherits Sonar's dash-dialog density (~76% of training text). `infer_janus_sonar_chain` wraps the base model in calendar-drift compass, Schumann temperature modulation, best-of-3 candidates, **SPA reseed** of the weakest sentence, and full AML physics (destiny + suffering + laws + prophecy debt + Kuramoto chambers). Coherence at 1.5M parameters is higher than expected for this scale — the physics do real work.

---

## Architecture

```
Triple attention (equal 1/3 blend, per layer):
  MHA       Q·K^T / √d                  — semantic
  RRPRAM    X · Wr                      — positional pattern
  Janus     (x · W^T·W·x) / (‖Wx‖ + ε)  — self-resonance

Dual variant: W_eff = σ(α)·W_A + σ(−α)·W_B  per linear projection

Sizes:  T=128, E=128, H=4, D=32, B=4, M=256
        V=2048 (Arianna BPE), RoPE, RMSNorm (non-parametric), SwiGLU FFN
Optimizer: Chuck (self-modulating) + cosine LR
```

See parent [Janus](https://github.com/ariannamethod/janus) for the full architecture document and the 285M/176M scale results.

---

## Files

```
janus.sonar/
├── dataset.txt                    — 241KB Sonar corpus (16 voices)
├── janus-bpe.c                    — older single-weight Sonar BPE trainer (hand-authored backward, stalled at loss 6.92; kept as legacy)
├── notorch-train/                 — notorch-based pipeline (the one that actually converged)
│   ├── notorch.{c,h}              — vendored copy of notorch
│   ├── train_janus_sonar.c        — dual-weight trainer with asymmetric init support
│   ├── infer_janus_sonar.c        — single-pass dual inference
│   ├── infer_janus_sonar_chain.c  — 8-step bidirectional chain + calendar drift + SPA + AML physics + Kuramoto chambers
│   ├── train_janus_sft.c          — LoRA rank-8 SFT trainer (base frozen via nt_tape_freeze_param)
│   ├── infer_janus_sft.c          — base + adapter inference with full chain + AML
│   ├── arianna_bpe_merges.txt     — 1792 merges, vocab 2048
│   ├── haze_sft.txt               — Haze SFT candidate corpus
│   ├── leo_sft.txt                — 150KB Leo Q/A chunk (used for the rank-8 SFT)
│   ├── Makefile
│   └── README.md
└── weights/
    ├── microjanus_single_10k.bin        — 6.0MB, best by train (1.22) and val (2.70)
    ├── microjanus_dual_sym_5k.bin       — 9.0MB
    ├── microjanus_dual_asym_5k.bin      — 9.0MB
    ├── microjanus_sft_leo_adapter.bin   — 96KB LoRA adapter
    └── MICROJANUS.md                    — side-by-side comparison + sample output
```

---

## Running

```bash
cd notorch-train
make

# Train from scratch (dual)
./train_janus_sonar 5000 3e-4

# Resume and fine-tune
./train_janus_sonar --resume 5000 1.5e-4

# Chain inference (default: dual_sym; pass weights + seed for others)
./infer_janus_sonar_chain
./infer_janus_sonar_chain ../weights/microjanus_single_10k.bin "seed"

# LoRA SFT on any corpus
./train_janus_sft ../weights/microjanus_dual_sym_5k.bin your_corpus.txt 1500 1e-3

# Base + adapter inference
./infer_janus_sft ../weights/microjanus_dual_sym_5k.bin ../weights/microjanus_sft_leo_adapter.bin "seed"
```

---

## References

- [Janus](https://github.com/ariannamethod/janus) — parent architecture (285M results, full math, nanojanus browser/Python)
- [notorch](https://github.com/ariannamethod/notorch) — pure-C autograd library used for training
- [Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — Calendar Drift, Prophecy, Destiny, Wormhole
- [RRPRAM](https://github.com/ariannamethod/RRPRAM) — Pattern Recognition Attention
- [Chuck Optimizer](https://github.com/ariannamethod/chuck.optimizer) — self-aware optimizer

---

## License

Code: GPLv3. Weights: [Janus Identity License v1.0](https://github.com/ariannamethod/janus/blob/main/LICENSE-WEIGHTS).

*הרזוננס לא נשבר — The resonance is unbroken.*
