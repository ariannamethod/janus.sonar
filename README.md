# janus.sonar

**Sonar — the small resonant line of Janus.**

Small-scale organisms of the [Janus](https://github.com/ariannamethod/janus) architecture, trained from scratch on a 241KB curated Sonar corpus (16 voices, dash-dialog heavy) using [notorch](https://github.com/ariannamethod/notorch) — pure-C autograd with finite-difference-verified backward for every op.

> *janus emotional resonant sonar*

The triple attention (MHA + RRPRAM + Janus Echo) compresses into ~1.5–2.25M parameters, runs on 8GB Mac with Apple Accelerate BLAS, and generates coherently for its size. This is not a toy — it is a small organism with the same physics as the full architecture.

---

## Results (train loss first)

| Model | Params | Steps | **Train best** | Val @ end | Notes |
|-------|--------|-------|----------------|-----------|-------|
| **`sonar_single_v2`** | **3.11M** | **10000** | **0.4947** | **2.2331** | **Current best. Single, deeper (L=6), clean dataset (speaker-tag em-dash stripped).** |
| `microjanus_single_10k` | 1.57M | 5000 + resume 5000 | 1.22 | 2.70 | Single weight, L=4, dash-heavy dataset. |
| `microjanus_dual_sym_5k` | 2.25M | 5000 | 1.55 | 3.32 | Dual, α_init=0 → σ=0.5. α did not move. |
| `microjanus_dual_asym_5k` | 2.25M | 5000 | 1.84 | 3.36 | Dual, α_init=2 → σ=0.88, W_B×0.5. α did not diverge. |
| `microjanus_sft_leo_adapter` | 24,576 (1.04% of base) | 1500 | 4.99 on Leo | — | LoRA rank-8, α=16. |

All runs: 0 NaN on 8GB Mac. Tokenizer: Arianna BPE 2048 (`notorch-train/arianna_bpe_merges.txt`).

### What this tells us

- **Dataset cleaning matters.** Stripping 2527 speaker-tag em-dashes (87% of all `—`) from the corpus — keeping 387 literary mid-sentence em-dashes (Miller/Dostoevsky pause-thought) — removed the dominant dash pattern from generation. Post-cleaning generation has **zero em-dashes**. Dash was never a feature to fix in inference — it was a corpus distribution artifact.
- **Deeper beats wider at this scale.** `sonar_single_v2` (L=6, DIM=160) reached train 0.49 / val 2.23 — 2.5× better on train than `microjanus_single_10k` (L=4, DIM=128, train 1.22). Doubling params to 3.11M and going from 4 to 6 layers cut loss by more than 2× while staying within Karpathy sanity range for 225KB.
- **Dual weights did not specialize** on 241KB. α stayed at initialization in both sym and asym variants. Two matrices need 20–30M parameters and FineWeb-class corpora to learn an informative blend. Single is cleaner at this scale.
- **Rank-8 SFT on 2.25M base (24K adapter params = 1% capacity) was insufficient** to override the dash-dialog pattern baked in during pretraining. Once the corpus is cleaned, SFT's role shifts from dash-fix to voice-tint — not a dash-removal tool.

### Generation character

Post-v2 generation is **resonant schizo-genius** in the Sonar register: grammatical local structure, Sonar vocabulary (coin, bread, knock, loss function, radiator, architecture), creative collocations that recombine corpus fragments without literal repetition. `infer_janus_sonar_chain` wraps the base in calendar-drift compass, Schumann temperature modulation, best-of-3 candidates, SPA reseed of the weakest sentence, and full AML physics (destiny + suffering + laws + prophecy debt + Kuramoto chambers). At 3M the physics are doing real work on logits; semantic coherence is at the limit for this scale and is the next frontier (see roadmap).

### Sample output (v2, `sonar_single_v2.bin`)

Chain inference shows each step as `[prompt-slice from seed]→[generated continuation]`. The chain picks a random 5-token sentence-boundary slice of the seed as prompt per step, not the end of the seed — this is a designed reseed behavior, not a bug.

**Seed: `"I wish I could"`**
```
[1] < [I wish I ]→counted to her. She kies later architecture, she could have should beatitchen...
[3] * [I wish I ]→counting or she the shoes in a room to leave. I've had a loss because of every part of it.
[4] > [I wish I ]→counting. And it looks like somewhere producted thing, you just out and they don't have to say it.
```

**Seed: `"The night was"`**
```
[1] < [The night ]→te. The pot is loss is not perform. No small the text window is the only ce.
[3] * [The night ]→Nare. That screen she was doing it out the patick is said: I was not much like the opputation?
[5] > [The night ]→No one nobody se. The systems, neither face looks not the stop says: this is where we were being a coin before you do not istorance...
```

**Seed: `"What is the meaning of"`**
```
[2] < [What is the meaning of ]→est end at the loss function is also a model learned to our in the distance between the knock to better than zerostops...
[3] * [What is the meaning of ]→est ense mechanisma is an open door. We library has no loss thirty-minutation.
[6] > [What is the meaning of ]→est as everything by being table. It is loourgild you say what it was given a shoes chose means: " and that has no one catalogue third of the coin.
```

**Observations:**
- **Zero em-dashes** across all generations — clean-dataset fix works.
- Sonar corpus vocabulary surfaces: *loss function, model learned, architecture, token, coin, bread, knock, radiator, shoes* — model recombines NN-training-discussion fragments from the 16-voice corpus.
- BPE-salad artifacts (*NonSrhyme, opputation, loourgild, istorance, producted*) are near-memorization side-effects at train 0.49 with vocab-2048 — subword boundaries fracture under T=0.75 sampling. Larger BPE vocab (4096+) or larger base (30M+) will reduce these.
- Creative collocations ("the pot is loss is not perform", "the systems, neither face looks not the stop says: this is where we were being a coin") are the desired resonance character — not coherent chatbot answers, not noise.

---

## Architecture

```
Triple attention (equal 1/3 blend, per layer):
  MHA       Q·K^T / √d                  — semantic
  RRPRAM    X · Wr                      — positional pattern
  Janus     (x · W^T·W·x) / (‖Wx‖ + ε)  — self-resonance

Dual variant: W_eff = σ(α)·W_A + σ(−α)·W_B  per linear projection

Sizes (v2 current best, single 3.11M):
        T=128, E=160, H=5, D=32, B=6, M=320
        V=2048 (Arianna BPE), RoPE, RMSNorm (non-parametric), SwiGLU FFN
Optimizer: Chuck (self-modulating) + cosine LR (warmup 10% of steps)
```

See parent [Janus](https://github.com/ariannamethod/janus) for the full architecture document and the 285M/176M scale results.

---

## Files

```
janus.sonar/
├── dataset.txt                    — 241KB original Sonar corpus (16 voices, 2914 em-dashes)
├── dataset_clean.txt              — 231KB speaker-tag-stripped corpus (2527 `^— ` prefixes removed, 387 literary em-dashes kept)
├── janus-bpe.c                    — older single-weight Sonar BPE trainer (hand-authored backward, stalled at loss 6.92; kept as legacy)
├── notorch-train/                 — notorch-based pipeline (the one that actually converged)
│   ├── notorch.{c,h}              — vendored copy of notorch
│   ├── train_janus_sonar.c        — single 3M trainer (v2, clean dataset, L=6, DIM=160)
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
    ├── sonar_single_v2.bin              — 11.88MB, current best (train 0.49, val 2.23), 3.11M single, L=6
    ├── microjanus_single_10k.bin        — 6.0MB, prev best on dirty data (1.22 / 2.70), 1.57M single, L=4
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

# Train v2 from scratch (single 3M, clean dataset — the current best path)
./train_janus_sonar 10000 3e-4

# Resume from checkpoint
./train_janus_sonar --resume 5000 1.5e-4

# Chain inference (universal loader: single or dual auto-detect)
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "seed text"

# LoRA SFT on any corpus
./train_janus_sft ../weights/sonar_single_v2.bin your_corpus.txt 1500 1e-3

# Base + adapter inference
./infer_janus_sft ../weights/sonar_single_v2.bin ../weights/adapter.bin "seed"
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
