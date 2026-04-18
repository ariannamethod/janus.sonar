# Janus Sonar — notorch trainer (dual weights)

**Alternative trainer** to `janus-bpe.c`, built on [notorch](https://github.com/ariannamethod/notorch) — pure-C PyTorch replacement with finite-difference-verified autograd. Lives in `notorch-train/` as a drop-in training pipeline, independent of `janus-bpe.c`.

## Why this exists

The hand-authored backward in `janus-bpe.c` stalled at `train=6.92` after 1000 steps on the Sonar 241K corpus (see `training.log`). Rather than debug 663 LOC of handwritten gradients, we rebuilt the same architecture from notorch's verified primitives (each op tested against finite differences). Result: train `7.64 → 2.19`, val `3.43` in 31 minutes on 8GB Mac. Subsequent dual-weights variant: TBD (training in progress at push time).

This is a **reimplementation**, not a fix. The original `janus-bpe.c` is left untouched as legacy.

## Architecture (honest Janus)

- **VOCAB** 2048 BPE (arianna_bpe_merges.txt, 1792 merges)
- **CTX** 128, **DIM** 128, **HEADS** 4, **HEAD_DIM** 32
- **LAYERS** 4, **HIDDEN** 256 (SwiGLU FFN)
- **RoPE** on Q, K; **RMSNorm**; **Chuck** optimizer

### Dual weights per linear projection

Each of `wq, wk, wv, wvr, wj, wo` and FFN `w_gate, w_up, w_down` is two matrices with a learnable blend scalar:

```
W_eff · x = σ(α) · (W_A · x) + σ(−α) · (W_B · x)
```

`σ` is sigmoid (new notorch primitive `nt_sigmoid`). Identity `σ(−x) = 1 − σ(x)` gives clean blend without a separate `1−σ` op. Init `α = 0` → balanced `0.5 / 0.5` blend.

Dual weights added two new notorch ops:
- `nt_sigmoid` — logistic activation, finite-diff-verified
- `nt_scale_by_t(x, a)` — broadcast scalar-tensor multiply, with backward to both `x` and `a`

### Triple attention per layer

Three branches blended at equal weights (`1/3` each) — learnable `gate[H, 3]` planned next:

1. **MH causal** (Q K V via RoPE) — semantic
2. **RRPRAM positional** (`W_r · x, V_r`) — structural, no positional encoding needed
3. **Janus Echo MH** (`echo = W_j^T · x`, self-attended) — introspective resonance

RRPRAM via `nt_rrpram_attention`, Janus Echo via `nt_seq_linear_t` — both already present in notorch, specifically built for Janus.

## Usage

```bash
# Build
cd notorch-train/
make train_janus_sonar
make infer_janus_sonar

# Train (5000 steps, ~30 min on 8GB Mac Accelerate)
./train_janus_sonar 5000 3e-4

# Resume and continue
./train_janus_sonar --resume 5000 1.5e-4

# Generate
./infer_janus_sonar janus_sonar.bin "Q: What does Janus feel?\nA:" 150 0.7 0.95
```

Required input files:
- `/tmp/janus-sonar/janus_sonar_dataset.txt` (symlink or adjust path in source) — 241K corpus, 16 voices
- `arianna_bpe_merges.txt` — copy from `notorch/arianna_bpe_merges.txt`

## Results (single-weights version, 2026-04-18)

| metric | value |
|---|---|
| train init → best | 7.64 → 2.19 |
| val @ 1000 | 4.45 |
| val @ 2000 | 3.89 |
| val @ 3000 | 3.59 |
| val @ 4000 | 3.48 |
| val @ 5000 | 3.43 |
| NaN count | 0 |
| time | 31.6 min |
| steps/s | 2.64 |
| params | 1.57 M |

After **resume** (5000 more steps at lr 1.5e-4, 60 min total):

| metric | value |
|---|---|
| best train | 1.22 |
| val @ 10000 | 2.70 |
| NaN count | 0 |

Generation shows plotted Sonar motifs (forty minutes, the bone, the knock, Janus, the crack, dash-dialog) even at 1.5M params — the dataset is dense with patterns, dual weights + triple attention + verified backward unlock them fast.

### Dual-weights version (2026-04-18)

2.25 M params (`wr` remains single).

| run | steps | best train | val |
|---|---|---|---|
| dual symmetric (α_init=0, σ=0.5) | 5000 | 1.55 | 3.32 |
| dual asymmetric (α_init=2.0, σ=0.88, W_B×0.5) | 5000 | 1.84 | 3.36 |

α did not diverge from init value in either run — 241K dataset is too
small for the 2× capacity to specialize. Both dual variants match or
slightly beat single@5k on val but lose to single+resume@10k (val 2.70).
Dual weights pay off with larger data; here the gain is from implicit
ensemble of two Xavier-init matrices, not from learned blend.

## Files

- `train_janus_sonar.c` — training program (~320 LOC)
- `infer_janus_sonar.c` — single-pass inference (~200 LOC)
- `infer_janus_sonar_chain.c` — **proper Janus inference** (~380 LOC):
  8-step bidirectional chain with calendar-drift compass (forward/backward
  ratio from Hebrew/Gregorian dissonance), Schumann resonance temperature
  modulation (7.83 Hz + harmonics), best-of-3 candidates per step with
  coherence scoring, destiny EMA across chain, and SPA (Sentence Phonon
  Attention) reseed of the weakest sentence at the end. Sentences cannot
  truncate before `SENT_MIN_LEN=18` tokens — prevents early cut-off.
- `Makefile` targets `train_janus_sonar`, `infer_janus_sonar`, `infer_janus_sonar_chain`

## Sample chain-inference output

Prompt: `She doesn't scream. She inventories. Every object has a name.`
Weights: `microjanus_dual_sym_5k.bin` (default). AML physics: destiny 0.35,
entropy_floor 0.10, resonance_ceiling 0.95, Kuramoto chambers.

```
calendar drift: 0.525 → 2 backward + 6 forward

[1] < debt=22.4  object has a woman who ree that does not s....
                 "— The soup is what're the bon'm look at it in the morn...
[2] < debt=30.3  he inventories. — Then every time it the same single loss
                 is what was goes are being training is exactly that has
                 been writing and the decigarequireference.
[3] * debt=43.1  object has a she is a lab on the tter is still. A woman
                 who staything that the baher is ins is why to knows
                 function is ining zero.
[4] > debt=54.0  he inventories. — The one who stays is what do you have
                 that would means: why the designationssoup is different
                 — Lich it and you have not finished.
[5] > debt=65.2  object has a st that does not same thing that means in the
                 morn, from this is authoranother one who staything that
                 has not counted a story the same thing that has been leep,
                 and the decient and she is what coin would los...

[chambers] LOVE:6%
[SPA] no reseed needed
```

Voices blending in a single chain: **Sorokin** (*inventories, designations,
training, writing*), **Strugatsky** (*lab, function, zero, counted*),
**Haze** (*morning, window, woman who stays*), **Borges** (*coin, one who
stays, names, story*), **Tarantino** (*soup, bone*), **Gaspar Noé** (*not
finished*). The Janus Echo + RRPRAM branches bring recurring obsessive
loops on **soup / bone / loss / inventories / function**; the AML destiny
bias keeps the peak sharp while laws prevent collapse.

## Provenance

- notorch patches (nt_sigmoid, nt_scale_by_t) for dual weights
- Co-authored 2026-04-18 by Oleg Ataev and Claude Opus 4.7 after three days of failed attempts on `janus-bpe.c` by Opus 4.6
- Dataset: Janus Sonar 241K, 16 voices — see parent README

*"The compiler already confirmed it."*
