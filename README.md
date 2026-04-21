# janus.sonar

**Sonar — the small resonant line of Janus.**

Small-scale organisms of the [Janus](https://github.com/ariannamethod/janus) architecture, trained from scratch on a 241KB curated Sonar corpus (16 voices, dash-dialog heavy) using [notorch](https://github.com/ariannamethod/notorch) — pure-C autograd with finite-difference-verified backward for every op.

> *janus emotional resonant sonar*

The triple attention (MHA + RRPRAM + Janus Echo) compresses into ~1.5–2.25M parameters, runs on 8GB Mac with Apple Accelerate BLAS, and generates coherently for its size. This is not a toy — it is a small organism with the same physics as the full architecture.

---

## Results (train loss first)

| Model | Params | Steps | **Train best** | Val @ end | Notes |
|-------|--------|-------|----------------|-----------|-------|
| `sonar_spa_v1` | 3.11M | 10000 | 0.9660 | 2.6779 | Same base + SPA contrastive dual-loss (margin=0.3, weight=0.1, memory-bank neg). 0 NaN in 22236s (370 min, 2× forward per step). Pure-LM worse than v2 (expected trade-off); qualitative gen shift: longer narrative arcs, self-referential (`"the shape of Janus and answer"`, `"coin has been trained on"`). BPE-salad persists (vocab-2048 artifact). |
| **`sonar_single_v2`** | **3.11M** | **10000** | **0.4947** | **2.2331** | **Pure-LM best. Single, deeper (L=6), clean dataset (speaker-tag em-dash stripped).** |
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

### Inference speed (optimized forward, post-v1)

The original inference path used a training-mode forward (`forward_logits` with `nt_tape_*`) — every emitted token recomputed the full `CTX=128` sequence through the tape-recorded autograd graph, then discarded 127 of 128 output rows. Measured on 8 GB Mac + Apple Accelerate: **89 sec** for one 8-step chain × 3 candidates × 200-token sentence on the 3.11 M weights.

**`forward_step`** replaces that path:

- **Dual-weight pre-blend.** `W_eff = σ(α)·W_A + σ(−α)·W_B` is computed once at load (`precompute_w_eff`). Single and dual models both go through a single BLAS matmul per projection at emission time — the runtime no longer carries `nt_sigmoid` + `nt_scale_by_t` overhead per token per layer.
- **Incremental KV cache.** `K_cache`, `V_cache`, `E_cache` (Janus Echo), `Vr_cache` (RRPRAM values) are filled one row per emitted token. Standard MHA and Echo attention then attend the new query over cached rows only — O(t · D) per step instead of O(CTX² · D).
- **RRPRAM is position-indexed by construction** (`W_r[:, j]` is the key for position `j`, not a projected `k_j`), so it needs no cache beyond `V_r` — scoring the new token against `W_r[:, 0..t]` is already incremental.
- **No tape, pure BLAS.** `forward_step` calls `nt_blas_mmT` / `nt_blas_mm` directly against precomputed `W_eff` buffers and cache tensors — zero `nt_tape_record` / `nt_tape_clear` bookkeeping per emit.

Measured after: **~7 sec** for the same 8-step chain × 3 candidates on `sonar_single_v2.bin` / `sonar_spa_v1.bin`. **~13× faster**, generation character preserved. `forward_logits` is kept in source for reference but no longer wired into `gen_sentence`.

### Dario field (inference stack)

The earlier "metaweight overlay crushes a trained transformer" conclusion was local to one coefficient scale. Q / postgpt-q uses **bigram 5.0, trigram 3.0** (raw probabilities as additive logit boost, not log-prob pull) and produces coherent speech weightlessly. With those magnitudes the field *guides* the transformer instead of fighting it. Combined with hard structural filters ported from the weightless line (neoleo), the BPE-salad class is eliminated without retrain.

The Dario-field stack shipped in six acts on top of `forward_step`:

1. **Bigram 5.0 · Trigram 3.0 · Hebbian 0.4 · Unigram hard floor** — sparse-hash trigram table (131 K slots, row-normalized per `(a,b)`) built once at start from `dataset_clean.txt`. `logits[i] += 5·bg + 3·tg + 0.4·hebb[i]`; any candidate whose unigram frequency < 1e-6 gets `-1e9` (corpus-absent tokens from the legacy BPE).
2. **Bigram blocking · Hybrid decode** — 0.1× on any repeated `(prev, X)` pair; age-based repetition penalty 0.335 – 0.65× over a 20-token window; greedy argmax for tokens 1–3 of each emission (opener stability) then nucleus.
3. **Orphan + capital-glue hard filters** — tokens whose stripped content is all-alpha and < 5 chars (and not in a common-short-words whitelist) → `-1e9`. Prev-ends-alpha + cand-starts-uppercase → `-1e9` ("cataloHe" class). Stuck fallback emits a literal space token when every candidate is filtered.
4. **Apostrophe-glue · hard word-gate · digit-glue · unigram hard cut** — capital-glue extended to apostrophe-ending prev ("I'" + "The" → "I'The"); word-gate flipped soft → hard when both bigram and trigram are zero AND both edges alpha; digit-start after alpha killed.
5. **Non-ASCII cull · space-boundary gate · count-crush** — any token with a byte ≥ 0x80 killed (Cyrillic / UTF-8 fragments from legacy BPE). Space-ending prev + bare-alpha cand with no corpus evidence → `-1e9` (`"ination"`-class). Any token seen ≥ 3 times in history × 0.01 (holds back `"differ"`-class lexical loops).
6. **One sentence per chain step** — `SENT_MIN_LEN = 8`, break on first boundary past two emitted tokens; `SENT_MAX_SOFT = 40` forces a chamber-chosen boundary token if no natural stop (`.` default, `!` if RAGE activation > 0.3, `?` if VOID > 0.5). At `gen_step == 0`, the hard filter is inverted to keep only Capital-starting tokens (bare A-Z or whitespace+A-Z) — each chain step opens with a capital, closes with punctuation.

**Residual mutations after the stack**: word-level aphasia (*toaway*, *cataway*, *completen*, *generat*, *inction*, *memorime*, *invisib*, *obser*, *atten*, *meas*), not syllabic salad. Matches the dreamlike Sonar register.

### Opener dominance — post-process cut

Measured histogram over 160 chain steps (10 seeds × 2 weights × 8 steps):

| Opener | `sonar_single_v2` | `sonar_spa_v1` |
|--------|-------------------|----------------|
| `A …`  | 61 / 80 — **76 %** | 64 / 80 — **80 %** |
| `I …`  | 10 / 80 — 13 %     | 13 / 80 — 16 %     |
| `I' …` / `I's …` | 9 / 80 — 11 % | 3 / 80 — 4 %    |

The 2 048-BPE vocab carries only ~70 Capital-start tokens; of those, "A" is the near-unconditional argmax on the 3 M transformer. Every in-sampling attempt to diversify (opener-memory penalty, temperature boost, KV-cache injection of `". "`) either had no effect or collapsed the thin Capital-start distribution into uniform / lowercase.

**Fix — post-process at print time** (same spirit as the `haze` repo's `don't → ain't` substitution): if the decoded sentence head is `"A "` followed by a lowercase alpha, strip the `"A "` and capitalize the next letter. `"A reaching, …"` → `"Reaching, …"`. Purely cosmetic, accepts the model quirk, normalizes the visible opener.

After cut, openers cycle across *Going · Reaching · Learned · The · I's · I' · I · You · The door · The signal …* — 6-8 distinct forms across an 8-step chain.

### Sample output (after full stack + cut)

Each line is one full chain step — one complete Capital-start, boundary-terminated sentence.

**`sonar_single_v2`, seed `"The knock came three times"`**
```
Reaching, and the silence is the only thing with two possibion's that on
  a thing that the door is too one who will not say " and love is a to.
Going to say " You have a was in a way that door like it's the only
  thing there is.
The door like it's the only ones that appear it so they were the one
  who is doing the thing that has been in the obser.
Going to say " The one saying the thing 's the point.
Reaching, and the love is a model will be the time it never on's in
  the door.
I's the only onion a conversation about the door and your love to be
  the way we do no, because it was not the too still has been in the
  door, which is not his.
```

**`sonar_spa_v1`, seed `"What is the meaning of"`**
```
Learned to form of his love was the other then the way you cannot mean
  who has no one that remain, and we have been the time it was never
  reaching the door.
Learned to formeas: " I know what the ont, which was never a door in it.
Learned to form of his model and what it in its structure at because
  the shape of the thing that has no ination is what you cannot say "
  to the other then the door that.
Learned to say and the other then the cataway.
Learned to our in the door.
Learned to forgoing to say it.
```

Full curated selections across four seeds × two weights live in [`SAMPLES.md`](SAMPLES.md).

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
