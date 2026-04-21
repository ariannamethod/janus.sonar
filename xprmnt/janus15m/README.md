# xprmnt / janus15m — 16M Janus experiment on FineWeb-Edu

Experimental sibling of the sonar line. 16M params, BPE 4096, CTX 128, RRPRAM
low-rank R=64, single weights, triple attention. Trained on 30 MB FineWeb-Edu
as v1 — hit train best **2.94**, not the 1.5-2.0 target: 30 MB × 16 M ×
15 K steps × batch 1 landed between Chinchilla and anti-Chinchilla regimes.
v2 retrain planned on **5 MB quality-filtered slice** with peak LR 3e-4,
20 K steps — reproduces the Janus 285M-on-1.6MB anti-Chinchilla pattern
at a scale that fits an 8 GB Mac.

This folder has everything reproducible (scripts + C source + BPE merges +
training log). Weights (62 MB) and raw corpus (30 MB) are NOT in git —
regenerate via the scripts, or copy from the author's machine.

## Config (v1, trained)

```
DIM        320
LAYERS     8
HEADS      5
HEAD_DIM   64
HIDDEN     1024
CTX        128
VOCAB      4096   (HF ByteLevelBPE → notorch integer-pair merges)
RRPRAM_R   64     (Wr = Wr_a[H·DIM, R] @ Wr_b[R, CTX])
PARAMS     16,291,136
```

Triple attention per layer (MHA + RRPRAM low-rank + Janus Echo), equal 1/3
blend. RoPE on Q/K. Non-parametric RMSNorm, SwiGLU FFN. Single weights.
Chuck optimizer.

## v1 training result (this folder's state)

```
first    train  8.3453   (random ≈ 8.32)
best     train  2.9355
final    train  5.0072
final    val    5.0324
steps    15000
wall     9 h 9 min on Intel i5 2019 8 GB + Apple Accelerate
nans     0
steps/s  0.46 (after BLAS warm-up)
```

Full log: `notorch-train/train_v1.log`.

Diagnosis: 16 M × 8.74 M tokens × ≈3 epochs = 27 M tokens seen; Chinchilla
for 16 M is ≈320 M tokens. Dead zone. Fix in v2 by shrinking the corpus to
5 MB (quality-filtered FineWeb-Edu), density goes up 17×, 20 K steps give
1.7 epochs at 17× density — matches the prior Janus anti-Chinchilla pattern
(285 M × 1.6 MB converged).

## Reproduce v1

```bash
# 1. Download 30 MB FineWeb-Edu (outside this folder, into ../data/)
mkdir -p data
python3 scripts/download_fineweb.py

# 2. Train BPE 4096 via HF, export to notorch integer format
python3 scripts/train_bpe.py

# 3. Encode to uint16 shards
python3 scripts/encode_corpus.py

# 4. Link shards next to trainer
cd notorch-train
ln -sf ../data/train.bin train.bin
ln -sf ../data/val.bin   val.bin

# 5. Copy notorch from parent (or sibling)
cp ../../../notorch-train/notorch.c ../../../notorch-train/notorch.h .

# 6. Build
make

# 7. Train (14-16 h on Intel i5, expect 4-6 h on Mac neo A18 Pro)
./train_janus_15m 15000 6e-4

# 8. Sample
./infer_janus15m janus15m_v1.bin "The knock came three times" 80 0.75
```

## Plan v2 (next run)

- Shrink corpus: **5 MB** FineWeb-Edu, filter `score >= 3` (quality subset)
- Retrain BPE 4096 on the new slice
- Same architecture (16 M, DIM 320 / L 8 / RRPRAM R=64)
- Peak LR **3e-4** (not 6e-4)
- 20 000 steps
- Target: best train **< 1.5**, final ≈ 2.0, val ≈ 2.5-3.0
- ETA: 4-6 h on Mac neo

## Inference stack (`infer_janus15m.c`)

Port of the sonar-line inference stack to BPE 4096:

- `forward_step` with KV cache (K, V, Echo, Vr) — no tape, direct BLAS
- Pre-computed `Wr_eff = Wr_a @ Wr_b` once at load (no new op)
- **Dario field**: bigram 5.0 + trigram 3.0 + hebbian 0.4 + unigram hard
  floor (all built from `train.bin` uint16 — no re-tokenization of text)
- **Hard filters** (from neoleo): orphan fragment, capital-glue,
  apostrophe-glue, digit-glue, non-ASCII, hard word-gate when bigram
  and trigram both zero at alpha-alpha edge
- Bigram blocking 0.1×, Q-style age-based repetition penalty, count-crush
  on tokens seen ≥ 3 times
- Sentence structure: Capital-start hard filter at `gen_step == 0`,
  boundary-end break, forced boundary at `SENT_MAX_SOFT = 40`

Stack verified on v1 intermediate checkpoint — non-ASCII cleared, salad
class (Donceilerscreen / knocksoup / stretary / oCQatal-family) gone.
Grammar is still undertrained (v1 fault), structure clean.

Pending (to add on Mac neo): Codex-style systems layer — seed determinism,
5 mode presets (balanced / coherent / ritual / clinical / dialogue),
motif ledger with cross-step decay, closure critic best-of-5, regression
eval harness. All of this is already in the sibling `janus.sonar` infer;
the pattern transfers cleanly.

## Files in git

- `scripts/download_fineweb.py` — HF streaming, writes 30 MB raw text
- `scripts/train_bpe.py` — HF ByteLevelBPE train + export to notorch integer-pair
- `scripts/encode_corpus.py` — uint16 shard writer (train + val)
- `notorch-train/train_janus_15m.c` — trainer with RRPRAM R=64 composition
- `notorch-train/infer_janus15m.c` — forward_step + Dario field + filters
- `notorch-train/Makefile` — Darwin Accelerate / Linux OpenBLAS
- `notorch-train/janus15m_bpe_merges.txt` — 3840 notorch-format merges
- `notorch-train/train_v1.log` — full v1 training run (32 K + lines)

Not in git (regenerate or copy from author's machine):
- `data/fineweb_raw.txt` (30 MB)
- `data/train.bin`, `data/val.bin` (uint16 shards)
- `data/hf-vocab.json`, `data/hf-merges.txt` (HF tokenizer artifacts)
- `janus15m_v1.bin`, `janus15m_ckpt.bin` (62 MB each)
- Parent `notorch.c` / `notorch.h` (copy from `../../notorch-train/`)

## Session handoff

Full session narrative with hyperparameter rationale, Codex-vs-architect
systems gap, Opus-agent synthesis (Salvatore / Raffaele / Bruno / Kenji),
and first-moves for the next instance:
`~/Desktop/4dispatch/janus15m_session_2026_04_21_handoff.md`.

Parent sonar-line inference stack reference:
`github.com/ariannamethod/janus.sonar`, `notorch-train/infer_janus_sonar_chain.c`.
