# janus.sonar

> speech is not content. speech is the shape of the room the words walked through.

`janus.sonar` is a tiny resonant language organism: 3.11M parameters, 231 KB
of hand-written Sonar corpus, byte-level BPE, pure C, no PyTorch.

It is not a useful assistant. It is a pressure vessel for style, memory,
motif, broken grammar, and emotional sonar. The transformer learns the
sentence. The corpus field bends the sentence. The inference stack keeps the
dream from dissolving into BPE soup.

## Run It

```bash
cd notorch-train
make infer_janus_sonar_chain
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times"
```

Deterministic run:

```bash
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times" 123
```

Use these weights:

- `weights/sonar_single_v2.bin` — best pure-LM Sonar voice
- `weights/sonar_spa_v1.bin` — SPA contrastive sibling, slightly stranger arcs

`infer_janus_sonar.c` is legacy 4-layer dual inference. For the current 3M
weights, use `infer_janus_sonar_chain`.

## What It Is

Current model:

```text
DIM      160
LAYERS   6
HEADS    5
CTX      128
VOCAB    2048 byte-level BPE
PARAMS   3.11M
```

Architecture:

- MHA causal attention with RoPE on Q/K
- RRPRAM positional attention
- Janus Echo attention over `echo = x @ Wj`
- equal 1/3 attention blend
- RMSNorm
- SwiGLU FFN
- trained with `notorch`, a pure-C autograd stack

The corpus is `dataset_clean.txt`: 231 KB, 74,698 BPE tokens, 16 stylized
voices, speaker tags stripped, literary damage preserved.

## What Makes It Weird

The transformer alone is too small and starts hallucinating token glue.
So inference is a compound:

```text
theta = epsilon + gamma + alpha-delta
```

- `epsilon`: the trained 3M transformer
- `gamma`: corpus field, bigram/trigram/Hebbian/unigram statistics injected into logits
- `alpha-delta`: AML field, prophecy debt, entropy/resonance laws, Kuramoto chambers

Then the filters get brutal:

- no corpus-absent tokens
- no orphan BPE fragments
- no capital glue
- no digit glue
- no non-ASCII leak
- no newline tokens in generated sentences
- no BPE token that contains punctuation and then keeps talking
- minimum generated sentence length before punctuation can stop

The result is not “clean English”. It is Sonar English: doors, bone, soup,
signal, model, threshold, loss, love, architecture, the thing that cannot say
what it is.

## Weights

| weight | params | train best | val |
|---|---:|---:|---:|
| `sonar_single_v2.bin` | 3.11M | 0.4947 | 2.2331 |
| `sonar_spa_v1.bin` | 3.11M | 0.9660 | 2.6779 |
| `microjanus_single_10k.bin` | 1.57M | 1.22 | 2.70 |
| `microjanus_dual_sym_5k.bin` | 2.25M | 1.55 | 3.32 |
| `microjanus_dual_asym_5k.bin` | 2.25M | 1.84 | 3.36 |

Dual weights did not specialize on this corpus size. The 3M single model is
the useful Sonar line.

## Build And Test

```bash
cd notorch-train
make all
./test_sonar_ops
```

Expected:

```text
== OK: all gradients match within 0.010 ==
```

`test_sonar_ops` finite-diff verifies:

- `NT_OP_SEQ_ROW`
- `NT_OP_TRIPLET_LOSS`
- active and inactive triplet hinge regimes

## Train

```bash
cd notorch-train
make train_janus_sonar
./train_janus_sonar 10000 3e-4
```

SPA sibling:

```bash
make train_sonar_spa
./train_sonar_spa 10000 3e-4
```

Training was done on an 8 GB Intel i5 MacBook Pro with Apple Accelerate.
BLAS matters.

## Known Damage

- It is tiny.
- It is overfit by design.
- It is not factual.
- Word-level aphasia remains: `meas`, `inction`, `obser`, `catamean`.
- Sentence openers still collapse; `"A "` display surgery is intentional.
- SPA v1 scores chain resonance but does not yet inject back into hidden states.

## Layout

```text
dataset_clean.txt
weights/
notorch-train/
  notorch.c / notorch.h
  train_janus_sonar.c
  train_sonar_spa.c
  infer_janus_sonar_chain.c
  test_sonar_ops.c
SAMPLES.md
```

## Next

- 30M Janus
- richer corpus mix
- SPA v2 as gated hidden-state injection
- Griffin-style retention path
- browser demo of `forward_step`

---

## Credits / License

GPLv3.

Sonar corpus and architecture line by
[Oleg Ataeff](https://github.com/ariannamethod), with Claude Opus 4.7 in the
loop.

The inference stack borrows pressure from `postgpt`, `postgpt-q`, `neoleo`,
`haze`, `me`, and `notorch`.

The soup is never done.
