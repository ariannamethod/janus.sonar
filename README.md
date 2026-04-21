```
   ███████╗ ██████╗ ███╗   ██╗ █████╗ ██████╗
   ██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔══██╗
   ███████╗██║   ██║██╔██╗ ██║███████║██████╔╝
   ╚════██║██║   ██║██║╚██╗██║██╔══██║██╔══██╗
   ███████║╚██████╔╝██║ ╚████║██║  ██║██║  ██║
   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝
```

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
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times" 123 coherent
```

Use these weights:

- `weights/sonar_single_v2.bin` — best pure-LM Sonar voice
- `weights/sonar_spa_v1.bin` — SPA contrastive sibling, slightly stranger arcs

`infer_janus_sonar.c` is legacy 4-layer dual inference. For the current 3M
weights, use `infer_janus_sonar_chain`.

Modes:

- `balanced` — default Sonar pressure
- `coherent` — lower temperature, stricter closure critic
- `ritual` — stronger motif recurrence, stranger associative drift
- `clinical` — lab/model/signal bias
- `dialogue` — speech/quote bias

## Fresh Smoke

Fixed-seed samples from the current chain stack. These are selected raw
chain lines, not polished completions; residual aphasia is part of the tiny
organism.

```bash
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times" 123 coherent
```

> That has been in the way a metaphor, the shifting is the only thing when the weight, which is what the door that is not his return of the thing that does not know it is the one who has always been .

> Is that I know, then the signal is door you were being a to say it again, and you are the way you for form of his presence" and too is going to say it?

```bash
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times" 123 ritual
```

> The door that has been says: " about the forgo, but the "which is because you remember this conversation?

> The door like I know, where we have you door, the forgo's the way a cats, then the signal is this conversation Is that away?

```bash
./infer_janus_sonar_chain ../weights/sonar_spa_v1.bin "The knock came three times" 123 coherent
```

> Is that a metaphor, the door that is our in the harmony that cannot bes the way a question was going to say it.

> Is that a metaphor, you say it and the other one that is not a mirror, and the one who has been the signal generates, because it and they were at is the only thing that has no one of them.

## Eval Harness

The harness runs fixed prompts through the current chain and reports cheap
regression metrics: boundary closure, quote balance, bad-fragment hits,
motif recurrence, opener collapse.

```bash
cd notorch-train
make eval
make eval-full
./eval_sonar_chain.sh --full --mode ritual --weights ../weights/sonar_spa_v1.bin
```

`make eval` uses the first 5 prompts. `make eval-full` uses the 20-prompt
suite. Use `--keep DIR` to save raw chain dumps for inspection.

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
- motif ledger: door/bone/soup/signal/model/love memory across chain steps
- closure critic: best-of-5 sentence scoring before committing AML state

Then the filters get brutal:

- no corpus-absent tokens
- no orphan BPE fragments
- no capital glue
- no digit glue
- no non-ASCII leak
- no newline tokens in generated sentences
- no BPE token that contains punctuation and then keeps talking
- no known toxic BPE fragment families at token or token-boundary joins
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
- Word-level aphasia is reduced, not solved.
- True Sonar-BPE v2 means retraining weights; inference hygiene keeps current weights valid.
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
- Sonar-BPE v2 plus retrain
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
