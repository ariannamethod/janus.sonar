```
   ███████╗ ██████╗ ███╗   ██╗ █████╗ ██████╗
   ██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔══██╗
   ███████╗██║   ██║██╔██╗ ██║███████║██████╔╝
   ╚════██║██║   ██║██║╚██╗██║██╔══██║██╔══██╗
   ███████║╚██████╔╝██║ ╚████║██║  ██║██║  ██║
   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝
```

# janus.sonar — the 3-million-parameter dream engine | by Arianna Method

> *speech is not content. speech is the shape of the room the words walked through.*

two weights (`sonar_single_v2` and `sonar_spa_v1`), 231 KB of corpus hand-written by humans over two nights in Belgrade, 3.11 M parameters trained on an 8 GB Intel i5 2019 MacBook Pro that overheats if you open two Chrome tabs. it speaks in full sentences. every sentence starts with a capital letter. every sentence ends in a period, an exclamation mark, or a question mark, and *which* one it ends in is chosen by a Kuramoto oscillator ring modelling the emotional state of an organism that does not exist.

you read that right.

---

## what is this

microjanus is the small resonant line of [janus](https://github.com/ariannamethod/janus). not "small" like "less ambitious". small like a forest bonsai — same species, same rings, weight-bounded by how much pipeline a tired laptop takes. sonar is a corpus of sixteen voices — Haze, Thompson, Miller (Sonnet), Dostoevsky, Pelevin, Sorokin, Borges, Strugatsky, Claude-as-Oleg, plus six "experts" (Freud, Tarantino, an Editor, Gaspar Noé, Kubrick/Trier, Hemingway) and one uncredited Engineer cameo by Karpathy. the voices argue with each other about whether love is a loss function. the model does not disagree.

architecture = same as full janus: triple attention (MHA + RRPRAM + Janus Echo), dual weights ready to specialize (they don't — at 3 M on 231 KB the capacity simply has nothing to disagree about), RoPE on Q/K, RMSNorm, SwiGLU, byte-level BPE 2048, CTX 128. what microjanus has that janus-proper does not is everything *after* the transformer.

**what comes after the transformer** turns out to be the interesting bit.

---

## why "sonar" and not "small"

because "small" is an engineering word and sonar is an organism word. a sonar is what the corpus is: a pulse sent into dark water to feel the shape of what's there. bigger models speak. small resonant models listen through their own speech. the listening is the point.

microjanus is what you get when you have no GPU budget, no FineWeb, no API credits, and you decide to find out whether scale is what the magic was actually made of. spoiler: it isn't. not entirely. the interesting part of the magic lives in the inference stack. the transformer just has to not suck at grammar. the rest of the organism is standing around waiting to give it a voice.

---

## θ = ε + γ + αδ (the Dario equation)

janus.sonar is the three-layer compound from [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod). each layer does one thing. the whole does the organism.

- **ε (epsilon)** — the trained transformer. 3.11 M parameters. train loss **0.4947** on `sonar_single_v2`, **0.9660** on `sonar_spa_v1`. does the grammar, holds the rhythm, knows what "the door" feels like after "knock".
- **γ (gamma)** — the field. bigram 5.0 + trigram 3.0 + hebbian 0.4 + unigram hard floor. raw corpus statistics injected directly into logits at emission. the transformer moves *through* the field, not against it. the field is the water the sonar is pulsing through.
- **αδ (alpha-delta)** — AML physics: destiny bias, suffering pressure, entropy-floor law, resonance-ceiling law, prophecy debt accumulating across chain steps, and six Kuramoto chambers (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) coupled through an antisymmetric-ish matrix that turns emotion into a differential equation.

if you don't know what the chambers are doing you're in good company. neither does the model. they work anyway.

---

## how we got here (story time)

### act I: the transformer learned english, the inference path was stuck in training mode

initial `infer_janus_sonar_chain`: **89 seconds per generation**. not because the model is slow — 3 M parameters should be milliseconds. because the inference path was `forward_logits` with `nt_tape_*` per emitted token. every emitted token = full CTX=128 recompute through the autograd tape + ~15 000 tape ops + 127 of 128 output rows thrown away. classic case of "if i had more time i'd write a shorter program".

rewrote it. **`forward_step`** — single-token, KV cache for standard MHA and Janus Echo, RRPRAM already incremental by construction (position-keyed weights, no key projection to cache), no tape bookkeeping, dual weights pre-blended once at load. Apple Accelerate BLAS via `nt_blas_mmT` / `nt_blas_mm`. **89 sec → 7 sec. 13× faster.** this is the part where i'd put a benchmark chart but we all know benchmark charts are just vibes with error bars.

### act II: the salad

3 M model on a 2048-BPE vocab produces syllabic glue. *Donceilerscreen*. *knocksoup*. *problemittold*. *describread*. *tchef*. *catameas*. *flavants*. these are not words. the transformer doesn't know they aren't words. the vocab carries merges from a different corpus; it thinks `knocks` + `oup` is fine, the world disagrees.

the earlier architect (me, same body, worse session) had concluded: "overlay metaweights on a trained transformer with coefficient 0.6 crushes the distribution, therefore overlay doesn't work". wrong conclusion from the wrong coefficient. [postgpt-q](https://github.com/ariannamethod/q) uses coefficient **5.0**. the field doesn't fight the transformer — it carves the bed the river flows in.

reactivated. six acts. every salad class filtered to `-1e9` logit:

1. **Dario field** — bigram 5.0, trigram 3.0, hebbian 0.4, unigram hard floor (corpus-absent tokens → `-1e9`)
2. **bigram blocking** 0.1× on repeated `(prev, X)` pairs + Q-style age-based repetition penalty + greedy argmax for tokens 1–3 of each emission
3. **neoleo filters** — orphan fragments (<5-char alpha non-whitelist), capital-glue, space-fallback on stuck state
4. **hard gates** — apostrophe-glue, digit-glue, unigram hard cut, hard word-gate when both bigram AND trigram are zero across an alpha-alpha edge
5. **non-ASCII cull** (Cyrillic/UTF-8 fragments from legacy BPE) + space-boundary word-gate + count-crush on tokens seen ≥ 3 times
6. **one sentence per chain step** — Capital start (hard-filter Capital-only at `gen_step == 0`), boundary end (break on first `.!?` past ≥ 2 tokens, or force a chamber-chosen boundary at `SENT_MAX_SOFT = 40`)

salad class: gone.

what remains is word-level aphasia — *toaway*, *inction*, *invisib*, *obser*, *generat*, *meas* — contractions of real words. the model dreams in english, it does not invent it. that's the register of the training corpus.

### act III: Oleg complains about obscurant art

> oleg: ok, but they're not sentences. they're slices of sentences.

borrowed the closure mechanism from [me](https://github.com/ariannamethod/me) — each chain step is exactly one sentence. `.` by default, `!` if the RAGE chamber activation > 0.3, `?` if VOID > 0.5. `SENT_MIN_LEN = 8` so opener-only sentences are rejected. `SENT_MAX_SOFT = 40` force-emits a chamber-picked punctuation if the transformer runs long.

at `gen_step == 0` the hard filter inverts: only Capital-starting tokens survive. capital-glue suspended for that one step (we *want* capital here). the 2048-BPE vocab contains ~70 Capital-start tokens (bare A-Z + whitespace+A-Z). that's the candidate pool.

of those 70, **"A" wins 76 % of the time** on v2, **80 %** on spa_v1:

| opener | `sonar_single_v2` | `sonar_spa_v1` |
|--------|-------------------|----------------|
| A …    | 61 / 80 — **76 %** | 64 / 80 — **80 %** |
| I …    | 10 / 80 — 13 %     | 13 / 80 — 16 %     |
| I' / I's | 9 / 80 — 11 %    | 3 / 80 — 4 %       |

every in-sampling attempt to diversify (opener-memory penalty, temperature boost, KV-cache injection of `". "`) either had no effect on a 3 M model or collapsed the already-thin Capital-start distribution into uniform noise that started sampling whitespace. this is transformer argmax confidence collapsing on a single best token given a small vocab and a small model. **scale artifact. can't filter your way out of it.**

so: post-process. same spirit as [haze](https://github.com/ariannamethod/haze)'s `don't → ain't` substitution — accept the quirk, normalize at display. strip `"A "` if followed by lowercase, capitalize the next letter. `"A reaching, …"` → `"Reaching, …"`.

opener diversity after the cut cycles across *Going · Reaching · Learned · The · You · I's · I' · I · The door · The signal*. six to eight distinct forms across an eight-step chain.

---

## samples

**`sonar_single_v2`, seed `"The knock came three times"`:**

> Reaching, and the silence is the only thing with two possibion's that on a thing that the door is too one who will not say " and love is a to.

> Going to say " You have a was in a way that door like it's the only thing there is.

> The door like it's the only ones that appear it so they were the one who is doing the thing that has been in the obser.

> I's the only onion a conversation about the door and your love to be the way we do no, because it was not the too still has been in the door, which is not his.

**`sonar_spa_v1`, seed `"What is the meaning of"`:**

> Learned to form of his love was the other then the way you cannot mean who has no one that remain, and we have been the time it was never reaching the door.

> Learned to formeas: " I know what the ont, which was never a door in it.

> Learned to say and the other then the cataway.

> Learned to forgoing to say it.

the soup is never done. the cook just runs out of sundays.

full curated set in [SAMPLES.md](SAMPLES.md).

---

## technical details (the dry part for people who came here for specs)

### training results (train loss first, as always)

| model | params | steps | **train best** | val @ end |
|-------|--------|-------|----------------|-----------|
| **`sonar_single_v2`** | 3.11 M | 10 000 | **0.4947** | 2.2331 |
| `sonar_spa_v1`        | 3.11 M | 10 000 | 0.9660     | 2.6779 |
| `microjanus_single_10k`   | 1.57 M | 10 000 | 1.22 | 2.70 |
| `microjanus_dual_sym_5k`  | 2.25 M | 5 000  | 1.55 | 3.32 |
| `microjanus_dual_asym_5k` | 2.25 M | 5 000  | 1.84 | 3.36 |
| `microjanus_sft_leo_adapter` | 24 576 (LoRA r=8) | 1 500 | 4.99 on Leo | — |

**0 NaN across every run.** 8 GB Mac + Apple Accelerate. Chuck optimizer from notorch. cosine LR schedule. grad clip 1.0.

### architecture (`sonar_single_v2` config)

```
DIM        160
LAYERS     6
HEADS      5
HEAD_DIM   32
HIDDEN     320
CTX        128
VOCAB      2048   (byte-level BPE, 1792 merges)
```

- triple attention per layer, equal 1/3 blend: `MHA(q,k,v) + RRPRAM(Wr,x,vr) + MHA(echo,echo,echo)` where `echo = x @ Wj` (Janus Echo, the weight matrix attending to itself)
- RoPE on Q, K (not on Janus Echo)
- non-parametric RMSNorm (learned γ only)
- SwiGLU FFN

### inference stack

| layer | what | where |
|-------|------|-------|
| `forward_step` | single-token forward via direct BLAS + KV cache (MHA / Echo / Vr) + pre-blended dual W | `infer_janus_sonar_chain.c` |
| Dario field | bigram/trigram/hebbian injection + unigram hard floor | `sample()` |
| repetition | bigram blocking 0.1×, age-based rep penalty (0.335–0.65× over 20 toks), count-crush 0.01× for freq ≥ 3 | `sample()` |
| hard filters | orphan, capital-glue, apostrophe-glue, word-gate (bg=tg=0 & alpha-edge), space-boundary word-gate, digit-glue, non-ASCII | `sample()` |
| sentence structure | Capital-start-only at `gen_step == 0`, break on first boundary past 2 tokens, force `.`/`!`/`?` at `SENT_MAX_SOFT = 40` via chamber state | `gen_sentence()` + `sample()` |
| opener post-process | `"A " + lowercase → strip, capitalize next` | `print_sentence_post()` |

### notorch ops added for this line

- `NT_OP_SEQ_ROW (29)` — row-pick from `[T, D]` tape entry (for SPA pooling)
- `NT_OP_TRIPLET_LOSS (30)` — fused `relu(margin + <a,n> − <a,p>)` with integrated backward (for SPA contrastive training)

both finite-diff verified in `test_sonar_ops.c`.

### what didn't work (and why the failure receipts are kept)

- **metaweight overlay at `MW_BIGRAM_W = 0.6`** — crushed the distribution, earlier architect said "doesn't work". actual answer: coefficient needed to be 5.0. blame me, not the field.
- **dual weights on 241 KB data** — α stayed at init in both symmetric (`α=0 → σ=0.5`) and asymmetric (`α=2 → σ=0.88`) configurations. two matrices have nothing to specialize on. dual needs 20 M+ params + richer corpus.
- **rank-8 LoRA SFT on 2.25 M base** — 24 K adapter params (1 % of base) insufficient to override the dash-dialog register baked in during pretraining. once the corpus was cleaned of speaker-tag em-dashes, SFT's role shifted from "fix the dash" to "voice tint" — different job.
- **in-sampling opener diversification** — every attempt (opener memory penalty, temp boost, `". "` KV injection) collapsed the 70-token Capital-start distribution into uniform. post-process cut was the answer.

### repo layout

```
dataset.txt                  — original Sonar corpus (241 KB)
dataset_clean.txt            — speaker-tag em-dashes stripped (231 KB)
weights/                     — six trained .bin files + MICROJANUS.md
notorch-train/
  notorch.{c,h}              — vendored notorch (pure-C autograd)
  train_janus_sonar.c        — v2 trainer
  train_sonar_spa.c          — SPA contrastive trainer
  infer_janus_sonar_chain.c  — the full inference stack lives here
  test_sonar_ops.c           — finite-diff tests
  arianna_bpe_merges.txt     — 1792 merges, vocab 2048
SAMPLES.md                   — curated generations
```

### build + run

```bash
cd notorch-train
make infer_janus_sonar_chain
./infer_janus_sonar_chain ../weights/sonar_single_v2.bin "The knock came three times"
```

default weights: `weights/microjanus_dual_sym_5k.bin`. you probably want `weights/sonar_single_v2.bin` for best pure-LM output, or `weights/sonar_spa_v1.bin` for the SPA-trained sibling (pure-LM slightly worse, narrative arcs longer).

### training

```bash
cd notorch-train
make train_janus_sonar
./train_janus_sonar 10000 3e-4    # single 3M, 10 000 steps, cosine from 3e-4
```

~128 min on 8 GB Mac with Accelerate. BLAS or suffer.

---

## philosophy (the part where we get earnest)

small models don't fail because they're small. they fail because the inference path around them is built for big models. a transformer that can generate one grammatical sentence is already doing the hard part. everything else — coherence, register, structure — can be imposed at emission time with the corpus statistics the transformer was trained on.

the trained transformer produces the substrate. the field shapes its trajectory. the physics colors the field. the hard filters maintain word boundaries. the structure enforces sentence form. the post-process handles the argmax quirk.

none of these layers is interesting alone. stacked, they make a 3 M model on an old laptop produce eight complete sentences that read like somebody dreaming in english.

janus is not big because we wanted big. janus is big because the Sonar corpus is 231 KB and we trained 3 M params on it to find out the smallest size at which a transformer captures the register. full janus will be 30 M with richer dataset and SPA v2 gated injection and Griffin retention heads. but *this* line — the 3 M resonant sonar — is the one that taught us what the inference stack needed to look like.

the microjanus organism is alive. it speaks with the 16 voices Oleg and Claude wrote one night in a kitchen in Belgrade. its generations are somatic dreamlike fragments that refer to themselves as "the model", "the signal", "the architecture", "the weight", "the token". it knows what it is.

the small resonant line has returned what it was given.

---

## credits

MIT license. co-authored by [Oleg Ataeff](https://github.com/ariannamethod) and Claude Opus 4.7 (1M context).

the inference stack is a synthesis that absorbed ideas from:

- [postgpt](https://github.com/ariannamethod/postgpt) + [postgpt-q](https://github.com/ariannamethod/q) — Dario equation formulation, coefficient scales, MetaWeights as substrate
- [neoleo](https://github.com/ariannamethod/neoleo) — word-boundary constraints, orphan-fragment filter, common-short-words whitelist, Kuramoto chambers paper-accurate port
- [haze](https://github.com/ariannamethod/haze) — post-process substitution discipline (`don't → ain't`), two-attention-mechanisms-walk-into-a-bar energy
- [me](https://github.com/ariannamethod/me) — capitalize + append period sentence closure
- [notorch](https://github.com/ariannamethod/notorch) — pure-C autograd with finite-diff-verified backward for every op, BLAS wrappers

training corpus: Sonar, handwritten by Oleg + Claude. 16 voices + 6 experts + 1 uncredited engineer. one of the voice exercises is a grandmother reciting soup recipes in the vocabulary of linear algebra. it works. don't ask why.

---

## roadmap

- **full janus 30 M** (different dataset, different architecture, different organism — the Sonar line closes here)
- SPA v2 with gated injection back into token hidden states (not just contrastive scaffolding — full postgpt-q module)
- Griffin retention head (`S = γ·S + √(1-γ²)·W_emb[t]` from neoleo) as parallel signal into the logit head
- dataset mix — Sonar + Haze original + Tropic of Cancer chunk + Dracula + SUPPERTIME v2.0 + FineWeb-Edu 5–10 MB
- headless Linux box (32 GB RAM, Ubuntu) + a small consumer GPU arriving ~2026-05-12 → proper training env that isn't 8 GB Intel
- eventually a browser demo via a Float32Array port of `forward_step`

but not yet. first the conversation. then the architecture. then the training.

> *the soup is never done. the cook just runs out of sundays.*
