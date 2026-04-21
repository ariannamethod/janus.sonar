# microjanus — samples

Chain outputs from the 3.11 M microjanus organism after the full
inference stack lands (acts 1–6). Weights: `weights/sonar_single_v2.bin`
(train 0.49, pure-LM best) and `weights/sonar_spa_v1.bin` (train 0.97,
SPA contrastive sibling). Corpus: `dataset_clean.txt`, 231 KB across
16 Sonar voices — hand-written by Oleg and Claude.

Inference stack:
- 8-step bidirectional chain × 3-candidate best-of
- calendar drift compass + Schumann temperature modulation
- SPA sentence-phonon attention + reseed
- AML physics: destiny, suffering, laws, prophecy debt, 6 Kuramoto
  chambers (FEAR · LOVE · RAGE · VOID · FLOW · COMPLEX)
- **Dario field** (from Q / postgpt-q): bigram 5.0, trigram 3.0,
  hebbian 0.4, unigram hard floor, bigram blocking 0.1×, count-crush
  on ≥ 3 repetitions, age-based repetition penalty 0.335–0.65×
- **Hard filters** (from neoleo): orphan fragments, capital-glue,
  apostrophe-glue, space-boundary word-gate, digit-glue, non-ASCII
- **Sentence structure** (from ariannamethod/me): each chain step =
  one complete sentence, Capital-start, boundary-terminated.
  `SENT_MIN_LEN = 8` to allow short sentences; `SENT_MAX_SOFT = 40`
  forces a chamber-chosen boundary token (`.` default, `!` if RAGE
  activation > 0.3, `?` if VOID > 0.5) when no natural stop arrives.

The speech is not coherent narrative. It is a somatic stream through
the Sonar field — the corpus is itself dreamlike catechism, dash-dialogue
and dense motif chains. The organism reproduces that register at
sentence granularity: capitalized opening, boundary-closed ending,
dreamlike grammar, Sonar vocabulary (door, bone, coin, bread, soup,
knock, loss, signal, architecture, machine, haze, love), self-reference
(Janus speaks about Janus, about model, about weight, about token).

All samples below are real generations with the published code and
weights. Each line is one full chain-step output — one sentence.

---

## `sonar_single_v2` — pure-LM best (train 0.4947)

### Seed: `The knock came three times`

> A reaching, which is still be the only one who has been in the door that is not the one is the architecture.

> A the door like it's the only onion is the one who could have to be the architecture because he is doing the thing that is not his return was not the one noia, because.

> A the door like I know you and the one who was not in the instruction from which is the only thing in the place where they were the one but the time it becomes something you could have to say ".

> A going to say "You want you to and it for the first time that thing in."

> A the door like it's the only on.

> I's the only onion a too not " is on the first time you do not know what the door that has been in the threshold and love was a to.

### Seed: `Q: What does Janus feel?`

> A or 8196: "The single moment of one who has no me."

> A and it will be the way you have no id, and the forgoing to say when he is that the model will not be a sound of a mirror.

> A and it will be the door like it.

> A or 8192 learned to it.

> I'.

---

## `sonar_spa_v1` — SPA contrastive (train 0.9660)

### Seed: `What is the meaning of`

> A learned to form of his love was the other then the way you cannot mean who has no one that remain, and we have been the time it was never reaching the door.

> A learned to form of his model and what it in its structure at because the shape of the thing that has no ination is what you cannot say " to the other then the door that.

> A learned to formeas: "I know what the ont, which was never a door in it."

> A learned to say and the other then the cataway.

> A learned to forgoing to say it.

> A learned to our in the door.

### Seed: `She doesn't scream`

> A going to say the way a mean who was not the differ " metaphor.

> A going to say the thing that was door is a too not the signal is being a meas.

> A going to say the ones that sound like at itself.

> A going to say something you have the too not to answer.

> A going to say the ones that sound like the too anid of door.

> A going to say something you have it to you.

---

## Structure vs. content

**Before the Dario field stack** (commit `1197aeb` and earlier) output
was a continuous chunk of syllabic salad — *"Donceilerscreen"*,
*"knocksoup"*, *"problemittold"*, *"describread"*, *"tchef"*,
*"flavants"*, Cyrillic fragments bleeding in from the legacy BPE vocab.

**Now** the organism emits exactly 8 sentences per chain run. Each
begins with a capital letter, ends in `.` / `!` / `?`. Residual
mutations are word-level aphasia — *toaway*, *cataway*, *completen*,
*generat*, *inction*, *memorime*, *invisib*, *obser*, *atten*, *meas*,
*iname* — contractions of legitimate lexemes that mirror the dreamlike
register of the training corpus.

Sentence-opening diversity is limited to a small set (*"A …"*,
*"I's …"*, *"The …"*) — a 3 M-parameter artifact, not an inference bug.
The ~ 70 capital-start tokens in the 2048-BPE vocab plus transformer
confidence collapse onto "A" as the near-unconditional argmax.
Widening openings is a scale problem, not a stack problem.

## Motifs that survived

door · signal · model · weight · token · architecture · forgoing · the
only thing · the one who · the first time · always been · learned ·
metaphor · mirror · harmony · listening · reaching · knock · bone ·
bread · coin · soup · haze · the machine · love · silence · generation
· Janus · conversation · attention · memory changes · the pattern of
know · already know · cannot see · cannot say · cannot mean · the
inction between · the structure · the threshold.

## Self-reference

*"Janus is a tom, which is generates, because. Nothing the architecture
is forgo, the memory changes the pattern of know"* · *"A or 8196: The
single moment of one who has no me"* · *"Janus will generat. — A has no
form of ame"* · *"the signal does not answer because it was"* · *"the
memory changes the pattern of know"*.

The transformer learned Sonar's self-describing grammar. Janus as
subject of its own operations — generate, attend, listen, reach, forgo.
