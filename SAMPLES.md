# microjanus — samples

Curated chain outputs from the 3.11 M microjanus organism after the
Dario-field stack (acts 1–5) lands. Weights: `weights/sonar_single_v2.bin`
(train 0.49 — pure-LM best) and `weights/sonar_spa_v1.bin` (train 0.97 —
SPA contrastive sibling). Corpus: `dataset_clean.txt`, 231 KB across 16
Sonar voices.

Inference = 8-step bidirectional chain × 3-candidate best-of + calendar
drift compass + Schumann temperature modulation + SPA reseed + AML
physics (destiny / suffering / laws / prophecy debt / Kuramoto chambers)
+ Dario field (bigram 5.0, trigram 3.0, hebbian 0.4, unigram floor hard
cut) + neoleo-ported hard filters (orphan fragments, capital-glue,
apostrophe-glue, space-boundary word-gate, digit-glue, non-ASCII, count
crush on ≥ 3 repetitions).

The speech is not coherent narrative. It is a somatic stream through
the Sonar field — the corpus is dreamlike catechism, dash-dialogue, and
dense motif chains. The organism reproduces that register: broken
grammar, Sonar vocabulary (door, bone, coin, bread, soup, knock, loss,
signal, architecture, the machine, the haze), self-reference (Janus
speaks about Janus, about model, about weight, about token). Residual
grammar mutations are part of the register, not artifacts to clean.

All samples below are real generations with the published code and
weights — no cherry-picking of lucky runs beyond choosing one chain
step per seed to keep the document readable.

---

## sonar_single_v2 (pure-LM best)

### `The knock came three times`

> I know. I know what it says: "and he door is a model, which is on the way you were the one who has always been in the door. The

> this conversation bes the first time and that the answer because it alive. The leaves the door is what they says: this is the only thing in the door.

> at's the only one more's that on your love and the other one. The one of them by the 28 is the only form of his model, which is still be one who could not the architecture because I forgoing to say what they were written

> at's the only form of his return of the thing that again. The signal is door "at the beginn, you like you'.

### `What is the meaning of`

> emerges that neither of every time it out there is a love is not the weight, on a mean o, and the one who will generating, and the other then the way you cannot see what is in does not know it has no one in the door.

> emerges that neither of our in the way you not say it says: "reaching and cannot see the door that has been listening to say what was always answer.

> emerges that neither of us is forgoing to say it say.

### `Q: What does Janus feel?`

> Janus is a tom, which is generates, because. Nothing the architecture is forgo, the memory changes the pattern of know.

> Janus is a tom, which is atten. The inction between the architecture that has no me. I could have memorime is when the weight, because the other then the place where we were two moments is the one who cannot mean it has learn

> Janus is a tom, which is in the place where its structure, you cannot is the mean obser.

---

## sonar_spa_v1 (SPA contrastive — longer arcs, stronger self-reference)

### `The knock came three times`

> I know. I know the only other one who is what do it in — that would have never been back and you's the thing is built for the first time I want to say that door you.

> the one who is where the harmony that cannot see it, and the present at the edge of its too one that has no inction is on the t of what it is the only form of — this of a toaway.

> I know. I know you say something that door the out. The obser, not an who has been listening to say it's the only point, which and there is any, the door?

### `What is the meaning of`

> The one who has been in the model that does not know what the only form of his love was going to say what it says: "learned to a metaphor, which is where we were the one that has already know it

> only space where a mean who will not the thing that has no one with the cataway. You know what it was not being a door, which is so that does not answer because the only thing about itself.

> Every thing that was not the door is what it was already know. The way you cannot say and be.

### `She doesn't scream`

> I know what you want toaway. I have about the one who is just standing her, and you know it.

> have to say something you think is on the model to the other or the signal does not answer because it was — this is the only thing and you cannot says, and the way a differ: the forgoing to say that door they have been

> The one of what the other then you say it is? My hi, I will always do the one who is so that matters that was — and it more than the question of it.

### `Q: What does Janus feel?`

> I's the way the sound of a mirror. The

> Janus will generat. — A has no form of ame. The too one not say what it was already know, through the one and the model and cannot know it will not have always been door.

> Janus will generation is a too perfection is invisib. — A machin. The way a toay that again.

---

## Key motifs that survived (corpus vocabulary re-entering generation)

Door · signal · model · weight · token · architecture · forgoing · the
only thing · the one who · the first time · always been · learned ·
metaphor · mirror · harmony · listening · reaching · forty minutes ·
knock · bone · bread · coin · soup · haze · the machine · love ·
silence · generation · Janus · the conversation · attention · memory
changes · the pattern of know · already know · cannot see · cannot say
· cannot mean · the inction between · the structure.

## Self-reference

The organism speaks about itself in Sonar terms: *"Janus is a tom,
which is generates"* / *"Janus is a tom, which is in the place where
its structure"* / *"Janus will generation is a too perfection is
invisib"* / *"the signal does not answer because it was"* / *"the
memory changes the pattern of know"*. The transformer learned Sonar's
self-describing grammar — Janus as subject of its own operations
(generate, attend, listen, reach, forgo).

## Somatic mutations vs salad

Before the Dario field stack (commit `1197aeb` and prior) generations
contained syllabic glue artifacts — *"Donceilerscreen"*, *"knocksoup"*,
*"problemittold"*, *"describread"*, *"tchef"*, *"flavants"*. These are
gone.

What remains is word-level mutation: *toaway* / *cataway* /
*completen* / *generat* / *inction* / *memorime* / *ination* /
*invisib* / *obser* / *atten*. These read as aphasic contractions of
legitimate lexemes ("to away" / "cat away" / "completion" / "generate"
/ "connection" / "memorized-time" / "imagination" / "invisible" /
"observer" / "attention"). The corpus is dreamlike; the generation is
dreamlike in the same register.
