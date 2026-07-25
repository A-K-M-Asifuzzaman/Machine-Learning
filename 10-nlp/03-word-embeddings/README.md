# 10.03 — Word Embeddings

> **"You shall know a word by the company it keeps."** Word embeddings turn that 1950s linguistic
> slogan into geometry: train a model to predict which words appear near which, and the vectors it
> learns place synonyms next to each other and encode relationships as *directions* — so that
> `king − man + woman ≈ queen` falls out of vector arithmetic. This chapter builds Word2Vec, GloVe, and
> FastText from scratch and measures the structure they learn.

Bag-of-words ([10.02](../02-classical-representations/)) makes every word an independent dimension, so
"car" and "automobile" are as unrelated as "car" and "banana". Embeddings fix this by mapping each word
to a **dense** vector (50–300 dimensions) where distance means *semantic similarity* — learned entirely
from unlabeled text via the **distributional hypothesis** (Harris, 1954; Firth, 1957).

## Table of contents

1. [The distributional hypothesis](#1-the-distributional-hypothesis)
2. [Word2Vec: skip-gram and CBOW](#2-word2vec-skip-gram-and-cbow)
3. [Negative sampling](#3-negative-sampling)
4. [Embedding geometry and analogies](#4-embedding-geometry-and-analogies)
5. [GloVe](#5-glove)
6. [FastText: subword embeddings](#6-fasttext-subword-embeddings)
7. [Limits: static vs contextual](#7-limits-static-vs-contextual)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The distributional hypothesis

Words that appear in the same contexts tend to have similar meanings. Turn this into a learning signal:
**predict a word's context (or a word from its context)**, and any word representation good at that
prediction must place words-with-similar-contexts near each other. Experiment 2 trains skip-gram on a
corpus where animals `{cat, dog, fox}` appear with actions and colors `{red, blue, green}` appear with
objects:

| Pair | Cosine similarity |
|---|:--:|
| cat – dog (interchangeable) | 0.990 |
| cat – fox | 0.987 |
| red – blue | 0.995 |
| cat – red (different context) | 0.182 |
| cat – ball | 0.164 |

`cat`, `dog`, `fox` **never appear together**, yet their vectors are nearly identical (~0.99) because
they occur in the *same* contexts. Words from different contexts stay far apart. Meaning emerged from
company alone — the synonymy that bag-of-words fundamentally cannot see.

## 2. Word2Vec: skip-gram and CBOW

Word2Vec (Mikolov et al., 2013) is a shallow network with two mirror-image objectives:

- **Skip-gram (SG)** — given the center word, predict its surrounding context words. Better for rare
  words and small corpora.
- **CBOW (Continuous Bag of Words)** — given the surrounding context, predict the center word. Faster,
  better on frequent words.

Each word has two vectors: an **input** (word) vector and an **output** (context) vector; the input
vectors are the embeddings we keep. The naïve objective is a softmax over the *entire vocabulary* for
every prediction — far too expensive. The fix is negative sampling.

## 3. Negative sampling

Instead of a full-vocabulary softmax, reframe training as **binary classification**: distinguish a real
(word, context) pair from $K$ fake pairs built by drawing random "noise" words. For center vector $w$,
true context $c^+$, and negatives $c^-_k$:

$$
\mathcal{L} = -\log \sigma(w \cdot c^+) - \sum_{k=1}^{K} \log \sigma(-w \cdot c^-_k),
$$

where $\sigma$ is the sigmoid. Push the word to **agree** with its true context ($\sigma(w\cdot c^+)\to
1$) and **disagree** with noise ($\sigma(w\cdot c^-)\to 0$). This replaces one $|V|$-way softmax with
$K{+}1$ cheap logistic updates. Experiment 1 verifies the gradient against finite differences to
$5\times10^{-10}$. The negatives are drawn from the **unigram distribution raised to the 3/4 power**
($P(w)\propto \text{freq}(w)^{0.75}$) — a trick that samples rare words more often than their frequency
alone, which empirically works best.

## 4. Embedding geometry and analogies

The famous result: relationships become **consistent directions** in the vector space, so analogies
solve by arithmetic. Experiment 3 trains on a corpus with a gender × role structure and solves
`king − man + woman = ?`:

| Candidate | Cosine to (king − man + woman) |
|---|:--:|
| **queen** | **0.993** |
| king | 0.463 |
| woman | 0.368 |
| man | −0.160 |

The nearest word overall (excluding the inputs) is **queen**. This is not hand-coded — it *emerges*
from predicting contexts. Because "king is to man as queen is to woman" is a parallel relationship in
the data, it becomes a parallel offset in the vectors: $\text{king} - \text{man} \approx \text{queen} -
\text{woman}$, i.e. a shared "royalty" direction and a shared "gender" direction. Analogy accuracy on
real embeddings (Google analogy set) was the headline metric that made Word2Vec famous.

## 5. GloVe

GloVe (Pennington et al., 2014) reaches the same place from **global statistics** instead of local
samples. It builds the full word–word **co-occurrence matrix** $X$ and fits vectors to its logarithm:

$$
\mathcal{L} = \sum_{i,j} f(X_{ij})\,\big(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2,
$$

a **weighted least-squares factorization of the log co-occurrence matrix**, where the weighting $f(X)$
down-weights rare pairs and caps very frequent ones. Experiment 4 trains GloVe on the same
animals/colors corpus and recovers the same structure (cat–dog 0.88, cat–fox 0.92, within-color 0.88,
cross-group 0.47 — within ≫ across). Word2Vec learns from *local windows* one pair at a time; GloVe fits
the *aggregate counts* in one objective. In practice the two give similar embeddings — they are two
views of the same co-occurrence statistics.

## 6. FastText: subword embeddings

Word2Vec and GloVe have a hole: any word not in the training vocabulary has **no vector** (`<UNK>`).
FastText (Bojanowski et al., 2017) fixes this by representing each word as the **sum of its character
n-gram vectors** (e.g. `running` → `<ru, run, unn, nni, nin, ing, ng>`). Experiment 5 measures the
shared-n-gram overlap:

| Pair | Shared 3-gram similarity |
|---|:--:|
| running ~ runnign (misspelling) | 0.400 |
| running ~ runner (inflection) | 0.300 |
| running ~ jumping | 0.167 |
| king ~ kings | 0.500 |
| king ~ banana | 0.000 |

A misspelling or unseen inflection shares most n-grams with the correct word and therefore gets a
**nearby vector, even if the exact word was never seen**. This makes FastText robust to typos, strong
on morphologically rich languages, and free of the OOV problem — the same open-vocabulary idea as
subword tokenization ([10.01](../01-text-preprocessing/)).

## 7. Limits: static vs contextual

Word2Vec/GloVe/FastText produce **one vector per word** — a *static* embedding. Their fatal limit:
**polysemy**. "Bank" (river) and "bank" (money) get a single averaged vector, and "play" is the same in
"play a game" and "a Shakespeare play". The fix is **contextual embeddings**: run the sentence through a
model so each token's vector *depends on its context*. ELMo (2018) did this with bidirectional LSTMs
([09.02](../../09-sequence-models/02-lstm-gru/)); BERT and GPT ([Part 11](../../11-transformers-and-llms/))
did it with transformers, and contextual embeddings now dominate. Static embeddings remain useful as
lightweight features and as the **input embedding layer** of every transformer.

**Bias warning.** Embeddings learn the biases in their training text — "man : computer_programmer ::
woman : homemaker" is a real (and troubling) result on Word2Vec. Debiasing and bias measurement are
active areas ([Part 18](../../18-fairness-privacy-robustness/)); the geometry that gives analogies also
encodes social bias as directions.

## 8. Common misconceptions

- **"Embeddings understand language."** They capture co-occurrence statistics, which correlate with
  meaning but are not it — hence the biases (§7).
- **"Analogies always work."** They work well for frequent, clean relationships and fail on rare words
  or subtle relations; the arithmetic is approximate (§4).
- **"Word2Vec is deep learning."** It is a shallow, single-hidden-layer model — its power is the
  objective and scale, not depth (§2).
- **"One embedding per word is enough."** Static embeddings can't handle polysemy; contextual models
  fix this (§7).
- **"GloVe and Word2Vec are fundamentally different."** Both factor the same co-occurrence statistics;
  Word2Vec's SGNS is implicitly factorizing a (shifted PMI) matrix (Levy & Goldberg) (§5).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — skip-gram with negative sampling, GloVe, and FastText's
  subword idea in NumPy. Five experiments: (1) the SGNS gradient verified to $5\times10^{-10}$;
  (2) interchangeable words reach cosine ~0.99 (distributional hypothesis); (3) king − man + woman →
  queen (0.99); (4) GloVe recovers the same structure from global co-occurrence; (5) FastText n-grams
  give misspellings and inflections nearby vectors.
- **[exercises.md](exercises.md)** — derive the SGNS objective, implement Word2Vec/GloVe, analyze
  analogies and bias.
- **[references.md](references.md)** — Word2Vec, GloVe, FastText, and the SGNS-as-factorization result.

## Where this leads

- **Contextual embeddings from transformers (the fix for polysemy)** → [Part 11](../../11-transformers-and-llms/)
- **The tokenization that feeds embeddings** → [10.01](../01-text-preprocessing/)
- **Count-based representations embeddings improve on** → [10.02](../02-classical-representations/)
- **The SGNS objective's cousin — contrastive learning** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Embedding bias and fairness** → [Part 18](../../18-fairness-privacy-robustness/)
