# 10.02 — Classical Text Representations

> **Before embeddings, text became numbers by counting.** A document is a bag of words; a corpus is a
> giant sparse count matrix; and everything — search, classification, topic discovery — is linear
> algebra on that matrix. These methods are simple, fast, interpretable, and still the right first
> tool for many tasks. They also expose the two problems that embeddings ([10.03](../03-word-embeddings/))
> were invented to fix: they ignore word *order* and word *meaning*. This chapter builds the classical
> pipeline and verifies each piece against scikit-learn.

The **vector space model** (Salton, 1970s) represents each document as a vector in
$\mathbb{R}^{|V|}$, one dimension per vocabulary word. Once documents are vectors, similarity is
geometry (cosine), classification is a linear boundary, and topics are matrix factorizations.

## Table of contents

1. [Bag-of-words](#1-bag-of-words)
2. [TF-IDF](#2-tf-idf)
3. [N-grams: recovering a little order](#3-n-grams-recovering-a-little-order)
4. [Similarity and retrieval](#4-similarity-and-retrieval)
5. [Latent Semantic Analysis](#5-latent-semantic-analysis)
6. [Topic models: LSA, NMF, LDA](#6-topic-models-lsa-nmf-lda)
7. [Strengths, limits, and the bridge to embeddings](#7-strengths-limits-and-the-bridge-to-embeddings)
8. [Common misconceptions](#8-common-misconceptions)

## 1. Bag-of-words

The simplest representation: count how many times each vocabulary word appears in each document,
ignoring order. A corpus becomes a **document–term matrix** $X \in \mathbb{R}^{n \times |V|}$ where
$X_{ij}$ is the count of word $j$ in document $i$. It is a **multiset** — "the cat sat" and "sat the
cat" are identical. [`from_scratch.py`](from_scratch.py) builds this and matches
`sklearn.CountVectorizer` exactly (Experiment 1). The matrix is huge and **sparse** (most words absent
from most documents), which is why it is stored and computed sparsely.

## 2. TF-IDF

Raw counts overweight common words: "the" appears in everything and dominates. **TF-IDF** fixes this by
multiplying each count (term frequency, **TF**) by an **inverse document frequency** (**IDF**) that
downweights words appearing in many documents:

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t), \qquad \text{idf}(t) = \ln\frac{1+n}{1+\text{df}(t)} + 1,
$$

then each document vector is L2-normalized. (The $+1$s are scikit-learn's smoothing; the $+1$ outside
keeps words that appear everywhere from vanishing entirely.) The from-scratch version matches
`sklearn.TfidfVectorizer` to $6\times10^{-17}$ (Experiment 2). Experiment 2b shows the effect —
IDF weights across 4 documents:

| Word | Doc frequency | IDF |
|---|:--:|:--:|
| the, today | 4 (all) | **1.000** (minimum) |
| market | 2 | 1.511 |
| rose, fell, weather, … | 1 | **1.916** (maximum) |

Words in every document get the smallest weight; distinctive words get the largest. This one reweighting
makes TF-IDF a strong baseline for search and text classification — often within a point or two of
neural models on topic-classification tasks, at a fraction of the cost.

## 3. N-grams: recovering a little order

Bag-of-words throws away order entirely, which can invert meaning. Experiment 3:

> A: "the movie was good not bad"  B: "the movie was bad not good"

| Representation | A and B identical? |
|---|:--:|
| unigrams (bag-of-words) | **True** — order lost |
| bigrams | **False** — order captured |

The two sentences have opposite sentiment but the *same words*, so their unigram vectors are identical.
Adding **bigrams** (adjacent word pairs) brings back local order: "not bad" and "not good" become
distinct features. N-grams are the classical fix for word order — cheap and effective for short-range
patterns — but they blow up the vocabulary (many pairs/triples) and become very sparse, so they are
usually capped at bi- or tri-grams.

## 4. Similarity and retrieval

With documents as TF-IDF vectors, **cosine similarity** ranks relevance:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [0, 1] \text{ for non-negative vectors}.
$$

Cosine ignores document length (a long and short document on the same topic still match), which is why
it is preferred over Euclidean distance for text. This is the core of classical **information
retrieval**: represent the query as a vector, rank documents by cosine, return the top few (the basis
of TF-IDF search, BM25's ancestor).

## 5. Latent Semantic Analysis

Bag-of-words has a fatal blind spot: it cannot see **synonymy**. Two documents about space that share
no words are orthogonal — similarity 0 — even though they mean the same thing. **LSA** (Deerwester et
al., 1990) fixes this with a **truncated SVD** of the document–term matrix:

$$
X \approx U_k \Sigma_k V_k^\top,
$$

keeping the top $k$ singular directions as latent **topics**. Documents projected into this
$k$-dimensional space are compared there. Experiment 4 (8 docs, 4 space + 4 food):

| Pair | Raw TF-IDF cosine | LSA cosine |
|---|:--:|:--:|
| doc1–doc3 (both space, **no shared words**) | 0.000 | **1.000** |
| doc1–doc4 (space vs food) | — | 0.000 |

Two space documents with *no words in common* become identical (1.00) in LSA space, while a space–food
pair stays at 0. LSA linked them through their **co-occurrence patterns** — both space docs co-occur
(across the corpus) with the same bridging words. This is LSA's power: it recovers latent semantic
structure that surface words hide, at the cost of losing interpretability (SVD directions can be
negative and hard to read).

## 6. Topic models: LSA, NMF, LDA

Three ways to factor a corpus into topics:

- **LSA** (§5) — SVD; fast, but topic vectors have negative entries and are hard to interpret.
- **NMF** (non-negative matrix factorization) — factor $X \approx W H$ with **all entries $\ge 0$**.
  Non-negativity forces **parts-based, additive** topics: each topic is a set of co-occurring words,
  each document a non-negative mixture. Experiment 5 factors 6 documents (3 finance, 3 sports) into 2
  topics and cleanly recovers them:

  | Topic | Top words |
  |---|---|
  | 0 | money, bank, market, loan |
  | 1 | win, team, play, score |

  with each document assigned to its correct topic. Interpretable and unsupervised.
- **LDA** (Latent Dirichlet Allocation, Blei et al., 2003) — a *probabilistic* generative model: each
  document is a distribution over topics, each topic a distribution over words, with Dirichlet priors.
  Gives proper probabilities and is the classic topic model, inferred by variational EM or Gibbs
  sampling.

## 7. Strengths, limits, and the bridge to embeddings

**Strengths.** Simple, fast, interpretable, no training data or GPUs; a strong baseline for topic
classification and retrieval; still used in production search (TF-IDF/BM25).

**Two fundamental limits** motivate everything after:

1. **No word order** — bag-of-words is a multiset; n-grams patch it locally but explode combinatorially
   (§3). Sequence models ([Part 9](../../09-sequence-models/)) and transformers model order natively.
2. **No word meaning** — each word is an independent dimension, so "car" and "automobile" are as
   unrelated as "car" and "banana". LSA/NMF recover *some* latent structure from co-occurrence, but the
   real fix is dense **word embeddings** ([10.03](../03-word-embeddings/)) that place synonyms near each
   other by design — the "distributional hypothesis" made into a vector.

## 8. Common misconceptions

- **"TF-IDF understands meaning."** It only reweights counts; it has no notion that "car" ≈
  "automobile" (§2, §7).
- **"Bag-of-words is obsolete."** TF-IDF + linear model is a fast, strong, interpretable baseline that
  often rivals neural models on topic tasks — always try it first.
- **"LSA and PCA are different."** LSA *is* (truncated) SVD of the term–document matrix — the same
  machinery as PCA ([04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)).
- **"More n-grams always help."** They help order but explode the (already sparse) feature space; past
  tri-grams they usually overfit.
- **"Topic models find the 'true' topics."** They find statistical co-occurrence structure; the number
  of topics is a hyperparameter and the topics are an interpretation, not ground truth.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — bag-of-words, TF-IDF, n-grams, LSA, and NMF in NumPy,
  verified against scikit-learn. Five experiments: (1–2) BoW and TF-IDF match `CountVectorizer` /
  `TfidfVectorizer` exactly; (2b) IDF downweights ubiquitous words; (3) bigrams distinguish
  opposite-sentiment sentences unigrams cannot; (4) LSA gives two no-shared-word space docs similarity
  1.0; (5) NMF recovers a finance and a sports topic.
- **[exercises.md](exercises.md)** — derive TF-IDF, implement LSA/NMF, reason about the order and
  meaning limits.
- **[references.md](references.md)** — the vector-space, LSA, NMF, and LDA sources.

## Where this leads

- **Dense word embeddings that fix the meaning problem** → [10.03](../03-word-embeddings/)
- **Tokenization that feeds these representations** → [10.01](../01-text-preprocessing/)
- **SVD/PCA, the machinery behind LSA** → [04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)
- **Sequence models that capture order natively** → [Part 9](../../09-sequence-models/)
- **NLP tasks these features feed** → [10.04](../04-nlp-tasks/)
