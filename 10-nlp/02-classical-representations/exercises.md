# 10.02 — Exercises: Classical Text Representations

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Define the document–term matrix and explain why it is sparse. Give its dimensions for a corpus
of $n$ documents and vocabulary $|V|$.

**D2.** Derive scikit-learn's TF-IDF: $\text{tf} \cdot (\ln\frac{1+n}{1+\text{df}} + 1)$ followed by L2
normalization. Explain each smoothing term.

**D3.** Explain why raw counts overweight common words and how IDF corrects it. What is the IDF of a
word appearing in every document?

**D4.** Show that bag-of-words is a multiset (order-invariant), and that adding bigrams distinguishes
"not good" from "good not".

**D5.** Derive cosine similarity and explain why it is preferred over Euclidean distance for documents
of different lengths.

**D6.** Explain LSA as a truncated SVD $X \approx U_k \Sigma_k V_k^\top$ and how projecting into the
top-$k$ space reveals synonymy through co-occurrence.

**D7.** Show that LSA is the same operation as PCA on the term–document matrix; relate to
[04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/).

**D8.** State the NMF objective $\min_{W,H \ge 0}\lVert X - WH\rVert_F^2$ and explain why non-negativity
yields parts-based, interpretable topics.

**D9.** Describe LDA's generative story (document → topic distribution → word) and how it differs from
NMF.

**D10.** State the two fundamental limits of count-based representations (order, meaning) and how
embeddings address the second.

---

## Tier 2 — Implementation

**I1.** Implement a count vectorizer; verify against `sklearn.CountVectorizer` (Experiment 1).

**I2.** Implement TF-IDF with sklearn's exact smoothing and L2 norm; verify against `TfidfVectorizer`
(Experiment 2).

**I3.** Reproduce Experiment 2b: compute IDF weights and show ubiquitous words are downweighted.

**I4.** Implement n-gram features; reproduce Experiment 3 (bigrams separate opposite-sentiment
sentences).

**I5.** Build a TF-IDF cosine search engine over a document collection and evaluate retrieval on a few
queries.

**I6.** Implement LSA via truncated SVD; reproduce Experiment 4 (synonymy) and verify against
`TruncatedSVD`.

**I7.** Implement NMF (multiplicative updates) and reproduce Experiment 5; verify topics against
`sklearn.NMF`.

**I8.** Train an LDA topic model (`sklearn.LatentDirichletAllocation` or Gibbs sampling) and compare
topics to NMF.

**I9.** Train a TF-IDF + logistic-regression text classifier and compare accuracy to a bag-of-words
baseline.

**I10.** *(Analysis.)* Measure how classification accuracy changes as you add bigrams and trigrams, and
where it starts to overfit.

---

## Tier 3 — Interview

**Q1.** What is a bag-of-words representation?

**Q2.** What is TF-IDF and what problem does it solve?

**Q3.** Why is cosine similarity used for documents?

**Q4.** What does bag-of-words lose, and how do n-grams help?

**Q5.** What is LSA and what does it discover?

**Q6.** How does LSA relate to PCA/SVD?

**Q7.** What is the difference between NMF and LDA topic models?

**Q8.** Why does NMF give interpretable topics?

**Q9.** What are the two fundamental limitations of count-based text features?

**Q10.** When would you still use TF-IDF over embeddings/transformers today?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Build a document–term matrix and TF-IDF from scratch
- [ ] Explain and derive the IDF weighting
- [ ] Use n-grams to recover local order and know their cost
- [ ] Build a cosine-similarity retrieval system
- [ ] Explain LSA as truncated SVD and what it recovers
- [ ] Distinguish LSA, NMF, and LDA topic models
- [ ] State the order/meaning limits that motivate embeddings
