# 10.03 — Exercises: Word Embeddings

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the distributional hypothesis and explain how "predict the context" turns it into a
learning objective.

**D2.** Write the skip-gram softmax objective and explain why the full-vocabulary softmax is
prohibitively expensive.

**D3.** Derive the negative-sampling objective
$-\log\sigma(w\cdot c^+) - \sum_k \log\sigma(-w\cdot c^-_k)$ and its gradient w.r.t. $w$.

**D4.** Explain the $\text{freq}^{0.75}$ noise distribution and why it beats sampling by raw frequency
or uniformly.

**D5.** Contrast skip-gram and CBOW: which predicts what, and when each is preferred.

**D6.** Explain why parallel relationships (king:man :: queen:woman) become parallel offsets, so that
$\text{king}-\text{man}+\text{woman}\approx\text{queen}$.

**D7.** Write the GloVe objective and explain it as a weighted least-squares factorization of the log
co-occurrence matrix. What does the weighting $f(X)$ do?

**D8.** Sketch Levy & Goldberg's result that SGNS implicitly factorizes a shifted PMI matrix, linking
Word2Vec and GloVe.

**D9.** Explain FastText's subword representation and why it solves the OOV problem.

**D10.** Explain why static embeddings fail on polysemy and how contextual embeddings fix it.

---

## Tier 2 — Implementation

**I1.** Implement the SGNS objective and verify its gradient against finite differences (Experiment 1).

**I2.** Implement full skip-gram training with negative sampling and the $0.75$ noise distribution.

**I3.** Reproduce Experiment 2: train on a structured corpus and show interchangeable words reach cosine
~0.99.

**I4.** Reproduce Experiment 3: build a corpus with a relational structure and solve an analogy by
vector arithmetic.

**I5.** Implement CBOW and compare its embeddings to skip-gram on the same corpus.

**I6.** Implement GloVe (co-occurrence matrix + weighted least squares) and reproduce Experiment 4.

**I7.** Implement FastText-style subword vectors (sum of char n-gram embeddings) and show an OOV word
gets a sensible vector.

**I8.** Load pretrained GloVe/Word2Vec vectors and evaluate analogy accuracy on the Google analogy set.

**I9.** Measure embedding bias (e.g. the WEAT test or a gender-direction projection) on pretrained
vectors.

**I10.** *(Nearest neighbors.)* Build a cosine nearest-neighbor lookup and explore the neighborhoods of
several words.

---

## Tier 3 — Interview

**Q1.** What is a word embedding and why is it better than one-hot / bag-of-words?

**Q2.** What is the distributional hypothesis?

**Q3.** How does Word2Vec's skip-gram work?

**Q4.** What is negative sampling and why is it used?

**Q5.** How do word analogies work in embedding space?

**Q6.** What is the difference between Word2Vec and GloVe?

**Q7.** What problem does FastText solve?

**Q8.** Why do static embeddings struggle with polysemy?

**Q9.** How are word embeddings biased, and why?

**Q10.** How do contextual embeddings differ from static ones?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the distributional hypothesis and its learning objective
- [ ] Derive the negative-sampling objective and gradient
- [ ] Implement skip-gram from scratch
- [ ] Explain why analogies emerge as vector offsets
- [ ] Relate GloVe and Word2Vec as co-occurrence factorizations
- [ ] Explain FastText's OOV handling
- [ ] Explain the static-vs-contextual limit and embedding bias
