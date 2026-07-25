# 10.02 — References: Classical Text Representations

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1, §4 | Vector space model, bag-of-words | Salton et al. (1975) |
| §2 | TF-IDF | Spärck Jones (1972); Salton & Buckley (1988) |
| §3 | N-grams | Manning & Schütze (1999) |
| §4 | Cosine, retrieval, BM25 | Manning, Raghavan & Schütze (2008) |
| §5 | Latent Semantic Analysis | Deerwester et al. (1990) |
| §6 | NMF | Lee & Seung (1999) |
| §6 | LDA | Blei, Ng & Jordan (2003) |

---

## Foundational papers

- **Salton, G., Wong, A. & Yang, C. (1975).** "A Vector Space Model for Automatic Indexing." *CACM*. —
  the **vector space model** (§1, §4).
- **Spärck Jones, K. (1972).** "A statistical interpretation of term specificity and its application in
  retrieval." *Journal of Documentation*. — the origin of **IDF** (§2).
- **Salton, G. & Buckley, C. (1988).** "Term-weighting approaches in automatic text retrieval."
  *Information Processing & Management*. — the **TF-IDF** weighting family (§2).
- **Deerwester, S. et al. (1990).** "Indexing by Latent Semantic Analysis." *JASIS*. — **LSA** via SVD
  and the synonymy argument (§5). <https://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>.
- **Lee, D. & Seung, H. (1999).** "Learning the parts of objects by non-negative matrix
  factorization." *Nature*. — **NMF** and its parts-based interpretation (§6).
  <https://www.nature.com/articles/44565>.
- **Blei, D., Ng, A. & Jordan, M. (2003).** "Latent Dirichlet Allocation." *JMLR*. — **LDA**, the
  probabilistic topic model (§6). <https://www.jmlr.org/papers/v3/blei03a.html>.

---

## Textbooks

- **Manning, C., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*.** — the
  reference for TF-IDF, the vector space model, cosine ranking, and BM25. Free at
  <https://nlp.stanford.edu/IR-book/>.
- **Manning, C. & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*.** —
  n-grams and count-based NLP.
- **Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed. draft).** — vector semantics,
  TF-IDF, and LSA. Free at <https://web.stanford.edu/~jurafsky/slp3/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.feature_extraction.text`](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | `CountVectorizer`, `TfidfVectorizer` — verified against here |
| [`sklearn.decomposition`](https://scikit-learn.org/stable/modules/decomposition.html) | `TruncatedSVD` (LSA), `NMF`, `LatentDirichletAllocation` |
| [`gensim`](https://radimrehurek.com/gensim/) | LSA, LDA, and topic modeling at scale |
| [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) | BM25, the modern TF-IDF retrieval scorer |

---

## Deferred to later chapters

- **Word embeddings — the fix for the meaning problem** → [10.03](../03-word-embeddings/)
- **Tokenization feeding these features** → [10.01](../01-text-preprocessing/)
- **PCA/SVD, the machinery of LSA** → [04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)
- **Sequence models for word order** → [Part 9](../../09-sequence-models/)
- **NLP tasks and metrics** → [10.04](../04-nlp-tasks/)
