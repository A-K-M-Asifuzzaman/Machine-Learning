# 10.03 — References: Word Embeddings

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Distributional hypothesis | Harris (1954); Firth (1957) |
| §2-§4 | Word2Vec, skip-gram, CBOW, analogies | Mikolov et al. (2013a, 2013b) |
| §3 | Negative sampling | Mikolov et al. (2013b) |
| §5 | GloVe | Pennington et al. (2014) |
| §5 | SGNS as PMI factorization | Levy & Goldberg (2014) |
| §6 | FastText | Bojanowski et al. (2017) |
| §7 | Contextual embeddings | Peters et al. (2018, ELMo) |
| §7 | Embedding bias | Bolukbasi et al. (2016); Caliskan et al. (2017) |

---

## The core papers

- **Mikolov, T. et al. (2013a).** "Efficient Estimation of Word Representations in Vector Space." — the
  **skip-gram and CBOW** architectures (§2). <https://arxiv.org/abs/1301.3781>.
- **Mikolov, T. et al. (2013b).** "Distributed Representations of Words and Phrases and their
  Compositionality." *NeurIPS*. — **negative sampling**, the $0.75$ noise distribution, and the analogy
  results (§3-§4). <https://arxiv.org/abs/1310.4546>.
- **Pennington, J., Socher, R. & Manning, C. (2014).** "GloVe: Global Vectors for Word Representation."
  *EMNLP*. — the **global co-occurrence** factorization objective (§5).
  <https://nlp.stanford.edu/pubs/glove.pdf>.
- **Levy, O. & Goldberg, Y. (2014).** "Neural Word Embedding as Implicit Matrix Factorization."
  *NeurIPS*. — proves **SGNS implicitly factorizes a shifted PMI matrix**, unifying Word2Vec and GloVe
  (§5). <https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization>.
- **Bojanowski, P. et al. (2017).** "Enriching Word Vectors with Subword Information" (**FastText**).
  *TACL*. — character n-gram embeddings for OOV and morphology (§6). <https://arxiv.org/abs/1607.04606>.

## Foundations and follow-ups

- **Harris, Z. (1954).** "Distributional Structure." *Word*. — the **distributional hypothesis** (§1).
- **Firth, J. R. (1957).** "A synopsis of linguistic theory." — "You shall know a word by the company
  it keeps" (§1).
- **Peters, M. et al. (2018).** "Deep contextualized word representations" (**ELMo**). *NAACL*. — the
  move to **contextual** embeddings (§7). <https://arxiv.org/abs/1802.05365>.

## Bias

- **Bolukbasi, T. et al. (2016).** "Man is to Computer Programmer as Woman is to Homemaker? Debiasing
  Word Embeddings." *NeurIPS*. — gender bias as a direction in embedding space (§7).
  <https://arxiv.org/abs/1607.06520>.
- **Caliskan, A., Bryson, J. & Narayanan, A. (2017).** "Semantics derived automatically from language
  corpora contain human-like biases." *Science*. — the WEAT test (§7).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`gensim`](https://radimrehurek.com/gensim/models/word2vec.html) | reference Word2Vec / FastText training and pretrained vectors |
| [GloVe (Stanford)](https://github.com/stanfordnlp/GloVe) | the original GloVe code and pretrained vectors |
| [fastText (Facebook)](https://github.com/facebookresearch/fastText) | FastText training + pretrained multilingual vectors |
| [Embedding Projector](https://projector.tensorflow.org/) | interactively explore embedding neighborhoods |

---

## Deferred to later chapters

- **Contextual embeddings / transformers** → [Part 11](../../11-transformers-llms/)
- **Tokenization feeding embeddings** → [10.01](../01-text-preprocessing/)
- **Count-based representations** → [10.02](../02-classical-representations/)
- **Contrastive learning (the SGNS objective's cousin)** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Embedding bias and fairness** → [Part 18](../../18-fairness-privacy-robustness/)
