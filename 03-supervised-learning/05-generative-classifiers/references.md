# 03.05 — References: Generative Classifiers

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Generative vs discriminative | Ng & Jordan (2001); Bishop, *PRML*, §1.5.4, §4.3 |
| §2 | Bayes optimal classifier | Hastie et al., *ESL*, §2.4; Devroye, Györfi & Lugosi (1996) |
| §3 | Naive Bayes and its variants | McCallum & Nigam (1998); Manning et al., *IIR*, Ch. 13 |
| §4 | Smoothing as a Dirichlet prior | Manning et al., *IIR*, §13.2; Murphy, *PML*, §4.6.3 |
| §5 | Why NB works despite the assumption | Domingos & Pazzani (1997); Zhang (2004) |
| §6 | Miscalibration | Zadrozny & Elkan (2001); Niculescu-Mizil & Caruana (2005) |
| §7-§8 | LDA and QDA | Hastie et al., *ESL*, §4.3; Bishop §4.2 |
| §8 | Regularized discriminant analysis | Friedman (1989) |
| §9 | The covariance spectrum | Hastie et al., *ESL*, §4.3.1; Bickel & Levina (2004) |
| §10 | LDA vs logistic regression | Efron (1975); Hastie et al., *ESL*, §4.4.5 |
| §11 | Fisher's discriminant | Fisher (1936); Hastie et al., *ESL*, §4.3.3 |
| §12 | Sample complexity | Ng & Jordan (2001) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
**Chapter 4 is the source for §7-§11.** §4.3 derives LDA and QDA and contains the parameter-count
argument of §8; §4.3.1 has the regularized version; §4.3.3 covers the reduced-rank/Fisher view;
§4.4.5 is the LDA-vs-logistic comparison of §10.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.** — free
§4.2 for probabilistic generative models (the LDA derivation, done carefully with all the algebra
shown), §4.3 for the discriminative counterpart. §1.5.4 has the cleanest short statement of the
generative/discriminative distinction.

**Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*.**
— free at <https://nlp.stanford.edu/IR-book/>
**Chapter 13 is the reference for naive Bayes as a text classifier** — the multinomial/Bernoulli
distinction of §3.1, smoothing, and the practical details that make it work on real corpora.

**Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*.** — free at
<https://probml.github.io/pml-book/> — Chapter 9 for discriminant analysis, §4.6.3 for the
Dirichlet-prior view of smoothing.

**Devroye, L., Györfi, L. & Lugosi, G. (1996). *A Probabilistic Theory of Pattern Recognition*.
Springer.** — the rigorous treatment of the Bayes error and what it means for a classifier to be
consistent.

---

## Papers

### Naive Bayes
- **Domingos, P. & Pazzani, M. (1997).** "On the Optimality of the Simple Bayesian Classifier
  under Zero-One Loss." *Machine Learning* 30(2-3), 103-130. — **the paper that answers §5.**
  Shows NB is optimal over a much larger region of dependency space than the independence
  assumption would suggest, because zero-one loss depends only on the sign of the discriminant.
- **Zhang, H. (2004).** "The Optimality of Naive Bayes." *FLAIRS*. — a cleaner characterization:
  what matters is whether the *dependencies cancel* between classes, not whether they exist.
- **McCallum, A. & Nigam, K. (1998).** "A Comparison of Event Models for Naive Bayes Text
  Classification." *AAAI Workshop*. — the multinomial vs Bernoulli comparison of §3.1, with the
  finding that the right choice depends on document length.
- **Rennie, J. D. M. et al. (2003).** "Tackling the Poor Assumptions of Naive Bayes Text
  Classifiers." *ICML*. — TWCNB: several practical fixes (TF-IDF weighting, length
  normalization, complement classes) that close much of the gap to discriminative methods.

### Calibration
- **Zadrozny, B. & Elkan, C. (2001).** "Obtaining calibrated probability estimates from decision
  trees and naive Bayesian classifiers." *ICML*. — documents §6's failure and introduces binning
  and isotonic fixes.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting Good Probabilities With Supervised
  Learning." *ICML*. — the systematic empirical study; naive Bayes has the characteristic
  sigmoid-shaped reliability curve of a badly over-confident model.

### Discriminant analysis
- **Fisher, R. A. (1936).** "The Use of Multiple Measurements in Taxonomic Problems." *Annals of
  Eugenics* 7(2), 179-188. — the origin of both the method and the iris dataset.
- **Friedman, J. H. (1989).** "Regularized Discriminant Analysis." *JASA* 84(405), 165-175. —
  the $\gamma$ interpolation of §8, implemented as `RDA` in this chapter.
- **Bickel, P. J. & Levina, E. (2004).** "Some theory for Fisher's linear discriminant function,
  'naive Bayes', and some alternatives when there are many more variables than observations."
  *Bernoulli* 10(6), 989-1010. — the theory behind §9: why the diagonal (naive) covariance can
  beat the full one when $d$ is large relative to $n$.
- **Ledoit, O. & Wolf, M. (2004).** "A well-conditioned estimator for large-dimensional covariance
  matrices." *Journal of Multivariate Analysis* 88(2), 365-411. — the shrinkage estimator behind
  sklearn's `shrinkage="auto"`.

### Generative vs discriminative
- **Efron, B. (1975).** "The Efficiency of Logistic Regression Compared to Normal Discriminant
  Analysis." *JASA* 70(352), 892-898. — the original efficiency comparison of §10: when the
  Gaussian assumption holds, LDA needs roughly 30% less data for the same error.
- **Ng, A. Y. & Jordan, M. I. (2001).** "On Discriminative vs. Generative Classifiers: A
  comparison of logistic regression and naive Bayes." *NeurIPS*. — **§12's source.** Read the
  actual statement rather than the folklore version: it is about convergence *rates to each
  method's own asymptote*, which is why Experiment 4 finds the effect is conditional on how badly
  the assumption is violated.
- **Bouchard, G. & Triggs, B. (2004).** "The Tradeoff Between Generative and Discriminative
  Classifiers." *COMPSTAT*. — hybrid objectives that interpolate between the two.
- **Jaakkola, T. & Haussler, D. (1999).** "Exploiting Generative Models in Discriminative
  Classifiers." *NeurIPS*. — Fisher kernels: use a generative model to define features for a
  discriminative one.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `naive_bayes.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/naive_bayes.py) | all four variants; note everything is in log space, and note `_joint_log_likelihood` is the single method each subclass overrides |
| [scikit-learn `discriminant_analysis.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/discriminant_analysis.py) | `LinearDiscriminantAnalysis` with three solvers (`svd`, `lsqr`, `eigen`) and Ledoit-Wolf shrinkage; `_class_cov` shows the ÷n convention discussed in §7 |
| [scikit-learn `calibration.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) | the fix for §6 |

> **A note on the covariance convention.** sklearn's LDA computes the pooled covariance as
> $\sum_k \pi_k \hat{\boldsymbol{\Sigma}}_k$ with each $\hat{\boldsymbol{\Sigma}}_k$ the *biased*
> (÷$n_k$) estimate, which totals to ÷$n$. Most statistics texts, including ESL, use the unbiased
> ÷$(n-K)$. The two differ by a factor $(n-K)/n$, which leaves the argmax essentially unchanged but
> shifts the posterior probabilities slightly — because the log-prior term is *not* rescaled with
> the discriminants. `from_scratch.py` implements both and verifies against sklearn using its
> convention.

---

## A note on the name collision

**LDA** means two entirely unrelated things:

| | Linear Discriminant Analysis | Latent Dirichlet Allocation |
|---|---|---|
| Field | classification, this chapter | topic modelling |
| Supervised | yes | no |
| Output | a classifier / projection | topics over words |
| Reference | Fisher (1936) | Blei, Ng & Jordan (2003) |

In an NLP context "LDA" almost always means the second. Disambiguate explicitly.

---

## Deferred to later chapters

- **Gaussian mixture models and EM — generative models with latent structure** → [04.04](../../04-unsupervised-learning/04-gaussian-mixtures/)
- **PCA in full, and why it can discard the signal** → [04.06](../../04-unsupervised-learning/06-linear-dim-reduction/)
- **Calibration: reliability diagrams, Platt scaling, isotonic regression** → [05.06](../../05-model-evaluation/06-calibration/)
- **Topic models (the *other* LDA), and NMF** → [10.02](../../10-nlp/02-classical-representations/)
- **Text classification pipelines end to end** → [10.04](../../10-nlp/04-nlp-tasks/)
- **Generative models for anomaly detection** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
- **Modern deep generative models** → [Part 12](../../12-generative-models/)
