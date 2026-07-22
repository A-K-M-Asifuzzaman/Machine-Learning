# 00.04 — References: Statistics and Inference

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-3 | Sampling distributions, estimator properties | Wasserman, *All of Statistics*, Ch. 6 |
| §3 | Bias-variance decomposition of an estimator | Wasserman §6.3; Hastie et al., *ESL*, §7.3 |
| §4 | Maximum likelihood and its properties | Wasserman Ch. 9; Casella & Berger, *Statistical Inference*, Ch. 7 |
| §5 | Bias of the variance MLE | Wasserman §9.4; Casella & Berger §7.3 |
| §6 | Fisher information, Cramér-Rao | Casella & Berger §7.3; Lehmann & Casella, *Theory of Point Estimation*, Ch. 2 |
| §7 | Bayesian estimation, Bernstein-von Mises | Gelman et al., *BDA3*, Ch. 2, 4 |
| §8 | Confidence intervals | Wasserman §6.3.2; Brown, Cai & DasGupta (2001) for proportions |
| §8.3 | Wilson interval | Wilson (1927); Agresti & Coull (1998) |
| §9-10 | Hypothesis testing, p-value misuse | Wasserman Ch. 10; Wasserstein & Lazar (2016) |
| §11 | Multiple comparisons | Benjamini & Hochberg (1995); Holm (1979) |
| §12 | The bootstrap | Efron & Tibshirani, *An Introduction to the Bootstrap* |
| §13 | Permutation tests | Good, *Permutation, Parametric and Bootstrap Tests*; Phipson & Smyth (2010) |
| §15.2 | Comparing classifiers | Dietterich (1998); Demšar (2006) |
| §15.3 | CV standard errors | Bengio & Grandvalet (2004) |

---

## Books

**Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.**
The right book for someone who codes. Covers this entire chapter in Chapters 6-10 with no wasted
words. If you read one statistics book, read this.

**Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd ed. Duxbury.**
The standard graduate text. Consult for rigorous proofs of the MLE's properties (Ch. 7) and the
Cramér-Rao bound. Heavier going than Wasserman, and more complete.

**Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.**
The bootstrap book, by its inventor. Chapters 6, 12-14 cover the percentile and BCa intervals;
Chapter 8 covers where it fails. Efron's original 1979 paper is also worth reading for how simple
the core idea is.

**Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.** — free at
<http://www.stat.columbia.edu/~gelman/book/>
The Bayesian reference. Chapters 2 and 4 for §7, including the Bernstein-von Mises result that
posteriors become Gaussian around the MLE.

**Efron, B. & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge.** — free at
<https://hastie.su.domains/CASI/>
The bridge between classical statistics and machine learning, written by two people who built
both. Excellent on where the two traditions agree and where they do not.

**Reinhart, A. (2015). *Statistics Done Wrong*. No Starch Press.** — free draft at
<https://www.statisticsdonewrong.com/>
Short, readable catalogue of exactly the errors in §10, §11, and §16. Read it in an afternoon.

---

## Papers

### The p-value problem
- **Wasserstein, R. L. & Lazar, N. A. (2016).** "The ASA Statement on p-Values: Context, Process,
  and Purpose." *The American Statistician* 70(2), 129-133. — the American Statistical
  Association's formal statement, issued after decades of misuse. Short; read it once.
- **Ioannidis, J. P. A. (2005).** "Why Most Published Research Findings Are False." *PLoS Medicine*
  2(8), e124. — the consequences of §9.1 and §11 compounded across a literature.
- **Gelman, A. & Loken, E. (2014).** "The Statistical Crisis in Science." *American Scientist*
  102(6), 460. — the "garden of forking paths": multiple comparisons that occur even without
  explicit multiple testing, because analysis choices are made after seeing data. Directly
  relevant to ML experimentation.
- **Benjamin, D. J. et al. (2018).** "Redefine statistical significance." *Nature Human Behaviour*
  2, 6-10.

### Multiple comparisons
- **Holm, S. (1979).** "A simple sequentially rejective multiple test procedure." *Scandinavian
  Journal of Statistics* 6(2), 65-70.
- **Benjamini, Y. & Hochberg, Y. (1995).** "Controlling the False Discovery Rate." *JRSS B* 57(1),
  289-300. — one of the most-cited statistics papers ever written, and the reason large-scale
  screening is possible at all.

### Confidence intervals for proportions
- **Wilson, E. B. (1927).** "Probable inference, the law of succession, and statistical inference."
  *JASA* 22(158), 209-212.
- **Brown, L. D., Cai, T. T. & DasGupta, A. (2001).** "Interval Estimation for a Binomial
  Proportion." *Statistical Science* 16(2), 101-133. — the definitive comparison; the source of
  the recommendation in §8.3 to never use the Wald interval. Their coverage plots are what
  Experiment 1 reproduces.

### Comparing classifiers
- **Dietterich, T. G. (1998).** "Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms." *Neural Computation* 10(7), 1895-1923. — where McNemar's
  test for classifiers comes from, with a comparison of the alternatives' Type I error rates.
- **Demšar, J. (2006).** "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*
  7, 1-30. — the standard reference for comparing many models across many datasets.
- **Bengio, Y. & Grandvalet, Y. (2004).** "No Unbiased Estimator of the Variance of K-Fold
  Cross-Validation." *JMLR* 5, 1089-1105. — the result behind §15.3.
- **Bouthillier, X. et al. (2021).** "Accounting for Variance in Machine Learning Benchmarks."
  *MLSys*. — measures how much reported improvements are seed variance. Sobering.

### Permutation tests
- **Phipson, B. & Smyth, G. K. (2010).** "Permutation P-values Should Never Be Zero."
  *Statistical Applications in Genetics and Molecular Biology* 9(1). — the source of the `+1` in
  the permutation p-value in `from_scratch.py`.

---

## Courses

| Course | Institution | Link |
|---|---|---|
| Statistical Rethinking (Bayesian) | McElreath | <https://xcelab.net/rm/> — lectures free on YouTube |
| 36-401 Modern Regression | CMU | <https://www.stat.cmu.edu/~cshalizi/mreg/> |
| Stat 110 → All of Statistics path | Harvard / CMU | see [00.03 references](../03-probability/references.md) |

**Statistical Rethinking** deserves the recommendation even if you never intend to be Bayesian:
McElreath's treatment of what models actually assume, and of the difference between statistical
and causal claims, is unusually good.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`scipy.stats`](https://github.com/scipy/scipy/tree/main/scipy/stats) | `_stats_py.py` — `ttest_ind`, `bootstrap`, `permutation_test`; `_binomtest.py` for exact intervals |
| [`statsmodels`](https://github.com/statsmodels/statsmodels) | `stats/multitest.py` (all the corrections in §11); `stats/proportion.py` (Wilson, Agresti-Coull, Clopper-Pearson) |
| [`scikit-learn`](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/model_selection) | `_split.py` — why `GroupKFold` and `TimeSeriesSplit` exist, and what independence assumption each protects |
| [`pingouin`](https://github.com/raphaelvallat/pingouin) | a statistics library that reports effect sizes and confidence intervals *by default* — a good model for how results should be presented |

---

## Deferred to later chapters

- **Bias-variance for predictions, learning curves, double descent** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Cross-validation done correctly, nested CV** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Calibration — whether predicted probabilities are honest** → [05.06](../../05-model-evaluation/06-calibration/)
- **PAC learning, VC dimension, generalization bounds** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **A/B testing and online experimentation in production** → [19.03](../../19-mlops/03-monitoring/)
- **Conformal prediction — distribution-free prediction intervals** → [18.01](../../18-responsible-ml/01-fairness/)
