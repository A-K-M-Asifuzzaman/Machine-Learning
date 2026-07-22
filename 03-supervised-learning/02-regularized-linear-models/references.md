# 03.02 — References: Regularized Linear Models

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Bias-variance rationale | Hastie et al., *ESL*, §3.4; Hoerl & Kennard (1970) |
| §2 | Ridge derivation and properties | Hastie et al., *ESL*, §3.4.1; Hoerl & Kennard (1970) |
| §3 | Ridge through the SVD | Hastie et al., *ESL*, §3.4.1 — the shrinkage-factor form |
| §4 | Effective degrees of freedom | Hastie et al., *ESL*, §3.4.4, §7.6 |
| §5-§6 | Lasso and sparsity | Tibshirani (1996); Hastie, Tibshirani & Wainwright (2015), Ch. 2 |
| §7 | Coordinate descent | Friedman, Hastie & Tibshirani (2010) |
| §8 | Elastic Net, grouping effect | Zou & Hastie (2005) |
| §9 | Bayesian interpretation | Park & Casella (2008); Murphy, *PML*, §11.4 |
| §11 | Choosing λ, the 1-SE rule | Hastie et al., *ESL*, §7.10; Friedman et al. (2010) |
| §12 | Regularization paths, LARS | Efron et al. (2004) |

---

## Books

**Hastie, T., Tibshirani, R. & Wainwright, M. (2015). *Statistical Learning with Sparsity: The
Lasso and Generalizations*. CRC Press.** — free at
<https://hastie.su.domains/StatLearnSparsity/>
**The book for this chapter**, by the people who invented most of it. Chapter 2 covers the Lasso
and the three sparsity arguments of §6; Chapter 5 covers coordinate descent and paths. Free and
not long.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§3.4 is the canonical treatment: ridge via the SVD, Lasso geometry, and the effective-degrees-of-
freedom argument. Figure 3.11 (the diamond-vs-circle picture) is the source of §6.1.

**James, G. et al. (2021). *An Introduction to Statistical Learning*, 2nd ed.** — free at
<https://www.statlearning.com/>
Chapter 6 is the gentler version, with good intuition for why sparsity matters.

**Bühlmann, P. & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.**
The theory: when does Lasso recover the true support, and under what conditions on the design
matrix (irrepresentable condition, restricted eigenvalue). The rigorous answer to "does Lasso pick
the right features?" — and the answer is "only under conditions real data rarely satisfies."

**Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*.** — free at
<https://probml.github.io/pml-book/> — §11.4 for the Bayesian view of §9, including why the
Bayesian Lasso posterior mean is not sparse.

---

## Papers

### Ridge
- **Hoerl, A. E. & Kennard, R. W. (1970).** "Ridge Regression: Biased Estimation for
  Nonorthogonal Problems." *Technometrics* 12(1), 55-67. — the original. Contains the proof that
  there always exists a $\lambda>0$ with lower MSE than OLS (exercise D5), which is the formal
  version of the Gauss-Markov loophole.
- **Theobald, C. M. (1974).** "Generalizations of Mean Square Error Applied to Ridge Regression."
  *JRSS B* 36(1), 103-106. — the multivariate strengthening of that result.

### Lasso
- **Tibshirani, R. (1996).** "Regression Shrinkage and Selection via the Lasso." *JRSS B* 58(1),
  267-288. — the paper. One of the most-cited in statistics, and very readable.
- **Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004).** "Least Angle Regression."
  *Annals of Statistics* 32(2), 407-499. — LARS, and the proof that the Lasso path is piecewise
  linear (§12).
- **Friedman, J., Hastie, T. & Tibshirani, R. (2010).** "Regularization Paths for Generalized
  Linear Models via Coordinate Descent." *Journal of Statistical Software* 33(1), 1-22. — the
  `glmnet` paper. §7's algorithm, plus the warm-start and active-set tricks that make paths cheap.
  Read this before implementing anything.
- **Zhao, P. & Yu, B. (2006).** "On Model Selection Consistency of Lasso." *JMLR* 7, 2541-2563. —
  the **irrepresentable condition**: the precise requirement for Lasso to recover the true
  support. It is restrictive, and it is why §14 warns against reading a Lasso zero as "irrelevant".
- **Meinshausen, N. & Bühlmann, P. (2010).** "Stability Selection." *JRSS B* 72(4), 417-473. — the
  principled fix for Experiment 5's instability: subsample repeatedly and keep features selected
  often.

### Elastic Net
- **Zou, H. & Hastie, T. (2005).** "Regularization and Variable Selection via the Elastic Net."
  *JRSS B* 67(2), 301-320. — §8. Both failures of Lasso and the grouping-effect theorem are
  stated and proved here. Worth checking their Theorem 1 against Experiment 5's measured
  threshold.

### Bayesian view
- **Park, T. & Casella, G. (2008).** "The Bayesian Lasso." *JASA* 103(482), 681-686. — the Laplace
  prior made properly Bayesian, and the demonstration that the posterior mean is **not** sparse.
- **Carvalho, C. M., Polson, N. G. & Scott, J. G. (2010).** "The horseshoe estimator for sparse
  signals." *Biometrika* 97(2), 465-480. — what you use when you want genuine Bayesian sparsity.

### Related shrinkage
- **Zou, H. (2006).** "The Adaptive Lasso and Its Oracle Properties." *JASA* 101(476), 1418-1429.
  — weighting the penalty per coefficient to fix Lasso's bias on large coefficients.
- **Meinshausen, N. (2007).** "Relaxed Lasso." *Computational Statistics & Data Analysis* 52(1),
  374-393. — exercise I14: select with Lasso, estimate with OLS.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_coordinate_descent.pyx`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_cd_fast.pyx) | the production coordinate descent — soft thresholding, active sets, and the duality-gap stopping rule |
| [scikit-learn `_ridge.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) | note how many solvers it has (`cholesky`, `svd`, `lsqr`, `sag`, `saga`) and when each is chosen |
| [scikit-learn `_ridge.py` — `_RidgeGCV`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) | leave-one-out CV in closed form from the SVD — the trick behind `RidgeCV` (§11) |
| [`glmnet`](https://glmnet.stanford.edu/) | the Fortran/R original. Its parameterization (loss divided by $n$) differs from sklearn's — the source of most confusion when transferring $\lambda$ values |
| [`celer`](https://github.com/mathurinm/celer) | a modern accelerated Lasso solver; good for seeing how far the active-set idea can be pushed |

**On parameterizations.** sklearn's `Ridge(alpha)` does not divide the loss by $n$; `Lasso(alpha)`
and `ElasticNet(alpha)` do (they use $\frac{1}{2n}\Vert y-Xw\Vert^2 + \alpha\Vert w\Vert_1$).
glmnet divides everywhere and calls it $\lambda$. This is why §14 says $\lambda$ values never
transfer — always re-tune, and read the docstring before comparing.

---

## Deferred to later chapters

- **Logistic regression with the same penalties** → [03.04](../04-logistic-regression/)
- **Group Lasso, fused Lasso, and structured sparsity** → [03.03](../03-basis-expansion/)
- **Cross-validation done correctly, nested CV** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Weight decay in deep learning — and why AdamW exists** → [07.08](../../07-deep-learning/08-regularization/)
- **Feature selection more broadly (filter, wrapper, embedded)** → [02.03](../../02-data/03-feature-engineering/)
- **Data leakage from preprocessing outside the fold** → [02.06](../../02-data/06-data-leakage/)
- **Post-selection inference — why regularized p-values are hard** → [00.04 §11](../../00-mathematical-foundations/04-statistics-and-inference/)
