# 03.03 — References: Basis Expansion

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | Basis expansion, polynomial pitfalls | Hastie et al., *ESL*, §5.1; Trefethen, *ATAP*, Ch. 13-15 |
| §3 | Runge phenomenon | Runge (1901); Trefethen, *Approximation Theory and Approximation Practice*, Ch. 13 |
| §4-§5 | Piecewise polynomials, regression splines | Hastie et al., *ESL*, §5.2; de Boor, *A Practical Guide to Splines*, Ch. 1 |
| §6 | B-splines, Cox-de Boor | de Boor (1978), Ch. 9; Cox (1972); de Boor (1972) |
| §7 | Natural cubic splines | Hastie et al., *ESL*, §5.2.1 and eq. 5.4-5.5 |
| §8 | Smoothing splines | Hastie et al., *ESL*, §5.4; Green & Silverman (1994); Wahba (1990) |
| §9 | GCV | Craven & Wahba (1979); Golub, Heath & Wahba (1979) |
| §10 | GAMs, backfitting | Hastie & Tibshirani (1986, 1990); Wood (2017) |
| §12 | Kernels and learned bases | Hastie et al., *ESL*, §5.8; Wahba (1990) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
**Chapter 5 is the source for most of this chapter.** §5.2 for regression splines and the natural
cubic basis (equations 5.4-5.5, implemented in `natural_cubic_basis`), §5.4 for smoothing splines
and the theorem in §8, §5.8 for the reproducing-kernel connection.

**Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*, 2nd ed. CRC Press.**
**The definitive GAM book**, by the author of R's `mgcv` — the best GAM implementation in
existence. Covers penalized regression splines, automatic smoothness selection (REML, GCV), and
inference for GAMs. If you intend to use GAMs seriously, read this rather than anything else.

**de Boor, C. (1978, revised 2001). *A Practical Guide to Splines*. Springer.**
The reference for splines as a numerical object. Chapter 9 has the Cox-de Boor recursion and its
stability analysis.

**Green, P. J. & Silverman, B. W. (1994). *Nonparametric Regression and Generalized Linear Models:
A Roughness Penalty Approach*. Chapman & Hall.**
The rigorous treatment of §8 — why the penalized problem has a natural-cubic-spline solution, and
what happens with other penalties.

**Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.**
The theoretical foundation, and the source of the RKHS view: smoothing splines *are* kernel
methods (§12, exercise D15).

**Trefethen, L. N. (2019). *Approximation Theory and Approximation Practice*, extended ed. SIAM.**
The best modern treatment of what polynomials can and cannot do. Chapters 13-15 cover Runge,
Chebyshev nodes, and why equally spaced interpolation is hopeless. Every chapter comes with
runnable Chebfun code.

**Hastie, T. & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.**
The original GAM monograph. Backfitting, its convergence, and the degrees-of-freedom accounting.

---

## Papers

- **Runge, C. (1901).** "Über empirische Funktionen und die Interpolation zwischen äquidistanten
  Ordinaten." *Zeitschrift für Mathematik und Physik* 46, 224-243. — the original example
  reproduced in Experiment 1.
- **Cox, M. G. (1972).** "The numerical evaluation of B-splines." *IMA Journal of Applied
  Mathematics* 10(2), 134-149.
- **de Boor, C. (1972).** "On calculating with B-splines." *Journal of Approximation Theory* 6(1),
  50-62. — Cox and de Boor independently; hence the recursion's name.
- **Craven, P. & Wahba, G. (1979).** "Smoothing noisy data with spline functions: estimating the
  correct degree of smoothing by the method of generalized cross-validation." *Numerische
  Mathematik* 31(4), 377-403. — GCV (§9).
- **Golub, G. H., Heath, M. & Wahba, G. (1979).** "Generalized Cross-Validation as a Method for
  Choosing a Good Ridge Parameter." *Technometrics* 21(2), 215-223. — the same idea for ridge,
  which is why `RidgeCV` is cheap ([03.02 §11](../02-regularized-linear-models/)).
- **Hastie, T. & Tibshirani, R. (1986).** "Generalized Additive Models." *Statistical Science*
  1(3), 297-310. — the paper that introduced them.
- **Eilers, P. H. C. & Marx, B. D. (1996).** "Flexible smoothing with B-splines and penalties."
  *Statistical Science* 11(2), 89-121. — **P-splines**: B-spline basis plus a difference penalty
  on the coefficients. This is the pragmatic middle ground between regression and smoothing
  splines, and the approach `mgcv` mostly uses. Note the difference-penalty subtlety of exercise
  I7 is discussed here.
- **Wood, S. N. (2011).** "Fast stable restricted maximum likelihood and marginal likelihood
  estimation of semiparametric generalized linear models." *JRSS B* 73(1), 3-36. — REML for
  smoothness selection, now preferred over GCV in `mgcv`.
- **Lou, Y., Caruana, R. & Gehrke, J. (2012).** "Intelligible Models for Classification and
  Regression." *KDD*. — GA²M / Explainable Boosting Machines: GAMs fitted by boosting, with
  optional pairwise interactions. The modern answer to §10's limitation.
- **Caruana, R. et al. (2015).** "Intelligible Models for HealthCare: Predicting Pneumonia Risk
  and Hospital 30-day Readmission." *KDD*. — the case study that made the interpretability
  argument concrete, including the famous "asthma lowers pneumonia risk" discovery that a black
  box would have hidden.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`scipy.interpolate`](https://github.com/scipy/scipy/tree/main/scipy/interpolate) | `BSpline`, `make_smoothing_spline`, `UnivariateSpline`; note the `extrapolate` flag and what it defaults to |
| [`sklearn.preprocessing.SplineTransformer`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_polynomial.py) | B-spline basis as a `Pipeline` step, with `extrapolation` options including `"periodic"` |
| [`sklearn.preprocessing.PolynomialFeatures`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_polynomial.py) | raw powers and interactions — note it does **not** orthogonalize, so §2's conditioning warning applies |
| [`patsy`](https://patsy.readthedocs.io/) | `bs()`, `cr()`, `cc()` — R-style formula splines in Python |
| [`pyGAM`](https://github.com/dswah/pyGAM) | GAMs in Python, with penalized B-splines and automatic smoothness selection |
| [`mgcv`](https://cran.r-project.org/package=mgcv) (R) | the gold standard for GAMs. Worth reading even if you work in Python — its documentation is a course in itself |
| [`interpret`](https://github.com/interpretml/interpret) | Explainable Boosting Machines — GAMs fitted by boosting, competitive with GBMs on tabular data |

---

## A note on extrapolation

Experiment 3 shows three different out-of-range behaviours: explosion (polynomial), **silent zero**
(B-spline), and linear extension (natural spline). The middle one is the trap, because it produces
a plausible-looking number with no warning. Before deploying any basis-expansion model, check what
your library does outside the training range:

| Library | Default outside range |
|---|---|
| `scipy.interpolate.BSpline` | `extrapolate=True` → extrapolates the polynomial; set `False` for `nan` |
| `sklearn.SplineTransformer` | `extrapolation="constant"` → clamps to the boundary value |
| `patsy.bs` | raises unless `include_intercept` handling is set up |
| `numpy.polynomial` | always extrapolates, unboundedly |

The defensible production choice is usually to detect out-of-range inputs and refuse, rather than
to answer.

---

## Deferred to later chapters

- **Kernel methods and the RKHS view** → [03.07](../07-svm/)
- **Trees as adaptive piecewise-constant basis expansion** → [03.08](../08-decision-trees/)
- **Gradient boosting as adaptive basis expansion** → [06.04](../../06-ensembles/04-gradient-boosting/)
- **Neural networks as *learned* basis expansion** → [07.01](../../07-deep-learning/01-neural-network-basics/)
- **Fourier features for periodic seasonality** → [15.01](../../15-time-series/01-classical/)
- **Partial dependence plots — the model-agnostic version of a GAM's $f_j$** → [17.02](../../17-explainable-ai/02-post-hoc/)
- **Explainable Boosting Machines** → [17.01](../../17-explainable-ai/01-intrinsic/)
