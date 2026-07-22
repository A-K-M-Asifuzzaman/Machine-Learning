# 03.01 — References: Linear Regression

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-3 | The model, three derivations | Hastie et al., *ESL*, §3.2; Bishop, *PRML*, §3.1 |
| §4 | Numerical solution | Golub & Van Loan, *Matrix Computations*, Ch. 5; Trefethen & Bau, Lecture 11 |
| §5 | The assumptions | Wooldridge, *Introductory Econometrics*, Ch. 2-3; Fox, *Applied Regression Analysis*, Ch. 6 |
| §6 | Gauss-Markov | Greene, *Econometric Analysis*, §4.3; Wooldridge Thm 3.4 |
| §7-8 | Sampling distribution, inference | Wasserman, *All of Statistics*, §13.3; Greene Ch. 4-5 |
| §9 | R², adjusted R² | James et al., *ISL*, §3.1.3; Hastie et al., *ESL*, §3.2 |
| §10 | Leverage, Cook's distance | Cook (1977); Belsley, Kuh & Welsch (1980); Fox Ch. 11 |
| §11.1 | Omitted-variable bias | Wooldridge §3.3 — the conditional-on-correlation form |
| §12 | Multicollinearity, VIF | Belsley, Kuh & Welsch (1980); Fox Ch. 13 |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
Chapter 3 is the ML-oriented treatment: least squares, its geometry, and the direct route from
there into ridge and lasso. §3.2 covers everything in §2-§9 here from a statistician's angle.

**James, G., Witten, D., Hastie, T. & Tibshirani, R. (2021). *An Introduction to Statistical
Learning*, 2nd ed.** — free at <https://www.statlearning.com/>
Chapter 3 is the gentle version of the same material, and the best first read if any derivation
here moved too fast. The Python edition (ISLP) has runnable labs.

**Wooldridge, J. M. (2019). *Introductory Econometrics: A Modern Approach*, 7th ed. Cengage.**
The best treatment of the *assumptions* — what each one buys, what breaks without it, and what to
do instead. Chapters 2-3 for the basics, 8 for heteroscedasticity, 12 for autocorrelation.
Economists take assumption violations far more seriously than ML texts do, and §5, §11, and §11.1
of this chapter follow their framing.

**Fox, J. (2015). *Applied Regression Analysis and Generalized Linear Models*, 3rd ed. Sage.**
The reference for diagnostics. Chapters 11-13 cover leverage, influence, and collinearity in the
detail §10 and §12 summarize.

**Greene, W. H. (2018). *Econometric Analysis*, 8th ed. Pearson.**
The comprehensive graduate reference. §4.3 has the cleanest statement and proof of Gauss-Markov.

**Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential
Data and Sources of Collinearity*. Wiley.**
Where leverage, DFFITS, DFBETAS, condition indices, and VIF were systematized. Still the
authoritative source for §10 and §12.

**Gelman, A., Hill, J. & Vehtari, A. (2020). *Regression and Other Stories*. Cambridge.** — free at
<https://avehtari.github.io/ROS-Examples/>
Unusually good on *interpretation* — what a coefficient means, when "holding others constant" is a
meaningful phrase, and when a regression supports a causal claim. Read it after the mechanics.

---

## Papers

- **Cook, R. D. (1977).** "Detection of Influential Observation in Linear Regression."
  *Technometrics* 19(1), 15-18. — Cook's distance, in four pages.
- **Anscombe, F. J. (1973).** "Graphs in Statistical Analysis." *The American Statistician* 27(1),
  17-21. — Anscombe's quartet: four datasets with identical regression summaries and completely
  different structure. The original argument for §10.
- **Matejka, J. & Fitzmaurice, G. (2017).** "Same Stats, Different Graphs: Generating Datasets with
  Varied Appearance and Identical Statistics through Simulated Annealing." *CHI*. — the
  "Datasaurus Dozen", Anscombe generalized. <https://www.autodesk.com/research/publications/same-stats-different-graphs>
- **Frisch, R. & Waugh, F. V. (1933).** "Partial Time Regressions as Compared with Individual
  Trends." *Econometrica* 1(4), 387-401. — the FWL theorem of exercise D15, which is what
  "controlling for" actually means.
- **White, H. (1980).** "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct
  Test for Heteroskedasticity." *Econometrica* 48(4), 817-838. — robust standard errors; the fix
  for Experiment 4's heteroscedastic row.
- **Long, J. S. & Ervin, L. H. (2000).** "Using Heteroscedasticity Consistent Standard Errors in
  the Linear Regression Model." *The American Statistician* 54(3), 217-224. — why HC3 rather than
  HC0 in small samples.
- **Berk, R. et al. (2013).** "Valid Post-Selection Inference." *Annals of Statistics* 41(2),
  802-837. — why the p-values in §8 are invalid after model selection, and what a valid procedure
  looks like.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_base.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_base.py) | `LinearRegression.fit` — note the dispatch to `scipy.linalg.lstsq` (SVD), and the separate sparse path |
| [statsmodels `linear_model.py`](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/linear_model.py) | `OLS` — the full inference machinery this chapter reimplements; read `RegressionResults` |
| [statsmodels `outliers_influence.py`](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/outliers_influence.py) | leverage, Cook's distance, DFFITS, VIF in production form |
| [`scipy.linalg.lstsq`](https://github.com/scipy/scipy/blob/main/scipy/linalg/_basic.py) | the LAPACK driver options (`gelsd`, `gelsy`, `gelss`) and what each trades |

**A note on the two libraries.** scikit-learn gives you `coef_` and nothing else — no standard
errors, no p-values, by design, because its audience is predicting. statsmodels gives you the full
inferential apparatus. If you are *explaining* rather than predicting, use statsmodels; if you
reach for sklearn and then wish you had a p-value, you were doing statistics, not machine
learning.

---

## Datasets for practice

| Dataset | Why | Source |
|---|---|---|
| Anscombe's quartet | the §10 lesson in four scatter plots | `seaborn.load_dataset("anscombe")` |
| Ames Housing | many features, real collinearity, needs transformation | `sklearn.datasets.fetch_openml("house_prices")` |
| Auto MPG | classic, visibly nonlinear in one feature | UCI |
| Diabetes | small, well-behaved, sklearn built-in | `sklearn.datasets.load_diabetes` |

> ⚠️ **Not the Boston Housing dataset.** It was removed from scikit-learn in 1.2 because one of
> its features was constructed on an explicitly racist premise. `fetch_california_housing` is the
> intended replacement; Ames is better for teaching anyway.

---

## Deferred to later chapters

- **Ridge, Lasso, Elastic Net — the Gauss-Markov loophole** → [03.02](../02-regularized-linear-models/)
- **Polynomials, splines, GAMs** → [03.03](../03-basis-expansion/)
- **Logistic regression and GLMs** → [03.04](../04-logistic-regression/)
- **Regression metrics beyond R²** → [05.02](../../05-model-evaluation/02-regression-metrics/)
- **Cross-validation instead of adjusted R²** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Time-series regression, autocorrelation done properly** → [15.01](../../15-time-series/01-classical/)
- **Causal interpretation of coefficients** → [17.01](../../17-explainable-ai/01-intrinsic/)
