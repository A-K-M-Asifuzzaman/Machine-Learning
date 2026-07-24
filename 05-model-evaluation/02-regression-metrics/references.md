# 05.02 — References: Regression Metrics

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Loss vs metric | ESL §7.2; Kuhn & Johnson §5 |
| §2-§4 | MSE/MAE, optimal constants | ESL §2.4; Hastie et al. §7.2 |
| §5 | R², adjusted R² | ESL §3.2; Draper & Smith Ch. 5 |
| §6 | MAPE / sMAPE and asymmetry | Hyndman & Koehler (2006); Makridakis (1993) |
| §7 | RMSLE | Kaggle competition conventions; Tofallis (2015) |
| §8 | Huber, pinball loss | Huber (1964); Koenker & Bassett (1978) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §2.4 derives the conditional mean as the MSE
minimizer and the conditional median as the MAE minimizer (§4 here); §3.2 covers R² and the geometry
of least squares; Chapter 7 frames loss vs metric and the decomposition of the error being measured.

**Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed.**
— free at <https://otexts.com/fpp3/>. The best free reference on forecast accuracy metrics: §5.8
"Evaluating point forecast accuracy" covers MAE, RMSE, MAPE, and scale-free alternatives (MASE), with
clear warnings about MAPE's failures (§6 here).

**Koenker, R. (2005). *Quantile Regression*.** The definitive book on the pinball loss and quantile
estimation (§8); the theory behind prediction intervals.

**Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*.** Chapter 5 "Measuring Performance in
Regression Models" is a clean applied companion, with the practical loss-vs-metric distinction (§1).

---

## Papers

- **Hyndman, R. J. & Koehler, A. B. (2006).** "Another look at measures of forecast accuracy."
  *International Journal of Forecasting* 22(4), 679-688. — **the reference critique of MAPE and sMAPE**
  (§6), proposing the scale-free MASE as a robust alternative. Free at
  <https://robjhyndman.com/papers/mase.pdf>.
- **Makridakis, S. (1993).** "Accuracy measures: theoretical and practical concerns."
  *International Journal of Forecasting* 9(4), 527-529. — documents MAPE's asymmetry and its
  under-forecast bias (§6, Experiment 4).
- **Tofallis, C. (2015).** "A better measure of relative prediction accuracy for model selection and
  model estimation." *JORS* 66(8), 1352-1362. — argues for the log-accuracy ratio (closely related to
  RMSLE, §7) as a symmetric relative-error alternative to MAPE.
- **Huber, P. J. (1964).** "Robust estimation of a location parameter." *Annals of Mathematical
  Statistics* 35(1), 73-101. — introduces the **Huber loss** (§8); the origin of robust regression.
- **Koenker, R. & Bassett, G. (1978).** "Regression quantiles." *Econometrica* 46(1), 33-50. — the
  **pinball loss** and quantile regression (§8).
- **Willmott, C. J. & Matsuura, K. (2005).** "Advantages of the mean absolute error (MAE) over the
  root mean square error (RMSE)." *Climate Research* 30, 79-82. — the classic argument for MAE's
  interpretability; useful counterpoint reading for §2-§3.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.metrics` regression](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_regression.py) | `mean_squared_error`, `mean_absolute_error`, `r2_score`, `mean_absolute_percentage_error`, `mean_squared_log_error`, `mean_pinball_loss` — every metric `from_scratch.py` is checked against |
| [`sklearn` docs: regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) | the formulas and caveats, including R²'s negativity |
| [`statsmodels` OLS summary](https://github.com/statsmodels/statsmodels) | R², adjusted R², and F-statistics computed the classical way |

---

## Deferred to later chapters

- **Bias-variance — what these metrics are decomposing** → [05.01](../01-bias-variance-and-theory/)
- **Classification metrics — the threshold and imbalance problems** → [05.03](../03-classification-metrics/)
- **Cross-validation — how to estimate these metrics honestly** → [05.04](../04-cross-validation/)
- **Quantile & probabilistic forecasting — pinball loss in full** → [15.xx time series]
- **Proper scoring rules & calibration for probabilistic predictions** → [05.06](../06-calibration/)
