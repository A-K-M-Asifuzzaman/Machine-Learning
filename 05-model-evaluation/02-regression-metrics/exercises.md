# 05.02 — Exercises: Regression Metrics

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Prove that the constant $c$ minimizing $\sum_i (y_i - c)^2$ is the mean $\bar y$, and that the
constant minimizing $\sum_i |y_i - c|$ is the median.

**D2.** Derive the constant minimizing the pinball loss $L_\tau$ and show it is the empirical
$\tau$-quantile. (Hint: differentiate the expected loss and set the subgradient to zero.)

**D3.** Show that minimizing MSE corresponds to Gaussian maximum likelihood and minimizing MAE to
Laplace maximum likelihood. What noise assumption does each metric encode?

**D4.** Prove $\mathrm{RMSE} \ge \mathrm{MAE}$ always (Jensen / power-mean inequality), with equality
iff all absolute errors are equal. Interpret the ratio RMSE/MAE.

**D5.** Derive $R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$ and show that for an OLS fit with
intercept on the training data, $R^2 = \mathrm{corr}(y,\hat y)^2$. Where does the proof use the
intercept and the OLS normal equations?

**D6.** Show by example that $R^2 < 0$ is possible on held-out data. What does $R^2 = 0$ correspond
to?

**D7.** Derive adjusted $R^2$ and explain why plain $R^2$ never decreases when a feature is added,
while adjusted $R^2$ can.

**D8.** Show MAPE is asymmetric: fix $y$ and show the percentage error is unbounded above (over-
prediction) but capped at 100% below (under-prediction to 0). Conclude the direction of its bias.

**D9.** Show RMSLE is RMSE applied to $\log(1+y)$, that it approximates the relative error for small
errors, and that it penalizes under-prediction more than over-prediction.

**D10.** Write the Huber loss and show it is $C^1$ (continuous value and derivative) at $|r|=\delta$.
What happens to it as $\delta\to 0$ and $\delta\to\infty$?

---

## Tier 2 — Implementation

**I1.** Implement MSE, RMSE, MAE, R², adjusted R², MAPE, sMAPE, RMSLE, Huber, and pinball loss.
Verify all against `sklearn.metrics`.

**I2.** Reproduce Experiment 1: for skewed targets, find each metric's minimizing constant and
confirm it is the mean / median / quantile.

**I3.** Reproduce Experiment 2: sweep the number of injected outliers and plot RMSE, MAE, and their
ratio. At what contamination does RMSE double?

**I4.** Reproduce Experiment 3: confirm $R^2 = \mathrm{corr}^2$ in-sample, then construct a held-out
set where $R^2 < 0$.

**I5.** Reproduce Experiment 4: measure MAPE's optimal constant on skewed data and confirm it sits
below the median. Then fit two regressors, one minimizing MSE and one minimizing MAPE, and show the
MAPE model systematically under-predicts.

**I6.** Reproduce Experiment 5: show RMSLE is scale-invariant for a fixed ratio error and asymmetric
in direction.

**I7.** Reproduce Experiment 6: construct two models whose RMSE and MAE rankings disagree.

**I8.** Implement quantile regression by minimizing pinball loss (via `scipy.optimize` or gradient
descent). Produce a 10/50/90 prediction interval and check empirical coverage.

**I9.** *(Weighted metrics.)* Implement sample-weighted MSE and MAE, and show how weights change the
optimal constant (weighted mean / weighted median).

**I10.** Take a real dataset, fit one model, and report all metrics. Write one sentence on which
metric you would headline and why, from the error cost.

---

## Tier 3 — Interview

**Q1.** What's the difference between a loss and a metric?

**Q2.** RMSE or MAE — how do you choose?

**Q3.** What does R² mean, and can it be negative?

**Q4.** Is R² the same as squared correlation?

**Q5.** Why is MAPE dangerous?

**Q6.** When would you use RMSLE?

**Q7.** What's the optimal constant prediction under MSE? Under MAE?

**Q8.** Your RMSE is much larger than your MAE. What does that tell you?

**Q9.** How do you evaluate a model that must predict a range, not a point?

**Q10.** Two models: one wins on RMSE, the other on MAE. Which is better?

**Q11.** You trained with MSE but the business cares about median error. Is that a problem?

**Q12.** Why might you evaluate on $\log(y)$ instead of $y$?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive each metric's optimal constant and name the estimator it implies
- [ ] Choose RMSE vs MAE from the error's cost, not habit
- [ ] Explain R²'s baseline, its negativity, and when it equals corr²
- [ ] State MAPE's asymmetry and RMSLE's opposite asymmetry
- [ ] Explain how the metric defines what "best model" means
- [ ] Produce a prediction interval with quantile loss
