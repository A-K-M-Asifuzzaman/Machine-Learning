# 05.02 — Regression Metrics

> **Prerequisites**: [05.01](../01-bias-variance-and-theory/) (the error we are now measuring),
> [00.04](../../00-mathematical-foundations/04-statistics-and-inference/) (mean, median, quantiles as
> estimators), [03.01](../../03-supervised-learning/01-linear-regression/) (squared error and its
> Gaussian assumption).
> **You will be able to**: choose a regression metric from the *cost structure of the errors* rather
> than habit, explain what each metric's optimal constant predictor reveals about it, and avoid the
> standard traps (R² on the wrong baseline, MAPE's asymmetry, RMSE's outlier domination).

---

## Table of contents

1. [Loss vs metric — you optimize one and report the other](#1-loss-vs-metric--you-optimize-one-and-report-the-other)
2. [MSE and RMSE](#2-mse-and-rmse)
3. [MAE — robustness and the median](#3-mae--robustness-and-the-median)
4. [The optimal constant reveals the metric](#4-the-optimal-constant-reveals-the-metric)
5. [R² — variance explained, and its baseline](#5-r--variance-explained-and-its-baseline)
6. [Percentage errors: MAPE and sMAPE](#6-percentage-errors-mape-and-smape)
7. [RMSLE — errors across orders of magnitude](#7-rmsle--errors-across-orders-of-magnitude)
8. [Huber and quantile losses](#8-huber-and-quantile-losses)
9. [Choosing a metric is a modelling decision](#9-choosing-a-metric-is-a-modelling-decision)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Loss vs metric — you optimize one and report the other

Two different objects are easy to conflate:

- The **loss** is what the model minimizes during training (squared error, log loss). It must be
  differentiable and well-behaved for optimization.
- The **metric** is what you report and use to *compare* models. It should reflect the real-world
  cost of errors, and need not be differentiable.

They are often different on purpose. You might train with squared error (smooth, easy to optimize)
but report mean absolute error (matches the business cost) and select models on it. A metric encodes
*what a mistake actually costs* — and, as §4 makes precise, choosing a metric is implicitly choosing
what "the right answer" even means. Getting the metric wrong means optimizing a real model against
the wrong yardstick, which no amount of modelling can fix.

---

## 2. MSE and RMSE

**Mean squared error** and its root:

$$
\mathrm{MSE} = \frac1n\sum_{i=1}^n (y_i - \hat y_i)^2, \qquad
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}.
$$

- **Squares the error**, so a prediction off by 10 counts 100× a prediction off by 1. MSE is
  **dominated by the largest errors** — one gross outlier can swamp it.
- **RMSE is in the units of $y$** (MSE is in $y^2$), which is why RMSE is the one usually reported.
- **MSE is minimized by the conditional mean** $\mathbb{E}[y\mid\mathbf{x}]$ (§4), and corresponds to
  a **Gaussian noise** assumption — minimizing MSE is Gaussian maximum likelihood
  ([03.01](../../03-supervised-learning/01-linear-regression/)). It is also the quantity the
  bias-variance decomposition ([05.01](../01-bias-variance-and-theory/)) splits.

Use MSE/RMSE when large errors are disproportionately bad (their quadratic penalty is a *feature*)
and when the noise is roughly Gaussian without heavy outliers. Avoid it as your headline metric when
a few outliers would otherwise dictate the entire number.

---

## 3. MAE — robustness and the median

**Mean absolute error**:

$$
\mathrm{MAE} = \frac1n\sum_{i=1}^n |y_i - \hat y_i|.
$$

- **Linear in the error**: an error of 10 counts 10× an error of 1, not 100×. MAE is **robust** — a
  single outlier moves it by a bounded amount, not a squared one.
- **In the units of $y$**, directly interpretable as "typical absolute miss."
- **Minimized by the conditional median** (§4), so it corresponds to a **Laplace noise** assumption.

The RMSE-vs-MAE choice is exactly the mean-vs-median choice, and it is the single most consequential
metric decision in regression. RMSE $\ge$ MAE always, and the *gap between them* is a diagnostic:
a large RMSE/MAE ratio signals a heavy-tailed error distribution (a few big misses inflating RMSE).
Experiment 2 shows RMSE tracking a handful of injected outliers while MAE barely moves.

---

## 4. The optimal constant reveals the metric

Here is the cleanest way to understand *any* regression metric: ask what single constant $c$ best
predicts a set of targets under it. The answer is the metric's implied notion of "center."

| Metric | Minimizing constant $c^\star = \arg\min_c \sum L(y_i, c)$ |
|---|---|
| MSE | the **mean** $\bar y$ |
| MAE | the **median** |
| MAPE | (weighted) — pulls toward **smaller** values |
| Pinball loss ($\tau$) | the **$\tau$-quantile** |

This is not a curiosity — it is the metric's soul. Choosing MSE *is* choosing to be judged against
the conditional mean; choosing MAE is choosing the median; choosing the $\tau$-pinball loss is asking
for the $\tau$-quantile ([§8](#8-huber-and-quantile-losses)). If your business cares about the median
outcome but you optimize and report MSE, you have quietly asked for the mean — and on skewed data the
mean and median differ substantially. Experiment 1 confirms each metric's optimal constant is exactly
the estimator in the table, to numerical precision.

---

## 5. R² — variance explained, and its baseline

The **coefficient of determination** rescales MSE against a baseline:

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}
= 1 - \frac{\mathrm{SS}_{\text{res}}}{\mathrm{SS}_{\text{tot}}}.
$$

- The baseline is the **mean predictor** $\bar y$: $R^2$ is the fraction of the target's variance the
  model explains *beyond* just guessing the mean. $R^2 = 1$ is perfect; $R^2 = 0$ means "no better
  than predicting the mean."
- **$R^2$ can be negative** — a model *worse* than the mean predictor (common on a test set with a
  bad model, or when the test mean differs from the train mean). This surprises people who think of
  $R^2$ as a squared quantity bounded in $[0,1]$; on held-out data it is not.
- On the **training set of an OLS model with intercept**, $R^2$ equals the squared correlation between
  $y$ and $\hat y$ — but *only there*. On test data, or for nonlinear/biased models, $R^2 \ne
  \mathrm{corr}^2$, and conflating them is a classic error.
- **Adjusted $R^2$** penalizes adding features: $R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-p-1}$,
  so it does not rise just because you added a useless predictor (plain $R^2$ never decreases when you
  add features). Use it for *in-sample* feature-count comparisons; on a proper held-out set, plain
  $R^2$ already reflects overfitting.

Experiment 3 drives $R^2$ negative on a mismatched test set and shows $R^2 = \mathrm{corr}^2$ holding
for in-sample OLS but breaking on test data.

---

## 6. Percentage errors: MAPE and sMAPE

When errors should be judged **relative** to the target's size (a 10-unit miss matters more on a
20-unit item than on a 2000-unit one), percentage metrics are tempting:

$$
\mathrm{MAPE} = \frac{100}{n}\sum_i \left|\frac{y_i - \hat y_i}{y_i}\right|, \qquad
\mathrm{sMAPE} = \frac{100}{n}\sum_i \frac{|y_i - \hat y_i|}{(|y_i| + |\hat y_i|)/2}.
$$

They are scale-independent and intuitive ("we're off by 8% on average"), but carry sharp traps:

- **Undefined / explosive at $y_i = 0$** and huge for small $y_i$. Useless for targets that hit or
  approach zero.
- **Asymmetric.** MAPE penalizes **over-prediction more than under-prediction**: an over-forecast can
  push the ratio above 100%, but an under-forecast's error is capped at 100% of $y$. A model
  minimizing MAPE therefore learns to **under-predict systematically** — a real, silent bias.
  Experiment 4 measures this asymmetry directly.
- **sMAPE** softens the zero problem with a symmetric denominator but has its own oddities (it is
  bounded in $[0, 200]$ and is not actually symmetric in the way its name suggests).

Reach for percentage errors only when the target is safely bounded away from zero and relative error
is genuinely what the business tracks — and even then, know that MAPE quietly rewards under-forecasting.

---

## 7. RMSLE — errors across orders of magnitude

When the target spans several orders of magnitude (populations, prices, counts), squared error on the
raw scale is dominated entirely by the largest targets. The **root mean squared log error** measures
error in *log space*:

$$
\mathrm{RMSLE} = \sqrt{\frac1n\sum_i \big(\log(1 + y_i) - \log(1 + \hat y_i)\big)^2}.
$$

- It effectively measures the **relative (ratio) error**: $\log(1+\hat y) - \log(1+y) \approx$ the
  proportional miss, so being off by a factor of 2 costs the same whether $y = 10$ or $y = 10^6$.
- It is **asymmetric — it penalizes under-prediction more than over-prediction** (the opposite of
  MAPE). Predicting 50 when the truth is 100 costs more than predicting 200. This is often *desirable*
  (under-forecasting demand or capacity is the costlier mistake), which is why RMSLE is common in
  demand/price competitions.
- The `1 +` guards $y = 0$. Predictions are assumed non-negative.

Equivalently, RMSLE is just RMSE after a $\log(1+\cdot)$ transform — a reminder that **the scale you
evaluate on is part of the metric.** Experiment 5 shows RMSLE treating a factor-of-2 miss identically
across scales while RMSE does not, and its under-prediction penalty.

---

## 8. Huber and quantile losses

Two metrics that sit *between* or *beside* the classics:

**Huber** interpolates MSE and MAE — quadratic for small residuals, linear for large ones (the loss
of [06.04 §5](../../06-ensembles/04-gradient-boosting/)):

$$
L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r|\le\delta \\ \delta(|r| - \tfrac12\delta) & |r|>\delta \end{cases}, \qquad r = y - \hat y.
$$

It keeps MSE's smooth sensitivity near the fit while capping an outlier's influence like MAE — a
robust default when you want differentiability *and* outlier resistance.

**Pinball (quantile) loss** measures error for a specific quantile $\tau\in(0,1)$:

$$
L_\tau(y, \hat y) = \begin{cases} \tau\,(y - \hat y) & y \ge \hat y \\ (1-\tau)(\hat y - y) & y < \hat y \end{cases}
$$

Minimizing it yields the conditional **$\tau$-quantile** (§4), the basis of quantile regression and
prediction intervals: fit $\tau = 0.05$ and $\tau = 0.95$ to bracket 90% of outcomes. $\tau = 0.5$
recovers MAE (up to a factor). It is how you evaluate a model that must predict a *range*, not a point.

---

## 9. Choosing a metric is a modelling decision

There is no universally correct regression metric; the right one is dictated by **what an error
costs** in the problem at hand. A short decision guide:

| If... | Use | Because |
|---|---|---|
| Large errors are disproportionately bad; noise ~Gaussian | **RMSE** | quadratic penalty; conditional mean |
| Outliers present; want a typical miss | **MAE** | robust; conditional median |
| Want a unitless "how good, vs guessing the mean" | **R²** | variance explained (mind the negatives) |
| Relative error matters; target bounded away from 0 | **MAPE** (cautiously) | scale-free (but under-forecast bias) |
| Target spans orders of magnitude; under-forecast costlier | **RMSLE** | ratio error; under-prediction penalty |
| Need robustness *and* differentiability | **Huber** | MSE core, MAE tails |
| Need intervals / a specific quantile | **Pinball** | targets the $\tau$-quantile |

Two disciplines follow. First, **fix the metric before modelling** — choosing it after seeing results
invites cherry-picking. Second, **know that the metric can flip your model ranking**: a model that
wins on RMSE can lose on MAE because they reward different behaviors (fitting the mean vs the median).
Experiment 6 exhibits exactly such a rank reversal — proof that "which model is best?" is not
well-posed until the metric is named.

---

## 10. Common misconceptions

**"RMSE and MAE just differ by a square root; pick either."**
No — RMSE targets the mean and is outlier-dominated; MAE targets the median and is robust (§2–§4).
They can rank models differently (§9). The choice is the mean-vs-median choice.

**"R² is between 0 and 1."**
Only in-sample for a model with intercept. On held-out data $R^2$ can be **negative** — worse than
predicting the mean (§5).

**"R² is the squared correlation between $y$ and $\hat y$."**
Only for in-sample OLS with intercept. On test data or for biased/nonlinear models it is not (§5).

**"MAPE is a neutral, scale-free error."**
It is scale-free but **asymmetric** — it penalizes over-prediction more, so minimizing it biases the
model to under-forecast, and it explodes near $y = 0$ (§6).

**"The metric is just for reporting; it doesn't affect the model."**
It affects **model selection and hyperparameter tuning**, which is where most of a model's quality is
decided. And each metric implies a different optimal predictor (§4), so it defines what "best" means.

**"Evaluate on whatever scale is convenient."**
The scale *is* part of the metric: RMSLE is RMSE in log space and rewards entirely different behavior
(§7). Evaluating on transformed vs raw targets can reverse conclusions.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — every metric implemented in NumPy and verified against
  `sklearn.metrics`. Six experiments: (1) each metric's optimal constant is exactly mean/median/
  quantile; (2) RMSE's outlier domination vs MAE's robustness; (3) R² going negative and $R^2 =
  \mathrm{corr}^2$ only in-sample; (4) MAPE's under-prediction asymmetry; (5) RMSLE's scale-invariance
  and under-prediction penalty; (6) a model-ranking reversal between RMSE and MAE.
- **[exercises.md](exercises.md)** — derive each optimal constant, implement quantile loss and
  prediction intervals, reproduce every experiment.
- **[references.md](references.md)** — ESL Ch. 7, forecasting texts, the MAPE/sMAPE literature.

**Next**: [05.03 — Classification Metrics](../03-classification-metrics/) — where the threshold, class
imbalance, and the confusion matrix make "accuracy" one of the most misleading numbers in ML.
