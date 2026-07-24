# 05.06 — Calibration

> **Prerequisites**: [05.03 §8](../03-classification-metrics/) (proper scoring rules — the metrics
> that see calibration), [03.04](../../03-supervised-learning/04-logistic-regression/) (log loss, the
> loss that produces calibrated probabilities), [05.04](../04-cross-validation/) (the held-out data a
> calibrator must be fit on).
> **You will be able to**: distinguish calibration from discrimination, measure it with reliability
> diagrams and ECE, know which models are miscalibrated and why, and recalibrate with Platt scaling,
> isotonic regression, or temperature scaling — without touching accuracy.

---

## Table of contents

1. [What calibration is](#1-what-calibration-is)
2. [Calibration vs discrimination](#2-calibration-vs-discrimination)
3. [Why calibration matters](#3-why-calibration-matters)
4. [Measuring calibration: reliability diagrams and ECE](#4-measuring-calibration-reliability-diagrams-and-ece)
5. [Which models are miscalibrated, and why](#5-which-models-are-miscalibrated-and-why)
6. [Platt scaling](#6-platt-scaling)
7. [Isotonic regression](#7-isotonic-regression)
8. [Temperature scaling](#8-temperature-scaling)
9. [Fit the calibrator on held-out data](#9-fit-the-calibrator-on-held-out-data)
10. [Calibration is not enough: sharpness](#10-calibration-is-not-enough-sharpness)
11. [Common misconceptions](#11-common-misconceptions)

---

## 1. What calibration is

A classifier is **calibrated** if its predicted probabilities mean what they say:

> Among all the cases where the model predicts $p = 0.7$, about **70%** should actually be positive.

Formally, calibration asks $\mathbb{P}(y = 1 \mid \hat p = p) = p$ for all $p$. A weather model that
says "70% chance of rain" is calibrated if it rains on 70% of such days. This is a property of the
*probabilities*, not of the *decisions*: a model can classify well and still lie about its confidence.
Most metrics so far — accuracy, precision, recall, AUC — never check it, which is why so many deployed
models output numbers that look like probabilities but are not.

---

## 2. Calibration vs discrimination

Two independent qualities of a probabilistic classifier:

- **Discrimination** — can it *rank* positives above negatives? Measured by **AUC**
  ([05.03 §6](../03-classification-metrics/)). A model with AUC 1.0 separates the classes perfectly.
- **Calibration** — are the predicted probabilities *truthful*? Measured by reliability / ECE (§4).

They are orthogonal. A model can have **perfect AUC and terrible calibration**: take any perfectly
discriminating model and squash all its scores into $[0.5, 0.6]$ — the ranking (and AUC) is untouched,
but now every "55%" prediction might correspond to a 90% true rate. Conversely, a model can be
perfectly calibrated and useless at discrimination (predict the base rate for everyone — §10).

The crucial consequence: **you cannot fix calibration by improving AUC, and improving AUC does not fix
calibration.** Recalibration (§6–§8) applies a *monotonic* transform to the scores, so it changes the
probabilities without changing the ranking — it improves calibration while leaving AUC (and, if the
threshold moves with it, accuracy) exactly intact. Experiment 1 shows a high-AUC model with large ECE,
and recalibration collapsing the ECE while AUC does not budge.

---

## 3. Why calibration matters

If you only ever take the top-1 prediction, calibration may not matter. It matters — often
critically — whenever the **probability itself drives a decision**:

- **Cost-sensitive thresholds.** The Bayes-optimal threshold $t^\star = c_{FP}/(c_{FP}+c_{FN})$
  ([05.03 §11](../03-classification-metrics/)) is correct *only for calibrated probabilities*. Feed it
  miscalibrated scores and the "optimal" threshold is wrong.
- **Expected value / risk.** Any decision that multiplies a probability by a payoff (expected revenue,
  expected loss, medical risk) needs the probability to be real. A credit model that says "8% default"
  had better mean 8%.
- **Combining models and evidence.** Stacking ([06.06](../../06-ensembles/06-stacking/)), Bayesian
  updating, and thresholding across sources all assume the inputs are true probabilities.
- **Communicating uncertainty.** "90% confident" must mean 90% to a human relying on it — in
  medicine, forecasting, or safety.

In all of these, a discriminating-but-miscalibrated model gives confidently wrong numbers, and the
downstream decision inherits the error. Calibration is what makes a probability *usable* as a
probability.

---

## 4. Measuring calibration: reliability diagrams and ECE

**Reliability diagram.** Bin the predictions (say into 10 bins by predicted probability). In each bin,
plot the *average predicted probability* (x) against the *observed positive frequency* (y). A
perfectly calibrated model lies on the diagonal $y = x$. Points **below** the diagonal mean the model
is **overconfident** (predicts higher than the truth); **above** means **underconfident**.

**Expected Calibration Error (ECE)** summarizes the diagram in one number — the average gap between
confidence and accuracy, weighted by bin population:

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\,\big|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\big|,
$$

where $\mathrm{conf}(b)$ is the bin's mean predicted probability and $\mathrm{acc}(b)$ its observed
frequency. **Maximum Calibration Error (MCE)** takes the worst bin instead of the average — relevant
when any miscalibration is dangerous. Both depend on the binning, so report the bin count.

The **Brier score** ([05.03 §8](../03-classification-metrics/)) also measures calibration, and
**Murphy's decomposition** splits it exactly: $\mathrm{Brier} = \mathrm{reliability} - \mathrm{resolution}
+ \mathrm{uncertainty}$, where *reliability* is a calibration term (lower better), *resolution* rewards
spreading predictions away from the base rate (discrimination), and *uncertainty* is the irreducible
base-rate variance. Experiment's code verifies the three terms sum to the Brier score.

---

## 5. Which models are miscalibrated, and why

Miscalibration is systematic and predictable from the training loss:

- **Logistic regression — calibrated by construction.** It minimizes log loss, a proper scoring rule
  ([05.03 §8](../03-classification-metrics/)), so at optimum its outputs *are* calibrated
  probabilities. This is why it is the natural calibrator (§6).
- **SVMs — pushed to the extremes.** The hinge loss cares only about the margin, not probabilities;
  its raw scores are not probabilities at all, and the sigmoid-ish squashing people apply is
  overconfident near the boundary. Needs calibration.
- **Naive Bayes — overconfident.** The independence assumption multiplies correlated evidence as if
  independent, driving probabilities toward 0 and 1. Good ranking, terrible calibration.
- **Random forests — underconfident.** Averaging many trees pulls predictions *toward the base rate*:
  a bagged vote rarely reaches 0 or 1 because some trees always dissent, so the forest is
  underconfident at the extremes (reliability curve above the diagonal near 0 and below near 1).
- **Boosted trees — overconfident.** Boosting drives the margin up ([06.03](../../06-ensembles/03-boosting-theory/)),
  pushing probabilities toward the extremes; usually overconfident.
- **Modern deep nets — badly overconfident.** High-capacity nets trained to near-zero loss become
  very overconfident (Guo et al., 2017) — a 99% softmax that is right only 90% of the time. Standard
  practice is to temperature-scale them (§8).

Experiment 2 measures ECE and draws reliability curves for logistic (near-calibrated), a random forest
(underconfident), and a boosted model (overconfident), reproducing these signatures.

---

## 6. Platt scaling

**Platt scaling** fits a **logistic regression on the model's scores**: learn $a, b$ such that

$$
\hat p_{\text{cal}} = \sigma(a\,s + b),
$$

where $s$ is the raw model score. It is a one-dimensional logistic regression (two parameters),
fit on a held-out calibration set (§9). Properties:

- **Parametric and low-variance** — only 2 parameters, so it works with **little calibration data**
  and rarely overfits.
- **Assumes a sigmoidal distortion** — it can only stretch/shift the scores through a sigmoid, so it
  corrects the common S-shaped miscalibration (SVMs, boosting) well but cannot fix arbitrary shapes.
- **Monotonic** — preserves ranking, so AUC is untouched (§2).

Platt scaling is the right default when calibration data is limited or the miscalibration is roughly
sigmoidal. Experiment 3 shows it stabilizing calibration where isotonic overfits on small data.

---

## 7. Isotonic regression

**Isotonic regression** fits a **non-parametric monotonic (non-decreasing) function** mapping scores
to calibrated probabilities — no functional form assumed, just monotonicity. It is solved exactly by
the **Pool Adjacent Violators (PAV)** algorithm: sort by score, then repeatedly average any adjacent
predictions that violate monotonicity until the sequence is non-decreasing (minimizing squared error).

- **Flexible** — fits *any* monotonic distortion, so it corrects miscalibration Platt scaling cannot.
- **Higher variance** — being non-parametric, it **overfits on small calibration sets** and produces a
  jagged step function; it needs more data (rule of thumb: hundreds to thousands of calibration
  points).
- **Monotonic** — again preserves ranking and AUC.

The choice is the usual bias-variance trade: **Platt (low variance) for small calibration data or
sigmoidal distortion; isotonic (low bias) for large data or oddly-shaped distortion.** Experiment 3
measures both crossing over as calibration-set size grows.

---

## 8. Temperature scaling

For neural networks, the miscalibration is almost entirely *overconfidence*, and a single parameter
fixes it. **Temperature scaling** divides the logits by a scalar $T > 0$ before the softmax:

$$
\hat p = \mathrm{softmax}(\mathbf{z} / T).
$$

- $T = 1$ leaves the model unchanged; $T > 1$ **softens** the probabilities (less confident); $T < 1$
  sharpens them. You fit the single $T$ by minimizing log loss on a held-out set.
- **It preserves accuracy *exactly*.** Dividing all logits by the same $T$ does not change which logit
  is largest, so the top-1 prediction — and hence accuracy and AUC — is identical. It only rescales
  confidence. This is why it is the standard recalibration for deep nets: free calibration, zero
  accuracy cost.
- **One parameter**, so it barely overfits and needs little data.

Temperature scaling is the special case of Platt scaling with $b = 0$ and a shared slope across
classes. Experiment 4 fits $T$ to an overconfident model and shows ECE dropping sharply while accuracy
stays bit-for-bit identical.

---

## 9. Fit the calibrator on held-out data

The calibrator (§6–§8) learns from data, so — exactly like any other learned step
([05.04 §7](../04-cross-validation/)) — it must be fit on data the base model **did not train on**.
Fit Platt/isotonic on the *training* predictions and you calibrate against scores the model has already
overfit, producing a calibrator that looks perfect in-sample and fails in deployment.

The correct protocol is a **held-out calibration set** (or cross-validated calibration, as
scikit-learn's `CalibratedClassifierCV` does): train the model on one split, fit the calibrator on the
predictions of a *disjoint* split, and evaluate on a third. This is the same use-only-held-out-data
discipline as cross-validation ([05.04](../04-cross-validation/)) and stacking's out-of-fold features
([06.06](../../06-ensembles/06-stacking/)). Experiment 5 shows calibrating on training predictions
giving a deceptively low in-sample ECE that balloons on test, while held-out calibration generalizes.

---

## 10. Calibration is not enough: sharpness

Calibration is necessary but not sufficient. The model that **predicts the base rate for every input**
is *perfectly calibrated* — if 30% of cases are positive, it always says 0.30, and among all its 0.30
predictions exactly 30% are positive — yet it is **useless**: it never discriminates. The missing
quality is **sharpness** (or resolution): a good model pushes its probabilities *away* from the base
rate toward 0 and 1 *while staying calibrated*.

The goal is therefore **"sharp, subject to calibration"** (Gneiting): make confident predictions that
are also truthful. This is why proper scoring rules (log loss, Brier) are the right training and
evaluation targets — they reward calibration *and* sharpness together, unlike ECE, which a
base-rate predictor games perfectly. Recalibration improves calibration without touching sharpness
(it is monotonic); it makes an already-discriminating model's probabilities honest, but it cannot
manufacture discrimination that was not there. Experiment 6 shows a base-rate predictor scoring ECE ≈ 0
yet a terrible Brier and AUC 0.5 — calibrated and worthless.

---

## 11. Common misconceptions

**"High accuracy / AUC means the probabilities are trustworthy."**
No — those measure discrimination, not calibration (§2). A model can rank perfectly and output wildly
miscalibrated probabilities. Only reliability/ECE and proper scoring rules check calibration.

**"Recalibration will improve my accuracy."**
Recalibration is monotonic, so it leaves the ranking — and AUC, and (for temperature scaling)
accuracy — unchanged (§2, §8). It fixes the *probabilities*, not the *decisions*.

**"Isotonic regression is strictly better than Platt scaling."**
Isotonic is more flexible but higher variance; it overfits on small calibration sets where Platt's two
parameters are more robust (§6–§7). Choose by data size and distortion shape.

**"I calibrated on the training set."**
That leaks (§9). Fit the calibrator on held-out predictions, or use cross-validated calibration.

**"Neural networks output probabilities."**
Modern nets are badly overconfident (§5); the softmax is a confidence, not a calibrated probability.
Temperature-scale them (§8).

**"A calibrated model is a good model."**
Not necessarily — the base-rate predictor is perfectly calibrated and useless (§10). You need
calibration *and* sharpness; evaluate with a proper scoring rule.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — reliability diagrams, ECE/MCE, **Platt scaling**,
  **isotonic regression via the PAV algorithm**, and **temperature scaling**, all in NumPy, verified
  against scikit-learn's `CalibratedClassifierCV` and `IsotonicRegression`. Six experiments: (1)
  discrimination vs calibration — recalibration collapses ECE while AUC is unchanged; (2) the
  miscalibration signatures of logistic / random forest / boosting; (3) Platt vs isotonic vs
  calibration-set size; (4) temperature scaling dropping ECE at zero accuracy cost; (5) calibrating on
  training data leaks; (6) the perfectly-calibrated-but-useless base-rate predictor (sharpness).
- **[exercises.md](exercises.md)** — derive the PAV algorithm and Murphy's decomposition, implement
  each calibrator, reproduce every experiment.
- **[references.md](references.md)** — Platt, Zadrozny-Elkan, Niculescu-Mizil-Caruana, Guo et al.

**This completes Part 5 — Evaluation & Model Selection.** From the bias-variance decomposition
([05.01](../01-bias-variance-and-theory/)) through metrics, cross-validation, tuning, and now
calibration, you can evaluate a model honestly end to end. **Next**:
[Part 4 — Unsupervised Learning](../../04-unsupervised-learning/), or
[Part 7 — Deep Learning](../../07-deep-learning/).
