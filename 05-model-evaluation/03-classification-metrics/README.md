# 05.03 — Classification Metrics

> **Prerequisites**: [05.02](../02-regression-metrics/) (the loss-vs-metric distinction),
> [03.04](../../03-supervised-learning/04-logistic-regression/) (probabilistic scores and the
> threshold), [00.03](../../00-mathematical-foundations/03-probability/) (conditional probability,
> Bayes).
> **You will be able to**: read a confusion matrix, know why accuracy lies under imbalance, choose
> between precision/recall/F1/AUC/AP/log-loss from the *cost of the two error types*, and interpret
> AUC as a ranking probability rather than an accuracy.

---

## Table of contents

1. [The confusion matrix — everything comes from four numbers](#1-the-confusion-matrix--everything-comes-from-four-numbers)
2. [Accuracy and why it lies under imbalance](#2-accuracy-and-why-it-lies-under-imbalance)
3. [Precision, recall, and the two error types](#3-precision-recall-and-the-two-error-types)
4. [F1 and Fβ](#4-f1-and-fβ)
5. [The threshold is a choice](#5-the-threshold-is-a-choice)
6. [ROC and AUC](#6-roc-and-auc)
7. [Precision-Recall and average precision](#7-precision-recall-and-average-precision)
8. [Proper scoring rules: log loss and Brier](#8-proper-scoring-rules-log-loss-and-brier)
9. [Balanced single numbers: MCC and Cohen's kappa](#9-balanced-single-numbers-mcc-and-cohens-kappa)
10. [Multiclass averaging](#10-multiclass-averaging)
11. [Choosing the operating point by cost](#11-choosing-the-operating-point-by-cost)
12. [Which metric, and common misconceptions](#12-which-metric-and-common-misconceptions)

---

## 1. The confusion matrix — everything comes from four numbers

Every metric for binary classification is a function of the same four counts, formed by crossing the
true label with the predicted label at a fixed threshold:

| | predicted positive | predicted negative |
|---|---|---|
| **actual positive** | TP (true positive) | FN (false negative) — a *miss* |
| **actual negative** | FP (false positive) — a *false alarm* | TN (true negative) |

The two kinds of mistake are **not interchangeable**: a false negative (miss a fraud, miss a tumor)
and a false positive (block a legitimate charge, alarm a healthy patient) usually carry very
different costs. Nearly every classification metric is a particular way of summarizing the balance
between FP and FN, and choosing a metric is choosing *which error you care about more*. Keep the
four cells in mind — precision, recall, specificity, F1, and the rest are all just ratios of them.

---

## 2. Accuracy and why it lies under imbalance

**Accuracy** is the fraction correct:

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.
$$

It is the default metric and, under class imbalance, one of the most misleading numbers in machine
learning. If 1% of transactions are fraud, the model that predicts "never fraud" for everyone scores
**99% accuracy** — while catching zero fraud. It has a perfect-looking headline and is completely
useless. This is the **accuracy paradox**: accuracy rewards a model for getting the majority class
right, which is trivial when one class dominates.

The lesson is not "accuracy is bad" but "accuracy is only meaningful when classes are roughly
balanced *and* the two error types cost the same." Under imbalance you need metrics that look at the
minority class specifically (recall, precision, AP) or that correct for chance (MCC, balanced
accuracy). Experiment 1 shows the 99%-accurate all-negative classifier scoring 0 on every metric
that matters.

---

## 3. Precision, recall, and the two error types

Two ratios split accuracy into the parts that matter under imbalance:

$$
\mathrm{Precision} = \frac{TP}{TP + FP}, \qquad \mathrm{Recall} = \frac{TP}{TP + FN}.
$$

- **Precision** — of everything the model *flagged*, what fraction was right? It penalizes **false
  alarms** (FP). High precision = "when it says positive, believe it." Matters when acting on a
  positive is expensive (a costly intervention, an accusation).
- **Recall** (= sensitivity = TPR) — of everything that was *actually positive*, what fraction did the
  model catch? It penalizes **misses** (FN). High recall = "it rarely lets a positive slip." Matters
  when missing a positive is dangerous (disease screening, fraud, safety).

Two more, for completeness:

- **Specificity** (= TNR) $= TN/(TN+FP)$ — recall for the negative class; the axis of the ROC curve.
- **Negative predictive value** $= TN/(TN+FN)$ — precision for the negative class.

Precision and recall **trade off**: lowering the decision threshold flags more positives, raising
recall but usually lowering precision, and vice versa (§5). You cannot read one without the other —
a model with 100% recall by predicting everything positive has precision equal to the base rate.
Reporting recall alone (or precision alone) is a classic way to hide a bad model.

---

## 4. F1 and Fβ

To collapse precision and recall into one number, use their **harmonic** mean, the **F1 score**:

$$
F_1 = \frac{2\,\mathrm{P}\cdot\mathrm{R}}{\mathrm{P} + \mathrm{R}}.
$$

Why harmonic and not arithmetic? The harmonic mean is **dominated by the smaller** of the two: if
precision is 1.0 but recall is 0.01, the arithmetic mean is a flattering 0.5 but $F_1 \approx 0.02$.
F1 refuses to reward a model that sacrifices one of the two entirely — you must be decent at *both*.

**Fβ** generalizes it to weight recall $\beta$ times as much as precision:

$$
F_\beta = (1 + \beta^2)\,\frac{\mathrm{P}\cdot\mathrm{R}}{\beta^2\,\mathrm{P} + \mathrm{R}}.
$$

$\beta > 1$ (e.g. $F_2$) favors recall — use it when misses are costlier than false alarms (disease
screening). $\beta < 1$ (e.g. $F_{0.5}$) favors precision — use it when false alarms are costlier
(spam filtering a critical inbox). $\beta = 1$ weights them equally. Note F1 **ignores TN entirely**,
which is exactly why it is useful under imbalance (the giant TN count cannot inflate it) — but also
why it is not symmetric between the classes.

---

## 5. The threshold is a choice

A probabilistic classifier ([03.04](../../03-supervised-learning/04-logistic-regression/)) outputs a
**score** $p \in [0,1]$; turning it into a decision requires a **threshold** $t$ (predict positive if
$p \ge t$). All the metrics above depend on $t$, and $t = 0.5$ is a *default*, not a law. Moving the
threshold slides you along the precision-recall tradeoff:

- **Lower $t$** → flag more → recall up, precision down, more false alarms.
- **Higher $t$** → flag fewer → precision up, recall down, more misses.

The right threshold depends on the **relative cost** of FP and FN (§11) — and on the base rate, which
is why a threshold tuned on balanced data is wrong on imbalanced data. This dependence motivates
**threshold-free** metrics that summarize a classifier across *all* thresholds at once (AUC, average
precision, §6–§7), separating the quality of the *scores* from the choice of *operating point*.
Experiment 2 sweeps the threshold and traces the precision-recall tradeoff.

---

## 6. ROC and AUC

The **ROC curve** plots the true-positive rate against the false-positive rate as the threshold sweeps
from 1 to 0:

$$
\mathrm{TPR} = \frac{TP}{TP+FN} = \mathrm{Recall}, \qquad \mathrm{FPR} = \frac{FP}{FP+TN} = 1 - \mathrm{Specificity}.
$$

Each threshold is a point; the curve traces the whole tradeoff. A perfect classifier hugs the
top-left; random guessing is the diagonal. **AUC** (area under the ROC curve) summarizes it in one
number, and it has a beautiful probabilistic meaning:

> **AUC is the probability that a randomly chosen positive is scored higher than a randomly chosen
> negative.** (The Wilcoxon-Mann-Whitney statistic.)

So AUC $= 1$ means the model ranks every positive above every negative; AUC $= 0.5$ is random; AUC
$< 0.5$ means it ranks *backwards*. Three properties make AUC popular:

- **Threshold-free** — it measures ranking quality, independent of where you set $t$.
- **Prevalence-independent** — it does not change if you resample the class balance, because TPR and
  FPR are each computed *within* a class.
- It answers "how well does the model *rank* positives above negatives?" — the right question when the
  operating point is not yet decided.

Experiment 3 verifies the ranking interpretation: AUC computed by the ROC integral equals AUC computed
by directly counting positive-negative pairs the model orders correctly. But AUC's prevalence-
independence is a double-edged sword (§7): under heavy imbalance it can look impressive while the model
is useless in practice.

---

## 7. Precision-Recall and average precision

Under **heavy imbalance**, ROC/AUC can badly mislead. The FPR denominator $FP + TN$ is dominated by
the huge TN count, so even thousands of false positives barely move the FPR — the ROC curve looks
great while the model floods you with false alarms. The **precision-recall (PR) curve** exposes this:
it plots precision against recall across thresholds, and precision *does* feel every false positive
(its denominator $TP+FP$ has no TN to hide behind).

**Average precision (AP)** summarizes the PR curve (roughly, the area under it):

$$
\mathrm{AP} = \sum_k (\mathrm{R}_k - \mathrm{R}_{k-1})\,\mathrm{P}_k.
$$

Key differences from ROC/AUC:

- **The PR baseline is the prevalence**, not 0.5. On 1%-positive data, random guessing gives
  AP $\approx 0.01$ — so an AP of 0.3 is genuinely good, whereas the same model might show AUC 0.9 and
  look near-perfect.
- **PR/AP is prevalence-*dependent*** — it reflects the actual difficulty of finding rare positives,
  which is usually what you care about under imbalance.

**Rule of thumb**: balanced classes or ranking quality matters → ROC/AUC. Rare positives and you care
about the minority class → PR/AP. Experiment 4 shows one model with AUC 0.9 but a mediocre AP on
imbalanced data — the same scores, two very different verdicts.

---

## 8. Proper scoring rules: log loss and Brier

Every metric so far judges *decisions* (or rankings). But a probabilistic classifier outputs
*probabilities*, and a good 0.9 is better than a lucky 0.51. **Proper scoring rules** evaluate the
probabilities themselves and are **minimized only by the true probabilities** — they reward honesty.

**Log loss** (binary cross-entropy):

$$
\mathrm{LogLoss} = -\frac1n\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big].
$$

It punishes confident wrong predictions *savagely* (a $p=0.99$ on a negative contributes $-\log(0.01)
\approx 4.6$) and unboundedly as $p \to 0$ or $1$. It is the training loss of logistic regression and
the natural metric when calibrated probabilities matter.

**Brier score** — mean squared error on the probabilities:

$$
\mathrm{Brier} = \frac1n\sum_i (p_i - y_i)^2.
$$

Bounded in $[0,1]$, gentler on confident mistakes than log loss, and it **decomposes** into
calibration + refinement (Murphy's decomposition, [05.06](../06-calibration/)). Both are *proper*:
you cannot game them by shading your probabilities away from your true belief. This is exactly what
accuracy and AUC miss — a model can rank perfectly (AUC 1.0) yet be badly **miscalibrated** (all its
"90%" predictions are really 60%), and only a proper scoring rule sees it. Experiment 6 shows log
loss separating two models that share the same accuracy and AUC. Calibration is [05.06](../06-calibration/)'s
whole subject.

---

## 9. Balanced single numbers: MCC and Cohen's kappa

Two metrics use all four confusion-matrix cells and correct for chance, giving a single trustworthy
number even under imbalance:

**Matthews correlation coefficient (MCC)** — the correlation between predicted and true labels:

$$
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in [-1, 1].
$$

MCC is high only when the model does well on **both** classes: $+1$ perfect, $0$ random, $-1$ inverted.
Unlike F1 it uses TN and is symmetric between the classes, which makes it arguably the **best single
number for imbalanced binary classification** — it cannot be fooled by the all-negative classifier
(which gets MCC = 0, correctly).

**Cohen's kappa** — agreement between prediction and truth, corrected for the agreement expected by
chance: $\kappa = (p_o - p_e)/(1 - p_e)$, where $p_o$ is observed accuracy and $p_e$ the chance
accuracy from the marginals. $\kappa = 1$ perfect, $0$ chance-level. Common for inter-rater agreement
and multiclass problems.

Both punish the accuracy-paradox model correctly (Experiment 1): where accuracy says 0.99, MCC and
kappa say ~0.

---

## 10. Multiclass averaging

With $K > 2$ classes, per-class precision/recall/F1 must be **averaged**, and the averaging scheme
changes the story under imbalance:

- **Macro** — unweighted mean of per-class scores. Every class counts equally, so **rare classes
  matter as much as common ones**. Use it when minority-class performance matters.
- **Micro** — pool all TP/FP/FN across classes, then compute once. Dominated by frequent classes; for
  single-label problems micro-F1 equals accuracy.
- **Weighted** — mean of per-class scores weighted by class frequency. A compromise; hides poor
  performance on rare classes less than micro but more than macro.

The gap between macro and micro is itself diagnostic: a large gap means the model does well on common
classes and poorly on rare ones. Report macro when you care about the tail.

---

## 11. Choosing the operating point by cost

If false positives and false negatives have known costs $c_{FP}$ and $c_{FN}$, the threshold is not a
matter of taste — it is an optimization. The expected cost at threshold $t$ is

$$
\mathbb{E}[\text{cost}](t) = c_{FP}\cdot FP(t) + c_{FN}\cdot FN(t),
$$

and for a *calibrated* classifier the cost-minimizing decision rule is the Bayes-optimal threshold

$$
t^\star = \frac{c_{FP}}{c_{FP} + c_{FN}}.
$$

If a miss costs 10× a false alarm, $t^\star = 1/11 \approx 0.09$ — flag aggressively. The default 0.5
is optimal *only* when the two errors cost the same and the classifier is calibrated. In practice you
sweep $t$ over the validation set, compute expected cost at each, and pick the minimizer — which is
almost never 0.5. Experiment 5 finds the cost-optimal threshold and shows it far from the default.

---

## 12. Which metric, and common misconceptions

**Decision guide:**

| Situation | Metric |
|---|---|
| Balanced classes, equal error cost | Accuracy (fine here) |
| Imbalanced, care about minority | **Precision/Recall, F1/Fβ, AP, MCC** |
| Ranking quality, operating point undecided | **AUC** |
| Rare positives, false alarms matter | **PR curve / average precision** |
| Probabilities must be trustworthy | **Log loss / Brier** (+ calibration, 05.06) |
| One trustworthy number under imbalance | **MCC** |
| Costs known | Expected cost at the optimal threshold (§11) |

**"99% accuracy means a great model."**
Not under imbalance — the all-negative classifier gets 99% on 1%-positive data and catches nothing
(§2). Check MCC/AP/recall.

**"AUC is the model's accuracy."**
No — AUC is the probability a random positive outranks a random negative (§6). It measures *ranking*,
is threshold-free, and can be 0.9 while the deployed model at $t=0.5$ is poor.

**"High AUC means the model is good under imbalance."**
AUC is prevalence-independent and can look great while precision is terrible because false positives
are drowned by the huge TN count. Use PR/AP for rare positives (§7).

**"Use 0.5 as the threshold."**
0.5 is optimal only for equal error costs and a calibrated classifier. Set the threshold from the cost
ratio (§11).

**"F1 is a good default single number."**
F1 ignores TN and is asymmetric between classes; MCC uses all four cells and is a better single number
under imbalance (§9). F1 is fine when the positive class is the focus.

**"Good AUC means good probabilities."**
Ranking and calibration are different — a model can rank perfectly and be badly miscalibrated. Only a
*proper scoring rule* (log loss, Brier) checks the probabilities (§8, [05.06](../06-calibration/)).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — every metric in NumPy (confusion matrix, precision/recall/
  F1/Fβ, specificity/NPV, ROC+AUC, PR+AP, log loss, Brier, MCC, Cohen's kappa, macro/micro/weighted),
  verified against `sklearn.metrics`. Six experiments: (1) the accuracy paradox; (2) the
  precision-recall threshold tradeoff; (3) AUC = the ranking probability, by direct pair counting;
  (4) ROC-looks-great-but-AP-is-mediocre under imbalance; (5) the cost-optimal threshold; (6) log loss
  separating two models with identical accuracy and AUC (a calibration preview).
- **[exercises.md](exercises.md)** — derive the AUC ranking identity and the Bayes threshold,
  implement ROC/PR/AP, reproduce every experiment.
- **[references.md](references.md)** — ESL, Fawcett's ROC primer, the imbalanced-learning literature.

**Next**: [05.04 — Cross-Validation](../04-cross-validation/) — how to *estimate* any of these metrics
without fooling yourself, and the leakage traps that make a great validation score a lie.
