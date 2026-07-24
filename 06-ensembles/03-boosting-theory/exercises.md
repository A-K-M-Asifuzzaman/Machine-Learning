# 06.03 — Exercises: Boosting Theory & AdaBoost

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the weak-learnability question (Kearns & Valiant) and what Schapire proved. Why is the
answer surprising?

**D2.** *(The full AdaBoost derivation.)* Starting from forward stagewise minimization of
$\sum_i \exp(-y_i F(\mathbf{x}_i))$, derive: (a) that the sample weights are
$w_i^{(m)} = \exp(-y_iF_{m-1}(\mathbf{x}_i))$; (b) that the best $h_m$ minimizes weighted error;
(c) the vote $\alpha_m = \tfrac12\ln\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$; (d) the reweighting
$w_i \leftarrow w_i\exp(-\alpha_m y_ih_m(\mathbf{x}_i))$.

**D3.** Show $\alpha_m = 0$ when $\mathrm{err}_m = \tfrac12$ (chance) and $\alpha_m < 0$ when
$\mathrm{err}_m > \tfrac12$. Interpret the negative case.

**D4.** Derive the training-error bound
$\frac1n\sum_i\mathbb{1}[H\ne y]\le\prod_m 2\sqrt{\mathrm{err}_m(1-\mathrm{err}_m)}$, and show it
implies $\le\exp(-2\sum_m\gamma_m^2)$ with $\gamma_m = \tfrac12 - \mathrm{err}_m$.

**D5.** How many rounds does AdaBoost need to drive training error below $\epsilon$ if every weak
learner has edge $\gamma$? Derive the $O(\log(1/\epsilon)/\gamma^2)$ rate.

**D6.** Plot (or tabulate) the exponential, logistic, hinge, and 0/1 losses against the margin
$yF$. Show all four upper-bound 0/1, and explain why the exponential's steep left tail makes
AdaBoost aggressive and noise-fragile.

**D7.** Show that a mislabelled point's weight grows by a factor $\prod_m e^{\alpha_m}$, and explain
why this makes AdaBoost overfit noise.

**D8.** Define the normalized margin. Explain why the *minimum* margin (not the median) is the
quantity Schapire's bound depends on, and what Experiment 5 shows about the two.

**D9.** Show why AdaBoost cannot be parallelized across rounds, and contrast with bagging.

**D10.** *(SAMME.)* Derive the multiclass AdaBoost vote $\alpha_m = \ln\frac{1-\mathrm{err}_m}{\mathrm{err}_m} + \ln(K-1)$
and explain why the $\ln(K-1)$ term lets a learner beating $1/K$ (not $1/2$) contribute.

---

## Tier 2 — Implementation

**I1.** Implement a weighted decision stump. Verify it minimizes weighted error over all
(feature, threshold, polarity) triples.

**I2.** Implement AdaBoost (binary). Verify against `sklearn.ensemble.AdaBoostClassifier` with
`algorithm="SAMME"`.

**I3.** Reproduce Experiment 4: verify numerically that the AdaBoost sample weights equal the
exponential-loss weights $\exp(-y_iF_{m-1})$ at every round.

**I4.** Reproduce Experiment 2: measure training error against rounds and against the
$\prod 2\sqrt{\mathrm{err}(1-\mathrm{err})}$ bound. Confirm exponential decay.

**I5.** Reproduce Experiment 3: show AdaBoost's test error turning up with rounds on noisy data,
while a random forest's plateaus. This is the key contrast — make it stark.

**I6.** Reproduce Experiment 5: show the minimum margin climbing after zero training error, and
confirm the median does *not* (the subtlety).

**I7.** Implement LogitBoost (boosting with logistic loss). Compare its noise robustness against
AdaBoost on Experiment 3's noisy data.

**I8.** *(Why a stump?)* Boost trees of increasing depth. Show that deep base learners overfit
quickly, confirming the "weak on purpose" principle.

**I9.** Implement SAMME.R (real-valued AdaBoost using class probabilities). Compare convergence
speed against SAMME.

**I10.** Extract calibrated probabilities from a boosted model via
$p = 1/(1+e^{-2F})$ and check the calibration ([05.06](../../05-model-evaluation/06-calibration/)).

**I11.** *(The XOR trap.)* Show that AdaBoost with stumps fails on symmetric XOR
($y = \mathbb{1}[x_1x_2>0]$) because every stump is exactly at chance, and explain the connection to
a single tree's greedy failure ([03.08 §3](../../03-supervised-learning/08-decision-trees/)).

---

## Tier 3 — Interview

**Q1.** What does boosting do that bagging cannot, and vice versa?

**Q2.** Derive AdaBoost's vote weight from first principles.

**Q3.** Why does AdaBoost use weak learners on purpose?

**Q4.** What loss is AdaBoost minimizing, and how do you know?

**Q5.** Can AdaBoost overfit? How does this differ from a random forest?

**Q6.** Why is AdaBoost sensitive to noisy labels?

**Q7.** AdaBoost's test error kept dropping after training error hit zero. Explain.

**Q8.** Is `n_estimators` a regularizer for boosting? For bagging?

**Q9.** How does AdaBoost relate to gradient boosting?

**Q10.** Why can't boosting be parallelized across trees like a random forest?

**Q11.** You have noisy labels. Boosting or bagging?

**Q12.** What's the connection between AdaBoost and the SVM?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive every AdaBoost formula from exponential-loss minimization
- [ ] Prove the exponential training-error bound
- [ ] Explain the exponential loss's aggression *and* its noise fragility from its tail
- [ ] State precisely why boosting can overfit and bagging cannot
- [ ] Explain the margin story — and the honest limit of it (minimum, not median)
- [ ] See gradient boosting as "AdaBoost with the loss generalized"
