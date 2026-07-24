# 06.01 — Exercises: Bagging

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the variance of the average of $B$ **independent** estimators each with variance
$\sigma^{2}$, and show it is $\sigma^{2}/B$.

**D2.** Derive the correlated case:

$$\mathrm{Var}\!\left(\tfrac1B\textstyle\sum_b f_b\right) = \rho\sigma^{2} + \tfrac{1-\rho}{B}\sigma^{2}$$

Count the variance and covariance terms explicitly.

**D3.** Take the $B\to\infty$ limit and identify the floor. Explain why more trees cannot break it.

**D4.** Show that the fraction of distinct points in a bootstrap sample tends to $1-1/e\approx0.632$.
Derive the out-of-bag fraction.

**D5.** Explain, using D2, why decorrelating the trees is the *only* way to improve a bagged
ensemble past a few hundred trees. Connect this to the random forest.

**D6.** Show bagging leaves bias unchanged. *Hint*: consider $\mathbb{E}[\tfrac1B\sum_b f_b]$.

**D7.** Explain why the overlap between two bootstrap samples (~37% shared) is the source of the
tree correlation $\rho$.

**D8.** Derive why OOB error is a nearly-unbiased estimate of test error. What assumption about the
number of trees does it require?

**D9.** Explain why bagging linear regression achieves almost nothing, in terms of $\rho$ and the
variance floor.

**D10.** For classification, show that soft voting (averaging probabilities) is at least as
informative as hard voting, and give a case where they disagree.

---

## Tier 2 — Implementation

**I1.** Implement `bootstrap_sample` returning in-bag and OOB indices. Verify the ~63.2% coverage
empirically.

**I2.** Implement a generic bagging ensemble that accepts any base learner. Verify against
`sklearn.ensemble.BaggingRegressor`.

**I3.** Implement OOB scoring. Reproduce Experiment 2 and show the OOB estimate converges to test
error as $B$ grows.

**I4.** Reproduce Experiment 1: measure the bagged variance across independent datasets and show it
flattens at $\rho\sigma^2$, matching the predicted curve. (Reuse a tree pool per dataset for
efficiency.)

**I5.** Reproduce Experiment 3: show bagging transforms a deep tree and does essentially nothing
for linear regression. Explain the difference in one sentence.

**I6.** Reproduce Experiment 4: sweep `max_features` and show tree correlation dropping and ensemble
accuracy improving, up to the point where trees become too weak.

**I7.** Implement both soft and hard voting. Measure the accuracy difference on a problem with
uncertain trees.

**I8.** *(Learning curve of $B$.)* Plot test error against $B$ for a bagged tree ensemble and
confirm it decreases monotonically to a floor — never increasing. Contrast with what you would
expect from boosting.

**I9.** *(Base learner depth.)* Bag trees of increasing depth and show that deeper (higher-variance)
base learners benefit *more* from bagging, confirming §8's advice not to prune.

**I10.** Implement pasting (sampling without replacement) and compare against bootstrapping. When
does each help?

---

## Tier 3 — Interview

**Q1.** What is bagging and what problem does it solve?

**Q2.** Derive how much variance averaging removes.

**Q3.** Why doesn't the variance go to zero as you add trees?

**Q4.** Does bagging reduce bias, variance, or both?

**Q5.** Can adding more trees to a bagged model overfit?

**Q6.** What is out-of-bag error, and why is it useful?

**Q7.** Would you bag a linear regression? Why or why not?

**Q8.** Should you prune the trees inside a bagged ensemble?

**Q9.** What is the difference between bagging and a random forest?

**Q10.** Bagging vs boosting — what does each reduce, and can each be parallelized?

**Q11.** Soft voting or hard voting, and why?

**Q12.** Why is the ~63% bootstrap coverage relevant to the correlation between trees?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ formula and read both terms
- [ ] Explain why the floor is $\rho\sigma^2$ and how a random forest lowers it
- [ ] Explain why bagging helps trees but not linear regression
- [ ] Use OOB error instead of a validation set, and know when it is unreliable
- [ ] State that bagging only reduces variance, and design the base learner accordingly
- [ ] Explain the bagging→random-forest→boosting progression in terms of bias and variance
