# 06.02 — Exercises: Random Forests

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Starting from the bagging variance formula
$\rho\sigma^{2}+\frac{1-\rho}{B}\sigma^{2}$ ([06.01 §2](../01-bagging/)), explain precisely how
feature subsampling changes each of $\rho$ and $\sigma^{2}$, and why the *product* $\rho\sigma^2$
has an interior optimum in `max_features`.

**D2.** Explain why the feature subset must be redrawn at **each split**, not once per tree, for
effective decorrelation.

**D3.** Derive Breiman's default `max_features`: why $\sqrt{d}$ for classification and $d/3$ for
regression?

**D4.** Explain why `n_estimators` cannot overfit a random forest, but *can* overfit gradient
boosting. What is structurally different?

**D5.** Show that a random forest's prediction is bounded by the range of training targets, and
hence that it cannot extrapolate.

**D6.** Explain why MDI feature importance is biased toward high-cardinality features, and why
averaging over trees reduces but does not eliminate the bias.

**D7.** Explain the two opposite failures of MDI and permutation importance on *correlated*
features. Which understates and which hides, and why?

**D8.** Define the forest proximity and show it is a valid (PSD) kernel. What does "close" mean
under it?

**D9.** Explain the connection between a random forest and adaptive nearest neighbours: in what
sense does a forest learn a metric?

**D10.** Compare Extra-Trees with random forests in terms of $\rho$, per-tree bias, and training
cost.

---

## Tier 2 — Implementation

**I1.** Extend a CART tree to draw a random `max_features` subset at each split. Verify a forest of
them beats plain bagging (all features) on correlated-feature data.

**I2.** Implement `RandomForestClassifier` and `RandomForestRegressor`. Verify test accuracy, OOB
score, and top-feature ranking against sklearn.

**I3.** Reproduce Experiment 1: measure tree correlation and ensemble error as `max_features`
falls, and show the floor coming down (then the trees becoming too weak).

**I4.** Reproduce Experiment 2's two regimes (many vs few informative features) and explain why the
optimal `max_features` differs.

**I5.** Reproduce Experiment 3: sweep the real signal's strength and find where the forest's MDI is
fooled by high-cardinality noise. Compare against a single tree's threshold from
[03.08](../../03-supervised-learning/08-decision-trees/).

**I6.** Reproduce Experiment 4: build correlated features and show MDI splits the importance while
permutation hides it. Then implement conditional permutation and show it recovers the group.

**I7.** Reproduce Experiment 5: confirm the forest flatlines beyond the training range.

**I8.** Implement forest proximities. Use them to (a) impute a missing value, (b) rank outliers,
and (c) feed a t-SNE for supervised-similarity visualization.

**I9.** Implement Extra-Trees (random thresholds). Compare training time, $\rho$, and accuracy
against your random forest.

**I10.** *(Vectorized traversal.)* Implement batch tree prediction by routing all query rows
through the tree with boolean masks, and measure the speedup over per-sample traversal.

**I11.** Verify empirically that OOB error tracks test error for a forest, and note the $B$ below
which it is unreliable.

---

## Tier 3 — Interview

**Q1.** What is a random forest, in one sentence, relative to bagging?

**Q2.** Why does feature subsampling help? Answer in terms of the variance floor.

**Q3.** How do you set `max_features`, and why is the default $\sqrt{d}$?

**Q4.** Can adding trees overfit a random forest?

**Q5.** How does a random forest differ from gradient boosting?

**Q6.** Your `feature_importances_` ranks a feature highly. Do you trust it?

**Q7.** Two of your features are correlated. What do the importance methods do?

**Q8.** Can a random forest extrapolate?

**Q9.** What is out-of-bag error and when is it unreliable?

**Q10.** What are forest proximities good for?

**Q11.** When would you pick a random forest over XGBoost?

**Q12.** What are Extra-Trees and when might they help?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the bagging→forest step as attacking $\rho$ in $\rho\sigma^2$
- [ ] Tune `max_features` from the bias-variance tradeoff, not folklore
- [ ] State why `n_estimators` cannot overfit (and why boosting differs)
- [ ] Distrust MDI importance correctly, and know permutation's separate failure
- [ ] Explain why a forest cannot extrapolate
- [ ] Place random forests and gradient boosting at opposite ends of the bias-variance strategy
