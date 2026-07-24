# 03.08 — Exercises: Decision Trees

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that Gini impurity equals the probability that two independent draws from a node have
different labels. Then show it equals $\sum_{k\ne k'}p_kp_{k'}$.

**D2.** Show that information gain (entropy reduction) at a split equals the mutual information
$I(Y; X_j > t)$ between the label and the binary split indicator
([00.05 §9](../../00-mathematical-foundations/05-information-theory/)).

**D3.** Prove Gini is the first-order Taylor expansion of $\tfrac12 H$ around the uniform
distribution $p_k = 1/K$.

**D4.** Show that for regression, minimizing the weighted child variance is equivalent to
minimizing the total within-node sum of squared errors, and that the leaf mean is the
squared-error-optimal prediction.

**D5.** Show that only midpoints between adjacent distinct feature values are candidate
thresholds. How many are there for $n$ points?

**D6.** Show how to update the class counts (and hence Gini) in $O(1)$ as the split boundary moves
one point. Conclude the per-feature cost is $O(n\log n)$.

**D7.** Derive the incremental variance update for regression using running sums of $y$ and
$y^{2}$. What numerical caveat applies ([00.06 §10](../../00-mathematical-foundations/06-numerical-methods/)),
and why is it mild here?

**D8.** Explain why a categorical feature with $q$ levels has $2^{q-1}-1$ binary partitions. Then
explain Breiman's sorting trick for two-class problems and why it is optimal.

**D9.** Explain why information gain is biased toward high-cardinality features. Derive C4.5's gain
ratio and show how it corrects the bias.

**D10.** Explain why one-hot encoding a high-cardinality categorical hurts a tree, in terms of how
the signal is distributed across the encoded columns.

**D11.** Show that a fully grown tree achieves zero training error (with distinct points), and
explain why this is memorization rather than learning.

**D12.** Explain tree instability: why does changing one training point potentially reorganize the
entire tree? Connect it to the greedy recursion.

**D13.** Write the cost-complexity objective $R_\alpha(T)=R(T)+\alpha|T|$ and explain the analogy
to ridge's $\lambda$. Describe how the pruning path is generated as $\alpha$ increases.

**D14.** Explain why greedy tree induction fails on XOR. What is the information gain of the first
split, and why?

**D15.** Show that a regression tree cannot extrapolate: its predictions are bounded by the range
of training targets.

---

## Tier 2 — Implementation

**I1.** Implement `gini`, `entropy`, `variance`. Verify the boundary values (pure = 0, uniform =
max).

**I2.** Implement the naive $O(n^{2})$ split scan and the incremental $O(n\log n)$ one. Verify
they find identical splits and measure the speedup as $n$ grows.

**I3.** Implement a CART classifier. Verify its tree structure matches `sklearn` exactly (mind the
float32 threshold storage) for several depths and both criteria.

**I4.** Implement a CART regressor with the incremental variance scan. Verify predictions match
sklearn, and account for any structural differences as tie-breaks.

**I5.** Reproduce Experiment 2: verify Gini is entropy's Taylor approximation, and measure the
fraction of splits on which the two criteria agree.

**I6.** Reproduce Experiment 3: bootstrap-resample a dataset, retrain, and measure how often the
root split changes. Then average the trees' predictions and show the variance drops (a preview of
bagging).

**I7.** Reproduce Experiment 4 on XOR. Find the minimum depth at which the tree solves it, and
compare against a model that sees the interaction directly.

**I8.** Implement cost-complexity pruning. Reproduce the pruning path of Experiment 5 and verify
`ccp_alpha` matches sklearn's `cost_complexity_pruning_path`.

**I9.** Implement both MDI and permutation importance. Reproduce Experiment 6: construct data where
a high-cardinality noise feature outranks a real one under MDI but not permutation.

**I10.** Implement surrogate splits for missing values. On data with missingness, compare against
dropping rows and against mean imputation.

**I11.** *(Learned default direction.)* Implement the XGBoost-style approach: for each split, try
sending missing values left and right, keep the better. Show it beats imputation when missingness
is informative.

**I12.** Build a tree with native categorical support using Breiman's sorting trick, and compare
against one-hot + numeric splitting on a high-cardinality categorical.

**I13.** *(The staircase.)* Fit a tree to data with a diagonal decision boundary. Visualize the
axis-aligned approximation and measure how depth trades off against the staircase error.

---

## Tier 3 — Interview

**Q1.** How does a decision tree make a prediction?

**Q2.** Do trees need feature scaling? Why or why not?

**Q3.** What is the difference between Gini and entropy? Which should you use?

**Q4.** What does a split actually optimize? Connect it to information theory.

**Q5.** Why are trees built greedily rather than optimally?

**Q6.** Give an example where greedy tree induction fails.

**Q7.** Why does a single tree overfit? Name two independent mechanisms.

**Q8.** What is tree instability, and why does it matter for ensembles?

**Q9.** What is cost-complexity pruning, and what is it analogous to?

**Q10.** Your `feature_importances_` says feature X is most important. Should you trust it?

**Q11.** How do trees handle missing values?

**Q12.** Why does one-hot encoding hurt tree performance?

**Q13.** Can a regression tree predict a value outside the training range?

**Q14.** Why is a single tree rarely used in practice?

**Q15.** What is the relationship between a decision tree and a random forest / gradient boosting?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the splitting criteria and connect information gain to mutual information
- [ ] Explain the incremental split scan and why it makes trees fast
- [ ] Explain why greedy is suboptimal *and* why we use it anyway
- [ ] Name the two mechanisms of tree overfitting and how each is controlled
- [ ] Explain tree instability and why it is the raw material for ensembles
- [ ] Explain why MDI importance lies and what to use instead
- [ ] State the single-tree weaknesses that [Part 6](../../06-ensembles/) exists to fix
