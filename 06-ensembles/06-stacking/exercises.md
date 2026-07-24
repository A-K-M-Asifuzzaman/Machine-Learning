# 06.06 — Exercises: Stacking & Blending

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the stacking idea in one sentence, and say precisely how it differs from bagging and
boosting in *how the combination is chosen*.

**D2.** *(The leakage argument.)* Let a base model be a 1-nearest-neighbour classifier. Show its
in-sample training predictions equal the training labels exactly. Explain why feeding those to the
meta-learner makes it assign the 1-NN near-total weight, and why the stack then fails on test data.

**D3.** Define out-of-fold (OOF) predictions and prove that each OOF meta-feature for row $i$ is
produced by a model that never trained on row $i$. Why is this the only regime that matches test
time?

**D4.** Write the full stacking algorithm (Wolpert 1992): OOF meta-feature construction, meta-learner
fit, base-model refit, and prediction. Explain why step 4 refits base models on the full data.

**D5.** Compare stacking ($K$-fold OOF) and blending (single holdout) in terms of data used by the
meta-learner, base-model retraining cost, and variance of the meta-features.

**D6.** Argue why the meta-learner should be simple. Frame it as a bias-variance argument on a
regression with $M$ correlated, strong inputs and $n$ rows.

**D7.** Explain, using the $\rho\sigma^2$ variance floor of [06.01 §2](../01-bagging/), why stacking
near-duplicate base models gains almost nothing, and why heterogeneous base models gain the most.

**D8.** Show that a uniform average of base predictions is a special case of a linear stacker. Under
what conditions on base-model quality does a learned stacker strictly beat the average?

**D9.** Derive the Euclidean projection onto the probability simplex $\lbrace w \ge 0, \sum w = 1\rbrace$
used by a non-negative sum-to-one meta-learner.

**D10.** *(Connection to CatBoost.)* Explain how the OOF trick of §3 is the same idea as CatBoost's
ordered target statistics ([06.05 §8](../05-modern-gbdts/)): both avoid using a point's own label to
build the feature that predicts it.

---

## Tier 2 — Implementation

**I1.** Implement $K$-fold OOF meta-feature generation. For a classifier, use predicted probabilities
(one column per base model). Verify each row's meta-feature came from a fold that excluded it.

**I2.** Implement `StackingClassifier` and `StackingRegressor`. Verify against
`sklearn.ensemble.StackingClassifier`/`StackingRegressor` (accuracy / $R^2$).

**I3.** Reproduce Experiment 1: include a 1-NN base model; compare naive in-sample stacking against
OOF stacking, printing the meta-learner's weight on the 1-NN and the test accuracy of each.

**I4.** Reproduce Experiment 2: show a stack of diverse models beating both the best single model and
a stack of clones. Construct data where each base model has a genuine blind spot.

**I5.** Reproduce Experiment 3: compare a linear meta-learner against a GBDT meta-learner and show
the simple one generalizing better.

**I6.** Implement a non-negative, sum-to-one meta-learner via projected gradient descent on the
simplex. Reproduce Experiment 4: show learned weights beating a uniform average under uneven base
quality, and tying it under even quality.

**I7.** Implement blending (single holdout). Reproduce Experiment 5: compare to $K$-fold stacking on
data efficiency and accuracy.

**I8.** *(Feature-weighted stacking.)* Give the meta-learner the raw features **and** the base
predictions. Show where it helps (base models reliable in different regions) and where it overfits.

**I9.** *(Multi-layer stacking.)* Add a second meta-layer and measure the marginal gain. Confirm the
diminishing returns of [§9](README.md).

**I10.** *(Leakage audit.)* Deliberately fit a scaler / target encoder on all data before OOF, and
show the leakage creeping back in. Then fix it by putting preprocessing inside each fold.

**I11.** Compare a two-model stack against the plain average on 3 datasets. Report when stacking is
worth the complexity.

---

## Tier 3 — Interview

**Q1.** What is stacking, and how does it differ from bagging and boosting?

**Q2.** What is the single most important thing to get right in stacking?

**Q3.** Why can't you train base models and predict the training set to make meta-features?

**Q4.** What are out-of-fold predictions and why do you need them?

**Q5.** What meta-learner do you use, and why not a powerful one?

**Q6.** What makes a good set of base models?

**Q7.** When does stacking beat a simple average?

**Q8.** Stacking vs blending — what's the difference?

**Q9.** Why is stacking common in competitions but rare in production?

**Q10.** How is stacking's leakage fix related to CatBoost's ordered target statistics?

**Q11.** You stacked five models and it's worse than your best single model. What went wrong?

**Q12.** How many stacking layers would you use?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the leakage trap and why OOF meta-features fix it
- [ ] Write the full stacking algorithm including the base-model refit
- [ ] Choose a simple meta-learner and justify it
- [ ] Explain why diversity, not individual accuracy, drives stacking gains
- [ ] Say when a plain average is the better engineering choice
- [ ] Connect stacking's OOF trick to CatBoost's ordered statistics
