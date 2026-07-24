# 06.05 — Exercises: XGBoost, LightGBM & CatBoost

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write XGBoost's regularized objective at round $t$ and identify each term of
$\Omega(f) = \gamma T + \tfrac12\lambda\sum_j w_j^2$.

**D2.** Take the second-order Taylor expansion of $L(y_i, F_{t-1} + f_t)$ and drop the constant.
Define $g_i$ and $h_i$.

**D3.** Group the quadratic objective by leaf and derive the optimal leaf weight
$w_j^\star = -G_j/(H_j + \lambda)$. Show the resulting structure score is
$-\tfrac12\sum_j G_j^2/(H_j+\lambda) + \gamma T$.

**D4.** Derive the split-gain formula
$\tfrac12[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}] - \gamma$.
Explain why a split is taken only when this is positive, and how $\gamma$ acts as pre-pruning.

**D5.** For squared loss show $g_i = F_i - y_i$, $h_i = 1$, and hence $w_j^\star$ is the (shrunken)
mean residual. Recover plain gradient boosting as $\lambda = \gamma = 0$.

**D6.** For log loss show $g_i = p_i - y_i$, $h_i = p_i(1-p_i)$, and that the leaf weight becomes the
gradient/Hessian ratio of [06.04 §6](../04-gradient-boosting/) with an $L_2$ term. Explain the sense
in which this is a Newton step.

**D7.** Explain `min_child_weight` as a floor on $H_j$. For log loss, what quantity does it bound,
and why is that "an effective number of well-classified points"?

**D8.** *(Histogram subtraction.)* Show that a parent's histogram equals the sum of its children's,
and explain how this halves histogram-building cost.

**D9.** Explain why leaf-wise growth reaches lower training loss than level-wise at equal leaf count,
and why it overfits more. Why tune `num_leaves`, not `max_depth`?

**D10.** *(Prediction shift.)* Explain why using $F_{t-1}$ — trained on point $i$ — to compute
point $i$'s gradient biases the training gradients, and how ordered boosting removes the bias.

**D11.** *(Target-statistic leakage.)* Show that naive mean encoding of a category seen once equals
that row's own label. Derive CatBoost's ordered (preceding-rows-only) target statistic and explain
why it is unbiased.

---

## Tier 2 — Implementation

**I1.** Implement the split-gain formula and the leaf weight $-G/(H+\lambda)$. Verify a second-order
booster on top matches `xgboost` (regression $R^2$ and classification accuracy) within tolerance.

**I2.** Reproduce Experiment 1: ablate the Hessian (set $h=1$) and compare test loss vs rounds
against the true second-order step. Confirm Newton converges in fewer rounds.

**I3.** Reproduce Experiment 2: sweep $\gamma$ and show average leaves per tree falling; sweep
$\lambda$ and show mean leaf-weight magnitude falling.

**I4.** Implement histogram split finding with quantile bins. Reproduce Experiment 3: match exact
accuracy while scanning far fewer candidate thresholds. Add the **subtraction trick**.

**I5.** Implement the sparsity-aware default direction: at each split, try sending missing values
left vs right and keep the higher-gain choice. Verify it beats mean-imputation on data with
informative missingness.

**I6.** Implement leaf-wise growth with a priority queue. Reproduce Experiment 4: lower training
loss at equal leaves, more overfitting.

**I7.** Reproduce Experiment 5: build a pure-noise high-cardinality categorical and show naive mean
encoding leaking (high train correlation, ~0 test) while ordered target statistics stay honest.

**I8.** *(Ordered boosting, simplified.)* Implement a one-permutation ordered booster: compute each
point's gradient from a model fit only on preceding points. Compare test error to standard boosting
on a small dataset.

**I9.** Implement column subsampling per tree and per split. Measure its effect on test accuracy and
tree correlation.

**I10.** Compare your from-scratch booster, `xgboost`, `lightgbm`, and `catboost` on one dataset with
matched `n_estimators`, `learning_rate`, and depth. Tabulate accuracy and training time.

**I11.** *(Oblivious trees.)* Constrain your tree to use the same (feature, threshold) across a whole
level (CatBoost-style). Measure the accuracy cost and the inference-speed benefit.

---

## Tier 3 — Interview

**Q1.** How is XGBoost different from Friedman's gradient boosting?

**Q2.** Derive XGBoost's leaf weight and split gain.

**Q3.** What do $\gamma$ and $\lambda$ do, and where do they enter the algorithm?

**Q4.** Why use the second derivative? Isn't the gradient enough?

**Q5.** What is histogram split finding and why is it fast?

**Q6.** LightGBM vs XGBoost — what actually differs?

**Q7.** What problem does CatBoost's ordered boosting solve?

**Q8.** How should categorical features be encoded, and what is the danger?

**Q9.** How does XGBoost handle missing values?

**Q10.** Which of the three do you reach for, and when?

**Q11.** Why do gradient-boosted trees still beat neural nets on tabular data?

**Q12.** You set `num_leaves` high in LightGBM and it overfits. Why, and what do you change?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the leaf weight and split gain from a second-order Taylor expansion
- [ ] Explain $\gamma$ and $\lambda$ as terms of $\Omega$ that enter the split criterion directly
- [ ] Explain histogram split finding and the subtraction trick
- [ ] State how leaf-wise growth differs from level-wise and how to control it
- [ ] Explain prediction shift and CatBoost's ordered fix (boosting and target stats)
- [ ] Say why GBDTs remain the tabular default over deep nets
