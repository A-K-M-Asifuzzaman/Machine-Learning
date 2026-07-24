# 06.04 — Exercises: Gradient Boosting

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State gradient boosting as gradient descent in function space. What plays the role of the
parameter vector, what is the gradient, and why must a base learner be fit to it?

**D2.** For squared loss $L = \tfrac12(y-F)^2$, show the pseudo-residual is $y - F$ and the optimal
leaf value is the mean residual. Conclude that gradient boosting under squared loss is "fit the
residuals."

**D3.** For absolute loss $|y - F|$, show the pseudo-residual is $\mathrm{sign}(y - F)$ and the
optimal leaf value is the **median** residual. Explain why this is robust.

**D4.** Write the Huber loss and its gradient. Show it interpolates between squared and absolute, and
explain Friedman's rule of setting $\delta$ to a residual quantile each round.

**D5.** For binary log loss $L = \log(1 + e^{F}) - yF$ with $p = \sigma(F)$, derive
$r_i = y_i - p_i$. Then derive the one-Newton-step leaf value
$\gamma = \frac{\sum(y_i - p_i)}{\sum p_i(1 - p_i)}$ and identify the numerator and denominator as
gradient and Hessian.

**D6.** Derive the initial constant $F_0 = \arg\min_\gamma \sum_i L(y_i, \gamma)$ for squared,
absolute, and log loss. (Answers: mean, median, base-rate log-odds.)

**D7.** *(Quantile loss.)* Derive the pseudo-residual and optimal leaf value for the pinball loss
$L_\tau(y, F) = \tau(y-F)^+ + (1-\tau)(F-y)^+$. Explain how this yields quantile regression.

**D8.** Explain, from the "sum that descends the training loss" view, why $M$ (number of trees)
overfits gradient boosting but not a random forest. Make the argument precise.

**D9.** Argue that shrinkage $\nu$ behaves like an $L_1$ penalty on the tree coefficients. Why do
$\nu$ and $M$ trade off reciprocally?

**D10.** Show that boosting stumps ($J = 1$) yields a purely additive model (a GAM), and that a
depth-$J$ tree captures interactions of order up to $J$. Connect to
[03.05](../../03-supervised-learning/05-splines-and-gams/).

**D11.** *(AdaBoost as a special case.)* Show that gradient boosting with the exponential loss
$e^{-yF}$ recovers AdaBoost's reweighting and vote. Where does the closed form come from?

---

## Tier 2 — Implementation

**I1.** Implement a regression-tree base learner that reports a leaf id per input. Verify a
squared-loss booster on top of it matches `sklearn.ensemble.GradientBoostingRegressor`.

**I2.** Implement the negative-gradient method for squared, absolute, and Huber loss. Reproduce
Experiment 1: print each loss's pseudo-residual range and confirm which are bounded.

**I3.** Reproduce Experiment 2: hold $\nu M$ roughly constant and show test error falling then
plateauing as $\nu$ shrinks. Find where diminishing returns set in.

**I4.** Reproduce Experiment 3: corrupt 4% of training targets and show Huber/absolute beating
squared on a clean test set. Sweep the outlier fraction and find where squared loss breaks.

**I5.** Reproduce Experiment 4: plot test error vs $M$ for a booster and a forest on noisy data;
show the booster's turning up and the forest's flattening. Implement early stopping.

**I6.** Reproduce Experiment 5: sweep `subsample` and show stochastic boosting helping. Add
**column** subsampling and compare.

**I7.** Implement binary log-loss boosting with the Newton leaf value. Verify $r = y - p$ each round
and match sklearn's accuracy and log loss.

**I8.** Implement multiclass gradient boosting (softmax deviance, $K$ trees per round,
$r_{ik} = \mathbb{1}[y_i=k] - p_{ik}$). Verify against sklearn on a 3-class problem.

**I9.** *(Quantile regression.)* Implement pinball-loss boosting and produce a 10%/50%/90%
prediction interval. Check empirical coverage.

**I10.** Extract feature importances (summed split gain) and a partial-dependence plot from your
booster. Compare the importance ranking to a random forest's.

**I11.** *(The gradient/Hessian preview.)* Modify the leaf value to the second-order form
$-\frac{\sum g_i}{\sum h_i + \lambda}$ with an $L_2$ penalty $\lambda$, and show it is XGBoost's leaf
weight ([06.05](../05-modern-gbdts/)). Measure the effect of $\lambda$.

---

## Tier 3 — Interview

**Q1.** What is gradient boosting in one sentence?

**Q2.** "Gradient boosting fits the residuals." When is that true, and when is it not?

**Q3.** How do you turn a regression booster into a classifier?

**Q4.** Why is the learning rate a regularizer? How does it relate to the number of trees?

**Q5.** Can gradient boosting overfit as you add trees? How does that differ from a random forest?

**Q6.** How do you choose the number of trees?

**Q7.** Your targets have outliers. What do you change?

**Q8.** How deep should the trees be, and why?

**Q9.** What does `subsample < 1` do, and why does it help?

**Q10.** How is gradient boosting related to AdaBoost?

**Q11.** What is the leaf value for log loss, and why does it look like a Newton step?

**Q12.** Random forest or gradient boosting for a new tabular problem — how do you decide?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the pseudo-residual for any loss and the matching leaf-value line search
- [ ] Explain why "fit the residuals" is squared loss only
- [ ] State precisely why $M$ overfits boosting but not a forest, and use early stopping
- [ ] Explain shrinkage and the $\nu$–$M$ tradeoff without folklore
- [ ] Pick a robust loss for noisy targets and say why it is robust
- [ ] See the log-loss leaf value as the gradient/Hessian ratio that becomes XGBoost's objective
