# 05.05 — Exercises: Hyperparameter Optimization

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Distinguish parameters from hyperparameters. Why can't you fit hyperparameters by minimizing
training loss? What objective do you minimize instead?

**D2.** Show that grid search over $d$ hyperparameters at $v$ values each costs $v^d$ evaluations, and
explains why each individual hyperparameter is explored at only $v$ distinct values regardless of the
total budget.

**D3.** State the Bergstra-Bengio argument for random search: given low effective dimensionality, why
does random search explore the important dimensions at higher resolution than grid at equal budget?

**D4.** Write the Gaussian-process posterior mean and variance for regression with an RBF kernel,
derived by conditioning the joint Gaussian of observed and query points.

**D5.** Derive the closed form of Expected Improvement under a Gaussian posterior,
$\mathrm{EI} = (f^\star - \mu)\,\Phi(z) + \sigma\,\phi(z)$ with $z = (f^\star-\mu)/\sigma$. Show it
combines exploitation and exploration.

**D6.** Compare EI, UCB/LCB, and PI as acquisition functions. Why is PI too greedy? What does $\kappa$
control in UCB?

**D7.** Analyze successive halving: with $N$ initial configs and geometric budgets, show the total
cost and compare to $N$ full evaluations. What assumption about early vs final performance does it
rely on?

**D8.** Explain what Hyperband adds over a single successive-halving run, and why hedging the
budget-vs-count tradeoff is necessary.

**D9.** Show why a learning rate should be sampled log-uniformly: what fraction of a linear-uniform
sample over $[10^{-4}, 10^{-1}]$ lands below $10^{-2}$?

**D10.** Explain, as an extreme-value argument, why the maximum of $N$ noisy CV estimates is biased
above the true value, and why the bias grows with $N$.

---

## Tier 2 — Implementation

**I1.** Implement grid and random search with $K$-fold CV. Verify grid finds the same optimum as
`sklearn.model_selection.GridSearchCV` on an SVM with matched folds.

**I2.** Implement a Gaussian process (RBF kernel, exact posterior via Cholesky). Verify it
interpolates its training points and reports growing uncertainty away from them.

**I3.** Implement Expected Improvement and a Bayesian optimization loop. Reproduce Experiment 2: show
it beating random search on a bumpy 2D objective at equal budget.

**I4.** Reproduce Experiment 1: an objective where few dimensions matter, and show random beating grid
at equal budget. Vary the number of active dimensions and see the gap shrink.

**I5.** Implement successive halving. Reproduce Experiment 3: near-best config for a fraction of the
full cost. Then implement Hyperband's outer bracket loop.

**I6.** Reproduce Experiment 4: log vs linear sampling of a learning rate; measure how often each
finds the optimum.

**I7.** Reproduce Experiment 5: show the winner's CV optimism growing with the number of configs, and
then show nested CV removing it.

**I8.** Implement a simple TPE: model $p(\theta\mid \text{good})$ and $p(\theta\mid\text{bad})$ with
kernel density estimates and sample where their ratio is high. Compare to your GP optimizer.

**I9.** Add UCB and PI acquisition functions and compare their exploration behavior against EI on the
same objective.

**I10.** Tune a real GBDT end to end: log-scale learning rate + early stopping, then depth and
subsample, with random search. Report the final score with nested CV.

---

## Tier 3 — Interview

**Q1.** Grid vs random search — which and why?

**Q2.** Why does random search beat grid?

**Q3.** When would you use Bayesian optimization?

**Q4.** What is an acquisition function and what does it balance?

**Q5.** Explain Expected Improvement.

**Q6.** What is successive halving / Hyperband?

**Q7.** How should you sample a learning rate?

**Q8.** Your tuned CV score is 92%. Is that your deployment estimate?

**Q9.** When do you stop tuning?

**Q10.** Which hyperparameters would you tune first for XGBoost? For an SVM?

**Q11.** How do you tune when each training run takes a day?

**Q12.** What is the risk of trying thousands of configurations?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain and reproduce why random beats grid at equal budget
- [ ] Implement a GP surrogate and Expected Improvement
- [ ] Choose grid / random / Bayesian / Hyperband from the problem shape
- [ ] Design a search space with log scales and sensible ranges
- [ ] Explain over-tuning and estimate honest performance with nested CV
- [ ] Prioritize the few hyperparameters that actually matter
