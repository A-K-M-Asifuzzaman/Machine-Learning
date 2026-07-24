# 05.05 — Hyperparameter Optimization

> **Prerequisites**: [05.04](../04-cross-validation/) (the CV score being optimized, and nested CV
> for honest estimates), [05.01](../01-bias-variance-and-theory/) (hyperparameters are the
> bias-variance knobs), [00.03](../../00-mathematical-foundations/03-probability/) (Gaussian
> conditioning, for the GP surrogate).
> **You will be able to**: choose grid vs random vs Bayesian search from the shape of the problem,
> explain why random search beats grid when few dimensions matter, implement a Gaussian-process
> Bayesian optimizer and successive halving from scratch, and avoid over-tuning to CV noise.

---

## Table of contents

1. [Parameters vs hyperparameters, and the search problem](#1-parameters-vs-hyperparameters-and-the-search-problem)
2. [Grid search and the curse of dimensionality](#2-grid-search-and-the-curse-of-dimensionality)
3. [Random search — why it beats grid](#3-random-search--why-it-beats-grid)
4. [Bayesian optimization](#4-bayesian-optimization)
5. [The acquisition function](#5-the-acquisition-function)
6. [Successive halving and Hyperband](#6-successive-halving-and-hyperband)
7. [Designing the search space](#7-designing-the-search-space)
8. [Over-tuning: the validation set fights back](#8-over-tuning-the-validation-set-fights-back)
9. [A practical recipe](#9-a-practical-recipe)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Parameters vs hyperparameters, and the search problem

**Parameters** are learned by fitting (the weights of a linear model, the splits of a tree).
**Hyperparameters** are set *before* fitting and control the fitting itself — the regularization
strength $\lambda$, the tree depth, the learning rate, the number of neighbors. They are almost
always the bias-variance knobs of [05.01](../01-bias-variance-and-theory/): tuning them is choosing
where on the U-curve to sit.

We cannot learn hyperparameters by minimizing training loss (that would just pick zero regularization
and maximal capacity — overfitting). Instead we minimize a **held-out** estimate, the cross-validation
score ([05.04](../04-cross-validation/)):

$$
\boldsymbol\theta^\star = \arg\min_{\boldsymbol\theta \in \Theta}\ \mathrm{CV}(\boldsymbol\theta).
$$

This is a black-box optimization: $\mathrm{CV}(\boldsymbol\theta)$ has no gradient we can use, each
evaluation is **expensive** (it trains $K$ models), and it is **noisy** (the CV estimate has
variance). The whole subject is how to find a good $\boldsymbol\theta^\star$ in as few expensive,
noisy evaluations as possible — and how to keep the reported score honest afterwards (§8).

---

## 2. Grid search and the curse of dimensionality

**Grid search** picks a finite set of values per hyperparameter and tries every combination. Simple,
reproducible, embarrassingly parallel — and it explodes. With $d$ hyperparameters and $v$ values
each, the grid has $v^d$ points: 5 hyperparameters at 5 values each is $5^5 = 3125$ model fits, each
itself a $K$-fold CV. Add one more hyperparameter and you multiply the cost by $v$.

Worse, grid search **wastes evaluations on dimensions that do not matter.** If only 2 of your 5
hyperparameters actually affect performance, the grid still spends its entire budget varying all 5 in
lockstep — it evaluates the 2 important ones at only $v$ distinct values each, no matter how large the
total budget, because the other 3 dimensions eat the rest. This is the specific weakness random search
exploits (§3). Grid search is fine for 1–2 hyperparameters with known good ranges; beyond that it is
the wrong tool.

---

## 3. Random search — why it beats grid

**Random search** samples each hyperparameter independently from a distribution and evaluates $N$
random points. For the same budget $N$, it reliably **beats grid search** — a result that surprises
people until they see the reason (Bergstra & Bengio, 2012):

> **The key insight**: performance usually depends strongly on a *few* hyperparameters and weakly on
> the rest (low *effective dimensionality*). Grid search evaluates each important hyperparameter at
> only $v$ distinct values (the grid resolution). Random search with $N$ points evaluates each
> hyperparameter at $N$ *distinct* values — because every sample has a fresh value in every
> dimension. So on the dimension that matters, random search explores $N$ settings while grid
> explores only $v \ll N$.

Picture a 2D space where only the horizontal axis matters. A $3\times3$ grid tries 3 horizontal
values (with 3 redundant copies of each). Nine random points try ~9 distinct horizontal values. The
random search has 3× the resolution on the axis that counts, for the same 9 evaluations. Experiment 1
makes this concrete: on an objective where one dimension dominates, random search finds a markedly
better optimum than grid at equal budget. **Random search should be your default over grid** for
3+ hyperparameters.

---

## 4. Bayesian optimization

Random search is *non-adaptive* — it ignores everything it has already learned. **Bayesian
optimization** is adaptive: it builds a probabilistic model of the objective from the points seen so
far and uses it to choose the next point intelligently, spending evaluations where they are most
likely to help. It is the method of choice when each evaluation is very expensive (training a large
model) and the budget is small (tens of evaluations).

The loop:

1. **Surrogate model** — fit a cheap probabilistic model of $\mathrm{CV}(\boldsymbol\theta)$ over the
   points evaluated so far, giving a predicted mean $\mu(\boldsymbol\theta)$ and uncertainty
   $\sigma(\boldsymbol\theta)$ everywhere. The classic surrogate is a **Gaussian process** (§4 of
   [00.03](../../00-mathematical-foundations/03-probability/)): it interpolates the observations and
   reports high uncertainty far from them.
2. **Acquisition function** — score every candidate $\boldsymbol\theta$ by how *promising* it is,
   balancing exploiting low predicted mean against exploring high uncertainty (§5).
3. **Evaluate** the acquisition's maximizer (the single most promising point), add the result to the
   data, and repeat.

Because the surrogate is cheap, we can search it densely to pick each expensive real evaluation.
`from_scratch.py` implements a Gaussian process with an RBF kernel — the exact posterior mean and
variance from conditioning a multivariate Gaussian — and drives the loop with Expected Improvement.
Experiment 2 shows it reaching a good optimum in far fewer evaluations than random search on the same
objective.

A cheaper, tree-structured alternative — **TPE** (Tree-structured Parzen Estimator, the default in
Hyperopt and Optuna) — models $p(\boldsymbol\theta \mid \text{good})$ and
$p(\boldsymbol\theta \mid \text{bad})$ separately and samples where their ratio is high. It scales to
higher dimensions and conditional spaces better than a GP.

---

## 5. The acquisition function

The acquisition function turns the surrogate's mean and uncertainty into a single "evaluate here next"
score, and it is where the **exploration-exploitation** tradeoff lives. Let $f^\star$ be the best
value seen so far (minimizing). Three standard choices:

- **Expected Improvement (EI)** — the expected amount by which a point will improve on $f^\star$:
  $\mathrm{EI}(\boldsymbol\theta) = \mathbb{E}\big[\max(f^\star - f(\boldsymbol\theta),\, 0)\big]$.
  Under the GP's Gaussian posterior this has a closed form in terms of the standard normal CDF and
  PDF. EI automatically balances the two forces: it is large where the mean is low (exploitation) *or*
  where the uncertainty is high near promising regions (exploration). The default choice.
- **Upper/Lower Confidence Bound (UCB/LCB)** — $\mu(\boldsymbol\theta) - \kappa\,\sigma(\boldsymbol\theta)$
  (minimizing). The constant $\kappa$ tunes exploration explicitly: large $\kappa$ chases uncertainty,
  small $\kappa$ chases the mean.
- **Probability of Improvement (PI)** — the probability of beating $f^\star$ at all; simple but too
  greedy (it prefers small certain gains over large uncertain ones), so EI is usually preferred.

The reason Bayesian optimization is sample-efficient is entirely in this step: instead of sampling
blindly (random) or exhaustively (grid), it computes *where the expected payoff of an expensive
evaluation is highest* and spends its budget there. Experiment 2 visualizes EI concentrating
evaluations around the optimum after a few initial random probes.

---

## 6. Successive halving and Hyperband

A different idea: most bad configurations reveal themselves *early* — after a few training epochs, or
on a small data subset — so do not waste a full training budget on them. **Successive halving** turns
tuning into a bandit problem:

1. Start with $N$ configurations and a small budget $b$ each (few epochs / small subsample).
2. Evaluate all $N$; **keep the top fraction** (say the best half), throw away the rest.
3. **Double the budget** for the survivors and repeat, until one configuration remains on the full
   budget.

The total cost is roughly $N$ small evaluations + $N/2$ medium + ... — far less than $N$ full
evaluations, so you can afford to *start* with many more configurations. It reallocates budget from
clear losers to promising survivors. Experiment 3 shows successive halving finding a near-best
configuration for a fraction of the cost of evaluating them all fully.

**Hyperband** wraps successive halving in an outer loop that hedges the budget-vs-count tradeoff (how
aggressively to cut): it runs several "brackets" from "many configs, tiny budgets, aggressive cutting"
to "few configs, full budgets, no cutting," so it is robust to whether early performance actually
predicts final performance. **BOHB** combines Hyperband's early stopping with Bayesian optimization's
smart proposals — a strong modern default.

---

## 7. Designing the search space

The search *space* matters as much as the search *method*, and a bad space wastes any budget:

- **Log scale for multiplicative hyperparameters.** Learning rate, regularization $\lambda$, and
  such span orders of magnitude, and their effect is roughly log-linear. Sample them
  **log-uniformly** (uniform in $\log\theta$), so $10^{-4}$ and $10^{-1}$ get equal attention. Sample
  a learning rate uniformly in $[10^{-4}, 10^{-1}]$ and 90% of your samples land above $10^{-2}$,
  starving the small values that often matter. Experiment 4 shows log-sampling finding the optimum
  far more reliably.
- **Sensible ranges.** Too narrow and you miss the optimum; too wide and you waste budget. Use domain
  knowledge and widen only if the best value sits at a boundary.
- **Conditional spaces.** Some hyperparameters exist only given others (`degree` matters only for a
  polynomial kernel). TPE and Optuna handle these natively; grid/GP need care.
- **The right scale for integers and categoricals.** Number of trees, layers, units — often better on
  a log-ish scale too; categoricals (optimizer, kernel) are one-hot to the surrogate.

---

## 8. Over-tuning: the validation set fights back

Every evaluation reads the CV score, and the CV score is **noisy** ([05.04](../04-cross-validation/)).
Try enough configurations and one will look good *by luck* — you are, in effect, overfitting the
model-selection process to the CV noise. The reported CV score of the winner is therefore
**optimistically biased**, and the bias grows with the number of configurations tried (it is the
maximum of many noisy estimates).

Two consequences:

- **The winner's CV score is not its true performance.** Estimate performance with **nested CV** or a
  **locked test set** the tuning never saw ([05.04 §8](../04-cross-validation/)). Experiment 5
  measures the optimism climbing as you try more configurations.
- **There is a noise floor.** Once configurations are within a CV standard error of each other,
  further search is chasing noise, not signal. Stop, or apply the one-standard-error rule and take the
  simplest configuration in the top band ([05.04 §6](../04-cross-validation/)).

This is the deep reason "more tuning is always better" is false: past a point you are optimizing the
sampling error of your validation estimate, and the gains evaporate on truly unseen data.

---

## 9. A practical recipe

A pragmatic workflow that respects the above:

1. **Start coarse and broad.** Random search (log scale where appropriate) over wide ranges, modest
   budget — find the promising region cheaply.
2. **Refine.** Narrow the ranges around the best region; go finer with random or Bayesian search.
3. **Use early stopping** (successive halving / Hyperband) if training is iterative — it multiplies
   how many configurations you can afford.
4. **Reach for Bayesian optimization** when each evaluation is very expensive and the budget is small.
5. **Do not over-tune.** Stop when configurations are within a CV standard error; the last 0.5% is
   usually noise (§8).
6. **Report honestly.** Estimate final performance with nested CV or a held-out test set, never the
   tuning score (§8, [05.04 §8](../04-cross-validation/)).
7. **Prioritize the few knobs that matter.** For GBDTs: learning rate + number of trees (early
   stopping), then depth, then subsampling. For SVMs: $C$ and $\gamma$. Effort on the important
   dimensions beats a bigger grid on all of them (§3).

The single highest-leverage move is usually not a fancier optimizer but a better search *space* (§7)
and spending the budget on the *right* hyperparameters (§3).

---

## 10. Common misconceptions

**"Grid search is the thorough, correct way to tune."**
Grid search scales as $v^d$ and wastes budget on unimportant dimensions; random search beats it at
equal budget for 3+ hyperparameters (§2–§3). Reserve grid for 1–2 well-understood knobs.

**"Random search is just a lazy fallback."**
It is provably better than grid when effective dimensionality is low (§3, Bergstra & Bengio), which is
almost always. It is a principled default, not a compromise.

**"Bayesian optimization is always best."**
It shines when evaluations are expensive and few, in low-to-moderate dimensions. For cheap evaluations
or high-dimensional/conditional spaces, random search + Hyperband or TPE often wins, and the GP
overhead is not worth it (§4, §6).

**"Sample the learning rate uniformly in [1e-4, 1e-1]."**
That starves the small values; multiplicative hyperparameters must be sampled on a **log scale** (§7,
Experiment 4).

**"My tuned CV score is my expected performance."**
It is optimistically biased by the search itself — the max of many noisy estimates. Report nested CV
or a locked test set (§8, [05.04 §8](../04-cross-validation/)).

**"More tuning always helps."**
Past the CV noise floor you are overfitting the validation estimate; gains vanish on unseen data (§8).
Know when to stop.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — grid search, random search, a **Gaussian-process Bayesian
  optimizer** (RBF kernel, exact posterior, Expected Improvement), and successive halving, all in
  NumPy; grid/random verified against scikit-learn. Five experiments: (1) random beats grid when few
  dimensions matter; (2) Bayesian optimization reaching the optimum in fewer evaluations than random;
  (3) successive halving finding a near-best config cheaply; (4) log- vs linear-scale sampling for a
  learning rate; (5) over-tuning — the winner's optimism growing with the number of configs tried.
- **[exercises.md](exercises.md)** — derive Expected Improvement, implement TPE and Hyperband,
  reproduce every experiment.
- **[references.md](references.md)** — Bergstra & Bengio, the GP-optimization and Hyperband papers,
  Optuna.

**Next**: [05.06 — Calibration](../06-calibration/) — the last piece: making a model's probabilities
mean what they say, which accuracy, AUC, and tuning all leave unchecked.
