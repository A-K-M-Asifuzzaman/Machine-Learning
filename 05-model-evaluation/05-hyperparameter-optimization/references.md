# 05.05 — References: Hyperparameter Optimization

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§3 | Grid vs random search | Bergstra & Bengio (2012) |
| §4-§5 | Bayesian optimization, GP surrogate, EI | Jones et al. (1998); Snoek et al. (2012); Shahriari et al. (2016) |
| §4 | TPE | Bergstra et al. (2011) |
| §6 | Successive halving, Hyperband | Jamieson & Talwalkar (2016); Li et al. (2017) |
| §6 | BOHB | Falkner et al. (2018) |
| §8 | Over-tuning / selection bias | Cawley & Talbot (2010) |

---

## Books

**Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.**
— free at <https://gaussianprocess.org/gpml/>. **The reference for the GP surrogate** (§4): Chapter 2
gives the exact posterior mean and variance our `GaussianProcess` implements. Chapter 5 covers kernel
and hyperparameter choice.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§7.10 for the CV objective being optimized, and the model-selection bias that §8 warns about.

**Feurer, M. & Hutter, F. (2019). "Hyperparameter Optimization."** In *Automated Machine Learning*
(Hutter, Kotthoff & Vanschoren, eds.), Springer — free at
<https://www.automl.org/book/>. **The definitive modern survey** of everything in this chapter:
grid/random, Bayesian optimization, multi-fidelity (Hyperband, BOHB), and practical guidance.

---

## Papers

- **Bergstra, J. & Bengio, Y. (2012).** "Random Search for Hyper-Parameter Optimization." *JMLR* 13,
  281-305. — **the paper behind §3 and Experiment 1**: random search beats grid because of low
  effective dimensionality. Read it. Free at
  <https://jmlr.org/papers/v13/bergstra12a.html>.
- **Jones, D. R., Schonlau, M. & Welch, W. J. (1998).** "Efficient Global Optimization of Expensive
  Black-Box Functions." *J. Global Optimization* 13, 455-492. — **the EGO algorithm**: GP surrogate +
  Expected Improvement (§4-§5), the foundation of Bayesian optimization.
- **Snoek, J., Larochelle, H. & Adams, R. P. (2012).** "Practical Bayesian Optimization of Machine
  Learning Algorithms." *NeurIPS*. — **Bayesian optimization for ML hyperparameters** (§4); popularized
  the approach. Free at <https://arxiv.org/abs/1206.2944>.
- **Shahriari, B. et al. (2016).** "Taking the Human Out of the Loop: A Review of Bayesian
  Optimization." *Proc. IEEE* 104(1), 148-175. — **the definitive survey** of surrogates and
  acquisition functions (§4-§5). Free at <https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf>.
- **Bergstra, J. et al. (2011).** "Algorithms for Hyper-Parameter Optimization." *NeurIPS*. — **TPE**
  (§4), the tree-structured Parzen estimator behind Hyperopt/Optuna.
- **Jamieson, K. & Talwalkar, A. (2016).** "Non-stochastic Best Arm Identification and Hyperparameter
  Optimization." *AISTATS*. — **successive halving** as a bandit problem (§6).
- **Li, L. et al. (2017).** "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization."
  *JMLR* 18. — **Hyperband** (§6). Free at <https://arxiv.org/abs/1603.06560>.
- **Falkner, S., Klein, A. & Hutter, F. (2018).** "BOHB: Robust and Efficient Hyperparameter
  Optimization at Scale." *ICML*. — **BOHB**, Hyperband + Bayesian optimization (§6). Free at
  <https://arxiv.org/abs/1807.01774>.
- **Cawley, G. C. & Talbot, N. L. C. (2010).** "On Over-fitting in Model Selection..." *JMLR* 11. —
  the selection-bias result behind §8 (also cited in [05.04](../04-cross-validation/)).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Optuna](https://github.com/optuna/optuna) | the modern default: TPE, pruning (Hyperband/successive halving), conditional spaces, log-scale distributions (§7) |
| [Hyperopt](https://github.com/hyperopt/hyperopt) | the original TPE implementation (§4) |
| [scikit-learn `GridSearchCV` / `RandomizedSearchCV`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py) | grid and random search, verified against here |
| [scikit-learn `HalvingGridSearchCV`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search_successive_halving.py) | successive halving (§6) |
| [scikit-optimize (`skopt`)](https://github.com/scikit-optimize/scikit-optimize) | GP-based Bayesian optimization with EI/UCB/PI (§4-§5) |

---

## Deferred to later chapters

- **Cross-validation & nested CV — the objective and the honest estimate** → [05.04](../04-cross-validation/)
- **Bias-variance — hyperparameters are its knobs** → [05.01](../01-bias-variance-and-theory/)
- **Neural-architecture search & training-time tuning** → [07.09](../../07-deep-learning/09-training-dynamics/)
- **AutoML pipelines** → [19.xx MLOps]
- **Gaussian processes as models in their own right** → [12.xx / probabilistic ML]
