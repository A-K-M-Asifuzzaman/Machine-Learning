# 06.04 — References: Gradient Boosting

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | GBM, functional gradient descent | Friedman (2001); Mason et al. (2000) |
| §4 | Squared loss = residual fitting | Friedman (2001) §4.1; ESL §10.10 |
| §5 | Absolute / Huber loss | Friedman (2001) §4.3-4.4; ESL §10.6 |
| §6 | Log loss, Newton leaf value | Friedman (2001) §4.5-4.6; Friedman, Hastie & Tibshirani (2000) |
| §7 | Shrinkage | Friedman (2001) §5; ESL §10.12.1 |
| §8 | Stochastic gradient boosting | Friedman (2002) |
| §9 | Tree depth / interaction order | ESL §10.11; Friedman (2001) §6 |
| §10 | Why $M$ overfits | ESL §10.12; contrast with §15 (forests) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. **Chapter 10** is the reference treatment:
§10.9-10.10 the GBM algorithm and loss functions (§2-§6 here), §10.11 the right tree size (§9),
§10.12 regularization — shrinkage, subsampling, and why the number of trees is a complexity knob
(§7, §8, §10). Read alongside Chapter 15 (forests) for the overfitting contrast.

**Schapire, R. E. & Freund, Y. (2012). *Boosting: Foundations and Algorithms*.** Chapter 7 develops
the coordinate-descent / functional-gradient view that unifies AdaBoost ([06.03](../03-boosting-theory/))
and gradient boosting.

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** §14.3 on boosting for the
additive-model framing.

---

## Papers

- **Friedman, J. H. (2001).** "Greedy function approximation: a gradient boosting machine." *Annals
  of Statistics* 29(5), 1189-1232. — **THE paper.** Introduces the GBM, the pseudo-residual /
  functional-gradient view (§2), the per-loss instantiations (squared, absolute, Huber, deviance;
  §4-§6), shrinkage (§7), and tree-size guidance (§9). Every formula in this chapter traces here.
  Free at <https://jerryfriedman.su.domains/ftp/trebst.pdf>.
- **Friedman, J. H. (2002).** "Stochastic gradient boosting." *Computational Statistics & Data
  Analysis* 38(4), 367-378. — **subsampling** (§8): fit each tree on a random fraction of the rows,
  for accuracy, regularization, and speed. Free at
  <https://jerryfriedman.su.domains/ftp/stobst.pdf>.
- **Mason, L., Baxter, J., Bartlett, P. & Frean, M. (2000).** "Boosting algorithms as gradient
  descent." *NeurIPS*. — the parallel derivation of boosting as gradient descent in function space;
  the conceptual foundation for §2 ("AnyBoost").
- **Friedman, J., Hastie, T. & Tibshirani, R. (2000).** "Additive logistic regression: a statistical
  view of boosting." *Annals of Statistics* 28(2), 337-407. — the forward-stagewise / exponential-loss
  bridge from AdaBoost ([06.03 §7](../03-boosting-theory/)) to gradient boosting; introduces
  LogitBoost, the log-loss instance of §6.
- **Ridgeway, G. (2007).** "Generalized Boosted Models: A guide to the gbm package." — a clear,
  practical companion to Friedman's papers; the R `gbm` package's manual, useful for loss zoo and
  tuning intuition.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_gb.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_gb.py) | `GradientBoostingRegressor` / `GradientBoostingClassifier`; the loss classes, the init estimator, and the per-leaf update our `from_scratch.py` is checked against |
| [scikit-learn `_gb_losses`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_gb_losses.py) | the negative-gradient and leaf-value formulas for each loss (squared, LAD, Huber, deviance) — exactly §4-§6 |
| [scikit-learn `HistGradientBoosting`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/) | the histogram-based, second-order successor — the bridge to [06.05](../05-modern-gbdts/) |
| [`gbm` (R)](https://github.com/gbm-developers/gbm) | Ridgeway's original implementation, close to Friedman's papers |

---

## Deferred to later chapters

- **XGBoost / LightGBM / CatBoost — second-order boosting, regularized objective, histogram splits**
  → [06.05](../05-modern-gbdts/) (the gradient/Hessian leaf of §6 becomes the whole objective)
- **Stacking — a different way to combine models** → [06.06](../06-stacking/)
- **The bias-variance decomposition — why boosting attacks bias** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Early stopping and validation curves in general** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Calibrating a boosted classifier's probabilities** → [05.06](../../05-model-evaluation/06-calibration/)
- **Partial dependence and feature importance done honestly** → [17.02](../../17-explainable-ai/02-post-hoc/)
- **Quantile regression in full** → [15.xx time series / probabilistic forecasting]
