# 06.06 — References: Stacking & Blending

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1, §4 | Stacked generalization | Wolpert (1992) |
| §2-§3 | Leakage, out-of-fold predictions | Wolpert (1992); Breiman (1996b) |
| §5 | Blending vs stacking | Netflix Prize write-ups (Töscher et al. 2009) |
| §6 | Simple / non-negative meta-learner | Breiman (1996b); Duchi et al. (2008) (simplex) |
| §7-§8 | Diversity, learned vs average | ESL §8.8; van der Laan et al. (2007) |
| §9 | Multi-layer stacks | Sill et al. (2009); Netflix Prize |

---

## Papers

- **Wolpert, D. H. (1992).** "Stacked generalization." *Neural Networks* 5(2), 241-259. — **the
  original.** Introduces stacking and, crucially, the out-of-fold ("leave-one-out"/cross-validation)
  construction of level-1 inputs that avoids the leakage of §2-§3. The whole chapter is a modern
  reading of this paper.
- **Breiman, L. (1996).** "Stacked regressions." *Machine Learning* 24(1), 49-64. — stacking for
  regression, with the key practical finding that the meta-learner should use **non-negative**
  weights (§6). Also emphasizes that OOF predictions are essential.
- **van der Laan, M. J., Polley, E. C. & Hubbard, A. E. (2007).** "Super Learner." *Statistical
  Applications in Genetics and Molecular Biology* 6(1). — a rigorous, cross-validated stacking
  framework with an oracle guarantee: the Super Learner performs asymptotically as well as the best
  combination in its library. The theory behind "stacking can't do much worse than your best model,
  if done right."
- **Ting, K. M. & Witten, I. H. (1999).** "Issues in stacked generalization." *JAIR* 10, 271-289. —
  careful empirical study: use **class probabilities** (not labels) as meta-features, and a simple
  (multi-response linear) meta-learner (§4, §6).
- **Sill, J., Takács, G., Mackey, L. & Lin, D. (2009).** "Feature-weighted linear stacking."
  *arXiv:0911.0460*. — lets stacking weights depend on (meta-)features; from the Netflix Prize
  (§8-§9). Free at <https://arxiv.org/abs/0911.0460>.
- **Töscher, A., Jahrer, M. & Bell, R. (2009).** "The BigChaos Solution to the Netflix Grand Prize."
  — the canonical account of large multi-layer stacks and blending in practice (§5, §9).
- **Duchi, J., Shalev-Shwartz, S., Singer, Y. & Chandra, T. (2008).** "Efficient projections onto the
  L1-ball for learning in high dimensions." *ICML*. — the simplex projection used by the non-negative
  sum-to-one meta-learner (§6, `from_scratch.py`).

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§8.8 "Model Averaging and Stacking" gives the compact statistical treatment: stacking as a
cross-validated linear combination, and why it beats naive combination. Free at
<https://hastie.su.domains/ElemStatLearn/>.

**Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*.** Chapter 4 covers combination
methods including stacking, with the diversity analysis of §7 made precise.

**Kuncheva, L. I. (2014). *Combining Pattern Classifiers*, 2nd ed.** The most thorough single book on
combiners and on ensemble **diversity** (§7) — how to measure it and why it matters.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `StackingClassifier`/`StackingRegressor`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_stacking.py) | the OOF `cross_val_predict` meta-feature construction and the `final_estimator`; our `from_scratch.py` is checked against this |
| [`mlxtend` StackingCVClassifier](https://github.com/rasbt/mlxtend) | a clear, widely-used stacking implementation with the CV variant spelled out |
| [ML-Ensemble (`mlens`)](https://github.com/flennerhag/mlens) | multi-layer stacking (§9) with careful leakage handling |
| [SuperLearner (R)](https://github.com/ecpolley/SuperLearner) | van der Laan's Super Learner, the cross-validated stacker with the oracle property |

---

## Deferred to later chapters

- **Cross-validation and model selection in depth** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Data leakage — the general problem stacking must avoid** → [02.xx data leakage]
- **Calibrating stacked probabilities** → [05.06](../../05-model-evaluation/06-calibration/)
- **Bayesian model averaging — the probabilistic cousin** → [12.xx / 05.xx]
- **AutoML ensembles (auto-sklearn, H2O) — stacking automated** → [19.xx MLOps]

---

*This completes **Part 6 — Ensembles**: [bagging](../01-bagging/) → [random forests](../02-random-forests/)
→ [boosting/AdaBoost](../03-boosting-theory/) → [gradient boosting](../04-gradient-boosting/) →
[XGBoost/LightGBM/CatBoost](../05-modern-gbdts/) → stacking. The through-line is the bias-variance
decomposition ([05.01](../../05-model-evaluation/01-bias-variance-and-theory/)): bagging attacks
variance, boosting attacks bias, and stacking exploits complementary errors across models built by
either strategy.*
