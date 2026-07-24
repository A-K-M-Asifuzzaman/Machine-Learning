# 06.02 — References: Random Forests

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | Forests, the $\rho$ floor | Breiman (2001); Hastie et al., *ESL*, §15.2 |
| §4 | `max_features`, defaults | Breiman (2001); ESL §15.3 |
| §6 | OOB error | Breiman (1996, 2001); ESL §15.3.1 |
| §7 | Importance bias | Strobl et al. (2007); Louppe et al. (2013) |
| §8 | Extra-Trees | Geurts, Ernst & Wehenkel (2006) |
| §9 | Proximities, RF as kernel | Breiman (2001); Lin & Jeon (2006); Scornet (2016) |
| §10-§11 | Strengths, vs boosting | ESL Ch. 15-16; Fernández-Delgado et al. (2014) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
**Chapter 15 is the definitive treatment.** §15.2 has the $\rho\sigma^2$ variance analysis that is
the analytical core of both this chapter and [06.01](../01-bagging/); §15.3 covers `max_features`,
OOB, and importances; §15.4 the connection to adaptive nearest neighbours (§9). Chapter 16
(boosting) is the natural contrast (§11).

**Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*.** Chapter 4 on random forests,
with careful theory on why decorrelation works.

**Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice*.** PhD thesis, free at
<https://arxiv.org/abs/1407.7502>. The most thorough single document on random forests — variance,
importances, and implementation, by a core scikit-learn contributor. If you want one deep reference
beyond ESL, this is it.

---

## Papers

- **Breiman, L. (2001).** "Random Forests." *Machine Learning* 45(1), 5-32. — **the original.**
  Introduces feature subsampling, the OOB estimate, importances, and proximities, all in one paper.
  The $\rho$-floor argument (§3), the defaults (§4), and the proximity/kernel view (§9) are all
  here. Highly readable; read it.
- **Breiman, L. (1996).** "Bagging Predictors." *Machine Learning* 24(2), 123-140. — the predecessor
  ([06.01](../01-bagging/)).
- **Ho, T. K. (1998).** "The Random Subspace Method for Constructing Decision Forests." *IEEE
  TPAMI* 20(8), 832-844. — feature subsampling predates Breiman's forest; this is the earlier idea.
- **Geurts, P., Ernst, D. & Wehenkel, L. (2006).** "Extremely randomized trees." *Machine Learning*
  63(1), 3-42. — Extra-Trees (§8).
- **Strobl, C. et al. (2007).** "Bias in random forest variable importance measures." *BMC
  Bioinformatics* 8, 25. — **the importance-bias paper** (§7). Documents both the cardinality bias
  and the correlated-feature problem, and proposes conditional permutation importance. Experiments
  3 and 4 reproduce its findings — including the nuance that a forest is less biased than a single
  tree but not unbiased.
- **Louppe, G. et al. (2013).** "Understanding variable importances in forests of randomized trees."
  *NeurIPS*. — a rigorous analysis of what MDI measures in a forest.
- **Lin, Y. & Jeon, Y. (2006).** "Random Forests and Adaptive Nearest Neighbors." *JASA* 101(474),
  578-590. — the formal statement of §9's "adaptive nearest neighbours" view.
- **Scornet, E. (2016).** "Random forests and kernel methods." *IEEE Trans. Information Theory*
  62(3), 1485-1500. — the forest-as-kernel connection made precise.
- **Wager, S. & Athey, S. (2018).** "Estimation and Inference of Heterogeneous Treatment Effects
  using Random Forests." *JASA* 113(523), 1228-1242. — causal forests; where the proximity/kernel
  view leads.
- **Fernández-Delgado, M. et al. (2014).** "Do we Need Hundreds of Classifiers to Solve Real World
  Classification Problems?" *JMLR* 15, 3133-3181. — the large benchmark study that found random
  forests among the best off-the-shelf classifiers across 121 datasets (§10).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_forest.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py) | `RandomForestClassifier/Regressor`, `ExtraTreesClassifier/Regressor`; the `max_features`, `oob_score`, and importance logic; note the base classes shared with bagging |
| [scikit-learn `_forest.py` importances](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py) | `feature_importances_` is MDI — sklearn's own docs now warn about its bias and point to `permutation_importance` |
| [`sklearn.inspection.permutation_importance`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/inspection/_permutation_importance.py) | the honest importance of §7 |
| [`ranger`](https://github.com/imbs-hl/ranger) (R/C++) | a fast forest implementation with conditional importance and more importance variants |

> **A note on how sklearn factors this.** `RandomForestClassifier` is essentially a `BaggingClassifier`
> with a tree base learner and `max_features` set at the split level — exactly the "bagging + one
> ingredient" framing of §1. Reading `_forest.py` alongside `_bagging.py` makes the relationship
> concrete.

---

## Deferred to later chapters

- **Boosting — the bias-reducing opposite** → [06.03](../03-boosting-theory/), [06.04](../04-gradient-boosting/)
- **XGBoost / LightGBM / CatBoost — why they edge out forests** → [06.05](../05-modern-gbdts/)
- **SHAP and conditional importance — the honest fix for correlated features** → [17.02](../../17-explainable-ai/02-post-hoc/)
- **The bias-variance decomposition in full** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Manifold learning for proximity visualization** → [04.07](../../04-unsupervised-learning/07-manifold-learning/)
- **Adaptive nearest neighbours / learned metrics** → [03.06 §8.4](../../03-supervised-learning/06-knn/)
