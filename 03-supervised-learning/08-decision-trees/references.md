# 03.08 — References: Decision Trees

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §3 | Greedy induction, NP-completeness | Hyafil & Rivest (1976); Breiman et al. (1984) |
| §4-§5 | Impurity criteria | Breiman et al. (1984), Ch. 2-4; Quinlan (1986) |
| §5 | Gini as entropy's Taylor approximation | Raileanu & Stoffel (2004) |
| §6 | Regression trees | Breiman et al. (1984), Ch. 8 |
| §7 | Efficient split finding | Breiman et al. (1984); modern: Chen & Guestrin (2016) |
| §8 | Categorical splits, sorting trick | Breiman et al. (1984), §4.2.2 |
| §9 | Overfitting, instability | Hastie et al., *ESL*, §9.2; Breiman (1996b) |
| §10 | Cost-complexity pruning | Breiman et al. (1984), Ch. 3 |
| §11 | Missing values | Breiman et al. (1984), §5.3 (surrogates); Chen & Guestrin (2016) |
| §12 | Feature importance bias | Strobl et al. (2007); Louppe et al. (2013) |
| §14 | CART / ID3 / C4.5 | Breiman et al. (1984); Quinlan (1986, 1993) |

---

## Books

**Breiman, L., Friedman, J. H., Olshen, R. A. & Stone, C. J. (1984). *Classification and Regression
Trees*. Wadsworth.**
**The CART book — the foundational text for everything in this chapter.** Binary splits, Gini,
regression trees, cost-complexity pruning, surrogate splits, and the categorical sorting trick all
originate here. Still the definitive reference forty years on; what sklearn implements.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§9.2 is the compact modern treatment: trees, their instability, and the direct segue into bagging
and boosting that motivates the chapter's framing.

**Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.**
The C4.5 book: gain ratio, multiway splits, error-based pruning, and fractional-instance missing
value handling. The competing tradition to CART, and the source of §7's gain-ratio fix.

**Mitchell, T. (1997). *Machine Learning*, Ch. 3.** — the classic pedagogical introduction to ID3
and information gain, if §4 moved too fast.

---

## Papers

### Foundations
- **Hyafil, L. & Rivest, R. L. (1976).** "Constructing optimal binary decision trees is
  NP-complete." *Information Processing Letters* 5(1), 15-17. — the result that forces greedy
  induction (§3).
- **Quinlan, J. R. (1986).** "Induction of Decision Trees." *Machine Learning* 1(1), 81-106. —
  ID3 and information gain.

### Criteria
- **Raileanu, L. E. & Stoffel, K. (2004).** "Theoretical comparison between the Gini index and
  information gain criteria." *Annals of Mathematics and AI* 41(1), 77-93. — the proof behind §5:
  Gini and entropy agree on the vast majority of splits, and Gini approximates entropy. Confirms
  Experiment 2.

### Instability and ensembles (the "why Part 6 exists" thread)
- **Breiman, L. (1996b).** "Heuristics of instability and stabilization in model selection."
  *Annals of Statistics* 24(6), 2350-2383. — formalizes tree instability (§9), the property
  bagging exploits.
- **Breiman, L. (1996a).** "Bagging Predictors." *Machine Learning* 24(2), 123-140. — where the
  instability of §9 becomes a feature. The bridge to [06.01](../../06-ensembles/01-bagging/).

### Feature importance
- **Strobl, C. et al. (2007).** "Bias in random forest variable importance measures:
  Illustrations, sources and a solution." *BMC Bioinformatics* 8, 25. — **the paper behind §12.**
  Documents MDI's bias toward high-cardinality and continuous features and its consequences.
  Experiment 6 reproduces its central illustration.
- **Louppe, G. et al. (2013).** "Understanding variable importances in forests of randomized
  trees." *NeurIPS*. — a rigorous analysis of what MDI actually measures.
- **Lundberg, S. M. & Lee, S.-I. (2017).** "A Unified Approach to Interpreting Model Predictions."
  *NeurIPS*. — SHAP, the principled importance method §12 recommends; developed in full in
  [17.02](../../17-explainable-ai/02-post-hoc/).

### Modern split finding
- **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD*. — the
  approximate and sparsity-aware (missing-value) split-finding of §7 and §11, at scale. Read after
  [06.05](../../06-ensembles/05-modern-gbdts/).
- **Ke, G. et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
  *NeurIPS*. — histogram-based splitting and native categorical handling (§8).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_tree.pyx` / `_splitter.pyx`](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/tree) | the production CART; `_splitter.pyx` is the incremental scan of §7 in Cython; note thresholds are stored `float32` |
| [scikit-learn `_classes.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py) | the `ccp_alpha`, `min_impurity_decrease`, and importance interfaces |
| [`dtreeviz`](https://github.com/parrt/dtreeviz) | the best decision-tree visualizer; invaluable for building intuition |
| [XGBoost `updater_exact.cc` / `updater_histmaker.cc`](https://github.com/dmlc/xgboost/tree/master/src/tree) | how split finding is done at scale, including the default-direction missing-value logic |

**A note on sklearn's importances.** `feature_importances_` is MDI, and §12 explains why it
misleads. sklearn also ships `sklearn.inspection.permutation_importance`, which is what you should
actually use — and its own documentation now warns about the MDI bias, citing Strobl et al.

---

## Deferred to later chapters

- **Bagging — averaging unstable trees** → [06.01](../../06-ensembles/01-bagging/)
- **Random forests — decorrelating the trees** → [06.02](../../06-ensembles/02-random-forests/)
- **AdaBoost and gradient boosting — trees as weak learners** → [06.03](../../06-ensembles/03-boosting-theory/), [06.04](../../06-ensembles/04-gradient-boosting/)
- **XGBoost, LightGBM, CatBoost — production trees** → [06.05](../../06-ensembles/05-modern-gbdts/)
- **SHAP and permutation importance in depth** → [17.02](../../17-explainable-ai/02-post-hoc/)
- **Missing-data theory (MCAR/MAR/MNAR)** → [02.02](../../02-data/02-cleaning-and-missing-data/)
- **Explainable Boosting Machines — interpretable trees** → [17.01](../../17-explainable-ai/01-intrinsic/)
