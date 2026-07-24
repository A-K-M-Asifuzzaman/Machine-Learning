# 06.05 — References: XGBoost, LightGBM & CatBoost

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§5 | Regularized objective, 2nd-order step, leaf weight, split gain | Chen & Guestrin (2016) |
| §6 | Histogram split finding | Chen & Guestrin (2016) §4; Ke et al. (2017) |
| §7 | GOSS, EFB, leaf-wise growth | Ke et al. (2017) |
| §8 | Ordered boosting, ordered target statistics | Prokhorenkova et al. (2018); Dorogush et al. (2018) |
| §9 | Sparsity-aware split finding | Chen & Guestrin (2016) §3.4 |
| §11 | GBDTs vs deep nets on tabular data | Grinsztajn et al. (2022); Shwartz-Ziv & Armon (2022) |

---

## Papers — the three libraries

- **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD*. — **the
  XGBoost paper, and the source for §2-§6.** The regularized objective, the second-order Taylor
  expansion, the leaf weight $-G/(H+\lambda)$, the split-gain formula, the sparsity-aware (default-
  direction) split finding, and the approximate/histogram algorithm are all here, derived cleanly.
  Read §2-§4. Free at <https://arxiv.org/abs/1603.02754>.
- **Ke, G. et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*.
  — **GOSS, EFB, and the histogram algorithm** (§6-§7). The leaf-wise growth strategy is LightGBM's
  practical default. Free at
  <https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>.
- **Prokhorenkova, L. et al. (2018).** "CatBoost: unbiased boosting with categorical features."
  *NeurIPS*. — **ordered boosting and prediction shift** (§8); the theoretical heart of CatBoost.
  Free at <https://arxiv.org/abs/1706.09516>.
- **Dorogush, A. V., Ershov, V. & Gulin, A. (2018).** "CatBoost: gradient boosting with categorical
  features support." *arXiv:1810.11363*. — the systems companion: ordered target statistics,
  oblivious trees, GPU training. Free at <https://arxiv.org/abs/1810.11363>.

---

## Papers — tabular deep learning benchmarks (§11)

- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** "Why do tree-based models still outperform
  deep learning on typical tabular data?" *NeurIPS Datasets & Benchmarks*. — **the definitive recent
  study.** Isolates *why*: non-smooth target functions, robustness to uninformative features, and
  rotational non-invariance all favor trees. Free at <https://arxiv.org/abs/2207.08815>.
- **Shwartz-Ziv, R. & Armon, A. (2022).** "Tabular Data: Deep Learning is Not All You Need." *Information
  Fusion* 81, 84-90. — XGBoost beats several proposed tabular deep nets across datasets, even
  before accounting for tuning and speed. Free at <https://arxiv.org/abs/2106.03253>.
- **Gorishniy, Y. et al. (2021).** "Revisiting Deep Learning Models for Tabular Data." *NeurIPS*. —
  the strongest deep-tabular architectures (FT-Transformer, ResNet); useful for the other side of
  §11. Free at <https://arxiv.org/abs/2106.11959>.

---

## Books & background

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
Chapter 10 (gradient boosting, [06.04](../04-gradient-boosting/)) is the prerequisite; XGBoost is
that algorithm plus the regularized second-order objective. §10.14 discusses tree size and additive
structure relevant to §7.

**Boehmke, B. & Greenwell, B. (2019). *Hands-On Machine Learning with R*.** Chapters 12 (GBM) and 13
(XGBoost) are a clear applied companion, with tuning workflows for the knobs of §5 and §10.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [dmlc/xgboost](https://github.com/dmlc/xgboost) | the objective in `src/objective/`, the split evaluation in `src/tree/`; the `hist` and `exact` tree methods; our `from_scratch.py` is checked against this library |
| [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | histogram construction, GOSS, EFB, and leaf-wise growth in `src/treelearner/` |
| [catboost/catboost](https://github.com/catboost/catboost) | ordered boosting and ordered target statistics; oblivious tree construction |
| [scikit-learn `HistGradientBoosting`](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/ensemble/_hist_gradient_boosting) | a clean, readable histogram GBDT in Python/Cython — the best code to read to understand §6 |
| [XGBoost docs: "Introduction to Boosted Trees"](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) | Chen's own derivation of §2-§5, the same notation used here |

---

## Deferred to later chapters

- **Stacking — combining heterogeneous models** → [06.06](../06-stacking/)
- **SHAP — the honest feature attribution for GBDTs** → [17.02](../../17-explainable-ai/02-post-hoc/)
- **Hyperparameter tuning at scale (Bayesian / Optuna)** → [05.05](../../05-model-evaluation/05-hyperparameter-optimization/)
- **Calibrating a boosted classifier** → [05.06](../../05-model-evaluation/06-calibration/)
- **Deep learning for tabular data (FT-Transformer, TabNet)** → [11.xx / 07.xx]
- **Target/mean encoding done safely (out-of-fold)** → [02.xx feature engineering & leakage]
