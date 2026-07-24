# 05.04 — References: Cross-Validation & Model Selection

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | K-fold, bias-variance of the estimate | ESL §7.10; Kohavi (1995) |
| §4 | LOOCV, the hat-matrix shortcut, GCV | ESL §7.10; Golub, Heath & Wahba (1979) |
| §6 | One-standard-error rule | Breiman et al. (1984); ESL §7.10 |
| §7 | Leakage / selection bias in CV | Ambroise & McLachlan (2002); Kaufman et al. (2012) |
| §8 | Nested CV, selection bias | Cawley & Talbot (2010); Varma & Simon (2006) |
| §9 | Time-series and grouped CV | Bergmeir & Benítez (2012); Roberts et al. (2017) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. **§7.10 is the reference for this chapter**:
$K$-fold CV, the bias-variance of the estimate (§3), the LOOCV/GCV shortcut (§4), and — critically —
§7.10.2 "The Wrong and Right Way to Do Cross-validation," the feature-selection leakage of §7 that
Experiment 3 reproduces. Read it.

**Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed.**
— free at <https://otexts.com/fpp3/>. §5.10 "Time series cross-validation" is the reference for
forward chaining (§9).

**James, G., Witten, D., Hastie, T. & Tibshirani, R. (2021). *An Introduction to Statistical
Learning*, 2nd ed.** — free at <https://www.statlearning.com/>. Chapter 5 is the gentlest correct
introduction to CV and the bootstrap; a good first read before ESL §7.

---

## Papers

- **Kohavi, R. (1995).** "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model
  Selection." *IJCAI*. — **the classic empirical study** establishing stratified 10-fold CV as the
  default (§3, §5). Free at <https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf>.
- **Ambroise, C. & McLachlan, G. J. (2002).** "Selection bias in gene extraction on the basis of
  microarray gene-expression data." *PNAS* 99(10), 6562-6566. — **the definitive demonstration of the
  feature-selection leak** (§7, Experiment 3): selection outside CV makes noise look predictive. Free
  at <https://www.pnas.org/doi/10.1073/pnas.102102699>.
- **Cawley, G. C. & Talbot, N. L. C. (2010).** "On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation." *JMLR* 11, 2079-2107. — **the reference on nested CV and
  model-selection bias** (§8, Experiment 4). Free at
  <https://jmlr.org/papers/v11/cawley10a.html>.
- **Varma, S. & Simon, R. (2006).** "Bias in error estimation when using cross-validation for model
  selection." *BMC Bioinformatics* 7, 91. — quantifies the optimism of non-nested CV after tuning
  (§8).
- **Golub, G. H., Heath, M. & Wahba, G. (1979).** "Generalized cross-validation as a method for
  choosing a good ridge parameter." *Technometrics* 21(2), 215-223. — **GCV and the LOOCV shortcut**
  (§4).
- **Bergmeir, C. & Benítez, J. M. (2012).** "On the use of cross-validation for time series predictor
  evaluation." *Information Sciences* 191, 192-213. — **why random CV fails on time series** and what
  to do instead (§9, Experiment 6).
- **Roberts, D. R. et al. (2017).** "Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure." *Ecography* 40(8), 913-929. — blocked / grouped / spatial
  CV (§9).
- **Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012).** "Leakage in Data Mining:
  Formulation, Detection, and Avoidance." *ACM TKDD* 6(4). — the general treatment of leakage that
  §7 is one instance of.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.model_selection._split`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py) | `KFold`, `StratifiedKFold`, `LeaveOneOut`, `TimeSeriesSplit`, `GroupKFold` — the splitters `from_scratch.py` mirrors |
| [`sklearn.model_selection._validation`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_validation.py) | `cross_val_score`, `cross_validate`, `permutation_test_score` |
| [`sklearn.pipeline.Pipeline`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py) | the tool that makes inside-fold preprocessing automatic (§7) — cross-validate the pipeline |
| [`sklearn.model_selection.GridSearchCV`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py) | nested CV = `cross_val_score(GridSearchCV(...))` (§8) |

---

## Deferred to later chapters

- **Hyperparameter search over the inner loop** → [05.05](../05-hyperparameter-optimization/)
- **The bootstrap — CV's resampling cousin** → [00.04](../../00-mathematical-foundations/04-statistics-and-inference/)
- **Out-of-fold predictions for stacking — the same idea** → [06.06](../../06-ensembles/06-stacking/)
- **Data leakage in the pipeline more broadly** → [02.06](../../02-data/06-data-leakage/)
- **Bias-variance of the model (vs of the estimate)** → [05.01](../01-bias-variance-and-theory/)
