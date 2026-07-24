# 05.03 — References: Classification Metrics

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§4 | Confusion matrix, precision/recall/F1 | ESL §9; Kuhn & Johnson §11 |
| §2 | Accuracy paradox under imbalance | He & Garcia (2009) |
| §5-§6 | Threshold, ROC, AUC | Fawcett (2006); Hanley & McNeil (1982) |
| §7 | PR curves, average precision | Davis & Goadrich (2006); Saito & Rehmsmeier (2015) |
| §8 | Proper scoring rules, log loss, Brier | Brier (1950); Gneiting & Raftery (2007) |
| §9 | MCC, Cohen's kappa | Matthews (1975); Chicco & Jurman (2020) |
| §11 | Cost-sensitive thresholds | Elkan (2001) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §9.2 covers the confusion matrix and the
ROC curve; the loss-function framing of Chapter 10 connects to proper scoring rules (§8).

**Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*.** Chapter 11 "Measuring Performance
in Classification Models" is the best applied reference: precision/recall/F1, ROC/AUC, the
imbalance problem, and calibration, with practical guidance for §2-§8.

**Provost, F. & Fawcett, T. (2013). *Data Science for Business*.** Chapters 7-8 give the clearest
intuition for ROC curves, expected value / cost-based thresholding (§11), and why accuracy misleads.

---

## Papers

- **Fawcett, T. (2006).** "An introduction to ROC analysis." *Pattern Recognition Letters* 27(8),
  861-874. — **the definitive ROC primer** (§6): construction, the AUC ranking interpretation, and
  common pitfalls. Read this for §5-§6. Free at
  <https://www.math.ucdavis.edu/~saito/data/roc/fawcett-roc.pdf>.
- **Hanley, J. A. & McNeil, B. J. (1982).** "The meaning and use of the area under a ROC curve."
  *Radiology* 143(1), 29-36. — the classic statement of **AUC = the Wilcoxon-Mann-Whitney
  probability** (§6, Experiment 3).
- **Davis, J. & Goadrich, M. (2006).** "The relationship between Precision-Recall and ROC curves."
  *ICML*. — **why PR curves are more informative than ROC under imbalance** (§7, Experiment 4). Free
  at <https://www.biostat.wisc.edu/~page/rocpr.pdf>.
- **Saito, T. & Rehmsmeier, M. (2015).** "The Precision-Recall Plot Is More Informative than the ROC
  Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE* 10(3), e0118432. — a
  thorough empirical case for PR/AP on rare-positive problems (§7).
- **Brier, G. W. (1950).** "Verification of forecasts expressed in terms of probability." *Monthly
  Weather Review* 78(1), 1-3. — the origin of the **Brier score** (§8).
- **Gneiting, T. & Raftery, A. E. (2007).** "Strictly Proper Scoring Rules, Prediction, and
  Estimation." *JASA* 102(477), 359-378. — **the reference on proper scoring rules** (§8): why log
  loss and Brier are minimized only by the true probabilities. Free at
  <https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>.
- **Matthews, B. W. (1975).** "Comparison of the predicted and observed secondary structure of T4
  phage lysozyme." *BBA* 405(2), 442-451. — the origin of the **Matthews correlation coefficient**
  (§9).
- **Chicco, D. & Jurman, G. (2020).** "The advantages of the Matthews correlation coefficient (MCC)
  over F1 score and accuracy in binary classification evaluation." *BMC Genomics* 21, 6. — **the case
  for MCC as the best single number under imbalance** (§9). Free at
  <https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7>.
- **Elkan, C. (2001).** "The Foundations of Cost-Sensitive Learning." *IJCAI*. — **the Bayes-optimal
  cost-sensitive threshold** (§11, Experiment 5). Free at
  <https://cseweb.ucsd.edu/~elkan/rescale.pdf>.
- **He, H. & Garcia, E. A. (2009).** "Learning from Imbalanced Data." *IEEE TKDE* 21(9), 1263-1284.
  — the survey framing the imbalance problem and its metrics (§2).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.metrics` classification](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py) | `precision_score`, `recall_score`, `f1_score`, `fbeta_score`, `matthews_corrcoef`, `cohen_kappa_score`, `log_loss`, `brier_score_loss` — checked against here |
| [`sklearn.metrics._ranking`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py) | `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score` — note the tie handling our `roc_curve` mirrors |
| [`sklearn` model evaluation docs](https://scikit-learn.org/stable/modules/model_evaluation.html) | the formulas, averaging schemes, and caveats for every metric |
| [`imbalanced-learn`](https://github.com/scikit-learn-contrib/imbalanced-learn) | `geometric_mean_score`, `classification_report_imbalanced` — metrics built for §2's problem |

---

## Deferred to later chapters

- **Calibration — making probabilities trustworthy** → [05.06](../06-calibration/)
- **Cross-validation — estimating these metrics honestly** → [05.04](../04-cross-validation/)
- **Class imbalance — resampling, class weights, thresholding** → [02.05](../../02-data/05-class-imbalance/)
- **Regression metrics — the continuous-target counterpart** → [05.02](../02-regression-metrics/)
- **Multiclass & multilabel losses in deep nets** → [07.04](../../07-deep-learning/04-loss-functions/)
