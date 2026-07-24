# 05.06 — References: Calibration

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§4 | Calibration, reliability, ECE, Brier decomposition | Murphy (1973); DeGroot & Fienberg (1983) |
| §5 | Miscalibration by model family | Niculescu-Mizil & Caruana (2005); Guo et al. (2017) |
| §6 | Platt scaling | Platt (1999) |
| §7 | Isotonic regression, PAV | Zadrozny & Elkan (2002); Ayer et al. (1955) |
| §8 | Temperature scaling | Guo et al. (2017) |
| §10 | Sharpness, proper scoring | Gneiting, Balabdaoui & Raftery (2007) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§9 and §10 for the models whose calibration behavior §5 describes; the log-loss/proper-scoring
framing connects to §8 of [05.03](../03-classification-metrics/).

**Barber, D. (2012). *Bayesian Reasoning and Machine Learning*.** — free at
<http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/>. Good background on why probabilistic outputs and
their honesty matter for decision-making (§3).

---

## Papers

- **Platt, J. (1999).** "Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*. — **Platt scaling**
  (§6): fit a sigmoid to SVM scores. Free at
  <https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf>.
- **Zadrozny, B. & Elkan, C. (2002).** "Transforming classifier scores into accurate multiclass
  probability estimates." *KDD*. — **isotonic regression for calibration** (§7), and multiclass
  extensions. Free at <https://cseweb.ucsd.edu/~elkan/calibrated.pdf>.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting Good Probabilities With Supervised
  Learning." *ICML*. — **the empirical study of which models miscalibrate and how** (§5): boosting
  overconfident, SVMs sigmoidal, and Platt vs isotonic. The basis of Experiment 2. Free at
  <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>.
- **Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017).** "On Calibration of Modern Neural
  Networks." *ICML*. — **modern nets are badly overconfident; temperature scaling fixes it** (§5, §8,
  Experiment 4). Free at <https://arxiv.org/abs/1706.04599>.
- **Murphy, A. H. (1973).** "A New Vector Partition of the Probability Score." *J. Applied
  Meteorology* 12(4), 595-600. — **Murphy's decomposition** of the Brier score into reliability −
  resolution + uncertainty (§4).
- **DeGroot, M. H. & Fienberg, S. E. (1983).** "The Comparison and Evaluation of Forecasters." *The
  Statistician* 32, 12-22. — the formal statistical treatment of calibration and refinement.
- **Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. & Silverman, E. (1955).** "An empirical
  distribution function for sampling with incomplete information." *Annals of Mathematical Statistics*
  26(4), 641-647. — the **Pool Adjacent Violators** algorithm (§7).
- **Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007).** "Probabilistic forecasts, calibration and
  sharpness." *JRSS-B* 69(2), 243-268. — **"sharp, subject to calibration"** (§10): why calibration
  alone is not enough. Free at <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf>.
- **Kull, M., Silva Filho, T. & Flach, P. (2017).** "Beta calibration." *AISTATS*. — a parametric
  calibrator between Platt and isotonic; useful when the sigmoid is too rigid but data is limited.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.calibration.CalibratedClassifierCV`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) | cross-validated Platt (`sigmoid`) and isotonic calibration — the leakage-free protocol of §9 |
| [`sklearn.isotonic.IsotonicRegression`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/isotonic.py) | PAV isotonic regression, verified against `from_scratch.py` |
| [`sklearn.calibration.calibration_curve`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) | reliability-diagram bins (§4) |
| [`netcal`](https://github.com/EFS-OpenSource/calibration-framework) | a dedicated calibration library: ECE/MCE, temperature/beta/histogram binning, for deep nets |

---

## Deferred to later chapters

- **Proper scoring rules & classification metrics** → [05.03 §8](../03-classification-metrics/)
- **Cross-validation — the held-out discipline the calibrator needs** → [05.04](../04-cross-validation/)
- **Cost-sensitive decisions that need calibrated probabilities** → [05.03 §11](../03-classification-metrics/)
- **Deep-net calibration at scale (temperature, ensembles, focal loss)** → [07.08](../../07-deep-learning/08-regularization/)
- **Conformal prediction — distribution-free uncertainty** → [18.xx robustness / uncertainty]

---

*This completes **Part 5 — Evaluation & Model Selection** (6/6):
[bias-variance & theory](../01-bias-variance-and-theory/) → [regression metrics](../02-regression-metrics/)
→ [classification metrics](../03-classification-metrics/) → [cross-validation](../04-cross-validation/)
→ [hyperparameter optimization](../05-hyperparameter-optimization/) → calibration. Together they are the
toolkit for evaluating a model honestly: decompose its error, measure it with the right metric,
estimate it without leakage, tune it without over-tuning, and make its probabilities mean what they
say.*
