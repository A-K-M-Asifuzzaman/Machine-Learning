# 04.08 — References: Anomaly Detection

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | Framing, taxonomy | Chandola, Banerjee & Kumar (2009) |
| §3 | Density / statistical methods | Aggarwal (2017) Ch. 2-3 |
| §4 | LOF | Breunig, Kriegel, Ng & Sander (2000) |
| §5 | Isolation Forest | Liu, Ting & Zhou (2008) |
| §6 | One-Class SVM | Schölkopf et al. (2001) |
| §7 | Reconstruction-based | Aggarwal (2017) Ch. 3; Sakurada & Yairi (2014) |
| §8 | Evaluation | Campos et al. (2016) |

---

## Books

**Aggarwal, C. C. (2017). *Outlier Analysis*, 2nd ed.** — **the definitive book on anomaly
detection.** Covers every method here — statistical/density (§3), distance and LOF (§4), isolation
(§5), one-class (§6), and reconstruction (§7) — with the theory and the practical caveats. If you want
one reference, this is it.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§14.3.11 (outliers in clustering) and the density-estimation chapters give the statistical grounding.

---

## Papers

- **Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008).** "Isolation Forest." *ICDM*. — **the Isolation
  Forest paper** (§5): the path-length principle, the anomaly score, and the linear-time,
  distance-free algorithm. Won the ICDM Test of Time award. Free at
  <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf>.
- **Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2012).** "Isolation-Based Anomaly Detection." *ACM TKDD*
  6(1). — the extended journal version with more analysis.
- **Breunig, M. M., Kriegel, H.-P., Ng, R. T. & Sander, J. (2000).** "LOF: Identifying Density-Based
  Local Outliers." *SIGMOD*. — **the LOF paper** (§4): local reachability density and the LOF ratio.
  Free at <https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf>.
- **Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J. & Williamson, R. C. (2001).**
  "Estimating the Support of a High-Dimensional Distribution." *Neural Computation* 13(7). — **the
  One-Class SVM** (§6).
- **Chandola, V., Banerjee, A. & Kumar, V. (2009).** "Anomaly Detection: A Survey." *ACM Computing
  Surveys* 41(3). — **the standard survey** and the source for the taxonomy of §1-§2 (point/contextual/
  collective, outlier/novelty). Free at
  <https://conservancy.umn.edu/handle/11299/215731>.
- **Sakurada, M. & Yairi, T. (2014).** "Anomaly Detection Using Autoencoders with Nonlinear
  Dimensionality Reduction." *MLSDA*. — **autoencoder reconstruction** for anomaly detection (§7).
- **Campos, G. O. et al. (2016).** "On the evaluation of unsupervised outlier detection: measures,
  datasets, and an empirical study." *Data Mining and Knowledge Discovery* 30(4). — **how to evaluate**
  anomaly detectors (§8), with a careful benchmark.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.ensemble.IsolationForest`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_iforest.py) | the isolation trees, path-length scoring, `contamination`; verified against here |
| [`sklearn.neighbors.LocalOutlierFactor`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_lof.py) | reachability distance, local reachability density, LOF; verified against here |
| [`sklearn.svm.OneClassSVM`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_classes.py) | the one-class boundary method (§6) |
| [`PyOD`](https://github.com/yzhao062/pyod) | a comprehensive anomaly-detection library: 40+ detectors, ensembling, and benchmarks |

---

## Deferred to later chapters

- **Gaussian mixtures as density models for anomaly detection** → [04.04](../04-gaussian-mixtures/)
- **PCA reconstruction — the linear reconstruction method** → [04.06](../06-linear-dimensionality-reduction/)
- **Autoencoders — nonlinear reconstruction-based detection** → [07.xx / 12.xx]
- **Random forests / trees — the base of Isolation Forest** → [06.02](../../06-ensembles/02-random-forests/)
- **Precision/recall/AP for rare-class evaluation** → [05.03](../../05-model-evaluation/03-classification-metrics/)
- **Time-series and streaming anomaly detection** → [15.xx time series]
