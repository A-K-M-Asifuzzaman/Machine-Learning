# 06.03 — References: Boosting Theory & AdaBoost

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Weak learnability | Kearns & Valiant (1989); Schapire (1990) |
| §2-§3 | AdaBoost algorithm | Freund & Schapire (1997) |
| §4 | Forward stagewise / exponential loss | Friedman, Hastie & Tibshirani (2000); ESL §10.4 |
| §5 | Training-error bound | Freund & Schapire (1997); Schapire & Freund (2012) Ch. 3 |
| §6 | Exponential loss shape, noise fragility | ESL §10.6; Long & Servedio (2010) |
| §7 | Overfitting on noise | Dietterich (2000); ESL §10.7 |
| §8-§9 | Margins theory, and its limits | Schapire et al. (1998); Reyzin & Schapire (2006) |
| §10 | SAMME multiclass | Zhu, Zou, Rosset & Hastie (2009) |
| §11 | Boosting vs bagging | ESL Ch. 10 & 15; Breiman (1998) |

---

## Books

**Schapire, R. E. & Freund, Y. (2012). *Boosting: Foundations and Algorithms*.** MIT Press.
— **the definitive book, by AdaBoost's own authors.** Chapter 1 introduces the algorithm, Chapter 3
proves the training-error bound (§5), Chapter 5 develops the margins theory (§8), and Chapter 7 gives
the exponential-loss / coordinate-descent view (§4). If you read one thing beyond ESL, read this.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. **Chapter 10** is the statistical view of
boosting that this chapter follows: §10.1-10.4 the forward-stagewise / exponential-loss derivation
(§2-§4 here), §10.5-10.6 why the exponential loss and its consequences (§6), §10.7 off-the-shelf
comparison. This is the chapter that reframed AdaBoost as additive logistic regression.

**Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*.** Chapter 2 (Boosting) gives a
careful, self-contained treatment of the error bound and margins, complementary to ESL.

---

## Papers

- **Kearns, M. & Valiant, L. (1989).** "Cryptographic limitations on learning Boolean formulae and
  finite automata." *STOC*. — poses the **weak vs strong learnability** question (§1).
- **Schapire, R. E. (1990).** "The strength of weak learnability." *Machine Learning* 5(2), 197-227.
  — **the affirmative answer**: weak learners can be boosted to strong ones. The theoretical birth
  of boosting.
- **Freund, Y. & Schapire, R. E. (1997).** "A decision-theoretic generalization of on-line learning
  and an application to boosting." *J. Computer and System Sciences* 55(1), 119-139. — **the AdaBoost
  paper.** The algorithm (§2), the vote weight (§3), and the training-error bound (§5) are all here.
  Won the 2003 Gödel Prize.
- **Friedman, J., Hastie, T. & Tibshirani, R. (2000).** "Additive logistic regression: a statistical
  view of boosting." *Annals of Statistics* 28(2), 337-407. — **reinterprets AdaBoost as forward
  stagewise minimization of exponential loss** (§4), the pivot that leads to gradient boosting
  ([06.04](../04-gradient-boosting/)). Introduces LogitBoost. The single most influential paper for
  how this chapter is framed.
- **Schapire, R. E., Freund, Y., Bartlett, P. & Lee, W. S. (1998).** "Boosting the margin: a new
  explanation for the effectiveness of voting methods." *Annals of Statistics* 26(5), 1651-1686. —
  **the margins explanation** for why test error keeps falling after training error hits zero (§8).
- **Reyzin, L. & Schapire, R. E. (2006).** "How boosting the margin can also boost classifier
  complexity." *ICML*. — **the honest sequel**: shows the *minimum* margin, not the margin
  distribution, is what the bound controls — the subtlety Experiment 5 reproduces (§8).
- **Dietterich, T. G. (2000).** "An experimental comparison of three methods for constructing
  ensembles of decision trees." *Machine Learning* 40(2), 139-157. — **documents AdaBoost's
  fragility to label noise** (§7); the empirical basis for Experiment 3.
- **Long, P. M. & Servedio, R. A. (2010).** "Random classification noise defeats all convex
  potential boosters." *Machine Learning* 78(3), 287-304. — a hardness result: *any* convex-loss
  booster (AdaBoost included) can be defeated by noise. The theory behind §6-§7.
- **Zhu, J., Zou, H., Rosset, S. & Hastie, T. (2009).** "Multi-class AdaBoost." *Statistics and Its
  Interface* 2(3), 349-360. — **SAMME and SAMME.R** (§10); the $\ln(K-1)$ term.
- **Breiman, L. (1998).** "Arcing classifiers." *Annals of Statistics* 26(3), 801-849. — Breiman's
  own study of boosting-type ("arcing") algorithms, and the bias-variance contrast with bagging
  (§11).
- **Mason, L., Baxter, J., Bartlett, P. & Frean, M. (2000).** "Boosting algorithms as gradient
  descent." *NeurIPS*. — the other route to the functional-gradient view; the bridge to
  [06.04](../04-gradient-boosting/).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_weight_boosting.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_weight_boosting.py) | `AdaBoostClassifier` / `AdaBoostRegressor`; the SAMME and SAMME.R updates our `from_scratch.py` is checked against |
| [Freund & Schapire's original pseudocode](https://www.cs.princeton.edu/~schapire/papers/explaining-adaboost.pdf) | "Explaining AdaBoost" (2013) — Schapire's own compact tutorial, pseudocode and margins in a few pages |
| [`multiboost`](https://github.com/rmnmnr/multiboost) | a C++ boosting library with many variants; useful for seeing SAMME.R, LogitBoost, and cascades in one place |

---

## Deferred to later chapters

- **Gradient boosting — the loss generalized** → [06.04](../04-gradient-boosting/)
- **XGBoost / LightGBM / CatBoost — regularized second-order boosting** → [06.05](../05-modern-gbdts/)
- **Stacking — the other way to combine models** → [06.06](../06-stacking/)
- **The exponential vs logistic loss, in full** → [03.02 §6](../../03-supervised-learning/02-logistic-regression/)
- **Bias-variance decomposition — why boosting attacks bias** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Calibrating a boosted model's scores** → [05.06](../../05-model-evaluation/06-calibration/)
