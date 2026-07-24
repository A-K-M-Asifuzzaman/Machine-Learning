# 06.01 — References: Bagging

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Bagging idea | Breiman (1996a) |
| §2-§3 | Variance reduction, the $\rho$ floor | Hastie et al., *ESL*, §8.7, §15.2 |
| §4 | The bootstrap, 63.2% | Efron & Tibshirani (1993); ESL §7.11 |
| §6 | Out-of-bag estimation | Breiman (1996b, 2001); ESL §15.3.1 |
| §7-§8 | Which models benefit; bias-variance | Breiman (1996a); ESL §8.7 |
| §9 | Voting for classification | Breiman (1996a) |
| §10 | Toward random forests | Breiman (2001) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§8.7 introduces bagging; **§15.2 has the $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ derivation** that
is the heart of this chapter (stated there for random forests, but it is the bagging identity); §7.11
for the bootstrap and §15.3.1 for OOB. The whole of Chapter 15 is the natural continuation into
[06.02](../02-random-forests/).

**Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.**
The bootstrap itself, whose resampling bagging is built on. Chapter 2 for the 63.2% fact.

**Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press.**
The dedicated ensemble-methods reference. Chapter 3 on bagging, with a careful treatment of *why*
it works and when it does not — a good complement to §7's "which models benefit" discussion.

---

## Papers

- **Breiman, L. (1996a).** "Bagging Predictors." *Machine Learning* 24(2), 123-140. — **the
  original.** Introduces bootstrap aggregating, the variance-reduction argument, and the crucial
  observation that it helps *unstable* learners (§7). Short and very readable.
- **Breiman, L. (1996b).** "Out-of-Bag Estimation." Technical report, UC Berkeley. — the OOB error
  estimate of §6.
- **Breiman, L. (1996c).** "Heuristics of instability and stabilization in model selection."
  *Annals of Statistics* 24(6), 2350-2383. — the theory of why bagging stabilizes high-variance
  learners; the formal version of §7-§8.
- **Breiman, L. (2001).** "Random Forests." *Machine Learning* 45(1), 5-32. — where bagging becomes
  a random forest by adding feature subsampling (§3, §10), and where the $\rho$-floor argument is
  made precise. The direct sequel; read after [06.02](../02-random-forests/).
- **Bühlmann, P. & Yu, B. (2002).** "Analyzing Bagging." *Annals of Statistics* 30(4), 927-961. —
  a rigorous analysis of exactly why and when bagging reduces variance, including the role of the
  base learner's instability.
- **Friedman, J. H. & Hall, P. (2007).** "On bagging and nonlinear estimation." *Journal of
  Statistical Planning and Inference* 137(3), 669-683. — clarifies that bagging's benefit comes
  from smoothing hard-thresholding decisions, sharpening the intuition of §2.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_bagging.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_bagging.py) | `BaggingRegressor` / `BaggingClassifier`; note `oob_score`, `max_samples`, `max_features`, and `bootstrap` — the last three are what turn bagging into a random forest |
| [scikit-learn `_forest.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py) | how the OOB machinery is shared between bagging and forests |

> **A note on generality.** sklearn's `BaggingClassifier`/`BaggingRegressor` accept *any* base
> estimator, not just trees — exactly the generic design of `from_scratch.py`. Bagging a k-NN or a
> neural net is valid; bagging a linear model is (§7) a waste. The `RandomForest` classes are, under
> the hood, `BaggingClassifier` with a tree base learner and `max_features` set — which is the whole
> content of the bagging→forest step.

---

## Deferred to later chapters

- **Random forests — bagging + feature subsampling** → [06.02](../02-random-forests/)
- **Boosting — the bias-reducing counterpart** → [06.03](../03-boosting-theory/)
- **The bias-variance decomposition in full** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Cross-validation, the alternative to OOB** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **The bootstrap for confidence intervals** → [00.04 §12](../../00-mathematical-foundations/04-statistics-and-inference/)
- **Stacking — a different way to combine models** → [06.06](../06-stacking/)
