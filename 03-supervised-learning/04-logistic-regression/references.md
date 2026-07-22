# 03.04 — References: Logistic Regression

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Why not OLS for classification | Hastie et al., *ESL*, §4.2 |
| §2 | The model, logit link, GLM framing | McCullagh & Nelder, *GLMs*, Ch. 4; Bishop, *PRML*, §4.3.2 |
| §3 | Odds ratios, odds vs risk | Hosmer, Lemeshow & Sturdivant, Ch. 3; Agresti, *Categorical Data Analysis*, Ch. 4 |
| §4-§5 | Loss, gradient, Hessian | Bishop, *PRML*, §4.3.3; Murphy, *PML*, §10.2 |
| §6 | Convexity | Boyd & Vandenberghe, §3.1.5; Murphy §10.2.3 |
| §8 | IRLS | McCullagh & Nelder §2.5; Hastie et al., *ESL*, §4.4.1 |
| §9 | Perfect separation | Albert & Anderson (1984); Heinze & Schemper (2002) |
| §10 | Regularization | Hastie et al., *ESL*, §4.4.4; Friedman et al. (2010) |
| §11 | Multinomial / softmax | Bishop §4.3.4; Murphy §10.3 |
| §12-§13 | Thresholds, calibration | Elkan (2001); Niculescu-Mizil & Caruana (2005) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§4.4 is the compact treatment: the model, IRLS, regularization, and the comparison with LDA that
[03.05](../05-generative-classifiers/) picks up. §4.2 has the argument of §1.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.** — free
§4.3 is the best derivation available: logistic and softmax regression, IRLS, and the Bayesian
treatment. §4.3.2-4.3.4 map directly onto §2-§11 here.

**Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013). *Applied Logistic Regression*,
3rd ed. Wiley.**
**The book if you need to interpret coefficients rather than just predict.** Odds ratios,
confidence intervals, model building, goodness of fit, and diagnostics — the applied-statistics
view that ML texts skip entirely. Chapter 3 is the source for §3.

**Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed. Wiley.**
The comprehensive reference for binary and multinomial responses. Chapter 4-6. Also the clearest
explanation anywhere of when an odds ratio approximates a risk ratio and when it badly does not.

**McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. Chapman & Hall.**
The GLM framework: logistic regression as one member of a family, and IRLS as *the* fitting
algorithm for all of them. §2.5 for IRLS.

**Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*.** — free at
<https://probml.github.io/pml-book/> — Chapter 10 covers logistic regression with modern framing
and good numerical detail.

---

## Papers

### Separation
- **Albert, A. & Anderson, J. A. (1984).** "On the existence of maximum likelihood estimates in
  logistic regression models." *Biometrika* 71(1), 1-10. — the definitive characterization of
  when the MLE exists (complete separation, quasi-complete separation, overlap). §9's theory.
- **Heinze, G. & Schemper, M. (2002).** "A solution to the problem of separation in logistic
  regression." *Statistics in Medicine* 21(16), 2409-2419. — **Firth's penalized likelihood**, the
  principled fix when you need interpretable coefficients and cannot simply regularize.
- **Firth, D. (1993).** "Bias reduction of maximum likelihood estimates." *Biometrika* 80(1),
  27-38. — the original penalty, motivated by bias reduction rather than by separation.

### Optimization
- **Minka, T. P. (2003).** "A comparison of numerical optimizers for logistic regression."
  — a short, extremely practical comparison of IRLS, conjugate gradient, and quasi-Newton on this
  specific problem. Worth reading before choosing a solver.
  <https://tminka.github.io/papers/logreg/>
- **Friedman, J., Hastie, T. & Tibshirani, R. (2010).** "Regularization Paths for Generalized
  Linear Models via Coordinate Descent." *JSS* 33(1). — penalized logistic regression by
  coordinate descent; the `glmnet` algorithm.
- **Defazio, A., Bach, F. & Lacoste-Julien, F. (2014).** "SAGA: A Fast Incremental Gradient Method
  With Support for Non-Strongly Convex Composite Objectives." *NeurIPS*. — sklearn's `saga`
  solver, the one to use for large sparse problems with $\ell_1$.

### Calibration and thresholds
- **Elkan, C. (2001).** "The Foundations of Cost-Sensitive Learning." *IJCAI*. — the threshold
  formula of §12, and the argument that adjusting the threshold is preferable to resampling.
  Short and worth reading in full.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting Good Probabilities With Supervised
  Learning." *ICML*. — the empirical study behind Experiment 4: which classifiers are calibrated
  out of the box (logistic regression, bagged trees) and which are not (SVMs, boosting, naive
  Bayes), plus what Platt scaling and isotonic regression each fix.
- **Platt, J. (1999).** "Probabilistic Outputs for Support Vector Machines." — Platt scaling: fit
  a logistic regression to another model's scores. This is what `SVC(probability=True)` does.
- **Guo, C. et al. (2017).** "On Calibration of Modern Neural Networks." *ICML*.
  [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) — modern networks are badly miscalibrated
  despite being accurate; temperature scaling fixes most of it.

### Historical
- **Cox, D. R. (1958).** "The Regression Analysis of Binary Sequences." *JRSS B* 20(2), 215-242.
- **Berkson, J. (1944).** "Application of the Logistic Function to Bio-Assay." *JASA* 39(227),
  357-365. — where the name "logit" comes from.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_logistic.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py) | `_logistic_loss_and_grad` — note the stable formulation; and the solver dispatch table showing when each of `lbfgs`, `liblinear`, `newton-cg`, `sag`, `saga` is used |
| [statsmodels `discrete_model.py`](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/discrete/discrete_model.py) | `Logit` — the inferential version, with standard errors, Wald tests, and a separation warning |
| [`torch.nn.BCEWithLogitsLoss`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp) | the fused stable loss of §4 in C++ |
| [`glmnet`](https://glmnet.stanford.edu/) | penalized logistic regression paths; the reference for regularized GLMs |
| [`sklearn.calibration`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) | `CalibratedClassifierCV` — Platt scaling and isotonic regression |

**On the two libraries again.** As with [03.01](../01-linear-regression/): sklearn gives you
`coef_` and is regularized by default; statsmodels gives you standard errors, p-values, and a
separation warning, and is unregularized. If you are interpreting coefficients, use statsmodels.
If your sklearn coefficients look shrunk compared to a textbook example, `C=1.0` is why.

---

## Deferred to later chapters

- **LDA/QDA and naive Bayes — the generative counterparts** → [03.05](../05-generative-classifiers/)
- **The generative vs discriminative comparison** → [03.05](../05-generative-classifiers/)
- **SVMs: the same linear boundary, a different loss** → [03.07](../07-svm/)
- **Classification metrics, ROC/PR, threshold selection** → [05.03](../../05-model-evaluation/03-classification-metrics/)
- **Calibration: reliability diagrams, Platt, isotonic, temperature** → [05.06](../../05-model-evaluation/06-calibration/)
- **Class imbalance: resampling vs thresholds vs class weights** → [02.05](../../02-data/05-class-imbalance/)
- **Softmax and cross-entropy in deep learning** → [07.04](../../07-deep-learning/04-loss-functions/)
- **Logistic regression as a one-layer network** → [07.01](../../07-deep-learning/01-neural-network-basics/)
