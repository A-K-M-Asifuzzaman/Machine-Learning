# 05.01 — References: Bias-Variance & Learning Theory

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§4 | Bias-variance decomposition, the U-curve | Geman et al. (1992); ESL §7.3 |
| §5 | Classification decomposition | Domingos (2000); Kong & Dietterich (1995) |
| §6 | Learning curves | ESL §7.10; Ng, CS229 notes |
| §7-§8 | ERM, Hoeffding, finite-class bound | Shalev-Shwartz & Ben-David, Ch. 2-4 |
| §9 | VC dimension, fundamental theorem | Vapnik (1998); SSBD Ch. 6 |
| §10 | Double descent | Belkin et al. (2019); Hastie et al. (2019) |

---

## Books

**Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to
Algorithms*.** — free at <https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/>. **The
definitive modern treatment of learning theory** and the source for §7-§9. Chapters 2-4 build ERM,
the finite-class bound (§8), and PAC learning from scratch; Chapter 6 is VC dimension and the
fundamental theorem (§9); Chapters 26-28 cover Rademacher complexity and covering numbers (the
sharper tools beyond VC). If you read one book for the theory half of this chapter, read this.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. **Chapter 7 ("Model Assessment and Selection")**
is the source for the decomposition and the U-curve (§2-§4), effective degrees of freedom, and
learning/validation curves (§6). §7.3 has the bias-variance decomposition; §7.10 cross-validation and
the pitfalls this whole part elaborates.

**Vapnik, V. (1998). *Statistical Learning Theory*.** The original, comprehensive source for VC
theory (§9) and structural risk minimization. Dense; SSBD is the friendlier route in.

**Mohri, M., Rostamizadeh, A. & Talwalkar, A. (2018). *Foundations of Machine Learning*, 2nd ed.**
— free at <https://cs.nyu.edu/~mohri/mlbook/>. A rigorous alternative to SSBD, especially strong on
Rademacher complexity and margin bounds.

---

## Papers

- **Geman, S., Bienenstock, E. & Doursat, R. (1992).** "Neural networks and the bias/variance
  dilemma." *Neural Computation* 4(1), 1-58. — **the paper that named the tradeoff** (§2-§4) and
  framed it as the central dilemma of learning.
- **Domingos, P. (2000).** "A Unified Bias-Variance Decomposition and its Applications." *ICML*. —
  **the source for §5**: a decomposition valid for 0/1 (and other) losses, showing variance can
  *reduce* error and that bias and variance interact multiplicatively. Experiment 3 reproduces its
  central effect.
- **Kong, E. B. & Dietterich, T. G. (1995).** "Error-correcting output coding corrects bias and
  variance." *ICML*. — an earlier 0/1-loss decomposition; useful context for §5.
- **Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019).** "Reconciling modern machine-learning practice
  and the classical bias-variance trade-off." *PNAS* 116(32), 15849-15854. — **the double-descent
  paper** (§10). Names the interpolation threshold and the second descent. Free at
  <https://arxiv.org/abs/1812.11118>.
- **Hastie, T., Montanari, A., Rosset, S. & Tibshirani, R. J. (2022).** "Surprises in
  High-Dimensional Ridgeless Least Squares Interpolation." *Annals of Statistics* 50(2), 949-986. —
  **the precise analysis of double descent for linear min-norm least squares** — the exact setup of
  Experiment 7. Free at <https://arxiv.org/abs/1903.08560>.
- **Nakkiran, P. et al. (2020).** "Deep Double Descent: Where Bigger Models and More Data Hurt."
  *ICLR*. — double descent in deep nets, including *epoch-wise* double descent. Free at
  <https://arxiv.org/abs/1912.02292>.
- **Zhang, C. et al. (2017).** "Understanding deep learning requires rethinking generalization."
  *ICLR*. — the experiments (nets fitting random labels) that showed classical VC bounds cannot
  explain deep-net generalization; the empirical motivation for §10. Free at
  <https://arxiv.org/abs/1611.03530>.

---

## Lecture notes

- **Ng, A. — CS229 notes on learning theory** (<https://cs229.stanford.edu/>). The clearest short
  derivation of the Hoeffding/union-bound generalization bound (§8) and VC bound (§9), and of the
  bias-variance/learning-curve diagnosis (§6). Start here if SSBD is too dense.
- **Abu-Mostafa, Y. — *Learning From Data* (Caltech CS156)** (<https://work.caltech.edu/telecourse>).
  The best gentle introduction to VC dimension and the generalization bound, with the growth function
  and Sauer's lemma built up carefully.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`mlxtend.evaluate.bias_variance_decomp`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py) | a clean library version of the Monte-Carlo decomposer of `from_scratch.py` |
| [`sklearn.model_selection.learning_curve`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_validation.py) | learning and validation curves (§6) |
| [Belkin double-descent notebooks](https://github.com/mbelkin/double_descent) | reproductions of the double-descent experiments (§10) |

---

## Deferred to later chapters

- **Regression metrics — measuring the error we decomposed** → [05.02](../02-regression-metrics/)
- **Cross-validation — locating the U's minimum in practice** → [05.04](../04-cross-validation/)
- **Regularization — the bias-variance knob in action** → [03.02](../../03-supervised-learning/02-regularized-linear-models/)
- **Ensembles — bagging attacks variance, boosting attacks bias** → [Part 6](../../06-ensembles/)
- **Deep learning generalization — where double descent lives** → [Part 7](../../07-deep-learning/)
- **Rademacher complexity & margin bounds — sharper than VC** → [advanced learning theory]
