# 03.07 — References: Support Vector Machines

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | Max-margin, the primal | Boser, Guyon & Vapnik (1992); Vapnik (1995) |
| §4 | The dual | Boyd & Vandenberghe, Ch. 5; Burges (1998) |
| §5 | KKT and support vectors | Burges (1998) §3; Boyd & Vandenberghe §5.5.3 |
| §6 | Soft margin | Cortes & Vapnik (1995) |
| §7 | Hinge-loss view | Hastie et al., *ESL*, §12.3.2 |
| §8 | The kernel trick | Aizerman et al. (1964); Schölkopf & Smola (2002) |
| §9 | Mercer's condition | Mercer (1909); Schölkopf & Smola, Ch. 2 |
| §11 | SMO | Platt (1998); Keerthi et al. (2001) |
| §12 | SVR | Smola & Schölkopf (2004) |
| §13 | Multiclass | Hsu & Lin (2002) |

---

## Books

**Schölkopf, B. & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.**
**The definitive kernel-methods book.** Chapter 2 for kernels and Mercer's theorem, Chapter 7 for
SVMs, Chapter 9 for SVR. Comprehensive and rigorous; the reference to reach for when §8-§9 leave
you wanting more.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
**Chapter 12** is the statistician's view. §12.3.2 is the source for §7's hinge-loss framing and
the comparison table — reading SVMs as regularized ERM rather than as a geometric special case is
the single most useful reframing in this chapter.

**Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer.**
Where the margin-based generalization theory comes from. The VC-dimension argument for why
margins control complexity independently of dimension is the theoretical foundation of §1.

**Cristianini, N. & Shawe-Taylor, J. (2000). *An Introduction to Support Vector Machines*.
Cambridge.**
The gentlest complete treatment; good if the dual derivation of §4 moved too fast.

**Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*.** — free at
<https://web.stanford.edu/~boyd/cvxbook/>
Chapter 5 for the Lagrangian duality and KKT machinery this chapter consumes. §5.5.3 states
complementary slackness in the form used in §5. If §4-§5 felt like magic, the magic is here.

---

## Papers

### Foundational
- **Boser, B. E., Guyon, I. M. & Vapnik, V. N. (1992).** "A Training Algorithm for Optimal Margin
  Classifiers." *COLT*. — the paper that introduced the kernel trick to margin classifiers.
- **Cortes, C. & Vapnik, V. (1995).** "Support-Vector Networks." *Machine Learning* 20(3),
  273-297. — the **soft margin** of §6. One of the most-cited papers in machine learning.
- **Aizerman, M. A., Braverman, E. M. & Rozonoer, L. I. (1964).** "Theoretical foundations of the
  potential function method in pattern recognition learning." — the kernel trick, three decades
  before it was applied to SVMs.
- **Mercer, J. (1909).** "Functions of positive and negative type and their connection with the
  theory of integral equations." *Phil. Trans. Royal Society A* 209, 415-446.

### Tutorials worth reading in full
- **Burges, C. J. C. (1998).** "A Tutorial on Support Vector Machines for Pattern Recognition."
  *Data Mining and Knowledge Discovery* 2(2), 121-167. — **the best single document on SVMs.**
  §3 works the KKT conditions carefully and is the direct source for §5 here. Read this if you
  read nothing else.
- **Smola, A. J. & Schölkopf, B. (2004).** "A tutorial on support vector regression." *Statistics
  and Computing* 14(3), 199-222. — §12's source.

### Algorithms
- **Platt, J. C. (1998).** "Sequential Minimal Optimization: A Fast Algorithm for Training Support
  Vector Machines." Microsoft Research MSR-TR-98-14. — **SMO**, including the pair-selection
  heuristics implemented in `from_scratch.py`. The naive version without those heuristics
  technically converges and practically does not, which this chapter's development shows.
- **Keerthi, S. S. et al. (2001).** "Improvements to Platt's SMO Algorithm for SVM Classifier
  Design." *Neural Computation* 13(3), 637-649. — the two-threshold refinement that libsvm uses.
- **Chang, C.-C. & Lin, C.-J. (2011).** "LIBSVM: A Library for Support Vector Machines."
  *ACM TIST* 2(3). — the implementation `sklearn.svm.SVC` wraps. The
  [practical guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) that accompanies it
  is the single most useful page of advice on actually using SVMs, and is the source of §10's
  recommendation to grid-search $C$ and $\gamma$ jointly on a log scale.
- **Fan, R.-E. et al. (2008).** "LIBLINEAR: A Library for Large Linear Classification." *JMLR* 9,
  1871-1874. — the primal solver behind `LinearSVC`, which scales to millions where the dual
  cannot.

### Theory and extensions
- **Schölkopf, B. et al. (2001).** "Estimating the Support of a High-Dimensional Distribution."
  *Neural Computation* 13(7), 1443-1471. — **one-class SVM** for anomaly detection
  ([04.08](../../04-unsupervised-learning/08-anomaly-detection/)).
- **Hsu, C.-W. & Lin, C.-J. (2002).** "A Comparison of Methods for Multiclass Support Vector
  Machines." *IEEE Trans. Neural Networks* 13(2), 415-425. — §13's source; the finding that
  one-vs-one is generally preferable.
- **Platt, J. (1999).** "Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods." — **Platt scaling**, what `probability=True` runs.
- **Rahimi, A. & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
  — approximate an RBF kernel with an explicit random feature map, recovering linear-time training.
  The practical answer when $n$ is too large for the dual.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`libsvm`](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | the reference SMO implementation; the code is compact and readable |
| [`liblinear`](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) | primal solvers for the linear case — dual coordinate descent, and why it scales |
| [scikit-learn `svm/`](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/svm) | thin wrappers over both; `_classes.py` documents the `C`, `gamma`, and `probability` semantics |
| [`sklearn.kernel_approximation`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_approximation.py) | `RBFSampler`, `Nystroem` — the Rahimi-Recht trick, letting a linear model imitate a kernel one |
| [`thundersvm`](https://github.com/Xtra-Computing/thundersvm) | GPU SVM, for when $n$ is large but you still want the kernel |

**A practical note.** `SVC` (libsvm, dual) is $O(n^{2})$–$O(n^{3})$ and becomes impractical above
~$10^{5}$ samples. `LinearSVC` (liblinear, primal) scales to millions but gives up the kernel. If
you need a nonlinear boundary at that scale, use `RBFSampler` + `SGDClassifier`, or stop using
SVMs and use gradient boosting.

---

## Deferred to later chapters

- **Kernel methods beyond SVMs — kernel ridge, Gaussian processes** → [03.03 §12](../03-basis-expansion/) has the connection; GPs in [Part 12](../../12-generative-models/)
- **One-class SVM for anomaly detection** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
- **Margin-based generalization bounds, VC dimension** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Calibrating SVM scores (Platt scaling, isotonic)** → [05.06](../../05-model-evaluation/06-calibration/)
- **Hinge loss in deep learning** → [07.04](../../07-deep-learning/04-loss-functions/)
- **Why boosting displaced SVMs on tabular data** → [06.05](../../06-ensembles/05-modern-gbdts/)
