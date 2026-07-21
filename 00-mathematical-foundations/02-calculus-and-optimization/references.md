# 00.02 — References: Calculus and Optimization

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-3 | Gradients, Jacobians, Hessians | Deisenroth et al., *MML*, Ch. 5; Goodfellow et al., *Deep Learning*, §4.3 |
| §4 | Taylor expansion as the basis of algorithms | Nocedal & Wright, *Numerical Optimization*, §2.1 |
| §5 | Optimality conditions | Nocedal & Wright, §2.1, Thm 2.3-2.4 |
| §5.1 | Saddle points dominate in high dimensions | Dauphin et al. (2014); Choromanska et al. (2015) |
| §6 | Convexity | Boyd & Vandenberghe, *Convex Optimization*, Ch. 2-3 |
| §7 | Gradient descent, descent lemma | Nesterov, *Lectures on Convex Optimization*, §1.2, §2.1 |
| §8 | Conditioning and convergence rate | Boyd & Vandenberghe, §9.3; Goh (2017), *Why Momentum Really Works* |
| §9 | SGD, Robbins-Monro conditions | Robbins & Monro (1951); Bottou et al. (2018) |
| §10 | Momentum, Nesterov acceleration | Polyak (1964); Nesterov (1983); Sutskever et al. (2013) |
| §10 | Optimality of $O(\sqrt{\kappa})$ | Nemirovski & Yudin (1983); Nesterov, §2.1.4 |
| §11 | AdaGrad, RMSProp, Adam, AdamW | Duchi et al. (2011); Tieleman & Hinton (2012); Kingma & Ba (2015); Loshchilov & Hutter (2019) |
| §12 | Newton, BFGS, L-BFGS | Nocedal & Wright, Ch. 3, 6, 7 |
| §13-14 | Lagrange, KKT, duality | Boyd & Vandenberghe, Ch. 5 |
| §15 | Subgradients, proximal methods, ISTA | Beck & Teboulle (2009); Parikh & Boyd (2014) |
| §16 | Convergence rates | Nesterov, *Lectures on Convex Optimization*, Ch. 2 |
| §17 | Practical tuning | Google Research, *Deep Learning Tuning Playbook* |

---

## Books

**Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge.** — free at
<https://web.stanford.edu/~boyd/cvxbook/>
The definitive treatment of convexity, duality, and KKT. Chapter 5 is the source for §13-14 and
the single best explanation of Lagrange duality in print. Lecture videos and slides are also free.

**Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.**
The reference for the *algorithms*: line search (Ch. 3), Newton and quasi-Newton (Ch. 6),
L-BFGS and the two-loop recursion (Ch. 7, Alg. 7.4). The Wolfe conditions and the bracket/zoom
line search implemented in `from_scratch.py` are Alg. 3.5-3.6.

**Nesterov, Y. (2018). *Lectures on Convex Optimization*, 2nd ed. Springer.**
Where the accelerated method and the matching lower bound come from. Rigorous and terse.

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.** — free at
<https://www.deeplearningbook.org/>
Chapter 4 (numerical computation) and Chapter 8 (optimization for training deep models) are the
deep-learning-specific counterpart to this chapter.

**Deisenroth, M. P., Faisal, A. A. & Ong, C. S. (2020). *Mathematics for Machine Learning*.**
— free at <https://mml-book.github.io/> — Chapter 5 (vector calculus) and Chapter 7 (optimization).

**Bertsekas, D. P. (2016). *Nonlinear Programming*, 3rd ed. Athena Scientific.**
The most thorough treatment of constrained optimization theory if Boyd leaves you wanting more.

---

## Papers

### Stochastic optimization
- **Robbins, H. & Monro, S. (1951).** "A Stochastic Approximation Method." *Annals of
  Mathematical Statistics* 22(3), 400-407. — the conditions in §9.
- **Bottou, L., Curtis, F. E. & Nocedal, J. (2018).** "Optimization Methods for Large-Scale
  Machine Learning." *SIAM Review* 60(2), 223-311.
  [arXiv:1606.04838](https://arxiv.org/abs/1606.04838) — the best single survey of the area.
- **Keskar, N. S. et al. (2017).** "On Large-Batch Training for Deep Learning: Generalization Gap
  and Sharp Minima." [arXiv:1609.04836](https://arxiv.org/abs/1609.04836) — why gradient noise
  may be a feature.

### Momentum and acceleration
- **Polyak, B. T. (1964).** "Some methods of speeding up the convergence of iteration methods."
  *USSR Computational Mathematics and Mathematical Physics* 4(5), 1-17. — heavy ball.
- **Nesterov, Y. (1983).** "A method for solving the convex programming problem with convergence
  rate $O(1/k^2)$." *Soviet Mathematics Doklady* 27, 372-376.
- **Sutskever, I. et al. (2013).** "On the importance of initialization and momentum in deep
  learning." *ICML*. — the reformulation of NAG used in deep learning frameworks.
- **Goh, G. (2017).** "Why Momentum Really Works." *Distill*.
  <https://distill.pub/2017/momentum/> — **read this.** The interactive eigenvalue-by-eigenvalue
  decomposition is the clearest exposition of §8 and §10 anywhere.

### Adaptive methods
- **Duchi, J., Hazan, E. & Singer, Y. (2011).** "Adaptive Subgradient Methods for Online Learning
  and Stochastic Optimization." *JMLR* 12, 2121-2159. — AdaGrad.
- **Tieleman, T. & Hinton, G. (2012).** "Lecture 6.5 — RMSProp." *COURSERA: Neural Networks for
  Machine Learning.* — RMSProp was never published as a paper.
- **Kingma, D. P. & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." *ICLR*.
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) — §2 and §3 contain the bias-correction
  argument analysed in §11 and measured in Experiment 3.
- **Reddi, S. J., Kale, S. & Kumar, S. (2018).** "On the Convergence of Adam and Beyond." *ICLR*.
  — shows Adam can fail to converge even on convex problems; introduces AMSGrad.
- **Loshchilov, I. & Hutter, F. (2019).** "Decoupled Weight Decay Regularization." *ICLR*.
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) — AdamW.
- **Liu, L. et al. (2020).** "On the Variance of the Adaptive Learning Rate and Beyond." *ICLR*.
  [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) — RAdam; the argument that early
  second-moment *variance*, not just bias, is what warmup addresses (§11, Experiment 3).

### Loss landscapes and saddle points
- **Dauphin, Y. et al. (2014).** "Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization." *NeurIPS*.
  [arXiv:1406.2572](https://arxiv.org/abs/1406.2572) — the basis of §5.1.
- **Choromanska, A. et al. (2015).** "The Loss Surfaces of Multilayer Networks." *AISTATS*.
- **Li, H. et al. (2018).** "Visualizing the Loss Landscape of Neural Nets." *NeurIPS*.
  [arXiv:1712.09913](https://arxiv.org/abs/1712.09913) — why residual connections smooth the
  landscape.

### Proximal and non-smooth methods
- **Beck, A. & Teboulle, M. (2009).** "A Fast Iterative Shrinkage-Thresholding Algorithm for
  Linear Inverse Problems." *SIAM J. Imaging Sciences* 2(1), 183-202. — ISTA and FISTA.
- **Parikh, N. & Boyd, S. (2014).** "Proximal Algorithms." *Foundations and Trends in
  Optimization* 1(3), 127-239. — free; the reference for proximal operators.

---

## Courses and lectures

| Course | Institution | Link |
|---|---|---|
| EE364A — Convex Optimization I | Stanford (Boyd) | <https://web.stanford.edu/class/ee364a/> |
| 10-725 — Convex Optimization | CMU (Tibshirani) | <https://www.stat.cmu.edu/~ryantibs/convexopt/> |
| CS231n — Optimization notes | Stanford | <https://cs231n.github.io/optimization-1/> |

CMU 10-725 is the best free course specifically for the ML-relevant subset — proximal methods,
coordinate descent, and duality are covered far better there than in most ML courses.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.optim`](https://github.com/pytorch/pytorch/tree/main/torch/optim) | `sgd.py`, `adam.py`, `adamw.py` — compare line by line with §10-11; note how closely the code tracks the update rules |
| [`scipy.optimize`](https://github.com/scipy/scipy/tree/main/scipy/optimize) | `_linesearch.py` (Wolfe conditions), `lbfgsb_py.py`, `_minimize.py` |
| [scikit-learn `_logistic.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py) | why L-BFGS is the default solver, and where Newton/IRLS is used instead |
| [scikit-learn `_coordinate_descent.pyx`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_cd_fast.pyx) | soft thresholding (§15) in production Cython |
| [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook) | the practical counterpart to §17 |

---

## Deferred to later chapters

- **Backpropagation as reverse-mode autodiff** → [07.02](../../07-deep-learning/02-backpropagation/)
- **Learning-rate schedules, warmup in practice** → [07.06](../../07-deep-learning/06-optimizers/)
- **The SVM dual, fully worked** → [03.07](../../03-supervised-learning/07-svm/)
- **Coordinate descent for Lasso** → [03.02](../../03-supervised-learning/02-regularized-linear-models/)
- **EM as alternating optimization** → [04.04](../../04-unsupervised-learning/04-gaussian-mixtures/)
- **Bayesian optimization for hyperparameters** → [05.05](../../05-model-evaluation/05-hyperparameter-optimization/)
