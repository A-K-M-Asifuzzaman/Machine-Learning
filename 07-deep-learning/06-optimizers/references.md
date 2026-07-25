# 07.06 — References: Optimizers

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | SGD, noise, saddle escape | Bottou (2010); Ge et al. (2015) |
| §4 | Momentum, Nesterov | Polyak (1964); Sutskever et al. (2013) |
| §5 | AdaGrad, RMSProp | Duchi et al. (2011); Tieleman & Hinton (2012) |
| §6 | Adam, AdamW | Kingma & Ba (2015); Loshchilov & Hutter (2019) |
| §7 | Schedules, warmup, one-cycle | Smith (2017); Goyal et al. (2017) |
| §8 | Second-order, saddles | Dauphin et al. (2014) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. **Chapter 8 "Optimization for Training Deep Models"** is the
reference: SGD (§2), momentum (§4), adaptive methods (§5-§6), and the loss-landscape / saddle-point
discussion (§8).

**Ruder, S. (2016). "An overview of gradient descent optimization algorithms."** — a widely-read survey
covering every optimizer here in one place. Free at <https://arxiv.org/abs/1609.04747>.

---

## Papers

- **Kingma, D. P. & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." *ICLR*. — **the Adam
  paper** (§6): first/second moments, bias correction, and the default hyperparameters. Free at
  <https://arxiv.org/abs/1412.6980>.
- **Loshchilov, I. & Hutter, F. (2019).** "Decoupled Weight Decay Regularization." *ICLR*. — **AdamW**
  (§6): why weight decay must be decoupled from the adaptive update. The modern default. Free at
  <https://arxiv.org/abs/1711.05101>.
- **Duchi, J., Hazan, E. & Singer, Y. (2011).** "Adaptive Subgradient Methods for Online Learning and
  Stochastic Optimization." *JMLR* 12. — **AdaGrad** (§5). Free at
  <https://jmlr.org/papers/v12/duchi11a.html>.
- **Tieleman, T. & Hinton, G. (2012).** "RMSProp." *Coursera lecture 6e*. — **RMSProp** (§5); the EMA
  fix for AdaGrad's decay.
- **Polyak, B. T. (1964).** "Some methods of speeding up the convergence of iteration methods." — the
  **heavy-ball momentum** method (§4).
- **Sutskever, I., Martens, J., Dahl, G. & Hinton, G. (2013).** "On the importance of initialization and
  momentum in deep learning." *ICML*. — **Nesterov momentum** in deep learning and its interaction with
  init (§4). Free at <https://proceedings.mlr.press/v28/sutskever13.html>.
- **Bottou, L. (2010).** "Large-Scale Machine Learning with Stochastic Gradient Descent." *COMPSTAT*. —
  the case for SGD at scale (§2).
- **Dauphin, Y. et al. (2014).** "Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization." *NeurIPS*. — **saddles, not local minima, are the obstacle**
  (§8, Experiment 5). Free at <https://arxiv.org/abs/1406.2572>.
- **Smith, L. N. (2017).** "Cyclical Learning Rates for Training Neural Networks." *WACV*, and "Super-
  Convergence" (2018). — the **one-cycle** schedule and LR range test (§3, §7). Free at
  <https://arxiv.org/abs/1506.01186>.
- **Goyal, P. et al. (2017).** "Accurate, Large Minibatch SGD." *arXiv*. — **learning-rate warmup** for
  large-batch training (§7). Free at <https://arxiv.org/abs/1706.02677>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.optim`](https://github.com/pytorch/pytorch/tree/main/torch/optim) | `SGD` (with momentum/nesterov), `Adam`, `AdamW`, `RMSprop`, `Adagrad`; our Adam is verified against `torch.optim.Adam` |
| [`torch.optim.lr_scheduler`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py) | `CosineAnnealingLR`, `OneCycleLR`, `LinearLR` (warmup), `StepLR` (§7) |
| [timm optimizers](https://github.com/huggingface/pytorch-image-models/tree/main/timm/optim) | a broad zoo (LAMB, Lion, AdaBelief, ...) with clean implementations |

---

## Deferred to later chapters

- **Backprop — the gradients optimizers consume** → [07.02](../02-backpropagation/)
- **Gradient descent / momentum / Adam foundations** → [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/)
- **Hyperparameter search over the learning rate** → [05.05](../../05-model-evaluation/05-hyperparameter-optimization/)
- **Normalization — stabilizes the landscape optimizers descend** → [07.07](../07-normalization/)
- **Large-batch / distributed training and LR scaling** → [19.xx MLOps]
