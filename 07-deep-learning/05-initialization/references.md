# 07.05 — References: Weight Initialization

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Symmetry breaking | Goodfellow et al. §8.4 |
| §3-§5 | Variance preservation, Glorot | Glorot & Bengio (2010) |
| §6 | He init (ReLU) | He et al. (2015) |
| §8 | Orthogonal init | Saxe et al. (2014) |
| §8 | LSUV | Mishkin & Matas (2016) |
| §9 | Init + normalization / residuals | Ioffe & Szegedy (2015); Zhang et al. (2019, Fixup) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. §8.4 "Parameter Initialization Strategies" covers symmetry
breaking (§2), the variance heuristics (§3-§6), and the practical guidance of this chapter.

---

## Papers

- **Glorot, X. & Bengio, Y. (2010).** "Understanding the difficulty of training deep feedforward neural
  networks." *AISTATS*. — **Xavier/Glorot initialization** (§3-§5): the variance-preservation
  derivation for symmetric activations, and the demonstration that it fixes deep-tanh training. Free at
  <https://proceedings.mlr.press/v9/glorot10a.html>.
- **He, K., Zhang, X., Ren, S. & Sun, J. (2015).** "Delving Deep into Rectifiers: Surpassing
  Human-Level Performance on ImageNet Classification." *ICCV*. — **He/Kaiming initialization** (§6): the
  factor-of-2 correction for ReLU that Experiment 3 confirms. Free at <https://arxiv.org/abs/1502.01852>.
- **Saxe, A. M., McClelland, J. L. & Ganguli, S. (2014).** "Exact solutions to the nonlinear dynamics
  of learning in deep linear neural networks." *ICLR*. — **orthogonal initialization** (§8) and its
  norm-preserving dynamics. Free at <https://arxiv.org/abs/1312.6120>.
- **Mishkin, D. & Matas, J. (2016).** "All you need is a good init." *ICLR*. — **LSUV** (§8): iterative
  data-driven unit-variance initialization. Free at <https://arxiv.org/abs/1511.06422>.
- **Zhang, H., Dauphin, Y. N. & Ma, T. (2019).** "Fixup Initialization: Residual Learning Without
  Normalization." *ICLR*. — **init that lets deep residual nets train without normalization** (§9).
  Free at <https://arxiv.org/abs/1901.09321>.
- **Sutskever, I., Martens, J., Dahl, G. & Hinton, G. (2013).** "On the importance of initialization and
  momentum in deep learning." *ICML*. — the joint role of init and momentum (§9, [07.06](../06-optimizers/)).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.init`](https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py) | `kaiming_normal_`, `xavier_normal_`, `orthogonal_`, `calculate_gain`; the He/Glorot scales verified against here |
| [Keras initializers](https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py) | `GlorotNormal`, `HeNormal`, `Orthogonal` and their `fan_in`/`fan_out` logic |

---

## Deferred to later chapters

- **Vanishing/exploding gradients — the problem init addresses** → [07.02 §8](../02-backpropagation/)
- **Activations — He/Glorot are matched to the activation** → [07.03](../03-activations/)
- **Normalization — the runtime fix that makes init less critical** → [07.07](../07-normalization/)
- **Optimizers — init interacts with momentum and learning rate** → [07.06](../06-optimizers/)
- **Residual connections — the architectural fix for gradient flow** → [08.xx CNNs / 11.xx transformers]
