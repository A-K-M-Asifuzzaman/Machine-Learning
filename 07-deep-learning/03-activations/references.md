# 07.03 — References: Activation Functions

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§3 | Sigmoid, tanh, saturation | LeCun et al. (1998, "Efficient BackProp") |
| §4 | ReLU | Nair & Hinton (2010); Glorot, Bordes & Bengio (2011) |
| §5 | Dying ReLU, Leaky/PReLU | Maas et al. (2013); He et al. (2015) |
| §6 | ELU, SELU | Clevert et al. (2016); Klambauer et al. (2017) |
| §7 | GELU, Swish | Hendrycks & Gimpel (2016); Ramachandran et al. (2017) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. §6.3 "Hidden Units" surveys activation functions and the case for
ReLU (§4); §6.2.2 covers output activations (softmax, §8).

**LeCun, Y., Bottou, L., Orr, G. & Müller, K.-R. (1998). "Efficient BackProp."** In *Neural Networks:
Tricks of the Trade*. — the classic on why zero-centered, non-saturating activations help (§2-§3).
Free at <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>.

---

## Papers

- **Nair, V. & Hinton, G. E. (2010).** "Rectified Linear Units Improve Restricted Boltzmann Machines."
  *ICML*. — **introduces ReLU** for neural networks (§4).
- **Glorot, X., Bordes, A. & Bengio, Y. (2011).** "Deep Sparse Rectifier Neural Networks." *AISTATS*.
  — **the case for ReLU in deep supervised networks**: non-saturation, sparsity, faster training (§4,
  Experiment 5). Free at <https://proceedings.mlr.press/v15/glorot11a.html>.
- **Maas, A. L., Hannun, A. Y. & Ng, A. Y. (2013).** "Rectifier Nonlinearities Improve Neural Network
  Acoustic Models." *ICML WDLASL*. — **Leaky ReLU** (§5).
- **He, K., Zhang, X., Ren, S. & Sun, J. (2015).** "Delving Deep into Rectifiers: Surpassing
  Human-Level Performance on ImageNet Classification." *ICCV*. — **PReLU** and the matching He
  initialization ([07.05](../05-initialization/)) (§5). Free at <https://arxiv.org/abs/1502.01852>.
- **Clevert, D.-A., Unterthiner, T. & Hochreiter, S. (2016).** "Fast and Accurate Deep Network Learning
  by Exponential Linear Units (ELUs)." *ICLR*. — **ELU** (§6). Free at <https://arxiv.org/abs/1511.07289>.
- **Klambauer, G., Unterthiner, T., Mayr, A. & Hochreiter, S. (2017).** "Self-Normalizing Neural
  Networks." *NeurIPS*. — **SELU** and self-normalization (§6). Free at
  <https://arxiv.org/abs/1706.02515>.
- **Hendrycks, D. & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*. —
  **GELU** (§7), the standard Transformer activation. Free at <https://arxiv.org/abs/1606.08415>.
- **Ramachandran, P., Zoph, B. & Le, Q. V. (2017).** "Searching for Activation Functions." *arXiv*. —
  **Swish/SiLU** (§7), found by architecture search. Free at <https://arxiv.org/abs/1710.05941>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.functional`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py) | `relu`, `leaky_relu`, `elu`, `gelu`, `silu`, `softmax`, `log_softmax` — verified against here |
| [`torch.nn.modules.activation`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py) | the module wrappers and PReLU's learnable parameter |
| [CS231n — activation functions notes](https://cs231n.github.io/neural-networks-1/#actfun) | the clearest survey with the dying-ReLU discussion |

---

## Deferred to later chapters

- **Backprop — where $\sigma'$ enters the gradient** → [07.02](../02-backpropagation/)
- **Loss functions & output activations (softmax + cross-entropy)** → [07.04](../04-loss-functions/)
- **Initialization — He/Glorot init matched to the activation** → [07.05](../05-initialization/)
- **Normalization — the other fix for gradient flow** → [07.07](../07-normalization/)
- **GELU/Swish in Transformers** → [11.xx transformers]
