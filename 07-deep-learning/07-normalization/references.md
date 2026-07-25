# 07.07 — References: Normalization

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2, §4 | Batch normalization | Ioffe & Szegedy (2015) |
| §3 | Why BN works (landscape) | Santurkar et al. (2018) |
| §6 | Layer normalization | Ba, Kiros & Hinton (2016) |
| §7 | Instance / group norm | Ulyanov et al. (2016); Wu & He (2018) |
| §8 | Pre-norm vs post-norm | Xiong et al. (2020) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. §8.7.1 covers batch normalization and the practical training
benefits (§2-§3).

---

## Papers

- **Ioffe, S. & Szegedy, C. (2015).** "Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift." *ICML*. — **the batch-norm paper** (§2, §4): the forward pass,
  the learnable $\gamma/\beta$, running statistics, and the (now-revised) covariate-shift motivation.
  Free at <https://arxiv.org/abs/1502.03167>.
- **Santurkar, S., Tsipras, D., Ilyas, A. & Madry, A. (2018).** "How Does Batch Normalization Help
  Optimization?" *NeurIPS*. — **the paper that debunked "internal covariate shift"** and showed the
  real mechanism is a smoother loss landscape (§3). Read this. Free at
  <https://arxiv.org/abs/1805.11604>.
- **Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016).** "Layer Normalization." *arXiv:1607.06450*. —
  **layer norm** (§6): per-example normalization over features, batch-independent. Free at
  <https://arxiv.org/abs/1607.06450>.
- **Wu, Y. & He, K. (2018).** "Group Normalization." *ECCV*. — **group norm** (§7): batch-independent
  normalization that matches BN at small batch sizes. Free at <https://arxiv.org/abs/1803.08494>.
- **Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2016).** "Instance Normalization: The Missing Ingredient
  for Fast Stylization." *arXiv*. — **instance norm** (§7). Free at <https://arxiv.org/abs/1607.08022>.
- **Xiong, R. et al. (2020).** "On Layer Normalization in the Transformer Architecture." *ICML*. — the
  **pre-norm vs post-norm** analysis (§8): why pre-norm trains more stably. Free at
  <https://arxiv.org/abs/2002.04745>.
- **Bjorck, N. et al. (2018).** "Understanding Batch Normalization." *NeurIPS*. — complementary analysis
  of why BN enables larger learning rates (§3).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.BatchNorm1d/2d`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py) | forward, running-stats update, train/eval switch; verified against here |
| [`torch.nn.LayerNorm`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py) | per-example feature normalization; verified against here |
| [`torch.nn.GroupNorm`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py) | group norm (§7) |
| [CS231n — batch norm backward](https://cs231n.github.io/) | the clearest derivation of the BN backward pass (§2) |

---

## Deferred to later chapters

- **Initialization — the static counterpart normalization makes less critical** → [07.05](../05-initialization/)
- **Optimizers — the landscape normalization smooths** → [07.06](../06-optimizers/)
- **Regularization — BN's mild regularizing effect and dropout** → [07.08](../08-regularization/)
- **Layer norm in Transformers, and pre-norm/post-norm** → [11.xx transformers]
- **Residual connections — the other gradient-flow fix** → [08.xx CNNs]
