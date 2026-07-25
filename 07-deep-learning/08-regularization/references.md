# 07.08 — References: Regularization

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Capacity, overfitting | Goodfellow et al. Ch. 7; Zhang et al. (2017) |
| §2 | Weight decay, AdamW | Krogh & Hertz (1992); Loshchilov & Hutter (2019) |
| §3-§4 | Dropout | Srivastava et al. (2014); Hinton et al. (2012) |
| §5 | Early stopping | Prechelt (1998); Goodfellow et al. §7.8 |
| §6 | Data augmentation | Shorten & Khoshgoftaar (2019); Cubuk et al. (2019) |
| §7 | Mixup, stochastic depth | Zhang et al. (2018); Huang et al. (2016) |
| §8 | Implicit regularization, flat minima | Keskar et al. (2017) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. **Chapter 7 "Regularization for Deep Learning"** is the reference:
weight decay (§7.1), early stopping (§7.8), dropout (§7.12), augmentation (§7.4), and the bias-variance
framing of all of them.

---

## Papers

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014).** "Dropout: A
  Simple Way to Prevent Neural Networks from Overfitting." *JMLR* 15. — **the dropout paper** (§3-§4):
  co-adaptation, the ensemble view, and the inference scaling. Free at
  <https://jmlr.org/papers/v15/srivastava14a.html>.
- **Loshchilov, I. & Hutter, F. (2019).** "Decoupled Weight Decay Regularization." *ICLR*. — **AdamW**
  (§2): why weight decay must be decoupled from the adaptive update. Free at
  <https://arxiv.org/abs/1711.05101>.
- **Krogh, A. & Hertz, J. (1992).** "A Simple Weight Decay Can Improve Generalization." *NeurIPS*. — the
  classic analysis of **weight decay** (§2).
- **Zhang, C. et al. (2017).** "Understanding deep learning requires rethinking generalization." *ICLR*.
  — nets fit random labels (unbounded capacity, §1) yet generalize on real data — the puzzle motivating
  implicit regularization (§8). Free at <https://arxiv.org/abs/1611.03530>.
- **Keskar, N. S. et al. (2017).** "On Large-Batch Training for Deep Learning: Generalization Gap and
  Sharp Minima." *ICLR*. — **flat vs sharp minima** and SGD's implicit bias (§8). Free at
  <https://arxiv.org/abs/1609.04836>.
- **Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. (2018).** "mixup: Beyond Empirical Risk
  Minimization." *ICLR*. — **mixup** (§7). Free at <https://arxiv.org/abs/1710.09412>.
- **Huang, G. et al. (2016).** "Deep Networks with Stochastic Depth." *ECCV*. — **stochastic depth**
  (§7). Free at <https://arxiv.org/abs/1603.09382>.
- **Cubuk, E. D. et al. (2019).** "AutoAugment: Learning Augmentation Strategies from Data." *CVPR*, and
  RandAugment (2020). — learned **data augmentation** policies (§6). Free at
  <https://arxiv.org/abs/1805.09501>.
- **Prechelt, L. (1998).** "Early Stopping — But When?" In *Neural Networks: Tricks of the Trade*. — the
  practical guide to **early stopping** criteria (§5).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.Dropout`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/dropout.py) | inverted dropout, train/eval switch; verified against here |
| [`torch.optim.AdamW`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) | decoupled weight decay (§2) |
| [`torchvision.transforms`](https://github.com/pytorch/vision/tree/main/torchvision/transforms) | the standard data-augmentation library (§6) |
| [timm `Mixup`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py) | mixup / CutMix implementations (§7) |

---

## Deferred to later chapters

- **Bias-variance & overfitting — the theory regularization addresses** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **Ridge/Lasso — the linear-model roots of weight decay** → [03.02](../../03-supervised-learning/02-regularized-linear-models/)
- **Cross-validation — tuning the regularization strength** → [05.04](../../05-model-evaluation/04-cross-validation/)
- **Batch norm's regularizing effect** → [07.07](../07-normalization/)
- **Label smoothing and calibration** → [07.04 §8](../04-loss-functions/), [05.06](../../05-model-evaluation/06-calibration/)
- **Data augmentation for vision / NLP in depth** → [08.xx / 10.xx]
