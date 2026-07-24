# 07.04 — References: Loss Functions

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Losses as negative log-likelihood | Bishop §4-§5; Goodfellow et al. §5.5, §6.2 |
| §3 | Regression losses | ESL §10.6; Huber (1964) |
| §4-§5 | Cross-entropy, softmax gradient | Goodfellow et al. §6.2.2 |
| §6 | Cross-entropy vs MSE | Nielsen (2015) Ch. 3 |
| §7 | Numerical stability | Goodfellow et al. §4.1 |
| §8 | Focal loss, label smoothing | Lin et al. (2017); Szegedy et al. (2016) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. **§6.2 "Gradient-Based Learning"** is the reference for output
units and their matched losses (§2, §4-§5); §5.5 frames maximum likelihood as the origin of losses
(§2); §4.1 covers numerical stability / log-sum-exp (§7).

**Nielsen, M. (2015). *Neural Networks and Deep Learning*, Chapter 3.** — free at
<http://neuralnetworksanddeeplearning.com/chap3.html>. **The clearest exposition of why cross-entropy
beats MSE for classification** (§6): the learning-slowdown argument that Experiment 3 reproduces. Read
this for §6.

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** Chapters 4-5 derive the losses from
the corresponding output distributions (Bernoulli, categorical, Gaussian) — the NLL view of §2.

---

## Papers

- **Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017).** "Focal Loss for Dense Object
  Detection." *ICCV*. — **focal loss** (§8): cross-entropy modulated by $(1-\hat p_t)^\gamma$ for
  extreme foreground-background imbalance. Free at <https://arxiv.org/abs/1708.02002>.
- **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. & Wojna, Z. (2016).** "Rethinking the Inception
  Architecture for Computer Vision." *CVPR*. — introduces **label smoothing** (§8). Free at
  <https://arxiv.org/abs/1512.00567>.
- **Müller, R., Kornblith, S. & Hinton, G. (2019).** "When Does Label Smoothing Help?" *NeurIPS*. — a
  careful study of label smoothing's effect on accuracy and calibration (§8). Free at
  <https://arxiv.org/abs/1906.02629>.
- **Huber, P. J. (1964).** "Robust estimation of a location parameter." *Annals of Mathematical
  Statistics* 35(1). — the **Huber loss** (§3).
- **Hinton, G., Vinyals, O. & Dean, J. (2015).** "Distilling the Knowledge in a Neural Network."
  *arXiv*. — **KL-divergence loss** with soft targets (§8). Free at <https://arxiv.org/abs/1503.02531>.
- **Gutmann, M. & Hyvärinen, A. (2010).** "Noise-contrastive estimation." *AISTATS*. — the basis of
  contrastive losses (§8), used across modern representation learning.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.functional`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py) | `mse_loss`, `l1_loss`, `huber_loss`, `binary_cross_entropy_with_logits`, `cross_entropy` — verified against here; note the fused, stable with-logits forms |
| [`torch.nn.CrossEntropyLoss`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py) | the combined log-softmax + NLL, with `label_smoothing` and class `weight` support |
| [`sklearn.metrics.log_loss`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py) | cross-entropy as a metric (the evaluation counterpart) |

---

## Deferred to later chapters

- **Output activations (sigmoid, softmax) these losses pair with** → [07.03](../03-activations/)
- **Backprop — how the loss gradient starts the backward pass** → [07.02](../02-backpropagation/)
- **Classification & regression metrics (the evaluation counterparts)** → [05.02](../../05-model-evaluation/02-regression-metrics/)–[05.03](../../05-model-evaluation/03-classification-metrics/)
- **Calibration — where label smoothing and proper scoring meet** → [05.06](../../05-model-evaluation/06-calibration/)
- **Contrastive / triplet losses for representation learning** → [11.xx]
- **ELBO / variational losses for generative models** → [12.xx]
