# 08.01 — References: Convolution

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Locality + weight sharing as constraints on a dense layer | Goodfellow et al. Ch. 9; LeCun et al. (1998) |
| §2 | Kernels as feature detectors; learned edge filters | Zeiler & Fergus (2014); Krizhevsky et al. (2012) |
| §3 | The operation, stride/padding/dilation, output size | Dumoulin & Visin (2016) |
| §4 | im2col / convolution as GEMM | Chellapilla et al. (2006); cuDNN |
| §5 | Parameter economy, weight sharing | LeCun et al. (1998); Goodfellow et al. §9.2 |
| §6 | Receptive fields; dilated convolutions | Yu & Koltun (2016); Luo et al. (2016) |
| §7 | Translation equivariance | Goodfellow et al. §9.4; Cohen & Welling (2016) |
| §8 | Pooling and invariance | Goodfellow et al. §9.3; Springenberg et al. (2015) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — **Chapter 9 "Convolutional
Networks"** is the reference: the sparse-interaction / parameter-sharing / equivariance framing (§9.1-9.4),
pooling (§9.3), and the variants (§9.5). Free at <https://www.deeplearningbook.org/>.

---

## Foundational papers

- **LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998).** "Gradient-Based Learning Applied to
  Document Recognition." *Proc. IEEE*. — **LeNet-5**, the paper that established convolutional networks
  and weight sharing (§1, §5). Free at <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
- **Krizhevsky, A., Sutskever, I. & Hinton, G. (2012).** "ImageNet Classification with Deep
  Convolutional Neural Networks" (**AlexNet**). *NeurIPS*. — the result that made CNNs the default for
  vision; first-layer filters look like Gabor/edge detectors (§2). Free at
  <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>.
- **Zeiler, M. & Fergus, R. (2014).** "Visualizing and Understanding Convolutional Networks." *ECCV*.
  — shows what learned kernels detect layer by layer, confirming §2's "edges then textures then parts"
  hierarchy. Free at <https://arxiv.org/abs/1311.2901>.

---

## Mechanics and receptive fields

- **Dumoulin, V. & Visin, F. (2016).** "A Guide to Convolution Arithmetic for Deep Learning." — the
  definitive treatment of stride / padding / dilation and the output-size formula (§3), with figures.
  Free at <https://arxiv.org/abs/1603.07285>.
- **Chellapilla, K., Puri, S. & Simard, P. (2006).** "High Performance Convolutional Neural Networks
  for Document Processing." — the **im2col** GEMM formulation used in §4. Free at
  <https://inria.hal.science/inria-00112631/>.
- **Yu, F. & Koltun, V. (2016).** "Multi-Scale Context Aggregation by Dilated Convolutions." *ICLR*.
  — **dilated convolutions** and the exponential receptive-field growth of §6. Free at
  <https://arxiv.org/abs/1511.07122>.
- **Luo, W. et al. (2016).** "Understanding the Effective Receptive Field in Deep Convolutional Neural
  Networks." *NeurIPS*. — the *effective* receptive field is Gaussian and smaller than the theoretical
  one (§6). Free at <https://arxiv.org/abs/1701.04128>.

---

## Equivariance and pooling

- **Cohen, T. & Welling, M. (2016).** "Group Equivariant Convolutional Networks." *ICML*. — formalizes
  the equivariance of §7 and generalizes it to rotations/reflections. Free at
  <https://arxiv.org/abs/1602.07576>.
- **Springenberg, J. T. et al. (2015).** "Striving for Simplicity: The All Convolutional Net." *ICLR
  Workshop*. — replaces pooling with strided convolutions (§8, §10). Free at
  <https://arxiv.org/abs/1412.6806>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.functional.conv2d`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp) | the operation this chapter verifies against |
| [cuDNN convolution](https://docs.nvidia.com/deeplearning/cudnn/latest/) | im2col / Winograd / FFT algorithm selection (§4) |
| [CS231n conv notes](https://cs231n.github.io/convolutional-networks/) | im2col, receptive fields, arithmetic — the standard course reference |

---

## Deferred to later chapters

- **CNN architectures assembled from these blocks** → [08.02](../02-cnn-architectures/)
- **Transfer learning on the learned filters** → [08.03](../03-transfer-learning/)
- **Detection/segmentation heads (FPN, U-Net, dilated backbones)** → [08.04](../04-detection-and-segmentation/)
- **Vision transformers — dropping the convolutional prior** → [08.05](../05-vision-transformers/)
- **Backpropagation, the general algorithm** → [07.02](../../07-deep-learning/02-backpropagation/)
