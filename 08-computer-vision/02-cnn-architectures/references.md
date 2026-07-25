# 08.02 — References: CNN Architectures

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | LeNet, AlexNet — the template | LeCun et al. (1998); Krizhevsky et al. (2012) |
| §3 | VGG, small filters | Simonyan & Zisserman (2015) |
| §4 | ResNet, degradation, residuals | He et al. (2016) |
| §5 | Bottleneck blocks | He et al. (2016) §4 |
| §6 | Depthwise-separable convolution | Howard et al. (2017); Chollet (2017) |
| §7 | Global average pooling | Lin et al. (2014) |
| §8 | Inception, DenseNet, EfficientNet, ConvNeXt | Szegedy (2015); Huang (2017); Tan & Le (2019); Liu et al. (2022) |

---

## The landmark papers, one per architecture

- **LeCun, Y. et al. (1998).** "Gradient-Based Learning Applied to Document Recognition" (**LeNet-5**).
  *Proc. IEEE*. — the conv+pool+FC template. <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
- **Krizhevsky, A., Sutskever, I. & Hinton, G. (2012).** "ImageNet Classification with Deep CNNs"
  (**AlexNet**). *NeurIPS*. — ReLU + dropout + GPUs; started the era.
  <https://papers.nips.cc/paper/4824>.
- **Simonyan, K. & Zisserman, A. (2015).** "Very Deep Convolutional Networks for Large-Scale Image
  Recognition" (**VGG**). *ICLR*. — only $3\times3$ convs, go deep (§3).
  <https://arxiv.org/abs/1409.1556>.
- **Szegedy, C. et al. (2015).** "Going Deeper with Convolutions" (**GoogLeNet/Inception**). *CVPR*. —
  parallel multi-scale blocks + $1\times1$ bottlenecks (§8). <https://arxiv.org/abs/1409.4842>.
- **He, K., Zhang, X., Ren, S. & Sun, J. (2016).** "Deep Residual Learning for Image Recognition"
  (**ResNet**). *CVPR*. — the degradation problem and the residual fix (§4-§5); bottleneck blocks.
  <https://arxiv.org/abs/1512.03385>.
- **Huang, G. et al. (2017).** "Densely Connected Convolutional Networks" (**DenseNet**). *CVPR*. —
  connect every layer to every later one (§8). <https://arxiv.org/abs/1608.06993>.
- **Howard, A. et al. (2017).** "MobileNets: Efficient CNNs for Mobile Vision." — **depthwise-separable
  convolution** for efficiency (§6). <https://arxiv.org/abs/1704.04861>.
- **Chollet, F. (2017).** "Xception: Deep Learning with Depthwise Separable Convolutions." *CVPR*. —
  the depthwise-separable idea taken to the extreme (§6). <https://arxiv.org/abs/1610.02357>.
- **Tan, M. & Le, Q. (2019).** "EfficientNet: Rethinking Model Scaling for CNNs." *ICML*. — **compound
  scaling** of depth/width/resolution (§8). <https://arxiv.org/abs/1905.11946>.
- **Liu, Z. et al. (2022).** "A ConvNet for the 2020s" (**ConvNeXt**). *CVPR*. — modernizing a ResNet
  with transformer-era tricks (§8). <https://arxiv.org/abs/2201.03545>.

---

## Supporting

- **Lin, M., Chen, Q. & Yan, S. (2014).** "Network In Network." *ICLR*. — introduced **global average
  pooling** and the $1\times1$ conv as a channel-mixing layer (§7). <https://arxiv.org/abs/1312.4400>.
- **He, K. et al. (2016).** "Identity Mappings in Deep Residual Networks." *ECCV*. — the pre-activation
  ResNet and why the *clean* identity path matters for gradient flow (§4).
  <https://arxiv.org/abs/1603.05027>.
- **Ioffe, S. & Szegedy, C. (2015).** "Batch Normalization." *ICML*. — the normalization that keeps a
  ResNet's forward activations bounded (§4). <https://arxiv.org/abs/1502.03167>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [torchvision models](https://github.com/pytorch/vision/tree/main/torchvision/models) | reference ResNet / VGG / DenseNet / EfficientNet / ConvNeXt code |
| [`timm`](https://github.com/huggingface/pytorch-image-models) | the definitive collection of CNN (and ViT) backbones + training recipes |
| [`F.conv2d(..., groups=C)`](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html) | grouped/depthwise conv this chapter verifies against |

---

## Deferred to later chapters

- **The convolution primitive** → [08.01](../01-convolution/)
- **Transfer learning on these backbones** → [08.03](../03-transfer-learning/)
- **Detection/segmentation heads** → [08.04](../04-detection-and-segmentation/)
- **Vision transformers** → [08.05](../05-vision-transformers/)
- **Why residuals work — backprop and initialization** → [07.02](../../07-deep-learning/02-backpropagation/), [07.05](../../07-deep-learning/05-initialization/)
- **BatchNorm in depth** → [07.07](../../07-deep-learning/07-normalization/)
