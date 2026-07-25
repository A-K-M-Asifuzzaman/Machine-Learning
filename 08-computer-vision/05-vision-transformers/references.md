# 08.05 — References: Vision Transformers

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | ViT, patch embedding | Dosovitskiy et al. (2021) |
| §3-§4 | Permutation equivariance, positional embeddings | Vaswani et al. (2017); Dosovitskiy et al. (2021) |
| §5 | Inductive bias, data efficiency, distillation | Dosovitskiy et al. (2021); Touvron et al. (2021, DeiT) |
| §6 | Windowed attention | Liu et al. (2021, Swin) |
| §7 | Self-supervised vision | He et al. (2022, MAE); Caron et al. (2021, DINO); Radford et al. (2021, CLIP) |

---

## The core papers

- **Dosovitskiy, A. et al. (2021).** "An Image is Worth 16x16 Words: Transformers for Image Recognition
  at Scale" (**ViT**). *ICLR*. — the paper: patch embedding, `[CLS]` token, positional embeddings, and
  the finding that ViT beats CNNs only with very large pretraining data (§1-§5). Free at
  <https://arxiv.org/abs/2010.11929>.
- **Vaswani, A. et al. (2017).** "Attention Is All You Need." *NeurIPS*. — the transformer and the
  permutation-equivariance / positional-encoding argument (§3-§4). Free at
  <https://arxiv.org/abs/1706.03762>. (Full treatment in [11.01](../../11-transformers-llms/01-attention/).)

## Data efficiency and architecture

- **Touvron, H. et al. (2021).** "Training data-efficient image transformers & distillation through
  attention" (**DeiT**). *ICML*. — trains ViT on ImageNet alone via augmentation and distillation from a
  CNN teacher (§5). <https://arxiv.org/abs/2012.12877>.
- **Liu, Z. et al. (2021).** "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."
  *ICCV*. — windowed (linear-cost) attention and hierarchical features, making ViTs general backbones
  (§6). <https://arxiv.org/abs/2103.14030>.
- **Liu, Z. et al. (2022).** "A ConvNet for the 2020s" (**ConvNeXt**). *CVPR*. — a modernized CNN that
  matches ViT, isolating how much of ViT's gain was the training recipe (§5, §8).
  <https://arxiv.org/abs/2201.03545>.

## Self-supervised vision

- **He, K. et al. (2022).** "Masked Autoencoders Are Scalable Vision Learners" (**MAE**). *CVPR*. —
  mask 75% of patches and reconstruct; strong label-free pretraining (§7).
  <https://arxiv.org/abs/2111.06377>.
- **Caron, M. et al. (2021).** "Emerging Properties in Self-Supervised Vision Transformers" (**DINO**).
  *ICCV*. — self-distillation; attention maps segment objects with no labels (§7).
  <https://arxiv.org/abs/2104.14294>.
- **Radford, A. et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision"
  (**CLIP**). *ICML*. — image–text contrastive pretraining and zero-shot classification (§7).
  <https://arxiv.org/abs/2103.00020>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`timm`](https://github.com/huggingface/pytorch-image-models) | reference ViT / DeiT / Swin implementations and weights |
| [google-research/vision_transformer](https://github.com/google-research/vision_transformer) | the original ViT code |
| [facebookresearch/mae](https://github.com/facebookresearch/mae), [dino](https://github.com/facebookresearch/dino) | MAE and DINO code |
| [openai/CLIP](https://github.com/openai/CLIP) | CLIP code and pretrained models |

---

## Deferred to later chapters

- **Attention and the transformer block in full** → [11.01](../../11-transformers-llms/01-attention/), [11.02](../../11-transformers-llms/02-transformer-architecture/)
- **CNNs, the prior ViT drops** → [08.02](../02-cnn-architectures/)
- **Transfer and self-supervised pretraining** → [08.03](../03-transfer-learning/), [11.05](../../11-transformers-llms/05-pretraining-scaling/)
- **Text-to-image generation (CLIP-guided)** → [Part 12](../../12-generative-models/)
