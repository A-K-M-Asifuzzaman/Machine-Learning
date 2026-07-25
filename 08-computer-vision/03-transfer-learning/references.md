# 08.03 — References: Transfer Learning

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | Feature extractor / head split, feature hierarchy | Zeiler & Fergus (2014); Yosinski et al. (2014) |
| §3 | Transferability, related vs unrelated source | Yosinski et al. (2014); Kornblith et al. (2019) |
| §4-§5 | Feature extraction vs fine-tuning, data efficiency | Donahue et al. (2014); Razavian et al. (2014) |
| §6 | Catastrophic forgetting | McCloskey & Cohen (1989); Kirkpatrick et al. (2017) |
| §7 | Fine-tuning recipe, discriminative LR | Howard & Ruder (2018) |
| §8 | Domain shift / adaptation | Ganin & Lempitsky (2015); Ben-David et al. (2010) |

---

## The transferability papers

- **Yosinski, J., Clune, J., Bengio, Y. & Lipson, H. (2014).** "How transferable are features in deep
  neural networks?" *NeurIPS*. — the definitive study: early layers are general, late layers are
  specific, and transferability decreases with source-target distance (§2-§3). Free at
  <https://arxiv.org/abs/1411.1792>.
- **Zeiler, M. & Fergus, R. (2014).** "Visualizing and Understanding Convolutional Networks." *ECCV*.
  — visualizes the edge → texture → part hierarchy that makes early features generic (§2).
  <https://arxiv.org/abs/1311.2901>.
- **Kornblith, S., Shlens, J. & Le, Q. (2019).** "Do Better ImageNet Models Transfer Better?" *CVPR*.
  — better source models generally transfer better, and fine-tuning beats feature extraction with
  enough data (§3-§5). <https://arxiv.org/abs/1805.08974>.

---

## Feature extraction and fine-tuning

- **Donahue, J. et al. (2014).** "DeCAF: A Deep Convolutional Activation Feature for Generic Visual
  Recognition." *ICML*. — pretrained features + a linear classifier beat hand-crafted features on many
  tasks (§5). <https://arxiv.org/abs/1310.1531>.
- **Razavian, A. S. et al. (2014).** "CNN Features off-the-shelf: an Astounding Baseline for
  Recognition." *CVPR Workshop*. — frozen CNN features as a strong general baseline (§5).
  <https://arxiv.org/abs/1403.6382>.
- **Howard, J. & Ruder, S. (2018).** "Universal Language Model Fine-tuning (ULMFiT)." *ACL*. —
  discriminative learning rates, gradual unfreezing, and the modern fine-tuning recipe (§6-§7), stated
  for NLP but general. <https://arxiv.org/abs/1801.06146>.

---

## Catastrophic forgetting and domain shift

- **McCloskey, M. & Cohen, N. (1989).** "Catastrophic Interference in Connectionist Networks." — the
  original description of forgetting (§6).
- **Kirkpatrick, J. et al. (2017).** "Overcoming Catastrophic Forgetting in Neural Networks" (EWC).
  *PNAS*. — a method to retain old-task knowledge while learning new tasks (§6).
  <https://arxiv.org/abs/1612.00796>.
- **Ganin, Y. & Lempitsky, V. (2015).** "Unsupervised Domain Adaptation by Backpropagation." *ICML*. —
  adapting features across a domain shift (§8). <https://arxiv.org/abs/1409.7495>.
- **Ben-David, S. et al. (2010).** "A Theory of Learning from Different Domains." *Machine Learning*. —
  the theoretical bound on transfer error under domain shift (§8).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [PyTorch transfer-learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | feature extraction vs fine-tuning, in code |
| [`torchvision.models`](https://github.com/pytorch/vision/tree/main/torchvision/models) | pretrained backbones to transfer from |
| [`timm`](https://github.com/huggingface/pytorch-image-models) | pretrained models + fine-tuning recipes |
| [Hugging Face `Trainer`](https://huggingface.co/docs/transformers/training) | fine-tuning APIs (the same paradigm for NLP) |

---

## Deferred to later chapters

- **The backbones being transferred** → [08.02](../02-cnn-architectures/)
- **Self-supervised pretraining (DINO, MAE, CLIP)** → [08.05](../05-vision-transformers/)
- **Pretrain-then-fine-tune for language / LLMs** → [11.02](../../11-transformers-and-llms/02-pretraining/)
- **Learning-rate schedules and warmup** → [07.06](../../07-deep-learning/06-optimizers/)
- **Distribution shift and robustness** → [18.xx]
