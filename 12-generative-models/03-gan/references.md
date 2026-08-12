# 12.03 — References: Generative Adversarial Networks

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | GAN, minimax, JS divergence | Goodfellow et al. (2014) |
| §3-§4 | Vanishing gradients, WGAN | Arjovsky et al. (2017) |
| §4 | WGAN-GP | Gulrajani et al. (2017) |
| §5 | Mode collapse, training | Salimans et al. (2016); Metz et al. (2017) |
| §6 | DCGAN, StyleGAN, BigGAN | Radford et al. (2016); Karras et al. (2019); Brock et al. (2019) |

---

## Foundational

- **Goodfellow, I. et al. (2014).** "Generative Adversarial Networks." *NeurIPS*. — the GAN, the minimax
  game, the optimal-discriminator / JS-divergence theorem (§1-§2). <https://arxiv.org/abs/1406.2661>.
- **Arjovsky, M., Chintala, S. & Bottou, L. (2017).** "Wasserstein GAN." *ICML*. — the vanishing-gradient
  analysis and the Wasserstein objective (§3-§4). <https://arxiv.org/abs/1701.07875>.
- **Gulrajani, I. et al. (2017).** "Improved Training of Wasserstein GANs" (**WGAN-GP**). *NeurIPS*. —
  the gradient penalty (§4). <https://arxiv.org/abs/1704.00028>.

## Training and stability

- **Salimans, T. et al. (2016).** "Improved Techniques for Training GANs." *NeurIPS*. — minibatch
  discrimination, feature matching, the Inception Score (§5). <https://arxiv.org/abs/1606.03498>.
- **Metz, L. et al. (2017).** "Unrolled Generative Adversarial Networks." *ICLR*. — unrolling to reduce
  mode collapse (§5). <https://arxiv.org/abs/1611.02163>.
- **Heusel, M. et al. (2017).** "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
  Equilibrium" — the **FID** metric (§7 eval). <https://arxiv.org/abs/1706.08500>.

## Architectures

- **Radford, A., Metz, L. & Chintala, S. (2016).** "Unsupervised Representation Learning with Deep
  Convolutional GANs" (**DCGAN**). *ICLR*. (§6). <https://arxiv.org/abs/1511.06434>.
- **Karras, T. et al. (2019).** "A Style-Based Generator Architecture for GANs" (**StyleGAN**). *CVPR*.
  (§6). <https://arxiv.org/abs/1812.04948>.
- **Brock, A. et al. (2019).** "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
  (**BigGAN**). *ICLR*. (§6). <https://arxiv.org/abs/1809.11096>.
- **Isola, P. et al. (2017).** "Image-to-Image Translation with Conditional GANs" (**pix2pix**). *CVPR*.
  <https://arxiv.org/abs/1611.07004>.
- **Zhu, J.-Y. et al. (2017).** "Unpaired Image-to-Image Translation" (**CycleGAN**). *ICCV*.
  <https://arxiv.org/abs/1703.10593>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) | a full conv GAN |
| [StyleGAN2/3 (NVIDIA)](https://github.com/NVlabs/stylegan3) | the state-of-the-art GAN generator |
| [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) | many GAN variants (WGAN-GP, cGAN, CycleGAN, …) |

---

## Deferred to later chapters

- **VAEs** → [12.02](../02-vae/)
- **Diffusion (the successor)** → [12.04](../04-diffusion/)
- **Normalizing flows** → [12.05](../05-flows-and-autoregressive/)
- **KL/JS divergences** → [00.05](../../00-mathematical-foundations/05-information-theory/)
- **CNN architectures** → [08.02](../../08-computer-vision/02-cnn-architectures/)
