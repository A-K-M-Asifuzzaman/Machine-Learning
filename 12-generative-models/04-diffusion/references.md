# 12.04 — References: Diffusion Models

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Diffusion probabilistic models | Sohl-Dickstein et al. (2015) |
| §2-§4 | DDPM, noise-prediction objective | Ho et al. (2020) |
| §5 | Score-based models, SDE | Song & Ermon (2019); Song et al. (2021) |
| §6 | DDIM | Song et al. (2021) |
| §7 | Classifier-free guidance | Ho & Salimans (2022) |
| §7 | Latent diffusion | Rombach et al. (2022) |

---

## Foundational

- **Sohl-Dickstein, J. et al. (2015).** "Deep Unsupervised Learning using Nonequilibrium
  Thermodynamics." *ICML*. — the original diffusion probabilistic model (§1).
  <https://arxiv.org/abs/1503.03585>.
- **Ho, J., Jain, A. & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models" (**DDPM**).
  *NeurIPS*. — the closed form, the simple noise-prediction MSE objective, high-quality samples (§2-§4).
  <https://arxiv.org/abs/2006.11239>.

## Score-based view

- **Song, Y. & Ermon, S. (2019).** "Generative Modeling by Estimating Gradients of the Data
  Distribution" (**score matching**). *NeurIPS*. — the score-based formulation (§5).
  <https://arxiv.org/abs/1907.05600>.
- **Song, Y. et al. (2021).** "Score-Based Generative Modeling through Stochastic Differential
  Equations." *ICLR*. — the unifying SDE view (§5). <https://arxiv.org/abs/2011.13456>.

## Sampling and guidance

- **Song, J., Meng, C. & Ermon, S. (2021).** "Denoising Diffusion Implicit Models" (**DDIM**). *ICLR*. —
  deterministic, few-step sampling (§6). <https://arxiv.org/abs/2010.02502>.
- **Lu, C. et al. (2022).** "DPM-Solver." *NeurIPS*. — fast high-order ODE solvers (§6).
  <https://arxiv.org/abs/2206.00927>.
- **Song, Y. et al. (2023).** "Consistency Models." *ICML*. — one/few-step generation (§6).
  <https://arxiv.org/abs/2303.01469>.
- **Ho, J. & Salimans, T. (2022).** "Classifier-Free Diffusion Guidance." *NeurIPS Workshop*. — the
  guidance scale (§7). <https://arxiv.org/abs/2207.12598>.
- **Dhariwal, P. & Nichol, A. (2021).** "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*. —
  the result that diffusion surpasses GANs (§4). <https://arxiv.org/abs/2105.05233>.

## Latent & text-to-image

- **Rombach, R. et al. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models"
  (**Stable Diffusion**). *CVPR*. — diffusion in a VAE latent (§7). <https://arxiv.org/abs/2112.10752>.
- **Ramesh, A. et al. (2022).** "Hierarchical Text-Conditional Image Generation with CLIP Latents"
  (**DALL·E 2**). <https://arxiv.org/abs/2204.06125>.
- **Saharia, C. et al. (2022).** "Photorealistic Text-to-Image Diffusion Models" (**Imagen**). *NeurIPS*.
  <https://arxiv.org/abs/2205.11487>.

---

## Reference implementations & tutorials

| Source | What to look at |
|---|---|
| [`diffusers`](https://github.com/huggingface/diffusers) | production diffusion pipelines and schedulers |
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | a clean DDPM implementation |
| [Karpathy / minDiffusion-style tutorials] | small from-scratch diffusion |
| [Lilian Weng, "What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | the standard derivation blog |

---

## Deferred to later chapters

- **GANs** → [12.03](../03-gan/)
- **VAEs (latent diffusion)** → [12.02](../02-vae/)
- **Denoising autoencoders** → [12.01](../01-autoencoders/)
- **U-Net / transformer backbones** → [08.04](../../08-computer-vision/04-detection-and-segmentation/), [11.01](../../11-transformers-and-llms/01-transformer/)
