# 12.02 — References: Variational Autoencoders

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§3 | VAE, ELBO, reparameterization | Kingma & Welling (2013) |
| §3 | Stochastic backprop / reparameterization | Rezende et al. (2014) |
| §5 | β-VAE, disentanglement | Higgins et al. (2017) |
| §5 | Posterior collapse | Bowman et al. (2016); He et al. (2019) |
| §6 | VQ-VAE (discrete latents) | van den Oord et al. (2017) |
| §6 | Latent diffusion (VAE + diffusion) | Rombach et al. (2022) |

---

## Foundational

- **Kingma, D. & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR 2014*. — the **VAE**: the
  ELBO, the reparameterization trick, and the amortized inference network (§2-§3).
  <https://arxiv.org/abs/1312.6114>.
- **Rezende, D., Mohamed, S. & Wierstra, D. (2014).** "Stochastic Backpropagation and Approximate
  Inference in Deep Generative Models." *ICML*. — concurrent derivation of the reparameterization
  gradient (§3). <https://arxiv.org/abs/1401.4082>.

## Extensions

- **Higgins, I. et al. (2017).** "β-VAE: Learning Basic Visual Concepts with a Constrained Variational
  Framework." *ICLR*. — the **β** weighting and disentanglement (§5).
  <https://openreview.net/forum?id=Sy2fzU9gl>.
- **Bowman, S. et al. (2016).** "Generating Sentences from a Continuous Space." *CoNLL*. — **posterior
  collapse** in text VAEs and KL annealing (§5). <https://arxiv.org/abs/1511.06349>.
- **He, J. et al. (2019).** "Lagging Inference Networks and Posterior Collapse in Variational
  Autoencoders." *ICLR*. — analysis and a fix for collapse (§5). <https://arxiv.org/abs/1901.05534>.
- **van den Oord, A. et al. (2017).** "Neural Discrete Representation Learning" (**VQ-VAE**). *NeurIPS*.
  — discrete latents for sharper samples (§6). <https://arxiv.org/abs/1711.00937>.
- **Rombach, R. et al. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*.
  — diffusion in a VAE's latent space (§6). <https://arxiv.org/abs/2112.10752>.

## Tutorials

- **Doersch, C. (2016).** "Tutorial on Variational Autoencoders." — the standard gentle derivation.
  <https://arxiv.org/abs/1606.05908>.
- **Kingma, D. & Welling, M. (2019).** "An Introduction to Variational Autoencoders." *Foundations and
  Trends in ML*. <https://arxiv.org/abs/1906.02691>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [PyTorch VAE example](https://github.com/pytorch/examples/tree/main/vae) | a minimal reference VAE |
| [`PyTorch-VAE`](https://github.com/AntixK/PyTorch-VAE) | many VAE variants (β-VAE, VQ-VAE, …) |
| [Stable Diffusion VAE](https://github.com/CompVis/latent-diffusion) | the VAE used in latent diffusion (§6) |

---

## Deferred to later chapters

- **Autoencoders** → [12.01](../01-autoencoders/)
- **GANs** → [12.03](../03-gan/)
- **Diffusion & latent diffusion** → [12.04](../04-diffusion/)
- **Normalizing flows (exact likelihood)** → [12.05](../05-flows-and-autoregressive/)
- **KL divergence / information theory** → [00.05](../../00-mathematical-foundations/05-information-theory/)
