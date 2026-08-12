# 12.04 — Exercises: Diffusion Models

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the forward process $q(x_t\mid x_{t-1})$ and explain why it needs no learned parameters.

**D2.** Derive the closed form $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
by composing the per-step Gaussians.

**D3.** Explain why $x_T \to \mathcal{N}(0,I)$ and why that is essential for generation.

**D4.** Derive the DDPM training objective (predict the noise) from the variational bound, and explain
the reweighting to plain MSE.

**D5.** Write the reverse sampling update and explain each term.

**D6.** Derive the identity $\epsilon^*(x_t) = -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$ linking
noise prediction and score matching.

**D7.** Explain the continuous-time (SDE) view and why reverse-time solution is generation.

**D8.** Explain DDIM: the deterministic (ODE) reinterpretation and why it allows fewer steps.

**D9.** Derive classifier-free guidance and explain the diversity/adherence trade-off of the guidance
scale.

**D10.** Explain latent diffusion and why running in a VAE latent is cheaper than pixels.

---

## Tier 2 — Implementation

**I1.** Verify the forward closed form against iterative noising (Experiment 1).

**I2.** Reproduce Experiment 2: track the SNR to near-zero.

**I3.** Implement a diffusion model (ε-predictor + DDPM sampling) and reproduce Experiment 3 (generates,
covers both modes).

**I4.** Verify the noise/score identity on Gaussian data (Experiment 4).

**I5.** Implement DDIM sampling and reproduce Experiment 5 (fewer steps).

**I6.** Train a diffusion model on MNIST/CIFAR with a small U-Net and sample images.

**I7.** Implement a cosine noise schedule and compare to linear.

**I8.** Implement classifier-free guidance (conditional + unconditional) and sweep the guidance scale.

**I9.** Implement a faster sampler (DPM-Solver or a few-step distillation) and compare quality/steps.

**I10.** *(Latent.)* Train a small VAE, then run diffusion in its latent space (latent diffusion) and
compare cost to pixel diffusion.

---

## Tier 3 — Interview

**Q1.** How does a diffusion model generate data?

**Q2.** What is the forward process and why does it have a closed form?

**Q3.** What does the network learn to predict, and what is the loss?

**Q4.** Why doesn't diffusion suffer mode collapse like GANs?

**Q5.** How are diffusion models related to score matching?

**Q6.** What is DDIM and why is it faster?

**Q7.** What is classifier-free guidance?

**Q8.** What is latent diffusion (Stable Diffusion)?

**Q9.** Why did diffusion replace GANs for image generation?

**Q10.** What is the main drawback of diffusion, and how is it addressed?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the forward closed form and the reparameterization
- [ ] Explain why the forward process must reach pure noise
- [ ] Derive and implement the DDPM noise-prediction objective
- [ ] Explain why diffusion is mode-covering
- [ ] Explain the noise/score equivalence
- [ ] Implement DDIM and explain the speedup
- [ ] Describe guidance and latent diffusion
