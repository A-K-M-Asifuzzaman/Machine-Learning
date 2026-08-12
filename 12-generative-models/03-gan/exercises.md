# 12.03 — Exercises: Generative Adversarial Networks

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the GAN minimax objective and explain the roles of the generator and discriminator.

**D2.** Derive the optimal discriminator $D^*(x) = \frac{p_{\text{data}}}{p_{\text{data}} +
p_{\text{gen}}}$ for a fixed generator.

**D3.** Substitute $D^*$ into the objective and show it equals $2\,\mathrm{JS}(p_{\text{data}}\Vert
p_{\text{gen}}) - \log 4$.

**D4.** Explain why JS divergence gives zero gradient when the supports are disjoint, and why this
causes vanishing gradients early in training.

**D5.** Explain the Wasserstein-1 distance and why it provides a useful gradient for disjoint supports.

**D6.** State the WGAN objective and the 1-Lipschitz constraint; describe weight clipping vs the WGAN-GP
gradient penalty.

**D7.** Explain the non-saturating generator loss and why it is preferred over the minimax form early in
training.

**D8.** Explain mode collapse mechanistically: why the objective doesn't penalize dropping modes.

**D9.** Compare GANs, VAEs, and diffusion on likelihood, encoder, sample sharpness, and stability.

**D10.** Explain StyleGAN's style-based generator (AdaIN injection) at a high level.

---

## Tier 2 — Implementation

**I1.** Implement a 1-D GAN and reproduce Experiment 1 (matches a Gaussian, D → 0.5).

**I2.** Numerically verify the optimal-discriminator / JS-divergence identity (Experiment 2).

**I3.** Reproduce Experiment 3: compare JS and Wasserstein for separated distributions.

**I4.** Reproduce Experiment 4: induce mode collapse on a bimodal target.

**I5.** Implement WGAN with weight clipping and compare stability to the vanilla GAN.

**I6.** Implement WGAN-GP (gradient penalty) and show it fixes vanishing gradients.

**I7.** Train a DCGAN on MNIST and inspect samples and mode coverage.

**I8.** Add minibatch discrimination and measure its effect on mode collapse.

**I9.** Implement a conditional GAN (condition on a label) and generate per class.

**I10.** *(Metric.)* Compute a GAN evaluation metric (FID or a 2-D proxy) and track it during training.

---

## Tier 3 — Interview

**Q1.** How does a GAN work?

**Q2.** What does the GAN objective optimize (in divergence terms)?

**Q3.** What is the optimal discriminator?

**Q4.** Why are GANs hard to train?

**Q5.** What problem does WGAN solve, and how?

**Q6.** What is the Lipschitz constraint and how is it enforced?

**Q7.** What is mode collapse and how do you mitigate it?

**Q8.** Why are GAN samples sharp but VAE samples blurry?

**Q9.** What did StyleGAN contribute?

**Q10.** Why has diffusion largely replaced GANs for image generation?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Write the minimax objective and explain the game
- [ ] Derive the optimal discriminator and the JS connection
- [ ] Explain the vanishing-gradient problem
- [ ] Explain WGAN and the Lipschitz constraint
- [ ] Explain and induce mode collapse
- [ ] Compare GANs to VAEs and diffusion
- [ ] Describe the DCGAN → StyleGAN progression
