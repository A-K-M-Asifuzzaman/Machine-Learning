# 12.02 — Exercises: Variational Autoencoders

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain the generative gap: why a plain autoencoder's latent space is not sampleable.

**D2.** Derive the ELBO from $\log p(x)$ using Jensen's inequality (or the KL-to-posterior form) and
identify the reconstruction and KL terms.

**D3.** Derive the closed-form KL between two diagonal Gaussians
$\mathrm{KL}(\mathcal{N}(\mu,\sigma^2)\Vert\mathcal{N}(0,I))$.

**D4.** Explain why $\mathbb{E}_{z\sim q}[\cdot]$ cannot be differentiated by backprop directly, and how
the reparameterization trick $z=\mu+\sigma\epsilon$ fixes it.

**D5.** Derive the score-function (REINFORCE) gradient estimator and explain why it has higher variance
than reparameterization.

**D6.** Show how the ELBO's KL term forces the aggregate posterior toward the prior, making sampling
valid.

**D7.** Write the β-VAE objective and explain how β trades reconstruction against disentanglement.

**D8.** Explain posterior collapse: the conditions (too-strong decoder, high β) and why the latent
becomes uninformative.

**D9.** Explain why VAE samples are blurry (the Gaussian likelihood / mode-covering objective).

**D10.** Compare VAEs, GANs, and diffusion on likelihood, encoder, and sample quality.

---

## Tier 2 — Implementation

**I1.** Verify the Gaussian KL closed form against Monte Carlo (Experiment 1).

**I2.** Implement both gradient estimators and reproduce Experiment 2's variance comparison.

**I3.** Implement a VAE with the reparameterization trick and hand-derived backprop.

**I4.** Reproduce Experiment 3: show the VAE generates and a plain AE does not.

**I5.** Reproduce Experiment 4: sweep β and observe reconstruction/KL and posterior collapse.

**I6.** Train a VAE on MNIST; visualize samples and latent-space interpolations.

**I7.** Add KL annealing (warm up β from 0) and show it mitigates collapse.

**I8.** Implement a β-VAE and measure disentanglement on a synthetic factored dataset.

**I9.** Implement a conditional VAE (condition on a label) and generate per class.

**I10.** *(Discrete latent.)* Implement a VQ-VAE-style discrete bottleneck and compare sample sharpness.

---

## Tier 3 — Interview

**Q1.** What is a VAE and how does it differ from a plain autoencoder?

**Q2.** What is the ELBO and what are its two terms?

**Q3.** What is the reparameterization trick and why is it needed?

**Q4.** Why does the Gaussian KL have a closed form?

**Q5.** How does a VAE generate new data?

**Q6.** What is β-VAE and what does β control?

**Q7.** What is posterior collapse and when does it happen?

**Q8.** Why are VAE samples blurry?

**Q9.** How do VAEs compare to GANs?

**Q10.** Where are VAEs used in modern systems (e.g. latent diffusion)?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain why a plain AE can't generate and a VAE can
- [ ] Derive the ELBO and the Gaussian KL closed form
- [ ] Explain and implement the reparameterization trick
- [ ] Show the reparameterization variance advantage
- [ ] Explain β-VAE and posterior collapse
- [ ] Explain why VAE samples are blurry
- [ ] Compare VAEs to GANs and diffusion
