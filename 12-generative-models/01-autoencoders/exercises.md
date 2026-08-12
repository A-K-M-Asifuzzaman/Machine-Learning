# 12.01 — Exercises: Autoencoders

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the autoencoder objective and explain why the bottleneck prevents trivial copying.

**D2.** Prove that a linear autoencoder with MSE loss recovers the PCA subspace (Baldi–Hornik). Show the
decoder's row space equals the span of the top-$k$ principal components.

**D3.** Explain why the encoder weights of a linear AE are the PCA components only *up to* an invertible
linear transformation.

**D4.** Explain how the bottleneck-vs-reconstruction-error curve estimates intrinsic dimension.

**D5.** Derive the denoising autoencoder objective and argue it forces the model to learn the data
manifold (project noisy points onto it).

**D6.** Write the sparse-autoencoder objective (L1 or KL sparsity) and explain why an overcomplete code
needs it.

**D7.** Define the contractive autoencoder penalty (encoder Jacobian norm) and what invariance it
induces.

**D8.** Explain why a plain autoencoder is not a generative model — what is wrong with sampling a random
latent code?

**D9.** Explain anomaly detection via reconstruction error and its assumptions.

**D10.** Relate denoising autoencoders to diffusion models (denoising across noise levels).

---

## Tier 2 — Implementation

**I1.** Implement a linear autoencoder and verify it recovers the PCA subspace (Experiment 1).

**I2.** Reproduce Experiment 2: sweep bottleneck size and find the intrinsic-dimension elbow.

**I3.** Implement a denoising autoencoder and reproduce Experiment 3.

**I4.** Implement a sparse (L1) autoencoder and reproduce Experiment 4's sparsity/reconstruction
trade-off.

**I5.** Reproduce Experiment 5: anomaly detection by reconstruction error; compute the AUC.

**I6.** Implement a KL-sparsity penalty toward a target activation rate and compare to L1.

**I7.** Implement a contractive autoencoder and measure its Jacobian norm vs a plain AE.

**I8.** Train a deep (multi-layer) autoencoder on image data (MNIST) and visualize the latent space.

**I9.** Show a plain AE's latent space is not sampleable: decode random codes and observe garbage.

**I10.** *(Interpretability.)* Train a sparse autoencoder on a network's hidden activations and inspect
the learned features.

---

## Tier 3 — Interview

**Q1.** What is an autoencoder and what does the bottleneck do?

**Q2.** How does a linear autoencoder relate to PCA?

**Q3.** How do you choose the bottleneck size?

**Q4.** What is a denoising autoencoder and why does it learn useful features?

**Q5.** What is a sparse autoencoder and when would you use one?

**Q6.** What is a contractive autoencoder?

**Q7.** How do you use an autoencoder for anomaly detection?

**Q8.** Why is a plain autoencoder not a generative model?

**Q9.** How do autoencoders relate to diffusion models?

**Q10.** When would you use an autoencoder over PCA?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the bottleneck and why it prevents copying
- [ ] Prove the linear-AE/PCA equivalence
- [ ] Estimate intrinsic dimension from the bottleneck curve
- [ ] Implement denoising, sparse, and contractive variants
- [ ] Use reconstruction error for anomaly detection
- [ ] Explain the generative gap that motivates the VAE
- [ ] Connect denoising autoencoders to diffusion
