# 12.05 — Exercises: Normalizing Flows & Autoregressive Models

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain why exact likelihood matters and which generative families provide it.

**D2.** Derive the change-of-variables formula $p_x(x) = p_z(f(x))|\det \partial f/\partial x|$ and
interpret the Jacobian factor.

**D3.** Write the affine coupling layer and prove it is exactly invertible.

**D4.** Show the coupling layer's Jacobian is triangular, so its log-determinant is the sum of the scale
outputs.

**D5.** Explain why a normalizing flow is an exactly normalized density (integrates to 1).

**D6.** Explain how stacking coupling layers (with alternating masks) builds an expressive invertible
map.

**D7.** Write the autoregressive factorization $p(x)=\prod_i p(x_i\mid x_{<i})$ and explain why it is
exact.

**D8.** Explain the sequential-generation bottleneck of autoregressive models and how it's mitigated
(caching, parallel training).

**D9.** Explain why a GPT is an autoregressive generative model with exact likelihood.

**D10.** Fill in the generative-family trade-off table (likelihood / speed / quality) and justify each
entry.

---

## Tier 2 — Implementation

**I1.** Verify the change-of-variables formula numerically (Experiment 1).

**I2.** Implement a RealNVP coupling layer; verify invertibility and the Jacobian (Experiment 2).

**I3.** Verify a flow's density integrates to 1 (Experiment 3).

**I4.** Verify an autoregressive chain-rule likelihood equals the exact joint (Experiment 4).

**I5.** Reproduce Experiment 5's trade-off table and explain each row.

**I6.** Train a normalizing flow (stacked couplings) on 2-D data by maximum likelihood; sample from it.

**I7.** Implement Glow's invertible $1\times1$ convolution and add it to your flow.

**I8.** Implement a small autoregressive model (MADE or a causal MLP) and train it by exact likelihood.

**I9.** Implement PixelCNN on a small image dataset and sample pixel by pixel.

**I10.** *(Comparison.)* Train a VAE, a flow, and an autoregressive model on the same 2-D data and
compare held-out log-likelihood and sample quality.

---

## Tier 3 — Interview

**Q1.** Which generative models give exact likelihood, and why does it matter?

**Q2.** What is the change-of-variables formula?

**Q3.** What is a normalizing flow?

**Q4.** How does a coupling layer stay invertible with a cheap Jacobian?

**Q5.** Why is a flow an exactly normalized density?

**Q6.** How do autoregressive models factorize the joint?

**Q7.** What is the main cost of autoregressive generation?

**Q8.** Why is a GPT an autoregressive generative model?

**Q9.** Compare flows, autoregressive models, VAEs, GANs, and diffusion.

**Q10.** When would you choose a flow over a VAE or diffusion?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive change of variables and the Jacobian factor
- [ ] Implement an invertible coupling layer with a tractable Jacobian
- [ ] Explain why flows give an exact normalized density
- [ ] Derive the autoregressive chain-rule likelihood
- [ ] Explain the sequential-generation trade-off
- [ ] Place every generative family in the likelihood/speed/quality map
- [ ] Choose the right family for a given need
