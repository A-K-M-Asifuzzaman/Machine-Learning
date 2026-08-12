# 12.03 — Generative Adversarial Networks

> **A GAN learns to generate by getting caught.** Two networks play a game: a **generator** turns noise
> into fake data, and a **discriminator** tries to tell fakes from real. As the discriminator gets
> better at catching fakes, the generator is forced to make them more realistic — until, at
> equilibrium, the fakes are indistinguishable from real data and the discriminator is reduced to a coin
> flip. This adversarial game produces the sharpest images of any generative model — and is
> notoriously hard to train, for reasons this chapter makes precise.

Where a VAE ([12.02](../02-vae/)) maximizes a likelihood bound and produces blurry samples, a GAN
(Goodfellow et al., 2014) optimizes an *implicit* objective with no likelihood at all — just "fool the
discriminator" — and produces sharp ones. The trade-off (sharp vs mode-covering) runs through all of
generative modeling.

## Table of contents

1. [The adversarial game](#1-the-adversarial-game)
2. [What the game optimizes: JS divergence](#2-what-the-game-optimizes-js-divergence)
3. [Why GANs are hard: vanishing gradients](#3-why-gans-are-hard-vanishing-gradients)
4. [Wasserstein GAN](#4-wasserstein-gan)
5. [Mode collapse](#5-mode-collapse)
6. [Architectures: DCGAN to StyleGAN](#6-architectures-dcgan-to-stylegan)
7. [GANs vs VAEs vs diffusion](#7-gans-vs-vaes-vs-diffusion)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The adversarial game

The generator $G$ maps noise $z \sim \mathcal{N}(0,I)$ to data; the discriminator $D$ outputs the
probability that its input is real. They optimize opposite objectives — a **minimax game**:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))].
$$

$D$ wants to assign high probability to real and low to fake; $G$ wants the opposite. Experiment 1 trains
this on a 1-D Gaussian target $\mathcal{N}(2, 0.5)$:

| | Value |
|---|---|
| generated mean | 1.71 (target 2.0) |
| discriminator score on real | **0.510** |

At equilibrium the generator matches the data, so the discriminator **can do no better than a coin
flip** — its score collapses to ~0.5. (The generated spread comes out a little narrow — GANs are
finicky and tend to under-cover, a mild version of mode collapse, §5.) In practice $G$ is trained with
the **non-saturating** loss $\max_G \mathbb{E}[\log D(G(z))]$ instead of $\min_G \mathbb{E}[\log(1-D)]$,
which gives stronger gradients early.

## 2. What the game optimizes: JS divergence

The adversarial game has a precise meaning. For a *fixed* generator, the **optimal discriminator** is

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{gen}}(x)},
$$

and plugging $D^*$ back into the objective gives exactly $2\,\mathrm{JS}(p_{\text{data}} \,\Vert\,
p_{\text{gen}}) - \log 4$. Experiment 2 verifies this to the digit:

| Mean gap | GAN value at $D^*$ | $2\,\mathrm{JS} - \log 4$ |
|:--:|:--:|:--:|
| 0.0 | −1.3863 | −1.3863 |
| 3.0 | −0.3327 | −0.3327 |
| 6.0 | −0.0078 | −0.0078 |

So **training the generator against an optimal discriminator minimizes the Jensen–Shannon divergence**
between the fake and real distributions. The adversarial game is divergence minimization in disguise —
the theoretical foundation of GANs.

## 3. Why GANs are hard: vanishing gradients

That foundation is also the problem. JS divergence is a *bad* loss when the distributions **don't
overlap** — which they don't early in training, when the generator is bad. Experiment 3 puts two narrow
distributions a distance $\theta$ apart:

| Separation $\theta$ | JS divergence | Wasserstein-1 |
|:--:|:--:|:--:|
| 0.5 | 0.693 | 0.5 |
| 1.0 | **0.693** | 1.0 |
| 4.0 | **0.693** | 4.0 |

For any positive separation, JS is **constant at $\log 2 \approx 0.693$** — flat, so it gives the
generator **zero gradient**. The generator gets no signal about which way to move, and training stalls.
This is the vanishing-gradient problem at the root of GAN instability.

## 4. Wasserstein GAN

The fix is a better distance. The **Wasserstein-1** ("earth-mover") distance grows **linearly** with the
separation (Experiment 3: 0.5, 1.0, 4.0) — a smooth, informative gradient everywhere, even for
non-overlapping distributions. **WGAN** (Arjovsky et al., 2017) replaces JS with Wasserstein, which
requires the discriminator (now called a **critic**, outputting a score not a probability) to be
**1-Lipschitz**. Enforcing that:

- **WGAN** — clip the critic's weights (crude, can misbehave).
- **WGAN-GP** (Gulrajani et al., 2017) — add a **gradient penalty** that pushes the critic's gradient
  norm toward 1. This is the standard, stable recipe.

WGANs train far more reliably, give a meaningful loss (the Wasserstein estimate correlates with sample
quality), and largely fix the vanishing-gradient problem.

## 5. Mode collapse

The signature GAN failure. Because the objective only rewards *fooling the discriminator*, not
*covering the data*, the generator can win by producing convincing samples from just **one mode** and
ignoring the rest. Experiment 4 trains a GAN on an equal mixture of two modes at $-5$ and $+5$:

| | Fraction of generated samples |
|---|:--:|
| left mode (−5) | **0.00** |
| right mode (+5) | **1.00** |

The generator piles **100%** of its mass onto a single mode instead of the 50/50 split the data has — it
collapsed. Fixes add explicit pressure for diversity: **WGAN** (the better metric helps),
**minibatch discrimination** (let $D$ see batch statistics), **unrolled GANs**, and packing. Mode
collapse is the flip side of the VAE's blur: **GANs are sharp but drop modes; VAEs cover modes but
blur.**

## 6. Architectures: DCGAN to StyleGAN

The GAN *idea* is orthogonal to the *network*; better architectures drove the image-quality explosion:

- **DCGAN** (2015) — the first stable conv GAN recipe (strided convs, batch norm, no pooling); the
  template.
- **Conditional GAN / pix2pix / CycleGAN** — condition on a label or another image (image-to-image
  translation, style transfer).
- **Progressive GAN → StyleGAN** (2018–) — grow resolution progressively; StyleGAN's
  **style-based generator** (inject a learned latent at every layer via AdaIN) gave unprecedented
  control and photorealism (the "this person does not exist" faces).
- **BigGAN** — scale + class conditioning for high-fidelity ImageNet generation.

## 7. GANs vs VAEs vs diffusion

| | Likelihood | Encoder | Samples | Training |
|---|:--:|:--:|:--:|:--:|
| **VAE** ([12.02](../02-vae/)) | bound (ELBO) | yes | blurry | stable |
| **GAN** | none (implicit) | no | **sharp** | unstable |
| **Diffusion** ([12.04](../04-diffusion/)) | bound | no | **sharp** | stable |

GANs dominated image generation from ~2016–2021 for their sharpness, but their instability and mode
collapse made them hard to work with. **Diffusion models** ([12.04](../04-diffusion/)) now lead: they
match or exceed GAN sample quality *and* train stably (no adversarial game), which is why text-to-image
systems (DALL·E 2/3, Stable Diffusion, Imagen) are diffusion-based. GANs remain valued for **fast**
single-step generation and specific tasks (super-resolution, editing).

## 8. Common misconceptions

- **"The discriminator is thrown away."** During training it *is* the loss function — its gradients
  train the generator; only at inference is it discarded (§1).
- **"A GAN maximizes likelihood."** It has no likelihood; it implicitly minimizes JS (or Wasserstein)
  divergence (§2).
- **"GAN instability is a bug."** It is inherent to the JS objective on non-overlapping supports (§3);
  WGAN addresses the cause (§4).
- **"Mode collapse means the GAN failed to train."** It can look perfectly trained (sharp samples) while
  covering only part of the data (§5).
- **"GANs are the state of the art for generation."** For images, diffusion has largely overtaken them
  (§7); GANs win where single-step speed matters.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a 1-D GAN in NumPy plus the divergence theory. Four
  experiments: (1) a GAN matches a Gaussian and the discriminator converges to 0.51; (2) the optimal
  discriminator makes the loss equal $2\,\mathrm{JS}-\log 4$ exactly; (3) JS is flat (0.693) for disjoint
  supports while Wasserstein is linear — the WGAN motivation; (4) mode collapse (100% of mass on one of
  two modes).
- **[exercises.md](exercises.md)** — derive the optimal discriminator and JS connection, implement
  WGAN-GP, analyze mode collapse.
- **[references.md](references.md)** — GAN, WGAN, DCGAN, StyleGAN, and follow-ups.

## Where this leads

- **Diffusion — the successor that fixed GAN instability** → [12.04](../04-diffusion/)
- **VAEs — the mode-covering, blurry alternative** → [12.02](../02-vae/)
- **The JS/KL divergences behind the objective** → [00.05](../../00-mathematical-foundations/05-information-theory/)
- **Normalizing flows — exact likelihoods** → [12.05](../05-flows-and-autoregressive/)
- **CNN architectures GANs generate with** → [08.02](../../08-computer-vision/02-cnn-architectures/)
