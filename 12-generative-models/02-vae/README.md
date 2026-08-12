# 12.02 — Variational Autoencoders

> **A VAE is an autoencoder whose latent space is a probability distribution you can sample.** A plain
> autoencoder ([12.01](../01-autoencoders/)) reconstructs beautifully but cannot generate — its latent
> space is full of holes, so a random code decodes to garbage. The VAE fixes this with three ideas: a
> *probabilistic* encoder that outputs a distribution, a *prior* the latent is regularized toward, and
> a loss — the **ELBO** — that balances reconstruction against that regularization. The reparameterization
> trick makes the whole thing trainable by gradient descent. The result is the first true deep
> generative model in this part.

The gap the VAE closes: an autoencoder learns to *reconstruct*, but its encoder maps data to arbitrary,
gap-riddled latent regions. To *generate*, you need to sample the latent — which requires it to be a
known, sampleable distribution. The VAE (Kingma & Welling, 2013) makes it one.

## Table of contents

1. [The generative gap](#1-the-generative-gap)
2. [The ELBO](#2-the-elbo)
3. [The reparameterization trick](#3-the-reparameterization-trick)
4. [Sampling: the VAE generates](#4-sampling-the-vae-generates)
5. [β-VAE and posterior collapse](#5-β-vae-and-posterior-collapse)
6. [What VAEs are good (and bad) at](#6-what-vaes-are-good-and-bad-at)
7. [Common misconceptions](#7-common-misconceptions)

## 1. The generative gap

To turn an autoencoder into a generator you must be able to **sample a latent code and decode it**. That
requires the latent codes to follow a *known* distribution. The VAE enforces this: the encoder outputs a
**distribution** $q(z \mid x) = \mathcal{N}(\mu(x), \sigma^2(x))$ instead of a point, and a **prior**
$p(z) = \mathcal{N}(0, I)$ is imposed so that, across the whole dataset, the codes fill out the prior.
Then generation is trivial: draw $z \sim \mathcal{N}(0, I)$, decode.

## 2. The ELBO

We want to maximize the data likelihood $\log p(x)$, but it is intractable (it integrates over all $z$).
The VAE maximizes a tractable lower bound, the **Evidence Lower BOund**:

$$
\log p(x) \;\ge\; \underbrace{\mathbb{E}_{q(z \mid x)}\!\big[\log p(x \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{\mathrm{KL}\big(q(z \mid x)\,\Vert\,p(z)\big)}_{\text{regularizer}} \;=\; \text{ELBO}.
$$

Two terms with opposite pulls: the **reconstruction** term wants the code to carry enough information to
rebuild $x$; the **KL** term pulls $q(z\mid x)$ toward the prior, keeping the latent space regular and
sampleable. Because both $q$ and $p$ are Gaussians, the KL has an **exact closed form**:

$$
\mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\Vert\, \mathcal{N}(0, I)\big) = \tfrac{1}{2}\sum_i \big(\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2\big).
$$

Experiment 1 confirms it matches Monte Carlo (1.786 vs 1.786) — so only the *reconstruction* term needs
sampling, and that is where the reparameterization trick comes in.

## 3. The reparameterization trick

To train, we must differentiate $\mathbb{E}_{z \sim q}[\cdot]$ with respect to $q$'s parameters $\mu,
\sigma$ — but you **cannot backpropagate through a random sampling operation**. The **reparameterization
trick** rewrites the sample to move the randomness *outside* the parameters:

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

Now $z$ is a *deterministic, differentiable* function of $\mu, \sigma$ (and a fixed noise $\epsilon$), so
gradients flow through it. The alternative — the **score-function / REINFORCE** estimator — also gives
unbiased gradients but with far higher variance. Experiment 2 estimates $\frac{d}{d\mu}\mathbb{E}[z^2]$
both ways:

| Estimator | Mean | Std (estimator variance) |
|---|:--:|:--:|
| reparameterization | 3.00 | **0.146** |
| score function (REINFORCE) | 2.99 | 0.516 |

Both are unbiased (correct mean $= 2\mu = 3$), but reparameterization has **4× lower variance** — which
is why VAEs train stably where black-box gradient estimators struggle. This trick is the technical heart
of the VAE.

## 4. Sampling: the VAE generates

Does the KL regularizer actually make the latent sampleable? Experiment 3 trains a VAE ($\beta=1$) and a
plain autoencoder ($\beta=0$) on 4 separated clusters, then samples $\mathcal{N}(0,I)$ and decodes:

| Model | Mean distance of prior-samples to real data |
|---|:--:|
| **VAE (β=1)** | **0.69** (near the clusters) |
| plain AE (β=0) | 2.18 (3.2× farther) |

The VAE's prior samples land *on* the data — it is a true generative model. The plain autoencoder packs
the clusters into arbitrary latent regions with **gaps** between them, so prior samples fall into those
gaps and decode to points *between* clusters (garbage). **Regularizing the latent toward a known prior
is exactly what makes generation possible** — the one idea separating a VAE from an autoencoder.

## 5. β-VAE and posterior collapse

The KL term can be weighted by a coefficient $\beta$: $\text{ELBO}_\beta = \text{reconstruction} -
\beta\,\text{KL}$. Experiment 4 sweeps it (KL measures latent usage):

| β | Reconstruction MSE | KL (latent usage) | Status |
|:--:|:--:|:--:|:--:|
| 0.0 | 0.0001 | 32.4 | no KL (plain AE) |
| 1.0 | 0.057 | 0.46 | standard VAE |
| 10 | 0.567 | 0.0001 | **collapsed** |
| 100 | 0.566 | 0.0000 | **collapsed** |

- **β < 1** favors reconstruction; **β > 1** (**β-VAE**, Higgins et al., 2017) pressures the latent to be
  more independent/**disentangled**, at a reconstruction cost.
- Push β too high and **posterior collapse** sets in: the encoder gives up and outputs the prior
  (KL → 0), the latent carries **no information**, and the decoder ignores it — reconstruction degrades
  to the data mean. Collapse also happens with too-powerful decoders (which can reconstruct without the
  latent), and is a central practical challenge, especially for VAEs on text.

β is the dial between reconstruction fidelity and a well-structured, disentangled latent, with collapse
as the failure mode at the extreme.

## 6. What VAEs are good (and bad) at

**Good:** a *principled* probabilistic generative model with an encoder (so you get an inference network
and a structured, interpretable latent for free), stable training, and a likelihood bound. Great for
representation learning, interpolation, and as a component (the "VAE" in **latent diffusion**,
[12.04](../04-diffusion/), compresses images into the latent that diffusion runs in).

**Bad:** samples are **blurry**. The Gaussian likelihood and the KL regularizer average over
possibilities, so VAE images look soft compared to GANs ([12.03](../03-gan/)) or diffusion. The blur is
the price of the probabilistic, mode-covering objective. Sharper variants (VQ-VAE, discrete latents;
NVAE; VAE+flows) address it.

## 7. Common misconceptions

- **"A VAE is just an autoencoder with noise."** The KL-to-prior regularizer and the probabilistic
  encoder are what make it *generative*; noise alone doesn't (§1, §4).
- **"The reparameterization trick is an optimization detail."** It is what makes the expectation
  differentiable at all — without it you'd need high-variance REINFORCE (§3).
- **"Higher β is better (more disentangled)."** Only up to a point; too high causes posterior collapse
  and destroys reconstruction (§5).
- **"VAEs and GANs are interchangeable."** VAEs give a likelihood and an encoder but blurry samples;
  GANs give sharp samples but no likelihood/encoder (§6).
- **"Blurry samples mean the VAE failed."** Blur is inherent to the objective, not a bug — it is the
  mode-covering behavior of maximum likelihood (§6).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a VAE in NumPy with the reparameterization trick and
  hand-derived backprop. Four experiments: (1) the Gaussian KL closed form matches Monte Carlo;
  (2) reparameterization gives 4× lower gradient variance than REINFORCE; (3) the VAE generates (prior
  samples land on data) where a plain AE's land 3.2× farther; (4) β-VAE trades reconstruction vs KL and
  collapses at high β.
- **[exercises.md](exercises.md)** — derive the ELBO and the KL closed form, implement the
  reparameterization trick, analyze collapse.
- **[references.md](references.md)** — the VAE, β-VAE, and follow-up papers.

## Where this leads

- **The autoencoder the VAE builds on** → [12.01](../01-autoencoders/)
- **GANs — sharp samples, no encoder** → [12.03](../03-gan/)
- **Diffusion — and latent diffusion, which runs in a VAE's latent** → [12.04](../04-diffusion/)
- **Normalizing flows — exact likelihoods** → [12.05](../05-flows-and-autoregressive/)
- **The KL divergence and information theory** → [00.05](../../00-mathematical-foundations/05-information-theory/)
