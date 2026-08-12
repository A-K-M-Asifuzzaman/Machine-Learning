# 12.05 — Normalizing Flows & Autoregressive Models

> **Two families of generative models pay for something the others can't offer: the exact likelihood.**
> A VAE gives only a lower bound; a GAN gives no likelihood at all; diffusion gives a bound. Normalizing
> flows and autoregressive models give the **exact** $\log p(x)$ — by construction. Flows do it with
> *invertible* transformations and the change-of-variables formula; autoregressive models do it with the
> *chain rule*. This chapter builds both, verifies their exactness to machine precision, and ends with
> the map of the whole generative-model landscape.

Exact likelihood matters wherever you need a *real* density: anomaly detection, model comparison,
compression, or as a building block in a larger probabilistic model. The cost is a constrained
architecture — and that constraint is the whole story of this chapter.

## Table of contents

1. [Why exact likelihood](#1-why-exact-likelihood)
2. [Change of variables](#2-change-of-variables)
3. [Normalizing flows: RealNVP](#3-normalizing-flows-realnvp)
4. [Flows give an exact normalized density](#4-flows-give-an-exact-normalized-density)
5. [Autoregressive models](#5-autoregressive-models)
6. [The generative-model landscape](#6-the-generative-model-landscape)
7. [Common misconceptions](#7-common-misconceptions)

## 1. Why exact likelihood

To *train by maximum likelihood* you need $p(x)$. VAEs ([12.02](../02-vae/)) can't compute it exactly, so
they maximize a bound (the ELBO); GANs ([12.03](../03-gan/)) never form a likelihood; diffusion
([12.04](../04-diffusion/)) optimizes a bound. Flows and autoregressive models compute $\log p(x)$
**exactly**, which gives them the best density estimates and makes them the right tool when you need a
calibrated probability, not just a sample.

## 2. Change of variables

The foundation of flows. If $z = f(x)$ is an **invertible** map and $z$ has a simple density $p_z$
(e.g. a standard normal), then the density of $x$ is:

$$
p_x(x) = p_z\big(f(x)\big)\,\left|\det \frac{\partial f}{\partial x}\right|.
$$

The Jacobian-determinant factor corrects for how $f$ **stretches or squeezes volume**. Experiment 1
verifies it: with $x \sim \mathcal{N}(0,1)$ and $z = \tanh(x)$, the predicted density of $z$ matches its
empirical histogram (e.g. at $z=0.8$: formula 0.606, empirical 0.615). So if you can compute $f$, its
**inverse**, and its **Jacobian determinant**, you can compute the exact density of $x$ from a simple
base density. Making all three tractable *and* the map expressive is the design challenge flows solve.

## 3. Normalizing flows: RealNVP

A **normalizing flow** stacks invertible transformations to map a simple base distribution to a complex
data distribution. The engineering problem: a general invertible net has an $O(d^3)$ Jacobian
determinant — too expensive. **RealNVP** (Dinh et al., 2017) solves it with the **affine coupling
layer**: split the input, pass one half through **unchanged**, and affine-transform the other half using
a network of the *unchanged* half:

$$
y_a = x_a, \qquad y_b = x_b \odot e^{s(x_a)} + t(x_a).
$$

This is (a) **exactly invertible** — knowing $y_a = x_a$ lets you recompute $s, t$ and undo the
transform: $x_b = (y_b - t(x_a)) \odot e^{-s(x_a)}$ — and (b) has a **triangular Jacobian**, so its
log-determinant is just $\sum s(x_a)$ (no determinant computation). Experiment 2 confirms both on a
4-layer flow:

| | Value |
|---|---|
| invertibility $\lvert x - f^{-1}(f(x))\rvert$ | **4.4 × 10⁻¹⁶** |
| $\log\lvert\det J\rvert$: analytic (sum of scales) | −0.25314 |
| $\log\lvert\det J\rvert$: numerical Jacobian | −0.25314 |

Stacking coupling layers (alternating which half is transformed) builds an expressive invertible map
with a cheap exact Jacobian. **Glow** (Kingma & Dhariwal, 2018) added invertible $1\times1$ convolutions
for high-quality images.

## 4. Flows give an exact normalized density

Because change of variables preserves probability mass, a flow is a **proper, exactly normalized
density**. Experiment 3 integrates the flow's $p(x)$ over a fine grid:

$$
\int p(x)\,dx = 0.9901 \approx 1 \quad(\text{the shortfall is only grid truncation}).
$$

It integrates to 1 — unlike a VAE's ELBO (a bound) or a GAN (no density). This is the flows' superpower:
**exact log-likelihood**, which is why they excel at density estimation, anomaly detection, and exact
model comparison. The price: the invertibility + tractable-Jacobian constraints limit architecture
flexibility, and flows generally need many layers and produce slightly less sharp samples than diffusion
or GANs.

## 5. Autoregressive models

The other exact-likelihood family takes a completely different route: the **chain rule** of probability.
Any joint factorizes as a product of conditionals:

$$
p(x) = \prod_{i=1}^{d} p(x_i \mid x_1, \dots, x_{i-1}).
$$

Model each conditional with a network, and you get the **exact** likelihood — the product of conditionals
*is* the joint, by definition. Experiment 4 verifies it on an AR(1) process: the chain-rule log-likelihood
equals the exact joint multivariate-Gaussian log-density to **$9\times10^{-16}$**. Autoregressive models
give the best density estimates of any family, and they include the most important models in ML:

- **PixelCNN / PixelRNN** — generate images pixel by pixel.
- **WaveNet** — generate raw audio sample by sample.
- **GPT / every LLM** — generate text token by token ([11.02](../../11-transformers-and-llms/02-pretraining/)).

**Every large language model is an autoregressive generative model.** The cost is **sequential
generation** — one element at a time, so sampling is slow (the very bottleneck that KV caches and
speculative decoding, [11.03](../../11-transformers-and-llms/03-efficient-attention/) /
[11.07](../../11-transformers-and-llms/07-inference/), attack).

## 6. The generative-model landscape

Every deep generative model trades off **likelihood** (exact / bound / none), **sampling speed**, and
**sample quality**. Experiment 5 lays out the whole map:

| Family | Likelihood | Sampling | Samples | Note |
|---|:--:|:--:|:--:|---|
| Autoencoder ([12.01](../01-autoencoders/)) | none | — | — | reconstruct only |
| VAE ([12.02](../02-vae/)) | bound | fast (1 pass) | blurry | has an encoder |
| GAN ([12.03](../03-gan/)) | none | fast (1 pass) | sharp | unstable, mode collapse |
| Diffusion ([12.04](../04-diffusion/)) | bound | slow (many steps) | sharp | stable, SOTA images |
| **Normalizing flow** | **exact** | fast (1 pass) | good | invertible, restricted arch |
| **Autoregressive** | **exact** | slow (sequential) | sharp | best density; LLMs |

**There is no free lunch.** Flows and autoregressive models buy exact likelihood by constraining the
architecture; GANs buy sharpness by giving up likelihood and stability; diffusion buys sharp-and-stable
at the cost of slow sampling; VAEs are fast with an encoder but blurry. The right choice depends on
whether you need **density**, **speed**, or **fidelity**. In practice today: **diffusion** for images,
**autoregressive transformers** for text and audio, **flows** where an exact density is essential, and
**VAEs** as components (latent diffusion).

## 7. Common misconceptions

- **"All generative models estimate a likelihood."** GANs don't; VAEs and diffusion give bounds; only
  flows and autoregressive models give the exact value (§1, §6).
- **"Flows are obscure."** The change-of-variables idea underlies exact-likelihood modeling, and coupling
  layers appear across ML (§2–§3).
- **"Autoregressive models are just for sequences."** PixelCNN applies them to images; the framework is
  general — any ordering works (§5).
- **"LLMs are a different kind of model."** A GPT *is* an autoregressive generative model with exact
  likelihood (§5).
- **"Exact likelihood means best samples."** Not necessarily — flows give exact likelihood but samples
  are often less sharp than diffusion/GANs (§4, §6).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — flows and autoregressive likelihood in NumPy. Five
  experiments: (1) the change-of-variables formula matches an empirical density; (2) a RealNVP coupling
  layer is invertible to $10^{-16}$ with a Jacobian matching the numerical one; (3) the flow density
  integrates to 1; (4) an autoregressive chain-rule likelihood equals the exact joint to $10^{-16}$;
  (5) the generative-family trade-off table.
- **[exercises.md](exercises.md)** — derive change of variables and the coupling Jacobian, implement a
  flow and an autoregressive model, compare families.
- **[references.md](references.md)** — RealNVP, Glow, PixelCNN, WaveNet, and survey papers.

## Where this leads

- **Diffusion — the other likelihood-bound family** → [12.04](../04-diffusion/)
- **VAEs and GANs** → [12.02](../02-vae/), [12.03](../03-gan/)
- **Autoregressive transformers (LLMs)** → [Part 11](../../11-transformers-and-llms/)
- **Change of variables & probability** → [00.03](../../00-mathematical-foundations/03-probability/)
- **Anomaly detection with exact densities** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
