# 12.01 — Autoencoders

> **An autoencoder learns by trying to copy its input to its output — through a bottleneck too narrow
> to let it cheat.** Forced to squeeze the data through a compressed code and rebuild it, the network
> must discover the structure that matters. Change the constraint on that code — make it narrow,
> denoising, or sparse — and you change what it learns. Autoencoders are the entry point to generative
> modeling: a linear one is literally PCA, and a denoising one is the direct ancestor of diffusion.

An autoencoder has two halves: an **encoder** $z = f(x)$ that compresses the input to a **latent code**,
and a **decoder** $\hat x = g(z)$ that reconstructs it, trained to minimize reconstruction error
$\lVert x - g(f(x))\rVert^2$. The interesting part is never the copy — it is what the *bottleneck*
forces the code to become. This is unsupervised representation learning, the bridge from
[dimensionality reduction](../../04-unsupervised-learning/06-linear-dimensionality-reduction/) to the
generative models in the rest of Part 12.

## Table of contents

1. [The bottleneck idea](#1-the-bottleneck-idea)
2. [Linear autoencoders are PCA](#2-linear-autoencoders-are-pca)
3. [The bottleneck and intrinsic dimension](#3-the-bottleneck-and-intrinsic-dimension)
4. [Denoising autoencoders](#4-denoising-autoencoders)
5. [Sparse and contractive autoencoders](#5-sparse-and-contractive-autoencoders)
6. [Anomaly detection](#6-anomaly-detection)
7. [What autoencoders are (and aren't) good for](#7-what-autoencoders-are-and-arent-good-for)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The bottleneck idea

If the code were as wide as the input, the network could just copy — learning nothing. The **bottleneck**
(an *undercomplete* code, narrower than the input) makes that impossible: the network must throw away
everything except the structure needed to reconstruct. What survives the bottleneck is a compressed
representation of the data's underlying **manifold**. Other constraints (denoising, sparsity) achieve the
same "can't just copy" pressure differently (§4–§5).

## 2. Linear autoencoders are PCA

The cleanest fact about autoencoders: **a linear autoencoder minimizing reconstruction error learns the
PCA subspace.** Experiment 1 trains a linear autoencoder (no nonlinearity) with a $k$-unit bottleneck on
data lying near a rank-3 subspace:

| | Value |
|---|---|
| autoencoder reconstruction MSE | 0.001718 |
| PCA (top-3) reconstruction MSE | 0.001714 |
| principal angles (AE subspace vs PCA) | [0.00°, 0.00°, 0.02°] |

The reconstruction errors are identical and the decoder's row space aligns with PCA's top-3 components
to a fraction of a degree. This is the **Baldi–Hornik theorem**: linear autoencoding *is* PCA
([04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)), up to a rotation within
the subspace. The value of an autoencoder, then, is what happens when you make it **nonlinear**: swap
the linear map for a deep net and the bottleneck learns a *curved* manifold instead of a flat subspace —
a nonlinear generalization of PCA.

## 3. The bottleneck and intrinsic dimension

The bottleneck width is a compression dial. Experiment 2 puts data on a nonlinear ~3-dimensional
manifold in $\mathbb{R}^{12}$ and sweeps the bottleneck:

| Bottleneck $k$ | Reconstruction MSE |
|:--:|:--:|
| 1 | 0.196 |
| 2 | 0.048 |
| 3 | **0.019** (intrinsic dim) |
| 4 | 0.010 |
| 6 | 0.003 |

Error falls **steeply** as the bottleneck approaches the data's intrinsic dimension (0.20 → 0.02 by
$k=3$), then with diminishing returns. The steep-then-shallow shape is a practical **intrinsic-dimension
estimator**: it reveals roughly how many dimensions the data really occupies, and the bottleneck just
past the elbow gives strong compression with little loss.

## 4. Denoising autoencoders

Instead of narrowing the code, **corrupt the input**: a **denoising autoencoder** (Vincent et al., 2008)
is trained to reconstruct the *clean* original from a *noisy* version. It cannot copy the input (the
input is corrupted), so it must learn the data manifold and **project noisy points onto it**.
Experiment 3 (noise std 0.4) on held-out data:

| | Distance to clean signal |
|---|:--:|
| noisy input | 0.160 |
| **denoised output** | **0.050** (3.2× closer) |

The autoencoder removes most of the noise on inputs it never saw. Denoising is both directly useful and a
powerful way to learn robust features — and, crucially, it is the **direct ancestor of diffusion models**
([12.04](../04-diffusion/)), which are denoising autoencoders trained across *many* noise levels.

## 5. Sparse and contractive autoencoders

You can even let the code be **overcomplete** (wider than the input) and still avoid trivial copying — by
penalizing the code. Experiment 4 uses a 32-unit code on 8-dim inputs with an **L1 sparsity penalty**:

| L1 penalty | % units near-zero | Recon MSE |
|:--:|:--:|:--:|
| 0.0 | 12% | 0.0001 |
| 0.5 | 95% | 0.0005 |
| 2.0 | 97% | 0.007 |
| 8.0 | 99% | 0.023 |

As the penalty grows, most units go to zero — each input is explained by a **few** active units — at a
small reconstruction cost. Two related ideas:

- **Sparse autoencoders** — penalize activations (L1 or a KL term toward a target sparsity). The code is
  interpretable and disentangled; this is the method now used to **interpret LLM activations**
  (decomposing them into monosemantic features).
- **Contractive autoencoders** — penalize the encoder's Jacobian, so the code is *insensitive* to small
  input changes, learning features robust to local perturbations.

## 6. Anomaly detection

A trained autoencoder reconstructs what it has seen. Train it on **normal data only**, and anomalies —
which lie *off* the learned manifold — reconstruct **poorly**, so reconstruction error is an
anomaly score. Experiment 5:

| | Reconstruction error |
|---|:--:|
| normal points | 0.022 |
| anomalies | 0.462 (21× higher) |
| separation AUC | **1.000** |

Anomalies have 21× the error and are perfectly separable. This is a standard **unsupervised anomaly
detector** — no labels, just "what does normal look like, and what fails to reconstruct?" — complementing
the methods in [04.08](../../04-unsupervised-learning/08-anomaly-detection/).

## 7. What autoencoders are (and aren't) good for

**Good for:** dimensionality reduction (nonlinear PCA), denoising, feature learning, anomaly detection,
and pretraining. The latent code is a useful learned representation.

**Not (directly) generative:** a plain autoencoder learns to *reconstruct*, but its latent space has no
enforced structure — sample a random code and the decoder usually produces garbage, because the code
distribution is unknown and full of holes. To *generate*, you must make the latent space a proper
probability distribution you can sample from. That is exactly the leap to the **variational autoencoder**
([12.02](../02-vae/)), which adds a prior and a probabilistic encoder — turning the autoencoder into a
true generative model.

## 8. Common misconceptions

- **"Autoencoders are generative models."** A plain autoencoder is not — its latent space isn't a
  sampleable distribution (§7); the VAE fixes this.
- **"A wider code is always better."** Without a bottleneck or penalty, the autoencoder just copies and
  learns nothing (§1).
- **"Autoencoders beat PCA."** Only if nonlinear — a linear autoencoder *is* PCA (§2), and for linear
  structure PCA is faster and exact.
- **"Reconstruction error means the model understands the data."** It means the data lies near the
  learned manifold; anomaly detection exploits exactly the *failure* to reconstruct (§6).
- **"Denoising is a niche trick."** It is the conceptual seed of diffusion models, today's
  state-of-the-art generators (§4).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — autoencoders in NumPy with Adam. Five experiments: (1) a
  linear autoencoder recovers the PCA subspace to 0.02°; (2) the bottleneck-vs-error curve reveals
  intrinsic dimension; (3) a denoising autoencoder gets 3.2× closer to the clean signal; (4) an L1
  penalty yields a 99%-sparse overcomplete code; (5) reconstruction error separates anomalies at AUC 1.0.
- **[exercises.md](exercises.md)** — derive the linear-AE/PCA equivalence, implement denoising/sparse
  variants, reason about the generative gap.
- **[references.md](references.md)** — the autoencoder, denoising, and sparse-coding literature.

## Where this leads

- **Variational autoencoders — making the latent space generative** → [12.02](../02-vae/)
- **Diffusion models — denoising at many noise levels** → [12.04](../04-diffusion/)
- **PCA, the linear special case** → [04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)
- **Anomaly detection methods** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
- **Sparse autoencoders for interpretability** → [Part 17](../../17-explainable-ai/)
