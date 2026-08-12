# 12.04 — Diffusion Models

> **Learn to generate by learning to denoise.** Take any image and gradually destroy it into pure noise
> over many small steps — a process so simple it has a closed form. Then train a single network to undo
> *one* step of that noising. To generate, start from pure noise and run the network backwards,
> denoising step by step until a sample emerges. That is a diffusion model, and it is the engine behind
> DALL·E, Stable Diffusion, Midjourney, and Imagen. This chapter builds a working one from scratch and
> derives why it beats GANs: it is stable, mode-covering, and produces sharp samples.

Diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020) resolves the generative trilemma that trapped
VAEs (blurry) and GANs (unstable, mode-collapsing). It trains by a simple regression loss — no
adversarial game — yet generates sharp, diverse samples. The price is slow, iterative sampling, which
the field has been steadily accelerating.

## Table of contents

1. [The forward process](#1-the-forward-process)
2. [The reparameterization that makes it work](#2-the-reparameterization-that-makes-it-work)
3. [The reverse process and the training objective](#3-the-reverse-process-and-the-training-objective)
4. [Generation, and no mode collapse](#4-generation-and-no-mode-collapse)
5. [Noise prediction is score matching](#5-noise-prediction-is-score-matching)
6. [DDIM: faster sampling](#6-ddim-faster-sampling)
7. [Guidance and latent diffusion](#7-guidance-and-latent-diffusion)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The forward process

The **forward (diffusion) process** gradually corrupts data $x_0$ into noise over $T$ steps, adding a
little Gaussian noise each step according to a variance schedule $\beta_t$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big).
$$

It has no learned parameters — it is a fixed destruction. Experiment 2 tracks the signal-to-noise ratio
as $t$ grows (with $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$ the surviving signal):

| Step $t$ | $\bar\alpha_t$ (signal) | $1-\bar\alpha_t$ (noise) | SNR |
|:--:|:--:|:--:|:--:|
| 0 | 1.000 | 0.000 | 9999 |
| 10 | 0.83 | 0.17 | 4.8 |
| 39 | **0.042** | 0.958 | **0.04** |

By the final step the signal is essentially gone: $x_T \approx \mathcal{N}(0, I)$, pure noise. **That is
the whole point** — if the forward process reliably turns *any* data into the *same* known distribution,
then learning to reverse it turns that known distribution back into data.

## 2. The reparameterization that makes it work

Iterating $T$ noising steps to train would be hopeless. The key trick: the composition of all those
Gaussian steps is **itself one Gaussian**, so you can jump to any noise level $t$ **in one step**:

$$
q(x_t \mid x_0) = \mathcal{N}\!\big(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t) I\big) \quad\Longrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \;\; \epsilon \sim \mathcal{N}(0,I).
$$

Experiment 1 confirms the iterative process matches this closed form (iterative mean 1.046 / var 0.726 vs
closed form 1.047 / 0.726). Now training a step at level $t$ needs no loop — sample $x_0$, sample $t$,
sample $\epsilon$, form $x_t$ directly. This closed form is what makes diffusion trainable at scale.

## 3. The reverse process and the training objective

Generation runs the process **backwards**: start from $x_T \sim \mathcal{N}(0,I)$ and repeatedly remove a
little noise. The reverse step is also Gaussian, and its mean depends on the noise that was added — which
we don't know, so we **train a network $\epsilon_\theta(x_t, t)$ to predict it**. The DDPM training
objective is stunningly simple — just **MSE on the predicted noise**:

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\rVert^2\Big].
$$

Take a clean sample, noise it to a random level $t$, and train the network to guess the noise. That's
it — no adversary, no likelihood estimation, no reparameterization gradient. (This loss is a reweighted
version of the variational bound on $\log p(x)$; the reweighting to plain MSE is what Ho et al. found
works best.) Sampling then applies the trained $\epsilon_\theta$ in the reverse update, one step at a
time.

## 4. Generation, and no mode collapse

Does it generate — and does it avoid the GAN's mode collapse? Experiment 3 trains the model on the
**same bimodal distribution** the GAN collapsed on ([12.03 §5](../03-gan/)):

| | Value |
|---|---|
| generated mean | −0.02 (real −0.02) |
| generated std | 2.11 (real 2.02) |
| fraction near left mode (−2) | **0.49** |
| fraction near right mode (+2) | **0.51** |

The diffusion model covers **both modes ~50/50** — where the GAN put 100% on one. Diffusion's objective
is maximum-likelihood-like (**mode-covering**), not adversarial, so it has no incentive to drop modes.
Stable training + mode coverage + sharp samples is exactly why diffusion **replaced GANs** for image
generation.

## 5. Noise prediction is score matching

There are two views of diffusion, and they are the same. Predicting the noise is equivalent to learning
the **score** — the gradient of the log-density $\nabla_x \log q(x_t)$ — of the noised data:

$$
\epsilon^*(x_t) = -\sqrt{1-\bar\alpha_t}\;\nabla_{x_t} \log q(x_t).
$$

Experiment 4 verifies this exactly on Gaussian data (the optimal $\epsilon^*$ equals
$-\sqrt{1-\bar\alpha_t}\cdot\text{score}$ to the digit). So a diffusion model is a **score-based
generative model** (Song & Ermon, 2019): it learns to point "uphill" toward higher data density at every
noise level, and sampling follows those gradients from noise to data (Langevin dynamics). The DDPM and
score-matching literatures describe the same object; the continuous-time view is a **stochastic
differential equation** whose reverse-time solution is generation.

## 6. DDIM: faster sampling

DDPM's stochastic sampling needs *all* $T$ steps (hundreds to thousands in real models) — slow. **DDIM**
(Song et al., 2021) reinterprets the reverse process as a **deterministic** path (an ODE) that can be
solved in far fewer steps. Experiment 5 samples the same trained model:

| Sampler | Steps | Gen std | Mode imbalance |
|---|:--:|:--:|:--:|
| DDPM | 40 | 2.11 | 0.005 |
| DDIM | 10 | 2.04 | 0.007 |
| **DDIM** | **5** | 1.96 | 0.011 |

Five DDIM steps recover nearly the full-quality distribution. Determinism also enables exact latent
inversion (find the noise that produced an image) and smooth interpolation. Accelerating sampling —
**DDIM, DPM-Solver, consistency models, distillation** (down to 1–4 steps) — is the main practical thrust
of diffusion research.

## 7. Guidance and latent diffusion

Two ideas turned diffusion into the text-to-image powerhouse:

- **Classifier-free guidance** (Ho & Salimans, 2022) — train the model both *conditioned* (on a text
  prompt) and *unconditioned*, then at sampling extrapolate: $\epsilon = \epsilon_{\text{uncond}} +
  w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$. The guidance scale $w$ trades **diversity for
  prompt-adherence** — higher $w$ follows the prompt more tightly. This is the knob behind every
  text-to-image model.
- **Latent diffusion** (Rombach et al., 2022, = Stable Diffusion) — running diffusion on full-resolution
  pixels is expensive. Instead, compress images into a small latent with a **VAE** ([12.02](../02-vae/)),
  run diffusion **in that latent space**, and decode. This cut the cost enough to make high-resolution
  text-to-image practical and open.

The full modern stack: text encoder (CLIP/T5) → conditioned latent diffusion with classifier-free
guidance → VAE decoder, sampled with a fast solver.

## 8. Common misconceptions

- **"Diffusion adds noise to generate."** The forward (noising) process is fixed and only used for
  *training*; *generation* is the learned reverse (denoising) process (§1, §3).
- **"The training loss is complicated."** It is plain MSE on predicted noise (§3) — the theory is deep,
  the objective is a one-liner.
- **"Diffusion and score-based models are different."** They are two views of the same thing (§5).
- **"Diffusion is slow because it's fundamentally iterative."** DDIM/solvers/distillation cut it to a
  handful of steps (§6).
- **"Stable Diffusion runs on pixels."** It runs in a VAE's latent space (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a working diffusion model in NumPy. Five experiments: (1) the
  forward closed form matches iterative noising; (2) the SNR collapses to near-zero (pure noise);
  (3) the model generates and covers both modes (no collapse, vs the GAN); (4) noise-prediction equals
  score matching exactly; (5) DDIM samples in 5 steps.
- **[exercises.md](exercises.md)** — derive the closed form, the DDPM objective, and the score
  connection; implement DDIM and guidance.
- **[references.md](references.md)** — DDPM, score-based models, DDIM, guidance, and latent diffusion.

## Where this leads

- **GANs — the predecessor diffusion replaced** → [12.03](../03-gan/)
- **The VAE inside latent diffusion** → [12.02](../02-vae/)
- **Denoising autoencoders — the conceptual seed** → [12.01 §4](../01-autoencoders/)
- **The U-Net / attention backbone diffusion uses** → [08.04](../../08-computer-vision/04-detection-and-segmentation/), [11.01](../../11-transformers-and-llms/01-transformer/)
- **CLIP text conditioning** → [08.05](../../08-computer-vision/05-vision-transformers/)
