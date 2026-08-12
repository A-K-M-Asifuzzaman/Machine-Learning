# 12.05 — References: Normalizing Flows & Autoregressive Models

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Change of variables, flows | Rezende & Mohamed (2015) |
| §3 | RealNVP coupling layers | Dinh et al. (2017) |
| §3 | Glow (1x1 conv) | Kingma & Dhariwal (2018) |
| §5 | Autoregressive (PixelCNN) | van den Oord et al. (2016) |
| §5 | WaveNet | van den Oord et al. (2016) |
| §5 | MADE | Germain et al. (2015) |
| §6 | Family comparison | Bond-Taylor et al. (2021, survey) |

---

## Normalizing flows

- **Rezende, D. & Mohamed, S. (2015).** "Variational Inference with Normalizing Flows." *ICML*. —
  introduced normalizing flows to deep learning (§2). <https://arxiv.org/abs/1505.05770>.
- **Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017).** "Density estimation using Real NVP." *ICLR*. —
  the **affine coupling layer** (§3). <https://arxiv.org/abs/1605.08803>.
- **Kingma, D. & Dhariwal, P. (2018).** "Glow: Generative Flow with Invertible 1x1 Convolutions."
  *NeurIPS*. — high-quality image flows (§3). <https://arxiv.org/abs/1807.03039>.
- **Papamakarios, G. et al. (2021).** "Normalizing Flows for Probabilistic Modeling and Inference."
  *JMLR*. — the definitive survey. <https://arxiv.org/abs/1912.02762>.

## Autoregressive models

- **van den Oord, A. et al. (2016).** "Pixel Recurrent Neural Networks" (**PixelRNN/PixelCNN**). *ICML*.
  — autoregressive image generation (§5). <https://arxiv.org/abs/1601.06759>.
- **van den Oord, A. et al. (2016).** "WaveNet: A Generative Model for Raw Audio." — autoregressive audio
  (§5). <https://arxiv.org/abs/1609.03499>.
- **Germain, M. et al. (2015).** "MADE: Masked Autoencoder for Distribution Estimation." *ICML*. — a
  masked autoregressive density estimator (§5). <https://arxiv.org/abs/1502.03509>.
- **Papamakarios, G. et al. (2017).** "Masked Autoregressive Flow." *NeurIPS*. — the flow/autoregressive
  connection. <https://arxiv.org/abs/1705.07057>.

## Surveys and comparison

- **Bond-Taylor, S. et al. (2021).** "Deep Generative Modelling: A Comparative Review of VAEs, GANs,
  Normalizing Flows, Energy-Based and Autoregressive Models." *IEEE TPAMI*. — the family map (§6).
  <https://arxiv.org/abs/2103.04922>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [RealNVP / Glow (openai)](https://github.com/openai/glow) | reference flow implementations |
| [`nflows`](https://github.com/bayesiains/nflows) | a normalizing-flow library |
| [PixelCNN++ (openai)](https://github.com/openai/pixel-cnn) | autoregressive images |
| [`normflows`](https://github.com/VincentStimper/normalizing-flows) | flows in PyTorch |

---

## Deferred to later chapters

- **Diffusion** → [12.04](../04-diffusion/)
- **VAEs / GANs** → [12.02](../02-vae/), [12.03](../03-gan/)
- **Autoregressive transformers (LLMs)** → [Part 11](../../11-transformers-and-llms/)
- **Probability & change of variables** → [00.03](../../00-mathematical-foundations/03-probability/)
