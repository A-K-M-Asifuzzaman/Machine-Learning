# 12.01 — References: Autoencoders

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Autoencoders, representation learning | Goodfellow et al. Ch. 14 |
| §2 | Linear AE = PCA | Baldi & Hornik (1989) |
| §4 | Denoising autoencoders | Vincent et al. (2008, 2010) |
| §5 | Sparse autoencoders | Ng (2011); Olshausen & Field (1996) |
| §5 | Contractive autoencoders | Rifai et al. (2011) |
| §5 | Sparse AEs for interpretability | Bricken et al. (2023) |
| §6 | Anomaly detection | Goodfellow et al. §14.1 |

---

## Foundational

- **Baldi, P. & Hornik, K. (1989).** "Neural networks and principal component analysis: Learning from
  examples without local minima." *Neural Networks*. — proves **linear autoencoders learn the PCA
  subspace** (§2).
- **Hinton, G. & Salakhutdinov, R. (2006).** "Reducing the Dimensionality of Data with Neural Networks."
  *Science*. — deep autoencoders for nonlinear dimensionality reduction (§1, §3).
  <https://www.science.org/doi/10.1126/science.1127647>.

## Denoising, sparse, contractive

- **Vincent, P. et al. (2008).** "Extracting and Composing Robust Features with Denoising Autoencoders."
  *ICML*. — the **denoising autoencoder** (§4). <https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf>.
- **Vincent, P. et al. (2010).** "Stacked Denoising Autoencoders." *JMLR*. — deep denoising AEs (§4).
- **Olshausen, B. & Field, D. (1996).** "Emergence of simple-cell receptive field properties by learning
  a sparse code for natural images." *Nature*. — the origin of **sparse coding** (§5).
- **Ng, A. (2011).** "Sparse Autoencoder." *CS294A lecture notes.* — the KL-sparsity autoencoder (§5).
- **Rifai, S. et al. (2011).** "Contractive Auto-Encoders." *ICML*. — the Jacobian penalty (§5).
  <https://icml.cc/2011/papers/455_icmlpaper.pdf>.
- **Bricken, T. et al. (2023).** "Towards Monosemanticity: Decomposing Language Models With Dictionary
  Learning." *Anthropic.* — sparse autoencoders to interpret LLM activations (§5).
  <https://transformer-circuits.pub/2023/monosemantic-features>.

## Textbook

- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Chapter 14 "Autoencoders."** —
  undercomplete, denoising, sparse, contractive AEs and their theory. Free at
  <https://www.deeplearningbook.org/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | the linear special case (§2) |
| [PyTorch autoencoder examples](https://github.com/pytorch/examples) | deep AE / denoising AE reference |
| [SAE (sparse autoencoder) tooling](https://github.com/jbloomAus/SAELens) | sparse autoencoders for interpretability (§5) |

---

## Deferred to later chapters

- **Variational autoencoders (making the latent generative)** → [12.02](../02-vae/)
- **Diffusion (denoising across noise levels)** → [12.04](../04-diffusion/)
- **PCA / SVD** → [04.06](../../04-unsupervised-learning/06-linear-dimensionality-reduction/)
- **Anomaly detection** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
