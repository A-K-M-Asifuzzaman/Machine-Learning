# 04.06 — Exercises: Linear Dimensionality Reduction (PCA)

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the first principal component as the maximizer of $\mathbf{w}^\top\mathbf{C}\mathbf{w}$
subject to $\lVert\mathbf{w}\rVert=1$, and show it is the top eigenvector of the covariance $\mathbf{C}$.

**D2.** Show the second PC maximizes remaining variance subject to orthogonality to the first, and is
the second eigenvector.

**D3.** State the minimum-reconstruction-error objective and prove (via Pythagoras / orthogonal
projection) that it is minimized by the same top-$k$ eigenvectors.

**D4.** Prove that the reconstruction error using $k$ components equals the sum of the discarded
eigenvalues $\sum_{j>k}\lambda_j$.

**D5.** Show that the right singular vectors of the centered $\mathbf{X}$ are the principal components,
and that $\lambda_j = \sigma_j^2/(n-1)$.

**D6.** Explain why forming $\mathbf{X}^\top\mathbf{X}$ squares the condition number, and why the SVD
is numerically preferable.

**D7.** Show that PCA on standardized data is PCA on the correlation matrix, and explain why scaling
matters.

**D8.** Explain the difference between the covariance and correlation matrix formulations, and when to
use each.

**D9.** Define whitening and show it produces features with identity covariance. What is the cost?

**D10.** Explain why PCA captures only linear structure, using the circle or Swiss roll as an example.

**D11.** *(Probabilistic PCA.)* State the latent-Gaussian model whose maximum likelihood recovers PCA,
and relate it to factor analysis.

---

## Tier 2 — Implementation

**I1.** Implement PCA by covariance eigendecomposition and by SVD. Verify components (up to sign),
explained variance, and transform against `sklearn.decomposition.PCA`.

**I2.** Reproduce Experiment 1: verify the Pythagoras identity and that PC1 minimizes reconstruction
error over random directions.

**I3.** Reproduce Experiment 2: plot the scree and cumulative-variance curves and pick $k$ at 95%.

**I4.** Reproduce Experiment 3: confirm reconstruction error equals the sum of discarded eigenvalues.

**I5.** Reproduce Experiment 4: build near-collinear data and show the covariance method losing the
smallest eigenvalue while the SVD keeps it.

**I6.** Reproduce Experiment 5: inflate one feature's scale and show the first PC chasing it; fix with
standardization.

**I7.** Reproduce Experiment 6: add noise to low-rank data and show truncated reconstruction denoising.

**I8.** Reproduce Experiment 7: run PCA on a circle and show no linear 1-D projection preserves it.

**I9.** Implement whitening and verify the transformed features have identity covariance.

**I10.** *(Eigenfaces.)* Run PCA on a face/digit dataset, visualize the top components, and reconstruct
images from $k$ components at several $k$.

**I11.** Implement randomized PCA (via a random projection + SVD) and compare accuracy and speed to
full SVD on a large matrix.

---

## Tier 3 — Interview

**Q1.** What does PCA do, in one sentence?

**Q2.** Derive the first principal component.

**Q3.** Why are max-variance and min-reconstruction the same objective?

**Q4.** How do you compute PCA — covariance or SVD? Why?

**Q5.** How do you choose the number of components?

**Q6.** Do you need to scale features before PCA?

**Q7.** Does PCA select features?

**Q8.** When does PCA fail?

**Q9.** Is high variance the same as importance?

**Q10.** How does PCA relate to SVD? To autoencoders?

**Q11.** What is whitening?

**Q12.** How would you reduce dimension on a nonlinear manifold?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive PCA from both max-variance and min-reconstruction
- [ ] Prove the reconstruction-error identity
- [ ] Compute PCA by SVD and explain its numerical advantage
- [ ] Choose $k$ from explained variance
- [ ] Explain why scaling matters and standardize correctly
- [ ] State PCA's linear limitation and name the nonlinear alternatives
