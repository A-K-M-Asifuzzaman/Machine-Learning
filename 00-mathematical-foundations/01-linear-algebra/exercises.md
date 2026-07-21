# 00.01 — Exercises: Linear Algebra

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).
Solutions are not provided for the derivations — the point is the struggle. Implementation
exercises can be checked against `from_scratch.py`.

---

## Tier 1 — Derivation

**D1.** Prove that the projection matrix $\mathbf{P} = \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}$
satisfies $\mathbf{P}^{2} = \mathbf{P}$ and $\mathbf{P}^{\top} = \mathbf{P}$. Then prove the
converse is what defines an orthogonal projection: any symmetric idempotent matrix projects onto
its own column space.

**D2.** Show $\operatorname{rank}(\mathbf{A}) = \operatorname{rank}(\mathbf{A}^{\top}\mathbf{A})$.
*Hint*: show the two matrices have the same null space, then apply rank-nullity.

**D3.** Derive the normal equations twice — once by setting $\nabla_{\mathbf{w}}\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^2 = 0$,
once by requiring the residual to be orthogonal to $C(\mathbf{X})$. State precisely where each
derivation uses the assumption that $\mathbf{X}$ has full column rank.

**D4.** Prove that eigenvectors of a real symmetric matrix corresponding to *distinct* eigenvalues
are orthogonal. Then explain why this argument fails for a general (non-symmetric) matrix, and
give a $2\times 2$ counterexample with non-orthogonal eigenvectors.

**D5.** Prove $\mathbf{A}^{\top}\mathbf{A} \succeq 0$ for any $\mathbf{A}$. Under what exact
condition is it positive *definite* rather than merely semidefinite?

**D6.** Derive $\nabla_{\mathbf{x}}(\mathbf{x}^{\top}\mathbf{A}\mathbf{x}) = (\mathbf{A}+\mathbf{A}^{\top})\mathbf{x}$
from components, without looking at §14.2. Then derive
$\nabla_{\mathbf{w}}\|\mathbf{y}-\mathbf{X}\mathbf{w}\|_2^2$ using it.

**D7.** Show that the singular values of $\mathbf{A}$ are the square roots of the eigenvalues of
$\mathbf{A}^{\top}\mathbf{A}$, and that $\mathbf{A}$ and $\mathbf{A}^{\top}$ have the same nonzero
singular values.

**D8.** Prove $\kappa(\mathbf{A}^{\top}\mathbf{A}) = \kappa(\mathbf{A})^{2}$ using the SVD.
This is the single most consequential identity in numerical ML.

**D9.** *(Ridge, spectrally.)* Using the SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$,
show that the ridge solution is

$$\hat{\mathbf{w}}_{\text{ridge}} = \sum_{i=1}^{r}\frac{\sigma_i}{\sigma_i^{2}+\lambda}(\mathbf{u}_i^{\top}\mathbf{y})\,\mathbf{v}_i$$

Compare term by term with OLS ($\lambda = 0$). Which directions get shrunk most, and why does
that make statistical sense?

**D10.** Prove the Eckart-Young-Mirsky theorem for the Frobenius norm. *Hint*: use the fact that
the Frobenius norm is invariant under orthogonal transformations.

**D11.** Show that for an orthogonal matrix $\mathbf{Q}$, $\kappa(\mathbf{Q}) = 1$, and that
$\kappa(\mathbf{Q}\mathbf{A}) = \kappa(\mathbf{A})$. Explain in one sentence why this makes
orthogonal transformations the building block of every stable numerical algorithm.

**D12.** *(Why depth needs nonlinearity.)* Prove that a composition of $L$ affine maps
$\mathbf{x}\mapsto \mathbf{W}_L(\dots(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)\dots)+\mathbf{b}_L$
is itself a single affine map. Give the resulting weight matrix and bias explicitly.

---

## Tier 2 — Implementation

Write each in NumPy without using the corresponding `np.linalg` function, then verify against it.

**I1.** Implement `modified_gram_schmidt` from the definition. Verify $\mathbf{Q}^{\top}\mathbf{Q}=\mathbf{I}$
and $\mathbf{Q}\mathbf{R}=\mathbf{A}$ to within $10^{-14}$.

**I2.** Implement classical Gram-Schmidt as well, and reproduce Experiment 1 in `from_scratch.py`.
At what condition number does CGS lose *all* orthogonality? Explain the mechanism.

**I3.** Implement `power_iteration`. Then extend it to find the top-$k$ eigenvectors by
**deflation**: after finding $(\lambda_1, \mathbf{v}_1)$, run it again on
$\mathbf{A} - \lambda_1\mathbf{v}_1\mathbf{v}_1^{\top}$. Verify against `np.linalg.eigh`.

**I4.** Implement PCA via the SVD. On a dataset of your choice, verify that:
(a) components are orthonormal, (b) explained variances equal the covariance eigenvalues,
(c) your `explained_variance_ratio_` matches sklearn's exactly.

**I5.** Empirically confirm $\kappa(\mathbf{X}^{\top}\mathbf{X}) = \kappa(\mathbf{X})^{2}$ over a
range of matrices. Then reproduce Experiment 2: at what $\kappa(\mathbf{X})$ do the normal
equations become useless in float64? Does the answer match the "lose $k$ digits when
$\kappa = 10^{k}$" rule of thumb?

**I6.** Implement image compression by truncated SVD. Plot reconstruction error and compression
ratio against $k$. At what $k$ does the image become visually indistinguishable from the original,
and what fraction of the singular values is that?

**I7.** Implement `pinv` via the SVD. Construct three systems — overdetermined, underdetermined,
and exactly determined — and verify the pseudoinverse gives the least-squares, minimum-norm, and
exact solution respectively.

**I8.** *(The dummy variable trap, measured.)* Build a design matrix that one-hot encodes a
3-category feature into 3 columns *plus* an intercept. Compute $\kappa(\mathbf{X})$ and
$\operatorname{rank}(\mathbf{X})$. Try to solve the normal equations. Now drop one category and
repeat. Explain the difference using §4.2.

**I9.** *(LoRA, in miniature.)* Take a $512 \times 512$ random matrix $\mathbf{W}$ and a target
$\mathbf{W} + \Delta$ where $\Delta$ is genuinely rank-8. Fit an approximation
$\mathbf{B}\mathbf{A}$ with $\mathbf{B}\in\mathbb{R}^{512\times r}$,
$\mathbf{A}\in\mathbb{R}^{r\times 512}$ for $r = 1,2,4,8,16$. Plot error vs $r$ and vs parameter
count. At what $r$ does the error collapse, and why exactly there?

---

## Tier 3 — Interview

Answer in 2-3 sentences, out loud, without notes.

**Q1.** What does the rank of a matrix tell you, in plain language?

**Q2.** Why can't you always solve $\mathbf{A}\mathbf{x} = \mathbf{b}$ exactly? What do you do
instead, and what makes that choice optimal?

**Q3.** Explain the difference between eigendecomposition and SVD. When can you only use SVD?

**Q4.** Why is PCA just an SVD? What is the role of centering, and what goes wrong without it?

**Q5.** Your linear regression returns coefficients of $+10^{6}$ and $-10^{6}$ on two features.
What is happening, how would you confirm it, and what are three fixes?

**Q6.** Why does `sklearn.linear_model.LinearRegression` not use
$(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$?

**Q7.** What does it mean geometrically for a matrix to be positive definite? Why do we care in
optimization?

**Q8.** Explain vanishing and exploding gradients using eigenvalues. Why do residual connections
help?

**Q9.** A colleague suggests speeding up your model by replacing a $4096\times 4096$ weight matrix
with the product of a $4096\times 8$ and an $8\times 4096$ matrix. What are they exploiting, what
do they gain, and what do they risk?

**Q10.** What is the condition number, and what should you do when it is large?

**Q11.** Why is the dot product the right way to measure similarity between embeddings? When is
cosine similarity better, and why?

**Q12.** You have 500 features and 100 training examples. What does that guarantee about the
null space of $\mathbf{X}$, and what does *that* imply about the uniqueness of your fitted
coefficients?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the normal equations from the geometry in one line
- [ ] Explain what breaks when features are collinear — at the level of subspaces, not vibes
- [ ] State what the SVD gives you about all four fundamental subspaces
- [ ] Explain why $\ell_1$ produces sparsity using the shape of its unit ball
- [ ] Predict, before running it, whether a matrix computation will be numerically stable
- [ ] Implement PCA from scratch without looking anything up
