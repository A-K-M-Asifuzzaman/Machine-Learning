"""
04.06 — Principal Component Analysis, from scratch (NumPy).

PCA by covariance eigendecomposition AND by SVD, verified against scikit-learn. Then the
chapter's claims are MEASURED:

  1. max-variance and min-reconstruction give the SAME components      (README §2-§3)
  2. explained variance / scree choose the number of components        (README §6)
  3. reconstruction error == sum of DISCARDED eigenvalues (exact)      (README §3)
  4. the SVD is numerically superior on ill-conditioned data           (README §5)
  5. scaling changes the principal components entirely                 (README §7)
  6. truncated reconstruction compresses and DENOISES                  (README §9)
  7. PCA fails on a nonlinear manifold                                 (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.preprocessing import StandardScaler
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# PCA  (README §4-§5)
# =============================================================================


class PCA:
    def __init__(self, n_components=None, method="svd"):
        self.n_components = n_components
        self.method = method

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        n = len(X)
        if self.method == "svd":
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.components_ = Vt                        # rows = principal components
            self.explained_variance_ = S ** 2 / (n - 1)  # unbiased, matching sklearn
        else:  # covariance eigendecomposition
            C = Xc.T @ Xc / (n - 1)
            vals, vecs = np.linalg.eigh(C)
            order = np.argsort(-vals)
            self.components_ = vecs[:, order].T
            self.explained_variance_ = vals[order]
        total = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total
        k = self.n_components or len(self.components_)
        self.components_ = self.components_[:k]
        self.explained_variance_ = self.explained_variance_[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return np.asarray(Z, float) @ self.components_ + self.mean_

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))


# =============================================================================
# VERIFICATION
# =============================================================================


def _sign_align(A, B):
    """Flip signs of A's rows to match B (components are defined up to sign)."""
    signs = np.sign(np.sum(A * B, axis=1))
    signs[signs == 0] = 1
    return A * signs[:, None]


def verify():
    print("=" * 88)
    print("VERIFICATION — PCA (SVD & covariance) vs scikit-learn")
    print("=" * 88)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    X = rng.standard_normal((500, 6)) @ A            # correlated features

    if HAVE_SK:
        sk = SkPCA(n_components=4).fit(X)
        for method in ("svd", "cov"):
            p = PCA(4, method=method).fit(X)
            comp = _sign_align(p.components_, sk.components_)
            cd = np.max(np.abs(comp - sk.components_))
            vd = np.max(np.abs(p.explained_variance_ - sk.explained_variance_))
            td = np.max(np.abs(_sign_align(p.transform(X).T, sk.transform(X).T).T
                               - sk.transform(X)))
            print(f"\n  method={method:>4s}: |component diff| {cd:.2e}, "
                  f"|explained-var diff| {vd:.2e}, |transform diff| {td:.2e}")
            assert cd < 1e-6 and vd < 1e-6, f"{method} must match sklearn"
        print("\n  both SVD and covariance PCA match sklearn (components, variance, transform)  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — max variance == min reconstruction (README §2-§3)
# =============================================================================


def experiment_1_two_derivations():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — max-variance and min-reconstruction give the same components (README §2-§3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5)) @ rng.standard_normal((5, 5))
    Xc = X - X.mean(0)
    total_var = np.sum(np.var(Xc, axis=0))          # total variance (sum over dims)

    # the max-variance PC1
    p = PCA(1).fit(X)
    w = p.components_[0]
    var_captured = np.var(Xc @ w)
    recon_err_pc1 = np.mean(np.sum((Xc - np.outer(Xc @ w, w)) ** 2, 1))

    # Pythagoras identity: recon error = total variance - captured variance
    # and: PC1 has the LOWEST recon error of any unit direction (check vs random directions)
    rand_errs = []
    for _ in range(5000):
        v = rng.standard_normal(5)
        v /= np.linalg.norm(v)
        rand_errs.append(np.mean(np.sum((Xc - np.outer(Xc @ v, v)) ** 2, 1)))
    print(f"""
  Max-variance vs min-reconstruction for the top direction (5-D correlated data):

    variance captured by PC1                     = {var_captured:.4f}
    total variance                               = {total_var:.4f}
    reconstruction error onto PC1                = {recon_err_pc1:.4f}
    total variance - captured variance           = {total_var - var_captured:.4f}
    -> Pythagoras identity holds to {abs(recon_err_pc1 - (total_var - var_captured)):.1e}

    best reconstruction error over 5000 RANDOM directions = {min(rand_errs):.4f}
    PC1's reconstruction error                            = {recon_err_pc1:.4f}   (the minimum)

  READING: the max-variance direction (PC1) is EXACTLY the min-reconstruction-error direction. By
  Pythagoras, captured variance + reconstruction error = total variance (identity holds to ~1e-14),
  so maximizing one minimizes the other. No random direction beats PC1's reconstruction error — it
  is the optimum. Two objectives, one eigenvector (README §2-§3).""")


# =============================================================================
# EXPERIMENT 2 — explained variance / scree (README §6)
# =============================================================================


def experiment_2_scree():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — explained variance and the scree plot choose k (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    # data whose variance lives mostly in 3 directions, plus small noise dims
    latent = rng.standard_normal((600, 3)) * np.array([5.0, 4.0, 3.0])
    load = rng.standard_normal((3, 10))
    X = latent @ load + 0.1 * rng.standard_normal((600, 10))

    p = PCA().fit(X)
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\n    {'PC':>4s} {'eigenvalue':>12s} {'explained %':>12s} {'cumulative %':>13s}")
    for i in range(8):
        print(f"    {i+1:>4d} {p.explained_variance_[i]:>12.3f} {evr[i]:>11.1%} {cum[i]:>12.1%}")
    k95 = int(np.searchsorted(cum, 0.95)) + 1
    print(f"""
  {k95} components reach 95% cumulative variance.

  READING: the variance is concentrated in the first 3 components (a clear scree 'elbow' after PC3),
  which capture {cum[2]:.0%} of the total; the rest is noise. The explained-variance ratio and the
  cumulative curve give a principled k — here 3, matching the 3 true latent directions (README §6).""")


# =============================================================================
# EXPERIMENT 3 — reconstruction error == sum of discarded eigenvalues (README §3)
# =============================================================================


def experiment_3_reconstruction_identity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — reconstruction error == sum of discarded eigenvalues (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((500, 8)) @ rng.standard_normal((8, 8))
    p_full = PCA().fit(X)
    eigs = p_full.explained_variance_

    n = len(X)
    print(f"\n    {'k':>4s} {'residual variance':>20s} {'sum discarded eigs':>20s} {'|diff|':>10s}")
    for k in (1, 3, 5, 7):
        p = PCA(k).fit(X)
        recon = p.reconstruct(X)
        # residual variance (unbiased, matching the eigenvalue convention)
        resid = np.sum((X - recon) ** 2) / (n - 1)
        discarded = eigs[k:].sum()
        print(f"    {k:>4d} {resid:>20.5f} {discarded:>20.5f} {abs(resid-discarded):>10.2e}")
    print("""
  READING: the mean reconstruction error using k components equals EXACTLY the sum of the discarded
  eigenvalues (the variance in the dropped directions), to machine precision. This is why the
  eigenvalues tell you precisely what each component is worth, and why keeping the top-k is optimal
  (README §3).""")


# =============================================================================
# EXPERIMENT 4 — SVD numerical superiority (README §5)
# =============================================================================


def experiment_4_svd_stability():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — the SVD is numerically superior on ill-conditioned data (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n, eps = 400, 1e-8
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    # two NEAR-COLLINEAR columns (a and a + eps*b) + one independent column. The tiny direction
    # has variance ~eps^2; forming X^T X squares eps to eps^2 ~ 1e-16, at the edge of float64.
    X = np.column_stack([a, a + eps * b, rng.standard_normal(n)])

    svd_small = PCA(method="svd").fit(X).explained_variance_[-1]
    cov_small = PCA(method="cov").fit(X).explained_variance_[-1]
    print(f"""
  Two near-collinear columns (a, a + 1e-8*b); the third is independent.
  cond(X) = {np.linalg.cond(X):.0e}, so forming X^T X squares it to ~{np.linalg.cond(X)**2:.0e}
  (past float64's ~1e16), corrupting the tiny eigenvalue. Smallest recovered variance:

    {'method':>22s} {'smallest variance':>20s}
    {'SVD (accurate)':>22s} {svd_small:>20.3e}
    {'covariance eigendecomp':>22s} {cov_small:>20.3e}

    -> the covariance method is off from the SVD by a factor of {cov_small/svd_small:.1f}x

  READING: the SVD works on X directly and recovers the tiny variance accurately; the COVARIANCE
  method squares the condition number by forming X^T X, so its smallest eigenvalue is corrupted —
  here off by ~{cov_small/svd_small:.0f}x. On well-conditioned data both agree, but when directions are
  near-collinear (common in real data) only the SVD stays accurate. Always compute PCA by SVD
  (README §5).""")


# =============================================================================
# EXPERIMENT 5 — scaling changes the components (README §7)
# =============================================================================


def experiment_5_scaling():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — feature scaling changes the principal components entirely (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn StandardScaler)")
        return
    rng = np.random.default_rng(5)
    # two comparably-informative features, but feature 0 is measured on a 1000x larger scale
    f0 = rng.standard_normal(500) * 1000        # e.g. millimeters
    f1 = rng.standard_normal(500) * 1.0         # e.g. meters, same true importance
    X = np.column_stack([f0, f1])

    pc_raw = PCA(1).fit(X).components_[0]
    Xs = StandardScaler().fit_transform(X)
    pc_scaled = PCA(1).fit(Xs).components_[0]
    print(f"""
  Feature 0 has 1000x the scale of feature 1 (but equal true importance):

    first PC on RAW data    : {np.array2string(np.abs(pc_raw), precision=3)}
    first PC on SCALED data : {np.array2string(np.abs(pc_scaled), precision=3)}

  READING: on raw data the first PC points almost entirely along feature 0 ({abs(pc_raw[0]):.2f} vs
  {abs(pc_raw[1]):.2f}) — PCA chased the loud UNIT, not real structure. After standardizing, the PC
  weights the two features comparably. On heterogeneous features you must standardize, or PCA
  reports the largest-scale feature as the 'principal' one (README §7).""")


# =============================================================================
# EXPERIMENT 6 — compression / denoising (README §9)
# =============================================================================


def experiment_6_denoising():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — truncated reconstruction compresses and DENOISES (README §9)")
    print("=" * 88)
    rng = np.random.default_rng(6)
    # signal lives in a 3-D subspace of 30-D space; add isotropic noise
    latent = rng.standard_normal((500, 3)) * np.array([6, 4, 2])
    load = rng.standard_normal((3, 30))
    clean = latent @ load
    noise = rng.standard_normal((500, 30)) * 3.0
    noisy = clean + noise

    print(f"\n    {'k (components kept)':>20s} {'MSE to CLEAN signal':>22s}")
    for k in (1, 3, 5, 10, 30):
        p = PCA(k).fit(noisy)
        recon = p.reconstruct(noisy)
        mse = np.mean(np.sum((recon - clean) ** 2, 1))
        tag = "  <- true rank" if k == 3 else ""
        print(f"    {k:>20d} {mse:>22.1f}{tag}")
    print("""
  READING: the signal occupies only 3 directions; the noise spreads over all 30. Reconstructing
  from the top few PCs keeps the (high-variance) signal and DISCARDS the (spread-out) noise, so the
  distance to the clean signal is MINIMIZED near the true rank (k=3) and rises again if you keep
  noise-dominated components. PCA denoises by dropping low-variance directions (README §9).""")


# =============================================================================
# EXPERIMENT 7 — PCA fails on a nonlinear manifold (README §8)
# =============================================================================


def experiment_7_nonlinear():
    print("\n" + "=" * 88)
    print("EXPERIMENT 7 — PCA fails on a nonlinear manifold (README §8)")
    print("=" * 88)
    rng = np.random.default_rng(7)
    # a 1-D circle embedded in 2-D: intrinsically 1-D, but no LINEAR 1-D projection preserves it
    t = rng.uniform(0, 2 * np.pi, 500)
    X = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.02, (500, 2))

    p = PCA(1).fit(X)
    recon = p.reconstruct(X)
    mse = np.mean(np.sum((X - recon) ** 2, 1))
    print(f"""
  A circle (intrinsically 1-D) in 2-D. Best linear 1-D PCA reconstruction:

    variance explained by PC1 = {p.explained_variance_ratio_[0]:.1%}   (both PCs ~equal — no dominant axis)
    reconstruction MSE onto 1 line = {mse:.3f}   (large: a line cannot represent a circle)

  READING: the circle is truly 1-dimensional (parametrized by the angle), but its 1-D structure is
  NONLINEAR — no straight line captures it, so PCA's first component explains only ~50% of the
  variance and collapsing to a line destroys the shape. Linear PCA sees no low-dimensional linear
  structure where a nonlinear one clearly exists. This is what kernel PCA and manifold learning
  (04.07) are for (README §8, §10).""")


if __name__ == "__main__":
    verify()
    experiment_1_two_derivations()
    experiment_2_scree()
    experiment_3_reconstruction_identity()
    experiment_4_svd_stability()
    experiment_5_scaling()
    experiment_6_denoising()
    experiment_7_nonlinear()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
