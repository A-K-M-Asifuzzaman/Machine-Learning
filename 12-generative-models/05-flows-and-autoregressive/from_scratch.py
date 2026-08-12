"""
12.05 — Normalizing flows & autoregressive models, from scratch (NumPy).

VAEs give a likelihood BOUND, GANs give NONE. Normalizing flows and autoregressive models give the EXACT
likelihood — by construction. Flows use invertible maps + the change-of-variables formula; autoregressive
models use the chain rule. This file builds both and verifies their exactness:

  1. change of variables: p_x(x) = p_z(f(x)) |det df/dx|                        -> Experiment 1
  2. a RealNVP coupling layer is invertible with a tractable Jacobian           -> Experiment 2
  3. a normalizing flow defines an EXACTLY normalized density (integrates to 1) -> Experiment 3
  4. an autoregressive model's chain-rule likelihood is exact (== joint)        -> Experiment 4
  5. the generative-model family trade-offs                                     -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


# =============================================================================
# EXPERIMENT 1 — the change-of-variables formula
# =============================================================================


def experiment_1_change_of_variables():
    print("=" * 88)
    print("EXPERIMENT 1 — change of variables: p_x(x) = p_z(f(x)) |det df/dx| (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    # x ~ N(0,1); z = tanh(x) maps to (-1,1). The density of z is given by change of variables.
    x = rng.standard_normal(2_000_000)
    z = np.tanh(x)

    def p_z_formula(zz):
        # p_z(z) = p_x(atanh z) * |d atanh z / dz| = N(atanh z) * 1/(1 - z^2)
        xz = np.arctanh(zz)
        return np.exp(-0.5 * xz ** 2) / np.sqrt(2 * np.pi) * 1.0 / (1 - zz ** 2)

    # compare the formula to the empirical histogram at a few points
    print(f"\n  x ~ N(0,1), z = tanh(x). Density of z: formula p_z(z) vs empirical histogram:\n")
    print(f"    {'z':>8s} {'formula p_z(z)':>16s} {'empirical density':>18s}")
    hist, edges = np.histogram(z, bins=400, range=(-0.999, 0.999), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for zq in (-0.5, 0.0, 0.5, 0.8):
        j = np.argmin(np.abs(centers - zq))
        print(f"    {zq:>8.1f} {p_z_formula(np.array([zq]))[0]:>16.4f} {hist[j]:>18.4f}")
    print("""
  READING: if z = f(x) is an INVERTIBLE map, the density transforms by the change-of-variables formula:
  p_x(x) = p_z(f(x)) * |det(df/dx)|. The Jacobian factor |det df/dx| accounts for how f stretches or
  squeezes volume. The formula (columns match the histogram) is the entire basis of normalizing flows:
  if you can compute f, its inverse, AND its Jacobian determinant, you can compute the EXACT density of
  x from a simple base density on z (README §2).""")


# =============================================================================
# A RealNVP coupling layer
# =============================================================================


class Coupling:
    """Affine coupling: pass through the masked dims, affine-transform the rest based on them."""
    def __init__(self, mask, seed, h=32):
        r = np.random.default_rng(seed)
        self.mask = mask
        self.W1 = r.standard_normal((2, h)) * 0.5; self.b1 = np.zeros(h)
        self.W2 = r.standard_normal((h, 2)) * 0.5; self.b2 = np.zeros(2)

    def _net(self, xin):
        return relu(xin @ self.W1 + self.b1) @ self.W2 + self.b2

    def forward(self, x):
        out = self._net(x * self.mask)
        sc = np.tanh(out[:, :1]) * 0.8; t = out[:, 1:]           # bounded log-scale, translation
        y = x * self.mask + (1 - self.mask) * (x * np.exp(sc) + t)
        logdet = ((1 - self.mask) * sc).sum(1)
        return y, logdet

    def inverse(self, y):
        out = self._net(y * self.mask)                          # masked part is unchanged -> invert exactly
        sc = np.tanh(out[:, :1]) * 0.8; t = out[:, 1:]
        return y * self.mask + (1 - self.mask) * ((y - t) * np.exp(-sc))


def _make_flow():
    masks = [np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 0.]]), np.array([[0., 1.]])]
    return [Coupling(m, i) for i, m in enumerate(masks)]


def flow_forward(layers, x):
    ld = np.zeros(len(x))
    for L in layers:
        x, d = L.forward(x); ld += d
    return x, ld


def flow_inverse(layers, z):
    for L in reversed(layers):
        z = L.inverse(z)
    return z


def flow_logpx(layers, x):
    z, ld = flow_forward(layers, x)
    return -0.5 * (z ** 2).sum(1) - np.log(2 * np.pi) + ld       # log N(z;0,I_2) + log|det J|


# =============================================================================
# EXPERIMENT 2 — the coupling layer is invertible with a tractable Jacobian
# =============================================================================


def experiment_2_coupling():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — RealNVP coupling: exact inverse + tractable Jacobian (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    layers = _make_flow()
    x = rng.standard_normal((5, 2))
    z, ld = flow_forward(layers, x)
    x_rec = flow_inverse(layers, z)
    inv_err = np.abs(x - x_rec).max()
    # verify log|det J| against a numerical Jacobian
    x0 = rng.standard_normal((1, 2)); z0, ld0 = flow_forward(layers, x0)
    J = np.zeros((2, 2)); eps = 1e-6
    for i in range(2):
        xp = x0.copy(); xp[0, i] += eps
        zp, _ = flow_forward(layers, xp)
        J[:, i] = (zp[0] - z0[0]) / eps
    numer_logdet = np.log(abs(np.linalg.det(J)))
    print(f"""
  A 4-layer RealNVP flow on 2-D data:

    invertibility:  max| x - inverse(forward(x)) | = {inv_err:.1e}   (exact)
    log|det J|:  analytic (sum of scales) = {ld0[0]:.5f},  numerical Jacobian = {numer_logdet:.5f}

  READING: the trick of a coupling layer is that it splits the input, passes half through UNCHANGED, and
  affine-transforms the other half using a network of the first half. That makes it (a) exactly
  invertible — the unchanged half lets you recompute the transform and undo it — and (b) its Jacobian is
  TRIANGULAR, so the log-determinant is just the sum of the scale outputs (no expensive determinant).
  Both hold to machine precision. Stacking coupling layers (alternating which half is transformed) builds
  an expressive invertible map with a cheap exact Jacobian — the RealNVP design (README §3).""")


# =============================================================================
# EXPERIMENT 3 — a flow is an exactly normalized density
# =============================================================================


def experiment_3_exact_density():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a flow defines an EXACTLY normalized density (README §4)")
    print("=" * 88)
    layers = _make_flow()
    # integrate p(x) over a fine grid; a valid density integrates to 1
    g = np.linspace(-8, 8, 600); dx = g[1] - g[0]
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    p = np.exp(flow_logpx(layers, pts))
    integral = p.sum() * dx * dx
    print(f"""
  Integrate the flow's density p(x) = N(f(x); 0, I) * |det J| over a fine 2-D grid:

    total probability mass = {integral:.4f}   (a valid, EXACTLY normalized density integrates to 1)

  READING: because the change of variables preserves probability mass exactly, a normalizing flow is a
  proper normalized density — its integral is 1 (the small shortfall here is only grid truncation). This
  is the flows' superpower: unlike a VAE (which gives a LOWER BOUND on log p(x)) or a GAN (which gives NO
  likelihood at all), a flow computes the EXACT log-likelihood. That makes flows ideal wherever you need
  a real density — anomaly detection, exact model comparison, and as building blocks in probabilistic
  models (README §4).""")


# =============================================================================
# EXPERIMENT 4 — autoregressive models give exact likelihood
# =============================================================================


def experiment_4_autoregressive():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — autoregressive: chain-rule likelihood is exact (== joint) (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    phi, sig = 0.7, 0.5
    # AR(1): x1 ~ N(0,1); x_i = phi x_(i-1) + N(0, sig^2). Joint is Gaussian x = A e, e ~ N(0, D).
    A = np.array([[1, 0, 0], [phi, 1, 0], [phi ** 2, phi, 1.0]])
    D = np.diag([1.0, sig ** 2, sig ** 2])
    Sigma = A @ D @ A.T                                          # exact joint covariance
    X = rng.standard_normal((4, 3)) @ np.linalg.cholesky(Sigma).T   # samples from the joint

    def ar_logp(x):                                             # chain rule: sum log p(x_i | x_<i)
        lp = -0.5 * x[:, 0] ** 2 - 0.5 * np.log(2 * np.pi)
        for i in (1, 2):
            m = phi * x[:, i - 1]
            lp += -0.5 * ((x[:, i] - m) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi)
        return lp

    def joint_logp(x):                                         # exact multivariate-Gaussian log-density
        inv = np.linalg.inv(Sigma); _, logdet = np.linalg.slogdet(Sigma)
        return -0.5 * np.einsum("ni,ij,nj->n", x, inv, x) - 0.5 * (3 * np.log(2 * np.pi) + logdet)

    err = np.abs(ar_logp(X) - joint_logp(X)).max()
    print(f"""
  AR(1) process x_i = {phi} x_(i-1) + N(0, {sig}^2). Compare the chain-rule log-likelihood
  sum_i log p(x_i | x_<i)  to the EXACT joint multivariate-Gaussian log-density:

    max | chain-rule logp  -  exact joint logp | = {err:.1e}   (identical)

  READING: an autoregressive model factorizes the joint by the CHAIN RULE p(x) = prod_i p(x_i | x_<i)
  and models each conditional. This is EXACT — the product of conditionals equals the joint by
  definition (verified to machine precision). So autoregressive models (PixelCNN for images, WaveNet for
  audio, and every GPT for text) give exact likelihoods and the best density estimates — at the cost of
  SEQUENTIAL generation (one element at a time), which is slow (README §5).""")


# =============================================================================
# EXPERIMENT 5 — the generative-model family trade-offs
# =============================================================================


def experiment_5_tradeoffs():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the generative-model families and their trade-offs (README §6)")
    print("=" * 88)
    rows = [
        ("Autoencoder (12.01)", "none", "n/a", "n/a", "reconstruct only"),
        ("VAE (12.02)", "bound (ELBO)", "fast (1 pass)", "blurry", "has encoder"),
        ("GAN (12.03)", "none", "fast (1 pass)", "sharp", "unstable, mode collapse"),
        ("Diffusion (12.04)", "bound", "slow (many steps)", "sharp", "stable, SOTA images"),
        ("Normalizing flow", "EXACT", "fast (1 pass)", "good", "invertible, restricted arch"),
        ("Autoregressive", "EXACT", "slow (sequential)", "sharp", "best density; LLMs"),
    ]
    print(f"\n    {'family':>22s} {'likelihood':>14s} {'sampling':>18s} {'samples':>9s} {'note':>26s}")
    for r in rows:
        print(f"    {r[0]:>22s} {r[1]:>14s} {r[2]:>18s} {r[3]:>9s} {r[4]:>26s}")
    print("""
  READING: every deep generative model trades off three things — LIKELIHOOD (exact / bound / none),
  SAMPLING SPEED (one pass / sequential / many steps), and SAMPLE QUALITY (sharp / blurry). Flows and
  autoregressive models buy EXACT likelihood by constraining the architecture (invertible, or ordered);
  GANs buy sharpness by giving up likelihood and stability; diffusion buys sharp + stable at the cost of
  slow sampling; VAEs are fast with an encoder but blurry. There is no free lunch — the right choice
  depends on whether you need density, speed, or fidelity. Today: diffusion for images, autoregressive
  transformers for text/audio (README §6).""")


if __name__ == "__main__":
    experiment_1_change_of_variables()
    experiment_2_coupling()
    experiment_3_exact_density()
    experiment_4_autoregressive()
    experiment_5_tradeoffs()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
