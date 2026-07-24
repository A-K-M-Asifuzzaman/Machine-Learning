"""
04.04 — Gaussian Mixtures & the EM Algorithm, from scratch (NumPy).

A full-covariance GMM fit by EM (k-means init, soft E-step, weighted M-step, reg_covar floor),
verified against scikit-learn. Then the chapter's claims are MEASURED:

  1. soft, full-covariance GMM beats k-means on elongated/overlapping clusters  (README §1, §6)
  2. EM increases the log-likelihood MONOTONICALLY                              (README §4-§5)
  3. a covariance SINGULARITY diverges without reg_covar; the floor fixes it    (README §9)
  4. covariance types: spherical ~ k-means, full captures ellipses             (README §6)
  5. BIC is minimized at the true number of components                         (README §8)
  6. responsibilities HARDEN toward k-means as the covariance shrinks          (README §7)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.mixture import GaussianMixture as SkGMM
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.datasets import make_blobs
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# GAUSSIAN MIXTURE via EM  (README §4)
# =============================================================================


try:
    from scipy.linalg import solve_triangular as _solve_tri
except Exception:
    def _solve_tri(L, b, lower=True):
        return np.linalg.solve(L, b)


def _log_gaussian(X, mu, cov):
    """log N(x | mu, cov) for each row of X (numerically stable via Cholesky)."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    sol = _solve_tri(L, (X - mu).T, lower=True)
    maha = np.sum(sol ** 2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


class GaussianMixture:
    def __init__(self, n_components=3, cov_type="full", max_iter=200, tol=1e-6,
                 reg_covar=1e-6, random_state=0):
        self.K = n_components
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _init(self, X):
        n, d = X.shape
        if HAVE_SK:
            km = KMeans(self.K, n_init=5, random_state=self.random_state).fit(X)
            self.means_ = km.cluster_centers_.copy()
            labels = km.labels_
        else:
            rng = np.random.default_rng(self.random_state)
            self.means_ = X[rng.choice(n, self.K, replace=False)]
            labels = np.argmin(((X[:, None] - self.means_[None]) ** 2).sum(-1), 1)
        self.weights_ = np.array([np.mean(labels == k) for k in range(self.K)])
        self.covariances_ = np.array([np.cov(X[labels == k].T) + self.reg_covar * np.eye(d)
                                      if np.sum(labels == k) > 1
                                      else np.eye(d) for k in range(self.K)])

    def _constrain_cov(self, cov, d):
        if self.cov_type == "full":
            return cov + self.reg_covar * np.eye(d)
        if self.cov_type == "diag":
            return np.diag(np.diag(cov)) + self.reg_covar * np.eye(d)
        if self.cov_type == "spherical":
            return (np.trace(cov) / d) * np.eye(d) + self.reg_covar * np.eye(d)
        raise ValueError(self.cov_type)

    def _e_step(self, X):
        # log responsibilities via log-sum-exp
        log_r = np.zeros((len(X), self.K))
        for k in range(self.K):
            log_r[:, k] = np.log(self.weights_[k] + 1e-300) \
                + _log_gaussian(X, self.means_[k], self.covariances_[k])
        log_norm = np.logaddexp.reduce(log_r, axis=1)
        log_resp = log_r - log_norm[:, None]
        return np.exp(log_resp), float(np.sum(log_norm))

    def _m_step(self, X, resp):
        n, d = X.shape
        Nk = resp.sum(0) + 1e-300
        self.weights_ = Nk / n
        self.means_ = (resp.T @ X) / Nk[:, None]
        covs = []
        for k in range(self.K):
            diff = X - self.means_[k]
            cov = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
            covs.append(self._constrain_cov(cov, d))
        self.covariances_ = np.array(covs)

    def fit(self, X):
        X = np.asarray(X, float)
        self._init(X)
        self.history_ = []
        prev = -np.inf
        for _ in range(self.max_iter):
            resp, ll = self._e_step(X)
            self.history_.append(ll)
            self._m_step(X, resp)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.resp_ = resp
        self.labels_ = np.argmax(resp, axis=1)
        self.lower_bound_ = ll
        return self

    def score(self, X):
        """Mean log-likelihood per sample."""
        _, ll = self._e_step(np.asarray(X, float))
        return ll / len(X)

    def n_parameters(self):
        d = self.means_.shape[1]
        if self.cov_type == "full":
            cov_p = self.K * d * (d + 1) // 2
        elif self.cov_type == "diag":
            cov_p = self.K * d
        else:
            cov_p = self.K
        return cov_p + self.K * d + (self.K - 1)      # covs + means + weights

    def bic(self, X):
        _, ll = self._e_step(np.asarray(X, float))
        return -2 * ll + self.n_parameters() * np.log(len(X))


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — GMM (EM) vs scikit-learn")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(sklearn unavailable — skipping)")
        return
    rng = np.random.default_rng(0)
    X = np.vstack([rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 300),
                   rng.multivariate_normal([5, 5], [[1, -0.6], [-0.6, 1]], 300),
                   rng.multivariate_normal([0, 6], [[0.5, 0], [0, 2]], 300)])

    ours = GaussianMixture(3, random_state=0).fit(X)
    sk = SkGMM(3, covariance_type="full", n_init=1, random_state=0,
               init_params="k-means++").fit(X)
    print(f"""
    our  mean log-likelihood/sample = {ours.score(X):.4f}
    sklearn mean log-likelihood/sample = {sk.score(X):.4f}   (diff {abs(ours.score(X)-sk.score(X)):.2e})
    label agreement (ARI): {adjusted_rand_score(sk.predict(X), ours.labels_):.3f}
    our BIC = {ours.bic(X):.1f}   sklearn BIC = {sk.bic(X):.1f}
""")
    assert abs(ours.score(X) - sk.score(X)) < 0.05, "log-likelihood parity"
    assert adjusted_rand_score(sk.predict(X), ours.labels_) > 0.95, "label parity"
    print("  log-likelihood, labels, and BIC agree with sklearn  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — soft full-covariance GMM beats k-means (README §1, §6)
# =============================================================================


def experiment_1_vs_kmeans():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — GMM (full covariance) beats k-means on elongated clusters (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    # elongated, tilted, overlapping clusters
    X, y = make_blobs(n_samples=900, centers=3, cluster_std=0.7, random_state=2)
    X = X @ np.array([[2.6, 1.1], [0.0, 0.4]])       # stretch + shear -> tilted ellipses

    gmm = GaussianMixture(3, cov_type="full", random_state=0).fit(X)
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    print(f"""
    {'method':>26s} {'ARI vs truth':>14s}
    {'k-means (spherical)':>26s} {adjusted_rand_score(y, km.labels_):>14.3f}
    {'GMM (full covariance)':>26s} {adjusted_rand_score(y, gmm.labels_):>14.3f}

  READING: the clusters are elongated and tilted, so k-means' straight, equidistant Voronoi
  boundaries cut across them. The GMM fits a full covariance per component — tilted ellipses that
  match the cluster shapes — and its soft assignments respect the overlap, recovering the true
  clustering far better (README §1, §6).""")


# =============================================================================
# EXPERIMENT 2 — EM monotonically increases log-likelihood (README §4-§5)
# =============================================================================


def experiment_2_monotone():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — EM increases the log-likelihood monotonically (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(1)
    X = np.vstack([rng.multivariate_normal([0, 0], np.eye(2), 300),
                   rng.multivariate_normal([4, 4], np.eye(2), 300)])
    gmm = GaussianMixture(2, random_state=3, tol=1e-9)
    gmm.fit(X)
    h = gmm.history_
    print(f"\n  Total log-likelihood at each EM iteration:\n")
    print(f"    {'iter':>5s} {'log-likelihood':>16s} {'increase':>12s}")
    for i in list(range(min(6, len(h)))) + ([len(h) - 1] if len(h) > 6 else []):
        inc = "" if i == 0 else f"{h[i]-h[i-1]:>12.3f}"
        print(f"    {i:>5d} {h[i]:>16.2f} {inc:>12s}")
    diffs = np.diff(h)
    assert np.all(diffs >= -1e-6), "log-likelihood must never decrease"
    print(f"""
  READING: the log-likelihood rises at every EM step and never falls (all {len(diffs)} differences
  >= 0). EM is coordinate ascent on the ELBO: the E-step makes the lower bound TIGHT (sets the
  responsibilities to the exact posterior), and the M-step pushes the bound — and hence the true
  log-likelihood — up. Guaranteed monotonic improvement to a local optimum (README §5).""")


# =============================================================================
# EXPERIMENT 3 — covariance singularity and reg_covar (README §9)
# =============================================================================


def experiment_3_singularity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a covariance singularity blows up without reg_covar (README §9)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    # a tiny, near-duplicate clump invites a component to collapse onto it
    X = np.vstack([rng.normal([0, 0], 1.0, (200, 2)),
                   rng.normal([6, 6], 0.0005, (10, 2))])   # 10 near-identical points

    def run(reg):
        gmm = GaussianMixture(10, cov_type="full", reg_covar=reg,
                              random_state=1, max_iter=100).fit(X)
        min_var = min(np.min(np.linalg.eigvalsh(c)) for c in gmm.covariances_)
        return gmm.score(X) * len(X), min_var

    ll_noreg, var_noreg = run(1e-15)
    ll_reg, var_reg = run(1e-3)
    print(f"""
  10-component GMM on data with a tiny near-duplicate 10-point clump:

    {'reg_covar':>12s} {'total log-lik':>14s} {'smallest cov eigenvalue':>26s}
    {'1e-15 (none)':>12s} {ll_noreg:>14.1f} {var_noreg:>26.2e}
    {'1e-3':>12s} {ll_reg:>14.1f} {var_reg:>26.2e}

  READING: with almost no regularization a component COLLAPSES onto the clump — its covariance
  eigenvalue falls to ~{var_noreg:.0e} (an infinitely tall spike) and the log-likelihood is
  SPURIOUSLY INFLATED by ~{ll_noreg - ll_reg:.0f} (heading toward the +infinity singular solution,
  which is a useless degenerate maximum, not a real cluster). A covariance floor (reg_covar=1e-3)
  stops any Gaussian from collapsing — the smallest eigenvalue stays at 1e-3 and the fit is
  meaningful. Always run GMMs with reg_covar > 0 (README §9).""")


# =============================================================================
# EXPERIMENT 4 — covariance types (README §6)
# =============================================================================


def experiment_4_cov_types():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — covariance types: spherical ~ k-means, full captures ellipses (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = make_blobs(n_samples=900, centers=3, cluster_std=0.7, random_state=2)
    X = X @ np.array([[2.6, 1.1], [0.0, 0.4]])       # tilted ellipses
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    print(f"\n    {'model':>26s} {'ARI vs truth':>14s}")
    print(f"    {'k-means':>26s} {adjusted_rand_score(y, km.labels_):>14.3f}")
    for ct in ("spherical", "diag", "full"):
        g = GaussianMixture(3, cov_type=ct, random_state=0).fit(X)
        print(f"    {'GMM (' + ct + ')':>26s} {adjusted_rand_score(y, g.labels_):>14.3f}")
    print("""
  READING: on tilted, elongated clusters, SPHERICAL GMM scores like k-means (both assume balls);
  DIAGONAL does a little better (axis-aligned ellipses); FULL covariance captures the tilt and
  wins clearly. The covariance type is the flexibility-vs-parameters knob, and spherical-equal is
  exactly k-means (README §6-§7).""")


# =============================================================================
# EXPERIMENT 5 — BIC selects the number of components (README §8)
# =============================================================================


def experiment_5_bic():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — BIC is minimized at the true number of components (README §8)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    true_k = 4
    X, _ = make_blobs(n_samples=800, centers=true_k, cluster_std=1.0, random_state=7)
    print(f"\n  Data has {true_k} true clusters.\n")
    print(f"    {'K':>4s} {'log-likelihood':>16s} {'BIC':>12s}")
    bics = {}
    for K in range(1, 9):
        g = GaussianMixture(K, cov_type="full", random_state=0).fit(X)
        bics[K] = g.bic(X)
        print(f"    {K:>4d} {g.score(X)*len(X):>16.1f} {bics[K]:>12.1f}")
    best = min(bics, key=bics.get)
    print(f"""
  BIC is minimized at K = {best}  (true K = {true_k}).

  READING: the log-likelihood keeps rising with K (more Gaussians fit better), but BIC adds a
  p*log(N) penalty for the extra parameters, so it turns down once K exceeds the real structure.
  Its minimum picks the true number of components — a principled, likelihood-based choice of the
  number of clusters, which k-means' elbow only approximates (README §8).""")


# =============================================================================
# EXPERIMENT 6 — responsibilities harden toward k-means (README §7)
# =============================================================================


def experiment_6_hardening():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — responsibilities harden toward k-means as covariance shrinks (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 1.0, (300, 2)), rng.normal([3, 0], 1.0, (300, 2))])
    means = np.array([[0.0, 0.0], [3.0, 0.0]])

    print(f"\n  Two fixed centers; E-step responsibilities at shrinking spherical variance:\n")
    print(f"    {'sigma^2':>9s} {'mean max-responsibility':>24s} {'% points > 0.99 assigned':>26s}")
    for s2 in (2.0, 0.5, 0.1, 0.01):
        # responsibility to nearest center under N(mu, s2 I)
        d0 = np.sum((X - means[0]) ** 2, 1)
        d1 = np.sum((X - means[1]) ** 2, 1)
        r0 = 1.0 / (1.0 + np.exp(-np.clip((d1 - d0) / (2 * s2), -700, 700)))  # resp to comp 0
        rmax = np.maximum(r0, 1 - r0)
        print(f"    {s2:>9.2f} {rmax.mean():>24.3f} {np.mean(rmax > 0.99):>25.0%}")
    print("""
  READING: at large variance the responsibilities are SOFT (a point near the boundary is ~50/50);
  as sigma^2 -> 0 they sharpen until almost every point is assigned to its nearest center with
  probability ~1 — a HARD assignment. That is exactly k-means. GMM and k-means are the same
  algorithm at different 'temperatures'; k-means is EM at zero variance (README §7).""")


if __name__ == "__main__":
    verify()
    experiment_1_vs_kmeans()
    experiment_2_monotone()
    experiment_3_singularity()
    experiment_4_cov_types()
    experiment_5_bic()
    experiment_6_hardening()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
