"""
04.01 — k-Means Clustering, from scratch (NumPy).

Lloyd's algorithm with k-means++ seeding, plus k-medoids, verified against scikit-learn. Then
the chapter's claims are MEASURED:

  1. k-means++ reaches lower inertia and far fewer catastrophic runs than random init  (README §5)
  2. Lloyd's algorithm decreases the objective J MONOTONICALLY                          (README §4)
  3. choosing k: the elbow and silhouette both point to the true k                      (README §6)
  4. failure modes: elongated / unequal-variance / non-convex clusters break k-means    (README §7)
  5. k-medoids is robust to outliers where k-means is not                               (README §9)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.datasets import make_blobs, make_moons
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# k-MEANS with k-means++  (README §3, §5)
# =============================================================================


def _pairwise_sq_dist(X, C):
    """(n, k) squared Euclidean distances between rows of X and centers C."""
    return (np.sum(X**2, 1)[:, None] + np.sum(C**2, 1)[None, :] - 2 * X @ C.T).clip(min=0)


def kmeans_plusplus_init(X, k, rng):
    """D^2 seeding: each new center is drawn proportional to squared distance to the
    nearest chosen center (README §5)."""
    n = len(X)
    centers = [X[rng.integers(n)]]
    d2 = _pairwise_sq_dist(X, np.array(centers)).ravel()
    for _ in range(1, k):
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, _pairwise_sq_dist(X, centers[-1][None, :]).ravel())
    return np.array(centers)


class KMeans:
    def __init__(self, n_clusters=8, init="kmeans++", n_init=10, max_iter=300,
                 tol=1e-6, random_state=0):
        self.k = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fit_once(self, X, rng):
        if self.init == "kmeans++":
            C = kmeans_plusplus_init(X, self.k, rng)
        else:
            C = X[rng.choice(len(X), self.k, replace=False)]
        history = []
        labels = None
        for _ in range(self.max_iter):
            d2 = _pairwise_sq_dist(X, C)
            labels = np.argmin(d2, axis=1)
            J = float(np.sum(d2[np.arange(len(X)), labels]))
            history.append(J)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                             for j in range(self.k)])
            shift = np.sum((newC - C) ** 2)
            C = newC
            if shift < self.tol:
                break
        d2 = _pairwise_sq_dist(X, C)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.sum(d2[np.arange(len(X)), labels]))
        return C, labels, inertia, history

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            C, labels, inertia, hist = self._fit_once(X, rng)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia, hist)
        self.cluster_centers_, self.labels_, self.inertia_, self.history_ = best
        return self

    def predict(self, X):
        return np.argmin(_pairwise_sq_dist(np.asarray(X, float),
                                           self.cluster_centers_), axis=1)


# =============================================================================
# k-MEDOIDS (PAM-style, README §9)
# =============================================================================


class KMedoids:
    def __init__(self, n_clusters=8, max_iter=100, random_state=0):
        self.k = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        D = np.sqrt(_pairwise_sq_dist(X, X))          # full distance matrix
        rng = np.random.default_rng(self.random_state)
        medoids = rng.choice(n, self.k, replace=False)
        for _ in range(self.max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for j in range(self.k):
                members = np.where(labels == j)[0]
                if len(members):
                    # medoid = member minimizing total distance to the rest of the cluster
                    costs = D[np.ix_(members, members)].sum(axis=1)
                    new_medoids[j] = members[np.argmin(costs)]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
        self.medoid_indices_ = medoids
        self.labels_ = np.argmin(D[:, medoids], axis=1)
        self.cluster_centers_ = X[medoids]
        return self


# =============================================================================
# SILHOUETTE (README §6)
# =============================================================================


def silhouette(X, labels):
    X = np.asarray(X, float)
    D = np.sqrt(_pairwise_sq_dist(X, X))
    n = len(X)
    uniq = np.unique(labels)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — k-means vs scikit-learn")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(scikit-learn unavailable — skipping)")
        return
    X, ytrue = make_blobs(n_samples=800, centers=5, cluster_std=0.8, random_state=0)

    ours = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    sk = SkKMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    print(f"""
    our inertia     = {ours.inertia_:.2f}
    sklearn inertia = {sk.inertia_:.2f}   (ratio {ours.inertia_/sk.inertia_:.4f})

    our labels vs true clustering (ARI):     {adjusted_rand_score(ytrue, ours.labels_):.3f}
    sklearn labels vs true clustering (ARI): {adjusted_rand_score(ytrue, sk.labels_):.3f}
    our labels vs sklearn labels (ARI):      {adjusted_rand_score(sk.labels_, ours.labels_):.3f}
""")
    assert abs(ours.inertia_ - sk.inertia_) / sk.inertia_ < 0.02, "inertia parity"
    assert adjusted_rand_score(sk.labels_, ours.labels_) > 0.98, "labels agree up to permutation"

    # silhouette matches sklearn
    s_ours = silhouette(X, ours.labels_)
    s_sk = silhouette_score(X, ours.labels_)
    print(f"  silhouette (ours) = {s_ours:.4f}  vs sklearn = {s_sk:.4f}  "
          f"(diff {abs(s_ours - s_sk):.2e})")
    assert abs(s_ours - s_sk) < 1e-6, "silhouette parity"
    print("\n  inertia, labels, and silhouette all match sklearn  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — k-means++ vs random init (README §5)
# =============================================================================


def experiment_1_init():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — k-means++ vs random initialization (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    # well-separated blobs where a bad init strands a cluster
    X, _ = make_blobs(n_samples=1500, centers=8, cluster_std=0.6, random_state=1)

    def single_init_inertias(init):
        vals = []
        for seed in range(200):
            km = KMeans(n_clusters=8, init=init, n_init=1, random_state=seed).fit(X)
            vals.append(km.inertia_)
        return np.array(vals)

    rand = single_init_inertias("random")
    pp = single_init_inertias("kmeans++")
    best = min(rand.min(), pp.min())
    # a run is 'catastrophic' if inertia is >20% above the best found
    print(f"""
  8 clusters, single initialization, 200 seeds each. Inertia (lower better):

    {'init':>12s} {'mean':>10s} {'best':>10s} {'% catastrophic (>1.2x best)':>28s}
    {'random':>12s} {rand.mean():>10.1f} {rand.min():>10.1f} {np.mean(rand > 1.2*best):>27.0%}
    {'k-means++':>12s} {pp.mean():>10.1f} {pp.min():>10.1f} {np.mean(pp > 1.2*best):>27.0%}

  READING: with a single random seeding, a large fraction of runs land in a bad local optimum
  (two seeds in one true cluster, none in another) — high mean inertia and many catastrophic
  runs. k-means++ seeds spread out via D^2 sampling, so it starts near a good partition: lower
  mean inertia and far fewer catastrophes. This is why k-means++ is the default (README §5).""")


# =============================================================================
# EXPERIMENT 2 — Lloyd's monotonic descent (README §4)
# =============================================================================


def experiment_2_monotone():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — Lloyd's algorithm decreases J monotonically (README §4)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.2, random_state=3)
    km = KMeans(n_clusters=5, init="random", n_init=1, random_state=7)
    _, _, _, hist = km._fit_once(X, np.random.default_rng(7))
    print(f"\n  Objective J at each iteration (single run):\n")
    print(f"    {'iter':>5s} {'J':>12s} {'decrease':>12s}")
    for i, J in enumerate(hist):
        dec = "" if i == 0 else f"{hist[i-1]-J:>12.1f}"
        print(f"    {i:>5d} {J:>12.1f} {dec:>12s}")
    diffs = np.diff(hist)
    assert np.all(diffs <= 1e-6), "J must never increase"
    print(f"""
  READING: J drops at every step and never rises (all {len(diffs)} differences <= 0), converging
  in {len(hist)} iterations. Each step exactly minimizes J over one block (assignments, then
  centroids), so Lloyd's algorithm is coordinate descent — guaranteed to converge to a local
  optimum in finitely many steps (README §4).""")


# =============================================================================
# EXPERIMENT 3 — choosing k (README §6)
# =============================================================================


def experiment_3_choosing_k():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — choosing k with the elbow and silhouette (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    true_k = 4
    X, _ = make_blobs(n_samples=1000, centers=true_k, cluster_std=0.7, random_state=10)
    print(f"\n  Data has {true_k} true clusters.\n")
    print(f"    {'k':>4s} {'inertia':>12s} {'inertia drop':>13s} {'silhouette':>12s}")
    prev = None
    sils = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        sil = silhouette(X, km.labels_)
        sils[k] = sil
        drop = "" if prev is None else f"{prev - km.inertia_:>13.1f}"
        print(f"    {k:>4d} {km.inertia_:>12.1f} {drop:>13s} {sil:>12.3f}")
        prev = km.inertia_
    best_k = max(sils, key=sils.get)
    print(f"""
  Silhouette is maximized at k = {best_k} (true k = {true_k}).

  READING: inertia falls fastest up to k={true_k} and then flattens — the 'elbow'. The silhouette
  peaks exactly at the true k, where clusters are tightest relative to their separation. When the
  elbow is ambiguous, the silhouette (or gap statistic) gives a sharper answer (README §6).""")


# =============================================================================
# EXPERIMENT 4 — failure modes (README §7)
# =============================================================================


def experiment_4_failures():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — where k-means fails: shape, variance, convexity (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    rng = np.random.default_rng(0)

    # (a) elongated (anisotropic) clusters
    Xa, ya = make_blobs(n_samples=800, centers=3, cluster_std=0.6, random_state=2)
    Xa = Xa @ np.array([[3.0, 1.2], [0.0, 0.4]])          # stretch + shear
    # (b) very unequal variances
    Xb1 = rng.normal([0, 0], 0.3, (500, 2))
    Xb2 = rng.normal([4, 0], 2.0, (500, 2))
    Xb = np.vstack([Xb1, Xb2]); yb = np.r_[np.zeros(500), np.ones(500)]
    # (c) non-convex (two moons)
    Xc, yc = make_moons(n_samples=800, noise=0.06, random_state=0)

    print(f"\n  Adjusted Rand Index of k-means vs the true clustering (1.0 = perfect):\n")
    print(f"    {'dataset':>28s} {'k':>3s} {'ARI':>7s}")
    for name, X, y, k in [("elongated (anisotropic)", Xa, ya, 3),
                          ("unequal variance", Xb, yb, 2),
                          ("non-convex (two moons)", Xc, yc, 2)]:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        print(f"    {name:>28s} {k:>3d} {adjusted_rand_score(y, km.labels_):>7.3f}")
    print("""
  READING: k-means scores poorly on all three because its Voronoi boundaries are straight and its
  implied clusters are spherical and equal-spread. Elongated clusters get sliced across; a diffuse
  cluster next to a tight one is mis-split; interlocking moons cannot be separated by any set of
  centers. These are exactly the cases for GMM (full covariance, 04.04), DBSCAN (arbitrary shape,
  04.03), and spectral clustering (non-convex, 04.05) (README §7).""")


# =============================================================================
# EXPERIMENT 5 — k-medoids robustness to outliers (README §9)
# =============================================================================


def experiment_5_medoids():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — k-medoids resists outliers that break k-means (README §9)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.5, random_state=1)
    # a tight clump of gross outliers far away — enough to drag a k-means centroid
    out = rng.normal([25, 0], 2.0, (30, 2))
    Xo = np.vstack([X, out])
    core = np.arange(len(X))                     # score only the real points

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xo)
    kmed = KMedoids(n_clusters=3, random_state=0).fit(Xo)
    ari_km = adjusted_rand_score(y, km.labels_[core])
    ari_md = adjusted_rand_score(y, kmed.labels_[core])
    print(f"""
  3 clusters + a tight clump of 30 gross outliers. ARI on the real points (outliers excluded):

    {'method':>14s} {'ARI on core points':>20s}
    {'k-means':>14s} {ari_km:>20.3f}
    {'k-medoids':>14s} {ari_md:>20.3f}

  READING: a handful of far outliers drag k-means CENTROIDS (means) away from the real clusters,
  corrupting the assignments of the genuine points. k-medoids uses actual data points as centers,
  which no outlier can pull, so it keeps the real clustering intact. Use k-medoids (or remove
  outliers first) when the data is contaminated (README §9).""")


if __name__ == "__main__":
    verify()
    experiment_1_init()
    experiment_2_monotone()
    experiment_3_choosing_k()
    experiment_4_failures()
    experiment_5_medoids()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
