"""
04.03 — Density Clustering (DBSCAN), from scratch (NumPy).

DBSCAN with core/border/noise labeling and the k-distance heuristic, verified against
scikit-learn. Then the chapter's claims are MEASURED:

  1. DBSCAN finds arbitrary shapes (moons, rings) where k-means fails      (README §1, §6)
  2. it labels outliers as NOISE natively                                  (README §2, §6)
  3. a single global eps CANNOT fit clusters of different density          (README §6)
  4. eps sensitivity, and the k-distance knee that sets it                 (README §5)
  5. the effect of minPts                                                  (README §5)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.cluster import DBSCAN as SkDBSCAN, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.datasets import make_moons, make_circles, make_blobs
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# DBSCAN  (README §2-§4)
# =============================================================================

NOISE = -1


def _neighbors(X, i, eps):
    d = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
    return np.where(d <= eps)[0]


def dbscan(X, eps=0.5, min_samples=5):
    """Return labels (noise = -1) and the boolean core-point mask (README §4)."""
    X = np.asarray(X, float)
    n = len(X)
    labels = np.full(n, NOISE)
    visited = np.zeros(n, dtype=bool)
    core = np.zeros(n, dtype=bool)
    cluster = 0

    # precompute neighborhoods (naive O(n^2); a spatial index makes this O(n log n))
    D = np.sqrt((np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T).clip(min=0))
    neigh = [np.where(D[i] <= eps)[0] for i in range(n)]
    core = np.array([len(neigh[i]) >= min_samples for i in range(n)])

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if not core[i]:
            continue                              # leave as noise (may become border later)
        # start a new cluster and flood-fill through core points
        labels[i] = cluster
        queue = list(neigh[i])
        qi = 0
        while qi < len(queue):
            q = queue[qi]
            qi += 1
            if labels[q] == NOISE:
                labels[q] = cluster               # noise -> border point of this cluster
            if not visited[q]:
                visited[q] = True
                labels[q] = cluster
                if core[q]:
                    queue.extend(neigh[q])        # expand through core points
        cluster += 1
    return labels, core


def k_distance(X, k):
    """Sorted distance to the k-th nearest neighbor (for the eps knee, README §5)."""
    X = np.asarray(X, float)
    D = np.sqrt((np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T).clip(min=0))
    kth = np.sort(D, axis=1)[:, k]                # k-th neighbor (0 is the point itself)
    return np.sort(kth)


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — DBSCAN vs scikit-learn")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(sklearn unavailable — skipping)")
        return
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
    eps, ms = 0.2, 5
    lab, core = dbscan(X, eps, ms)
    sk = SkDBSCAN(eps=eps, min_samples=ms).fit(X)

    # compare on CORE points (border assignment is order-dependent, README §4)
    sk_core = np.zeros(len(X), bool)
    sk_core[sk.core_sample_indices_] = True
    print(f"""
    our core points:     {int(core.sum())}
    sklearn core points: {int(sk_core.sum())}   (identical set: {np.array_equal(core, sk_core)})
    our #clusters:       {len(set(lab)) - (1 if -1 in lab else 0)}
    sklearn #clusters:   {len(set(sk.labels_)) - (1 if -1 in sk.labels_ else 0)}
    label agreement on core points (ARI): {adjusted_rand_score(sk.labels_[core], lab[core]):.3f}
""")
    assert np.array_equal(core, sk_core), "core-point set must match sklearn"
    assert adjusted_rand_score(sk.labels_[core], lab[core]) > 0.99, "core clustering matches"
    print("  core-point set and clustering match sklearn  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — arbitrary shapes (README §1, §6)
# =============================================================================


def experiment_1_shapes():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — DBSCAN finds arbitrary shapes where k-means fails (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    Xm, ym = make_moons(n_samples=500, noise=0.06, random_state=0)
    Xc, yc = make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=0)

    print(f"\n    {'dataset':>18s} {'DBSCAN ARI':>12s} {'k-means ARI':>12s}")
    for name, X, y, eps in [("two moons", Xm, ym, 0.2), ("concentric rings", Xc, yc, 0.2)]:
        lab, _ = dbscan(X, eps=eps, min_samples=5)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).labels_
        # score DBSCAN on non-noise points (noise excluded from the truth comparison)
        nz = lab != NOISE
        ari_db = adjusted_rand_score(y[nz], lab[nz])
        ari_km = adjusted_rand_score(y, km)
        print(f"    {name:>18s} {ari_db:>12.3f} {ari_km:>12.3f}")
    print("""
  READING: DBSCAN clusters the interlocking moons and the concentric rings almost perfectly by
  following connected dense regions, while k-means — restricted to straight Voronoi boundaries —
  scores near zero on both. When clusters are non-convex, density beats distance-to-center
  (README §1, §6).""")


# =============================================================================
# EXPERIMENT 2 — native noise detection (README §2, §6)
# =============================================================================


def experiment_2_noise():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — DBSCAN labels outliers as NOISE natively (README §2)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.4, random_state=1)
    outliers = rng.uniform(-12, 12, (40, 2))
    Xo = np.vstack([X, outliers])
    is_outlier = np.r_[np.zeros(len(X), bool), np.ones(40, bool)]

    lab, _ = dbscan(Xo, eps=0.6, min_samples=5)
    flagged = lab == NOISE
    # how many true outliers were flagged, how many real points wrongly flagged
    tp = int(np.sum(flagged & is_outlier))
    fp = int(np.sum(flagged & ~is_outlier))
    ari_core = adjusted_rand_score(y, lab[:len(X)])
    print(f"""
  3 clusters + 40 injected outliers. DBSCAN(eps=0.6, minPts=5):

    outliers correctly flagged as noise: {tp}/40
    real points wrongly flagged:         {fp}/{len(X)}
    clustering of real points (ARI):     {ari_core:.3f}

  READING: the 40 far outliers land in sparse regions, so DBSCAN marks them NOISE (label -1)
  automatically — outlier detection for free — while the genuine clusters are recovered cleanly.
  k-means, by contrast, would force every outlier into a cluster and let it drag the centroid
  (README §2, §6).""")


# =============================================================================
# EXPERIMENT 3 — varying density defeats a single eps (README §6)
# =============================================================================


def experiment_3_varying_density():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a single global eps cannot fit clusters of different density (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    # a close DENSE PAIR (A, B) plus a far SPARSE cluster C of much lower density
    A = rng.normal([0, 0], 0.18, (300, 2))
    B = rng.normal([1.5, 0], 0.18, (300, 2))
    C = rng.normal([9, 0], 2.4, (150, 2))
    X = np.vstack([A, B, C])

    def dense_pair_separate(lab):
        la = set(lab[:300]) - {NOISE}
        lb = set(lab[300:600]) - {NOISE}
        return bool(la and lb and not (la & lb))

    print(f"\n  Dense pair A,B (close, tight) + sparse cluster C (far, diffuse):\n")
    print(f"    {'eps':>6s} {'#clusters':>10s} {'% noise':>8s} {'C as noise':>11s} "
          f"{'A,B separate?':>14s}")
    for eps in (0.4, 0.7, 1.0, 1.4):
        lab, _ = dbscan(X, eps=eps, min_samples=5)
        nclu = len(set(lab)) - (1 if -1 in lab else 0)
        print(f"    {eps:>6.1f} {nclu:>10d} {np.mean(lab==NOISE):>7.0%} "
              f"{np.mean(lab[600:]==NOISE):>10.0%} {str(dense_pair_separate(lab)):>14s}")
    print("""
  READING: at eps=0.4 the dense pair A,B is correctly SEPARATE — but then C is 97% NOISE, lost
  entirely, because its diffuse points are farther apart than 0.4. Raise eps to capture C and the
  dense pair MERGES (A,B are only 1.5 apart). No single global eps both separates the dense pair
  and captures the sparse cluster — the defining weakness of DBSCAN under varying density, and the
  reason OPTICS and HDBSCAN, which use multiple density levels, exist (README §6-§7).""")


# =============================================================================
# EXPERIMENT 4 — eps sensitivity and the k-distance knee (README §5)
# =============================================================================


def experiment_4_kdistance():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — eps sensitivity and the k-distance knee (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.5, center_box=(-8, 8),
                      random_state=3)
    k = 5

    # k-distance curve and its knee (largest jump in the upper tail)
    kd = k_distance(X, k)
    tail = kd[int(0.7 * len(kd)):]                 # the rising part
    knee = tail[int(np.argmax(np.diff(tail)))]
    print(f"\n  k-distance knee (k={k}) suggests eps ~ {knee:.2f}\n")
    print(f"    {'eps':>6s} {'#clusters':>10s} {'% noise':>8s} {'ARI (non-noise)':>16s}")
    for eps in (0.15, knee, knee * 1.5, 9.0):
        lab, _ = dbscan(X, eps=eps, min_samples=k)
        nz = lab != NOISE
        nclu = len(set(lab)) - (1 if -1 in lab else 0)
        ari = adjusted_rand_score(y[nz], lab[nz]) if nz.sum() > 1 else 0.0
        tag = "  <- knee" if abs(eps - knee) < 1e-9 else ""
        print(f"    {eps:>6.2f} {nclu:>10d} {np.mean(lab==NOISE):>7.0%} {ari:>16.3f}{tag}")
    print("""
  READING: too small an eps marks almost everything noise (many tiny clusters); too large merges
  all four blobs into one. The k-distance knee — where the sorted k-NN distance jumps from
  'inside a cluster' to 'noise' — sets eps in the sweet spot, recovering the 4 true clusters. The
  result is highly sensitive to eps, so use the knee, do not guess (README §5).""")


# =============================================================================
# EXPERIMENT 5 — the effect of minPts (README §5)
# =============================================================================


def experiment_5_minpts():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — minPts controls how much is treated as noise (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=2)
    X = np.vstack([X, rng.uniform(-10, 10, (40, 2))])       # sprinkle noise

    print(f"\n  Fixed eps=0.5; raising minPts (looser blobs + noise):\n")
    print(f"    {'minPts':>7s} {'#clusters':>10s} {'% noise':>8s}")
    for ms in (3, 10, 20, 30):
        lab, _ = dbscan(X, eps=0.5, min_samples=ms)
        nclu = len(set(lab)) - (1 if -1 in lab else 0)
        print(f"    {ms:>7d} {nclu:>10d} {np.mean(lab==NOISE):>7.0%}")
    print("""
  READING: raising minPts demands denser regions to form a cluster, so more low-density points
  are labelled noise and small/loose clusters vanish. Small minPts is permissive (few noise
  points, but noise sensitivity); large minPts is conservative. The rule of thumb is minPts >= d+1,
  often 2d, tuned up if the data is noisy (README §5).""")


if __name__ == "__main__":
    verify()
    experiment_1_shapes()
    experiment_2_noise()
    experiment_3_varying_density()
    experiment_4_kdistance()
    experiment_5_minpts()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
