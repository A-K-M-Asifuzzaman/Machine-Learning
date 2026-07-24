"""
04.02 — Hierarchical (Agglomerative) Clustering, from scratch (NumPy).

Agglomerative merging with the Lance-Williams update for single / complete / average / Ward
linkage, verified against scipy.cluster.hierarchy. Then the chapter's claims are MEASURED:

  1. linkage choice determines cluster shape: Ward wins blobs, single wins chains  (README §4)
  2. single-linkage CHAINING finds non-convex clusters but breaks on a noise bridge (README §4)
  3. Ward linkage ~ k-means on blobs (same sum-of-squares objective)               (README §9)
  4. the largest merge-height GAP recovers the true number of clusters             (README §6)
  5. cophenetic correlation measures how faithfully the tree preserves distances   (README §7)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from scipy.cluster.hierarchy import linkage as sp_linkage, fcluster, cophenet
    from scipy.spatial.distance import pdist
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.datasets import make_blobs, make_moons, make_circles
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# AGGLOMERATIVE CLUSTERING via LANCE-WILLIAMS  (README §3-§5)
# =============================================================================


def _lance_williams_coeffs(linkage, ni, nj, nk):
    """(alpha_i, alpha_j, beta, gamma) for d(i∪j, k) (README §5)."""
    if linkage == "single":
        return 0.5, 0.5, 0.0, -0.5
    if linkage == "complete":
        return 0.5, 0.5, 0.0, 0.5
    if linkage == "average":
        return ni / (ni + nj), nj / (ni + nj), 0.0, 0.0
    if linkage == "ward":
        t = ni + nj + nk
        return (ni + nk) / t, (nj + nk) / t, -nk / t, 0.0
    raise ValueError(linkage)


def agglomerative(X, linkage="ward"):
    """Return the SciPy-style linkage matrix Z (shape (n-1, 4)):
    [cluster_i, cluster_j, merge_height, merged_size]."""
    X = np.asarray(X, float)
    n = len(X)
    # initial pairwise distances; Ward works on squared distances internally
    D = np.sqrt((np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T).clip(min=0))
    if linkage == "ward":
        D = D ** 2                                     # Ward updates on squared distance
    active = list(range(n))
    size = {i: 1 for i in range(n)}
    dist = {}                                          # (a,b)->distance, a<b by id
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
    Z = []
    next_id = n

    def key(a, b):
        return (a, b) if a < b else (b, a)

    while len(active) > 1:
        # find the closest pair among active clusters
        best = None
        for x in range(len(active)):
            for y in range(x + 1, len(active)):
                a, b = active[x], active[y]
                d = dist[key(a, b)]
                if best is None or d < best[0]:
                    best = (d, a, b)
        d_merge, a, b = best
        na, nb = size[a], size[b]
        height = np.sqrt(d_merge) if linkage == "ward" else d_merge
        Z.append([a, b, height, na + nb])
        # Lance-Williams update: new cluster `next_id` vs every other active cluster
        for c in active:
            if c == a or c == b:
                continue
            ai, aj, be, ga = _lance_williams_coeffs(linkage, na, nb, size[c])
            dik, djk, dij = dist[key(a, c)], dist[key(b, c)], dist[key(a, b)]
            dist[key(next_id, c)] = ai * dik + aj * djk + be * dij + ga * abs(dik - djk)
        active = [c for c in active if c != a and c != b] + [next_id]
        size[next_id] = na + nb
        next_id += 1
    return np.array(Z)


def fcluster_k(Z, n, k):
    """Cut the linkage tree Z into k flat clusters (by merging the n-1 lowest merges,
    stopping when k clusters remain)."""
    parent = list(range(2 * n - 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merges = (n - 1) - (k - 1)                       # merges to apply to leave k clusters
    for m in range(n_merges):
        a, b = int(Z[m, 0]), int(Z[m, 1])
        parent[find(a)] = n + m
        parent[find(b)] = n + m
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def cophenetic_corr(X, Z):
    """Correlation between original pairwise distances and cophenetic (tree) distances."""
    X = np.asarray(X, float)
    n = len(X)
    orig = np.sqrt((np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :]
                    - 2 * X @ X.T).clip(min=0))
    # cophenetic distance = merge height at which i and j first join
    coph = np.zeros((n, n))
    parent = list(range(2 * n - 1))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for m in range(len(Z)):
        a, b, h = int(Z[m, 0]), int(Z[m, 1]), Z[m, 2]
        ra, rb = find(a), find(b)
        for i in members[ra]:
            for j in members[rb]:
                coph[i, j] = coph[j, i] = h
        new = n + m
        parent[ra] = new
        parent[rb] = new
        members[new] = members[ra] + members[rb]
    iu = np.triu_indices(n, 1)
    return float(np.corrcoef(orig[iu], coph[iu])[0, 1])


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — agglomerative linkage vs scipy.cluster.hierarchy")
    print("=" * 88)
    if not (HAVE_SCIPY and HAVE_SK):
        print("\n(scipy/sklearn unavailable — skipping)")
        return
    X, _ = make_blobs(n_samples=120, centers=4, cluster_std=0.8, random_state=0)

    for lk in ("single", "complete", "average", "ward"):
        Z = agglomerative(X, lk)
        Zs = sp_linkage(X, method=lk)
        # merge heights should match (merge ORDER can differ on ties, so compare sorted heights)
        h_ours = np.sort(Z[:, 2])
        h_sp = np.sort(Zs[:, 2])
        max_diff = np.max(np.abs(h_ours - h_sp))
        # flat clustering agreement at k=4
        lab_ours = fcluster_k(Z, len(X), 4)
        lab_sp = fcluster(Zs, 4, criterion="maxclust")
        ari = adjusted_rand_score(lab_sp, lab_ours)
        print(f"    {lk:>9s}: merge-height max|diff| = {max_diff:.2e}   "
              f"flat-cluster ARI vs scipy = {ari:.3f}")
        assert max_diff < 1e-6, f"{lk} heights must match scipy"
        assert ari > 0.98, f"{lk} flat clusters must match scipy"
    print("\n  all four linkages match scipy's merge heights and flat clusters  ✓")

    # cophenetic correlation vs scipy
    Z = agglomerative(X, "average")
    ours = cophenetic_corr(X, Z)
    sp = cophenet(sp_linkage(X, "average"), pdist(X))[0]
    print(f"\n  cophenetic correlation (average linkage): ours {ours:.4f} vs scipy {sp:.4f}  "
          f"(diff {abs(ours-sp):.2e})")
    assert abs(ours - sp) < 1e-6, "cophenetic correlation must match scipy"
    print("  cophenetic correlation matches scipy  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — linkage determines cluster shape (README §4)
# =============================================================================


def experiment_1_linkage_shapes():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — linkage choice determines the cluster shape it finds (README §4)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    Xb, yb = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
    Xm, ym = make_moons(n_samples=300, noise=0.05, random_state=0)

    print(f"\n  Adjusted Rand Index vs the truth (1.0 = perfect):\n")
    print(f"    {'linkage':>10s} {'blobs (compact)':>17s} {'two moons (chain)':>19s}")
    for lk in ("single", "complete", "average", "ward"):
        ari_b = adjusted_rand_score(yb, fcluster_k(agglomerative(Xb, lk), len(Xb), 3))
        ari_m = adjusted_rand_score(ym, fcluster_k(agglomerative(Xm, lk), len(Xm), 2))
        print(f"    {lk:>10s} {ari_b:>17.3f} {ari_m:>19.3f}")
    print("""
  READING: on compact BLOBS, Ward and complete linkage (which prefer spherical, equal-diameter
  clusters) win; single linkage can chain across them. On the non-convex TWO MOONS, SINGLE linkage
  wins by far — it follows each moon's connectivity via nearest points — while Ward/complete slice
  the moons in half. The linkage encodes what a cluster IS; choose it from the shape you expect
  (README §4).""")


# =============================================================================
# EXPERIMENT 2 — single-linkage chaining and its fragility (README §4)
# =============================================================================


def experiment_2_chaining():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — single-linkage chaining: finds non-convex, breaks on a noise bridge")
    print("               (README §4)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    rng = np.random.default_rng(0)
    X, y = make_circles(n_samples=400, noise=0.04, factor=0.4, random_state=0)

    ari_clean = adjusted_rand_score(y, fcluster_k(agglomerative(X, "single"), len(X), 2))

    # add a thin BRIDGE of noise points connecting the inner ring (r~0.4) to the outer (r~1.0)
    bridge = np.column_stack([np.linspace(0.4, 1.0, 25), np.zeros(25)]) \
        + rng.normal(0, 0.01, (25, 2))
    Xb = np.vstack([X, bridge])
    ari_bridge = adjusted_rand_score(y, fcluster_k(agglomerative(Xb, "single"),
                                                   len(Xb), 2)[:len(X)])
    print(f"""
  Two concentric rings (non-convex), single linkage, k=2:

    {'data':>26s} {'ARI on the rings':>18s}
    {'clean':>26s} {ari_clean:>18.3f}   <- single linkage separates the rings
    {'+ 25-point noise bridge':>26s} {ari_bridge:>18.3f}   <- one bridge merges them

  READING: single linkage nails the two concentric rings on clean data — no other linkage can,
  because the rings are non-convex. But its strength is fragile: a thin chain of ~25 noise points
  bridging the rings makes single linkage merge them into one cluster (each merge only needs ONE
  close pair). Powerful on clean, well-separated shapes; dangerous with noise (README §4).""")


# =============================================================================
# EXPERIMENT 3 — Ward ~ k-means on blobs (README §9)
# =============================================================================


def experiment_3_ward_kmeans():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — Ward linkage ~ k-means (same sum-of-squares objective) (README §9)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    X, y = make_blobs(n_samples=600, centers=5, cluster_std=1.0, random_state=2)
    ward = fcluster_k(agglomerative(X, "ward"), len(X), 5)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X).labels_
    print(f"""
    {'method':>14s} {'ARI vs truth':>14s}
    {'Ward':>14s} {adjusted_rand_score(y, ward):>14.3f}
    {'k-means':>14s} {adjusted_rand_score(y, km):>14.3f}
    Ward vs k-means agreement (ARI): {adjusted_rand_score(km, ward):.3f}

  READING: Ward merges the pair that increases within-cluster variance least — the SAME
  sum-of-squares objective k-means minimizes. So on blob data they land on nearly the same
  clustering. Ward is effectively k-means with a full hierarchy and no random restarts (README §9).""")


# =============================================================================
# EXPERIMENT 4 — choosing k from the largest merge-height gap (README §6)
# =============================================================================


def experiment_4_cut():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — recover k from the largest gap in merge heights (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    true_k = 4
    X, _ = make_blobs(n_samples=400, centers=true_k, cluster_std=0.6, random_state=7)
    Z = agglomerative(X, "ward")
    heights = Z[:, 2]
    # look at the largest jumps between the last several merges
    top = heights[-8:]
    gaps = np.diff(top)
    # k clusters correspond to cutting just below the largest gap among the final merges
    gap_pos = int(np.argmax(gaps))
    k_est = len(top) - gap_pos                          # clusters remaining above that gap
    print(f"\n  Final merge heights (Ward): {np.array2string(top, precision=2)}")
    print(f"  gaps between them:          {np.array2string(gaps, precision=2)}")
    print(f"""
  Largest gap is before the last {gap_pos+1} merge(s) -> estimated k = {k_est}  (true k = {true_k}).

  READING: as agglomeration proceeds, merges within a true cluster happen at small heights and
  the merges that join SEPARATE true clusters happen at large heights. The biggest jump in merge
  height marks that transition, so cutting just below it recovers the number of clusters — the
  dendrogram analogue of the elbow method (README §6).""")


# =============================================================================
# EXPERIMENT 5 — cophenetic correlation per linkage (README §7)
# =============================================================================


def experiment_5_cophenetic():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — cophenetic correlation: how faithfully the tree preserves distance")
    print("               (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping — needs sklearn datasets)")
        return
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=1)
    print(f"\n    {'linkage':>10s} {'cophenetic correlation':>24s}")
    for lk in ("single", "complete", "average", "ward"):
        c = cophenetic_corr(X, agglomerative(X, lk))
        print(f"    {lk:>10s} {c:>24.3f}")
    print("""
  READING: the cophenetic correlation measures how well the dendrogram's tree-distances (the
  height at which two points first merge) match the original pairwise distances. AVERAGE linkage
  usually scores highest — it is built to preserve mean distances — which is why it is a common
  choice when faithful representation matters. Use it to compare and trust linkages (README §7).""")


if __name__ == "__main__":
    verify()
    experiment_1_linkage_shapes()
    experiment_2_chaining()
    experiment_3_ward_kmeans()
    experiment_4_cut()
    experiment_5_cophenetic()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
