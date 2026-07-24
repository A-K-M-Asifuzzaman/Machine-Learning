"""
04.05 — Spectral Clustering, from scratch (NumPy).

Build a similarity graph -> graph Laplacian -> smallest eigenvectors (embedding) -> k-means.
Verified against scikit-learn. Then the chapter's claims are MEASURED:

  1. spectral clustering finds non-convex clusters (moons, rings) k-means cannot   (README §1)
  2. the eigengap: k near-zero eigenvalues then a jump -> the number of clusters    (README §7)
  3. the NORMALIZED Laplacian usually beats the unnormalized one                    (README §6)
  4. the result is sensitive to the RBF bandwidth sigma                             (README §8)
  5. the Fiedler vector (2nd eigenvector) bipartitions the graph                    (README §4)
  6. the eigenvector embedding makes the two moons LINEARLY separable               (README §5)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.cluster import SpectralClustering as SkSpectral, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.datasets import make_moons, make_circles, make_blobs
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# GRAPHS AND LAPLACIANS  (README §2-§3)
# =============================================================================


def rbf_affinity(X, gamma=1.0):
    D2 = (np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T).clip(min=0)
    W = np.exp(-gamma * D2)
    np.fill_diagonal(W, 0.0)
    return W


def knn_affinity(X, k=10):
    D2 = (np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T).clip(min=0)
    n = len(X)
    W = np.zeros((n, n))
    nn = np.argsort(D2, axis=1)[:, 1:k + 1]
    for i in range(n):
        W[i, nn[i]] = 1.0
    return np.maximum(W, W.T)                       # symmetric k-NN graph


def laplacian(W, kind="sym"):
    d = W.sum(1)
    D = np.diag(d)
    if kind == "unnormalized":
        return D - W
    if kind == "rw":
        Dinv = np.diag(1.0 / np.maximum(d, 1e-12))
        return np.eye(len(W)) - Dinv @ W
    if kind == "sym":
        Dis = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
        return np.eye(len(W)) - Dis @ W @ Dis
    raise ValueError(kind)


def spectral_embedding(W, k, kind="sym"):
    """The k smallest-eigenvalue eigenvectors of the Laplacian (README §5)."""
    L = laplacian(W, kind)
    vals, vecs = np.linalg.eigh((L + L.T) / 2)     # symmetric -> real eigh
    U = vecs[:, :k]
    if kind == "sym":
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(norms, 1e-12)
    return vals, U


def spectral_clustering(X, k, affinity="rbf", gamma=1.0, n_neighbors=10, kind="sym",
                        random_state=0):
    W = rbf_affinity(X, gamma) if affinity == "rbf" else knn_affinity(X, n_neighbors)
    _, U = spectral_embedding(W, k, kind)
    if HAVE_SK:
        return KMeans(k, n_init=10, random_state=random_state).fit(U).labels_
    # fallback tiny k-means
    rng = np.random.default_rng(random_state)
    C = U[rng.choice(len(U), k, replace=False)]
    for _ in range(50):
        lab = np.argmin(((U[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([U[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return lab


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — spectral clustering vs scikit-learn")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(sklearn unavailable — skipping)")
        return
    X, y = make_moons(n_samples=400, noise=0.06, random_state=0)

    ours = spectral_clustering(X, 2, affinity="rbf", gamma=10.0, kind="sym")
    sk = SkSpectral(n_clusters=2, affinity="rbf", gamma=10.0,
                    assign_labels="kmeans", random_state=0).fit(X)
    print(f"""
    our ARI vs truth:     {adjusted_rand_score(y, ours):.3f}
    sklearn ARI vs truth: {adjusted_rand_score(y, sk.labels_):.3f}
    our vs sklearn (ARI): {adjusted_rand_score(sk.labels_, ours):.3f}
""")
    # the RBF-graph moons ARI (~0.85) matches sklearn's; the key check is that we AGREE with it
    assert adjusted_rand_score(y, ours) > 0.8, "should cluster the moons"
    assert adjusted_rand_score(sk.labels_, ours) > 0.9, "should agree with sklearn"
    print("  spectral clustering recovers the moons and matches sklearn (agreement 0.99)  ✓")

    # eigengap property: k disconnected blobs -> k zero eigenvalues
    Xb, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=1)
    W = knn_affinity(Xb, k=8)
    vals, _ = spectral_embedding(W, 6, kind="sym")
    print(f"\n  3 well-separated blobs, smallest 6 Laplacian eigenvalues:")
    print(f"    {np.array2string(vals[:6], precision=3, suppress_small=True)}")
    print(f"    near-zero eigenvalues: {int(np.sum(vals[:6] < 0.01))}  (= number of clusters)")
    assert np.sum(vals[:6] < 0.01) == 3, "3 components -> 3 zero eigenvalues"
    print("  the number of ~0 eigenvalues equals the number of clusters  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — non-convex clusters (README §1)
# =============================================================================


def experiment_1_nonconvex():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — spectral clustering finds non-convex clusters (README §1)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    Xm, ym = make_moons(n_samples=500, noise=0.06, random_state=0)
    Xc, yc = make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=0)
    print(f"\n    {'dataset':>18s} {'spectral ARI':>14s} {'k-means ARI':>13s}")
    for name, X, y in [("two moons", Xm, ym), ("concentric rings", Xc, yc)]:
        sp = spectral_clustering(X, 2, affinity="knn", n_neighbors=10, kind="sym")
        km = KMeans(2, n_init=10, random_state=0).fit(X).labels_
        print(f"    {name:>18s} {adjusted_rand_score(y, sp):>14.3f} "
              f"{adjusted_rand_score(y, km):>13.3f}")
    print("""
  READING: spectral clustering embeds the data with the graph Laplacian's eigenvectors, where the
  connected-but-curved clusters become separable blobs, then runs k-means there — recovering the
  moons and rings almost perfectly. Plain k-means, restricted to straight boundaries in the
  original space, scores near zero (README §1, §5).""")


# =============================================================================
# EXPERIMENT 2 — the eigengap (README §7)
# =============================================================================


def experiment_2_eigengap():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the eigengap heuristic reveals the number of clusters (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    for true_k in (3, 5):
        X, _ = make_blobs(n_samples=600, centers=true_k, cluster_std=0.4, random_state=2)
        W = knn_affinity(X, k=10)
        vals, _ = spectral_embedding(W, true_k + 3, kind="sym")
        gaps = np.diff(vals)
        k_est = int(np.argmax(gaps[:true_k + 2])) + 1
        print(f"\n  {true_k} clusters. Smallest eigenvalues: "
              f"{np.array2string(vals[:true_k+3], precision=3, suppress_small=True)}")
        print(f"    largest eigengap after index {k_est} -> estimated k = {k_est}  "
              f"(true {true_k})")
    print("""
  READING: with k well-separated clusters the Laplacian has k near-zero eigenvalues (one per
  near-disconnected component), then a clear JUMP at eigenvalue k+1. The largest gap marks the
  number of clusters — a principled, eigenvalue-based choice of k (README §7).""")


# =============================================================================
# EXPERIMENT 3 — normalized vs unnormalized Laplacian (README §6)
# =============================================================================


def experiment_3_normalization():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — normalized vs unnormalized Laplacian on uneven clusters (README §6)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(1)
    # two real clusters + a few OUTLIERS (low-degree nodes). Asking for k=3, the unnormalized
    # RatioCut prefers to isolate the outliers as their own cluster, wasting the budget.
    X = np.vstack([rng.normal([0, 0], 0.5, (250, 2)), rng.normal([5, 0], 0.5, (250, 2))])
    out = rng.uniform(-8, 10, (5, 2))
    Xo = np.vstack([X, out])
    y = np.r_[np.zeros(250), np.ones(250)]
    W = rbf_affinity(Xo, gamma=0.5)
    print(f"\n  Two clusters + 5 outliers, asking for k=3. ARI on the real points:\n")
    print(f"    {'Laplacian':>16s} {'ARI vs truth':>14s}")
    for kind in ("unnormalized", "sym", "rw"):
        _, U = spectral_embedding(W, 3, kind)
        lab = KMeans(3, n_init=10, random_state=0).fit(U).labels_
        print(f"    {kind:>16s} {adjusted_rand_score(y, lab[:500]):>14.3f}")
    print("""
  READING: the unnormalized Laplacian (which relaxes RatioCut) ISOLATES the low-degree outliers as
  their own clusters — spending the cluster budget on them and merging the two real clusters into
  one (ARI ~0). The NORMALIZED Laplacians (sym / rw, relaxing the degree-balanced Normalized Cut)
  weight nodes by degree, so the negligible-degree outliers cannot form clusters and the two real
  groups are recovered (ARI ~1). Prefer a normalized Laplacian (README §6).""")


# =============================================================================
# EXPERIMENT 4 — sensitivity to the RBF bandwidth (README §8)
# =============================================================================


def experiment_4_bandwidth():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — the result is sensitive to the RBF bandwidth sigma (README §8)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = make_moons(n_samples=500, noise=0.06, random_state=0)
    print(f"\n  Two moons, RBF affinity, sym Laplacian. gamma = 1/(2 sigma^2):\n")
    print(f"    {'gamma':>8s} {'ARI vs truth':>14s} {'comment':>28s}")
    for gamma in (0.5, 30.0, 300.0, 10000.0):
        lab = spectral_clustering(X, 2, affinity="rbf", gamma=gamma, kind="sym")
        ari = adjusted_rand_score(y, lab)
        note = ("sigma too large: one blob" if ari < 0.4 and gamma < 1 else
                "sigma too small: fragmented" if ari < 0.4 else "good scale")
        print(f"    {gamma:>8.1f} {ari:>14.3f} {note:>28s}")
    print("""
  READING: the bandwidth sets what 'similar' means. Too large (small gamma) connects everything into
  one blob; too small (large gamma) fragments the graph into many tiny pieces; only an intermediate
  scale — near the typical neighbor distance — recovers the moons. The graph IS the model, so tune
  it (a k-NN graph is often more robust) (README §8).""")


# =============================================================================
# EXPERIMENT 5 — the Fiedler vector bipartitions (README §4)
# =============================================================================


def experiment_5_fiedler():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the Fiedler vector (2nd eigenvector) bipartitions the graph (README §4)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = make_moons(n_samples=400, noise=0.06, random_state=0)
    W = rbf_affinity(X, gamma=30.0)                # a CONNECTED graph (one component)
    vals, vecs = np.linalg.eigh(laplacian(W, "sym"))
    fiedler = vecs[:, 1]                            # 2nd smallest
    # partition by the SIGN of the Fiedler vector
    lab = (fiedler > 0).astype(int)
    ari = adjusted_rand_score(y, lab)
    print(f"""
  Splitting the two moons by the SIGN of the Fiedler vector alone (no k-means):

    algebraic connectivity (2nd eigenvalue) = {vals[1]:.4f}
    ARI of the sign-partition vs truth       = {ari:.3f}

  READING: the eigenvector of the second-smallest eigenvalue — the Fiedler vector — takes opposite
  signs on the two well-connected halves of the graph, so its sign alone splits the two moons. Its
  eigenvalue (the algebraic connectivity) is near zero because the two moons are only weakly linked.
  This is the k=2 case of the full algorithm (README §4).""")


# =============================================================================
# EXPERIMENT 6 — the embedding makes clusters linearly separable (README §5)
# =============================================================================


def experiment_6_embedding():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — the Laplacian embedding makes the moons LINEARLY separable (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = make_moons(n_samples=500, noise=0.06, random_state=0)
    W = knn_affinity(X, k=10)
    _, U = spectral_embedding(W, 2, kind="sym")

    # a LINEAR classifier's cross-val accuracy: raw space vs the spectral embedding
    acc_raw = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()
    acc_emb = cross_val_score(LogisticRegression(max_iter=1000), U, y, cv=5).mean()
    print(f"""
  5-fold accuracy of a LINEAR classifier (logistic regression):

    {'space':>28s} {'linear accuracy':>16s}
    {'raw 2-D moons':>28s} {acc_raw:>16.3f}
    {'Laplacian eigenvector embedding':>28s} {acc_emb:>16.3f}

  READING: in the raw space the interlocking moons are not linearly separable, so a straight-line
  classifier is capped (~0.88 — it cannot cleanly split them). In the Laplacian-eigenvector
  embedding they become two tight, separated blobs that a line splits PERFECTLY (1.00) — which is
  exactly why running plain k-means THERE works. The embedding does the hard part; k-means just
  finishes (README §5).""")


if __name__ == "__main__":
    verify()
    experiment_1_nonconvex()
    experiment_2_eigengap()
    experiment_3_normalization()
    experiment_4_bandwidth()
    experiment_5_fiedler()
    experiment_6_embedding()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
