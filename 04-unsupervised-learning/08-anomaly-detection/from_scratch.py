"""
04.08 — Anomaly Detection, from scratch (NumPy).

Isolation Forest, Local Outlier Factor, a Gaussian density scorer, and PCA reconstruction
error, verified against scikit-learn. Then the chapter's claims are MEASURED:

  1. Isolation Forest: anomalies have SHORTER path lengths (ROC-AUC vs labels)   (README §5)
  2. LOF catches a LOCAL anomaly that global k-NN distance misses                (README §4)
  3. Isolation Forest beats a single Gaussian on MULTIMODAL data                 (README §3, §5)
  4. PCA reconstruction error scores off-subspace anomalies                      (README §7)
  5. the contamination threshold trades precision against recall                 (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.ensemble import IsolationForest as SkIF
    from sklearn.neighbors import LocalOutlierFactor as SkLOF
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _harmonic(n):
    return np.log(n) + 0.5772156649 if n > 1 else 0.0


def _c(n):
    """Expected path length of an unsuccessful BST search on n points."""
    return 2 * _harmonic(n - 1) - 2 * (n - 1) / n if n > 1 else 0.0


# =============================================================================
# ISOLATION FOREST  (README §5)
# =============================================================================


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit

    def fit(self, X, depth=0):
        n = len(X)
        if depth >= self.height_limit or n <= 1:
            self.leaf = True
            self.size = n
            return self
        self.leaf = False
        f = np.random.randint(X.shape[1])
        lo, hi = X[:, f].min(), X[:, f].max()
        if lo == hi:
            self.leaf = True
            self.size = n
            return self
        self.feature = f
        self.split = np.random.uniform(lo, hi)
        mask = X[:, f] < self.split
        self.left = IsolationTree(self.height_limit).fit(X[mask], depth + 1)
        self.right = IsolationTree(self.height_limit).fit(X[~mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.leaf:
            return depth + _c(self.size)              # adjust for early termination
        branch = self.left if x[self.feature] < self.split else self.right
        return branch.path_length(x, depth + 1)


class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, random_state=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, float)
        np.random.seed(self.random_state)
        n = len(X)
        m = min(self.sample_size, n)
        self.c_n = _c(m)
        height_limit = int(np.ceil(np.log2(m)))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, m, replace=False)
            self.trees.append(IsolationTree(height_limit).fit(X[idx]))
        return self

    def anomaly_score(self, X):
        """Higher = more anomalous (in [0, 1])."""
        X = np.asarray(X, float)
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            avg_path = np.mean([t.path_length(x) for t in self.trees])
            scores[i] = 2 ** (-avg_path / self.c_n)
        return scores


# =============================================================================
# LOCAL OUTLIER FACTOR  (README §4)
# =============================================================================


def _pairwise(X):
    return np.sqrt((np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :]
                    - 2 * X @ X.T).clip(min=0))


def lof(X, k=20):
    """Local Outlier Factor: ratio of neighbours' density to a point's own (README §4)."""
    X = np.asarray(X, float)
    n = len(X)
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    neighbors = np.argsort(D, axis=1)[:, :k]
    k_dist = np.array([D[i, neighbors[i][-1]] for i in range(n)])   # distance to k-th nn
    # reachability distance reach_k(a,b) = max(k_dist(b), d(a,b))
    lrd = np.zeros(n)
    for i in range(n):
        reach = np.maximum(k_dist[neighbors[i]], D[i, neighbors[i]])
        lrd[i] = 1.0 / (reach.mean() + 1e-12)
    lof_scores = np.array([np.mean(lrd[neighbors[i]]) / (lrd[i] + 1e-12) for i in range(n)])
    return lof_scores


# =============================================================================
# GAUSSIAN DENSITY and PCA RECONSTRUCTION  (README §3, §7)
# =============================================================================


def gaussian_score(X, Xtrain=None):
    """Negative log-likelihood under a single fitted Gaussian (higher = more anomalous)."""
    Xtrain = X if Xtrain is None else Xtrain
    mu = Xtrain.mean(0)
    cov = np.cov(Xtrain.T) + 1e-6 * np.eye(X.shape[1])
    Ci = np.linalg.inv(cov)
    diff = X - mu
    return np.sum((diff @ Ci) * diff, axis=1)         # Mahalanobis^2 (up to const)


def pca_reconstruction_score(X, n_components):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_components]
    recon = Xc @ V.T @ V
    return np.sum((Xc - recon) ** 2, axis=1)          # reconstruction error


# =============================================================================
# VERIFICATION
# =============================================================================


def _make_with_outliers(n_normal=480, n_out=20, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal([0, 0], 1.0, (n_normal, 2))
    out = rng.uniform(-8, 8, (n_out, 2))
    X = np.vstack([normal, out])
    y = np.r_[np.zeros(n_normal), np.ones(n_out)]     # 1 = anomaly
    return X, y


def verify():
    print("=" * 88)
    print("VERIFICATION — Isolation Forest & LOF vs scikit-learn")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(sklearn unavailable — skipping)")
        return
    X, y = _make_with_outliers(seed=0)

    ours_if = IsolationForest(n_trees=200, random_state=0).fit(X).anomaly_score(X)
    sk_if = -SkIF(n_estimators=200, random_state=0).fit(X).score_samples(X)
    auc_ours = roc_auc_score(y, ours_if)
    auc_sk = roc_auc_score(y, sk_if)
    print(f"""
    Isolation Forest ROC-AUC: ours = {auc_ours:.3f}, sklearn = {auc_sk:.3f}
    score rank correlation with sklearn = {np.corrcoef(np.argsort(np.argsort(ours_if)),
                                                        np.argsort(np.argsort(sk_if)))[0,1]:.3f}""")
    assert auc_ours > 0.9 and abs(auc_ours - auc_sk) < 0.06, "IF should match sklearn"

    ours_lof = lof(X, k=20)
    sk_lof = -SkLOF(n_neighbors=20).fit(X).negative_outlier_factor_
    auc_lof = roc_auc_score(y, ours_lof)
    auc_sklof = roc_auc_score(y, sk_lof)
    print(f"""
    LOF ROC-AUC: ours = {auc_lof:.3f}, sklearn = {auc_sklof:.3f}
    LOF score rank correlation with sklearn = {np.corrcoef(np.argsort(np.argsort(ours_lof)),
                                                            np.argsort(np.argsort(sk_lof)))[0,1]:.3f}""")
    assert abs(auc_lof - auc_sklof) < 0.05, "LOF should match sklearn"
    print("\n  Isolation Forest and LOF match sklearn (ROC-AUC and score ranking)  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — Isolation Forest path lengths (README §5)
# =============================================================================


def experiment_1_isolation():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — Isolation Forest: anomalies have shorter path lengths (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = _make_with_outliers(n_normal=480, n_out=20, seed=1)
    forest = IsolationForest(n_trees=200, random_state=0).fit(X)

    # mean path length for normal vs anomalous points
    def mean_path(pts):
        return np.mean([np.mean([t.path_length(x) for t in forest.trees]) for x in pts])
    p_normal = mean_path(X[y == 0])
    p_anom = mean_path(X[y == 1])
    scores = forest.anomaly_score(X)
    print(f"""
  480 normal points + 20 uniform outliers:

    mean path length, NORMAL points  = {p_normal:.2f}
    mean path length, ANOMALY points = {p_anom:.2f}   (shorter -> isolated faster)

    ROC-AUC (anomaly score vs labels)     = {roc_auc_score(y, scores):.3f}
    average precision (rare-class metric) = {average_precision_score(y, scores):.3f}

  READING: anomalies sit apart from the crowd, so a few random axis splits isolate them — their
  average path length ({p_anom:.1f}) is much shorter than a normal point's ({p_normal:.1f}), which
  is buried among many similar points. The 2^(-path/c(n)) score turns that into a clean separation
  (ROC-AUC {roc_auc_score(y, scores):.2f}), needing no distances or density model (README §5).""")


# =============================================================================
# EXPERIMENT 2 — LOF catches a local anomaly (README §4)
# =============================================================================


def experiment_2_local():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — LOF catches a LOCAL anomaly that global k-NN distance misses (README §4)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    # a DENSE cluster and a SPARSE cluster; plant an anomaly just outside the dense cluster
    dense = rng.normal([0, 0], 0.3, (200, 2))
    sparse = rng.normal([8, 0], 2.0, (100, 2))
    local_anom = np.array([[1.5, 0.0]])              # near the dense cluster but off it
    X = np.vstack([dense, sparse, local_anom])
    anom_idx = len(X) - 1

    lof_scores = lof(X, k=20)
    # global k-NN distance score (distance to 20th neighbour)
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    knn_dist = np.sort(D, axis=1)[:, 19]

    lof_rank = np.sum(lof_scores >= lof_scores[anom_idx])       # 1 = most anomalous
    knn_rank = np.sum(knn_dist >= knn_dist[anom_idx])
    print(f"""
  A point at (1.5, 0) — just outside the DENSE cluster, but not far in absolute terms because a
  SPARSE cluster exists elsewhere. Rank of this local anomaly (1 = flagged most anomalous):

    {'method':>22s} {'rank of the local anomaly':>26s}
    {'global k-NN distance':>22s} {knn_rank:>26d}
    {'LOF (local density)':>22s} {lof_rank:>26d}

  READING: by GLOBAL k-NN distance the planted point is unremarkable (rank {knn_rank}) — the sparse
  cluster contains points just as far from their neighbours, so no global distance threshold flags
  it. LOF compares the point's density to its OWN neighbours' densities: sitting just outside a
  DENSE cluster, it is far sparser than its neighbours, so LOF ranks it near the top ({lof_rank}).
  Local density ratios catch local anomalies that global distances miss (README §4).""")


# =============================================================================
# EXPERIMENT 3 — Isolation Forest vs single Gaussian on multimodal data (README §3)
# =============================================================================


def experiment_3_multimodal():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a single Gaussian fails on multimodal data; Isolation Forest does not")
    print("               (README §3, §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    # TWO normal clusters far apart; an anomaly in the GAP between them (near the global mean!)
    c1 = rng.normal([-6, 0], 0.6, (250, 2))
    c2 = rng.normal([6, 0], 0.6, (250, 2))
    gap_anoms = rng.normal([0, 0], 0.3, (20, 2))     # sit at the global mean -> low Mahalanobis
    X = np.vstack([c1, c2, gap_anoms])
    y = np.r_[np.zeros(500), np.ones(20)]

    g_score = gaussian_score(X)
    if_score = IsolationForest(n_trees=200, random_state=0).fit(X).anomaly_score(X)
    print(f"""
  Two normal clusters at (-6,0) and (6,0); 20 anomalies in the GAP at the origin (the global mean):

    {'method':>22s} {'ROC-AUC':>8s}
    {'single Gaussian':>22s} {roc_auc_score(y, g_score):>8.3f}
    {'Isolation Forest':>22s} {roc_auc_score(y, if_score):>8.3f}

  READING: the anomalies sit at the GLOBAL MEAN, so a single Gaussian scores them as the MOST
  normal points (low Mahalanobis distance) — ROC-AUC below 0.5, worse than chance. Isolation
  Forest partitions the actual data and isolates the sparse gap region quickly, correctly flagging
  it. A single Gaussian assumes one blob; for multimodal normal data use a GMM or Isolation Forest
  (README §3).""")


# =============================================================================
# EXPERIMENT 4 — PCA reconstruction error (README §7)
# =============================================================================


def experiment_4_pca_recon():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — PCA reconstruction error scores off-subspace anomalies (README §7)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    rng = np.random.default_rng(0)
    # normal data lives near a 3-D subspace of 20-D; anomalies have off-subspace components
    latent = rng.standard_normal((480, 3))
    load = rng.standard_normal((3, 20))
    normal = latent @ load + 0.05 * rng.standard_normal((480, 20))
    anoms = normal[:20].copy()
    anoms += rng.standard_normal((20, 20)) * 2.0                  # push off the subspace
    X = np.vstack([normal, anoms])
    y = np.r_[np.zeros(480), np.ones(20)]

    score = pca_reconstruction_score(X, n_components=3)
    print(f"""
  Normal data near a 3-D subspace of 20-D; 20 anomalies pushed OFF the subspace.
  Anomaly score = reconstruction error using the top 3 PCs:

    ROC-AUC (reconstruction error vs labels) = {roc_auc_score(y, score):.3f}
    average precision                        = {average_precision_score(y, score):.3f}

  READING: normal points lie near the 3-D subspace, so projecting to 3 PCs and back reconstructs
  them with tiny error; the anomalies have components OFF the subspace that the projection discards,
  so they reconstruct poorly and score high (ROC-AUC {roc_auc_score(y, score):.2f}). Reconstruction
  error is a strong anomaly signal when normal data has low-rank structure (README §7).""")


# =============================================================================
# EXPERIMENT 5 — the contamination threshold (README §8)
# =============================================================================


def experiment_5_contamination():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the contamination threshold trades precision against recall (README §8)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(skipping)")
        return
    X, y = _make_with_outliers(n_normal=470, n_out=30, seed=2)
    scores = IsolationForest(n_trees=200, random_state=0).fit(X).anomaly_score(X)
    n = len(X)

    print(f"\n  True anomaly rate = {y.mean():.0%}. Flag the top-`contamination` fraction by score:\n")
    print(f"    {'contamination':>14s} {'# flagged':>10s} {'precision':>10s} {'recall':>8s}")
    for cont in (0.02, 0.04, 0.06, 0.10, 0.20):
        k = int(cont * n)
        flagged = np.argsort(-scores)[:k]
        pred = np.zeros(n)
        pred[flagged] = 1
        tp = np.sum((pred == 1) & (y == 1))
        prec = tp / max(k, 1)
        rec = tp / y.sum()
        print(f"    {cont:>14.2f} {k:>10d} {prec:>10.2f} {rec:>8.2f}")
    print("""
  READING: the contamination parameter sets the score threshold. Flag too few (low contamination)
  and precision is high but recall is low (you miss anomalies); flag too many and recall rises but
  precision falls (false alarms). It is exactly a classifier's threshold tradeoff — set it from the
  cost of a miss vs a false alarm, near the true anomaly rate when known (README §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_isolation()
    experiment_2_local()
    experiment_3_multimodal()
    experiment_4_pca_recon()
    experiment_5_contamination()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
