"""
03.06 — k-Nearest Neighbours from Scratch
=========================================

KNN is trivial to implement and interesting to measure. The implementation is short; the
experiments are where the chapter lives.

Implemented here
----------------
    KNNClassifier / KNNRegressor    brute force and KD-tree backends
        weights="uniform"|"distance"                        README §6
        metric: euclidean, manhattan, chebyshev, minkowski,
                cosine, hamming, mahalanobis                README §3

    KDTree                          with correct pruning                README §10.1
    pairwise_distances              vectorized, all metrics

Run it
------
    python from_scratch.py

Verified against sklearn and scipy, then five experiments:
  1. The curse of dimensionality, measured three ways
  2. k as the bias-variance dial — and why training error cannot select it
  3. Feature scaling: one unscaled feature silently becomes the only feature
  4. KD-trees degenerate to brute force, and the crossover is where theory says
  5. Cover & Hart: 1-NN error is within a factor of 2 of the Bayes error

Reference: README.md sections 2-10.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# DISTANCES  (README §3)
# =============================================================================


def pairwise_distances(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean",
                       p: float = 2.0, VI: np.ndarray | None = None) -> np.ndarray:
    """Distance matrix D[i, j] = d(X[i], Y[j]).

    The Euclidean case uses the expansion ||x - y||^2 = ||x||^2 - 2 x.y + ||y||^2, which
    turns the whole computation into one matrix product — orders of magnitude faster than
    a loop. It is also numerically delicate: the subtraction of two large nearly-equal
    quantities is catastrophic cancellation (00.06 §4), so tiny negative values appear and
    must be clipped before the square root, or you get nan.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if metric == "euclidean":
        sq = (np.sum(X ** 2, axis=1)[:, None] - 2 * X @ Y.T
              + np.sum(Y ** 2, axis=1)[None, :])
        return np.sqrt(np.maximum(sq, 0.0))       # clip: cancellation can give -1e-16

    if metric == "manhattan":
        return np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)

    if metric == "chebyshev":
        return np.abs(X[:, None, :] - Y[None, :, :]).max(axis=2)

    if metric == "minkowski":
        return (np.abs(X[:, None, :] - Y[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)

    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300)
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-300)
        return 1.0 - Xn @ Yn.T

    if metric == "hamming":
        return (X[:, None, :] != Y[None, :, :]).mean(axis=2)

    if metric == "mahalanobis":
        # d^2 = (x-y)^T Sigma^-1 (x-y). Whitening first turns this into a plain Euclidean
        # distance, which is both faster and better conditioned than forming the quadratic
        # form directly — and makes the relationship explicit: Mahalanobis IS Euclidean
        # distance after whitening (README §3).
        if VI is None:
            raise ValueError("mahalanobis needs VI (inverse covariance)")
        L = np.linalg.cholesky(VI)
        return pairwise_distances(X @ L, Y @ L, "euclidean")

    raise ValueError(f"unknown metric {metric!r}")


# =============================================================================
# KD-TREE  (README §10.1)
# =============================================================================


class KDTree:
    """Axis-aligned space-partitioning tree for exact nearest-neighbour search.

    Build: recursively split on the median of the widest-spread axis. Query: descend to the
    query's own leaf first, then unwind, visiting a sibling branch ONLY IF the distance to
    its splitting plane is smaller than the current k-th best. That test is the pruning,
    and it is the entire reason the tree is fast.

    It is also exactly why the tree degenerates in high dimensions (README §8.2, §10.1):
    pruning requires proving a region is farther than the current best, and when all
    distances concentrate no region can be excluded. The tree then visits every node and
    pays traversal overhead on top of the brute-force work. Experiment 4 measures where
    that crossover happens.
    """

    def __init__(self, X: np.ndarray, leaf_size: int = 16):
        self.X = np.asarray(X, dtype=float)
        self.leaf_size = leaf_size
        self.n_nodes_visited = 0
        self.root = self._build(np.arange(self.X.shape[0]))

    def _build(self, indices):
        if indices.size <= self.leaf_size:
            return {"leaf": True, "indices": indices}

        points = self.X[indices]
        # Split on the axis with the largest spread — better balanced than cycling axes.
        axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
        order = np.argsort(points[:, axis])
        mid = order.size // 2
        sorted_idx = indices[order]

        return {
            "leaf": False,
            "axis": axis,
            "value": float(points[order[mid], axis]),
            "left": self._build(sorted_idx[:mid]),
            "right": self._build(sorted_idx[mid:]),
        }

    def query(self, x: np.ndarray, k: int = 1):
        """Return (distances, indices) of the k nearest neighbours of x."""
        x = np.asarray(x, dtype=float)
        # Max-heap of the k best so far, kept as a sorted list for clarity.
        best: list[tuple[float, int]] = []

        def search(node):
            self.n_nodes_visited += 1
            if node["leaf"]:
                for i in node["indices"]:
                    d = float(np.sqrt(np.sum((x - self.X[i]) ** 2)))
                    if len(best) < k:
                        best.append((d, int(i)))
                        best.sort()
                    elif d < best[-1][0]:
                        best[-1] = (d, int(i))
                        best.sort()
                return

            axis, value = node["axis"], node["value"]
            near, far = ((node["left"], node["right"]) if x[axis] < value
                         else (node["right"], node["left"]))
            search(near)
            # THE PRUNING TEST. Visit the far side only if the splitting plane is closer
            # than the current worst of our k best — otherwise nothing over there can win.
            if len(best) < k or abs(x[axis] - value) < best[-1][0]:
                search(far)

        search(self.root)
        distances = np.array([d for d, _ in best])
        indices = np.array([i for _, i in best])
        return distances, indices


# =============================================================================
# MODELS  (README §2, §5-§7)
# =============================================================================


class _KNNBase:
    def __init__(self, k: int = 5, weights: str = "uniform", metric: str = "euclidean",
                 p: float = 2.0, algorithm: str = "brute", leaf_size: int = 16):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        """'Training' is storage. All the work happens at prediction time (README §1)."""
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).ravel()
        self._VI = None
        if self.metric == "mahalanobis":
            cov = np.cov(self.X_.T) + 1e-8 * np.eye(self.X_.shape[1])
            self._VI = np.linalg.inv(cov)
        if self.algorithm == "kd_tree":
            self._tree = KDTree(self.X_, self.leaf_size)
        return self

    def _neighbours(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        k = min(self.k, self.X_.shape[0])

        if self.algorithm == "kd_tree":
            if self.metric != "euclidean":
                raise ValueError("kd_tree backend supports euclidean only")
            distances, indices = [], []
            for row in X:
                d, i = self._tree.query(row, k)
                distances.append(d)
                indices.append(i)
            return np.array(distances), np.array(indices)

        D = pairwise_distances(X, self.X_, self.metric, self.p, self._VI)
        # argpartition is O(n) per row against argsort's O(n log n) — only the k smallest
        # matter, and their internal order is then fixed with a small sort.
        idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(X.shape[0])[:, None]
        order = np.argsort(D[rows, idx], axis=1)
        idx = idx[rows, order]
        return D[rows, idx], idx

    def _weights(self, distances: np.ndarray) -> np.ndarray:
        if self.weights == "uniform":
            return np.ones_like(distances)
        if self.weights == "distance":
            # An exact hit gets all the weight, matching sklearn's behaviour. Without this
            # special case 1/d is inf and the arithmetic produces nan.
            with np.errstate(divide="ignore"):
                w = 1.0 / distances
            exact = np.isinf(w)
            if np.any(exact):
                w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), w)
            return w
        raise ValueError(f"unknown weights {self.weights!r}")


class KNNClassifier(_KNNBase):
    """Majority vote among the k nearest neighbours.

    Note the decision boundary for k=1 is exactly the Voronoi diagram of the training set,
    and that training accuracy is IDENTICALLY 1.0 for k=1 — every point is its own nearest
    neighbour. That is why k must never be tuned on training error (README §5).
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self._neighbours(X)
        weights = self._weights(distances)
        labels = self.y_[indices]

        out = np.zeros((len(indices), self.classes_.size))
        for c_idx, c in enumerate(self.classes_):
            out[:, c_idx] = np.sum(weights * (labels == c), axis=1)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


class KNNRegressor(_KNNBase):
    """Average of the k nearest neighbours' targets.

    Structurally CANNOT extrapolate: every prediction is a weighted average of observed y
    values, so it is bounded by [min y, max y] no matter how far outside the data the query
    lies (README §7).
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self._neighbours(X)
        weights = self._weights(distances)
        return np.sum(weights * self.y_[indices], axis=1) / np.sum(weights, axis=1)

    def score(self, X, y) -> float:
        y = np.asarray(y).ravel()
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total)


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<58s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("VERIFICATION")
    print("=" * 88)

    n, d = 300, 6
    X = rng.standard_normal((n, d))
    y_clf = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    y_reg = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.standard_normal(n) * 0.1
    X_test = rng.standard_normal((80, d))

    print("\nDistance metrics vs scipy (README §3)")
    try:
        from scipy.spatial.distance import cdist
        for metric, kwargs in [("euclidean", {}), ("manhattan", {}), ("chebyshev", {}),
                               ("cosine", {}), ("minkowski", {"p": 3.0})]:
            scipy_name = {"manhattan": "cityblock"}.get(metric, metric)
            mine = pairwise_distances(X[:40], X_test[:30], metric, **kwargs)
            ref = cdist(X[:40], X_test[:30], scipy_name,
                        **({"p": 3.0} if metric == "minkowski" else {}))
            ok &= _report(f"{metric} distance vs scipy",
                          float(np.abs(mine - ref).max()), 1e-10)

        cov = np.cov(X.T) + 1e-8 * np.eye(d)
        VI = np.linalg.inv(cov)
        ok &= _report("mahalanobis vs scipy",
                      float(np.abs(pairwise_distances(X[:40], X_test[:30], "mahalanobis",
                                                      VI=VI)
                                   - cdist(X[:40], X_test[:30], "mahalanobis", VI=VI)).max()),
                      1e-8)
    except ImportError:
        print("  [SKIP]  scipy not installed")

    print("\nClassifier and regressor vs sklearn (README §2, §6-§7)")
    try:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        for k in (1, 3, 5, 15):
            for weights in ("uniform", "distance"):
                mine = KNNClassifier(k=k, weights=weights).fit(X, y_clf)
                ref = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X, y_clf)
                ok &= _report(f"classifier k={k}, weights={weights!r}: probabilities",
                              float(np.abs(mine.predict_proba(X_test)
                                           - ref.predict_proba(X_test)).max()), 1e-10)

        for k in (1, 5, 15):
            for weights in ("uniform", "distance"):
                mine = KNNRegressor(k=k, weights=weights).fit(X, y_reg)
                ref = KNeighborsRegressor(n_neighbors=k, weights=weights).fit(X, y_reg)
                ok &= _report(f"regressor k={k}, weights={weights!r}",
                              float(np.abs(mine.predict(X_test)
                                           - ref.predict(X_test)).max()), 1e-10)

        for metric in ("manhattan", "chebyshev"):
            mine = KNNClassifier(k=5, metric=metric).fit(X, y_clf)
            ref = KNeighborsClassifier(n_neighbors=5, metric=metric).fit(X, y_clf)
            ok &= _report(f"classifier with metric={metric!r}",
                          float(np.mean(mine.predict(X_test) != ref.predict(X_test))), 1e-12)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nKD-tree (README §10.1)")
    tree = KDTree(X, leaf_size=8)
    brute = pairwise_distances(X_test, X, "euclidean")
    max_err = 0.0
    for i, row in enumerate(X_test):
        d_tree, i_tree = tree.query(row, k=5)
        order = np.argsort(brute[i])[:5]
        max_err = max(max_err, float(np.abs(d_tree - brute[i][order]).max()))
    ok &= _report("KD-tree returns EXACTLY the brute-force neighbours", max_err, 1e-12)

    tree_model = KNNClassifier(k=5, algorithm="kd_tree").fit(X, y_clf)
    brute_model = KNNClassifier(k=5, algorithm="brute").fit(X, y_clf)
    ok &= _report("kd_tree and brute backends agree",
                  float(np.mean(tree_model.predict(X_test) != brute_model.predict(X_test))),
                  1e-12)

    print("\nStructural properties (README §5, §7)")
    one_nn = KNNClassifier(k=1).fit(X, y_clf)
    ok &= _report("k=1 has training accuracy exactly 1.0",
                  abs(one_nn.score(X, y_clf) - 1.0), 1e-15)

    # k = n predicts the global majority for every query.
    all_nn = KNNClassifier(k=n).fit(X, y_clf)
    majority = np.bincount(y_clf).argmax()
    ok &= _report("k=n predicts the majority class everywhere",
                  float(np.mean(all_nn.predict(X_test) != majority)), 1e-12)

    # Regression cannot extrapolate: predictions stay inside the observed range.
    far = np.full((10, d), 50.0)
    preds = KNNRegressor(k=5).fit(X, y_reg).predict(far)
    ok &= _report("regressor cannot leave [min y, max y]",
                  float(max(0.0, preds.max() - y_reg.max(), y_reg.min() - preds.min())),
                  1e-12)

    # Cosine distance is invariant to vector magnitude.
    scaled = X_test * rng.uniform(0.1, 10, (X_test.shape[0], 1))
    ok &= _report("cosine distance ignores magnitude",
                  float(np.abs(pairwise_distances(X_test, X, "cosine")
                               - pairwise_distances(scaled, X, "cosine")).max()), 1e-10)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_curse() -> None:
    """README §8: the three geometric facts that break KNN in high dimensions."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the curse of dimensionality, measured  (README §8)")
    print("=" * 88)
    print("""
Three separate consequences of high-dimensional geometry, each fatal to KNN on its own.
""")
    rng = np.random.default_rng(1)

    print("  (a) NEIGHBOURHOODS STOP BEING LOCAL.  To capture a fraction r of a unit")
    print("      hypercube you need a sub-cube of edge r^(1/d):\n")
    print(f"      {'d':>5s}  {'edge for 1%':>13s}  {'edge for 10%':>14s}")
    print("      " + "-" * 36)
    for d in (1, 2, 10, 50, 100):
        print(f"      {d:5d}  {0.01 ** (1 / d):13.3f}  {0.10 ** (1 / d):14.3f}")

    print("""
      At d = 100, reaching your nearest 1% of neighbours needs a box spanning 95.5% of
      every feature's range. Those points are not 'nearby' in any useful sense.
""")

    print("  (b) DISTANCES CONCENTRATE.  The nearest and farthest points converge:\n")
    print(f"      {'d':>5s}  {'mean dist':>11s}  {'sd/mean':>9s}  "
          f"{'(max-min)/min':>14s}  {'1-NN unique?':>13s}")
    print("      " + "-" * 60)

    n = 1000
    for d in (1, 2, 5, 10, 20, 50, 100, 500):
        X = rng.random((n, d))
        queries = rng.random((60, d))
        D = pairwise_distances(queries, X, "euclidean")
        contrast = float(np.mean((D.max(axis=1) - D.min(axis=1)) / D.min(axis=1)))
        # How distinguishable is the nearest neighbour from the 2nd?
        part = np.sort(D, axis=1)
        gap_ratio = float(np.mean((part[:, 1] - part[:, 0]) / part[:, 0]))
        print(f"      {d:5d}  {D.mean():11.4f}  {D.std() / D.mean():9.4f}  "
              f"{contrast:14.4f}  {gap_ratio:12.5f}")

    print("""
      Column 3 (sd/mean) falls as 1/sqrt(d): the squared distance is a sum of d i.i.d.
      terms whose mean grows like d and whose standard deviation grows like sqrt(d).

      Column 4 is the one that matters. By d = 100 the farthest point is only ~40% farther
      than the nearest; by d = 500 it is ~17%. 'Nearest neighbour' has stopped being a
      meaningful designation.

      Column 5 shows the practical consequence: the relative gap between the 1st and 2nd
      neighbour collapses toward zero, so which point is 'nearest' is decided by noise.
""")

    print("  (c) EVERYTHING IS ON THE BOUNDARY.  Volume of the inscribed ball / cube:\n")
    from math import lgamma, log, pi, exp
    print(f"      {'d':>5s}  {'ball/cube volume':>18s}")
    print("      " + "-" * 26)
    for d in (2, 5, 10, 20, 50):
        log_ratio = (d / 2) * log(pi) - d * log(2) - lgamma(d / 2 + 1)
        print(f"      {d:5d}  {exp(log_ratio):18.3e}")

    print("""
      Almost all the volume of a high-dimensional cube is in its corners, so almost every
      point sits near a boundary with few neighbours 'all around' it. Every prediction
      becomes an extrapolation.

  THE ESCAPE (README §8.4): all three effects concern the INTRINSIC dimension, not the
  column count. Real data often lies near a low-dimensional manifold, and a learned
  embedding maps inputs to a space where L2 distance MEANS semantic similarity. That is
  why KNN over 1536-dimensional embeddings powers modern vector search while KNN over 100
  raw features does not. The algorithm is unchanged; the metric is not.""")


def experiment_k() -> None:
    """README §5: k is the bias-variance dial, and it runs backwards."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — k as the bias-variance dial  (README §5)")
    print("=" * 88)
    print("""
For most hyperparameters, larger means more capacity. For k it is the opposite: the
effective number of parameters is roughly n/k, so SMALL k is the complex model.
""")
    rng = np.random.default_rng(2)
    n = 400

    def make(size):
        X = rng.uniform(-3, 3, (size, 2))
        # A genuinely curved boundary, plus 10% label noise.
        clean = (X[:, 1] > np.sin(2 * X[:, 0])).astype(int)
        flip = rng.random(size) < 0.10
        return X, np.where(flip, 1 - clean, clean)

    X_tr, y_tr = make(n)
    X_te, y_te = make(4000)

    print(f"  n = {n}, 2 features, 10% label noise\n")
    print(f"  {'k':>5s}  {'~effective params':>18s}  {'train acc':>10s}  {'TEST acc':>10s}  "
          f"{'gap':>8s}")
    print("  " + "-" * 58)

    ks = (1, 3, 5, 11, 25, 51, 101, 201, n)
    rows = []
    for k in ks:
        model = KNNClassifier(k=k).fit(X_tr, y_tr)
        rows.append((k, model.score(X_tr, y_tr), model.score(X_te, y_te)))

    best_k = max(rows, key=lambda r: r[2])[0]
    for k, train, test in rows:
        marker = "  <- best" if k == best_k else ""
        print(f"  {k:5d}  {n / k:18.1f}  {train:10.4f}  {test:10.4f}  "
              f"{train - test:8.4f}{marker}")

    print(f"""
  TRAINING ACCURACY IS EXACTLY 1.0000 AT k=1, by construction — every point is its own
  nearest neighbour. It then falls monotonically. This is why k can never be selected on
  training error: it would always choose 1, and the choice would carry no information.

  TEST ACCURACY has an interior maximum, at k = {best_k}. Below it the model is fitting the
  10% label noise; above it the boundary is oversmoothed and the sine curve is lost. At
  k = n every prediction is the global majority.

  The gap column is the overfitting signal — large at small k, closing as k grows.

  Note the direction: increasing k SIMPLIFIES the model. If you carry the usual intuition
  that bigger hyperparameter = more capacity, KNN will surprise you.""")


def experiment_scaling() -> None:
    """README §4: without scaling, the widest-range feature is the only feature."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — feature scaling is not optional  (README §4)")
    print("=" * 88)
    print("""
Two features. Feature A is INFORMATIVE about the label. Feature B is pure noise. The only
difference between runs is the UNITS feature B happens to be measured in.
""")
    rng = np.random.default_rng(3)
    n = 600

    def make(size, noise_scale):
        y = rng.integers(0, 2, size)
        informative = rng.standard_normal(size) + y * 2.5      # carries the signal
        noise = rng.standard_normal(size) * noise_scale        # carries nothing
        return np.column_stack([informative, noise]), y

    print(f"  {'noise feature scale':>20s}  {'unscaled accuracy':>19s}  "
          f"{'standardized accuracy':>22s}")
    print("  " + "-" * 66)

    for scale in (0.1, 1.0, 10.0, 100.0, 1000.0):
        X_tr, y_tr = make(n, scale)
        X_te, y_te = make(3000, scale)

        raw = KNNClassifier(k=15).fit(X_tr, y_tr).score(X_te, y_te)

        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        scaled = KNNClassifier(k=15).fit((X_tr - mu) / sd, y_tr).score((X_te - mu) / sd, y_te)

        print(f"  {scale:20.1f}  {raw:19.4f}  {scaled:22.4f}")

    print("""
  Standardized accuracy is flat: the units of an irrelevant feature should not matter, and
  after scaling they do not.

  Unscaled accuracy collapses toward chance as the noise feature's scale grows. Euclidean
  distance sums SQUARED differences, so a feature with 1000x the range contributes 10^6
  times more to the distance. KNN is then effectively doing nearest-neighbour search on
  pure noise, and the informative feature is invisible.

  Nothing about the data changed — only the units someone recorded one column in. This is
  the same class of failure as unscaled ridge regression (03.02 §10), arriving through a
  different route: there the PENALTY was not scale-invariant, here the DISTANCE is not.

  Always put the scaler in a Pipeline so it is fitted on the training fold only (02.06).""")


def experiment_kdtree() -> None:
    """README §10.1: KD-trees degenerate to brute force as d grows."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — where KD-trees stop helping  (README §10.1)")
    print("=" * 88)
    print("""
A KD-tree prunes by proving a region is farther than the current best. Experiment 1 showed
that in high dimensions all distances converge — so nothing can be proved farther, nothing
is pruned, and the tree visits every node while paying traversal overhead on top.

Measuring the fraction of the tree actually visited:
""")
    import time
    rng = np.random.default_rng(4)
    n = 3000

    print(f"  n = {n}, k = 5, exact search\n")
    print(f"  {'d':>5s}  {'nodes visited':>14s}  {'% of tree':>10s}  {'KD-tree (s)':>12s}  "
          f"{'brute (s)':>11s}  {'speedup':>9s}")
    print("  " + "-" * 68)

    for d in (2, 4, 8, 16, 32):
        X = rng.random((n, d))
        queries = rng.random((150, d))

        tree = KDTree(X, leaf_size=16)
        # Count nodes in the tree, for a denominator.
        def count(node):
            return 1 if node["leaf"] else 1 + count(node["left"]) + count(node["right"])
        total_nodes = count(tree.root)

        tree.n_nodes_visited = 0
        start = time.perf_counter()
        for q in queries:
            tree.query(q, k=5)
        tree_time = time.perf_counter() - start
        visited_per_query = tree.n_nodes_visited / len(queries)

        start = time.perf_counter()
        D = pairwise_distances(queries, X, "euclidean")
        np.argpartition(D, kth=4, axis=1)[:, :5]
        brute_time = time.perf_counter() - start

        print(f"  {d:5d}  {visited_per_query:14.1f}  "
              f"{visited_per_query / total_nodes:9.1%}  {tree_time:12.4f}  "
              f"{brute_time:11.4f}  {brute_time / tree_time:8.2f}x")

    print("""
  The '% of tree' column is the mechanism. At d = 2 a query touches a small fraction of the
  nodes — pruning is working. By d = 16-32 it is visiting most of the tree, so the
  asymptotic advantage is gone and only the overhead remains.

  Note the timing comparison is not entirely fair to the tree: the brute-force path here is
  a single vectorized BLAS matrix product, while the tree is a Python recursion. That
  handicap is the point in practice, though — sklearn's C implementation shows the same
  qualitative crossover around d ~ 20, and `algorithm='auto'` silently switches to brute
  force above it.

  The lesson is structural rather than about constants: exact space-partitioning search
  cannot beat the curse of dimensionality, because pruning IS the ability to distinguish
  distances, and that is exactly what high dimensions destroy. Above d ~ 20 the answer is
  approximate search — HNSW, IVF-PQ, LSH — which trades exactness for a speedup that does
  survive.""")


def experiment_cover_hart() -> None:
    """README §9: 1-NN error is asymptotically within a factor of 2 of Bayes."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — Cover & Hart: 1-NN vs the Bayes error  (README §9)")
    print("=" * 88)
    print("""
The theorem: as n -> infinity,  R* <= R_1NN <= 2 R*(1 - R*) <= 2 R*.

We can test it because we can construct a problem whose Bayes error is known exactly: two
Gaussians with known means and shared covariance, where the optimal rule and its error rate
are both computable in closed form.
""")
    from math import erf, sqrt
    rng = np.random.default_rng(5)
    d = 2

    print(f"  {'separation':>11s}  {'Bayes error':>12s}  {'1-NN (n=20k)':>14s}  "
          f"{'ratio':>7s}  {'upper bound 2R*(1-R*)':>22s}  {'holds?':>7s}")
    print("  " + "-" * 80)

    for sep in (0.5, 1.0, 2.0, 3.0, 4.0):
        mu = np.zeros(d)
        mu[0] = sep
        # Bayes error for two equiprobable N(0,I) and N(mu,I): Phi(-||mu||/2).
        bayes = 0.5 * (1 - erf((sep / 2) / sqrt(2)))

        def make(size):
            y = rng.integers(0, 2, size)
            return rng.standard_normal((size, d)) + np.outer(y, mu), y

        X_tr, y_tr = make(20000)
        X_te, y_te = make(20000)
        error_1nn = 1 - KNNClassifier(k=1).fit(X_tr, y_tr).score(X_te, y_te)

        bound = 2 * bayes * (1 - bayes)
        holds = error_1nn <= bound + 0.01              # small allowance for finite n
        print(f"  {sep:11.1f}  {bayes:12.4f}  {error_1nn:14.4f}  "
              f"{error_1nn / bayes:7.2f}  {bound:22.4f}  {str(holds):>7s}")

    print("""
  The bound holds in every row, and the ratio sits between 1 and 2 exactly as the theorem
  requires. It also moves the right way: the ratio RISES as the Bayes error falls, because
  the bound itself is 2(1 - R*), which tightens toward 1 as R* grows and loosens toward 2
  as R* shrinks. At finite n the observed ratio stays comfortably inside the bound rather
  than saturating it.

  This is a genuinely remarkable guarantee. With NO model, NO training, NO assumptions
  beyond a sensible metric, 1-NN is asymptotically within a factor of two of the best any
  classifier can do. It says most of the information needed to classify a point is carried
  by its single nearest example.

  Two caveats that matter as much as the theorem:

  1. It is ASYMPTOTIC. The convergence rate is exponentially slow in d — which is
     Experiment 1 restated. At d = 100 the required n exceeds anything achievable.

  2. It assumes the metric is meaningful. Experiments 1 and 3 are both about that
     assumption failing, in different ways.

  The theorem is what makes k-NN worth understanding; the caveats are what make everything
  after this chapter necessary.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_curse()
    experiment_k()
    experiment_scaling()
    experiment_kdtree()
    experiment_cover_hart()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
