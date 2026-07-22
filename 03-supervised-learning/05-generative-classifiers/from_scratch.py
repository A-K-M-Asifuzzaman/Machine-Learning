"""
03.05 — Generative Classifiers from Scratch
===========================================

Naive Bayes, LDA, and QDA — three points on one covariance spectrum (README §9).

Everything is computed in LOG SPACE. A product of d per-feature probabilities underflows
to exactly zero for d in the hundreds (00.06 §5), which is the normal situation in text
classification, so sums of logs are not an optimization but a requirement.

Implemented here
----------------
    GaussianNB              continuous features, diagonal per-class covariance
    MultinomialNB           counts (bag of words), with add-alpha smoothing
    BernoulliNB             binary presence/absence
    LDA                     shared covariance -> LINEAR boundary       README §7
        .transform()        Fisher's supervised projection             README §11
    QDA                     per-class covariance -> quadratic boundary README §8
    RDA                     regularized: interpolates LDA <-> QDA      README §8

Run it
------
    python from_scratch.py

Verified against sklearn, then five experiments:
  1. Where the independence assumption actually breaks NB
  2. NB's probabilities collapse to 0/1 as correlated features are added — while its
     accuracy holds. A good classifier and a terrible probability estimator.
  3. The NB / LDA / QDA covariance spectrum, and where each wins
  4. Ng & Jordan: naive Bayes beats logistic regression at small n and loses at large n
  5. LDA vs PCA: the highest-variance direction can carry no class information

Reference: README.md sections 3-12.
"""

from __future__ import annotations

import numpy as np

LOG_2PI = np.log(2 * np.pi)


def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """Stable log-sum-exp (00.06 §7). Needed to normalize log-posteriors."""
    c = np.max(a, axis=axis, keepdims=True)
    c = np.where(np.isfinite(c), c, 0.0)
    out = c + np.log(np.sum(np.exp(a - c), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


class _BaseGenerative:
    """Shared prediction machinery: everything is argmax over log-joint scores."""

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self.predict_log_joint(X)
        return joint - _logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


# =============================================================================
# NAIVE BAYES  (README §3-§4)
# =============================================================================


class GaussianNB(_BaseGenerative):
    """p(x_j | y=k) ~ N(mu_jk, sigma^2_jk), features independent given the class.

    Equivalently (README §9): QDA with the covariance forced DIAGONAL. That is the whole
    difference — Gaussian NB is not a separate family, it is a point on the same spectrum
    with 2Kd parameters instead of K*d(d+1)/2.

    `var_smoothing` adds a fraction of the largest feature variance to every variance, so
    a feature that is constant within a class does not produce a division by zero. sklearn
    does the same, with the same default.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) + epsilon for c in self.classes_])
        self.class_prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self.classes_.size))
        for k in range(self.classes_.size):
            # log N(x; mu, sigma^2) summed over independent features.
            log_det = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k]))
            quad = -0.5 * np.sum((X - self.theta_[k]) ** 2 / self.var_[k], axis=1)
            out[:, k] = np.log(self.class_prior_[k]) + log_det + quad
        return out


class MultinomialNB(_BaseGenerative):
    """p(x | y=k) proportional to prod_j theta_jk^{x_j}  — counts, e.g. bag of words.

    Add-alpha smoothing (README §4):

        theta_jk = (N_jk + alpha) / (N_k + alpha * d)

    This is not a hack to avoid zeros — it is the posterior mean under a Dirichlet prior
    (00.03 §8.1), with alpha as a pseudo-count. sklearn's alpha=1.0 default is a prior you
    are using whether or not you chose it.

    Without it, a single unseen word gives log p = -inf and annihilates the class no matter
    what the other thousand features say.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        d = X.shape[1]

        counts = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log([np.mean(y == c) for c in self.classes_])
        return self

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.feature_log_prob_.T + self.class_log_prior_


class BernoulliNB(_BaseGenerative):
    """p(x_j | y=k) ~ Bernoulli(theta_jk) — binary presence/absence.

    The difference from MultinomialNB matters and is often missed: Bernoulli NB explicitly
    penalizes ABSENT features via the (1 - theta) term, while Multinomial NB simply ignores
    them. For short documents, where absence is informative, Bernoulli often wins.
    """

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        X = np.asarray(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        counts = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        n_per_class = np.array([np.sum(y == c) for c in self.classes_])[:, None]
        theta = (counts + self.alpha) / (n_per_class + 2 * self.alpha)

        self.feature_log_prob_ = np.log(theta)
        self.feature_log_prob_neg_ = np.log(1 - theta)
        self.class_log_prior_ = np.log([np.mean(y == c) for c in self.classes_])
        return self

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        # x * log(theta) + (1-x) * log(1-theta), summed over features.
        return (X @ (self.feature_log_prob_ - self.feature_log_prob_neg_).T
                + self.feature_log_prob_neg_.sum(axis=1)
                + self.class_log_prior_)


# =============================================================================
# DISCRIMINANT ANALYSIS  (README §7-§8, §11)
# =============================================================================


class LDA(_BaseGenerative):
    """Gaussian classes with a SHARED covariance -> a linear boundary.  README §7

    The derivation in one line: expanding the log-density gives a term
    x^T Sigma^-1 x that is IDENTICAL for every class, so it cancels in the argmax and only
    linear terms survive. The shared covariance is exactly what makes LDA linear.

        delta_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T Sigma^-1 mu_k + log pi_k

    Solved via the pooled covariance's Cholesky factor rather than by inverting it — same
    lesson as 00.01 §15.3.
    """

    def __init__(self, reg: float = 1e-6, n_components: int | None = None,
                 covariance: str = "unbiased"):
        self.reg = reg
        self.n_components = n_components
        self.covariance = covariance

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n, d = X.shape
        K = self.classes_.size

        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])

        # Pooled within-class covariance. TWO CONVENTIONS ARE IN USE, and they differ by
        # a factor of (n-K)/n:
        #
        #   "unbiased" (ESL, most statistics texts): divide by n - K. K means were
        #       estimated, so K degrees of freedom are gone — the multi-class version of
        #       Bessel's correction (00.04 §5).
        #   "mle" (what sklearn does): divide by n, i.e. the maximum likelihood estimate,
        #       biased low by exactly (n-K)/n.
        #
        # The choice does not affect the ARGMAX much, because scaling Sigma scales all the
        # discriminants together — but it does not scale the log-prior term with them, so
        # the posterior PROBABILITIES differ slightly. The verification below checks
        # against sklearn using "mle" so the comparison is exact rather than approximate.
        Sw = np.zeros((d, d))
        for k, c in enumerate(self.classes_):
            centered = X[y == c] - self.means_[k]
            Sw += centered.T @ centered

        denominator = (n - K) if self.covariance == "unbiased" else n
        self.covariance_ = Sw / denominator + self.reg * np.eye(d)

        # Precompute Sigma^-1 mu_k and the constant term.
        self._prec_means = np.linalg.solve(self.covariance_, self.means_.T).T
        self._const = (-0.5 * np.sum(self.means_ * self._prec_means, axis=1)
                       + np.log(self.priors_))

        self._fit_scalings(X, y)
        return self

    def _fit_scalings(self, X, y):
        """Fisher's projection: eigenvectors of Sw^-1 Sb.  README §11

        Maximizes between-class scatter relative to within-class scatter. Sb has rank at
        most K-1, so there are at most K-1 useful directions no matter how large d is —
        which is LDA-as-projection's main limitation compared with PCA.

        Solved as a symmetric eigenproblem in whitened coordinates rather than by
        eigendecomposing the non-symmetric Sw^-1 Sb, which would give complex eigenvalues
        in floating point.
        """
        d = X.shape[1]
        overall_mean = X.mean(axis=0)
        Sb = np.zeros((d, d))
        for k, c in enumerate(self.classes_):
            n_k = np.sum(y == c)
            diff = (self.means_[k] - overall_mean)[:, None]
            Sb += n_k * (diff @ diff.T)

        # Whiten by Sw, then the problem becomes symmetric.
        L = np.linalg.cholesky(self.covariance_)
        M = np.linalg.solve(L, np.linalg.solve(L, Sb).T).T
        eigenvalues, eigenvectors = np.linalg.eigh((M + M.T) / 2)
        order = np.argsort(eigenvalues)[::-1]

        max_components = min(self.classes_.size - 1, d)
        n_comp = max_components if self.n_components is None else min(self.n_components,
                                                                     max_components)
        self.scalings_ = np.linalg.solve(L.T, eigenvectors[:, order[:n_comp]])
        self.explained_variance_ratio_ = (eigenvalues[order[:n_comp]]
                                          / max(eigenvalues[order].sum(), 1e-300))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto the discriminant directions — supervised dimensionality reduction."""
        return (np.asarray(X, dtype=float) - self.means_.mean(axis=0)) @ self.scalings_

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self._prec_means.T + self._const


class QDA(_BaseGenerative):
    """Gaussian classes with a covariance EACH -> a quadratic boundary.  README §8

    Now x^T Sigma_k^-1 x depends on k and does not cancel, so the boundary is quadratic.

    The cost is parameters: K * d(d+1)/2 covariance entries against LDA's d(d+1)/2. QDA
    needs roughly n_k >> d^2/2 samples PER CLASS to estimate them reliably; below that its
    covariances are noisy or singular and it loses to LDA despite being the more correct
    model. Experiment 3 finds the crossover.

    log|Sigma_k| is computed from the Cholesky factor (2 * sum log diag L) rather than from
    a determinant, which would overflow or underflow even in modest dimensions.
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        d = X.shape[1]

        self.means_, self.priors_ = [], []
        self._chol, self._log_det = [], []

        for c in self.classes_:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            centered = Xc - mean
            cov = centered.T @ centered / max(Xc.shape[0] - 1, 1) + self.reg * np.eye(d)

            L = np.linalg.cholesky(cov)
            self.means_.append(mean)
            self.priors_.append(Xc.shape[0] / X.shape[0])
            self._chol.append(L)
            self._log_det.append(2.0 * np.sum(np.log(np.diag(L))))

        self.means_ = np.array(self.means_)
        self.priors_ = np.array(self.priors_)
        return self

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self.classes_.size))
        for k in range(self.classes_.size):
            diff = X - self.means_[k]
            # Mahalanobis via triangular solve — never forms Sigma^-1.
            solved = np.linalg.solve(self._chol[k], diff.T)
            mahalanobis = np.sum(solved ** 2, axis=0)
            out[:, k] = (-0.5 * mahalanobis - 0.5 * self._log_det[k]
                         - 0.5 * X.shape[1] * LOG_2PI + np.log(self.priors_[k]))
        return out


class RDA(_BaseGenerative):
    """Regularized discriminant analysis: Sigma_k(gamma) = gamma Sigma_k + (1-gamma) Sigma_pooled

    gamma = 1 is QDA, gamma = 0 is LDA, and anything between interpolates. Choosing gamma
    by cross-validation lets the data decide where on the bias-variance spectrum of
    README §9 to sit, rather than committing in advance.
    """

    def __init__(self, gamma: float = 0.5, reg: float = 1e-6):
        self.gamma = gamma
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n, d = X.shape
        K = self.classes_.size

        pooled = np.zeros((d, d))
        per_class = []
        self.means_, self.priors_ = [], []

        for c in self.classes_:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            centered = Xc - mean
            scatter = centered.T @ centered
            pooled += scatter
            per_class.append(scatter / max(Xc.shape[0] - 1, 1))
            self.means_.append(mean)
            self.priors_.append(Xc.shape[0] / n)

        pooled /= (n - K)
        self.means_ = np.array(self.means_)
        self.priors_ = np.array(self.priors_)

        self._chol, self._log_det = [], []
        for cov_k in per_class:
            cov = self.gamma * cov_k + (1 - self.gamma) * pooled + self.reg * np.eye(d)
            L = np.linalg.cholesky(cov)
            self._chol.append(L)
            self._log_det.append(2.0 * np.sum(np.log(np.diag(L))))
        return self

    predict_log_joint = QDA.predict_log_joint


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<56s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("VERIFICATION")
    print("=" * 88)

    n, d, K = 600, 5, 3
    means = rng.standard_normal((K, d)) * 2
    y = rng.integers(0, K, n)
    X = np.array([means[c] + rng.standard_normal(d) for c in y])

    counts = rng.poisson(3, (n, d)).astype(float)
    binary = (rng.random((n, d)) < 0.4).astype(float)

    try:
        from sklearn.naive_bayes import (GaussianNB as SKGaussianNB,
                                         MultinomialNB as SKMultinomialNB,
                                         BernoulliNB as SKBernoulliNB)
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis as SKLDA,
            QuadraticDiscriminantAnalysis as SKQDA)

        print("\nNaive Bayes vs sklearn (README §3-§4)")
        for name, mine, ref, data in [
            ("GaussianNB", GaussianNB(), SKGaussianNB(), X),
            ("MultinomialNB(alpha=1)", MultinomialNB(1.0),
             SKMultinomialNB(alpha=1.0), counts),
            ("MultinomialNB(alpha=0.1)", MultinomialNB(0.1),
             SKMultinomialNB(alpha=0.1), counts),
            ("BernoulliNB(alpha=1)", BernoulliNB(1.0), SKBernoulliNB(alpha=1.0), binary),
        ]:
            mine.fit(data, y)
            ref.fit(data, y)
            ok &= _report(f"{name}: log-probabilities",
                          float(np.abs(mine.predict_log_proba(data)
                                       - ref.predict_log_proba(data)).max()), 1e-9)
            ok &= _report(f"{name}: predictions",
                          float(np.mean(mine.predict(data) != ref.predict(data))), 1e-12)

        print("\nDiscriminant analysis vs sklearn (README §7-§8)")
        # sklearn uses the MLE covariance (divide by n); match it so the check is exact.
        lda, sk_lda = LDA(reg=0.0, covariance="mle"), SKLDA(solver="lsqr")
        lda.fit(X, y)
        sk_lda.fit(X, y)
        ok &= _report("LDA: predictions",
                      float(np.mean(lda.predict(X) != sk_lda.predict(X))), 1e-12)
        ok &= _report("LDA: probabilities (covariance='mle', sklearn's convention)",
                      float(np.abs(lda.predict_proba(X) - sk_lda.predict_proba(X)).max()),
                      1e-8)

        # The unbiased convention gives the same argmax but slightly different posteriors.
        lda_unbiased = LDA(reg=0.0, covariance="unbiased").fit(X, y)
        ok &= _report("...unbiased convention: same predictions",
                      float(np.mean(lda_unbiased.predict(X) != sk_lda.predict(X))), 1e-12)
        print(f"  [INFO]  {'...unbiased vs mle: max probability difference':<56s}  "
              f"{np.abs(lda_unbiased.predict_proba(X) - lda.predict_proba(X)).max():.3e}")

        qda, sk_qda = QDA(reg=0.0), SKQDA(store_covariance=True)
        qda.fit(X, y)
        sk_qda.fit(X, y)
        ok &= _report("QDA: predictions",
                      float(np.mean(qda.predict(X) != sk_qda.predict(X))), 1e-12)
        ok &= _report("QDA: probabilities",
                      float(np.abs(qda.predict_proba(X) - sk_qda.predict_proba(X)).max()),
                      1e-8)

        # Fisher projection: same subspace as sklearn's, up to sign and scale.
        sk_lda_eig = SKLDA(solver="eigen", n_components=K - 1).fit(X, y)
        mine_proj = lda.transform(X)
        ref_proj = sk_lda_eig.transform(X)
        correlations = [abs(np.corrcoef(mine_proj[:, i], ref_proj[:, i])[0, 1])
                        for i in range(K - 1)]
        ok &= _report("LDA projection matches sklearn's (|corr| = 1)",
                      float(1 - min(correlations)), 1e-6)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nStructural properties (README §7-§9)")

    # LDA's boundary must be LINEAR: the log-odds are affine in x.
    lda = LDA().fit(X, y)
    joint = lda.predict_log_joint(X)
    log_odds = joint[:, 1] - joint[:, 0]
    A = np.column_stack([np.ones(n), X])
    residual = log_odds - A @ np.linalg.lstsq(A, log_odds, rcond=None)[0]
    ok &= _report("LDA log-odds are exactly affine in x (linear boundary)",
                  float(np.abs(residual).max()), 1e-9)

    # QDA's must NOT be — otherwise the quadratic terms cancelled and it is just LDA.
    qda = QDA().fit(X, y)
    joint_q = qda.predict_log_joint(X)
    log_odds_q = joint_q[:, 1] - joint_q[:, 0]
    residual_q = log_odds_q - A @ np.linalg.lstsq(A, log_odds_q, rcond=None)[0]
    print(f"  [{'PASS' if np.abs(residual_q).max() > 1e-3 else 'FAIL'}]  "
          f"{'QDA log-odds are NOT affine (quadratic boundary)':<56s}  "
          f"residual = {np.abs(residual_q).max():.3e}")
    ok &= np.abs(residual_q).max() > 1e-3

    # RDA must interpolate: gamma=0 is LDA, gamma=1 is QDA.
    ok &= _report("RDA(gamma=1) matches QDA",
                  float(np.abs(RDA(gamma=1.0, reg=0.0).fit(X, y).predict_proba(X)
                               - QDA(reg=0.0).fit(X, y).predict_proba(X)).max()), 1e-9)
    rda0 = RDA(gamma=0.0, reg=0.0).fit(X, y)
    lda0 = LDA(reg=0.0).fit(X, y)
    ok &= _report("RDA(gamma=0) matches LDA",
                  float(np.abs(rda0.predict_proba(X) - lda0.predict_proba(X)).max()), 1e-9)

    # Probabilities must be normalized, computed via logsumexp.
    ok &= _report("posteriors sum to 1",
                  float(np.abs(GaussianNB().fit(X, y).predict_proba(X).sum(axis=1) - 1).max()),
                  1e-12)

    # Smoothing must eliminate zero probabilities (README §4).
    sparse_counts = np.zeros((60, 8))
    sparse_counts[:30, :4] = rng.poisson(2, (30, 4))
    sparse_counts[30:, 4:] = rng.poisson(2, (30, 4))
    y_sparse = np.array([0] * 30 + [1] * 30)
    unsmoothed = MultinomialNB(alpha=1e-12).fit(sparse_counts, y_sparse)
    smoothed = MultinomialNB(alpha=1.0).fit(sparse_counts, y_sparse)
    print(f"  [INFO]  {'min log p(feature|class): alpha=1e-12 vs alpha=1':<56s}  "
          f"{unsmoothed.feature_log_prob_.min():.1f} vs "
          f"{smoothed.feature_log_prob_.min():.1f}")
    ok &= np.all(np.isfinite(smoothed.feature_log_prob_))

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_independence() -> None:
    """README §5: where does the independence assumption actually break NB?"""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — when does the naive assumption actually hurt?  (README §5)")
    print("=" * 88)
    print("""
The assumption is essentially always false, yet NB classifies well. The claim of README §5
is that classification needs the ARGMAX to be right, not the posterior. Testing it by
sweeping the within-class correlation between features:
""")
    rng = np.random.default_rng(1)
    n, d = 1000, 6

    print(f"  {'within-class corr':>18s}  {'GaussianNB':>11s}  {'LDA':>8s}  {'QDA':>8s}  "
          f"{'logistic':>10s}  {'NB gap to best':>15s}")
    print("  " + "-" * 76)

    for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
        cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
        L = np.linalg.cholesky(cov)
        mean_shift = np.ones(d) * 0.8

        def make(size):
            y = rng.integers(0, 2, size)
            X = rng.standard_normal((size, d)) @ L.T
            X[y == 1] += mean_shift
            return X, y

        X_tr, y_tr = make(n)
        X_te, y_te = make(4000)

        scores = {"nb": GaussianNB().fit(X_tr, y_tr).score(X_te, y_te),
                  "lda": LDA().fit(X_tr, y_tr).score(X_te, y_te),
                  "qda": QDA().fit(X_tr, y_tr).score(X_te, y_te)}
        try:
            from sklearn.linear_model import LogisticRegression
            scores["lr"] = LogisticRegression(max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te)
        except ImportError:
            scores["lr"] = np.nan

        best = max(v for v in scores.values() if not np.isnan(v))
        print(f"  {rho:18.2f}  {scores['nb']:11.4f}  {scores['lda']:8.4f}  "
              f"{scores['qda']:8.4f}  {scores['lr']:10.4f}  "
              f"{best - scores['nb']:15.4f}")

    print("""
  At rho = 0 the assumption holds and NB is as good as anything. As correlation rises its
  gap to the best model widens — but look at the size of the gap. Even at rho = 0.99, where
  the six features are nearly one feature repeated, NB is only a few points behind.

  The mechanism is the one README §5 describes. NB double-counts the correlated evidence
  and produces a wildly over-confident posterior, but the over-confidence is in the SAME
  DIRECTION as the truth, so the argmax survives. Its decisions stay reasonable while its
  probabilities do not — which is Experiment 2.

  Note also that LDA, which models the shared covariance, tracks the best model throughout.
  The cost of NB's assumption is exactly the correlation structure LDA is estimating.""")


def experiment_calibration() -> None:
    """README §6: NB's probabilities collapse to 0/1 while accuracy holds."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — a good classifier, a terrible probability estimator  (README §6)")
    print("=" * 88)
    print("""
Adding REDUNDANT copies of an informative feature. Each copy carries no new information,
so a correct model's confidence should not change. NB multiplies the same evidence once
per copy, so its log-odds grow linearly in the number of copies.
""")
    rng = np.random.default_rng(2)
    n = 2000

    def calibration_error(p, y, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        total = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum():
                total += mask.sum() / len(p) * abs(p[mask].mean() - y[mask].mean())
        return total

    print(f"  {'copies of x1':>13s}  {'NB accuracy':>12s}  {'NB calib err':>13s}  "
          f"{'mean max prob':>14s}  {'% of p beyond 0.99':>19s}")
    print("  " + "-" * 78)

    for n_copies in (1, 2, 4, 8, 16):
        y = rng.integers(0, 2, n)
        base = rng.standard_normal(n) + y * 1.2
        # n_copies near-identical views of the SAME underlying signal.
        X = np.column_stack([base + 0.01 * rng.standard_normal(n) for _ in range(n_copies)])

        y_te = rng.integers(0, 2, 4000)
        base_te = rng.standard_normal(4000) + y_te * 1.2
        X_te = np.column_stack([base_te + 0.01 * rng.standard_normal(4000)
                                for _ in range(n_copies)])

        nb = GaussianNB().fit(X, y)
        p = nb.predict_proba(X_te)[:, 1]
        extreme = float(np.mean((p > 0.99) | (p < 0.01)))

        print(f"  {n_copies:13d}  {nb.score(X_te, y_te):12.4f}  "
              f"{calibration_error(p, y_te):13.4f}  "
              f"{nb.predict_proba(X_te).max(axis=1).mean():14.4f}  {extreme:18.1%}")

    print("""
  ACCURACY BARELY MOVES. The copies add no information, and NB's argmax correctly reflects
  that.

  THE PROBABILITIES COLLAPSE. By 16 copies almost every prediction is beyond 0.99 or below
  0.01, and the calibration error has grown several-fold. NB has multiplied one piece of
  evidence sixteen times and concluded it is certain.

  This is the precise sense in which naive Bayes is a good classifier and a bad probability
  estimator. Never feed its predict_proba into an expected-value decision, a cost-based
  threshold, or a downstream model without calibrating it first (05.06).

  It also explains a common surprise: adding more correlated features to a naive Bayes
  model makes its confidence go UP while its accuracy stays flat. Confidence is not
  evidence.""")


def experiment_covariance_spectrum() -> None:
    """README §9: NB / LDA / QDA are one spectrum, and n decides where to sit."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the covariance spectrum  (README §9)")
    print("=" * 88)
    print("""
Gaussian NB, LDA, and QDA differ ONLY in what they assume about the covariance: diagonal,
shared-full, and per-class-full. That is a bias-variance dial, and the right setting
depends on n relative to d^2.

Data generated with genuinely DIFFERENT per-class covariances, so QDA is the correct model
and the only question is whether there is enough data to fit it:
""")
    rng = np.random.default_rng(3)
    d, K = 8, 3

    covs = []
    for k in range(K):
        A = rng.standard_normal((d, d))
        covs.append(A @ A.T / d + 0.5 * np.eye(d))
    means = rng.standard_normal((K, d)) * 1.5

    def make(size):
        y = rng.integers(0, K, size)
        X = np.array([rng.multivariate_normal(means[c], covs[c]) for c in y])
        return X, y

    X_te, y_te = make(6000)

    print(f"  d = {d}, K = {K};  QDA needs ~d^2/2 = {d ** 2 // 2} samples PER CLASS\n")
    print(f"  {'n per class':>12s}  {'GaussianNB':>11s}  {'LDA':>8s}  {'QDA':>8s}  "
          f"{'RDA(0.5)':>10s}  {'winner':>12s}")
    print("  " + "-" * 70)

    for n_per in (12, 25, 50, 150, 600, 3000):
        X_tr, y_tr = make(n_per * K)
        scores = {}
        for name, model in [("GaussianNB", GaussianNB()), ("LDA", LDA(reg=1e-3)),
                            ("QDA", QDA(reg=1e-3)), ("RDA(0.5)", RDA(gamma=0.5, reg=1e-3))]:
            try:
                scores[name] = model.fit(X_tr, y_tr).score(X_te, y_te)
            except np.linalg.LinAlgError:
                scores[name] = float("nan")
        winner = max(scores, key=lambda k: (scores[k] if not np.isnan(scores[k]) else -1))
        print(f"  {n_per:12d}  {scores['GaussianNB']:11.4f}  {scores['LDA']:8.4f}  "
              f"{scores['QDA']:8.4f}  {scores['RDA(0.5)']:10.4f}  {winner:>12s}")

    print("""
  QDA is the CORRECT model here — the data really does have different covariances per class
  — and it still loses at small n, because estimating K full covariances from a handful of
  points per class is hopeless. LDA's wrong-but-cheap shared covariance wins until there is
  enough data to afford the truth.

  RDA, which interpolates between them, is competitive across the whole range without
  having to choose in advance. That is the practical recommendation when you do not know
  where you sit on the curve: regularize toward the pooled covariance and cross-validate
  gamma.

  This is the bias-variance trade in an unusually clean form: three models, one hypothesis
  space each nested in the next, and a crossover you can locate empirically.""")


def experiment_ng_jordan() -> None:
    """README §12: generative wins at small n, discriminative at large n."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — Ng & Jordan: the sample-complexity crossover  (README §12)")
    print("=" * 88)
    print("""
The often-quoted version is "naive Bayes wins at small n, logistic regression at large n".
The actual result is narrower: NB converges to its OWN (higher) asymptote at O(log d / n)
while LR converges to its own (lower) asymptote at O(d / n). Whether you SEE a crossover
depends on how big the gap between those two asymptotes is — which is to say, on how badly
NB's assumption is violated.

So we sweep both axes: how wrong the assumption is (within-class correlation) and n. Data
is class-conditional Gaussian, which is NB's model family when rho = 0 and progressively
less so as rho grows. Logistic regression is unregularized, to isolate the sample-
complexity effect from a shrinkage effect.
""")
    try:
        from sklearn.linear_model import LogisticRegression
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        print("  [SKIP]  sklearn not installed")
        return

    d = 20
    sizes = (20, 50, 100, 300, 1000, 5000)
    n_repeats = 25

    print(f"  d = {d};  each cell is (naive Bayes accuracy - logistic accuracy)\n")
    header = f"  {'rho':>6s}  " + "  ".join(f"{f'n={n}':>10s}" for n in sizes)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rho in (0.0, 0.2, 0.5, 0.8):
        rng = np.random.default_rng(4)
        cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
        L = np.linalg.cholesky(cov)
        mu = rng.standard_normal(d) * 0.45

        def make(size):
            y = rng.integers(0, 2, size)
            return rng.standard_normal((size, d)) @ L.T + np.outer(y, mu), y

        X_te, y_te = make(20000)

        cells = []
        for n_train in sizes:
            nb_scores, lr_scores = [], []
            for _ in range(n_repeats):
                X_tr, y_tr = make(n_train)
                if np.unique(y_tr).size < 2:
                    continue
                nb_scores.append(GaussianNB().fit(X_tr, y_tr).score(X_te, y_te))
                lr_scores.append(LogisticRegression(penalty=None, max_iter=3000)
                                 .fit(X_tr, y_tr).score(X_te, y_te))
            gap = float(np.mean(nb_scores) - np.mean(lr_scores))
            cells.append(f"{gap:+10.4f}")
        print(f"  {rho:6.1f}  " + "  ".join(cells))

    print("""
  Positive means naive Bayes is ahead. Read the table by rows and by columns.

  ALONG A ROW (increasing n): every row decays toward zero from n = 50 rightward. That is
  the Ng & Jordan effect — logistic regression's O(d/n) variance term is what is being paid
  down, so whichever way the gap points, it closes as data arrives.

  The n = 20 column is the exception and is worth understanding rather than ignoring: with
  d = 20 and n = 20, unregularized logistic regression is at its interpolation threshold
  and behaves erratically, which is why the first column does not fit the pattern. This is
  also a small demonstration of why sklearn regularizes by default (03.04 §10).

  DOWN A COLUMN (increasing rho): NB's advantage falls steadily as its assumption is
  violated harder. At rho = 0 it is the correctly-specified model and wins at every n. At
  rho = 0.2 it wins in the middle of the range and loses at both ends — an actual crossover.
  By rho = 0.5 and 0.8 it loses everywhere, and the gap is large.

  So the folklore is half right, and the half it gets wrong matters. NAIVE BAYES IS NOT
  FAVOURED BY SMALL n ALONE. It is favoured when small n is combined with an assumption
  that is close enough to true. When the assumption is badly violated its asymptotic
  penalty dominates at every sample size, and there is no crossover to find — the rho = 0.8
  row never comes close to zero.

  The practical rule is therefore not "use naive Bayes when n is small" but: "use it when n
  is small AND the features are close to conditionally independent given the class." That
  is exactly the situation in bag-of-words text with a small labelled set — which is why NB
  survived there for two decades and largely nowhere else.""")


def experiment_lda_vs_pca() -> None:
    """README §11: PCA's top direction can carry no class information at all."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — LDA vs PCA as projections  (README §11)")
    print("=" * 88)
    print("""
PCA finds the directions the DATA varies in; LDA finds the directions the CLASSES differ
in. These are different questions, and the answers can be orthogonal.

Constructed deliberately: a high-variance direction with no class signal, and a
low-variance direction that separates the classes perfectly.
""")
    rng = np.random.default_rng(5)
    n = 1000
    y = rng.integers(0, 2, n)

    # Axis 0: variance 25, identical for both classes -> no information.
    # Axis 1: variance 1, means separated by 3       -> all the information.
    X = np.column_stack([rng.standard_normal(n) * 5.0,
                         rng.standard_normal(n) * 1.0 + y * 3.0])
    # Rotate so neither axis is privileged.
    angle = np.pi / 6
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = X @ R.T

    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_dir = Vt[0]
    lda_dir = LDA().fit(X, y).scalings_[:, 0]
    lda_dir = lda_dir / np.linalg.norm(lda_dir)

    def separation(direction):
        """Between-class mean gap in units of within-class standard deviation."""
        proj = X @ direction
        gap = abs(proj[y == 1].mean() - proj[y == 0].mean())
        spread = np.sqrt(0.5 * (proj[y == 0].var() + proj[y == 1].var()))
        return gap / spread

    print(f"  {'projection':<22s}  {'variance captured':>18s}  "
          f"{'class separation':>17s}  {'1-D accuracy':>13s}")
    print("  " + "-" * 76)

    for name, direction in [("PCA (top component)", pca_dir), ("LDA (discriminant)", lda_dir)]:
        proj = (X @ direction)[:, None]
        variance = float(np.var(X @ direction) / np.var(X, axis=0).sum())
        acc = LDA().fit(proj, y).score(proj, y)
        print(f"  {name:<22s}  {variance:18.4f}  {separation(direction):17.4f}  {acc:13.4f}")

    print(f"\n  angle between the two directions: "
          f"{np.degrees(np.arccos(abs(pca_dir @ lda_dir))):.1f} degrees")

    print("""
  PCA's top component captures nearly all the variance and separates the classes barely at
  all — a 1-D classifier on it is close to chance. LDA's direction captures a small
  fraction of the variance and separates the classes almost perfectly.

  The two directions are close to orthogonal, which is the sharpest possible version of the
  point: the direction your data varies in most can be exactly the direction that carries
  no label information.

  The practical warning is about the common 'PCA to 10 components, then classify' pipeline.
  It is unsupervised, so it selects components by variance and can discard precisely the
  low-variance direction the classes live in. If you are reducing dimensions before a
  supervised task, either use a supervised method, or keep enough components that you have
  not thrown the signal away — and check.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_independence()
    experiment_calibration()
    experiment_covariance_spectrum()
    experiment_ng_jordan()
    experiment_lda_vs_pca()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
