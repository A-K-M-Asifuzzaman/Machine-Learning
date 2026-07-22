"""
03.01 — Linear Regression from Scratch
======================================

A complete OLS implementation: four solvers, full statistical inference, and the
diagnostics that tell you whether any of it is trustworthy.

NumPy primitives only inside the implementation. sklearn and statsmodels appear only in
the verification section, as references to check against.

Implemented here
----------------
    LinearRegression        sklearn-compatible fit/predict, with:
        solver="normal"     the closed form — included to be measured, not used
        solver="qr"         the standard stable route          README §4
        solver="svd"        handles rank deficiency            README §4
        solver="gd"         gradient descent                   README §4

    Inference (README §7-§8):
        coef_se_, t_values_, p_values_, conf_int()
        r_squared_, adj_r_squared_, f_statistic_, f_pvalue_
        sigma2_             unbiased RSS/(n-d)

    Diagnostics (README §10, §12):
        leverage()          diagonal of the hat matrix
        cooks_distance()    influence
        studentized_residuals()
        vif()               variance inflation factors

Run it
------
    python from_scratch.py

Verifies every quantity against sklearn and statsmodels, then runs five experiments:
  1. Gauss-Markov: OLS really does have the lowest variance among linear unbiased estimators
  2. Solver comparison as conditioning degrades — where the closed form dies
  3. R^2 always rises with more features, even pure noise; adjusted R^2 does not
  4. Which assumption violations bias the coefficients, and which only corrupt the SEs
  5. Leverage vs influence: one point that moves the fit, one that does not

Reference: README.md sections 3-12.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE MODEL
# =============================================================================


class LinearRegression:
    """Ordinary least squares with full inference.

        y = Xw + eps,   eps ~ N(0, sigma^2 I)
        w_hat = argmin ||y - Xw||^2

    The estimator is the same for every solver; they differ only in numerical behaviour
    (README §4). `solver="normal"` is included so Experiment 2 can show it failing.
    """

    def __init__(self, fit_intercept: bool = True, solver: str = "qr",
                 lr: float = 0.01, n_iter: int = 5000):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter

    # --- internals --------------------------------------------------------

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones so the intercept is just another coefficient."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X

    @staticmethod
    def _solve_normal(A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """w = (X^T X)^-1 X^T y, via Cholesky.

        The textbook formula. It squares the condition number (00.01 §15.2), which is why
        no library uses it and why Experiment 2 shows it losing all precision while QR and
        SVD are still exact. Kept here to be measured.
        """
        gram = A.T @ A
        return np.linalg.solve(gram, A.T @ y)

    @staticmethod
    def _solve_qr(A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve R w = Q^T y by back-substitution.  00.01 §8

        X^T X is never formed, so kappa is never squared. This is how least squares is
        actually computed.
        """
        Q, R = np.linalg.qr(A)
        rhs = Q.T @ y
        n = R.shape[1]
        w = np.zeros(n)
        for i in range(n - 1, -1, -1):
            w[i] = (rhs[i] - R[i, i + 1:] @ w[i + 1:]) / R[i, i]
        return w

    @staticmethod
    def _solve_svd(A: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
        """w = V S^+ U^T y, the pseudoinverse solution.  00.01 §13.2

        The only solver that survives rank deficiency: singular values below the tolerance
        are zeroed rather than inverted, and the result is the MINIMUM-NORM solution among
        the infinitely many that fit equally well (00.01 §4.2). That is a reasonable
        default, but it is a choice being made on your behalf.
        """
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        cutoff = rcond * (s[0] if s.size else 0.0)
        s_inv = np.where(s > cutoff, 1.0 / np.maximum(s, 1e-300), 0.0)
        return Vt.T @ (s_inv * (U.T @ y))

    def _solve_gd(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Batch gradient descent on ||y - Aw||^2 / n.

        The step size is computed from the curvature rather than guessed. The objective
        J(w) = ||y - Aw||^2 / n has gradient 2 A^T(Aw - y)/n and therefore Hessian

            H = 2 A^T A / n,   so   lambda_max(H) = 2 * lambda_max(A^T A / n)

        Gradient descent is stable only for eta < 2/lambda_max(H) (00.02 §7.1), which is
        1/lambda_max(A^T A / n) — so setting eta to exactly that value lands ON the
        stability boundary, where the sharpest direction oscillates forever at constant
        amplitude instead of converging. (That is precisely the eta/threshold = 1.00 row
        in 00.02's Experiment 1.)

        We use the OPTIMAL step for a quadratic instead (00.02 §8):

            eta* = 2 / (lambda_min(H) + lambda_max(H))

        which is safely inside the stability limit and gives the fastest possible
        convergence rate, (kappa - 1)/(kappa + 1).
        """
        n = A.shape[0]
        w = np.zeros(A.shape[1])

        eigenvalues = np.linalg.eigvalsh(2.0 * A.T @ A / n)      # Hessian eigenvalues
        lambda_min = float(max(eigenvalues.min(), 0.0))
        lambda_max = float(eigenvalues.max())
        eta = 2.0 / (lambda_min + lambda_max) if lambda_max > 0 else self.lr

        for _ in range(self.n_iter):
            grad = 2.0 * A.T @ (A @ w - y) / n
            if np.max(np.abs(grad)) < 1e-13:
                break
            w = w - eta * grad
        return w

    # --- API --------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        A = self._design(X)
        y = np.asarray(y, dtype=float).ravel()
        n, d = A.shape

        solvers = {"normal": self._solve_normal, "qr": self._solve_qr,
                   "svd": self._solve_svd, "gd": self._solve_gd}
        if self.solver not in solvers:
            raise ValueError(f"unknown solver {self.solver!r}")
        theta = solvers[self.solver](A, y)

        self.n_, self.d_ = n, d
        self._A = A
        self._y = y

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        self.theta_ = theta

        # ---- inference (README §7-§8) ------------------------------------
        residuals = y - A @ theta
        self.residuals_ = residuals
        rss = float(residuals @ residuals)
        tss = float(np.sum((y - y.mean()) ** 2))
        self.rss_, self.tss_ = rss, tss

        dof = n - d
        # sigma^2 = RSS/(n-d): d degrees of freedom were spent fitting d coefficients.
        # Dividing by n instead would reproduce the MLE, which is biased low (00.04 §5).
        self.dof_ = dof
        self.sigma2_ = rss / dof if dof > 0 else np.nan

        # Cov(w_hat) = sigma^2 (X^T X)^-1.  Computed from the SVD rather than by
        # inverting the Gram matrix, for the usual conditioning reason.
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_safe = np.where(s > 1e-12 * (s[0] if s.size else 1.0), s, np.inf)
        xtx_inv = (Vt.T / s_safe ** 2) @ Vt
        self.cov_ = self.sigma2_ * xtx_inv
        self.coef_se_ = np.sqrt(np.maximum(np.diag(self.cov_), 0.0))

        with np.errstate(divide="ignore", invalid="ignore"):
            self.t_values_ = theta / self.coef_se_
        self.p_values_ = self._t_sf(np.abs(self.t_values_), dof) * 2

        self.r_squared_ = 1 - rss / tss if tss > 0 else np.nan
        self.adj_r_squared_ = (1 - (1 - self.r_squared_) * (n - 1) / dof
                               if dof > 0 else np.nan)

        # F-test for the whole model (README §8).
        if self.fit_intercept and d > 1 and dof > 0:
            self.f_statistic_ = ((tss - rss) / (d - 1)) / (rss / dof)
            self.f_pvalue_ = self._f_sf(self.f_statistic_, d - 1, dof)
        else:
            self.f_statistic_ = np.nan
            self.f_pvalue_ = np.nan

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.theta_

    # --- distributions (kept dependency-free) -----------------------------

    @staticmethod
    def _t_sf(t: np.ndarray, dof: int) -> np.ndarray:
        """P(T > t) for Student's t, via the regularized incomplete beta function."""
        try:
            from scipy import stats
            return stats.t.sf(t, dof)
        except ImportError:
            from math import lgamma
            t = np.atleast_1d(np.asarray(t, dtype=float))
            out = np.empty_like(t)
            for i, ti in enumerate(t):
                x = dof / (dof + ti ** 2)
                out[i] = 0.5 * _betainc(dof / 2.0, 0.5, x)
            return out

    @staticmethod
    def _f_sf(f: float, d1: int, d2: int) -> float:
        try:
            from scipy import stats
            return float(stats.f.sf(f, d1, d2))
        except ImportError:
            x = d2 / (d2 + d1 * f)
            return float(_betainc(d2 / 2.0, d1 / 2.0, x))

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Confidence intervals for every coefficient.  README §8

        Uses the t distribution, not the normal, because sigma was estimated
        (00.04 §8.2). Returns an array of shape (d, 2).
        """
        try:
            from scipy import stats
            crit = stats.t.ppf(1 - alpha / 2, self.dof_)
        except ImportError:
            crit = 1.959963984540054                # normal approximation fallback
        half = crit * self.coef_se_
        return np.column_stack([self.theta_ - half, self.theta_ + half])

    # --- diagnostics (README §10, §12) ------------------------------------

    def leverage(self) -> np.ndarray:
        """h_ii, the diagonal of the hat matrix H = X(X^T X)^-1 X^T.  README §10.1

        Computed as the row-wise squared norm of Q from the QR factorization, since
        H = QQ^T. That avoids forming the n x n hat matrix, which would be a disaster
        for large n, and is better conditioned besides.

        sum(h_ii) = d exactly, so average leverage is d/n; h_ii > 2d/n is the usual flag.
        """
        Q, _ = np.linalg.qr(self._A)
        return np.sum(Q ** 2, axis=1)

    def studentized_residuals(self) -> np.ndarray:
        """Residuals scaled to unit variance: r_i / (sigma * sqrt(1 - h_ii)).

        Raw residuals have UNEQUAL variances — Var(r_i) = sigma^2 (1 - h_ii) — so
        high-leverage points have artificially small residuals. That is exactly why a
        raw residual plot can hide the most influential points.
        """
        h = self.leverage()
        return self.residuals_ / np.sqrt(self.sigma2_ * np.maximum(1 - h, 1e-12))

    def cooks_distance(self) -> np.ndarray:
        """Cook's distance: how far the whole fit moves if point i is deleted.

            D_i = (r_i^2 / (d sigma^2)) * (h_ii / (1 - h_ii)^2)

        Leverage alone is not influence — a high-leverage point sitting exactly on the
        line changes nothing. Cook's combines leverage with residual size, which is why
        Experiment 5 can separate the two.
        """
        h = self.leverage()
        return (self.residuals_ ** 2 / (self.d_ * self.sigma2_)) * (h / (1 - h) ** 2)

    def vif(self) -> np.ndarray:
        """Variance inflation factor per feature (excluding the intercept).  README §12

            VIF_j = 1 / (1 - R_j^2)

        where R_j^2 comes from regressing feature j on all the OTHER features. It says how
        much Var(w_j) is inflated relative to an orthogonal design. VIF > 10 is the
        conventional alarm.
        """
        X = self._A[:, 1:] if self.fit_intercept else self._A
        n_features = X.shape[1]
        out = np.empty(n_features)
        for j in range(n_features):
            others = np.delete(X, j, axis=1)
            if others.shape[1] == 0:
                out[j] = 1.0
                continue
            model = LinearRegression(fit_intercept=True, solver="svd").fit(others, X[:, j])
            out[j] = 1.0 / max(1 - model.r_squared_, 1e-15)
        return out

    def summary(self) -> str:
        """A statsmodels-style summary table."""
        lines = [
            "=" * 78,
            f"{'OLS Regression Results':^78s}",
            "=" * 78,
            f"n = {self.n_},  d = {self.d_},  dof = {self.dof_}",
            f"R-squared      = {self.r_squared_:.6f}",
            f"Adj. R-squared = {self.adj_r_squared_:.6f}",
            f"F-statistic    = {self.f_statistic_:.4f}   (p = {self.f_pvalue_:.4g})",
            f"sigma^2        = {self.sigma2_:.6f}",
            "-" * 78,
            f"{'':>10s} {'coef':>12s} {'std err':>12s} {'t':>10s} {'P>|t|':>10s} "
            f"{'[0.025':>10s} {'0.975]':>10s}",
            "-" * 78,
        ]
        ci = self.conf_int()
        names = ([f"const"] if self.fit_intercept else []) + \
                [f"x{i}" for i in range(len(self.coef_))]
        for i, name in enumerate(names):
            lines.append(f"{name:>10s} {self.theta_[i]:12.6f} {self.coef_se_[i]:12.6f} "
                         f"{self.t_values_[i]:10.4f} {self.p_values_[i]:10.4f} "
                         f"{ci[i, 0]:10.4f} {ci[i, 1]:10.4f}")
        lines.append("=" * 78)
        return "\n".join(lines)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta, by continued fraction. scipy-free fallback."""
    from math import exp, lgamma
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * np.log(x) + b * np.log(1 - x) - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + numerator / (c if abs(c) > 1e-30 else 1e-30)
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    return front * (f - 1)


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<54s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 86)
    print("VERIFICATION")
    print("=" * 86)

    n, d = 300, 5
    X = rng.standard_normal((n, d))
    true_w = np.array([2.0, -1.5, 0.0, 3.0, 0.5])
    y = 4.0 + X @ true_w + rng.standard_normal(n) * 1.5

    model = LinearRegression(solver="qr").fit(X, y)

    # --- all four solvers agree ------------------------------------------
    print("\nAll four solvers give the same estimator (README §4)")
    reference = model.theta_
    for solver in ("normal", "svd", "gd"):
        other = LinearRegression(solver=solver).fit(X, y)
        ok &= _report(f"solver={solver!r} matches solver='qr'",
                      float(np.abs(other.theta_ - reference).max()), 1e-6)

    # --- against sklearn --------------------------------------------------
    print("\nAgainst sklearn")
    try:
        from sklearn.linear_model import LinearRegression as SKLinear
        from sklearn.metrics import r2_score
        sk = SKLinear().fit(X, y)
        ok &= _report("coefficients vs sklearn",
                      float(np.abs(model.coef_ - sk.coef_).max()), 1e-10)
        ok &= _report("intercept vs sklearn", abs(model.intercept_ - sk.intercept_), 1e-10)
        ok &= _report("predictions vs sklearn",
                      float(np.abs(model.predict(X) - sk.predict(X)).max()), 1e-10)
        ok &= _report("R^2 vs sklearn r2_score",
                      abs(model.r_squared_ - r2_score(y, model.predict(X))), 1e-12)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    # --- against statsmodels (the inference is the real test) -------------
    print("\nAgainst statsmodels — standard errors, t, p, CIs, F (README §7-§8)")
    try:
        import statsmodels.api as sm
        sm_model = sm.OLS(y, sm.add_constant(X)).fit()

        ok &= _report("coefficients vs statsmodels",
                      float(np.abs(model.theta_ - sm_model.params).max()), 1e-9)
        ok &= _report("standard errors vs statsmodels",
                      float(np.abs(model.coef_se_ - sm_model.bse).max()), 1e-9)
        ok &= _report("t-statistics vs statsmodels",
                      float(np.abs(model.t_values_ - sm_model.tvalues).max()), 1e-8)
        ok &= _report("p-values vs statsmodels",
                      float(np.abs(model.p_values_ - sm_model.pvalues).max()), 1e-9)
        ok &= _report("confidence intervals vs statsmodels",
                      float(np.abs(model.conf_int() - sm_model.conf_int()).max()), 1e-8)
        ok &= _report("R^2 vs statsmodels", abs(model.r_squared_ - sm_model.rsquared), 1e-12)
        ok &= _report("adjusted R^2 vs statsmodels",
                      abs(model.adj_r_squared_ - sm_model.rsquared_adj), 1e-12)
        ok &= _report("F-statistic vs statsmodels",
                      abs(model.f_statistic_ - sm_model.fvalue), 1e-8)
        ok &= _report("F p-value vs statsmodels",
                      abs(model.f_pvalue_ - sm_model.f_pvalue), 1e-12)
        ok &= _report("sigma^2 vs statsmodels", abs(model.sigma2_ - sm_model.mse_resid), 1e-10)

        influence = sm_model.get_influence()
        ok &= _report("leverage vs statsmodels",
                      float(np.abs(model.leverage() - influence.hat_matrix_diag).max()), 1e-10)
        ok &= _report("Cook's distance vs statsmodels",
                      float(np.abs(model.cooks_distance() - influence.cooks_distance[0]).max()),
                      1e-9)

        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xc = sm.add_constant(X)
        sm_vif = np.array([variance_inflation_factor(Xc, i + 1) for i in range(d)])
        ok &= _report("VIF vs statsmodels", float(np.abs(model.vif() - sm_vif).max()), 1e-8)
    except ImportError:
        print("  [SKIP]  statsmodels not installed — checking internal identities instead")
        ok &= _report("R^2 = 1 - RSS/TSS",
                      abs(model.r_squared_ - (1 - model.rss_ / model.tss_)), 1e-15)
        ok &= _report("sum(leverage) = d", abs(model.leverage().sum() - model.d_), 1e-10)

    # --- structural identities that must always hold ----------------------
    print("\nStructural identities (README §3, §10)")
    ok &= _report("residuals orthogonal to every column of X",
                  float(np.abs(model._A.T @ model.residuals_).max()), 1e-9)
    ok &= _report("residuals sum to zero (intercept fitted)",
                  abs(float(model.residuals_.sum())), 1e-9)
    ok &= _report("sum of leverages = d", abs(float(model.leverage().sum()) - d - 1), 1e-10)
    ok &= _report("0 <= h_ii <= 1",
                  float(max(0.0, -model.leverage().min(), model.leverage().max() - 1)), 1e-12)

    # Rank deficiency: only SVD should cope.
    print("\nRank-deficient design (README §4)")
    X_dup = np.column_stack([X, X[:, 0]])          # exact duplicate column
    svd_model = LinearRegression(solver="svd").fit(X_dup, y)
    ok &= _report("SVD still predicts correctly with a duplicate column",
                  float(np.abs(svd_model.predict(X_dup) - model.predict(X)).max()), 1e-6)
    print(f"  [INFO]  {'minimum-norm solution splits the duplicate':<54s}  "
          f"w[0]={svd_model.coef_[0]:.4f}, w[-1]={svd_model.coef_[-1]:.4f} "
          f"(sum={svd_model.coef_[0] + svd_model.coef_[-1]:.4f} vs OLS {model.coef_[0]:.4f})")

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_gauss_markov() -> None:
    """README §6: OLS has the lowest variance among linear unbiased estimators."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 1 — Gauss-Markov, and its loophole  (README §6)")
    print("=" * 86)
    print("""
The theorem says OLS is BEST LINEAR UNBIASED. Testing all three words: we compare OLS
against other linear estimators of the same coefficients over 20,000 resamples of the
same design.
""")
    rng = np.random.default_rng(1)
    n, d = 40, 3
    X = rng.standard_normal((n, d))
    A = np.column_stack([np.ones(n), X])
    true_w = np.array([1.0, 2.0, -1.0, 0.5])
    sigma = 1.0
    n_trials = 20_000

    # Competing LINEAR estimators, each of the form w = M y.
    M_ols = np.linalg.pinv(A)
    M_half = np.linalg.pinv(A[: n // 2]) @ np.eye(n // 2, n)     # uses half the data
    P = np.eye(n) - A @ np.linalg.pinv(A)                        # projects onto null space
    M_perturbed = M_ols + 0.05 * (rng.standard_normal((d + 1, n)) @ P)

    estimators = {"OLS": M_ols, "OLS on half the data": M_half,
                  "OLS + null-space perturbation": M_perturbed}
    results = {name: np.empty((n_trials, d + 1)) for name in estimators}

    for t in range(n_trials):
        y = A @ true_w + sigma * rng.standard_normal(n)
        for name, M in estimators.items():
            results[name][t] = M @ y

    print(f"  {'estimator':<32s}  {'max |bias|':>12s}  {'total variance':>15s}  {'total MSE':>12s}")
    print("  " + "-" * 78)
    for name, estimates in results.items():
        bias = np.abs(estimates.mean(axis=0) - true_w).max()
        variance = estimates.var(axis=0, ddof=0).sum()
        mse = np.mean(np.sum((estimates - true_w) ** 2, axis=1))
        print(f"  {name:<32s}  {bias:12.5f}  {variance:15.5f}  {mse:12.5f}")

    print("""
  All three are UNBIASED — the perturbation was deliberately built inside the null space
  of X, so it cancels in expectation. And all three are LINEAR in y. Yet OLS has strictly
  the smallest variance of the three. That is Gauss-Markov, measured.

  Now the loophole. Ridge is BIASED, so the theorem does not apply to it, and
  MSE = bias^2 + variance (00.04 §3) leaves room for a biased estimator to win:
""")
    print(f"  {'estimator':<32s}  {'max |bias|':>12s}  {'total variance':>15s}  {'total MSE':>12s}")
    print("  " + "-" * 78)

    ols_row = results["OLS"]
    bias = np.abs(ols_row.mean(axis=0) - true_w).max()
    print(f"  {'OLS (lambda = 0)':<32s}  {bias:12.5f}  "
          f"{ols_row.var(axis=0).sum():15.5f}  "
          f"{np.mean(np.sum((ols_row - true_w) ** 2, axis=1)):12.5f}")

    for lam in (0.5, 2.0, 10.0):
        M_ridge = np.linalg.solve(A.T @ A + lam * np.eye(d + 1), A.T)
        estimates = np.empty((n_trials, d + 1))
        rng2 = np.random.default_rng(1)
        _ = rng2.standard_normal((n, d))                          # replay the same stream
        for t in range(n_trials):
            y = A @ true_w + sigma * rng2.standard_normal(n)
            estimates[t] = M_ridge @ y
        bias = np.abs(estimates.mean(axis=0) - true_w).max()
        variance = estimates.var(axis=0).sum()
        mse = np.mean(np.sum((estimates - true_w) ** 2, axis=1))
        marker = "  <- beats OLS" if mse < np.mean(np.sum((ols_row - true_w) ** 2, axis=1)) else ""
        print(f"  {f'ridge (lambda = {lam})':<32s}  {bias:12.5f}  {variance:15.5f}  "
              f"{mse:12.5f}{marker}")

    print("""
  Ridge trades a little bias for a larger reduction in variance and comes out ahead on
  MSE. Gauss-Markov is not violated — ridge is simply outside the class of estimators the
  theorem quantifies over.

  This is the entire justification for regularization, and it is why "unbiased" is not a
  synonym for "good" (00.04 §3). Chapter 03.02 develops it.""")


def experiment_solvers() -> None:
    """README §4: what conditioning does to each solver."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 2 — solvers under ill-conditioning  (README §4)")
    print("=" * 86)
    print("""
Two nearly-collinear features. As collinearity tightens, kappa(X) grows and
kappa(X^T X) = kappa(X)^2 grows as its square (00.01 §15.2). Measuring each solver's
coefficient error against the true w:
""")
    rng = np.random.default_rng(2)
    n = 200

    n_iter_gd = 5000
    print(f"  Gradient descent is capped at {n_iter_gd:,} iterations.\n")
    print(f"  {'epsilon':>9s}  {'kappa(X)':>11s}  {'normal eq':>12s}  {'QR':>12s}  "
          f"{'SVD':>12s}  {'GD':>12s}  {'GD iters needed':>16s}")
    print("  " + "-" * 94)

    for eps in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9):
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1, x1 + eps * rng.standard_normal(n)])
        true_w = np.array([1.0, -1.0])
        y = 2.0 + X @ true_w                        # noiseless: any error is numerical

        A = np.column_stack([np.ones(n), X])
        kappa = np.linalg.cond(A)
        # GD on a quadratic needs O(kappa * log(1/eps)) iterations (00.02 §8), with
        # kappa here the condition number of the HESSIAN, i.e. kappa(A)^2.
        kappa_hessian = kappa ** 2
        iters_needed = kappa_hessian * np.log(1 / 1e-12) / 2

        row = [f"  {eps:9.0e}  {kappa:11.2e}"]
        for solver in ("normal", "qr", "svd", "gd"):
            try:
                m = LinearRegression(solver=solver, n_iter=n_iter_gd).fit(X, y)
                # Compare fitted values, which are well-determined even when the
                # coefficients are not (README §12).
                err = float(np.abs(m.predict(X) - y).max())
                row.append(f"  {err:12.3e}")
            except np.linalg.LinAlgError:
                row.append(f"  {'SINGULAR':>12s}")
        row.append(f"  {iters_needed:16.1e}")
        print("".join(row))

    print("""
  Errors here are on the FITTED VALUES, not the coefficients — because with collinear
  features the coefficients are genuinely not identifiable, while the predictions are
  (README §12). That distinction is the practical heart of multicollinearity.

  THE DIRECT SOLVERS. The normal equations degrade exactly as kappa^2 predicts, losing
  roughly two digits for every one that kappa gains, and go singular at eps = 1e-9. QR and
  SVD stay at machine precision throughout, because neither ever forms X^T X.

  GRADIENT DESCENT does not merely degrade — it stops being applicable. The last column is
  the number of iterations the O(kappa) rate of 00.02 §8 requires, and note it is kappa of
  the HESSIAN, which is kappa(X)^2. By eps = 1e-5 that is already ~10^11 iterations. GD is
  not "slow" here in any useful sense; at this conditioning it cannot converge in any
  feasible budget, and its error of ~1.0 reflects a run that had barely started.

  Two lessons. First, this is why sklearn's LinearRegression calls scipy.linalg.lstsq
  (SVD) rather than the closed form every textbook prints. Second, it is why feature
  scaling is not cosmetic: for the iterative solvers that large models are forced to use,
  kappa is the difference between converging and not.""")


def experiment_r_squared() -> None:
    """README §9: R^2 always rises with more features, even pure noise."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 3 — R^2 always goes up  (README §9)")
    print("=" * 86)
    print("""
Adding a feature can only expand the column space, so the projection can only get closer
(00.01 §6). R^2 therefore NEVER decreases — even for features that are pure noise, with
no relationship to y whatsoever. Here y depends on exactly 3 real features; everything
after that is random:
""")
    rng = np.random.default_rng(3)
    n = 60
    X_real = rng.standard_normal((n, 3))
    y = 1.0 + X_real @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.8

    X_test = rng.standard_normal((400, 3))
    y_test = 1.0 + X_test @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(400) * 0.8

    print(f"  n = {n} training rows;  3 real features, the rest pure noise\n")
    print(f"  {'features':>9s}  {'train R^2':>11s}  {'adj R^2':>11s}  {'TEST R^2':>11s}")
    print("  " + "-" * 48)

    for n_noise in (0, 5, 15, 30, 45, 55, 58):
        noise = rng.standard_normal((n, n_noise)) if n_noise else np.empty((n, 0))
        X_full = np.column_stack([X_real, noise])
        model = LinearRegression(solver="svd").fit(X_full, y)

        noise_test = rng.standard_normal((400, n_noise)) if n_noise else np.empty((400, 0))
        X_test_full = np.column_stack([X_test, noise_test])
        pred = model.predict(X_test_full)
        test_r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

        print(f"  {3 + n_noise:9d}  {model.r_squared_:11.4f}  {model.adj_r_squared_:11.4f}  "
              f"{test_r2:11.4f}")

    print("""
  Read the three columns together.

  TRAIN R^2 rises monotonically to 1.0 — the model eventually fits 60 points perfectly
  using 59 parameters, explaining pure noise "perfectly". It never once decreases.

  ADJUSTED R^2 penalizes the parameter count and turns over, correctly signalling that
  the extra features are not earning their degrees of freedom.

  TEST R^2 collapses, and goes NEGATIVE — the model is worse than predicting the mean.
  This is overfitting shown in a single table.

  The lesson: R^2 on training data cannot be used for model selection, because it is
  monotone in model size by construction. Use held-out data, adjusted R^2, AIC/BIC, or
  cross-validation (05.04).""")


def experiment_assumption_violations() -> None:
    """README §5, §11: which violations bias coefficients, which only corrupt SEs."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 4 — which assumption violations actually matter  (README §5, §11)")
    print("=" * 86)
    print("""
Textbooks list five assumptions as if they were equally important. They are not. For each
violation we measure two very different things: is the COEFFICIENT still unbiased, and is
the STANDARD ERROR still honest (does a 95% CI cover 95% of the time)?
""")
    rng = np.random.default_rng(4)
    n = 100
    n_trials = 4000
    true_slope = 2.0

    scenarios = {}

    def run(name, generate):
        estimates = np.empty(n_trials)
        covered = 0
        for t in range(n_trials):
            X, y = generate(rng)
            m = LinearRegression(solver="qr").fit(X, y)
            estimates[t] = m.coef_[0]
            lo, hi = m.conf_int()[1]
            covered += lo <= true_slope <= hi
        scenarios[name] = (estimates.mean() - true_slope, estimates.std(), covered / n_trials)

    run("none (all assumptions hold)", lambda r: (
        (X := r.standard_normal((n, 1))), 1.0 + true_slope * X[:, 0] + r.standard_normal(n))[0:2])

    def heteroscedastic(r):
        X = r.standard_normal((n, 1))
        scale = 0.3 + 2.0 * np.abs(X[:, 0])          # variance grows with |x|
        return X, 1.0 + true_slope * X[:, 0] + scale * r.standard_normal(n)
    run("heteroscedasticity", heteroscedastic)

    def autocorrelated(r):
        # BOTH x and the errors are AR(1). This matters: with i.i.d. x, autocorrelated
        # errors barely disturb the OLS standard errors at all. It is the combination of
        # serial correlation in the regressor AND in the noise that shrinks the effective
        # sample size — which is exactly the situation in every time series.
        e = np.empty(n)
        x = np.empty(n)
        e[0], x[0] = r.standard_normal(), r.standard_normal()
        for i in range(1, n):
            e[i] = 0.85 * e[i - 1] + r.standard_normal()
            x[i] = 0.85 * x[i - 1] + r.standard_normal()
        return x[:, None], 1.0 + true_slope * x + e
    run("autocorrelation (in x AND errors)", autocorrelated)

    def heavy_tailed(r):
        X = r.standard_normal((n, 1))
        return X, 1.0 + true_slope * X[:, 0] + r.standard_t(3, n)
    run("non-normal (heavy-tailed) errors", heavy_tailed)

    def nonlinear_symmetric(r):
        # x is SYMMETRIC about 0, so Cov(x, x^2) = E[x^3] = 0: the omitted quadratic is
        # uncorrelated with the included linear term, and the slope is NOT biased.
        X = r.standard_normal((n, 1))
        return X, 1.0 + true_slope * X[:, 0] + 1.5 * X[:, 0] ** 2 + r.standard_normal(n)
    run("nonlinearity, x symmetric", nonlinear_symmetric)

    def nonlinear_asymmetric(r):
        # x is shifted away from 0, so now Cov(x, x^2) != 0 and the omitted term loads
        # onto the slope. THIS is omitted-variable bias.
        X = r.uniform(0.0, 3.0, (n, 1))
        return X, 1.0 + true_slope * X[:, 0] + 1.5 * X[:, 0] ** 2 + r.standard_normal(n)
    run("NONLINEARITY, x asymmetric", nonlinear_asymmetric)

    print(f"  {'violation':<36s}  {'coef bias':>11s}  {'coef sd':>9s}  "
          f"{'95% CI coverage':>17s}")
    print("  " + "-" * 80)
    for name, (bias, sd, coverage) in scenarios.items():
        flag = ""
        if abs(bias) > 0.10:
            flag = "  <- BIASED"
        elif abs(coverage - 0.95) > 0.03:
            flag = "  <- SEs wrong"
        print(f"  {name:<36s}  {bias:11.4f}  {sd:9.4f}  {coverage:16.1%}{flag}")

    print("""
  The table separates two failure modes that usually get lumped together, and adds a third
  distinction that most treatments omit.

  HETEROSCEDASTICITY and AUTOCORRELATION leave the coefficient UNBIASED — the point
  estimate is still right on average. What breaks is the confidence interval. Fix the
  STANDARD ERRORS (robust HC3, Newey-West, clustered), not the model.

    A detail worth knowing: autocorrelated ERRORS alone are nearly harmless if the
    regressor is i.i.d. It is serial correlation in x AND in the noise together that
    shrinks the effective sample size and wrecks coverage — which is precisely the
    situation in every time series, and why you cannot treat time-series rows as
    independent observations (15.01).

  HEAVY TAILS barely hurt at n = 100 — the CLT is doing its job (00.03 §12). Coverage
  holds; the cost is efficiency (a wider spread), not validity.

  NONLINEARITY is the one that can be genuinely fatal — but note the two rows, because the
  distinction is sharp and is usually stated wrongly. Omitting the x^2 term biases the
  slope ONLY IF the omitted term is correlated with the included one:

    - x symmetric about 0: Cov(x, x^2) = E[x^3] = 0, so the slope comes out UNBIASED
      despite the model being badly misspecified. The SEs are inflated and R^2 is poor,
      but the coefficient is fine.
    - x asymmetric: Cov(x, x^2) != 0, the omitted term loads onto the slope, and the
      estimate is badly biased.

  That is omitted-variable bias in general form: bias = (effect of the omitted variable) x
  (its regression on the included ones). It is why "I left out a variable" is only a
  problem when the missing variable correlates with the ones you kept — and why a
  residuals-vs-fitted plot, which detects misspecification whether or not it biases
  anything, is the first plot you look at (README §10).""")


def experiment_leverage_influence() -> None:
    """README §10: leverage is not influence."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 5 — leverage vs influence  (README §10)")
    print("=" * 86)
    print("""
High leverage means "unusual in X" and says nothing about y. A high-leverage point that
happens to sit on the line changes nothing. Cook's distance is what separates the two.
Same clean dataset, three different points added:
""")
    rng = np.random.default_rng(5)
    n = 40
    x = rng.uniform(-2, 2, n)
    y = 1.0 + 2.0 * x + rng.standard_normal(n) * 0.4

    cases = [
        ("no extra point", None, None),
        ("high leverage, ON the line", 10.0, 21.0),      # x far out, y consistent
        ("high leverage, OFF the line", 10.0, 0.0),      # x far out, y inconsistent
        ("low leverage, big residual", 0.0, 9.0),        # x typical, y wild
    ]

    print(f"  {'case':<30s}  {'slope':>8s}  {'max h_ii':>9s}  {'max Cook D':>11s}  "
          f"{'verdict':>16s}")
    print("  " + "-" * 82)

    for name, x_new, y_new in cases:
        if x_new is None:
            xa, ya = x, y
        else:
            xa = np.append(x, x_new)
            ya = np.append(y, y_new)
        m = LinearRegression(solver="qr").fit(xa[:, None], ya)
        h = m.leverage()
        cook = m.cooks_distance()
        if x_new is None:
            verdict = "baseline"
        elif cook.max() > 1.0:
            verdict = "INFLUENTIAL"
        elif h.max() > 4 * m.d_ / m.n_:
            verdict = "high h, harmless"
        else:
            verdict = "not influential"
        print(f"  {name:<30s}  {m.coef_[0]:8.4f}  {h.max():9.4f}  {cook.max():11.4f}  "
              f"{verdict:>16s}")

    print("""
  Rows 2 and 3 have nearly IDENTICAL leverage — both points sit at x = 10, far from the
  rest of the data. But their effect on the fit could not be more different.

  Row 2 lies on the true line, so the slope barely moves and Cook's distance stays small.
  Row 3 is off the line, and it drags the slope away from 2.0 with a large Cook's D.

  Row 4 has an enormous residual but ORDINARY leverage — it sits in the middle of the x
  range where it has little leverage to exert, so it moves the fit far less than row 3
  does despite looking worse on a residual plot.

  Two practical consequences. First, leverage and influence are different quantities, and
  only Cook's distance answers "would deleting this change my conclusions?". Second, raw
  residual plots systematically UNDER-show high-leverage points, because Var(r_i) =
  sigma^2(1 - h_ii) shrinks exactly where leverage is high — which is why you should plot
  STUDENTIZED residuals.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    rng = np.random.default_rng(42)
    X_demo = rng.standard_normal((120, 3))
    y_demo = 5.0 + X_demo @ np.array([1.5, -2.0, 0.0]) + rng.standard_normal(120)
    print("\n" + LinearRegression().fit(X_demo, y_demo).summary())

    experiment_gauss_markov()
    experiment_solvers()
    experiment_r_squared()
    experiment_assumption_violations()
    experiment_leverage_influence()

    print("\n" + "=" * 86)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 86)
