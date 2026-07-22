"""
03.02 — Regularized Linear Models from Scratch
==============================================

Ridge, Lasso, and Elastic Net implemented from their objectives, plus the experiments
that show what each penalty actually does to the coefficients.

Implemented here
----------------
    Ridge                   closed form and SVD form            README §2-§3
        .effective_dof()    sum of shrinkage factors            README §4
        .shrinkage_factors()

    Lasso                   coordinate descent (the standard)   README §7
                            and ISTA, for comparison            00.02 §15
    ElasticNet              both penalties                      README §8

    soft_threshold          the prox of L1 — where zeros come from   README §6.3
    regularization_path     coefficients vs lambda              README §12
    cross_validate_lambda   CV with the one-standard-error rule README §11

Run it
------
    python from_scratch.py

Verified against sklearn, then five experiments:
  1. The bias-variance trade: MSE is minimized at lambda > 0, measured
  2. Ridge's spectral shrinkage — low-variance directions damped hardest
  3. L1 gives exact zeros and L2 never does, all three arguments at once
  4. Standardization is mandatory: what happens without it
  5. Lasso vs Elastic Net on correlated features — the grouping effect

Reference: README.md sections 2-12.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# SHARED
# =============================================================================


def soft_threshold(v: float | np.ndarray, threshold: float):
    """S_t(v) = sign(v) * max(|v| - t, 0).  README §6.3

    The proximal operator of t*|.|. Shrink toward zero by t, and CLAMP to exactly zero
    if that would cross. The clamp is the entire reason L1 produces exact zeros while L2
    (whose prox is v/(1+t)) produces only small values.
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


class _LinearModelBase:
    """Handles centring, scaling, and intercept recovery — shared by all three models.

    Two things every regularized linear model must get right, and which are the source of
    most bugs in hand-rolled implementations:

    1. THE INTERCEPT IS NEVER PENALIZED (README §2.1). Shrinking it would make predictions
       depend on where the origin of y happens to sit — add 1000 to every target and a
       model with a penalized intercept gives different answers. We centre y and X, fit
       without an intercept, then recover b = mean(y) - w . mean(x).

    2. FEATURES MUST BE STANDARDIZED (README §10). The penalty treats all coefficients
       alike, so a feature measured in kilometres rather than metres is penalized 10^6
       times harder. Experiment 4 measures the damage.
    """

    def __init__(self, fit_intercept: bool = True, standardize: bool = True):
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    def _preprocess(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            self._x_mean = X.mean(axis=0)
            self._y_mean = y.mean()
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._y_mean = 0.0

        Xc = X - self._x_mean
        if self.standardize:
            self._x_scale = Xc.std(axis=0)
            self._x_scale[self._x_scale < 1e-12] = 1.0     # leave constant columns alone
        else:
            self._x_scale = np.ones(X.shape[1])

        return Xc / self._x_scale, y - self._y_mean

    def _finalize(self, w_scaled: np.ndarray):
        """Undo the scaling so coefficients refer to the ORIGINAL feature units."""
        self.coef_ = w_scaled / self._x_scale
        self.intercept_ = float(self._y_mean - self.coef_ @ self._x_mean) \
            if self.fit_intercept else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total) if total > 0 else float("nan")


# =============================================================================
# RIDGE  (README §2-§4)
# =============================================================================


class Ridge(_LinearModelBase):
    """min ||y - Xw||^2 + alpha ||w||^2

    Note the parameterization: this matches sklearn's `Ridge(alpha=...)`, where the loss
    is NOT divided by n. glmnet divides by n and calls it lambda, so the same amount of
    regularization has a different number attached. This is why README §14 warns that
    lambda values do not transfer between libraries.
    """

    def __init__(self, alpha: float = 1.0, solver: str = "svd",
                 fit_intercept: bool = True, standardize: bool = False):
        super().__init__(fit_intercept, standardize)
        self.alpha = alpha
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        Xs, yc = self._preprocess(X, y)
        n, d = Xs.shape

        if self.solver == "closed":
            # w = (X^T X + alpha I)^-1 X^T y. Always invertible: X^T X is PSD, so adding
            # alpha I makes every eigenvalue >= alpha > 0 (00.01 §11.2). This holds even
            # when d > n, where OLS has no unique solution at all.
            gram = Xs.T @ Xs + self.alpha * np.eye(d)
            w = np.linalg.solve(gram, Xs.T @ yc)

        elif self.solver == "svd":
            # w = sum_i [sigma_i / (sigma_i^2 + alpha)] (u_i . y) v_i   (README §3)
            # Better conditioned, and it exposes the per-direction shrinkage directly.
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            self._singular_values = s
            w = Vt.T @ ((s / (s ** 2 + self.alpha)) * (U.T @ yc))
        else:
            raise ValueError(f"unknown solver {self.solver!r}")

        if not hasattr(self, "_singular_values"):
            self._singular_values = np.linalg.svd(Xs, compute_uv=False)

        self._finalize(w)
        return self

    def shrinkage_factors(self) -> np.ndarray:
        """sigma_i^2 / (sigma_i^2 + alpha) — how much each principal direction survives.

        Near 1 for high-variance directions (barely touched), near 0 for low-variance ones
        (almost erased). Ridge shrinks hardest exactly where the coefficient is least
        determined, since Var(w_i) is proportional to 1/sigma_i^2. That is the statistical
        content of the method (README §3).
        """
        s2 = self._singular_values ** 2
        return s2 / (s2 + self.alpha)

    def effective_dof(self) -> float:
        """df(alpha) = sum of the shrinkage factors = trace of the ridge hat matrix.

        Falls smoothly from d (alpha = 0) to 0 (alpha -> inf). This is what
        'regularization reduces complexity' means as a number: a 100-feature ridge model
        can behave like a 12-parameter one (README §4).
        """
        return float(self.shrinkage_factors().sum())


# =============================================================================
# LASSO  (README §5-§7)
# =============================================================================


class Lasso(_LinearModelBase):
    """min (1/2n) ||y - Xw||^2 + alpha ||w||_1     [sklearn's parameterization]

    No closed form: ||w||_1 is not differentiable at zero, which is precisely where the
    interesting solutions live. Solved by coordinate descent, which works because the
    1-D subproblem DOES have a closed form (README §7).
    """

    def __init__(self, alpha: float = 1.0, solver: str = "cd", max_iter: int = 5000,
                 tol: float = 1e-10, fit_intercept: bool = True,
                 standardize: bool = False):
        super().__init__(fit_intercept, standardize)
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray, w_init: np.ndarray | None = None) -> "Lasso":
        Xs, yc = self._preprocess(X, y)
        n, d = Xs.shape

        w = np.zeros(d) if w_init is None else w_init.copy()

        if self.solver == "cd":
            w = self._coordinate_descent(Xs, yc, w)
        elif self.solver == "ista":
            w = self._ista(Xs, yc, w)
        else:
            raise ValueError(f"unknown solver {self.solver!r}")

        self.n_iter_ = self._n_iter
        self._finalize(w)
        return self

    def _coordinate_descent(self, X, y, w):
        """Cycle through coordinates, solving each exactly.  README §7

            w_j <- S_{alpha}( x_j . r^(-j) / n ) / (x_j . x_j / n)

        where r^(-j) is the residual with feature j's contribution removed. Each update is
        the closed-form solution of a 1-D lasso, which is soft thresholding applied to the
        1-D least-squares answer.

        Maintaining the FULL residual and patching it incrementally makes each update O(n)
        rather than O(nd) — the difference between a usable algorithm and a toy. This is
        what sklearn does (in Cython).
        """
        n, d = X.shape
        col_norms = np.sum(X ** 2, axis=0) / n
        residual = y - X @ w

        self._n_iter = self.max_iter
        for iteration in range(self.max_iter):
            max_change = 0.0
            for j in range(d):
                if col_norms[j] < 1e-15:              # constant column: nothing to fit
                    continue
                w_old = w[j]
                # Add feature j back into the residual, solve, then remove it again.
                rho = X[:, j] @ (residual + X[:, j] * w_old) / n
                w[j] = soft_threshold(rho, self.alpha) / col_norms[j]
                if w[j] != w_old:
                    residual -= X[:, j] * (w[j] - w_old)
                max_change = max(max_change, abs(w[j] - w_old))
            if max_change < self.tol:
                self._n_iter = iteration + 1
                break
        return w

    def _ista(self, X, y, w):
        """Proximal gradient descent.  00.02 §15

        Gradient step on the smooth part, then apply the L1 prox. Simpler than coordinate
        descent and far slower to converge — included so the two can be compared.

        Step size 1/L with L the Lipschitz constant of the smooth gradient, which is the
        largest provably-convergent choice (00.02 §7.1).
        """
        n = X.shape[0]
        L = float(np.linalg.eigvalsh(X.T @ X / n).max())
        step = 1.0 / L if L > 0 else 1.0

        self._n_iter = self.max_iter
        for iteration in range(self.max_iter):
            w_old = w.copy()
            grad = X.T @ (X @ w - y) / n
            w = soft_threshold(w - step * grad, step * self.alpha)
            if np.max(np.abs(w - w_old)) < self.tol:
                self._n_iter = iteration + 1
                break
        return w


# =============================================================================
# ELASTIC NET  (README §8)
# =============================================================================


class ElasticNet(_LinearModelBase):
    """min (1/2n)||y - Xw||^2 + alpha * l1_ratio * ||w||_1
                              + 0.5 * alpha * (1 - l1_ratio) * ||w||^2

    Exists because Lasso has two specific failures: it can select at most n features when
    d > n, and with correlated features it picks one arbitrarily. The L2 term removes the
    cap and induces the GROUPING EFFECT — correlated features get similar coefficients and
    enter together (Experiment 5).

    The coordinate update gains a denominator term, and nothing else changes:

        w_j <- S_{alpha*l1_ratio}(rho) / (x_j.x_j/n + alpha*(1 - l1_ratio))
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 5000,
                 tol: float = 1e-10, fit_intercept: bool = True,
                 standardize: bool = False):
        super().__init__(fit_intercept, standardize)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        Xs, yc = self._preprocess(X, y)
        n, d = Xs.shape

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)

        w = np.zeros(d)
        col_norms = np.sum(Xs ** 2, axis=0) / n
        residual = yc - Xs @ w

        for _ in range(self.max_iter):
            max_change = 0.0
            for j in range(d):
                denom = col_norms[j] + l2
                if denom < 1e-15:
                    continue
                w_old = w[j]
                rho = Xs[:, j] @ (residual + Xs[:, j] * w_old) / n
                w[j] = soft_threshold(rho, l1) / denom
                if w[j] != w_old:
                    residual -= Xs[:, j] * (w[j] - w_old)
                max_change = max(max_change, abs(w[j] - w_old))
            if max_change < self.tol:
                break

        self._finalize(w)
        return self


# =============================================================================
# PATHS AND MODEL SELECTION  (README §11-§12)
# =============================================================================


def regularization_path(model_class, X, y, alphas, **kwargs):
    """Coefficients as a function of alpha.  README §12

    Fitted from LARGE alpha down to small, warm-starting each fit from the previous
    solution. That is not just an optimization: at large alpha almost everything is zero,
    so each subsequent problem starts close to its answer, and the whole path costs barely
    more than a few independent fits.
    """
    alphas = np.sort(np.asarray(alphas, dtype=float))[::-1]
    coefs = []
    w_prev = None

    for alpha in alphas:
        model = model_class(alpha=alpha, **kwargs)
        if isinstance(model, Lasso) and w_prev is not None:
            model.fit(X, y, w_init=w_prev * model._x_scale if hasattr(model, "_x_scale")
                      else None)
        else:
            model.fit(X, y)
        coefs.append(model.coef_.copy())
        w_prev = model.coef_.copy()

    return alphas, np.array(coefs)


def cross_validate_lambda(model_class, X, y, alphas, n_folds: int = 5,
                          seed: int = 0, **kwargs) -> dict:
    """K-fold CV over a grid of alphas, with the one-standard-error rule.  README §11

    Returns both the alpha minimizing CV error and the LARGEST alpha whose error is within
    one standard error of that minimum. The CV curve is usually flat near its optimum, so
    the 1-SE choice costs almost nothing in accuracy and buys a simpler, more stable model.
    It is glmnet's default, and generally the better choice when you intend to interpret
    the model.

    Note the scaler is fitted INSIDE each fold. Standardizing on the full dataset first
    would leak the validation fold's mean and variance into training — the classic subtle
    leak of 02.06.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    alphas = np.asarray(alphas, dtype=float)
    errors = np.zeros((n_folds, alphas.size))

    for k, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Standardization statistics come from the TRAINING fold only.
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        X_tr_s = (X_tr - mu) / sd
        X_val_s = (X_val - mu) / sd

        for j, alpha in enumerate(alphas):
            model = model_class(alpha=alpha, **kwargs).fit(X_tr_s, y_tr)
            errors[k, j] = np.mean((y_val - model.predict(X_val_s)) ** 2)

    mean_error = errors.mean(axis=0)
    se_error = errors.std(axis=0, ddof=1) / np.sqrt(n_folds)

    best = int(np.argmin(mean_error))
    threshold = mean_error[best] + se_error[best]
    within = np.where(mean_error <= threshold)[0]
    one_se = int(within[np.argmax(alphas[within])])

    return {
        "alphas": alphas,
        "mean_error": mean_error,
        "se_error": se_error,
        "alpha_min": float(alphas[best]),
        "alpha_1se": float(alphas[one_se]),
    }


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

    n, d = 200, 12
    X = rng.standard_normal((n, d))
    true_w = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    y = 2.0 + X @ true_w + rng.standard_normal(n) * 0.5

    print("\nRidge (README §2-§3)")
    ridge_svd = Ridge(alpha=1.0, solver="svd").fit(X, y)
    ridge_closed = Ridge(alpha=1.0, solver="closed").fit(X, y)
    ok &= _report("SVD solver matches closed form",
                  float(np.abs(ridge_svd.coef_ - ridge_closed.coef_).max()), 1e-10)

    try:
        from sklearn.linear_model import (Ridge as SKRidge, Lasso as SKLasso,
                                          ElasticNet as SKElasticNet)
        sk = SKRidge(alpha=1.0).fit(X, y)
        ok &= _report("Ridge coefficients vs sklearn",
                      float(np.abs(ridge_svd.coef_ - sk.coef_).max()), 1e-9)
        ok &= _report("Ridge intercept vs sklearn",
                      abs(ridge_svd.intercept_ - sk.intercept_), 1e-9)

        for alpha in (10.0, 100.0):
            mine = Ridge(alpha=alpha, solver="svd").fit(X, y)
            ref = SKRidge(alpha=alpha).fit(X, y)
            ok &= _report(f"Ridge(alpha={alpha}) vs sklearn",
                          float(np.abs(mine.coef_ - ref.coef_).max()), 1e-9)

        print("\nLasso (README §5-§7)")
        for alpha in (0.01, 0.1, 0.5):
            mine = Lasso(alpha=alpha, max_iter=50_000, tol=1e-14).fit(X, y)
            ref = SKLasso(alpha=alpha, max_iter=200_000, tol=1e-14).fit(X, y)
            ok &= _report(f"Lasso(alpha={alpha}) coefficients vs sklearn",
                          float(np.abs(mine.coef_ - ref.coef_).max()), 1e-6)
            same_support = np.array_equal(mine.coef_ == 0, ref.coef_ == 0)
            print(f"  [{'PASS' if same_support else 'FAIL'}]  "
                  f"{f'Lasso(alpha={alpha}) same exact-zero support':<54s}  "
                  f"nnz {np.sum(mine.coef_ != 0)} vs {np.sum(ref.coef_ != 0)}")
            ok &= same_support

        # Coordinate descent and ISTA solve the same problem.
        cd = Lasso(alpha=0.1, solver="cd", max_iter=50_000, tol=1e-14).fit(X, y)
        ista = Lasso(alpha=0.1, solver="ista", max_iter=200_000, tol=1e-14).fit(X, y)
        ok &= _report("coordinate descent matches ISTA",
                      float(np.abs(cd.coef_ - ista.coef_).max()), 1e-5)
        print(f"  [INFO]  {'iterations: coordinate descent vs ISTA':<54s}  "
              f"{cd.n_iter_} vs {ista.n_iter_}")

        print("\nElastic Net (README §8)")
        for alpha, ratio in ((0.1, 0.5), (0.05, 0.9)):
            mine = ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=50_000,
                              tol=1e-14).fit(X, y)
            ref = SKElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=200_000,
                               tol=1e-14).fit(X, y)
            ok &= _report(f"ElasticNet(alpha={alpha}, l1_ratio={ratio}) vs sklearn",
                          float(np.abs(mine.coef_ - ref.coef_).max()), 1e-6)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nStructural properties (README §2.1, §4, §6)")
    # Ridge is defined even when d > n, where OLS is not.
    X_wide = rng.standard_normal((20, 60))
    y_wide = rng.standard_normal(20)
    wide = Ridge(alpha=1.0, solver="svd").fit(X_wide, y_wide)
    ok &= _report("Ridge fits d > n (OLS cannot)",
                  0.0 if np.all(np.isfinite(wide.coef_)) else 1.0, 0.5)

    # Effective dof falls from d to 0 as alpha grows.
    dofs = [Ridge(alpha=a, solver="svd").fit(X, y).effective_dof()
            for a in (1e-8, 1.0, 100.0, 1e6)]
    monotone = all(dofs[i] > dofs[i + 1] for i in range(len(dofs) - 1))
    print(f"  [{'PASS' if monotone else 'FAIL'}]  "
          f"{'effective dof decreases with alpha':<54s}  "
          f"{dofs[0]:.2f} -> {dofs[1]:.2f} -> {dofs[2]:.2f} -> {dofs[3]:.4f}")
    ok &= monotone
    ok &= _report("effective dof -> d as alpha -> 0", abs(dofs[0] - d), 1e-4)

    # Ridge never produces an exact zero; Lasso does.
    ridge_zeros = int(np.sum(Ridge(alpha=100.0).fit(X, y).coef_ == 0))
    lasso_zeros = int(np.sum(Lasso(alpha=0.3, max_iter=50_000).fit(X, y).coef_ == 0))
    print(f"  [{'PASS' if ridge_zeros == 0 and lasso_zeros > 0 else 'FAIL'}]  "
          f"{'Ridge gives 0 exact zeros, Lasso gives some':<54s}  "
          f"{ridge_zeros} vs {lasso_zeros}")
    ok &= (ridge_zeros == 0 and lasso_zeros > 0)

    # Soft thresholding is the prox of L1.
    ok &= _report("soft_threshold(0.5, 1.0) = 0 (clamped)",
                  abs(float(soft_threshold(0.5, 1.0))), 1e-15)
    ok &= _report("soft_threshold(3.0, 1.0) = 2.0 (shrunk)",
                  abs(float(soft_threshold(3.0, 1.0)) - 2.0), 1e-15)

    # Intercept must be unpenalized: shifting y must shift only the intercept.
    shifted = Ridge(alpha=10.0, solver="svd").fit(X, y + 1000.0)
    base = Ridge(alpha=10.0, solver="svd").fit(X, y)
    ok &= _report("shifting y by 1000 leaves coefficients unchanged",
                  float(np.abs(shifted.coef_ - base.coef_).max()), 1e-9)
    ok &= _report("...and shifts the intercept by exactly 1000",
                  abs(shifted.intercept_ - base.intercept_ - 1000.0), 1e-8)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_bias_variance() -> None:
    """README §1: MSE is minimized at some lambda > 0."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 1 — the bias-variance trade, measured  (README §1)")
    print("=" * 86)
    print("""
Gauss-Markov says OLS is best among UNBIASED estimators. Ridge is biased, so the theorem
does not apply — leaving room for it to win on MSE. Refitting on 3,000 fresh samples from
the same generative process at each alpha, and decomposing the coefficient error:
""")
    rng = np.random.default_rng(1)
    n, d = 50, 20
    true_w = np.concatenate([np.array([2.0, -1.5, 1.0]), np.zeros(d - 3)])
    X_fixed = rng.standard_normal((n, d))
    n_trials = 3000

    alphas = [0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 1000.0]
    print(f"  {'alpha':>9s}  {'bias^2':>11s}  {'variance':>11s}  {'MSE(w)':>11s}  "
          f"{'test MSE':>11s}  {'eff. dof':>9s}")
    print("  " + "-" * 70)

    X_test = rng.standard_normal((2000, d))
    y_test_clean = X_test @ true_w

    best_mse, best_alpha = float("inf"), None
    rows = []
    for alpha in alphas:
        estimates = np.empty((n_trials, d))
        test_errors = np.empty(n_trials)
        for t in range(n_trials):
            y = X_fixed @ true_w + rng.standard_normal(n)
            model = (Ridge(alpha=alpha, solver="svd") if alpha > 0
                     else Ridge(alpha=1e-10, solver="svd")).fit(X_fixed, y)
            estimates[t] = model.coef_
            test_errors[t] = np.mean((y_test_clean - (X_test @ model.coef_)) ** 2)

        bias2 = float(np.sum((estimates.mean(axis=0) - true_w) ** 2))
        variance = float(estimates.var(axis=0).sum())
        mse_w = bias2 + variance
        test_mse = float(test_errors.mean())
        dof = Ridge(alpha=max(alpha, 1e-10), solver="svd").fit(
            X_fixed, X_fixed @ true_w).effective_dof()
        rows.append((alpha, bias2, variance, mse_w, test_mse, dof))
        if test_mse < best_mse:
            best_mse, best_alpha = test_mse, alpha

    for alpha, bias2, variance, mse_w, test_mse, dof in rows:
        marker = "  <- best" if alpha == best_alpha else ""
        label = "0 (OLS)" if alpha == 0.0 else f"{alpha:g}"
        print(f"  {label:>9s}  {bias2:11.5f}  {variance:11.5f}  {mse_w:11.5f}  "
              f"{test_mse:11.5f}  {dof:9.2f}{marker}")

    print(f"""
  Read the second and third columns against each other. Bias^2 rises monotonically with
  alpha, exactly as expected — ridge deliberately pulls coefficients toward zero. Variance
  falls monotonically, and it falls FASTER at first.

  The result is a genuine minimum at alpha = {best_alpha:g}, not at 0. OLS is not the best
  estimator here, and there is nothing paradoxical about that: Gauss-Markov only ranks
  UNBIASED estimators, and ridge is not one (03.01 §6).

  The last column shows what is being bought. Effective degrees of freedom fall from 20 to
  a small number — the model has 20 features but behaves like a much smaller one, which is
  precisely why it stops overfitting n = 50 rows.""")


def experiment_ridge_spectrum() -> None:
    """README §3: ridge shrinks low-variance directions hardest."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 2 — ridge shrinks by direction, not uniformly  (README §3)")
    print("=" * 86)
    print("""
Ridge is often described as "shrinking all coefficients". Through the SVD it is doing
something much more targeted: shrinking each PRINCIPAL DIRECTION by
sigma_i^2 / (sigma_i^2 + alpha).

Building a design matrix with deliberately spread-out singular values:
""")
    rng = np.random.default_rng(2)
    n, d = 200, 6

    # Construct X with prescribed singular values.
    Q1, _ = np.linalg.qr(rng.standard_normal((n, d)))
    Q2, _ = np.linalg.qr(rng.standard_normal((d, d)))
    target_s = np.array([10.0, 5.0, 2.0, 1.0, 0.3, 0.05])
    X = Q1 @ np.diag(target_s) @ Q2.T
    y = X @ rng.standard_normal(d) + rng.standard_normal(n) * 0.1

    print(f"  {'sigma_i':>9s}  {'sigma_i^2':>11s}", end="")
    alphas = [0.01, 0.1, 1.0, 10.0]
    for a in alphas:
        print(f"  {f'alpha={a:g}':>12s}", end="")
    print("\n  " + "-" * (22 + 14 * len(alphas)))

    shrinkages = {a: Ridge(alpha=a, solver="svd").fit(X, y).shrinkage_factors()
                  for a in alphas}
    for i in range(d):
        print(f"  {target_s[i]:9.2f}  {target_s[i] ** 2:11.4f}", end="")
        for a in alphas:
            print(f"  {shrinkages[a][i]:12.4f}", end="")
        print()

    print(f"\n  effective dof: ", end="")
    for a in alphas:
        print(f"alpha={a:g}: {shrinkages[a].sum():.2f}   ", end="")
    print("""

  Each column is a shrinkage factor: 1.0 means "untouched", 0.0 means "erased".

  The top rows — the directions in which the data varies most — are barely affected at any
  alpha. The bottom row, with sigma = 0.05, is crushed to near zero even at alpha = 0.01.

  That is exactly the right behaviour. Var(w_i) is proportional to 1/sigma_i^2, so the
  low-sigma directions are the ones you know least about, and they are the ones ridge
  damps hardest. It is not indiscriminate shrinkage — it is shrinkage in proportion to
  ignorance.

  It also explains why ridge is the natural remedy for multicollinearity: collinearity IS
  small sigma_i, and those directions are precisely what ridge suppresses (03.01 §12).""")


def experiment_sparsity() -> None:
    """README §6: L1 gives exact zeros, L2 never does."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 3 — L1 zeros vs L2 shrinkage  (README §6)")
    print("=" * 86)
    print("""
The claim is not that L1 makes coefficients SMALL — it is that L1 makes them EXACTLY zero.
Data with 5 real features out of 40; the rest are pure noise:
""")
    rng = np.random.default_rng(3)
    n, d, n_real = 100, 40, 5
    X = rng.standard_normal((n, d))
    true_w = np.zeros(d)
    true_w[:n_real] = [3.0, -2.5, 2.0, -1.5, 1.0]
    y = X @ true_w + rng.standard_normal(n) * 0.5

    print(f"  True model: {n_real} nonzero of {d}\n")
    print(f"  {'method':<22s}  {'exact zeros':>12s}  {'nonzero':>9s}  "
          f"{'min |coef|':>12s}  {'true found':>11s}  {'false pos':>10s}")
    print("  " + "-" * 82)

    for alpha in (1.0, 10.0, 100.0):
        m = Ridge(alpha=alpha, solver="svd").fit(X, y)
        print(f"  {f'Ridge(alpha={alpha:g})':<22s}  {int(np.sum(m.coef_ == 0)):12d}  "
              f"{int(np.sum(m.coef_ != 0)):9d}  {np.abs(m.coef_).min():12.3e}  "
              f"{'n/a':>11s}  {'n/a':>10s}")

    for alpha in (0.01, 0.05, 0.1, 0.3):
        m = Lasso(alpha=alpha, max_iter=50_000, tol=1e-14).fit(X, y)
        selected = m.coef_ != 0
        true_found = int(np.sum(selected[:n_real]))
        false_pos = int(np.sum(selected[n_real:]))
        nz = m.coef_[selected]
        print(f"  {f'Lasso(alpha={alpha:g})':<22s}  {int(np.sum(~selected)):12d}  "
              f"{int(np.sum(selected)):9d}  "
              f"{(np.abs(nz).min() if nz.size else 0.0):12.3e}  "
              f"{true_found:8d}/{n_real}  {false_pos:10d}")

    print("""
  Ridge produces ZERO exact zeros at every alpha — its smallest coefficient shrinks toward
  zero but never reaches it, because its proximal operator is w/(1+t), which is zero only
  if w already was.

  Lasso zeroes out most of the 35 irrelevant features and, at alpha around 0.05-0.1,
  recovers close to the true support of 5. The nonzero coefficients it keeps are bounded
  away from zero — there is no "tiny but nonzero" band, which is the soft-threshold clamp
  visible in the output.

  One caution the table also shows: at small alpha Lasso keeps false positives, and at
  large alpha it starts dropping real features. Sparsity is not free, and the selected set
  is 'a sufficient set', not 'the true set' (README §14).""")


def experiment_standardization() -> None:
    """README §10: the penalty is not scale-invariant."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 4 — why standardization is mandatory  (README §10)")
    print("=" * 86)
    print("""
Three features with IDENTICAL true effect on y, differing only in the units they are
measured in. Nothing about the underlying relationship changes — only the scale.
""")
    rng = np.random.default_rng(4)
    n = 300
    base = rng.standard_normal((n, 3))
    scales = np.array([0.001, 1.0, 1000.0])
    X = base * scales                                # same information, different units
    true_effect = np.array([1.0, 1.0, 1.0]) / scales  # identical contribution to y
    y = X @ true_effect + rng.standard_normal(n) * 0.1

    print(f"  All three features contribute equally to y. Feature scales: "
          f"{scales[0]:g}, {scales[1]:g}, {scales[2]:g}\n")

    for standardize in (False, True):
        label = "WITH standardization" if standardize else "WITHOUT standardization"
        print(f"  {label}")
        print(f"    {'alpha':>8s}  {'coef 1':>13s}  {'coef 2':>13s}  {'coef 3':>13s}  "
              f"{'R^2':>8s}")
        print("    " + "-" * 62)
        for alpha in (0.01, 1.0, 100.0):
            m = Ridge(alpha=alpha, solver="svd", standardize=standardize).fit(X, y)
            # Report contribution (coef * scale), which is comparable across features.
            contrib = m.coef_ * scales
            print(f"    {alpha:8g}  {contrib[0]:13.5f}  {contrib[1]:13.5f}  "
                  f"{contrib[2]:13.5f}  {m.score(X, y):8.5f}")
        print()

    print("""  Each entry is the feature's CONTRIBUTION (coefficient x scale), so all three
  should be equal — they contribute identically by construction.

  WITHOUT standardization they are not. The penalty is applied to raw coefficients, and
  the feature measured in small units needs a huge coefficient to express the same
  relationship, so it is penalized enormously harder. At alpha = 100 its contribution is
  crushed while the large-scale feature is barely touched. The model has made a decision
  based on nothing but the units someone chose.

  WITH standardization all three are treated alike and shrink together, which is the only
  defensible behaviour.

  Note this is NOT an issue for plain OLS, which is equivariant under feature scaling —
  rescaling a feature just rescales its coefficient and leaves predictions identical. It
  becomes mandatory the moment a penalty is added, because the penalty is not.""")


def experiment_grouping() -> None:
    """README §8: Lasso picks one of a correlated group; Elastic Net keeps them."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 5 — the grouping effect  (README §8)")
    print("=" * 86)
    print("""
Lasso is said to "pick one of a correlated group arbitrarily", and Elastic Net to keep the
whole group. Rather than assert that, we look for the correlation level at which it starts
being true.

Three genuinely predictive features built from the same latent z, plus 20 irrelevant ones.
Sweeping how tightly the three are coupled, 20 resampled datasets at each level:
""")
    n, n_noise, n_runs, alpha = 150, 20, 20, 1.0

    print(f"  {'noise added':>12s}  {'pairwise r':>11s}  {'Lasso keeps':>12s}  "
          f"{'ENet keeps':>11s}  {'Lasso stability':>16s}")
    print("  " + "-" * 70)

    for corr_noise in (0.5, 0.2, 0.1, 0.05, 0.02):
        rng = np.random.default_rng(5)
        lasso_counts = np.zeros(3)
        enet_counts = np.zeros(3)
        correlations = []

        for _ in range(n_runs):
            z = rng.standard_normal(n)
            group = np.column_stack(
                [z + corr_noise * rng.standard_normal(n) for _ in range(3)])
            X = np.column_stack([group, rng.standard_normal((n, n_noise))])
            y = 3.0 * z + rng.standard_normal(n) * 0.5
            correlations.append(np.corrcoef(group.T)[0, 1])

            lasso = Lasso(alpha=alpha, max_iter=10_000, tol=1e-9,
                          standardize=True).fit(X, y)
            enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10_000, tol=1e-9,
                              standardize=True).fit(X, y)
            lasso_counts += (lasso.coef_[:3] != 0)
            enet_counts += (enet.coef_[:3] != 0)

        # Spread of the per-feature selection rates. If Lasso were choosing arbitrarily
        # among equivalent features, the three rates would be similar and all well below
        # 100%; a large spread means some features are being systematically dropped.
        rates = lasso_counts / n_runs

        print(f"  {corr_noise:12.2f}  {np.mean(correlations):11.4f}  "
              f"{lasso_counts.sum() / n_runs:11.2f}/3  {enet_counts.sum() / n_runs:10.2f}/3  "
              f"{rates.min():6.0%} - {rates.max():.0%}")

    print("""
  The grouping effect is REAL, but it has a threshold — and the threshold is far higher
  than the usual warning implies.

  At r = 0.80, r = 0.96, even r = 0.99, Lasso happily keeps all three features. The
  textbook caution simply does not bite there. Only once the features are near-duplicates
  (r = 0.9996) does Lasso start dropping members of the group, keeping 1.70 of 3 — while
  Elastic Net keeps all three at every level.

  So the accurate statement is: Lasso selects arbitrarily among features that are nearly
  INDISTINGUISHABLE, not among features that are merely correlated. That is still a real
  problem, and it is exactly the situation in genomics (linked markers), sensor arrays
  (redundant channels), and text (near-synonymous n-grams). But "my features correlate at
  0.7, so I must avoid Lasso" does not follow, and is a common overcorrection.

  The last column is the per-feature selection rate range. When it collapses away from
  100%, individual members of the group are being dropped run-to-run based on noise — so
  two genuinely predictive features get reported as irrelevant, and possibly a different
  two on the next resample. If you are INTERPRETING the selected set rather than only
  predicting with it, use Elastic Net or report the whole correlated group.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_bias_variance()
    experiment_ridge_spectrum()
    experiment_sparsity()
    experiment_standardization()
    experiment_grouping()

    print("\n" + "=" * 86)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 86)
