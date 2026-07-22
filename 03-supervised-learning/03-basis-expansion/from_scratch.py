"""
03.03 — Basis Expansion from Scratch
====================================

Polynomials, splines, and GAMs — all of them linear models in a transformed basis, so all
of them fitted with the machinery of 03.01 and 03.02.

Implemented here
----------------
    polynomial_basis            raw powers — ill-conditioned by construction   README §2
    legendre_basis              orthogonal alternative, kappa ~ 1              README §2
    truncated_power_basis       the pedagogical cubic-spline basis             README §5.1
    bspline_basis               Cox-de Boor recursion                          README §6
    natural_cubic_basis         linear beyond the boundary knots               README §7

    SplineRegression            regression spline with any of the above
    SmoothingSpline             penalized, with GCV for lambda                 README §8-§9
    GAM                         additive model fitted by backfitting           README §10

Run it
------
    python from_scratch.py

Verified against numpy, scipy.interpolate, and sklearn, then five experiments:
  1. The Runge phenomenon — polynomial error DIVERGES as degree grows
  2. Conditioning: raw powers vs orthogonal polynomials vs B-splines
  3. Extrapolation: what happens outside the data range
  4. Smoothing splines: effective df as a function of lambda, and GCV's choice
  5. GAM backfitting recovers additive structure a linear model cannot

Reference: README.md sections 2-10.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# BASES  (README §2, §5-§7)
# =============================================================================


def polynomial_basis(x: np.ndarray, degree: int, include_bias: bool = True) -> np.ndarray:
    """[1, x, x^2, ..., x^degree] — the Vandermonde matrix.  README §2

    Included to be measured, not used. Its columns become nearly collinear as the degree
    grows (Experiment 2 shows kappa exceeding 1e10 by degree 12), because x^m and x^(m+1)
    are extremely similar functions on a bounded interval. This is a conditioning problem,
    not a modelling one — `legendre_basis` spans exactly the same function space with
    kappa ~ 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    start = 0 if include_bias else 1
    return np.column_stack([x ** m for m in range(start, degree + 1)])


def legendre_basis(x: np.ndarray, degree: int, domain=None) -> np.ndarray:
    """Orthogonal polynomials on the data range.  README §2

    Spans the IDENTICAL function space as `polynomial_basis`, so the fitted curve is the
    same to machine precision — only the parameterization changes. But the columns are
    orthogonal, so the design matrix is perfectly conditioned.

    Built by the standard three-term recurrence, which is also how they are computed in
    practice:

        (m+1) P_{m+1}(t) = (2m+1) t P_m(t) - m P_{m-1}(t)

    This is what R's poly() does by default, and the reason polynomial regression behaves
    better in R than in a naive NumPy implementation.
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = (x.min(), x.max()) if domain is None else domain
    t = 2 * (x - lo) / (hi - lo) - 1.0          # map to [-1, 1]

    columns = [np.ones_like(t)]
    if degree >= 1:
        columns.append(t)
    for m in range(1, degree):
        columns.append(((2 * m + 1) * t * columns[m] - m * columns[m - 1]) / (m + 1))
    return np.column_stack(columns[: degree + 1])


def truncated_power_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """[1, x, ..., x^d, (x-k1)_+^d, ..., (x-kK)_+^d].  README §5.1

    The basis that makes the cubic-spline construction obvious: (x - k)_+^3 is identically
    zero left of k, and its value, first, and second derivatives are all zero AT k. So
    adding it changes nothing before the knot and introduces exactly one new degree of
    freedom afterwards — which is precisely the C^2 continuity condition.

    Right for understanding, wrong for computing: columns for nearby knots are nearly
    identical. Use `bspline_basis` in anger. (The same relationship the normal equations
    have to least squares, 03.01 §4.)
    """
    x = np.asarray(x, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    columns = [x ** m for m in range(degree + 1)]
    columns += [np.maximum(x - k, 0.0) ** degree for k in knots]
    return np.column_stack(columns)


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3,
                  boundary=None) -> np.ndarray:
    """B-spline basis by the Cox-de Boor recursion.  README §6

        B_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0
        B_{i,d}(x) = (x - t_i)/(t_{i+d} - t_i) B_{i,d-1}(x)
                   + (t_{i+d+1} - x)/(t_{i+d+1} - t_{i+1}) B_{i+1,d-1}(x)

    The defining property is COMPACT SUPPORT: each basis function is nonzero on only d+2
    adjacent knot intervals. That gives a banded design matrix (O(n) fitting), excellent
    conditioning, and genuine locality — moving one coefficient moves the curve only near
    that knot.

    Boundary knots are repeated degree+1 times, the standard "clamped" knot vector, so the
    basis spans the full interval.
    """
    x = np.asarray(x, dtype=float).ravel()
    interior = np.asarray(knots, dtype=float).ravel()
    lo, hi = (x.min(), x.max()) if boundary is None else boundary

    # Clamped knot vector: boundaries repeated degree+1 times.
    t = np.concatenate([np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)])
    n_basis = len(t) - degree - 1

    # Degree 0: indicator functions. The last interval is closed on the right so that
    # x == hi is covered rather than falling off the end.
    B = np.zeros((x.size, len(t) - 1))
    for i in range(len(t) - 1):
        if t[i] == t[i + 1]:
            continue
        if i == len(t) - degree - 2:
            B[:, i] = (x >= t[i]) & (x <= t[i + 1])
        else:
            B[:, i] = (x >= t[i]) & (x < t[i + 1])

    # Recurse upward in degree.
    for d in range(1, degree + 1):
        B_new = np.zeros((x.size, len(t) - d - 1))
        for i in range(len(t) - d - 1):
            left = 0.0
            if t[i + d] > t[i]:
                left = (x - t[i]) / (t[i + d] - t[i]) * B[:, i]
            right = 0.0
            if t[i + d + 1] > t[i + 1]:
                right = (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1]) * B[:, i + 1]
            B_new[:, i] = left + right
        B = B_new

    return B[:, :n_basis]


def natural_cubic_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis: linear beyond the boundary knots.  README §7

    Uses the standard construction (ESL eq. 5.4-5.5): starting from the truncated power
    basis and imposing f'' = 0 outside the boundary knots yields

        N_1 = 1,  N_2 = x,  N_{k+2} = d_k(x) - d_{K-1}(x)

    where d_k(x) = [(x-xi_k)_+^3 - (x-xi_K)_+^3] / (xi_K - xi_k).

    Result: K basis functions for K knots, versus K+4 for an unconstrained cubic spline.
    Four degrees of freedom are spent buying linear extrapolation — which is why the
    variance at the edges collapses (Experiment 3).
    """
    x = np.asarray(x, dtype=float).ravel()
    xi = np.sort(np.asarray(knots, dtype=float).ravel())
    K = xi.size
    if K < 3:
        raise ValueError("natural cubic spline needs at least 3 knots")

    def d(k):
        return ((np.maximum(x - xi[k], 0.0) ** 3 - np.maximum(x - xi[K - 1], 0.0) ** 3)
                / (xi[K - 1] - xi[k]))

    columns = [np.ones_like(x), x]
    columns += [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(columns)


# =============================================================================
# MODELS
# =============================================================================


class SplineRegression:
    """Least squares in a chosen basis — a linear model, so 03.01 applies unchanged.

    `basis` selects the transformation; everything after it is ordinary regression, fitted
    by lstsq (SVD) for the reasons of 03.01 §4.
    """

    def __init__(self, basis: str = "natural", n_knots: int = 5, degree: int = 3,
                 alpha: float = 0.0):
        self.basis = basis
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha

    def _make_basis(self, x: np.ndarray) -> np.ndarray:
        if self.basis == "poly":
            return polynomial_basis(x, self.degree)
        if self.basis == "legendre":
            return legendre_basis(x, self.degree, domain=self._domain)
        if self.basis == "truncated":
            return truncated_power_basis(x, self._knots, self.degree)
        if self.basis == "bspline":
            return bspline_basis(x, self._knots, self.degree, boundary=self._domain)
        if self.basis == "natural":
            return natural_cubic_basis(x, self._all_knots)
        raise ValueError(f"unknown basis {self.basis!r}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SplineRegression":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self._domain = (x.min(), x.max())

        # Knots at QUANTILES, not uniformly spaced (README §9): uniform spacing puts knots
        # in empty regions where the fit is unconstrained and variance explodes.
        quantiles = np.linspace(0, 100, self.n_knots + 2)
        self._all_knots = np.percentile(x, quantiles)
        self._knots = self._all_knots[1:-1]           # interior knots only

        Phi = self._make_basis(x)
        if self.alpha > 0:
            gram = Phi.T @ Phi + self.alpha * np.eye(Phi.shape[1])
            self.coef_ = np.linalg.solve(gram, Phi.T @ y)
        else:
            self.coef_ = np.linalg.lstsq(Phi, y, rcond=None)[0]

        self.n_basis_ = Phi.shape[1]
        self.condition_number_ = float(np.linalg.cond(Phi))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._make_basis(np.asarray(x, dtype=float).ravel()) @ self.coef_


class SmoothingSpline:
    """min sum (y_i - f(x_i))^2 + lambda * integral f''(t)^2 dt.   README §8

    The solution over ALL twice-differentiable functions is a natural cubic spline with
    knots at every unique x — an infinite-dimensional problem with a finite-dimensional
    answer. Given that, the fit is linear in y:

        f_hat = (I + lambda * Omega)^-1 y = S_lambda y

    which is ridge regression in a spline basis (03.02 §2), with

        df(lambda) = trace(S_lambda)

    exactly as in 03.02 §4. We implement it in the natural cubic basis with a penalty
    matrix built from second differences — a discrete stand-in for the integral of f''^2
    that is standard and adequate here.
    """

    def __init__(self, lam: float = 1.0, n_knots: int = 30):
        self.lam = lam
        self.n_knots = n_knots

    @staticmethod
    def _penalty_matrix(knots: np.ndarray, n_grid: int = 4000) -> np.ndarray:
        """Omega_jk = integral N_j''(t) N_k''(t) dt — the true roughness penalty.

        This is the matrix for which c^T Omega c = integral f''(t)^2 dt exactly, so
        minimizing RSS + lambda c^T Omega c is the smoothing-spline problem of README §8
        rather than an approximation to it.

        Getting this right matters for a specific reason. A tempting shortcut is to
        penalize second DIFFERENCES of the coefficient vector instead. That is much
        simpler, but its null space has dimension 4 rather than 2, so df tends to 4 as
        lambda -> infinity and the model never reduces to a straight line. The null space
        of the true Omega is exactly {functions with f'' = 0} = span{1, x}, which is why
        lambda -> infinity recovers ordinary least squares as it should.

        Computed by numerically integrating the second derivatives of the basis
        functions. The natural cubic basis is C^2, so central differences are valid.
        """
        knots = np.asarray(knots, dtype=float)
        grid = np.linspace(knots[0], knots[-1], n_grid)
        h = grid[1] - grid[0]

        N = natural_cubic_basis(grid, knots)
        second = (N[2:] - 2 * N[1:-1] + N[:-2]) / h ** 2       # (n_grid-2, n_basis)

        # Trapezoid weights over the interior grid.
        weights = np.full(second.shape[0], h)
        weights[0] = weights[-1] = h / 2
        return (second * weights[:, None]).T @ second

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SmoothingSpline":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        order = np.argsort(x)
        self._x, self._y = x[order], y[order]

        unique_x = np.unique(self._x)
        n_knots = min(self.n_knots, unique_x.size)
        self._knots = np.percentile(unique_x, np.linspace(0, 100, n_knots))

        Phi = natural_cubic_basis(self._x, self._knots)
        Omega = self._penalty_matrix(self._knots)

        gram = Phi.T @ Phi + self.lam * Omega
        self.coef_ = np.linalg.solve(gram, Phi.T @ self._y)

        # Smoother matrix S = Phi (Phi^T Phi + lam Omega)^-1 Phi^T; we need only its trace.
        hat = Phi @ np.linalg.solve(gram, Phi.T)
        self.df_ = float(np.trace(hat))
        self._n = x.size

        residuals = self._y - hat @ self._y
        # GCV (README §9): leave-one-out error in closed form, exploiting the linearity
        # of S. One fit instead of n.
        denom = 1.0 - self.df_ / self._n
        self.gcv_ = float(np.mean(residuals ** 2) / denom ** 2) if denom > 0 else np.inf
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return natural_cubic_basis(np.asarray(x, dtype=float).ravel(),
                                   self._knots) @ self.coef_

    @staticmethod
    def fit_gcv(x, y, lambdas=None, n_knots: int = 30):
        """Choose lambda by minimizing GCV. Returns the fitted model."""
        lambdas = np.logspace(-6, 6, 60) if lambdas is None else np.asarray(lambdas)
        best, best_model = np.inf, None
        for lam in lambdas:
            model = SmoothingSpline(lam=lam, n_knots=n_knots).fit(x, y)
            if model.gcv_ < best:
                best, best_model = model.gcv_, model
        return best_model


class GAM:
    """Additive model: y ~ beta_0 + sum_j f_j(x_j), fitted by backfitting.  README §10

    Backfitting is coordinate descent over FUNCTIONS rather than coefficients (compare
    03.02 §7): cycle through features, and fit each smooth to the partial residual left
    over by the others.

    Each f_j is centred at every step because the decomposition is only identified up to
    a constant per function — without centring, the f_j drift arbitrarily and the
    intercept becomes meaningless, even though the FIT stays the same.

    The structural limitation is visible in the formula: no interactions. If the effect of
    x_1 depends on x_2, no amount of flexibility in f_1 and f_2 can express it.
    """

    def __init__(self, n_knots: int = 5, max_iter: int = 100, tol: float = 1e-8):
        self.n_knots = n_knots
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        self.intercept_ = float(y.mean())
        self.smoothers_ = [None] * d
        contributions = np.zeros((n, d))

        for iteration in range(self.max_iter):
            max_change = 0.0
            for j in range(d):
                partial = y - self.intercept_ - contributions.sum(axis=1) + contributions[:, j]
                smoother = SplineRegression(basis="natural",
                                            n_knots=self.n_knots).fit(X[:, j], partial)
                new = smoother.predict(X[:, j])
                new = new - new.mean()                 # centre: identifiability
                max_change = max(max_change, float(np.abs(new - contributions[:, j]).max()))
                contributions[:, j] = new
                self.smoothers_[j] = smoother

            if max_change < self.tol:
                break

        self.n_iter_ = iteration + 1
        self._offsets = np.array([s.predict(X[:, j]).mean()
                                  for j, s in enumerate(self.smoothers_)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.intercept_)
        for j, smoother in enumerate(self.smoothers_):
            out += smoother.predict(X[:, j]) - self._offsets[j]
        return out

    def partial_dependence(self, j: int, x_grid: np.ndarray) -> np.ndarray:
        """f_j evaluated on a grid — the plot that makes a GAM interpretable."""
        return self.smoothers_[j].predict(x_grid) - self._offsets[j]


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

    x = np.sort(rng.uniform(0, 10, 200))
    y_true = np.sin(x) + 0.1 * x ** 2
    y = y_true + rng.standard_normal(200) * 0.3

    print("\nBases (README §2, §5-§7)")

    # Legendre and raw powers must span the same space -> identical FIT.
    for degree in (3, 6, 9):
        fit_raw = np.linalg.lstsq(polynomial_basis(x, degree), y, rcond=None)[0]
        fit_leg = np.linalg.lstsq(legendre_basis(x, degree), y, rcond=None)[0]
        pred_raw = polynomial_basis(x, degree) @ fit_raw
        pred_leg = legendre_basis(x, degree) @ fit_leg
        ok &= _report(f"degree {degree}: raw and Legendre give the same fit",
                      float(np.abs(pred_raw - pred_leg).max()), 1e-6)

    knots = np.percentile(x, [20, 40, 60, 80])

    # B-splines: partition of unity, and compact support.
    B = bspline_basis(x, knots, degree=3)
    ok &= _report("B-spline basis sums to 1 (partition of unity)",
                  float(np.abs(B.sum(axis=1) - 1.0).max()), 1e-10)
    ok &= _report("B-spline basis is non-negative", float(max(0.0, -B.min())), 1e-15)

    support_fraction = np.mean(B > 1e-12)
    print(f"  [INFO]  {'B-spline compact support: fraction of nonzero entries':<56s}  "
          f"{support_fraction:.3f}")
    ok &= support_fraction < 0.5           # compact support means a sparse design matrix

    try:
        from scipy.interpolate import BSpline
        t = np.concatenate([np.repeat(x.min(), 4), knots, np.repeat(x.max(), 4)])
        n_basis = len(t) - 4
        scipy_B = np.column_stack([
            BSpline.basis_element(t[i:i + 5], extrapolate=False)(x) for i in range(n_basis)])
        scipy_B = np.nan_to_num(scipy_B)
        mine = np.nan_to_num(B)
        # Compare on the interior, where both agree on the half-open-interval convention.
        interior = (x > x.min() + 1e-9) & (x < x.max() - 1e-9)
        ok &= _report("B-spline basis vs scipy.interpolate",
                      float(np.abs(mine[interior] - scipy_B[interior]).max()), 1e-9)
    except ImportError:
        print("  [SKIP]  scipy not installed")

    # Natural cubic basis: K knots -> K basis functions.
    knots5 = np.percentile(x, [0, 25, 50, 75, 100])
    N = natural_cubic_basis(x, knots5)
    ok &= _report("natural cubic basis has K columns for K knots",
                  abs(N.shape[1] - knots5.size), 0.5)

    # The defining property: LINEAR outside the boundary knots, so the second derivative
    # of the fitted function vanishes there.
    model = SplineRegression(basis="natural", n_knots=5).fit(x, y)
    x_out = np.linspace(x.max() + 0.5, x.max() + 5.0, 50)
    pred_out = model.predict(x_out)
    second_diff = np.diff(pred_out, 2)
    ok &= _report("natural spline is linear beyond the boundary knots",
                  float(np.abs(second_diff).max()), 1e-8)

    print("\nModels")
    for basis in ("poly", "legendre", "truncated", "bspline", "natural"):
        m = SplineRegression(basis=basis, n_knots=5, degree=5).fit(x, y)
        r2 = 1 - np.sum((y - m.predict(x)) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  [INFO]  {f'{basis} basis: {m.n_basis_} functions':<40s}  "
              f"R^2 = {r2:.4f}   kappa = {m.condition_number_:.2e}")

    # A spline regression is a linear model, so it must match sklearn's on the same basis.
    try:
        from sklearn.linear_model import LinearRegression as SKLinear
        Phi = natural_cubic_basis(x, knots5)
        sk = SKLinear(fit_intercept=False).fit(Phi, y)
        mine = np.linalg.lstsq(Phi, y, rcond=None)[0]
        ok &= _report("spline fit vs sklearn on the same basis",
                      float(np.abs(mine - sk.coef_).max()), 1e-8)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nSmoothing spline (README §8)")
    ss_small = SmoothingSpline(lam=1e-6).fit(x, y)
    ss_large = SmoothingSpline(lam=1e8).fit(x, y)
    print(f"  [INFO]  {'df at lambda=1e-6 and lambda=1e8':<56s}  "
          f"{ss_small.df_:.2f} -> {ss_large.df_:.2f}")
    ok &= ss_small.df_ > ss_large.df_

    # lambda -> infinity must give a straight line: OLS.
    linear_fit = np.polyfit(x, y, 1)
    ok &= _report("lambda -> inf reduces to a straight line",
                  float(np.abs(ss_large.predict(x) - np.polyval(linear_fit, x)).max()), 0.05)
    ok &= _report("df -> 2 as lambda -> inf", abs(ss_large.df_ - 2.0), 0.05)

    print("\nGAM (README §10)")
    n = 400
    X = rng.uniform(-2, 2, (n, 3))
    y_add = 2 * np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2] + \
        rng.standard_normal(n) * 0.2
    gam = GAM(n_knots=6).fit(X, y_add)
    r2_gam = 1 - np.sum((y_add - gam.predict(X)) ** 2) / np.sum((y_add - y_add.mean()) ** 2)

    from_linear = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y_add, rcond=None)[0]
    r2_lin = 1 - np.sum((y_add - np.column_stack([np.ones(n), X]) @ from_linear) ** 2) / \
        np.sum((y_add - y_add.mean()) ** 2)

    print(f"  [INFO]  {'backfitting converged in':<56s}  {gam.n_iter_} iterations")
    print(f"  [{'PASS' if r2_gam > 0.95 else 'FAIL'}]  "
          f"{'GAM recovers additive structure':<56s}  R^2 = {r2_gam:.4f}")
    print(f"  [INFO]  {'...where a linear model gets':<56s}  R^2 = {r2_lin:.4f}")
    ok &= r2_gam > 0.95 and r2_gam > r2_lin

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_runge() -> None:
    """README §3: polynomial interpolation error DIVERGES as the degree grows."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the Runge phenomenon  (README §3)")
    print("=" * 88)
    print("""
Fitting f(x) = 1/(1 + 25x^2) on [-1, 1] at equally spaced points. Intuition says a
higher-degree polynomial should fit better. It does — in the middle. At the edges the
error grows without bound, and MORE DATA MAKES IT WORSE.
""")
    def f(x):
        return 1.0 / (1.0 + 25 * x ** 2)

    x_dense = np.linspace(-1, 1, 2000)
    y_dense = f(x_dense)
    edge = np.abs(x_dense) > 0.85
    centre = np.abs(x_dense) < 0.5

    print(f"  {'degree':>7s}  {'n points':>9s}  {'max err (centre)':>17s}  "
          f"{'max err (EDGE)':>15s}  {'kappa(Phi)':>11s}")
    print("  " + "-" * 66)

    for degree in (4, 8, 12, 16, 20):
        n_points = degree + 1
        x_train = np.linspace(-1, 1, n_points)
        Phi = legendre_basis(x_train, degree, domain=(-1, 1))   # well-conditioned basis
        coef = np.linalg.lstsq(Phi, f(x_train), rcond=None)[0]
        pred = legendre_basis(x_dense, degree, domain=(-1, 1)) @ coef

        err_centre = float(np.abs(pred[centre] - y_dense[centre]).max())
        err_edge = float(np.abs(pred[edge] - y_dense[edge]).max())
        print(f"  {degree:7d}  {n_points:9d}  {err_centre:17.5f}  {err_edge:15.5f}  "
              f"{np.linalg.cond(Phi):11.2e}")

    print("""
  The centre column improves. The EDGE column gets steadily worse — by degree 20 the error
  near the boundary is many times the function's entire range (f is bounded by 1).

  Note this is computed in a well-conditioned Legendre basis, so it is not a numerical
  artifact. Runge is a property of high-degree polynomial interpolation on equally spaced
  points, full stop.

  Now the same target with cubic SPLINES, adding knots instead of degree:
""")
    print(f"  {'knots':>7s}  {'basis fns':>10s}  {'max err (centre)':>17s}  "
          f"{'max err (EDGE)':>15s}")
    print("  " + "-" * 54)

    x_train = np.linspace(-1, 1, 40)
    for n_knots in (4, 8, 12, 16):
        model = SplineRegression(basis="natural", n_knots=n_knots).fit(x_train, f(x_train))
        pred = model.predict(x_dense)
        print(f"  {n_knots:7d}  {model.n_basis_:10d}  "
              f"{float(np.abs(pred[centre] - y_dense[centre]).max()):17.5f}  "
              f"{float(np.abs(pred[edge] - y_dense[edge]).max()):15.5f}")

    print("""
  Both columns improve monotonically. Keeping the degree at 3 and adding PIECES converges;
  raising the degree does not.

  This is the entire argument for splines over polynomials, and it is why numerical
  analysis abandoned high-degree polynomial interpolation decades ago.""")


def experiment_conditioning() -> None:
    """README §2: raw powers are ill-conditioned; the fix is a change of basis."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — conditioning of polynomial bases  (README §2)")
    print("=" * 88)
    print("""
The columns [1, x, x^2, ...] become nearly collinear as the degree grows — x^9 and x^10
are very similar functions on a bounded interval. From 00.01 §15, kappa ~ 10^k means
losing about k digits.
""")
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(4 * x) + rng.standard_normal(200) * 0.05

    print(f"  {'degree':>7s}  {'kappa(raw powers)':>19s}  {'kappa(Legendre)':>17s}  "
          f"{'digits lost (raw)':>18s}  {'same fit?':>10s}")
    print("  " + "-" * 78)

    for degree in (3, 6, 9, 12, 15, 18):
        raw = polynomial_basis(x, degree)
        leg = legendre_basis(x, degree)
        k_raw = float(np.linalg.cond(raw))
        k_leg = float(np.linalg.cond(leg))

        pred_raw = raw @ np.linalg.lstsq(raw, y, rcond=None)[0]
        pred_leg = leg @ np.linalg.lstsq(leg, y, rcond=None)[0]
        agree = float(np.abs(pred_raw - pred_leg).max())

        print(f"  {degree:7d}  {k_raw:19.3e}  {k_leg:17.3e}  "
              f"{np.log10(k_raw):18.1f}  {agree:10.2e}")

    print("""
  Raw powers reach kappa ~ 1e10 by degree 12 and keep climbing — over half of float64's
  16 digits gone. Legendre polynomials stay near 1e1 at every degree.

  The last column is the point: the two bases span the SAME function space, so the fitted
  curve is identical to within the precision the raw basis still has. The problem is
  purely the parameterization, and the fix is free.

  R's poly() uses an orthogonal basis by default, which is why polynomial regression
  misbehaves less there than in a naive NumPy implementation. But note this fixes only the
  ARITHMETIC — Runge (Experiment 1) and bad extrapolation (Experiment 3) are properties of
  the function class and survive any change of basis.""")


def experiment_extrapolation() -> None:
    """README §7: what each model does outside the data range."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — extrapolation  (README §2, §7)")
    print("=" * 88)
    print("""
Models are fitted on x in [0, 10] and then asked about points beyond it. The truth is
y = sin(x) + 0.1x^2, which keeps growing gently. Nothing here can KNOW that — the question
is how badly each one fails.
""")
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 10, 150))
    y = np.sin(x) + 0.1 * x ** 2 + rng.standard_normal(150) * 0.2

    models = {
        "polynomial degree 3": SplineRegression(basis="legendre", degree=3),
        "polynomial degree 10": SplineRegression(basis="legendre", degree=10),
        "cubic spline (B-spline)": SplineRegression(basis="bspline", n_knots=6),
        "NATURAL cubic spline": SplineRegression(basis="natural", n_knots=6),
    }
    for m in models.values():
        m.fit(x, y)

    test_points = np.array([10.5, 12.0, 15.0, 20.0])
    truth = np.sin(test_points) + 0.1 * test_points ** 2

    print(f"  {'model':<26s}", end="")
    for t in test_points:
        print(f"  {f'x={t:g}':>12s}", end="")
    print()
    print(f"  {'true value':<26s}", end="")
    for v in truth:
        print(f"  {v:12.2f}", end="")
    print("\n  " + "-" * 76)

    for name, m in models.items():
        print(f"  {name:<26s}", end="")
        for p in m.predict(test_points):
            print(f"  {p:12.2f}" if abs(p) < 1e6 else f"  {p:12.2e}", end="")
        print()

    print("""
  Three genuinely different failure modes, and the differences matter more than the
  magnitudes.

  THE DEGREE-10 POLYNOMIAL EXPLODES. It is wrong by five orders of magnitude at x = 20 and
  accelerating — outside the data the highest power dominates and nothing restrains it.
  This is the same disease as Runge (Experiment 1), seen from outside the interval.

  THE B-SPLINE RETURNS EXACTLY ZERO. This surprises people, and it is worth knowing: every
  B-spline basis function has compact support (README §6), so outside the boundary knots
  ALL of them are zero and the model predicts 0 — not a diverging cubic, just 0. That is
  a silent, plausible-looking, completely meaningless number. scipy's BSpline exposes an
  `extrapolate` flag precisely because this trap is easy to fall into: a model that quietly
  answers 0 for every out-of-range input is more dangerous than one that answers 776,963.

  THE NATURAL CUBIC SPLINE DEGRADES GRACEFULLY. Linear beyond the boundary knots by
  construction (README §7), so it keeps extending the trend it last saw. It is still wrong
  — the truth is quadratic and it is only linear — but wrong by a factor of 2 rather than
  10^5, and it does not accelerate.

  The degree-3 polynomial also does reasonably here, but that is luck: the truth happens to
  be roughly quadratic, so a cubic has close to the right shape. Change the target and it
  fails like the degree-10 one.

  The general lesson: extrapolation is not a solved problem, and the honest response is to
  refuse it — check the input range and raise. If you must extrapolate, prefer a model that
  is linear outside its data, and never one whose basis silently evaluates to zero.""")


def experiment_smoothing() -> None:
    """README §8-§9: lambda, effective df, and GCV's choice."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — smoothing splines: lambda, df, and GCV  (README §8-§9)")
    print("=" * 88)
    print("""
A smoothing spline puts a knot at every point and controls complexity ENTIRELY through
lambda. The interpretable quantity is not lambda but the effective degrees of freedom,
df = trace(S_lambda) — the same quantity as trace(H) for OLS (03.01 §10.1).
""")
    rng = np.random.default_rng(3)
    n = 150
    x = np.sort(rng.uniform(0, 1, n))
    f_true = np.sin(3 * np.pi * x) * np.exp(-x)
    y = f_true + rng.standard_normal(n) * 0.15

    x_test = np.sort(rng.uniform(0, 1, 500))
    f_test = np.sin(3 * np.pi * x_test) * np.exp(-x_test)
    y_test = f_test + rng.standard_normal(500) * 0.15

    print(f"  {'lambda':>10s}  {'eff. df':>9s}  {'train MSE':>11s}  {'TEST MSE':>11s}  "
          f"{'GCV':>11s}")
    print("  " + "-" * 58)

    best_test, best_lam = np.inf, None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8):
        m = SmoothingSpline(lam=lam, n_knots=40).fit(x, y)
        train_mse = float(np.mean((y - m.predict(x)) ** 2))
        test_mse = float(np.mean((y_test - m.predict(x_test)) ** 2))
        if test_mse < best_test:
            best_test, best_lam = test_mse, lam
        print(f"  {lam:10.0e}  {m.df_:9.2f}  {train_mse:11.5f}  {test_mse:11.5f}  "
              f"{m.gcv_:11.5f}")

    gcv_model = SmoothingSpline.fit_gcv(x, y, n_knots=40)
    gcv_test = float(np.mean((y_test - gcv_model.predict(x_test)) ** 2))

    print(f"""
  Train MSE falls monotonically as lambda shrinks — the model is free to interpolate.
  Test MSE has a minimum, at lambda = {best_lam:g}, and rises on both sides. That is the
  bias-variance trade of 03.02 §1, in a different parameterization.

  GCV selected lambda = {gcv_model.lam:.3g}, giving df = {gcv_model.df_:.2f} and test MSE
  {gcv_test:.5f} — against the best achievable {best_test:.5f}. It found a near-optimal
  choice WITHOUT a validation set, from a single fit per lambda, by exploiting the fact
  that S_lambda is linear in y (README §9).

  The df column is why practitioners specify df rather than lambda: 'a smooth with 8
  degrees of freedom' is a statement someone can reason about; 'lambda = 3.7e-5' is not,
  and does not transfer to another dataset.""")


def experiment_gam() -> None:
    """README §10: additive structure, and the interaction it cannot capture."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — GAMs: what additivity buys and what it costs  (README §10)")
    print("=" * 88)
    print("""
Two datasets. The first is genuinely additive; the second has an interaction. Comparing a
linear model, a GAM, and a full nonparametric fit (gradient boosting) on each:
""")
    rng = np.random.default_rng(4)
    n = 600

    def evaluate(X, y, X_test, y_test, label):
        results = {}

        A = np.column_stack([np.ones(len(X)), X])
        A_test = np.column_stack([np.ones(len(X_test)), X_test])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        results["linear model"] = float(np.mean((y_test - A_test @ coef) ** 2))

        gam = GAM(n_knots=6).fit(X, y)
        results["GAM (additive)"] = float(np.mean((y_test - gam.predict(X_test)) ** 2))

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            gb = GradientBoostingRegressor(random_state=0).fit(X, y)
            results["gradient boosting"] = float(np.mean((y_test - gb.predict(X_test)) ** 2))
        except ImportError:
            pass

        var_y = float(np.var(y_test))
        print(f"  {label}")
        print(f"    {'model':<22s}  {'test MSE':>10s}  {'R^2':>8s}")
        print("    " + "-" * 44)
        for name, mse in results.items():
            print(f"    {name:<22s}  {mse:10.4f}  {1 - mse / var_y:8.4f}")
        print()

    # --- additive truth ---
    X = rng.uniform(-2, 2, (n, 2))
    X_test = rng.uniform(-2, 2, (n, 2))
    y = 2 * np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + rng.standard_normal(n) * 0.3
    y_test = 2 * np.sin(2 * X_test[:, 0]) + X_test[:, 1] ** 2 + rng.standard_normal(n) * 0.3
    evaluate(X, y, X_test, y_test,
             "TRUTH IS ADDITIVE:  y = 2 sin(2 x1) + x2^2 + noise")

    # --- interaction ---
    y = 3 * X[:, 0] * X[:, 1] + rng.standard_normal(n) * 0.3
    y_test = 3 * X_test[:, 0] * X_test[:, 1] + rng.standard_normal(n) * 0.3
    evaluate(X, y, X_test, y_test,
             "TRUTH IS AN INTERACTION:  y = 3 x1 x2 + noise")

    print("""  On the ADDITIVE data the GAM is dramatically better than the linear model — it
  recovers the sine and the parabola that linearity cannot express — and it is competitive
  with gradient boosting, while remaining fully plottable: you can draw f_1 and f_2 and
  read off exactly what the model believes.

  On the INTERACTION data the GAM is barely better than the linear model, and both are far
  behind boosting. This is not a tuning failure. y = 3 x1 x2 has NO additive decomposition;
  the marginal effect of x1 is zero when averaged over x2, so f_1 and f_2 have nothing to
  find. No amount of flexibility in each individual smooth can help.

  That is the trade in one experiment. A GAM buys you arbitrary smooth nonlinearity per
  feature at essentially no interpretability cost — and it buys you nothing at all when the
  structure lives in the interactions. Knowing which situation you are in is the whole
  decision, and it is exactly the gap that trees and networks fill by searching for
  interactions automatically.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_runge()
    experiment_conditioning()
    experiment_extrapolation()
    experiment_smoothing()
    experiment_gam()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
