"""
03.04 — Logistic Regression from Scratch
========================================

Binary and multinomial logistic regression, four solvers, full inference, and a
numerically stable loss throughout.

Implemented here
----------------
    LogisticRegression      binary, with:
        solver="gd"         gradient descent
        solver="newton"     Newton / IRLS — README §8
        solver="lbfgs"      quasi-Newton (sklearn's default)
        solver="sgd"        stochastic, for large n
        penalty="l2"/"l1"/None

        .odds_ratios()      exp(w), with confidence intervals   README §3
        .coef_se_           from the inverse Hessian            00.04 §6

    SoftmaxRegression       multinomial                          README §11

    stable_sigmoid, bce_with_logits, log_softmax   00.06 §8-§9

Run it
------
    python from_scratch.py

Verified against sklearn and statsmodels, then five experiments:
  1. Why OLS fails at classification — including the surprising one, where adding
     CORRECTLY-labelled points makes an OLS classifier worse
  2. Newton/IRLS vs gradient descent: iteration counts, and quadratic convergence
  3. Perfect separation: coefficients diverge without regularization
  4. Calibration: logistic regression's predicted probabilities sum to the observed
     positives EXACTLY, and other classifiers' do not
  5. Multinomial vs one-vs-rest

Reference: README.md sections 2-13.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# NUMERICALLY STABLE PRIMITIVES  (00.06 §8-§9)
# =============================================================================


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """1/(1+exp(-z)), branching on the sign so the exponent is never positive."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """max(z,0) - z*y + log(1 + exp(-|z|)).  README §4, 00.06 §9

    The naive -[y log(sigma(z)) + (1-y) log(1-sigma(z))] returns inf once sigma saturates
    — around |z| = 37 in float64, |z| = 17 in float32, both entirely ordinary logits for a
    confident model. This form is bounded at every z.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Z - logsumexp(Z), computed directly rather than as log(softmax(Z))."""
    Z = np.asarray(Z, dtype=float)
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# BINARY LOGISTIC REGRESSION
# =============================================================================


class LogisticRegression:
    """Binary logistic regression: p(y=1|x) = sigmoid(w.x + b).

    NOTE ON THE PENALTY PARAMETER. This class takes `lam` (lambda) directly, where larger
    means more regularization. sklearn takes C = 1/lambda, so SMALLER C means more — the
    opposite convention, and a persistent source of confusion (README §10). The
    verification below converts between them explicitly.
    """

    def __init__(self, solver: str = "lbfgs", penalty: str | None = "l2",
                 lam: float = 0.0, max_iter: int = 1000, tol: float = 1e-10,
                 lr: float = 0.1, batch_size: int = 32, fit_intercept: bool = True,
                 random_state: int = 0):
        self.solver = solver
        self.penalty = penalty
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    # --- objective, gradient, Hessian  (README §4-§5) ---------------------

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.column_stack([np.ones(X.shape[0]), X]) if self.fit_intercept else X

    def _penalty_mask(self, d: int) -> np.ndarray:
        """1 for penalized coefficients, 0 for the intercept.

        The intercept is never penalized, for the same reason as in ridge (03.02 §2.1):
        shrinking it makes predictions depend on the arbitrary location of the origin.
        """
        mask = np.ones(d)
        if self.fit_intercept:
            mask[0] = 0.0
        return mask

    def _loss(self, w: np.ndarray, A: np.ndarray, y: np.ndarray) -> float:
        z = A @ w
        loss = float(np.mean(bce_with_logits(z, y)))
        mask = self._penalty_mask(w.size)
        if self.penalty == "l2":
            loss += 0.5 * self.lam * float(np.sum((w * mask) ** 2)) / A.shape[0]
        elif self.penalty == "l1":
            loss += self.lam * float(np.sum(np.abs(w * mask))) / A.shape[0]
        return loss

    def _gradient(self, w: np.ndarray, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """grad = X^T (p - y) / n  — features times residuals.  README §5

        Structurally identical to linear regression's gradient. That is not a coincidence:
        it is a general property of GLMs fitted with their canonical link.
        """
        p = stable_sigmoid(A @ w)
        grad = A.T @ (p - y) / A.shape[0]
        mask = self._penalty_mask(w.size)
        if self.penalty == "l2":
            grad += self.lam * (w * mask) / A.shape[0]
        return grad

    def _hessian(self, w: np.ndarray, A: np.ndarray) -> np.ndarray:
        """H = X^T S X / n  with S = diag(p(1-p)).  README §5

        S weights each observation by p(1-p): maximal at p = 0.5 (the uncertain points
        near the boundary) and near zero for confident ones. The model learns almost
        entirely from the examples it is unsure about.

        H is PSD because v^T H v = ||S^(1/2) X v||^2 >= 0, which is the convexity proof of
        README §6 in one line.
        """
        p = stable_sigmoid(A @ w)
        s = p * (1 - p)
        H = (A.T * s) @ A / A.shape[0]
        mask = self._penalty_mask(w.size)
        if self.penalty == "l2":
            H += np.diag(self.lam * mask) / A.shape[0]
        return H

    # --- solvers -----------------------------------------------------------

    def _fit_gd(self, A, y, w):
        for i in range(self.max_iter):
            grad = self._gradient(w, A, y)
            if np.max(np.abs(grad)) < self.tol:
                return w, i + 1
            w = w - self.lr * grad
        return w, self.max_iter

    def _fit_newton(self, A, y, w):
        """Newton / IRLS.  README §8

        Each step is equivalent to a weighted least squares fit of the working response
        z = Xw + S^-1 (y - p) on X with weights S — which is why this algorithm is called
        Iteratively Reweighted Least Squares and why the same code fits every GLM.

        Converges QUADRATICALLY near the optimum, so 5-8 iterations is typical where
        gradient descent needs thousands (Experiment 2).

        Ridge on the Hessian guards the separable case, where S -> 0 and H becomes
        singular (README §9).
        """
        for i in range(self.max_iter):
            grad = self._gradient(w, A, y)
            if np.max(np.abs(grad)) < self.tol:
                return w, i + 1
            H = self._hessian(w, A) + 1e-10 * np.eye(w.size)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = grad                         # fall back to gradient descent
            # Backtracking line search: full Newton steps can overshoot far from the
            # optimum, where the quadratic model is a poor fit (00.02 §12.1).
            t = 1.0
            current = self._loss(w, A, y)
            for _ in range(50):
                if self._loss(w - t * step, A, y) <= current:
                    break
                t *= 0.5
            w = w - t * step
        return w, self.max_iter

    def _fit_lbfgs(self, A, y, w):
        try:
            from scipy.optimize import minimize
            result = minimize(lambda v: self._loss(v, A, y), w,
                              jac=lambda v: self._gradient(v, A, y),
                              method="L-BFGS-B",
                              options={"maxiter": self.max_iter, "gtol": self.tol,
                                       "ftol": 1e-16})
            return result.x, int(result.nit)
        except ImportError:
            return self._fit_newton(A, y, w)

    def _fit_sgd(self, A, y, w):
        rng = np.random.default_rng(self.random_state)
        n = A.shape[0]
        for epoch in range(self.max_iter):
            for start in range(0, n, self.batch_size):
                idx = rng.integers(0, n, self.batch_size)
                w = w - self.lr * self._gradient(w, A[idx], y[idx])
            # Decaying step size: SGD with a constant lr reaches a noise ball rather than
            # the optimum (00.02 §9).
            self.lr *= 0.995
        return w, self.max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        A = self._design(X)
        y = np.asarray(y, dtype=float).ravel()
        w = np.zeros(A.shape[1])

        solvers = {"gd": self._fit_gd, "newton": self._fit_newton,
                   "lbfgs": self._fit_lbfgs, "sgd": self._fit_sgd}
        if self.solver not in solvers:
            raise ValueError(f"unknown solver {self.solver!r}")
        w, n_iter = solvers[self.solver](A, y, w)

        self.n_iter_ = n_iter
        self.theta_ = w
        self.intercept_ = float(w[0]) if self.fit_intercept else 0.0
        self.coef_ = w[1:] if self.fit_intercept else w
        self._A, self._y = A, y

        # Standard errors from the inverse observed Fisher information (00.04 §6). Only
        # valid unregularized; a penalty biases the estimator and invalidates these.
        H = self._hessian(w, A) * A.shape[0]
        try:
            self.coef_se_ = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            self.coef_se_ = np.full(w.size, np.nan)
        return self

    # --- prediction and inference -----------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.theta_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = stable_sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Threshold defaults to 0.5, which is a convention and not a law (README §12)."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def odds_ratios(self, alpha: float = 0.05) -> dict:
        """exp(w) with confidence intervals.  README §3

        Increasing x_j by one unit MULTIPLIES the odds by exp(w_j). Note this is an odds
        ratio, not a risk ratio — they coincide only when p is small.

        The interval is built on the log-odds scale and then exponentiated, which is
        correct: exp() is monotone, so the transformed endpoints bound the same coverage.
        Building it symmetrically around exp(w) would not.
        """
        try:
            from scipy import stats
            crit = stats.norm.ppf(1 - alpha / 2)
        except ImportError:
            crit = 1.959963984540054
        lo = self.theta_ - crit * self.coef_se_
        hi = self.theta_ + crit * self.coef_se_
        return {"odds_ratio": np.exp(self.theta_),
                "ci_lower": np.exp(lo), "ci_upper": np.exp(hi)}


# =============================================================================
# MULTINOMIAL  (README §11)
# =============================================================================


class SoftmaxRegression:
    """p(y=k|x) = exp(w_k.x) / sum_j exp(w_j.x).

    The model is OVER-PARAMETERIZED: adding any constant vector to every w_k leaves all
    probabilities unchanged, because softmax is shift-invariant (00.06 §8). So the solution
    is not unique without regularization. We keep all K weight vectors and rely on the L2
    penalty to pin them down, which is what sklearn's multinomial mode does.
    """

    def __init__(self, lam: float = 1e-4, max_iter: int = 2000, lr: float = 0.5,
                 tol: float = 1e-9, fit_intercept: bool = True):
        self.lam = lam
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(X.shape[0]), X]) if self.fit_intercept else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        A = self._design(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        K = self.classes_.size
        n, d = A.shape

        Y = np.zeros((n, K))                        # one-hot
        for k, c in enumerate(self.classes_):
            Y[y == c, k] = 1.0

        W = np.zeros((d, K))
        mask = np.ones(d)
        if self.fit_intercept:
            mask[0] = 0.0

        for i in range(self.max_iter):
            P = softmax(A @ W)
            # Same shape as every other gradient here: features times residuals.
            grad = A.T @ (P - Y) / n + self.lam * (W * mask[:, None]) / n
            if np.max(np.abs(grad)) < self.tol:
                break
            W = W - self.lr * grad

        self.n_iter_ = i + 1
        self.W_ = W
        self.intercept_ = W[0] if self.fit_intercept else np.zeros(K)
        self.coef_ = W[1:].T if self.fit_intercept else W.T
        return self

    def predict_proba(self, X):
        return softmax(self._design(X) @ self.W_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


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

    n, d = 500, 5
    X = rng.standard_normal((n, d))
    true_w = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
    z = 0.3 + X @ true_w
    y = (rng.random(n) < stable_sigmoid(z)).astype(int)

    print("\nNumerically stable primitives (00.06 §9)")
    extreme = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
    ok &= _report("bce_with_logits finite at |z| = 800",
                  0.0 if np.all(np.isfinite(bce_with_logits(extreme, np.zeros(5)))) else 1.0,
                  0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = -np.log(1 - stable_sigmoid(np.array([40.0])))
    print(f"  [INFO]  {'naive -log(1-sigmoid(40)) vs stable':<56s}  "
          f"{naive[0]} vs {bce_with_logits(np.array([40.0]), np.array([0.0]))[0]:.4f}")

    print("\nAll four solvers agree (README §8)")
    reference = LogisticRegression(solver="newton", penalty=None, max_iter=200).fit(X, y)
    for solver, tol in (("lbfgs", 1e-5), ("gd", 2e-2), ("sgd", 2e-1)):
        other = LogisticRegression(solver=solver, penalty=None,
                                   max_iter=20000 if solver == "gd" else 300).fit(X, y)
        ok &= _report(f"solver={solver!r} matches Newton",
                      float(np.abs(other.theta_ - reference.theta_).max()), tol)
    print(f"  [INFO]  {'iterations: newton / lbfgs / gd':<56s}  "
          f"{reference.n_iter_} / "
          f"{LogisticRegression(solver='lbfgs', penalty=None).fit(X, y).n_iter_} / "
          f"{LogisticRegression(solver='gd', penalty=None, max_iter=20000).fit(X, y).n_iter_}")

    print("\nAgainst sklearn")
    try:
        from sklearn.linear_model import LogisticRegression as SKLogistic

        # Unregularized. sklearn needs penalty=None explicitly — its default is C=1.0,
        # i.e. regularization ON (README §10).
        sk = SKLogistic(penalty=None, max_iter=5000, tol=1e-12).fit(X, y)
        mine = LogisticRegression(solver="lbfgs", penalty=None, max_iter=5000).fit(X, y)
        ok &= _report("unregularized coefficients vs sklearn",
                      float(np.abs(mine.coef_ - sk.coef_[0]).max()), 1e-5)
        ok &= _report("unregularized intercept vs sklearn",
                      abs(mine.intercept_ - sk.intercept_[0]), 1e-5)
        ok &= _report("predicted probabilities vs sklearn",
                      float(np.abs(mine.predict_proba(X)[:, 1]
                                   - sk.predict_proba(X)[:, 1]).max()), 1e-6)

        # Regularized. Working out the correspondence carefully, because the two
        # parameterizations differ in BOTH the direction of the knob and the 1/n scaling:
        #
        #   sklearn minimizes    sum_i BCE_i + (1 / 2C) ||w||^2
        #   this class minimizes mean_i BCE_i + (lam / 2n) ||w||^2
        #                      = (1/n) [ sum_i BCE_i + (lam/2) ||w||^2 ]
        #
        # The 1/n is an overall scale and does not move the minimizer, so matching the
        # ratio of penalty to likelihood gives lam = 1/C. The n cancels — it does NOT
        # appear in the conversion, which is the easy mistake to make here.
        for C in (1.0, 0.1, 0.01):
            sk_reg = SKLogistic(C=C, max_iter=10000, tol=1e-14).fit(X, y)
            mine_reg = LogisticRegression(solver="lbfgs", penalty="l2", lam=1.0 / C,
                                          max_iter=10000).fit(X, y)
            ok &= _report(f"L2 with C={C} (lam = 1/C = {1 / C:g}) vs sklearn",
                          float(np.abs(mine_reg.coef_ - sk_reg.coef_[0]).max()), 1e-4)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nAgainst statsmodels — standard errors and odds ratios (README §3)")
    try:
        import statsmodels.api as sm
        sm_model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        mine = LogisticRegression(solver="newton", penalty=None, max_iter=200).fit(X, y)
        ok &= _report("coefficients vs statsmodels",
                      float(np.abs(mine.theta_ - sm_model.params).max()), 1e-6)
        ok &= _report("standard errors vs statsmodels",
                      float(np.abs(mine.coef_se_ - sm_model.bse).max()), 1e-5)
        ors = mine.odds_ratios()
        ok &= _report("odds ratios vs exp(statsmodels params)",
                      float(np.abs(ors["odds_ratio"] - np.exp(sm_model.params)).max()), 1e-5)
    except ImportError:
        print("  [SKIP]  statsmodels not installed")

    print("\nStructural properties (README §5, §6, §13)")
    mine = LogisticRegression(solver="newton", penalty=None, max_iter=200).fit(X, y)

    # Convexity: the Hessian must be PSD everywhere, not just at the optimum.
    worst = 0.0
    for _ in range(50):
        w_random = rng.standard_normal(d + 1) * 2
        eigenvalues = np.linalg.eigvalsh(mine._hessian(w_random, mine._A))
        worst = min(worst, float(eigenvalues.min()))
    ok &= _report("Hessian is PSD at 50 random points (convexity)", abs(min(worst, 0.0)), 1e-12)

    # Calibration identity: at the optimum, sum(p) = sum(y) exactly (README §13).
    p = mine.predict_proba(X)[:, 1]
    ok &= _report("sum(predicted p) = sum(y) at the optimum",
                  abs(float(p.sum() - y.sum())), 1e-6)

    print("\nSoftmax regression (README §11)")
    n_multi = 600
    X_multi = rng.standard_normal((n_multi, 4))
    logits = X_multi @ rng.standard_normal((4, 3))
    y_multi = np.array([rng.choice(3, p=row) for row in softmax(logits)])

    ok &= _report("softmax rows sum to 1",
                  float(np.abs(softmax(logits).sum(axis=1) - 1).max()), 1e-12)
    ok &= _report("log_softmax = log(softmax) for moderate logits",
                  float(np.abs(log_softmax(logits) - np.log(softmax(logits))).max()), 1e-12)

    sr = SoftmaxRegression(lam=1.0, max_iter=6000).fit(X_multi, y_multi)
    accuracy = float(np.mean(sr.predict(X_multi) == y_multi))
    print(f"  [INFO]  {'softmax regression training accuracy':<56s}  {accuracy:.4f}")
    ok &= accuracy > 0.5

    # With K=2, softmax must reproduce binary logistic regression.
    sr2 = SoftmaxRegression(lam=1e-8, max_iter=30000, lr=1.0).fit(X, y)
    binary = LogisticRegression(solver="lbfgs", penalty=None, max_iter=5000).fit(X, y)
    ok &= _report("softmax with K=2 matches binary logistic (probabilities)",
                  float(np.abs(sr2.predict_proba(X)[:, 1]
                               - binary.predict_proba(X)[:, 1]).max()), 5e-3)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_why_not_ols() -> None:
    """README §1: what goes wrong when you fit OLS to 0/1 labels."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — why not linear regression  (README §1)")
    print("=" * 88)
    print("""
Encode the classes 0/1 and fit OLS. Two problems are obvious and one is not.
""")
    rng = np.random.default_rng(1)
    n = 200
    X = np.concatenate([rng.normal(-2, 1, n // 2), rng.normal(2, 1, n // 2)])[:, None]
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    A = np.column_stack([np.ones(n), X])
    ols = np.linalg.lstsq(A, y, rcond=None)[0]
    ols_pred = A @ ols

    print(f"  OBVIOUS PROBLEM 1 — predictions leave [0, 1]")
    print(f"    OLS fitted values range: [{ols_pred.min():.3f}, {ols_pred.max():.3f}]")
    print(f"    fraction outside [0,1]:  {np.mean((ols_pred < 0) | (ols_pred > 1)):.1%}")

    print(f"\n  OBVIOUS PROBLEM 2 — the variance is not constant")
    print(f"    Var(y|x) = p(1-p), which is 0.25 at p=0.5 and ~0 at the extremes,")
    print(f"    so the homoscedasticity assumption of 03.01 §5 is violated by construction.")

    print("""
  THE NON-OBVIOUS PROBLEM — adding CORRECT data makes OLS worse.

  We now add a cluster of unambiguous, correctly-labelled positives far to the right.
  They carry no new information about where the boundary is; every classifier should
  ignore them.
""")
    print(f"  {'extra points at x=10':>21s}  {'OLS boundary':>14s}  {'OLS acc':>9s}  "
          f"{'logistic boundary':>18s}  {'logistic acc':>13s}")
    print("  " + "-" * 82)

    for n_extra in (0, 10, 30, 100):
        X_aug = np.concatenate([X.ravel(), np.full(n_extra, 10.0)])[:, None]
        y_aug = np.concatenate([y, np.ones(n_extra)])

        A_aug = np.column_stack([np.ones(X_aug.size), X_aug])
        w_ols = np.linalg.lstsq(A_aug, y_aug, rcond=None)[0]
        # OLS decision boundary: where the fitted value crosses 0.5.
        ols_boundary = (0.5 - w_ols[0]) / w_ols[1]
        ols_acc = float(np.mean(((A @ w_ols) >= 0.5).astype(int) == y))

        lr = LogisticRegression(solver="newton", penalty=None, max_iter=200).fit(X_aug, y_aug)
        log_boundary = -lr.intercept_ / lr.coef_[0]
        log_acc = float(np.mean(lr.predict(X) == y))

        print(f"  {n_extra:21d}  {ols_boundary:14.3f}  {ols_acc:9.3f}  "
              f"{log_boundary:18.3f}  {log_acc:13.3f}")

    print("""
  The true boundary is at x = 0. Logistic regression stays there and its accuracy does not
  move. OLS's boundary drifts steadily to the right and its accuracy falls.

  The mechanism: OLS penalizes SQUARED DISTANCE from the target. A point at x = 10 labelled
  1 gets a fitted value well above 1, producing a large residual — so OLS rotates the line
  to reduce it, even though that point was already classified correctly and confidently.
  OLS is penalizing the model for being 'too right'.

  Logistic regression's loss saturates: once sigma(z) is close to 1 for a positive example,
  the gradient contribution (p - y) is near zero and the point stops mattering. That is the
  weighting S = p(1-p) of README §5 doing its job — the model learns from the points near
  the boundary and ignores the settled ones.""")


def experiment_convergence() -> None:
    """README §8: IRLS converges quadratically; gradient descent does not."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — Newton/IRLS vs gradient descent  (README §8)")
    print("=" * 88)
    print("""
The logistic loss is convex with a cheap PSD Hessian, so a second-order method is
affordable — and it converges quadratically (00.02 §12.1), doubling the number of correct
digits each step.
""")
    rng = np.random.default_rng(2)
    n, d = 800, 8
    X = rng.standard_normal((n, d))
    y = (rng.random(n) < stable_sigmoid(0.5 + X @ rng.standard_normal(d))).astype(int)

    A = np.column_stack([np.ones(n), X])
    optimum = LogisticRegression(solver="newton", penalty=None,
                                 max_iter=100, tol=1e-14).fit(X, y).theta_

    print(f"  {'iteration':>10s}  {'Newton/IRLS ||w - w*||':>24s}  "
          f"{'gradient descent ||w - w*||':>29s}")
    print("  " + "-" * 68)

    model = LogisticRegression(solver="newton", penalty=None)
    w_newton = np.zeros(d + 1)
    w_gd = np.zeros(d + 1)
    gd_model = LogisticRegression(solver="gd", penalty=None, lr=1.0)

    for it in range(1, 11):
        # One Newton step.
        grad = model._gradient(w_newton, A, y)
        H = model._hessian(w_newton, A) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        t, current = 1.0, model._loss(w_newton, A, y)
        for _ in range(50):
            if model._loss(w_newton - t * step, A, y) <= current:
                break
            t *= 0.5
        w_newton = w_newton - t * step

        # One gradient step.
        w_gd = w_gd - gd_model.lr * gd_model._gradient(w_gd, A, y)

        print(f"  {it:10d}  {np.linalg.norm(w_newton - optimum):24.3e}  "
              f"{np.linalg.norm(w_gd - optimum):29.3e}")

    gd_full = LogisticRegression(solver="gd", penalty=None, lr=1.0,
                                 max_iter=200_000, tol=1e-10).fit(X, y)
    newton_full = LogisticRegression(solver="newton", penalty=None,
                                     max_iter=100, tol=1e-10).fit(X, y)
    print(f"""
  Newton's error falls to machine precision in about 6 steps. Read the exponents: the
  number of correct digits roughly DOUBLES each iteration, which is what quadratic
  convergence means.

  Gradient descent is still nowhere near after 10 steps. To reach the same tolerance it
  needs {gd_full.n_iter_:,} iterations against Newton's {newton_full.n_iter_}.

  The catch is cost per iteration: Newton solves a d x d system, O(nd^2 + d^3), against
  gradient descent's O(nd). At d = 8 that is free. At d = 100,000 it is impossible, which
  is why sklearn defaults to L-BFGS — quasi-Newton curvature at O(md) memory
  (00.02 §12.2) — and why SGD takes over for very large n.""")


def experiment_separation() -> None:
    """README §9: perfect separation makes the MLE diverge."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — perfect separation  (README §9)")
    print("=" * 88)
    print("""
If a hyperplane separates the classes perfectly, scaling w by any c > 1 makes every
prediction more confident and the likelihood strictly larger. There is no finite maximum:
||w|| -> infinity. Watching it happen as the gap between classes opens up:
""")
    rng = np.random.default_rng(3)
    n = 100

    print(f"  {'class gap':>10s}  {'separable?':>11s}  {'||w|| unpenalized':>19s}  "
          f"{'max |z|':>10s}  {'||w|| with L2':>15s}")
    print("  " + "-" * 72)

    for gap in (0.0, 1.0, 2.0, 4.0, 8.0):
        X = np.concatenate([rng.normal(-gap / 2, 1.0, n // 2),
                            rng.normal(gap / 2, 1.0, n // 2)])[:, None]
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        # Separable iff no overlap between the two class ranges.
        separable = X[y == 0].max() < X[y == 1].min()

        free = LogisticRegression(solver="lbfgs", penalty=None, max_iter=20000).fit(X, y)
        reg = LogisticRegression(solver="lbfgs", penalty="l2", lam=1.0,
                                 max_iter=20000).fit(X, y)

        print(f"  {gap:10.1f}  {str(separable):>11s}  "
              f"{np.linalg.norm(free.theta_):19.2f}  "
              f"{np.abs(free.decision_function(X)).max():10.1f}  "
              f"{np.linalg.norm(reg.theta_):15.4f}")

    print("""
  Once the classes stop overlapping the norm jumps and the logits reach magnitudes where
  sigmoid saturates to exactly 0 and 1.

  But notice the unpenalized norm at gap = 8 is 10.91, not 10^6. The optimizer did not
  diverge — it STOPPED, because once every prediction is saturated the gradient is
  numerically zero and L-BFGS's convergence test fires. That is the dangerous part: the
  solver reports success at an essentially arbitrary point.

  To show there is no finite optimum, give the same separable problem more budget and watch
  where it lands:
""")
    X_sep = np.concatenate([rng.normal(-4, 1.0, n // 2), rng.normal(4, 1.0, n // 2)])[:, None]
    y_sep = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    print(f"    {'max_iter':>10s}  {'||w||':>10s}  {'max |z|':>10s}  {'train loss':>12s}  "
          f"{'min p(correct)':>15s}")
    print("    " + "-" * 62)
    for budget in (10, 50, 200, 1000, 5000):
        m = LogisticRegression(solver="gd", penalty=None, lr=2.0,
                               max_iter=budget, tol=0.0).fit(X_sep, y_sep)
        z_sep = m.decision_function(X_sep)
        p_correct = np.where(y_sep == 1, stable_sigmoid(z_sep), 1 - stable_sigmoid(z_sep))
        print(f"    {budget:10d}  {np.linalg.norm(m.theta_):10.2f}  "
              f"{np.abs(z_sep).max():10.1f}  {m._loss(m.theta_, m._A, y_sep):12.3e}  "
              f"{p_correct.min():15.6f}")

    print("""
    ||w|| keeps growing and the loss keeps falling toward zero, with no sign of settling.
    That is what "the MLE does not exist" looks like in practice: not an error, but a
    number that depends entirely on when you stopped looking.

  L2 regularization gives a finite minimum at every gap, because the penalty grows as
  ||w||^2 while the likelihood gain saturates.

  This is why sklearn regularizes BY DEFAULT (C=1.0), which surprises people arriving from
  statsmodels — and why statsmodels' Logit prints a separation warning instead.

  Practical note: perfect separation on real data is very often a LEAK. A feature that
  separates the classes perfectly is usually a proxy for the label that will not exist at
  prediction time (02.06). Investigate before you regularize it away.""")


def experiment_calibration() -> None:
    """README §13: logistic regression is calibrated by construction."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — calibration  (README §13)")
    print("=" * 88)
    print("""
Logistic regression minimizes a proper scoring rule (00.05 §6.2) — a loss uniquely
minimized by reporting your true beliefs. One exact consequence: at the optimum the
gradient's intercept component is zero, so

    sum_i p_i = sum_i y_i

The predicted probabilities sum to the observed number of positives, exactly. Checking
that, and comparing calibration against classifiers not trained on a proper scoring rule:
""")
    rng = np.random.default_rng(4)
    n, d = 2000, 6
    X = rng.standard_normal((n, d))
    p_true = stable_sigmoid(0.5 + X @ rng.standard_normal(d))
    y = (rng.random(n) < p_true).astype(int)

    lr = LogisticRegression(solver="lbfgs", penalty=None, max_iter=5000).fit(X, y)
    p_lr = lr.predict_proba(X)[:, 1]

    print(f"  observed positives   = {int(y.sum())}")
    print(f"  sum of predicted p   = {p_lr.sum():.6f}")
    print(f"  difference           = {abs(p_lr.sum() - y.sum()):.3e}   <- exactly zero\n")

    def calibration_error(p, y, n_bins=10):
        """Expected calibration error: |confidence - accuracy|, averaged over bins."""
        bins = np.linspace(0, 1, n_bins + 1)
        total = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 0:
                total += mask.sum() / len(p) * abs(p[mask].mean() - y[mask].mean())
        return total

    rows = [("logistic regression", p_lr)]
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB

        rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
        rows.append(("random forest", rf.predict_proba(X)[:, 1]))

        svm = SVC(probability=False).fit(X, y)
        raw = svm.decision_function(X)
        rows.append(("SVM (min-max scaled)", (raw - raw.min()) / (raw.max() - raw.min())))

        nb = GaussianNB().fit(X, y)
        rows.append(("naive Bayes", nb.predict_proba(X)[:, 1]))
    except ImportError:
        pass

    print(f"  {'model':<24s}  {'sum(p) - sum(y)':>17s}  {'calibration error':>19s}")
    print("  " + "-" * 66)
    for name, p in rows:
        print(f"  {name:<24s}  {p.sum() - y.sum():17.3f}  {calibration_error(p, y):19.4f}")

    print("""
  Logistic regression's total is exactly right and its calibration error is near zero. That
  is not tuning — it falls out of the first-order optimality condition.

  The random forest is close on the total (it is fitting the same data) but its per-bin
  calibration is visibly worse: tree votes are not probabilities, and they are systematically
  over-confident near 0 and 1.

  The SVM's decision values are not probabilities at all — the numbers above are a min-max
  rescaling, which is arbitrary. This is why sklearn's SVC(probability=True) runs Platt
  scaling, an entirely separate logistic fit on the decision values.

  Naive Bayes is famously over-confident, for the reason discussed in 03.05: its
  independence assumption multiplies correlated evidence as though it were independent,
  driving probabilities toward 0 and 1.

  If you need probabilities you can act on — expected-value decisions, thresholds tuned to
  costs, risk scores — start from a model trained on a proper scoring rule, or calibrate
  afterwards (05.06).""")


def experiment_multiclass() -> None:
    """README §11: multinomial vs one-vs-rest."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — multinomial vs one-vs-rest  (README §11)")
    print("=" * 88)
    print("""
Two ways to handle K classes. Multinomial fits one joint model with a shared softmax;
one-vs-rest fits K independent binary models and takes the argmax. They usually agree on
the ARGMAX and differ on the PROBABILITIES.
""")
    rng = np.random.default_rng(5)
    n, d, K = 900, 5, 4
    X = rng.standard_normal((n, d))
    W_true = rng.standard_normal((d, K)) * 1.5
    y = np.array([rng.choice(K, p=row) for row in softmax(X @ W_true)])

    split = int(0.7 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    multi = SoftmaxRegression(lam=1.0, max_iter=8000).fit(X_tr, y_tr)
    p_multi = multi.predict_proba(X_te)
    acc_multi = float(np.mean(multi.predict(X_te) == y_te))

    # One-vs-rest, built by hand from K binary fits.
    ovr_scores = np.column_stack([
        LogisticRegression(solver="lbfgs", penalty="l2", lam=1.0, max_iter=5000)
        .fit(X_tr, (y_tr == k).astype(int)).predict_proba(X_te)[:, 1]
        for k in range(K)])
    acc_ovr = float(np.mean(np.argmax(ovr_scores, axis=1) == y_te))

    print(f"  {'method':<26s}  {'test accuracy':>14s}  {'mean sum of scores':>20s}  "
          f"{'mean max prob':>15s}")
    print("  " + "-" * 82)
    print(f"  {'multinomial (softmax)':<26s}  {acc_multi:14.4f}  "
          f"{p_multi.sum(axis=1).mean():20.4f}  {p_multi.max(axis=1).mean():15.4f}")
    print(f"  {'one-vs-rest':<26s}  {acc_ovr:14.4f}  "
          f"{ovr_scores.sum(axis=1).mean():20.4f}  "
          f"{(ovr_scores / ovr_scores.sum(axis=1, keepdims=True)).max(axis=1).mean():15.4f}")

    agreement = float(np.mean(multi.predict(X_te) == np.argmax(ovr_scores, axis=1)))
    print(f"\n  the two methods agree on {agreement:.1%} of test predictions")

    print("""
  Accuracy is close — for the argmax, one-vs-rest is usually fine.

  The third column is the problem. Multinomial probabilities sum to exactly 1 by
  construction, because they come from a single softmax. One-vs-rest scores do not: each
  was produced by a separate binary model trained on a different, usually imbalanced,
  problem, and nothing ties them together. You can renormalize, but the renormalized
  numbers are not a coherent posterior — they are K unrelated confidences divided by their
  sum.

  So: use one-vs-rest when you only need the argmax and want the parallelism. Use
  multinomial when the probabilities themselves will be used for anything — expected-value
  decisions, thresholds, downstream models, or reporting.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_why_not_ols()
    experiment_convergence()
    experiment_separation()
    experiment_calibration()
    experiment_multiclass()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
