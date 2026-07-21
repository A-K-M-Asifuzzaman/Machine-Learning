"""
00.02 — Optimization from Scratch
=================================

Every optimizer in this file is implemented from its update rule, using NumPy only.
No autograd, no torch.optim — the point is that the equations in README.md and the code
here are the same object.

Implemented here
----------------
    numerical_gradient          finite differences, for checking analytic gradients
    check_gradient              the gradient check every ML engineer should know
    backtracking_line_search    Armijo condition
    gradient_descent            README §7
    momentum / nesterov         README §10
    adagrad / rmsprop           README §11
    adam / adamw                README §11
    newton                      README §12.1
    bfgs / lbfgs                README §12.2
    soft_threshold / ista       README §15  (the Lasso proximal operator)

Test problems
-------------
    Quadratic       — the only case with a closed-form answer, used to verify the
                      convergence-rate theory of §8 exactly
    Rosenbrock      — the classic ill-conditioned non-convex banana valley
    LogisticLoss    — a real ML objective, convex, with analytic gradient and Hessian

Run it
------
    python from_scratch.py

It verifies every optimizer against a known optimum, checks all analytic gradients against
finite differences, and runs four experiments that measure README.md's claims:
  1. The stability threshold eta < 2/lambda_max is exact, to the digit
  2. GD needs O(kappa) iterations; Nesterov needs O(sqrt(kappa))
  3. Adam's bias correction is not optional
  4. Soft thresholding produces exact zeros; gradient descent does not

Reference: README.md sections 7-15.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# =============================================================================
# GRADIENT CHECKING  (README §2)
# =============================================================================


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                       h: float = 1e-5) -> np.ndarray:
    """Central-difference approximation to the gradient.

        df/dx_i ~ [f(x + h e_i) - f(x - h e_i)] / (2h)

    The *central* difference has error O(h^2), against O(h) for the forward difference
    [f(x+h) - f(x)]/h — one extra function evaluation per coordinate buys an entire order
    of accuracy. The h = 1e-5 default balances truncation error (falls with h) against
    floating-point cancellation (grows as h shrinks); see 00.06.
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        step = np.zeros_like(x, dtype=float)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def check_gradient(f: Callable[[np.ndarray], float],
                   grad_f: Callable[[np.ndarray], np.ndarray],
                   x: np.ndarray, tol: float = 1e-6) -> tuple[bool, float]:
    """Compare an analytic gradient against finite differences.

    Uses *relative* error, since absolute error is meaningless without knowing the scale
    of the gradient:

        rel = ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)

    Anything below ~1e-7 is correct; above ~1e-4 is a bug. This check costs one line and
    catches the single most common class of error in hand-written ML code — write it
    before you trust any gradient you derived yourself.
    """
    analytic = np.asarray(grad_f(x), dtype=float)
    numeric = numerical_gradient(f, x)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denominator == 0.0:
        return True, 0.0
    rel_error = float(np.linalg.norm(analytic - numeric) / denominator)
    return rel_error < tol, rel_error


# =============================================================================
# TEST PROBLEMS
# =============================================================================


class Quadratic:
    """f(x) = 1/2 x^T H x - b^T x,  with H symmetric positive definite.

    The one problem where every quantity in README §8 is known exactly:
        minimum      x* = H^-1 b
        curvature    eigenvalues of H
        condition    kappa = lambda_max / lambda_min
        GD stability eta < 2 / lambda_max
        optimal step eta* = 2 / (lambda_min + lambda_max)

    Because all of these are computable, this class is what makes the convergence-rate
    experiments below *measurements* rather than illustrations.
    """

    def __init__(self, H: np.ndarray, b: np.ndarray | None = None):
        self.H = np.asarray(H, dtype=float)
        self.b = np.zeros(self.H.shape[0]) if b is None else np.asarray(b, dtype=float)
        self.eigenvalues = np.linalg.eigvalsh(self.H)

    def __call__(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.H @ x - self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x - self.b

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.H

    @property
    def optimum(self) -> np.ndarray:
        return np.linalg.solve(self.H, self.b)

    @property
    def condition_number(self) -> float:
        return float(self.eigenvalues[-1] / self.eigenvalues[0])

    @property
    def max_stable_lr(self) -> float:
        """eta < 2/lambda_max, from the descent lemma (README §7.1, §8)."""
        return float(2.0 / self.eigenvalues[-1])

    @property
    def optimal_lr(self) -> float:
        """eta* = 2/(lambda_min + lambda_max)  (README §8, point 3)."""
        return float(2.0 / (self.eigenvalues[0] + self.eigenvalues[-1]))


class Rosenbrock:
    """f(x, y) = (a - x)^2 + b(y - x^2)^2,  minimum f(a, a^2) = 0.

    The standard hard test case: a narrow, curved, banana-shaped valley. The floor of the
    valley is the parabola y = x^2, so the descent direction rotates continuously as you
    travel along it — punishing any method that assumes a fixed local geometry.

    Its Hessian at the optimum has kappa ~ 2500 for the usual b = 100, which by README §8
    predicts gradient descent will be roughly 2500x slower than it would be on a sphere.
    Experiment 2 measures exactly that.
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a, self.b = a, b

    def __call__(self, x: np.ndarray) -> float:
        return float((self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2),
            2 * self.b * (x[1] - x[0] ** 2),
        ])

    def hess(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            [2 - 4 * self.b * (x[1] - 3 * x[0] ** 2), -4 * self.b * x[0]],
            [-4 * self.b * x[0], 2 * self.b],
        ])

    @property
    def optimum(self) -> np.ndarray:
        return np.array([self.a, self.a ** 2])


class LogisticLoss:
    """Regularized logistic regression — a genuine ML objective.

        J(w) = -1/n sum_i [ y_i log s_i + (1-y_i) log(1-s_i) ] + lambda/2 ||w||^2
        s_i  = sigmoid(x_i^T w)

    Gradient and Hessian have clean closed forms:

        grad = 1/n X^T (s - y) + lambda w
        H    = 1/n X^T diag(s(1-s)) X + lambda I

    H is positive semidefinite because s(1-s) > 0 always, so X^T D X is a Gram matrix
    (00.01 §11.2). Adding lambda I makes it strictly positive definite — so the objective
    is strongly convex and has a unique minimum (README §6.4, §6.5).

    Note the use of a numerically stable log-sigmoid: computing log(sigmoid(z)) naively
    underflows to -inf for z < -745. See 00.06.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-2):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.l2 = l2
        self.n = self.X.shape[0]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Stable logistic sigmoid; branch on the sign to avoid overflow in exp."""
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def __call__(self, w: np.ndarray) -> float:
        z = self.X @ w
        # log(1 + exp(z)) computed stably as max(z,0) + log1p(exp(-|z|))
        log_terms = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
        nll = np.mean(log_terms - self.y * z)
        return float(nll + 0.5 * self.l2 * w @ w)

    def grad(self, w: np.ndarray) -> np.ndarray:
        s = self._sigmoid(self.X @ w)
        return self.X.T @ (s - self.y) / self.n + self.l2 * w

    def hess(self, w: np.ndarray) -> np.ndarray:
        s = self._sigmoid(self.X @ w)
        d = s * (1 - s)
        return (self.X.T * d) @ self.X / self.n + self.l2 * np.eye(w.size)


# =============================================================================
# LINE SEARCH  (README §7)
# =============================================================================


def backtracking_line_search(f, grad_f, x, direction, alpha0=1.0,
                             c=1e-4, rho=0.5, max_iter=60) -> float:
    """Find a step size satisfying the Armijo (sufficient decrease) condition:

        f(x + a*p) <= f(x) + c * a * grad_f(x)^T p

    The right-hand side is the first-order Taylor prediction, discounted by c. We are
    asking for at least a fraction c of the decrease the linear model promised. Start
    optimistic at alpha0 and halve until satisfied.

    This is how you avoid tuning a learning rate on deterministic problems, and it is why
    L-BFGS needs no learning rate (README §17). It does not transfer to stochastic
    settings, where f is only estimable up to noise.

    Armijo alone is enough for gradient descent and Newton, but NOT for quasi-Newton —
    see `wolfe_line_search` below for why.
    """
    fx = f(x)
    slope = grad_f(x) @ direction
    if slope >= 0:                       # not a descent direction; refuse to step far
        return 0.0
    alpha = alpha0
    for _ in range(max_iter):
        if f(x + alpha * direction) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


def wolfe_line_search(f, grad_f, x, direction, c1=1e-4, c2=0.9, max_iter=60) -> float:
    """Step size satisfying the *Wolfe* conditions — Armijo plus a curvature condition:

        (1) Armijo:     f(x + a*p) <= f(x) + c1 * a * g^T p        [sufficient decrease]
        (2) Curvature:  grad_f(x + a*p)^T p >= c2 * g^T p          [not too short a step]

    Why quasi-Newton needs condition (2). BFGS and L-BFGS only accept a curvature pair
    (s, y) when s^T y > 0 — otherwise the inverse-Hessian approximation would lose positive
    definiteness. Expand that quantity:

        s^T y = a * (grad_f(x + a*p)^T p  -  g^T p) = a * (dphi - dphi0)

    The Wolfe curvature condition says dphi >= c2 * dphi0, and since dphi0 < 0 and
    0 < c2 < 1 we have c2*dphi0 > dphi0, hence dphi > dphi0 and therefore s^T y > 0.
    **The Wolfe condition is exactly what guarantees the BFGS update is well-defined.**

    Backtracking-Armijo gives no such guarantee: it accepts any sufficiently-decreasing
    step, including very short ones where the gradient has barely changed. Those get
    rejected by the s^T y > 0 test, the curvature history stops updating, and L-BFGS
    silently degrades into steepest descent. On Rosenbrock that is the difference between
    converging in ~30 iterations and not converging at all.

    Implemented by bisection on a bracket, which is the short version of the standard
    bracket/zoom algorithm (Nocedal & Wright, Alg. 3.5-3.6).
    """
    phi0 = f(x)
    dphi0 = grad_f(x) @ direction
    if dphi0 >= 0:                       # not a descent direction
        return 0.0

    alpha_lo, alpha_hi = 0.0, np.inf
    alpha = 1.0                          # quasi-Newton methods want to try 1.0 first

    for _ in range(max_iter):
        if f(x + alpha * direction) > phi0 + c1 * alpha * dphi0:
            alpha_hi = alpha             # overshot: Armijo violated
        elif grad_f(x + alpha * direction) @ direction < c2 * dphi0:
            alpha_lo = alpha             # undershot: still descending steeply
        else:
            return alpha                 # both conditions hold

        alpha = 0.5 * (alpha_lo + alpha_hi) if np.isfinite(alpha_hi) else 2.0 * alpha

    return alpha_lo if alpha_lo > 0 else alpha


# =============================================================================
# FIRST-ORDER OPTIMIZERS  (README §7, §10, §11)
# =============================================================================


def _run(update, x0, grad_f, f, n_iter, tol):
    """Shared driver: iterate `update`, record the trajectory, stop when converged."""
    x = np.asarray(x0, dtype=float).copy()
    history = {"x": [x.copy()], "f": [f(x)], "grad_norm": [float(np.linalg.norm(grad_f(x)))]}

    for _ in range(n_iter):
        g = grad_f(x)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)):
            break                        # diverged; stop rather than propagate NaN
        if np.linalg.norm(g) < tol:
            break
        x = update(x, g)
        history["x"].append(x.copy())
        history["f"].append(f(x))
        history["grad_norm"].append(float(np.linalg.norm(grad_f(x))))

    history["x"] = np.array(history["x"])
    history["f"] = np.array(history["f"])
    history["grad_norm"] = np.array(history["grad_norm"])
    return x, history


def gradient_descent(f, grad_f, x0, lr=0.01, n_iter=1000, tol=1e-8, line_search=False):
    """theta <- theta - eta * grad f(theta)        README §7

    With line_search=True the step size is chosen by Armijo instead of fixed, which makes
    the method robust to a badly-chosen `lr` at the cost of extra function evaluations.
    """
    def update(x, g):
        step = backtracking_line_search(f, grad_f, x, -g) if line_search else lr
        return x - step * g
    return _run(update, x0, grad_f, f, n_iter, tol)


def momentum(f, grad_f, x0, lr=0.01, beta=0.9, n_iter=1000, tol=1e-8):
    """Polyak heavy ball.                          README §10

        v <- beta*v + g
        theta <- theta - eta*v

    v is an exponentially weighted sum of past gradients with horizon 1/(1-beta). On a
    consistently downhill stretch v -> g/(1-beta), so steps are up to 1/(1-beta) times
    larger than plain GD — 10x at beta=0.9.
    """
    v = np.zeros_like(np.asarray(x0, dtype=float))

    def update(x, g):
        nonlocal v
        v = beta * v + g
        return x - lr * v
    return _run(update, x0, grad_f, f, n_iter, tol)


def nesterov(f, grad_f, x0, lr=0.01, beta=0.9, n_iter=1000, tol=1e-8):
    """Nesterov accelerated gradient.              README §10

        v <- beta*v + grad f(theta - eta*beta*v)   <- gradient at the LOOKAHEAD point
        theta <- theta - eta*v

    Momentum will carry us to theta - eta*beta*v regardless, so we measure the slope
    there. The correction acts as a brake before overshooting, and improves the rate from
    O(kappa) to O(sqrt(kappa)) — which is optimal for any first-order method.

    Note this optimizer ignores the `g` supplied by the driver: it needs the gradient at a
    different point than the current iterate.
    """
    v = np.zeros_like(np.asarray(x0, dtype=float))

    def update(x, g):
        nonlocal v
        v = beta * v + grad_f(x - lr * beta * v)
        return x - lr * v
    return _run(update, x0, grad_f, f, n_iter, tol)


def adagrad(f, grad_f, x0, lr=0.1, eps=1e-8, n_iter=1000, tol=1e-8):
    """Accumulate squared gradients forever.       README §11

    Great for sparse features. Fatal flaw: s only grows, so the effective step size decays
    monotonically to zero and training stalls. RMSProp exists to fix exactly this.
    """
    s = np.zeros_like(np.asarray(x0, dtype=float))

    def update(x, g):
        nonlocal s
        s = s + g ** 2
        return x - lr * g / (np.sqrt(s) + eps)
    return _run(update, x0, grad_f, f, n_iter, tol)


def rmsprop(f, grad_f, x0, lr=0.01, rho=0.9, eps=1e-8, n_iter=1000, tol=1e-8):
    """AdaGrad with an exponential moving average, so old gradients are forgotten."""
    s = np.zeros_like(np.asarray(x0, dtype=float))

    def update(x, g):
        nonlocal s
        s = rho * s + (1 - rho) * g ** 2
        return x - lr * g / (np.sqrt(s) + eps)
    return _run(update, x0, grad_f, f, n_iter, tol)


def adam(f, grad_f, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
         n_iter=1000, tol=1e-8, bias_correction=True):
    """Adam = momentum (1st moment) + RMSProp (2nd moment) + bias correction.  README §11

    `bias_correction=False` is provided so Experiment 3 can measure what the correction is
    worth. Since m_0 = 0, the estimate m_t is biased low by exactly (1 - beta1^t) — a 10x
    underestimate on the first step at beta1=0.9. Skipping the correction makes the first
    few dozen steps far too small.
    """
    m = np.zeros_like(np.asarray(x0, dtype=float))
    v = np.zeros_like(np.asarray(x0, dtype=float))
    t = 0

    def update(x, g):
        nonlocal m, v, t
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        if bias_correction:
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
        else:
            m_hat, v_hat = m, v
        return x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return _run(update, x0, grad_f, f, n_iter, tol)


def adamw(f, grad_f, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
          weight_decay=0.01, n_iter=1000, tol=1e-8):
    """Adam with *decoupled* weight decay.         README §11

    The difference from "Adam + L2 penalty" is one line, and it matters: in vanilla Adam an
    L2 term enters through g, so it gets divided by sqrt(v_hat) — meaning parameters with
    large gradient history receive *less* regularization, which is backwards. Here the
    decay is applied straight to the weights, untouched by the adaptive scaling.
    """
    m = np.zeros_like(np.asarray(x0, dtype=float))
    v = np.zeros_like(np.asarray(x0, dtype=float))
    t = 0

    def update(x, g):
        nonlocal m, v, t
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        return x - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)
    return _run(update, x0, grad_f, f, n_iter, tol)


# =============================================================================
# SECOND-ORDER OPTIMIZERS  (README §12)
# =============================================================================


def newton(f, grad_f, hess_f, x0, n_iter=100, tol=1e-10, damping=1e-8, line_search=True):
    """theta <- theta - H^-1 grad f               README §12.1

    Solves the second-order Taylor model exactly. In the eigenbasis this divides each
    direction by its own curvature, making the effective condition number 1 — which is why
    it solves any quadratic in a single step.

    Two practical guards, both standard:
      - `damping` adds delta*I so H stays invertible and positive definite. Without it, at
        a saddle point H is indefinite and the "Newton step" moves toward the saddle
        (README §18).
      - `line_search` keeps the full step from overshooting where the quadratic model is
        a poor fit — the difference between "Newton's method" and "damped Newton".
    """
    x = np.asarray(x0, dtype=float).copy()
    history = {"x": [x.copy()], "f": [f(x)], "grad_norm": [float(np.linalg.norm(grad_f(x)))]}

    for _ in range(n_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x) + damping * np.eye(x.size)

        try:
            direction = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            direction = -g                               # fall back to steepest descent
        if g @ direction >= 0:                           # not a descent direction
            direction = -g

        step = backtracking_line_search(f, grad_f, x, direction) if line_search else 1.0
        x = x + step * direction
        history["x"].append(x.copy())
        history["f"].append(f(x))
        history["grad_norm"].append(float(np.linalg.norm(grad_f(x))))

    for k in history:
        history[k] = np.array(history[k])
    return x, history


def bfgs(f, grad_f, x0, n_iter=500, tol=1e-8):
    """BFGS: build an approximation B ~ H^-1 from gradient differences.  README §12.2

    Each step enforces the secant condition B_{t+1} y_t = s_t, where

        s = theta_{t+1} - theta_t        (where we moved)
        y = grad_{t+1} - grad_t          (how the gradient responded)

    The rank-2 update below is the unique one satisfying the secant condition while
    staying symmetric and positive definite (given curvature condition s^T y > 0), and
    being closest to the previous B in a weighted Frobenius sense.

    Cost O(d^2) in time and memory — fine for d in the thousands, hopeless for a network.
    """
    x = np.asarray(x0, dtype=float).copy()
    d = x.size
    B = np.eye(d)                                        # inverse-Hessian approximation
    g = grad_f(x)
    history = {"x": [x.copy()], "f": [f(x)], "grad_norm": [float(np.linalg.norm(g))]}

    for _ in range(n_iter):
        if np.linalg.norm(g) < tol:
            break

        direction = -B @ g
        if g @ direction >= 0:                           # lost positive definiteness
            B = np.eye(d)
            direction = -g

        step = wolfe_line_search(f, grad_f, x, direction)
        if step == 0.0:
            break

        s = step * direction
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g
        sy = s @ y

        if sy > 1e-10:                                   # curvature condition
            rho = 1.0 / sy
            I = np.eye(d)
            V = I - rho * np.outer(s, y)
            B = V @ B @ V.T + rho * np.outer(s, s)

        x, g = x_new, g_new
        history["x"].append(x.copy())
        history["f"].append(f(x))
        history["grad_norm"].append(float(np.linalg.norm(g)))

    for k in history:
        history[k] = np.array(history[k])
    return x, history


def lbfgs(f, grad_f, x0, m=10, n_iter=500, tol=1e-8):
    """L-BFGS: BFGS without ever forming B.        README §12.2

    Store only the last m pairs (s_i, y_i) and recover the action of B on a vector by the
    *two-loop recursion* below. Cost drops from O(d^2) to O(md) in both time and memory,
    which is what makes it usable at d = 10^5 and up.

    The recursion is BFGS's update applied m times to an initial diagonal guess, unrolled
    so that the intermediate matrices are never materialized. The scaling
    gamma = s^T y / y^T y is Barzilai-Borwein: a scalar estimate of the curvature, and the
    detail that makes L-BFGS competitive rather than merely cheap.

    This is sklearn's default solver for LogisticRegression.
    """
    x = np.asarray(x0, dtype=float).copy()
    g = grad_f(x)
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    history = {"x": [x.copy()], "f": [f(x)], "grad_norm": [float(np.linalg.norm(g))]}

    for _ in range(n_iter):
        if np.linalg.norm(g) < tol:
            break

        # --- two-loop recursion: compute direction = -B g -------------------
        q = g.copy()
        alphas = []
        for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
            rho_i = 1.0 / (y_i @ s_i)
            a_i = rho_i * (s_i @ q)
            alphas.append(a_i)
            q = q - a_i * y_i

        if s_hist:                                       # Barzilai-Borwein scaling
            s_last, y_last = s_hist[-1], y_hist[-1]
            q = q * ((s_last @ y_last) / (y_last @ y_last))

        for (s_i, y_i), a_i in zip(zip(s_hist, y_hist), reversed(alphas)):
            rho_i = 1.0 / (y_i @ s_i)
            b_i = rho_i * (y_i @ q)
            q = q + (a_i - b_i) * s_i

        direction = -q
        # --------------------------------------------------------------------

        if g @ direction >= 0:
            direction = -g

        step = wolfe_line_search(f, grad_f, x, direction)
        if step == 0.0:
            break

        s = step * direction
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g

        if s @ y > 1e-10:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:                          # keep only the last m
                s_hist.pop(0)
                y_hist.pop(0)

        x, g = x_new, g_new
        history["x"].append(x.copy())
        history["f"].append(f(x))
        history["grad_norm"].append(float(np.linalg.norm(g)))

    for k in history:
        history[k] = np.array(history[k])
    return x, history


# =============================================================================
# NON-SMOOTH: PROXIMAL METHODS  (README §15)
# =============================================================================


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """The proximal operator of lambda*||.||_1 — soft thresholding.  README §15

        S_t(v) = sign(v) * max(|v| - t, 0)

    Shrink toward zero by t, and *clamp to exactly zero* if that would cross. The clamp is
    the whole story of why L1 gives exact zeros while L2 only gives small values: L2's prox
    is v/(1+t), which is never exactly zero for v != 0.
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def ista(X, y, lam=0.1, lr=None, n_iter=1000, tol=1e-10):
    """Iterative Shrinkage-Thresholding — proximal gradient for Lasso.  README §15

        minimize  1/(2n) ||y - Xw||^2  +  lambda ||w||_1

    Split into smooth + non-smooth: gradient step on the least-squares part, then apply the
    prox of the L1 part. The non-differentiability at zero — which stops plain gradient
    descent dead — is handled exactly, in closed form.

    Step size defaults to 1/L where L = lambda_max(X^T X)/n is the Lipschitz constant of
    the smooth part's gradient, which is the largest provably-convergent choice (README §7.1).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    if lr is None:
        L = np.linalg.eigvalsh(X.T @ X).max() / n
        lr = 1.0 / L

    w = np.zeros(d)
    history = []
    for _ in range(n_iter):
        w_old = w.copy()
        grad = X.T @ (X @ w - y) / n
        w = soft_threshold(w - lr * grad, lr * lam)
        obj = 0.5 * np.mean((y - X @ w) ** 2) + lam * np.abs(w).sum()
        history.append(obj)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w, np.array(history)


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<48s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # --- analytic gradients vs finite differences -------------------------
    print("\nGradient checks — analytic vs central differences (README §2)")

    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    quad = Quadratic(H, np.array([1.0, 2.0]))
    rosen = Rosenbrock()

    X = rng.standard_normal((80, 4))
    true_w = rng.standard_normal(4)
    y_bin = (rng.random(80) < 1 / (1 + np.exp(-X @ true_w))).astype(float)
    logistic = LogisticLoss(X, y_bin, l2=1e-2)

    for label, prob, point in [
        ("Quadratic", quad, np.array([0.7, -1.3])),
        ("Rosenbrock", rosen, np.array([-1.2, 1.0])),
        ("LogisticLoss", logistic, rng.standard_normal(4) * 0.5),
    ]:
        passed, rel = check_gradient(prob, prob.grad, point)
        ok &= _report(f"{label}: gradient", rel, 1e-6)

    # --- optimizers reach the known optimum -------------------------------
    print("\nOptimizers on a quadratic — known optimum x* = H^-1 b (README §7-§12)")
    x0 = np.array([5.0, -4.0])
    x_star = quad.optimum
    lr = quad.optimal_lr

    for label, result in [
        ("gradient_descent", gradient_descent(quad, quad.grad, x0, lr=lr, n_iter=5000)),
        ("momentum", momentum(quad, quad.grad, x0, lr=lr * 0.3, n_iter=5000)),
        ("nesterov", nesterov(quad, quad.grad, x0, lr=lr * 0.3, n_iter=5000)),
        ("adagrad", adagrad(quad, quad.grad, x0, lr=0.5, n_iter=20000)),
        ("adam", adam(quad, quad.grad, x0, lr=0.05, n_iter=20000)),
        ("bfgs", bfgs(quad, quad.grad, x0)),
        ("lbfgs", lbfgs(quad, quad.grad, x0)),
        ("newton", newton(quad, quad.grad, quad.hess, x0)),
    ]:
        ok &= _report(f"{label} finds x*", float(np.linalg.norm(result[0] - x_star)), 1e-5)

    # RMSProp is held to a *different* standard, and the reason is instructive.
    # As g -> 0 the EMA s -> 0 at the same rate, so g/sqrt(s) -> sign(g): the step size
    # stops shrinking and settles at ~lr. RMSProp therefore orbits the optimum at radius
    # O(lr) forever on a deterministic problem, rather than converging to it.
    # Adam does not have this problem: its momentum term m averages the sign-flipping
    # gradients toward zero, so m_hat/sqrt(v_hat) -> 0 and it converges properly.
    # This is a real and under-appreciated difference between the two.
    rms_lr = 0.01
    rms_x, _ = rmsprop(quad, quad.grad, x0, lr=rms_lr, n_iter=20000)
    rms_err = float(np.linalg.norm(rms_x - x_star))
    ok &= _report(f"rmsprop orbits x* within O(lr={rms_lr})", rms_err, 10 * rms_lr)

    # --- Newton solves a quadratic in ONE step ----------------------------
    print("\nNewton's method on a quadratic (README §12.1)")
    # damping=0 so the step is the exact Newton step. With the default damping of 1e-8 the
    # solve uses H + 1e-8*I instead of H, leaving a ~1e-8 residual that needs a second
    # step to clear — a small but real reminder that damped Newton is not Newton.
    _, hist = newton(quad, quad.grad, quad.hess, x0, line_search=False, damping=0.0)
    n_steps = len(hist["f"]) - 1
    print(f"  [{'PASS' if n_steps == 1 else 'FAIL'}]  "
          f"{'solves a quadratic in exactly 1 step':<48s}  steps = {n_steps}")
    ok &= (n_steps == 1)

    _, hist_damped = newton(quad, quad.grad, quad.hess, x0, line_search=False, damping=1e-8)
    print(f"  [INFO]  {'the same solve with damping=1e-8':<48s}  "
          f"steps = {len(hist_damped['f']) - 1}")

    # --- optimizers on Rosenbrock -----------------------------------------
    print("\nOptimizers on Rosenbrock — optimum (1, 1), kappa ~ 2500 (README §8)")
    x0_r = np.array([-1.2, 1.0])
    for label, result in [
        ("bfgs", bfgs(rosen, rosen.grad, x0_r, n_iter=500)),
        ("lbfgs", lbfgs(rosen, rosen.grad, x0_r, n_iter=500)),
        ("newton", newton(rosen, rosen.grad, rosen.hess, x0_r, n_iter=200)),
    ]:
        ok &= _report(f"{label} finds (1, 1)",
                      float(np.linalg.norm(result[0] - rosen.optimum)), 1e-4)

    # --- logistic regression vs sklearn -----------------------------------
    print("\nLogistic regression: our L-BFGS vs scipy and sklearn")
    w_ours, _ = lbfgs(logistic, logistic.grad, np.zeros(4), n_iter=500, tol=1e-10)

    try:
        from scipy.optimize import minimize
        ref = minimize(logistic, np.zeros(4), jac=logistic.grad, method="L-BFGS-B",
                       options={"gtol": 1e-12, "ftol": 1e-15})
        ok &= _report("our lbfgs vs scipy L-BFGS-B",
                      float(np.linalg.norm(w_ours - ref.x)), 1e-5)
    except ImportError:
        print("  [SKIP]  scipy comparison (scipy not installed)")

    w_newton, _ = newton(logistic, logistic.grad, logistic.hess, np.zeros(4))
    ok &= _report("our newton agrees with our lbfgs",
                  float(np.linalg.norm(w_ours - w_newton)), 1e-5)

    # --- ISTA vs sklearn Lasso --------------------------------------------
    print("\nISTA vs sklearn Lasso (README §15)")
    X_l = rng.standard_normal((120, 10))
    w_true = np.array([3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y_l = X_l @ w_true + 0.1 * rng.standard_normal(120)

    w_ista, _ = ista(X_l, y_l, lam=0.1, n_iter=20000, tol=1e-14)
    try:
        from sklearn.linear_model import Lasso
        sk = Lasso(alpha=0.1, fit_intercept=False, max_iter=100000, tol=1e-14).fit(X_l, y_l)
        ok &= _report("ista vs sklearn Lasso coefficients",
                      float(np.abs(w_ista - sk.coef_).max()), 1e-4)
        same_support = np.array_equal(w_ista == 0, sk.coef_ == 0)
        print(f"  [{'PASS' if same_support else 'FAIL'}]  "
              f"{'ista recovers the same exact-zero support':<48s}  "
              f"nnz = {np.sum(w_ista != 0)} vs {np.sum(sk.coef_ != 0)}")
        ok &= same_support
    except ImportError:
        print("  [SKIP]  sklearn comparison (sklearn not installed)")

    return ok


# =============================================================================
# EXPERIMENTS — the README's claims, measured
# =============================================================================


def experiment_stability_threshold() -> None:
    """README §7.1 and §8: gradient descent converges iff eta < 2/lambda_max."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 — the learning-rate stability threshold  (README §7.1, §8)")
    print("=" * 80)
    print("""
Theory says gradient descent on a quadratic converges if and only if eta < 2/lambda_max,
with no grey area. Not "roughly", not "usually" — exactly. Sweeping eta across the
predicted threshold:
""")
    H = np.array([[10.0, 0.0], [0.0, 1.0]])              # lambda_max = 10 -> eta* = 0.2
    quad = Quadratic(H)
    x0 = np.array([1.0, 1.0])
    threshold = quad.max_stable_lr

    print(f"  lambda_max = {quad.eigenvalues[-1]:.1f}   =>   predicted threshold "
          f"eta = 2/lambda_max = {threshold:.4f}\n")
    print(f"  {'eta':>10s}  {'eta / threshold':>16s}  {'final f':>14s}  {'outcome':>12s}")
    print("  " + "-" * 58)

    for factor in (0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5):
        eta = threshold * factor
        _, hist = gradient_descent(quad, quad.grad, x0, lr=eta, n_iter=500, tol=0.0)
        final = hist["f"][-1]
        if not np.isfinite(final) or final > 1e6:
            outcome, shown = "DIVERGED", float("inf")
        elif final < 1e-10:
            outcome, shown = "converged", final
        else:
            outcome, shown = "oscillating", final
        print(f"  {eta:10.5f}  {factor:16.2f}  {shown:14.3e}  {outcome:>12s}")

    print("""
  The transition happens precisely at the ratio 1.00. At eta exactly 2/lambda_max the
  iteration neither converges nor diverges: the sharp direction has multiplier
  |1 - eta*lambda_max| = 1, so it oscillates forever at constant amplitude.

  This is the single most useful fact for debugging a diverging loss (README §17): your
  maximum usable learning rate is set by the *sharpest* curvature in the problem — which
  is why changing feature scaling can suddenly break a learning rate that worked.""")


def experiment_condition_number() -> None:
    """README §8 and §10: GD needs O(kappa) iterations; Nesterov needs O(sqrt(kappa))."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 — conditioning, and what momentum buys  (README §8, §10)")
    print("=" * 80)
    print("""
Theory predicts iterations-to-converge scale as kappa for gradient descent and sqrt(kappa)
for Nesterov momentum. Building quadratics with kappa from 10 to 10,000 and counting:
""")
    print(f"  {'kappa':>8s}  {'GD iters':>10s}  {'Nesterov':>10s}  "
          f"{'GD/kappa':>10s}  {'Nesterov/sqrt(kappa)':>21s}  {'speedup':>9s}")
    print("  " + "-" * 76)

    for kappa in (10, 100, 1000, 10000):
        H = np.diag([float(kappa), 1.0])
        quad = Quadratic(H)
        x0 = np.array([1.0, 1.0])
        target = 1e-10

        def iters_to_target(result):
            f_hist = result[1]["f"]
            below = np.where(f_hist < target)[0]
            return int(below[0]) if below.size else len(f_hist)

        gd = gradient_descent(quad, quad.grad, x0, lr=quad.optimal_lr,
                              n_iter=400000, tol=0.0)
        # For a quadratic, the classical optimal momentum parameter.
        beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        nag = nesterov(quad, quad.grad, x0, lr=1.0 / quad.eigenvalues[-1],
                       beta=beta, n_iter=400000, tol=0.0)

        n_gd, n_nag = iters_to_target(gd), iters_to_target(nag)
        print(f"  {kappa:8d}  {n_gd:10d}  {n_nag:10d}  {n_gd / kappa:10.2f}  "
              f"{n_nag / np.sqrt(kappa):21.2f}  {n_gd / max(n_nag, 1):8.1f}x")

    print("""
  Columns 4 and 5 are the test. Both stay roughly *constant* down the table, which is what
  it means for the iteration counts to scale as kappa and sqrt(kappa) respectively. If the
  theory were wrong, these columns would drift.

  The last column is the practical payoff: at kappa = 10,000, momentum is ~50x fewer
  iterations for one extra vector of memory. O(sqrt(kappa)) is also optimal — no method
  that sees only gradients can beat it (Nemirovski-Yudin).""")


def experiment_adam_bias_correction() -> None:
    """README §11: uncorrected Adam takes steps that are too LARGE, not too small."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 — what Adam's bias correction actually corrects  (README §11)")
    print("=" * 80)
    print("""
Both moments start at zero, so both are biased toward zero. The naive expectation is that
the biases cancel in the ratio m/sqrt(v) — or that uncorrected Adam moves too slowly.
Neither is right. Because beta2 = 0.999 decays much more slowly than beta1 = 0.9, the
second moment is the more underestimated of the two, and it sits under a square root in
the denominator. Theory predicts the step is inflated by

        (1 - beta1^t) / sqrt(1 - beta2^t)

To measure the bias and nothing else, we walk ONE trajectory and, at each step, compute
both step rules from the *same* (m, v) state. Comparing two separately-run trajectories
would not isolate the bias: after a few steps they sit at different points with different
gradients, so any difference confounds the bias with where each run happened to end up.
""")
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    H = np.array([[3.0, 0.5], [0.5, 2.0]])
    quad = Quadratic(H, np.array([1.0, -1.0]))
    x0 = np.array([4.0, 4.0])

    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    rows = {}

    for t in range(1, 501):
        g = quad.grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        step_corrected = lr * m_hat / (np.sqrt(v_hat) + eps)
        step_uncorrected = lr * m / (np.sqrt(v) + eps)     # same m, v — bias only
        rows[t] = (np.linalg.norm(step_uncorrected), np.linalg.norm(step_corrected))

        x = x - step_corrected                             # follow the corrected path

    print(f"  {'step':>6s}  {'predicted':>11s}  {'measured':>11s}  "
          f"{'corrected step':>16s}  {'uncorrected step':>18s}")
    print("  " + "-" * 68)

    max_err = 0.0
    for t in (1, 2, 5, 10, 25, 50, 100, 500):
        predicted = (1 - beta1 ** t) / np.sqrt(1 - beta2 ** t)
        norm_u, norm_c = rows[t]
        measured = norm_u / norm_c
        max_err = max(max_err, abs(measured - predicted) / predicted)
        print(f"  {t:6d}  {predicted:10.4f}x  {measured:10.4f}x  "
              f"{norm_c:16.6f}  {norm_u:18.6f}")

    print(f"""
  Predicted and measured agree to {max_err:.1e} relative error — the formula is exact, and
  the only residual is the epsilon in the denominator.

  Read the shape of the first column: the inflation *peaks around step 10* at ~6.5x, and is
  still above 1.5x at step 500. So uncorrected Adam does not stall — it bolts, hardest
  exactly when the model is least able to absorb a bad step.

  This is the opposite of the usual intuition ("the estimates start at zero, so the steps
  must start small"). Both moments are biased toward zero, but v is far more biased than m
  because beta2 = 0.999 decays 100x slower than beta1 = 0.9 — and v sits under a square
  root in the *denominator*.

  It is also a large part of why learning-rate warmup is standard for transformers: warmup
  suppresses the window where this bias is worst. Note that bias correction does not fully
  solve the problem, because the early second-moment estimate is also extremely *noisy*,
  and no amount of bias correction fixes variance.""")


def experiment_sparsity() -> None:
    """README §15: soft thresholding produces exact zeros; gradient descent does not."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 — why L1 gives exact zeros and L2 does not  (README §15)")
    print("=" * 80)
    print("""
Same data, same objective value target, two regularizers. The claim is not that L1 makes
coefficients *small* — it is that L1 makes them *exactly zero*, because its proximal
operator clamps. Counting exact zeros:
""")
    rng = np.random.default_rng(3)
    n, d = 150, 20
    X = rng.standard_normal((n, d))
    w_true = np.zeros(d)
    w_true[:4] = [3.0, -2.5, 2.0, -1.5]                  # only 4 of 20 features matter
    y = X @ w_true + 0.1 * rng.standard_normal(n)

    print(f"  True model: {np.sum(w_true != 0)} nonzero of {d} features\n")
    print(f"  {'lambda':>8s}  {'L1 nonzero':>12s}  {'L1 exact zeros':>16s}  "
          f"{'L2 nonzero':>12s}  {'L2 min |w|':>12s}")
    print("  " + "-" * 68)

    for lam in (0.01, 0.05, 0.1, 0.3):
        w_l1, _ = ista(X, y, lam=lam, n_iter=20000, tol=1e-14)

        # Ridge, closed form: (X^T X + n*lam*I)^-1 X^T y
        w_l2 = np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T @ y)

        print(f"  {lam:8.3f}  {np.sum(w_l1 != 0):12d}  {np.sum(w_l1 == 0):16d}  "
              f"{np.sum(w_l2 != 0):12d}  {np.abs(w_l2).min():12.2e}")

    print("""
  L2 never produces a single exact zero at any lambda — its smallest coefficient shrinks
  but stays nonzero, because its prox is w/(1+t), which is zero only if w already was.
  L1 zeroes out most of the irrelevant features, and at lambda = 0.05-0.1 recovers close
  to the true support of 4.

  This is the analytic counterpart to the geometric story in 00.01 §5.2: the L1 ball has
  corners on the axes, and soft thresholding is what reaching a corner looks like in code.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_stability_threshold()
    experiment_condition_number()
    experiment_adam_bias_correction()
    experiment_sparsity()

    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 80)
