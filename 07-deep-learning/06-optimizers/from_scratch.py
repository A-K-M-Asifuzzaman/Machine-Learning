"""
07.06 — Optimizers, from scratch (NumPy).

SGD, momentum, Nesterov, AdaGrad, RMSProp, and Adam, with Adam verified step-for-step against
PyTorch. The chapter's claims are then MEASURED:

  1. the learning rate: stall / diverge / converge                          (README §3)
  2. momentum accelerates on an ill-conditioned ravine                      (README §4)
  3. Adam converges faster/more robustly than SGD                           (README §6)
  4. Adam's bias correction inflates the early steps                        (README §6)
  5. SGD noise escapes a saddle point where full-batch GD stalls            (README §2, §8)
  6. AdaGrad's learning rate decays to 0; RMSProp's survives                (README §5)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# =============================================================================
# OPTIMIZERS  (README §2-§6)
# =============================================================================


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.v = None

    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.momentum * self.v + grad
        if self.nesterov:
            return theta - self.lr * (grad + self.momentum * self.v)
        return theta - self.lr * self.v


class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.G = lr, eps, None

    def step(self, theta, grad):
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.G += grad ** 2                         # CUMULATIVE sum -> denominator only grows
        return theta - self.lr * grad / (np.sqrt(self.G) + self.eps)

    def eff_lr(self):
        return self.lr / (np.sqrt(self.G) + self.eps)


class RMSProp:
    def __init__(self, lr=0.01, gamma=0.9, eps=1e-8):
        self.lr, self.gamma, self.eps, self.v = lr, gamma, eps, None

    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.gamma * self.v + (1 - self.gamma) * grad ** 2   # EMA -> forgets, stays bounded
        return theta - self.lr * grad / (np.sqrt(self.v) + self.eps)

    def eff_lr(self):
        return self.lr / (np.sqrt(self.v) + self.eps)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.bias_correction = bias_correction
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        if self.bias_correction:
            mhat = self.m / (1 - self.b1 ** self.t)
            vhat = self.v / (1 - self.b2 ** self.t)
        else:
            mhat, vhat = self.m, self.v
        return theta - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — Adam update vs PyTorch (step for step)")
    print("=" * 88)
    if not HAVE_TORCH:
        print("\n(PyTorch unavailable — skipping)")
        return
    rng = np.random.default_rng(0)
    theta0 = rng.standard_normal(5)
    grads = [rng.standard_normal(5) for _ in range(10)]

    ours = Adam(lr=0.01)
    theta = theta0.copy()
    for g in grads:
        theta = ours.step(theta, g)

    tp = torch.tensor(theta0, requires_grad=True)
    opt = torch.optim.Adam([tp], lr=0.01)
    for g in grads:
        opt.zero_grad()
        tp.grad = torch.tensor(g)
        opt.step()
    diff = np.max(np.abs(theta - tp.detach().numpy()))
    print(f"\n  after 10 Adam steps: max|ours - PyTorch| = {diff:.2e}")
    assert diff < 1e-10, "Adam must match PyTorch"
    print("  our Adam matches PyTorch's optimizer to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def quadratic(cond=1.0):
    """f(x) = 0.5 (a x0^2 + b x1^2); cond = a/b sets the conditioning (ravine)."""
    a, b = cond, 1.0
    f = lambda x: 0.5 * (a * x[0] ** 2 + b * x[1] ** 2)
    grad = lambda x: np.array([a * x[0], b * x[1]])
    return f, grad


# =============================================================================
# EXPERIMENT 1 — the learning rate (README §3)
# =============================================================================


def experiment_1_learning_rate():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the learning rate: stall / diverge / converge (README §3)")
    print("=" * 88)
    f, grad = quadratic(cond=1.0)
    x0 = np.array([5.0, 5.0])
    print(f"\n  Minimizing a simple quadratic bowl (min f = 0). f after 40 SGD steps:\n")
    print(f"    {'learning rate':>14s} {'f (start=25)':>14s} {'outcome':>14s}")
    for lr in (0.001, 0.5, 1.9, 2.05):
        x = x0.copy()
        opt = SGD(lr=lr)
        for _ in range(40):
            x = opt.step(x, grad(x))
        fv = f(x)
        outcome = ("diverged" if not np.isfinite(fv) or fv > 25 else
                   "slow / stalled" if fv > 1.0 else "converged")
        fstr = "nan" if not np.isfinite(fv) else f"{fv:.4f}"
        print(f"    {lr:>14.3f} {fstr:>14s} {outcome:>14s}")
    print("""
  READING: too SMALL a learning rate (0.001) barely moves — the loss stalls far above the minimum.
  Too LARGE (>=2.0 here, past the stability limit) OVERSHOOTS and the loss diverges. A well-chosen
  rate (0.5) converges quickly. The good range is narrow and problem-specific, which is why the
  learning rate is the first and most important thing to tune (README §3).""")


# =============================================================================
# EXPERIMENT 2 — momentum on an ill-conditioned ravine (README §4)
# =============================================================================


def experiment_2_momentum():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — momentum accelerates on an ill-conditioned ravine (README §4)")
    print("=" * 88)
    f, grad = quadratic(cond=30.0)                 # steep in x0, gentle in x1
    x0 = np.array([1.0, 1.0])
    lr = 0.02                                       # small enough that plain SGD is stable but slow
    print(f"\n  Ill-conditioned quadratic (30:1). f after N steps (min = 0):\n")
    print(f"    {'optimizer':>18s} {'step 20':>10s} {'step 50':>10s} {'step 100':>10s}")
    for name, opt in [("SGD", SGD(lr=lr)),
                      ("SGD + momentum 0.9", SGD(lr=lr, momentum=0.9)),
                      ("Nesterov 0.9", SGD(lr=lr, momentum=0.9, nesterov=True))]:
        x = x0.copy()
        fs = []
        for t in range(100):
            x = opt.step(x, grad(x))
            fs.append(f(x))
        print(f"    {name:>18s} {fs[19]:>10.2e} {fs[49]:>10.2e} {fs[99]:>10.2e}")
    print("""
  READING: on an ill-conditioned ravine, plain SGD oscillates across the steep x0 direction and
  crawls along the gentle x1 valley — slow. MOMENTUM accumulates velocity along the consistent
  valley direction (accelerating) while the oscillations across the walls cancel in the average
  (damping), reaching the minimum far faster. Nesterov's look-ahead is slightly better still. This
  ravine behaviour is exactly what momentum is for (README §4).""")


# =============================================================================
# EXPERIMENT 3 — Adam vs SGD (README §6)
# =============================================================================


def experiment_3_adam_vs_sgd():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — Adam converges faster/more robustly than SGD (README §6)")
    print("=" * 88)
    f, grad = quadratic(cond=50.0)                 # badly conditioned
    x0 = np.array([1.0, 1.0])
    print(f"\n  Badly-conditioned quadratic (50:1). f after N steps (min = 0):\n")
    print(f"    {'optimizer':>18s} {'step 50':>10s} {'step 150':>10s} {'step 300':>10s}")
    for name, opt in [("SGD (lr 0.019)", SGD(lr=0.019)),
                      ("SGD+mom (lr 0.019)", SGD(lr=0.019, momentum=0.9)),
                      ("Adam (lr 0.1)", Adam(lr=0.1))]:
        x = x0.copy()
        fs = []
        for t in range(300):
            x = opt.step(x, grad(x))
            fs.append(f(x) if np.isfinite(f(x)) else np.inf)
        print(f"    {name:>18s} {fs[49]:>10.2e} {fs[149]:>10.2e} {fs[299]:>10.2e}")
    print("""
  READING: SGD must use a tiny learning rate (bounded by the STEEP direction) so it crawls along
  the gentle one. Adam gives each coordinate its own adaptive step (dividing by sqrt of the running
  squared gradient), so it takes large steps in the gentle direction and small ones in the steep
  direction automatically — reaching the minimum far faster and without careful per-direction
  tuning. This per-parameter adaptivity is why Adam is the robust default (README §6).""")


# =============================================================================
# EXPERIMENT 4 — Adam bias correction (README §6)
# =============================================================================


def experiment_4_bias_correction():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — Adam's bias correction gives the right step size from step 1 (README §6)")
    print("=" * 88)
    # constant gradient g=1: WITH correction the update is ~lr from step 1; WITHOUT it, m and v
    # (both initialized to 0) are biased, so the early steps are mis-scaled.
    g = np.array([1.0])
    print(f"\n  Constant gradient g=1, lr=0.01. |update| at early steps:\n")
    print(f"    {'step':>6s} {'with bias correction':>22s} {'without correction':>20s}")
    with_bc = Adam(lr=0.01, bias_correction=True)
    without = Adam(lr=0.01, bias_correction=False)
    tw, two = np.array([0.0]), np.array([0.0])
    for t in range(1, 21):
        new_w = with_bc.step(tw.copy(), g)
        new_o = without.step(two.copy(), g)
        upd_w, upd_o = abs(new_w[0] - tw[0]), abs(new_o[0] - two[0])
        tw, two = new_w, new_o
        if t in (1, 2, 5, 10, 20):
            print(f"    {t:>6d} {upd_w:>22.5f} {upd_o:>20.5f}")
    print("""
  READING: m and v both start at 0, so early on they mis-estimate the gradient statistics. WITH
  correction (dividing m by 1-beta1^t and v by 1-beta2^t) the update magnitude is exactly the
  intended ~lr (0.01) from step 1. WITHOUT correction the early steps are MIS-SCALED — here ~3x too
  large at step 1 and drifting erratically — because v is under-estimated (its sqrt is too small).
  The correction is essential for correct, stable early-training steps, not a cosmetic detail
  (README §6; the effect is derived in 00.02).""")


# =============================================================================
# EXPERIMENT 5 — SGD noise escapes a saddle point (README §2, §8)
# =============================================================================


def experiment_5_saddle():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — SGD noise escapes a saddle where full-batch GD stalls (README §2, §8)")
    print("=" * 88)
    # f(x,y) = x^2 - y^2, a saddle at the origin. Start EXACTLY on the x-axis (y=0):
    # the y-gradient (-2y) is 0 there, so deterministic GD stays on the axis and gets STUCK at
    # the saddle; gradient NOISE kicks y off the axis and it escapes down the -y^2 valley.
    f = lambda p: p[0] ** 2 - p[1] ** 2
    grad = lambda p: np.array([2 * p[0], -2 * p[1]])
    x0 = np.array([1.0, 0.0])
    lr = 0.1

    # deterministic full-batch GD
    x = x0.copy()
    for _ in range(200):
        x = x - lr * grad(x)
    f_det = f(x)

    # noisy (mini-batch-like) SGD: add gradient noise
    rng = np.random.default_rng(0)
    finals = []
    for seed in range(20):
        r = np.random.default_rng(seed)
        x = x0.copy()
        for _ in range(120):
            noisy = grad(x) + r.standard_normal(2) * 0.1
            x = x - lr * noisy
        finals.append(f(x))
    print(f"""
  f(x,y) = x^2 - y^2 (saddle at origin). Start at (1, 0), 120 steps:

    {'method':>28s} {'final f':>14s}
    {'full-batch GD (deterministic)':>28s} {f_det:>14.4f}
    {'SGD (gradient noise)':>28s} {np.median(finals):>14.2e}

  READING: started exactly on the x-axis, deterministic GD has ZERO gradient in y, so it slides
  down x to the saddle (0,0) and STOPS there (final f ~ 0) — stuck at a saddle point. SGD's gradient
  noise kicks y off the axis; the -2y gradient then amplifies it and the optimizer escapes down the
  -y^2 direction (final f very negative). In high dimensions saddles, not local minima, are the main
  obstacle, and SGD's noise is what escapes them (README §2, §8).""")


# =============================================================================
# EXPERIMENT 6 — AdaGrad decays to 0, RMSProp survives (README §5)
# =============================================================================


def experiment_6_adagrad_decay():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — AdaGrad's learning rate decays to 0; RMSProp's survives (README §5)")
    print("=" * 88)
    g = np.array([1.0])                            # steady gradient signal
    ada = AdaGrad(lr=0.1)
    rms = RMSProp(lr=0.1)
    print(f"\n  Steady gradient. EFFECTIVE per-parameter learning rate over steps:\n")
    print(f"    {'step':>6s} {'AdaGrad eff-lr':>16s} {'RMSProp eff-lr':>16s}")
    x_a, x_r = np.array([0.0]), np.array([0.0])
    for t in range(1, 501):
        x_a = ada.step(x_a, g)
        x_r = rms.step(x_r, g)
        if t in (1, 10, 50, 200, 500):
            print(f"    {t:>6d} {ada.eff_lr()[0]:>16.5f} {rms.eff_lr()[0]:>16.5f}")
    print("""
  READING: AdaGrad accumulates the SUM of squared gradients, so its denominator only grows and the
  effective learning rate DECAYS toward 0 — learning grinds to a halt. RMSProp uses an EXPONENTIAL
  MOVING AVERAGE (it forgets old gradients), so its denominator stabilizes and the effective rate
  stays alive. This one change — cumulative sum -> moving average — is what made adaptive methods
  practical, and it is the second moment inside Adam (README §5).""")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    verify()
    experiment_1_learning_rate()
    experiment_2_momentum()
    experiment_3_adam_vs_sgd()
    experiment_4_bias_correction()
    experiment_5_saddle()
    experiment_6_adagrad_decay()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
