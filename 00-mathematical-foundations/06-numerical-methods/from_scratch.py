"""
00.06 — Numerical Methods from Scratch
======================================

Naive and stable implementations of the same mathematics, side by side, so you can see
exactly where the naive one breaks and by how much.

Every "naive" function here is the formula as it appears in a textbook. Every "stable"
function computes the same quantity and is what your framework actually runs. The gap
between them is this chapter.

Implemented here
----------------
    naive_softmax / stable_softmax / log_softmax        README §8
    naive_logsumexp / logsumexp                         README §7
    naive_sigmoid / stable_sigmoid                      README §9
    naive_bce / bce_with_logits / softplus              README §9
    variance_naive / variance_two_pass / variance_welford   README §10
    naive_sum / kahan_sum                               README §11
    numerical_gradient / check_gradient                 README §13

Run it
------
    python from_scratch.py

Verifies every stable version against its naive counterpart where the naive one still
works, and against scipy/torch where available. Then six experiments:
  1. Float anatomy: why 0.1 + 0.2 != 0.3, and where the gaps are
  2. Softmax overflow: the exact logit at which the naive version dies
  3. Binary cross-entropy: where log(sigmoid(z)) becomes -inf
  4. Variance: the naive formula returning a NEGATIVE number
  5. Summation: naive vs pairwise vs Kahan vs exact on 10 million floats
  6. Gradient checking: the U-shaped error curve, and the optimal h

Reference: README.md sections 2-14.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# =============================================================================
# LOG-SUM-EXP AND SOFTMAX  (README §7, §8)
# =============================================================================


def naive_logsumexp(x: np.ndarray) -> float:
    """log(sum(exp(x))) — straight from the formula. Overflows for x > ~709 (float64)."""
    return float(np.log(np.sum(np.exp(np.asarray(x, dtype=float)))))


def logsumexp(x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """Stable log-sum-exp.  README §7

        log sum_i exp(x_i) = c + log sum_i exp(x_i - c)      for any c

    Taking c = max(x) makes the largest exponent exp(0) = 1, so:
      - nothing can overflow (every exponent is <= 0)
      - underflow is harmless (those terms were negligible anyway)
      - at least one term equals 1, so the log never sees zero

    This is the single most important numerical routine in machine learning: softmax,
    cross-entropy, HMM forward-backward, mixture likelihoods, and every `logaddexp` in
    your framework are all built on it.
    """
    x = np.asarray(x, dtype=float)
    c = np.max(x, axis=axis, keepdims=True)
    # Guard the all-(-inf) case, where c = -inf and x - c would be nan.
    c = np.where(np.isfinite(c), c, 0.0)
    out = c + np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    elif not keepdims and axis is None:
        out = out.reshape(())
    return out


def naive_softmax(x: np.ndarray) -> np.ndarray:
    """exp(x) / sum(exp(x)) — dies for logits above ~709 (float64) or ~88 (float32)."""
    e = np.exp(np.asarray(x, dtype=float))
    return e / e.sum()


def stable_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """Softmax with the max subtracted.  README §8

    Softmax is EXACTLY shift-invariant — multiplying numerator and denominator by exp(-c)
    changes nothing algebraically. So this is not an approximation that trades accuracy
    for stability; it is the same function, computed in an order that cannot overflow.
    """
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """log(softmax(x)) computed DIRECTLY as x - logsumexp(x).  README §8.1

    Never compute log(stable_softmax(x)). A probability that underflowed to 0 becomes
    -inf, destroying the loss — even though x_i - LSE(x) would have returned a perfectly
    usable finite number like -800.

    This is why nn.CrossEntropyLoss takes raw LOGITS and fuses log_softmax with nll_loss
    internally, and why feeding it softmax output is a bug.
    """
    x = np.asarray(x, dtype=float)
    c = np.max(x, axis=axis, keepdims=True)
    shifted = x - c
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


# =============================================================================
# SIGMOID AND BINARY CROSS-ENTROPY  (README §9)
# =============================================================================


def naive_sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-z)) — overflows in exp(-z) for large negative z."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Branch on the sign so the exponent is always negative.  README §9

        z >= 0:  1 / (1 + exp(-z))
        z <  0:  exp(z) / (1 + exp(z))

    Both branches evaluate only exp(negative), which underflows gracefully to 0 instead of
    overflowing to inf.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    """log(1 + exp(z)), computed stably as max(z,0) + log1p(exp(-|z|)).  README §9

    For z = 800 the naive form overflows; this returns 800.0, which is correct to every
    digit float64 has.
    """
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


def naive_bce(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-[y log(sigma(z)) + (1-y) log(1 - sigma(z))] — fails once sigma saturates.

    In float32 sigma(z) rounds to exactly 1.0 at around z = 37, an entirely ordinary logit
    for a confident model. Then log(1 - 1) = log(0) = -inf.
    """
    p = naive_sigmoid(z)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary cross-entropy with the sigmoid folded in.  README §9

        L = max(z, 0) - z*y + log(1 + exp(-|z|))

    Derivation: substituting sigma into the naive form and simplifying gives
    z - z*y + log(1 + exp(-z)); applying the max(z,0) shift to the softplus term makes
    every exponent negative.

    Bounded at every z — no log(0), no overflow. This is exactly what
    torch.nn.BCEWithLogitsLoss computes, and why it exists separately from BCELoss.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


# =============================================================================
# VARIANCE  (README §10)
# =============================================================================


def variance_naive(x: np.ndarray) -> float:
    """E[X^2] - (E[X])^2 — the textbook shortcut, and a cancellation trap.

    When mu >> sigma (timestamps, IDs, prices in cents), both terms are huge and nearly
    equal. Their difference can come out NEGATIVE, and the next sqrt gives NaN.
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean(x ** 2) - np.mean(x) ** 2)


def variance_two_pass(x: np.ndarray) -> float:
    """mean((x - mean(x))^2) — one extra pass, no cancellation."""
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - np.mean(x)) ** 2))


def variance_welford(x: np.ndarray) -> tuple[float, float]:
    """Welford's online algorithm — one pass AND stable.  README §10

        delta = x_n - mu_{n-1}
        mu_n  = mu_{n-1} + delta / n
        M2_n  = M2_{n-1} + delta * (x_n - mu_n)

    Never forms sum(x^2), so there is nothing large to cancel. Returns (mean, variance).

    This is what BatchNorm implementations and pandas' .var() use, and the reason
    centering your data is a numerical fix as well as a statistical one.
    """
    mean = 0.0
    m2 = 0.0
    for n, value in enumerate(np.asarray(x, dtype=float), start=1):
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)      # note: uses the UPDATED mean
    n = len(x)
    return mean, (m2 / n if n > 0 else 0.0)


# =============================================================================
# SUMMATION  (README §11)
# =============================================================================


def naive_sum(x: np.ndarray) -> float:
    """Sequential accumulation. Error grows as O(n * eps).

    Once the running total is large, small addends fall below its precision and vanish
    entirely — the sum literally stops growing.
    """
    total = 0.0
    for value in x:
        total += value
    return total


def kahan_sum(x: np.ndarray) -> float:
    """Kahan compensated summation. Error is O(eps), independent of n.  README §11

    `c` carries the low-order bits that were lost in the previous addition, and they are
    added back before the next one. The seemingly redundant `(t - sum) - y` is what
    recovers them — do not let an optimizer "simplify" it.
    """
    total = 0.0
    c = 0.0
    for value in x:
        y = value - c              # apply the correction carried from last time
        t = total + y              # this addition loses the low-order bits of y ...
        c = (t - total) - y        # ... and this recovers exactly what was lost
        total = t
    return total


# =============================================================================
# GRADIENT CHECKING  (README §13)
# =============================================================================


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                       h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient. Error O(h^2), vs O(h) for a forward difference."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        step = np.zeros_like(x, dtype=float)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def check_gradient(f, grad_f, x: np.ndarray, h: float = 1e-5) -> float:
    """Relative error between an analytic and a numerical gradient.  README §13

    Relative, not absolute — an absolute threshold is meaningless without knowing the
    gradient's scale. Below 1e-7 is correct; above 1e-4 is a bug; in between, suspect a
    kink (ReLU, abs, max) or the wrong h.
    """
    analytic = np.asarray(grad_f(x), dtype=float)
    numeric = numerical_gradient(f, x, h)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    return 0.0 if denominator == 0 else float(
        np.linalg.norm(analytic - numeric) / denominator)


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

    x_safe = rng.standard_normal(50) * 3          # small enough that naive still works

    print("\nStable versions agree with naive ones where naive still works")
    ok &= _report("logsumexp vs naive_logsumexp",
                  abs(float(logsumexp(x_safe)) - naive_logsumexp(x_safe)), 1e-12)
    ok &= _report("stable_softmax vs naive_softmax",
                  float(np.abs(stable_softmax(x_safe) - naive_softmax(x_safe)).max()), 1e-15)
    ok &= _report("stable_sigmoid vs naive_sigmoid",
                  float(np.abs(stable_sigmoid(x_safe) - naive_sigmoid(x_safe)).max()), 1e-15)
    ok &= _report("log_softmax vs log(stable_softmax)",
                  float(np.abs(log_softmax(x_safe) - np.log(stable_softmax(x_safe))).max()),
                  1e-13)

    y = (rng.random(50) < 0.5).astype(float)
    ok &= _report("bce_with_logits vs naive_bce",
                  float(np.abs(bce_with_logits(x_safe, y) - naive_bce(x_safe, y)).max()), 1e-12)
    ok &= _report("softplus vs log(1 + exp(z))",
                  float(np.abs(softplus(x_safe) - np.log1p(np.exp(x_safe))).max()), 1e-13)

    print("\nStable versions still work where naive ones do NOT")
    x_huge = np.array([1000.0, 1001.0, 1002.0])
    print(f"  [INFO]  {'naive_logsumexp([1000, 1001, 1002])':<54s}  {naive_logsumexp(x_huge)}")
    print(f"  [INFO]  {'logsumexp([1000, 1001, 1002])':<54s}  {float(logsumexp(x_huge)):.4f}")
    ok &= np.isinf(naive_logsumexp(x_huge)) and np.isfinite(float(logsumexp(x_huge)))

    # Softmax is shift-invariant, so the answer for [1000,1001,1002] must equal [0,1,2].
    ok &= _report("softmax([1000,1001,1002]) == softmax([0,1,2])",
                  float(np.abs(stable_softmax(x_huge)
                               - stable_softmax(np.array([0.0, 1.0, 2.0]))).max()), 1e-15)
    ok &= _report("stable_softmax sums to 1 at extreme logits",
                  abs(float(stable_softmax(x_huge).sum()) - 1.0), 1e-15)

    extreme = np.array([-800.0, 800.0])
    ok &= bool(np.all(np.isfinite(stable_sigmoid(extreme))))
    ok &= _report("stable_sigmoid(+/-800) is finite and saturates correctly",
                  float(np.abs(stable_sigmoid(extreme) - np.array([0.0, 1.0])).max()), 1e-15)
    ok &= _report("softplus(800) = 800 (naive overflows)",
                  abs(float(softplus(np.array([800.0]))[0]) - 800.0), 1e-10)
    ok &= _report("bce_with_logits finite at z = +/-800",
                  float(np.max(np.abs(bce_with_logits(np.array([-800.0, 800.0]),
                                                      np.array([1.0, 0.0]))) - 800.0)), 1e-9)

    print("\nAgainst scipy / torch")
    try:
        from scipy.special import logsumexp as scipy_lse, softmax as scipy_softmax
        ok &= _report("logsumexp vs scipy.special.logsumexp",
                      abs(float(logsumexp(x_safe)) - float(scipy_lse(x_safe))), 1e-12)
        ok &= _report("logsumexp vs scipy at extreme values",
                      abs(float(logsumexp(x_huge)) - float(scipy_lse(x_huge))), 1e-9)
        ok &= _report("stable_softmax vs scipy.special.softmax",
                      float(np.abs(stable_softmax(x_safe) - scipy_softmax(x_safe)).max()), 1e-15)
    except ImportError:
        print("  [SKIP]  scipy not installed")

    try:
        import torch
        z = torch.tensor(x_safe)
        ok &= _report("bce_with_logits vs torch BCEWithLogitsLoss",
                      float(np.abs(bce_with_logits(x_safe, y)
                                   - torch.nn.functional.binary_cross_entropy_with_logits(
                                       z, torch.tensor(y), reduction="none").numpy()).max()),
                      1e-12)
        ok &= _report("log_softmax vs torch.log_softmax",
                      float(np.abs(log_softmax(x_safe)
                                   - torch.log_softmax(z, dim=-1).numpy()).max()), 1e-12)
    except ImportError:
        print("  [SKIP]  torch not installed")

    print("\nVariance (README §10)")
    data = rng.standard_normal(1000) * 2 + 5
    ok &= _report("variance_two_pass vs np.var",
                  abs(variance_two_pass(data) - float(np.var(data))), 1e-12)
    mean_w, var_w = variance_welford(data)
    ok &= _report("variance_welford vs np.var", abs(var_w - float(np.var(data))), 1e-10)
    ok &= _report("welford mean vs np.mean", abs(mean_w - float(np.mean(data))), 1e-12)

    print("\nSummation (README §11)")
    values = rng.random(100_000)
    exact = math.fsum(values)
    ok &= _report("kahan_sum vs math.fsum", abs(kahan_sum(values) - exact), 1e-12)

    print("\nGradient checking (README §13)")
    ok &= _report("check_gradient on a correct gradient",
                  check_gradient(lambda v: float(v @ v), lambda v: 2 * v,
                                 rng.standard_normal(8)), 1e-8)
    bad = check_gradient(lambda v: float(v @ v), lambda v: 2.1 * v, rng.standard_normal(8))
    print(f"  [{'PASS' if bad > 1e-4 else 'FAIL'}]  "
          f"{'check_gradient CATCHES a 5% wrong gradient':<54s}  err = {bad:.3e}")
    ok &= bad > 1e-4

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_float_anatomy() -> None:
    """README §2-§3: what a float can and cannot represent."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 1 — what a float actually is  (README §2-§3)")
    print("=" * 86)
    print("""
Floats are logarithmically spaced: the gap between representable numbers grows with
magnitude. Almost every surprise in this chapter follows from that one fact.
""")
    print(f"  0.1 + 0.2 == 0.3  ->  {0.1 + 0.2 == 0.3}")
    print(f"  0.1 + 0.2          =  {0.1 + 0.2!r}")
    print(f"  0.3                =  {0.3!r}")
    print(f"  difference         =  {abs((0.1 + 0.2) - 0.3):.3e}")
    print(f"  np.isclose         ->  {np.isclose(0.1 + 0.2, 0.3)}   <- use this\n")

    print("  Gap between consecutive representable float64 values:")
    print(f"    {'near':>12s}  {'gap':>12s}  {'x + 1 == x?':>13s}")
    print("    " + "-" * 40)
    for magnitude in (1.0, 1e3, 1e8, 1e15, 1e16, 1e17):
        print(f"    {magnitude:12.0e}  {np.spacing(magnitude):12.3e}  "
              f"{str(magnitude + 1 == magnitude):>13s}")

    print("""
  Above 1e16 the gap exceeds 1.0, so adding 1 changes nothing. This is why accumulating a
  counter in a float, or storing a nanosecond timestamp as float32, silently loses data.
""")
    print("  Machine epsilon and range by dtype:")
    print(f"    {'dtype':<10s}  {'epsilon':>11s}  {'max':>11s}  {'smallest normal':>17s}")
    print("    " + "-" * 54)
    for dtype in (np.float64, np.float32, np.float16):
        info = np.finfo(dtype)
        print(f"    {dtype.__name__:<10s}  {info.eps:11.3e}  {info.max:11.3e}  "
              f"{info.tiny:17.3e}")

    print(f"""
  In float16, eps = {np.finfo(np.float16).eps:.2e}. So 1 + 0.0005 rounds back to exactly 1:
    float16(1.0) + float16(0.0005) = {np.float16(1.0) + np.float16(0.0005)}

  A typical SGD update eta*g is many orders of magnitude smaller than the weight it
  updates. In pure float16 the subtraction therefore does NOTHING — the gradient is
  computed correctly, applied, and rounded away. That is precisely why mixed-precision
  training keeps a float32 master copy of the weights (README §14).""")


def experiment_softmax_overflow() -> None:
    """README §8: find the exact logit at which naive softmax dies."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 2 — where naive softmax dies  (README §7-§8)")
    print("=" * 86)
    print("""
Logits grow during training. Below, the same two-element softmax computed both ways as the
logit magnitude increases:
""")
    print(f"  {'max logit':>10s}  {'naive softmax':>28s}  {'stable softmax':>28s}")
    print("  " + "-" * 70)

    for scale in (1, 10, 100, 500, 700, 710, 1000):
        x = np.array([float(scale), float(scale) + 1.0])
        with np.errstate(over="ignore", invalid="ignore"):
            naive = naive_softmax(x)
        stable = stable_softmax(x)
        naive_str = f"[{naive[0]:.4f}, {naive[1]:.4f}]" if np.all(np.isfinite(naive)) \
            else f"[{naive[0]}, {naive[1]}]"
        print(f"  {scale:10d}  {naive_str:>28s}  "
              f"{f'[{stable[0]:.4f}, {stable[1]:.4f}]':>28s}")

    print(f"""
  The naive version returns nan from logit ~710 onward, because exp(710) overflows float64.
  In float32 the threshold is only ~88 — well inside the range real logits reach.

  The stable version returns [0.2689, 0.7311] at EVERY scale, which is correct: softmax
  depends only on logit DIFFERENCES, and the difference is 1.0 in every row. That is not
  an approximation — softmax is exactly shift-invariant (README §8).

  And the reason never to compute log(softmax(x)) separately:
""")
    x_spread = np.array([0.0, -800.0])
    with np.errstate(divide="ignore"):
        via_log_of_softmax = np.log(stable_softmax(x_spread))
    print(f"    log(stable_softmax([0, -800])) = {via_log_of_softmax}")
    print(f"    log_softmax([0, -800])         = {log_softmax(x_spread)}")
    print("""
  The first underflowed to 0 and then took its log, giving -inf and destroying the loss.
  The second computed x - logsumexp(x) directly and returned a perfectly usable -800.
  This is why nn.CrossEntropyLoss takes raw logits and fuses the two operations.""")


def experiment_bce_saturation() -> None:
    """README §9: where log(sigmoid(z)) becomes -inf."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 3 — binary cross-entropy saturation  (README §9)")
    print("=" * 86)
    print("""
A confident model produces large-magnitude logits. Once sigmoid saturates to exactly 0 or
1, the naive loss takes log(0). Below: loss for a CONFIDENTLY WRONG prediction (y=0 with a
large positive logit), which is exactly the case that should produce a big gradient:
""")
    print(f"  {'logit z':>9s}  {'sigmoid(z)':>14s}  {'naive BCE':>14s}  {'BCEWithLogits':>15s}")
    print("  " + "-" * 58)

    for z_val in (10.0, 20.0, 30.0, 36.0, 37.0, 40.0, 100.0, 800.0):
        z = np.array([z_val])
        y0 = np.array([0.0])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            naive = naive_bce(z, y0)[0]
        stable = bce_with_logits(z, y0)[0]
        p = float(stable_sigmoid(z)[0])
        naive_str = f"{naive:.6f}" if np.isfinite(naive) else str(naive)
        print(f"  {z_val:9.1f}  {p:14.12f}  {naive_str:>14s}  {stable:15.6f}")

    print("""
  In float64 sigmoid(z) rounds to exactly 1.0 near z = 37, and the naive loss becomes inf
  from there on. In float32 it happens around z = 17 — a completely ordinary logit.

  The stable version returns z itself for large z, which is the correct answer: the loss
  for being confidently wrong grows LINEARLY in the logit, and its gradient stays a
  well-behaved 1.0. The naive version returns inf, whose gradient is nan, and one nan
  poisons every parameter it touches on the backward pass.

  This is the whole reason BCEWithLogitsLoss exists as a separate class from
  Sigmoid + BCELoss.""")


def experiment_variance_cancellation() -> None:
    """README §10: the naive variance formula returning a negative number."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 4 — catastrophic cancellation in variance  (README §4, §10)")
    print("=" * 86)
    print("""
The identity Var(X) = E[X^2] - (E[X])^2 is exactly correct in real arithmetic. In floating
point it subtracts two huge nearly-equal numbers. Here the data always has true variance
1.0; only the OFFSET changes — a shift that mathematically cannot affect the answer:
""")
    rng = np.random.default_rng(0)
    base = rng.standard_normal(10_000)

    print(f"  {'offset':>10s}  {'naive':>18s}  {'two-pass':>14s}  {'Welford':>14s}  "
          f"{'np.var':>14s}")
    print("  " + "-" * 76)

    for offset in (0.0, 1e3, 1e6, 1e7, 1e8, 1e9):
        x = base + offset
        naive = variance_naive(x)
        naive_str = f"{naive:.8f}" if abs(naive) < 1e6 else f"{naive:.3e}"
        print(f"  {offset:10.0e}  {naive_str:>18s}  {variance_two_pass(x):14.8f}  "
              f"{variance_welford(x)[1]:14.8f}  {float(np.var(x)):14.8f}")

    # Whether the naive formula lands on 0 or overshoots into negative territory depends
    # on the particular rounding, so it varies with the data. Both outcomes are fatal;
    # search a few seeds to exhibit the negative one concretely rather than claim it.
    print("\n  The failure is rounding-dependent — some datasets collapse to 0, others "
          "overshoot.\n  Searching seeds at offset 1e9 for a NEGATIVE result:")
    found = False
    for seed in range(8):
        sample = np.random.default_rng(seed).standard_normal(10_000) + 1e9
        naive = variance_naive(sample)
        if naive < 0:
            print(f"    seed {seed}:  naive = {naive:.6e}   <- NEGATIVE variance, "
                  f"sqrt() of this is nan")
            print(f"              Welford = {variance_welford(sample)[1]:.8f}  (correct)")
            found = True
            break
    if not found:
        print("    (no negative result in the first 8 seeds on this platform)")

    print("""
  The true variance is 1.0 in every row. The naive formula starts losing digits around
  offset 1e6, and by 1e8 it has lost ALL of them — every significant digit of the answer
  was cancelled away, leaving 0. A variance of exactly 0 is just as fatal as a negative
  one: standardizing by it divides by zero.

  And as the seed search shows, the error is not even one-sided. Depending on which way
  the rounding falls, the naive formula can return a NEGATIVE variance — after which
  sqrt() gives nan and the nan propagates through every downstream parameter.

  Note how ordinary the offending data is. Unix timestamps are ~1.7e9. Prices in cents,
  ID numbers, sensor readings with a DC offset — all sit in this range. Computing variance
  on a raw timestamp feature can hand you a nan before training starts.

  Two-pass and Welford are exact throughout. Welford does it in ONE pass, which is why
  BatchNorm and streaming statistics use it. And centering your features first is a
  numerical fix as much as a statistical one.""")


def experiment_summation() -> None:
    """README §11: summation error over many terms."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 5 — summation error  (README §11)")
    print("=" * 86)
    print("""
Adding n floats sequentially accumulates O(n*eps) error: once the running total is large,
small addends fall below its precision and vanish. Summing 1,000,000 values, each ~1e-8,
against the exactly-rounded answer from math.fsum:
""")
    rng = np.random.default_rng(1)
    values = rng.random(1_000_000) * 1e-8

    exact = math.fsum(values)
    results = [
        ("naive Python loop", naive_sum(values)),
        ("np.sum (pairwise)", float(np.sum(values))),
        ("Kahan compensated", kahan_sum(values)),
        ("math.fsum (exact)", exact),
    ]

    print(f"  {'method':<22s}  {'result':>22s}  {'abs error':>12s}  {'rel error':>12s}")
    print("  " + "-" * 74)
    for name, value in results:
        err = abs(value - exact)
        print(f"  {name:<22s}  {value:22.16f}  {err:12.3e}  {err / abs(exact):12.3e}")

    # A case designed to break naive summation completely.
    print("\n  A pathological case — one large value followed by a million tiny ones:")
    adversarial = np.concatenate([[1e16], np.ones(1_000_000)])
    exact_adv = math.fsum(adversarial)
    print(f"    true answer         = {exact_adv:.1f}")
    print(f"    naive Python loop   = {naive_sum(adversarial):.1f}")
    print(f"    np.sum (pairwise)   = {float(np.sum(adversarial)):.1f}")
    print(f"    Kahan compensated   = {kahan_sum(adversarial):.1f}")

    print("""
  In the pathological case the naive loop returns 1e16 exactly — every one of the million
  1.0s was completely lost, because 1e16 + 1 == 1e16 in float64 (Experiment 1).

  NumPy's pairwise summation gets most of it and costs nothing extra, which is why you
  should prefer np.sum to a Python loop as a matter of habit. Kahan recovers essentially
  all of it. math.fsum is exactly rounded but slow.

  This matters whenever a loss or metric is aggregated over millions of examples — and it
  matters much more in float16, where eps is 1e-3 rather than 1e-16. That is why
  mixed-precision training accumulates reductions in float32 (README §14).""")


def experiment_gradient_check_h() -> None:
    """README §13: the U-shaped error curve, and the optimal step size."""
    print("\n" + "=" * 86)
    print("EXPERIMENT 6 — choosing h for a gradient check  (README §13)")
    print("=" * 86)
    print("""
Finite differences trade two error sources against each other:

    truncation error  ~ h^2     (shrinks as h shrinks)
    cancellation error ~ eps/h  (GROWS as h shrinks — README §4)

So the total error is U-shaped, and there is an optimal h. Theory puts it near
eps^(1/3) ~ 6e-6 for central differences in float64. Measuring on a function whose
gradient we know exactly:
""")
    rng = np.random.default_rng(2)
    x = rng.standard_normal(6)

    def f(v):
        return float(np.sum(np.sin(v) * np.exp(v / 3)))

    def grad_f(v):
        return np.cos(v) * np.exp(v / 3) + np.sin(v) * np.exp(v / 3) / 3

    print(f"  {'h':>10s}  {'relative error':>16s}  {'dominated by':>22s}")
    print("  " + "-" * 52)

    best_h, best_err = None, float("inf")
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12):
        err = check_gradient(f, grad_f, x, h=h)
        if err < best_err:
            best_err, best_h = err, h
        if h > 1e-5:
            reason = "truncation (h^2)"
        elif h < 1e-7:
            reason = "cancellation (eps/h)"
        else:
            reason = "near optimal"
        print(f"  {h:10.0e}  {err:16.3e}  {reason:>22s}")

    print(f"""
  Best h = {best_h:.0e}, relative error {best_err:.2e} — matching the eps^(1/3) ~ 6e-6
  prediction closely.

  Read the shape: too LARGE an h and the quadratic Taylor remainder dominates; too SMALL
  and you are subtracting two nearly identical numbers and cancellation destroys the
  result (README §4). Using h = 1e-12 because "smaller is more accurate" makes the check
  WORSE than h = 1e-2.

  Two practical rules: run gradient checks in float64 (in float32 eps = 1e-7 and the
  useful window nearly vanishes), and check away from kinks — ReLU at exactly 0 will fail
  a gradient check correctly, because the derivative genuinely does not exist there.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_float_anatomy()
    experiment_softmax_overflow()
    experiment_bce_saturation()
    experiment_variance_cancellation()
    experiment_summation()
    experiment_gradient_check_h()

    print("\n" + "=" * 86)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 86)
