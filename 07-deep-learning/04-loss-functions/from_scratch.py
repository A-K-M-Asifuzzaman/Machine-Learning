"""
07.04 — Loss Functions, from scratch (NumPy).

Regression and classification losses (with numerically stable with-logits cross-entropy),
gradients verified against PyTorch autograd. Then the chapter's claims are MEASURED:

  1. each loss is an NLL: MSE's minimizer is the mean, MAE's is the median   (README §2-§3)
  2. the softmax + cross-entropy gradient is exactly p_hat - y               (README §5)
  3. THE KEY RESULT: MSE's gradient vanishes on confident errors; CE's does not (README §6)
  4. numerical stability: naive softmax->log gives NaN; the stable form is correct (README §7)
  5. focal loss / class weighting focuses training on the rare class          (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def log_sum_exp(Z, axis=-1):
    m = Z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softmax(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)


# =============================================================================
# LOSSES (value + gradient w.r.t. the network output/logits)  (README §3-§5)
# =============================================================================


def mse(yhat, y):
    return np.mean((yhat - y) ** 2)


def d_mse(yhat, y):
    return 2 * (yhat - y) / y.size


def mae(yhat, y):
    return np.mean(np.abs(yhat - y))


def huber(yhat, y, delta=1.0):
    r = np.abs(yhat - y)
    quad = np.minimum(r, delta)
    return np.mean(0.5 * quad ** 2 + delta * (r - quad))


def bce_with_logits(z, y):
    """Numerically stable binary cross-entropy from logits (README §7)."""
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))


def d_bce_with_logits(z, y):
    return (sigmoid(z) - y) / y.size          # p_hat - y (README §5)


def ce_with_logits(Z, y_idx):
    """Stable categorical cross-entropy from logits; y_idx = class indices (README §7)."""
    return np.mean(log_sum_exp(Z, axis=1) - Z[np.arange(len(Z)), y_idx])


def d_ce_with_logits(Z, y_idx):
    P = softmax(Z, axis=1)
    P[np.arange(len(Z)), y_idx] -= 1          # p_hat - y (README §5)
    return P / len(Z)


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — loss values and gradients vs PyTorch autograd")
    print("=" * 88)
    if not HAVE_TORCH:
        print("\n(PyTorch unavailable — checking the softmax+CE gradient by finite differences)")
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((5, 4))
        y = rng.integers(0, 4, 5)
        g = d_ce_with_logits(Z.copy(), y)
        num = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Zp, Zm = Z.copy(), Z.copy()
                Zp[i, j] += 1e-6; Zm[i, j] -= 1e-6
                num[i, j] = (ce_with_logits(Zp, y) - ce_with_logits(Zm, y)) / 2e-6
        print(f"  softmax+CE gradient vs finite diff: max|diff| = {np.max(np.abs(g-num)):.2e}")
        assert np.max(np.abs(g - num)) < 1e-6
        print("\nAll verification checks passed.")
        return

    rng = np.random.default_rng(0)
    # regression losses + gradients
    yhat = torch.tensor(rng.standard_normal((10, 3)), requires_grad=True)
    y = torch.tensor(rng.standard_normal((10, 3)))
    checks = []
    L = F.mse_loss(yhat, y); L.backward()
    checks.append(("MSE", mse(yhat.detach().numpy(), y.numpy()), L.item(),
                   d_mse(yhat.detach().numpy(), y.numpy()), yhat.grad.numpy()))
    # classification: BCE and CE from logits
    z = torch.tensor(rng.standard_normal(20), requires_grad=True)
    yb = torch.tensor((rng.uniform(size=20) < 0.5).astype(float))
    Lb = F.binary_cross_entropy_with_logits(z, yb); Lb.backward()
    checks.append(("BCE", bce_with_logits(z.detach().numpy(), yb.numpy()), Lb.item(),
                   d_bce_with_logits(z.detach().numpy(), yb.numpy()), z.grad.numpy()))
    Z = torch.tensor(rng.standard_normal((8, 5)), requires_grad=True)
    yi = torch.tensor(rng.integers(0, 5, 8))
    Lc = F.cross_entropy(Z, yi); Lc.backward()
    checks.append(("CE", ce_with_logits(Z.detach().numpy(), yi.numpy()), Lc.item(),
                   d_ce_with_logits(Z.detach().numpy(), yi.numpy()), Z.grad.numpy()))

    print(f"\n    {'loss':>6s} {'value |diff|':>14s} {'gradient |diff|':>16s}")
    for name, ov, tv, og, tg in checks:
        vd, gd = abs(ov - tv), np.max(np.abs(og - tg))
        print(f"    {name:>6s} {vd:>14.2e} {gd:>16.2e}")
        assert vd < 1e-6 and gd < 1e-6, f"{name} mismatch"
    print("\n  losses and gradients match PyTorch to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — each loss is an NLL (minimizing constant) (README §2-§3)
# =============================================================================


def experiment_1_nll():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — each loss's minimizing constant is its estimator (README §2-§3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    y = rng.lognormal(3, 1, 5000)                 # skewed -> mean != median
    grid = np.linspace(y.min(), np.quantile(y, 0.99), 4000)
    c_mse = grid[np.argmin([mse(np.full_like(y, c), y) for c in grid])]
    c_mae = grid[np.argmin([mae(np.full_like(y, c), y) for c in grid])]
    print(f"""
  Skewed targets: mean = {y.mean():.2f}, median = {np.median(y):.2f}

    {'loss':>6s} {'minimizing constant':>20s} {'matches':>14s}
    {'MSE':>6s} {c_mse:>20.2f} {'mean (Gaussian NLL)':>19s}
    {'MAE':>6s} {c_mae:>20.2f} {'median (Laplace NLL)':>20s}

  READING: MSE is minimized by the mean and MAE by the median — exactly the estimators of the
  Gaussian and Laplace negative log-likelihoods these losses ARE. Choosing a loss is choosing a
  probabilistic model of the output (README §2).""")


# =============================================================================
# EXPERIMENT 2 — softmax + CE gradient is p_hat - y (README §5)
# =============================================================================


def experiment_2_clean_gradient():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the softmax + cross-entropy gradient is exactly p_hat - y (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((6, 4))
    y_idx = rng.integers(0, 4, 6)
    Y = np.zeros((6, 4)); Y[np.arange(6), y_idx] = 1

    analytic = d_ce_with_logits(Z.copy(), y_idx) * len(Z)   # undo the 1/n for per-sample
    clean = softmax(Z, axis=1) - Y
    # finite-difference check
    num = np.zeros_like(Z)
    for i in range(6):
        for j in range(4):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, j] += 1e-6; Zm[i, j] -= 1e-6
            num[i, j] = (ce_with_logits(Zp, y_idx) - ce_with_logits(Zm, y_idx)) / 2e-6 * len(Z)
    print(f"""
    max | (p_hat - y)  -  analytic dL/dz | = {np.max(np.abs(clean - analytic)):.2e}
    max | (p_hat - y)  -  finite-diff dL/dz | = {np.max(np.abs(clean - num)):.2e}

  READING: the gradient of softmax cross-entropy w.r.t. the LOGITS is exactly the predicted
  probabilities minus the one-hot target (p_hat - y) — all the softmax and log derivatives cancel.
  This clean form (matching finite differences to ~1e-9) is why frameworks fuse softmax+CE into one
  'with_logits' op, and it is what gives cross-entropy its good learning behaviour (README §5).""")


# =============================================================================
# EXPERIMENT 3 — cross-entropy vs MSE for classification (README §6)
# =============================================================================


def experiment_3_ce_vs_mse():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — cross-entropy vs MSE: MSE's gradient vanishes on confident errors (§6)")
    print("=" * 88)
    print(f"\n  Binary output p_hat = sigmoid(z). |dL/dz| when the model is CONFIDENTLY WRONG")
    print(f"  (target y=0, but the logit z is large positive so p_hat -> 1):\n")
    print(f"    {'logit z':>8s} {'p_hat':>8s} {'|dL/dz| MSE':>14s} {'|dL/dz| cross-entropy':>22s}")
    for z in (0.0, 2.0, 4.0, 6.0, 10.0):
        p = sigmoid(np.array([z]))[0]
        # MSE on the sigmoid output: dL/dz = (p - y) * p(1-p)
        g_mse = abs((p - 0.0) * p * (1 - p))
        # cross-entropy: dL/dz = p - y
        g_ce = abs(p - 0.0)
        print(f"    {z:>8.1f} {p:>8.4f} {g_mse:>14.5f} {g_ce:>22.5f}")
    print("""
  READING: as the model becomes confidently WRONG (p_hat -> 1, y=0), MSE's gradient COLLAPSES to
  ~0 — the (p)(1-p) = sigma'(z) factor saturates, so the model cannot learn from its worst
  mistakes and training stalls. Cross-entropy's gradient is just (p_hat - y) -> 1: the sigma'
  factor CANCELS (Experiment 2), so the gradient is LARGE exactly when the error is large. This is
  why classification uses cross-entropy, not MSE (README §6).""")


# =============================================================================
# EXPERIMENT 4 — numerical stability (README §7)
# =============================================================================


def experiment_4_stability():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — numerical stability: naive softmax->log overflows (README §7)")
    print("=" * 88)
    Z = np.array([[1000.0, 1001.0, 999.0]])       # large logits
    y_idx = np.array([1])

    # NAIVE: exp then normalize then log
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.exp(Z)
        P_naive = E / E.sum(axis=1, keepdims=True)
        naive_loss = -np.log(P_naive[0, y_idx[0]])
    stable_loss = ce_with_logits(Z, y_idx)
    print(f"""
  Logits = {Z[0]} (large). Cross-entropy of the middle class:

    naive  (exp -> normalize -> log): {naive_loss}
    stable (log-sum-exp on logits)  : {stable_loss:.6f}

  READING: exp(1000) overflows to inf, so the naive softmax gives inf/inf = NaN and the loss is
  NaN — training would immediately break. The stable form computes -z_c + log-sum-exp(z) directly
  (subtracting the max), never forming exp of a huge number, and returns the correct finite loss.
  Always use the framework's cross_entropy_with_logits (README §7).""")


# =============================================================================
# EXPERIMENT 5 — focal loss / class weighting (README §8)
# =============================================================================


def experiment_5_focal():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — focal loss down-weights easy examples for imbalance (README §8)")
    print("=" * 88)
    # per-example CE vs focal loss, for an EASY correct example vs a HARD (rare, wrong) one
    print(f"\n  Per-example loss contribution (binary, gamma=2 for focal):\n")
    print(f"    {'case':>28s} {'p_true':>8s} {'CE':>8s} {'focal':>8s} {'focal/CE':>10s}")
    for name, p in [("easy correct", 0.95), ("moderate", 0.6), ("hard / misclassified", 0.1)]:
        ce = -np.log(p)
        focal = -(1 - p) ** 2 * np.log(p)
        print(f"    {name:>28s} {p:>8.2f} {ce:>8.3f} {focal:>8.3f} {focal/ce:>10.3f}")
    print("""
  READING: focal loss multiplies cross-entropy by (1 - p_true)^gamma. For an EASY example
  (p_true=0.95) that factor is ~0.0025, so its loss is almost erased — it stops dominating the
  gradient. For a HARD/rare example (p_true=0.1) the factor is ~0.81, so it keeps most of its loss.
  Under extreme imbalance (many easy negatives, few hard positives), this focuses training on the
  cases that matter — the reason focal loss powers dense object detection (README §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_nll()
    experiment_2_clean_gradient()
    experiment_3_ce_vs_mse()
    experiment_4_stability()
    experiment_5_focal()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
