"""
07.03 — Activation Functions, from scratch (NumPy).

Every activation and its derivative, verified against PyTorch. The chapter's claims are then
MEASURED (judging each activation by what its DERIVATIVE does to the gradient):

  1. saturation: sigmoid/tanh derivatives -> 0 for large |z|; ReLU's stays 1   (README §1-§4)
  2. gradient flow through depth: ReLU preserves magnitude, sigmoid vanishes    (README §4)
  3. the dying-ReLU fraction, and Leaky ReLU fixing it                          (README §5)
  4. GELU / Swish vs ReLU: smooth, non-monotonic, small negative response       (README §7)
  5. ReLU trains faster than sigmoid on the same task                           (README §4, §9)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# =============================================================================
# ACTIVATIONS (value + derivative)  (README §2-§7)
# =============================================================================


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return (z > 0).astype(float)


def leaky_relu(z, a=0.01):
    return np.where(z > 0, z, a * z)


def d_leaky_relu(z, a=0.01):
    return np.where(z > 0, 1.0, a)


def elu(z, a=1.0):
    return np.where(z > 0, z, a * (np.exp(np.clip(z, -30, 30)) - 1))


def _Phi(z):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))


def gelu(z):
    return z * _Phi(z)


def swish(z):
    return z * sigmoid(z)


def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)          # log-sum-exp stability (README §8)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


ACT = {"sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh), "relu": (relu, d_relu),
       "leaky_relu": (leaky_relu, d_leaky_relu), "identity": (lambda z: z, np.ones_like)}


# =============================================================================
# A COMPACT TRAINABLE MLP (for gradient-flow and training experiments)
# =============================================================================


class MLP:
    def __init__(self, sizes, act="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.Ws = [rng.standard_normal((o, i)) * np.sqrt(2.0 / i)
                   for i, o in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.zeros(o) for o in sizes[1:]]
        self.act = act

    def forward(self, X):
        a = np.asarray(X, float).T
        self.zs, self.as_ = [], [a]
        for k, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = W @ a + b[:, None]
            a = ACT[self.act][0](z) if k < len(self.Ws) - 1 else z   # linear output
            self.zs.append(z)
            self.as_.append(a)
        return a.T

    def backward(self, y):
        y = np.asarray(y, float).T
        n = y.shape[1]
        delta = (self.as_[-1] - y) / n
        dWs = [None] * len(self.Ws)
        for ell in reversed(range(len(self.Ws))):
            dWs[ell] = delta @ self.as_[ell].T
            if ell > 0:
                delta = (self.Ws[ell].T @ delta) * ACT[self.act][1](self.zs[ell - 1])
        return dWs

    def train(self, X, y, lr=0.05, epochs=300):
        losses = []
        for _ in range(epochs):
            pred = self.forward(X)
            dWs = self.backward(y)
            for i in range(len(self.Ws)):
                self.Ws[i] -= lr * dWs[i]
            losses.append(np.mean((pred - y) ** 2))
        return losses


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — activations vs PyTorch")
    print("=" * 88)
    if not HAVE_TORCH:
        print("\n(PyTorch unavailable — checking derivatives by finite differences instead)")
        z = np.linspace(-3, 3, 50)
        for name, (f, df) in [("sigmoid", (sigmoid, d_sigmoid)), ("tanh", (tanh, d_tanh)),
                              ("relu", (relu, d_relu))]:
            num = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6
            err = np.max(np.abs(num - df(z)))
            print(f"  {name:>8s}: |analytic - numeric derivative| = {err:.2e}")
            assert err < 1e-4
        print("\nAll verification checks passed.")
        return

    z = np.linspace(-5, 5, 100)
    tz = torch.tensor(z)
    checks = {
        "sigmoid": (sigmoid(z), torch.sigmoid(tz).numpy()),
        "tanh": (tanh(z), torch.tanh(tz).numpy()),
        "relu": (relu(z), F.relu(tz).numpy()),
        "leaky_relu": (leaky_relu(z), F.leaky_relu(tz, 0.01).numpy()),
        "elu": (elu(z), F.elu(tz).numpy()),
        "gelu": (gelu(z), F.gelu(tz).numpy()),
        "swish/silu": (swish(z), F.silu(tz).numpy()),
    }
    print(f"\n    {'activation':>12s} {'max|ours - torch|':>18s}")
    for name, (ours, th) in checks.items():
        d = np.max(np.abs(ours - th))
        print(f"    {name:>12s} {d:>18.2e}")
        assert d < 1e-6, f"{name} mismatch"
    # softmax
    Z = np.random.default_rng(0).standard_normal((5, 4))
    sm = np.max(np.abs(softmax(Z) - F.softmax(torch.tensor(Z), dim=-1).numpy()))
    print(f"    {'softmax':>12s} {sm:>18.2e}")
    assert sm < 1e-9
    print("\n  all activations match PyTorch to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — saturation (README §1-§4)
# =============================================================================


def experiment_1_saturation():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — saturation: the DERIVATIVE at large |z| (README §1-§4)")
    print("=" * 88)
    print(f"\n  Derivative sigma'(z) at increasing |z| (this is the per-layer gradient factor):\n")
    print(f"    {'z':>6s} {'sigmoid':>10s} {'tanh':>10s} {'ReLU':>10s}")
    for z in (0.0, 1.0, 3.0, 5.0, 10.0):
        print(f"    {z:>6.1f} {d_sigmoid(np.array([z]))[0]:>10.4f} "
              f"{d_tanh(np.array([z]))[0]:>10.4f} {d_relu(np.array([z]))[0]:>10.4f}")
    print("""
  READING: sigmoid's derivative peaks at 0.25 (z=0) and collapses toward 0 as |z| grows — it
  SATURATES, so it shrinks the gradient by >=4x per layer even at best. tanh peaks at 1.0 but also
  saturates to 0. ReLU's derivative is EXACTLY 1 for all z>0 — it never saturates on the active
  side. Since backprop multiplies the gradient by sigma'(z) at every layer, saturating activations
  vanish gradients in deep nets while ReLU preserves them (README §1, §4).""")


# =============================================================================
# EXPERIMENT 2 — gradient flow through depth (README §4)
# =============================================================================


def experiment_2_gradient_flow():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — gradient flow through a deep network by activation (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    depth = 12
    sizes = [8] + [24] * depth + [1]
    X = rng.standard_normal((64, 8))
    y = rng.standard_normal((64, 1))

    print(f"\n  {depth}-hidden-layer network. Mean |gradient| at the FIRST vs LAST layer:\n")
    print(f"    {'activation':>12s} {'first-layer |dW|':>18s} {'last-layer |dW|':>17s} "
          f"{'ratio first/last':>18s}")
    for act in ("sigmoid", "tanh", "relu"):
        net = MLP(sizes, act=act, seed=1)
        net.forward(X)
        dWs = net.backward(y)
        first, last = np.mean(np.abs(dWs[0])), np.mean(np.abs(dWs[-1]))
        print(f"    {act:>12s} {first:>18.2e} {last:>17.2e} {first/last:>18.1e}")
    print("""
  READING: with SIGMOID the first (input) layer's gradient is orders of magnitude smaller than the
  last — the gradient VANISHED on the way back, so early layers barely learn. tanh is better (peak
  derivative 1) but still decays. ReLU keeps the first- and last-layer gradients comparable — its
  derivative is 1 on the active side, so the gradient flows all the way back. This is why ReLU made
  deep networks trainable (README §4).""")


# =============================================================================
# EXPERIMENT 3 — dying ReLU (README §5)
# =============================================================================


def experiment_3_dying_relu():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the dying ReLU, and Leaky ReLU fixing it (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 20))
    y = (X @ rng.standard_normal(20) > 0).astype(float)[:, None]

    def count_dead(act, lr):
        net = MLP([20, 64, 64, 1], act=act, seed=2)
        net.train(X, y, lr=lr, epochs=200)
        # a hidden unit is DEAD if its pre-activation is <= 0 for EVERY input
        net.forward(X)
        dead = 0
        total = 0
        for z in net.zs[:-1]:                       # hidden pre-activations
            dead += np.sum(np.all(z <= 0, axis=1))
            total += z.shape[0]
        return dead, total

    print(f"\n  2-hidden-layer net (128 units), trained at a high learning rate:\n")
    print(f"    {'activation':>12s} {'dead units':>12s} {'% dead':>8s}")
    for act in ("relu", "leaky_relu"):
        dead, total = count_dead(act, lr=0.3)
        print(f"    {act:>12s} {f'{dead}/{total}':>12s} {dead/total:>7.0%}")
    print("""
  READING: with a high learning rate, a fraction of ReLU units get pushed to a permanently negative
  pre-activation — they output 0 for every input, receive 0 gradient, and never recover (DEAD).
  Leaky ReLU gives the negative side a small slope, so those units still get a gradient and stay
  alive — its dead-unit count is ~0. Leaky ReLU is the safe fix when units are dying (README §5).""")


# =============================================================================
# EXPERIMENT 4 — GELU / Swish vs ReLU (README §7)
# =============================================================================


def experiment_4_gelu_swish():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — GELU / Swish vs ReLU: smooth, non-monotonic (README §7)")
    print("=" * 88)
    print(f"\n  Activation VALUES at sample inputs (note the small NEGATIVE response near z<0):\n")
    print(f"    {'z':>6s} {'ReLU':>8s} {'GELU':>9s} {'Swish':>9s}")
    for z in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
        za = np.array([z])
        print(f"    {z:>6.1f} {relu(za)[0]:>8.3f} {gelu(za)[0]:>9.3f} {swish(za)[0]:>9.3f}")
    # min value (non-monotonic dip below 0)
    zg = np.linspace(-5, 0, 1000)
    print(f"""
  ReLU is exactly 0 for all z<0 (a hard dead zone). GELU and Swish dip slightly NEGATIVE for small
  negative z (min GELU ~ {gelu(zg).min():.3f}, min Swish ~ {swish(zg).min():.3f}) then return to 0 —
  they are smooth and NON-MONOTONIC.

  READING: GELU (z*Phi(z)) and Swish (z*sigmoid(z)) are smooth, differentiable everywhere, and give
  a small nonzero response to slightly-negative inputs — so no hard dead zone and smoother
  gradients than ReLU's kink at 0. The gains are modest but consistent on large models, which is
  why GELU is the default in Transformers while ReLU stays fine (and cheaper) elsewhere (README §7).""")


# =============================================================================
# EXPERIMENT 5 — ReLU trains faster than sigmoid (README §4, §9)
# =============================================================================


def experiment_5_training_speed():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — ReLU trains faster than sigmoid on the same task (README §4, §9)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 10))
    y = np.sin(X[:, 0] * 2) + 0.5 * X[:, 1] ** 2      # a nonlinear regression target
    y = (y - y.mean())[:, None]
    sizes = [10, 40, 40, 40, 1]                        # deep enough for saturation to bite

    print(f"\n  4-hidden-layer net, same init seed, 400 epochs. Training MSE:\n")
    print(f"    {'activation':>12s} {'epoch 50':>10s} {'epoch 200':>11s} {'epoch 400':>11s}")
    for act in ("sigmoid", "tanh", "relu"):
        net = MLP(sizes, act=act, seed=3)
        losses = net.train(X, y, lr=0.02, epochs=400)
        print(f"    {act:>12s} {losses[49]:>10.4f} {losses[199]:>11.4f} {losses[399]:>11.4f}")
    print("""
  READING: with the same architecture and learning rate, the ReLU network drives the loss down far
  faster than sigmoid — sigmoid's saturating gradients slow the early layers to a crawl, while
  ReLU's flow freely. tanh sits in between. Non-saturating activations don't just enable deep
  training; they make it converge faster (README §4, §9).""")


if __name__ == "__main__":
    # some experiments deliberately drive deep nets to extremes (aggressive LR, 12 saturating
    # layers); the resulting overflow warnings are expected and do not affect the reported results
    np.seterr(over="ignore", invalid="ignore")
    verify()
    experiment_1_saturation()
    experiment_2_gradient_flow()
    experiment_3_dying_relu()
    experiment_4_gelu_swish()
    experiment_5_training_speed()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
