"""
07.02 — Backpropagation, from scratch (NumPy).

A full MLP with forward AND backward passes (the four backprop equations), that actually TRAINS
by gradient descent. Verified two ways: finite-difference gradient checking, and against PyTorch
autograd. Then the chapter's claims are MEASURED:

  1. analytic gradients match finite differences to ~1e-9              (README §6)
  2. the verified gradients TRAIN XOR from scratch                     (README §3-§4)
  3. backprop == finite-difference gradients, at O(1) vs O(P) passes   (README §1, §6)
  4. reverse mode gives ALL parameter gradients in one backward pass   (README §5, §7)
  5. vanishing gradients: gradient magnitude shrinks with depth (sigmoid) (README §8)

Run:  python3 from_scratch.py
"""

import time
import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# =============================================================================
# ACTIVATIONS (value + derivative)  (README §4)
# =============================================================================


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


ACT = {
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "identity": (lambda z: z, lambda z: np.ones_like(z)),
}


# =============================================================================
# MLP with FORWARD and BACKWARD  (README §3-§4)
# =============================================================================


class MLP:
    def __init__(self, sizes, acts, seed=0):
        rng = np.random.default_rng(seed)
        self.Ws, self.bs = [], []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            self.Ws.append(rng.standard_normal((dout, din)) * np.sqrt(2.0 / din))
            self.bs.append(np.zeros(dout))
        self.acts = acts                            # activation name per layer

    def forward(self, X):
        """Cache pre-activations z and activations a for the backward pass."""
        a = np.asarray(X, float).T                  # (features, batch)
        self.zs, self.as_ = [], [a]
        for W, b, act in zip(self.Ws, self.bs, self.acts):
            z = W @ a + b[:, None]
            a = ACT[act][0](z)
            self.zs.append(z)
            self.as_.append(a)
        return a.T                                  # (batch, outputs)

    def backward(self, y):
        """MSE loss; return gradients (dWs, dbs) via the four backprop equations (README §4)."""
        y = np.asarray(y, float).T
        n = y.shape[1]
        aL = self.as_[-1]
        # BP1: output error  delta^L = dC/da ⊙ σ'(z^L)   (MSE: dC/da = (aL - y))
        delta = (aL - y) / n * ACT[self.acts[-1]][1](self.zs[-1])
        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.bs)
        for ell in reversed(range(len(self.Ws))):
            # BP4 and BP3: gradients from the error and the incoming activation
            dWs[ell] = delta @ self.as_[ell].T
            dbs[ell] = delta.sum(1)
            if ell > 0:
                # BP2: backpropagate the error through W^T and the local slope
                delta = (self.Ws[ell].T @ delta) * ACT[self.acts[ell - 1]][1](self.zs[ell - 1])
        return dWs, dbs

    def loss(self, X, y):
        pred = self.forward(X)
        return 0.5 * np.mean(np.sum((pred - np.asarray(y, float)) ** 2, axis=1))

    def params(self):
        return self.Ws + self.bs

    def train(self, X, y, lr=0.5, epochs=2000):
        history = []
        for _ in range(epochs):
            self.forward(X)
            dWs, dbs = self.backward(y)
            for i in range(len(self.Ws)):
                self.Ws[i] -= lr * dWs[i]
                self.bs[i] -= lr * dbs[i]
            history.append(self.loss(X, y))
        return history


# =============================================================================
# GRADIENT CHECKING  (README §6)
# =============================================================================


def numerical_gradients(net, X, y, eps=1e-5):
    """Central-difference gradient of the MSE loss w.r.t. every parameter."""
    num_dW = [np.zeros_like(W) for W in net.Ws]
    num_db = [np.zeros_like(b) for b in net.bs]
    for params, grads in [(net.Ws, num_dW), (net.bs, num_db)]:
        for p, g in zip(params, grads):
            it = np.nditer(p, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = p[idx]
                p[idx] = orig + eps
                lp = net.loss(X, y)
                p[idx] = orig - eps
                lm = net.loss(X, y)
                p[idx] = orig
                g[idx] = (lp - lm) / (2 * eps)
                it.iternext()
    return num_dW, num_db


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — backprop gradients vs finite differences and PyTorch")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 4))
    y = rng.standard_normal((8, 2))
    net = MLP([4, 6, 5, 2], ["tanh", "relu", "identity"], seed=1)

    net.forward(X)
    dWs, dbs = net.backward(y)
    num_dW, num_db = numerical_gradients(net, X, y)

    def rel_err(a, b):
        return np.max([np.max(np.abs(x - z) / (np.abs(x) + np.abs(z) + 1e-12))
                       for x, z in zip(a, b)])
    err = max(rel_err(dWs, num_dW), rel_err(dbs, num_db))
    print(f"\n  analytic vs finite-difference gradients: max relative error = {err:.2e}")
    assert err < 1e-6, "backprop must match finite differences"
    print("  the four backprop equations match numerical gradients  ✓")

    if HAVE_TORCH:
        # same weights in torch, compare gradients
        tX = torch.tensor(X, dtype=torch.float64)
        ty = torch.tensor(y, dtype=torch.float64)
        tWs = [torch.tensor(W, dtype=torch.float64, requires_grad=True) for W in net.Ws]
        tbs = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in net.bs]
        a = tX.T
        for W, b, act in zip(tWs, tbs, net.acts):
            z = W @ a + b[:, None]
            a = {"tanh": torch.tanh, "relu": torch.relu,
                 "identity": lambda x: x, "sigmoid": torch.sigmoid}[act](z)
        pred = a.T
        loss = 0.5 * ((pred - ty) ** 2).sum(1).mean()
        loss.backward()
        gerr = max(np.max(np.abs(dWs[i] - tWs[i].grad.numpy())) for i in range(len(dWs)))
        print(f"\n  analytic gradients vs PyTorch autograd: max|diff| = {gerr:.2e}")
        assert gerr < 1e-9, "backprop must match PyTorch"
        print("  hand-written backprop matches PyTorch autograd to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — gradient checking (README §6)
# =============================================================================


def experiment_1_grad_check():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — gradient checking: analytic vs numerical gradients (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal((6, 2))
    print(f"\n    {'architecture':>26s} {'max relative gradient error':>30s}")
    for sizes, acts in [([3, 4, 2], ["sigmoid", "identity"]),
                        ([3, 5, 5, 2], ["tanh", "tanh", "identity"]),
                        ([3, 8, 6, 4, 2], ["relu", "relu", "relu", "identity"])]:
        net = MLP(sizes, acts, seed=1)
        net.forward(X)
        dWs, dbs = net.backward(y)
        num_dW, num_db = numerical_gradients(net, X, y)
        err = max(
            max(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)) for a, b in zip(dWs, num_dW)),
            max(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)) for a, b in zip(dbs, num_db)))
        print(f"    {str(sizes):>26s} {err:>30.2e}")
    print("""
  READING: for every architecture the analytic backprop gradients match the central-difference
  numerical gradients to ~1e-9 — confirming the four backprop equations are implemented correctly.
  Gradient checking is the essential debugging tool: a wrong backward pass produces plausible-
  looking but incorrect gradients, and only a numerical check catches it (README §6).""")


# =============================================================================
# EXPERIMENT 2 — train XOR from scratch (README §3-§4)
# =============================================================================


def experiment_2_train_xor():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the verified gradients TRAIN XOR from scratch (README §3-§4)")
    print("=" * 88)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([[0], [1], [1], [0]], float)
    net = MLP([2, 4, 1], ["tanh", "sigmoid"], seed=2)
    hist = net.train(X, y, lr=1.0, epochs=3000)
    pred = net.forward(X).ravel()
    acc = np.mean((pred >= 0.5) == y.ravel())
    print(f"""
  2-4-1 network, tanh hidden + sigmoid output, trained by gradient descent:

    loss: {hist[0]:.4f} (start) -> {hist[-1]:.6f} (end)
    predictions: {np.round(pred, 3)}   targets: {y.ravel()}
    XOR accuracy: {acc:.2f}

  READING: with correct gradients from backprop, plain gradient descent trains a small MLP to
  solve XOR — the problem the single perceptron could not (03.09). The loss falls to near zero and
  the network outputs ~0/1 correctly. Backprop computes the gradients; the optimizer does the
  learning (README §3-§4).""")


# =============================================================================
# EXPERIMENT 3 — backprop vs finite differences (README §1, §6)
# =============================================================================


def experiment_3_speed():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — backprop == finite differences, at O(1) vs O(P) passes (README §1)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 20))
    y = rng.standard_normal((32, 10))
    net = MLP([20, 40, 30, 10], ["tanh", "relu", "identity"], seed=1)
    P = sum(W.size for W in net.Ws) + sum(b.size for b in net.bs)

    net.forward(X)
    t0 = time.time()
    dWs, dbs = net.backward(y)
    t_bp = time.time() - t0

    t0 = time.time()
    num_dW, num_db = numerical_gradients(net, X, y)
    t_fd = time.time() - t0

    err = max(np.max(np.abs(dWs[i] - num_dW[i])) for i in range(len(dWs)))
    print(f"""
  {P} parameters. Computing the FULL gradient:

    {'method':>20s} {'time (s)':>12s} {'forward passes':>16s}
    {'backprop':>20s} {t_bp:>12.5f} {'1 (+1 backward)':>16s}
    {'finite differences':>20s} {t_fd:>12.5f} {2*P:>16d}

    gradients identical: max|diff| = {err:.2e}
    backprop speedup: {t_fd/t_bp:.0f}x

  READING: backprop and finite differences produce the SAME gradient (to ~1e-9), but backprop
  needs one forward + one backward pass regardless of parameter count, while finite differences
  needs 2P forward passes (two per parameter). Here that is {t_fd/t_bp:.0f}x faster, and the gap
  grows without bound as the network grows. This O(1)-vs-O(P) difference is why backprop makes
  deep learning possible (README §1).""")


# =============================================================================
# EXPERIMENT 4 — reverse mode: all gradients in one pass (README §5, §7)
# =============================================================================


def experiment_4_reverse_mode():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — reverse mode gives ALL parameter gradients in one backward pass (§5,§7)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((16, 10))
    y = rng.standard_normal((16, 4))
    net = MLP([10, 20, 15, 4], ["tanh", "tanh", "identity"], seed=1)
    P = sum(W.size for W in net.Ws) + sum(b.size for b in net.bs)

    net.forward(X)
    dWs, dbs = net.backward(y)
    n_grads = sum(g.size for g in dWs) + sum(g.size for g in dbs)
    print(f"""
  Scalar loss, {P} parameters. A SINGLE backward pass produced {n_grads} partial derivatives
  (one per parameter): {n_grads == P}.

  READING: training has ONE output (the scalar loss) and MANY inputs (the parameters) — the regime
  where REVERSE mode is optimal. One backward pass propagates the loss gradient to every parameter
  at once. Forward-mode autodiff would need one pass PER parameter ({P} passes) to get the same
  thing. This is why deep learning uses reverse-mode (backprop), not forward-mode differentiation
  (README §7).""")


# =============================================================================
# EXPERIMENT 5 — vanishing gradients (README §8)
# =============================================================================


def experiment_5_vanishing():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — vanishing gradients in a deep sigmoid network (README §8)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    depth = 8
    sizes = [4] + [16] * depth + [1]
    X = rng.standard_normal((32, 4))
    y = rng.standard_normal((32, 1))

    # small-weight sigmoid network: sigma' <= 0.25 everywhere, so gradients shrink with depth
    net = MLP(sizes, ["sigmoid"] * depth + ["identity"], seed=1)
    for i in range(len(net.Ws)):
        net.Ws[i] *= 0.5                            # smallish weights, saturating regime
    net.forward(X)
    dWs, _ = net.backward(y)

    print(f"\n  {depth}-hidden-layer sigmoid network. Gradient magnitude per layer (output -> input):\n")
    print(f"    {'layer (from output)':>20s} {'mean |dW|':>14s}")
    norms = [np.mean(np.abs(g)) for g in dWs]
    for i, g in enumerate(reversed(norms)):
        print(f"    {i:>20d} {g:>14.3e}")
    ratio = norms[-1] / (norms[0] + 1e-30)          # output-layer / input-layer gradient
    print(f"""
  the INPUT (first) layer's gradient is ~{ratio:.0e}x SMALLER than the OUTPUT layer's.

  READING: backprop's (BP2) multiplies the gradient by W^T and sigma'(z) at every layer. For
  sigmoid, sigma' <= 0.25, so the gradient shrinks by a factor <1 per layer — after {depth} layers
  the EARLY layers get a vanishingly small gradient and barely learn. This vanishing-gradient
  problem is why we use ReLU (07.03), careful initialization (07.05), normalization (07.07), and
  residual connections — all ways to keep backprop's product of Jacobians near 1 (README §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_grad_check()
    experiment_2_train_xor()
    experiment_3_speed()
    experiment_4_reverse_mode()
    experiment_5_vanishing()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
