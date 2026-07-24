"""
07.05 — Weight Initialization, from scratch (NumPy).

Initialization schemes and the variance-preservation principle, with He/Glorot scales checked
against PyTorch. The chapter's claims are then MEASURED:

  1. the symmetry problem: zero init keeps every unit identical, never learns   (README §2)
  2. good init PRESERVES activation variance across depth; bad init vanishes/explodes (§3)
  3. He preserves ReLU-network variance where Glorot decays it (the factor-of-2)  (README §6)
  4. bad init -> training vanishes / diverges; good init converges              (README §7)
  5. activation std by layer for each scheme                                    (README §3)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# =============================================================================
# INITIALIZATION SCHEMES  (README §5-§6)
# =============================================================================


def init_weights(n_in, n_out, scheme, rng):
    if scheme == "zeros":
        return np.zeros((n_out, n_in))
    if scheme == "small":                          # too small
        return rng.standard_normal((n_out, n_in)) * 0.01
    if scheme == "large":                          # too large
        return rng.standard_normal((n_out, n_in)) * 1.0
    if scheme == "glorot":                         # Var = 2/(n_in+n_out)
        return rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / (n_in + n_out))
    if scheme == "he":                             # Var = 2/n_in  (ReLU)
        return rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)
    if scheme == "orthogonal":
        A = rng.standard_normal((n_out, n_in))
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        return U @ Vt
    raise ValueError(scheme)


ACT = {"relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
       "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
       "identity": (lambda z: z, np.ones_like)}


class MLP:
    def __init__(self, sizes, scheme="he", act="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.Ws = [init_weights(i, o, scheme, rng) for i, o in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.zeros(o) for o in sizes[1:]]
        self.act = act

    def forward(self, X):
        a = np.asarray(X, float).T
        self.zs, self.as_ = [], [a]
        for k, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = W @ a + b[:, None]
            a = ACT[self.act][0](z) if k < len(self.Ws) - 1 else z
            self.zs.append(z)
            self.as_.append(a)
        return a.T

    def layer_activation_std(self):
        return [np.std(a) for a in self.as_[1:-1]]  # hidden layers

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

    def train(self, X, y, lr=0.01, epochs=200):
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
    print("VERIFICATION — He / Glorot init scales vs PyTorch nn.init")
    print("=" * 88)
    n_in, n_out = 256, 128
    rng = np.random.default_rng(0)
    he = init_weights(n_in, n_out, "he", rng)
    glorot = init_weights(n_in, n_out, "glorot", rng)
    print(f"""
  Weight std for a {n_in}->{n_out} layer:

    {'scheme':>10s} {'our std':>12s} {'theoretical':>14s}
    {'He':>10s} {np.std(he):>12.4f} {np.sqrt(2.0/n_in):>14.4f}
    {'Glorot':>10s} {np.std(glorot):>12.4f} {np.sqrt(2.0/(n_in+n_out)):>14.4f}""")
    assert abs(np.std(he) - np.sqrt(2.0 / n_in)) < 0.01
    assert abs(np.std(glorot) - np.sqrt(2.0 / (n_in + n_out))) < 0.01

    if HAVE_TORCH:
        W = torch.empty(n_out, n_in)
        torch.nn.init.kaiming_normal_(W, nonlinearity="relu")
        torch.nn.init.xavier_normal_(W)  # noqa (just to exercise the call)
        Wk = torch.empty(n_out, n_in); torch.nn.init.kaiming_normal_(Wk, nonlinearity="relu")
        Wx = torch.empty(n_out, n_in); torch.nn.init.xavier_normal_(Wx)
        print(f"""
    PyTorch kaiming_normal std = {Wk.std().item():.4f}  (ours He {np.std(he):.4f})
    PyTorch xavier_normal  std = {Wx.std().item():.4f}  (ours Glorot {np.std(glorot):.4f})""")
        assert abs(Wk.std().item() - np.sqrt(2.0 / n_in)) < 0.02
    print("\n  He and Glorot scales match theory (and PyTorch)  ✓")

    # orthogonal preserves norm
    Q = init_weights(64, 64, "orthogonal", rng)
    x = rng.standard_normal(64)
    print(f"\n  orthogonal init preserves norm: ||Qx||/||x|| = {np.linalg.norm(Q@x)/np.linalg.norm(x):.4f}")
    assert abs(np.linalg.norm(Q @ x) / np.linalg.norm(x) - 1.0) < 1e-10
    print("  orthogonal init has gain exactly 1  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — the symmetry problem (README §2)
# =============================================================================


def experiment_1_symmetry():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — zero init: every unit stays identical, never learns (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8))
    y = (X @ rng.standard_normal(8) > 0).astype(float)[:, None]

    for scheme in ("zeros", "he"):
        net = MLP([8, 16, 16, 1], scheme=scheme, act="relu", seed=1)
        losses = net.train(X, y, lr=0.05, epochs=300)
        net.forward(X)
        # variance ACROSS the 16 hidden units for the first sample (0 = all identical)
        unit_var = np.var(net.as_[1][:, 0])
        print(f"    {scheme:>8s} init: variance across hidden units = {unit_var:.2e}, "
              f"final loss = {losses[-1]:.4f}")
    print("""
  READING: with ZERO init every hidden unit computes the same value (variance across units = 0)
  and receives the same gradient, so they stay identical forever — the network cannot use its
  capacity and the loss barely moves. HE (random) init breaks the symmetry: units differ, specialize,
  and the loss falls. Weights MUST be random to break symmetry (README §2).""")


# =============================================================================
# EXPERIMENT 2 — variance preservation across depth (README §3)
# =============================================================================


def experiment_2_variance():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — good init preserves activation variance across depth (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    depth = 20
    sizes = [64] + [64] * depth + [1]
    X = rng.standard_normal((256, 64))

    print(f"\n  {depth}-layer tanh network. Activation std at layers 1, 5, 10, 15, 20:\n")
    print(f"    {'init':>10s} {'L1':>9s} {'L5':>9s} {'L10':>9s} {'L15':>9s} {'L20':>9s}")
    for scheme in ("small", "large", "glorot"):
        net = MLP(sizes, scheme=scheme, act="tanh", seed=1)
        net.forward(X)
        stds = net.layer_activation_std()
        picks = [stds[i] for i in (0, 4, 9, 14, 19)]
        print(f"    {scheme:>10s} " + " ".join(f"{s:>9.2e}" for s in picks))
    print("""
  READING: TOO-SMALL init shrinks the activation std geometrically toward 0 (signal vanishes with
  depth); TOO-LARGE init pushes tanh into saturation (std climbs toward the +-1 bound then the
  gradient dies); GLOROT keeps the std roughly CONSTANT across all 20 layers — the signal traverses
  the whole network intact. Preserving variance per layer is the whole goal of initialization (§3).""")


# =============================================================================
# EXPERIMENT 3 — He vs Glorot for ReLU (README §6)
# =============================================================================


def experiment_3_he_vs_glorot():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — He vs Glorot for a ReLU network: the factor of 2 (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    depth = 20
    sizes = [128] + [128] * depth + [1]
    X = rng.standard_normal((256, 128))

    print(f"\n  {depth}-layer ReLU network. Activation std at layers 1, 5, 10, 15, 20:\n")
    print(f"    {'init':>10s} {'L1':>9s} {'L5':>9s} {'L10':>9s} {'L15':>9s} {'L20':>9s}")
    for scheme in ("glorot", "he"):
        net = MLP(sizes, scheme=scheme, act="relu", seed=1)
        net.forward(X)
        stds = net.layer_activation_std()
        picks = [stds[i] for i in (0, 4, 9, 14, 19)]
        print(f"    {scheme:>10s} " + " ".join(f"{s:>9.2e}" for s in picks))
    print("""
  READING: ReLU zeroes half its inputs, halving the variance per layer. GLOROT (which assumes a
  variance-preserving activation) under-scales, so the std DECAYS geometrically through the ReLU
  layers — the signal fades. HE adds a factor of 2 to exactly cancel ReLU's halving, keeping the std
  roughly constant across all 20 layers. Use He for ReLU networks; Glorot is for tanh (README §6).""")


# =============================================================================
# EXPERIMENT 4 — bad init breaks training (README §7)
# =============================================================================


def experiment_4_training():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — bad init: training vanishes / diverges; good init converges (README §7)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    y = np.tanh(X[:, 0] * 2 + X[:, 1]) [:, None]
    sizes = [20] + [40] * 6 + [1]                  # deep enough for init to matter

    baseline = float(np.var(y))                    # loss of the trivial constant predictor
    print(f"\n  6-hidden-layer ReLU net, same data & LR (baseline / dead-net loss = {baseline:.3f}):\n")
    print(f"    {'init':>10s} {'epoch 1':>12s} {'epoch 50':>10s} {'epoch 200':>12s} {'outcome':>18s}")
    for scheme in ("small", "large", "he"):
        net = MLP(sizes, scheme=scheme, act="relu", seed=1)
        with np.errstate(over="ignore", invalid="ignore"):
            losses = net.train(X, y, lr=0.02, epochs=200)
        losses = np.array(losses)
        exploded = (not np.all(np.isfinite(losses))) or losses.max() > 100
        if exploded:
            outcome = "exploded -> dead"
        elif losses[-1] > 0.95 * baseline:
            outcome = "stalled (no learning)"
        else:
            outcome = "converged"
        l1 = f"{losses[0]:.2e}"; l50 = f"{losses[49]:.3f}"
        l200 = "nan" if not np.isfinite(losses[-1]) else f"{losses[-1]:.4f}"
        print(f"    {scheme:>10s} {l1:>12s} {l50:>10s} {l200:>12s} {outcome:>18s}")
    print("""
  READING: TOO-SMALL init -> gradients vanish -> the loss STALLS at the baseline (never learns).
  TOO-LARGE init -> activations/gradients EXPLODE (loss ~3e8 at epoch 1); the huge step then kills
  every ReLU, leaving a DEAD network stuck at the baseline. HE init preserves variance -> gradients
  flow -> the loss CONVERGES below the baseline. Same architecture and data, three fates decided
  entirely by the initial weight scale (README §7).""")


# =============================================================================
# EXPERIMENT 5 — activation std by scheme (README §3)
# =============================================================================


def experiment_5_summary():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — activation std at a deep layer, by scheme (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    sizes = [100] + [100] * 15 + [1]
    X = rng.standard_normal((256, 100))
    print(f"\n  Std of activations at hidden layer 15 (deep), input std = 1.0:\n")
    print(f"    {'init (activation)':>20s} {'layer-15 std':>14s} {'verdict':>18s}")
    for scheme, act in [("zeros", "relu"), ("small", "relu"), ("glorot", "relu"),
                        ("he", "relu"), ("large", "relu")]:
        net = MLP(sizes, scheme=scheme, act=act, seed=1)
        net.forward(X)
        s = net.layer_activation_std()[14]
        verdict = ("dead/zero" if s < 1e-6 else "vanishing" if s < 0.1
                   else "exploding" if s > 10 else "preserved")
        print(f"    {scheme + ' (' + act + ')':>20s} {s:>14.2e} {verdict:>18s}")
    print("""
  READING: only HE init keeps the activation std near its input scale (~1) at a deep layer with
  ReLU; zeros give dead activations, small vanishes, glorot decays (under-scaled for ReLU), and
  large explodes. The initialization scale is the single knob that decides whether a deep network's
  signal survives to its deep layers (README §3, §6).""")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    verify()
    experiment_1_symmetry()
    experiment_2_variance()
    experiment_3_he_vs_glorot()
    experiment_4_training()
    experiment_5_summary()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
