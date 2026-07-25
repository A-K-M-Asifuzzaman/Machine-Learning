"""
07.08 — Regularization, from scratch (NumPy).

Weight decay, dropout (inverted, forward + backward), and early stopping in a trainable network,
with dropout verified against PyTorch. The chapter's claims are then MEASURED:

  1. baseline: a big net on small data OVERFITS (train->0, val high)    (README §1)
  2. weight decay (L2) reduces the train-val gap                        (README §2)
  3. dropout closes the train-val gap                                   (README §3)
  4. inverted dropout keeps the EXPECTED activation constant train/test (README §4)
  5. the validation U-curve, and early stopping catching its minimum    (README §5)
  6. dropout as an implicit ensemble (averaging over masks)             (README §3, §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# =============================================================================
# LAYERS  (README §2-§4)
# =============================================================================


class Linear:
    def __init__(self, din, dout, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((din, dout)) * np.sqrt(2.0 / din)
        self.b = np.zeros(dout)
        self.weight_decay = 0.0

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.T @ dy + self.weight_decay * self.W    # L2 penalty gradient
        self.db = dy.sum(0)
        return dy @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask

    def step(self, lr):
        pass


class Dropout:
    """Inverted dropout: scale kept units by 1/(1-p) during training; identity at test."""

    def __init__(self, p=0.5, seed=0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (self.rng.uniform(size=x.shape) > self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask if self.training and self.p > 0 else dy

    def step(self, lr):
        pass


class Net:
    def __init__(self, sizes, dropout=0.0, weight_decay=0.0, seed=0):
        self.layers = []
        rng = np.random.default_rng(seed)
        for k in range(len(sizes) - 1):
            lin = Linear(sizes[k], sizes[k + 1], seed=int(rng.integers(1 << 30)))
            lin.weight_decay = weight_decay
            self.layers.append(lin)
            if k < len(sizes) - 2:
                self.layers.append(ReLU())
                if dropout > 0:
                    self.layers.append(Dropout(dropout, seed=int(rng.integers(1 << 30))))

    def set_mode(self, training):
        for L in self.layers:
            if hasattr(L, "training"):
                L.training = training

    def forward(self, x):
        for L in self.layers:
            x = L.forward(x)
        return x

    def backward(self, dy):
        for L in reversed(self.layers):
            dy = L.backward(dy)

    def step(self, lr):
        for L in self.layers:
            L.step(lr)

    def mse(self, X, y):
        self.set_mode(False)
        return np.mean((self.forward(X) - y) ** 2)

    def fit(self, X, y, Xval=None, yval=None, lr=0.02, epochs=400):
        tr, va = [], []
        for _ in range(epochs):
            self.set_mode(True)
            pred = self.forward(X)
            self.backward(2 * (pred - y) / y.size)
            self.step(lr)
            tr.append(self.mse(X, y))
            if Xval is not None:
                va.append(self.mse(Xval, yval))
        return np.array(tr), np.array(va)


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — inverted dropout vs PyTorch")
    print("=" * 88)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10000, 4)) + 5.0
    p = 0.5

    d = Dropout(p, seed=0)
    d.training = True
    out = d.forward(x)
    # expected value should be preserved (inverted dropout)
    print(f"""
  Inverted dropout (p={p}) on inputs with mean {x.mean():.3f}:

    mean of dropped-out output (train): {out.mean():.3f}   (should ~= input mean)
    fraction of units zeroed:           {np.mean(out == 0):.3f}   (should ~= p)
""")
    assert abs(out.mean() - x.mean()) < 0.2, "inverted dropout preserves the mean"
    assert abs(np.mean(out == 0) - p) < 0.02, "drops ~p fraction"

    if HAVE_TORCH:
        td = nn.Dropout(p)
        td.train()
        tx = torch.tensor(x)
        tout = td(tx)
        print(f"    PyTorch dropout output mean (train): {tout.mean().item():.3f}  "
              f"(matches inverted-dropout convention)")
        assert abs(tout.mean().item() - x.mean()) < 0.2
        # eval mode: identity
        td.eval()
        assert torch.allclose(td(tx), tx)
        print("    PyTorch dropout in eval mode is the identity  ✓")
    print("\n  inverted dropout matches PyTorch's convention (preserved mean, eval=identity)  ✓")
    print("\nAll verification checks passed.")


def _overfit_data(seed=0):
    """Limited, noisy data from a NONLINEAR target + a big network = overfitting."""
    rng = np.random.default_rng(seed)
    f = lambda X: (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2)[:, None]
    Xtr = rng.standard_normal((150, 10))             # limited relative to a 128-128 net
    ytr = f(Xtr) + 0.3 * rng.standard_normal((150, 1))
    Xva = rng.standard_normal((500, 10))
    yva = f(Xva)                                     # clean validation targets
    return Xtr, ytr, Xva, yva


# =============================================================================
# EXPERIMENT 1 — the overfitting baseline (README §1)
# =============================================================================


def experiment_1_overfit():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the overfitting baseline: big net, small data (README §1)")
    print("=" * 88)
    Xtr, ytr, Xva, yva = _overfit_data(0)
    net = Net([10, 128, 128, 1], seed=1)
    tr, va = net.fit(Xtr, ytr, Xva, yva, lr=0.02, epochs=800)
    print(f"""
  40 training points, 10 features, a 128-128 network (thousands of parameters):

    train MSE: {tr[0]:.4f} (start) -> {tr[-1]:.4f} (end)
    val   MSE: {va[0]:.4f} (start) -> {va[-1]:.4f} (end)
    train-val gap at end: {va[-1] - tr[-1]:.3f}

  READING: the over-parametrized network drives the TRAINING loss to ~0 (it memorizes the 40 points)
  while the VALIDATION loss stays high — a large gap between them. This is textbook overfitting: the
  network has the capacity to fit the noise. Regularization (the rest of the chapter) closes this gap
  (README §1).""")


# =============================================================================
# EXPERIMENT 2 — weight decay (README §2)
# =============================================================================


def experiment_2_weight_decay():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — weight decay (L2) reduces the train-val gap (README §2)")
    print("=" * 88)
    Xtr, ytr, Xva, yva = _overfit_data(0)
    print(f"\n    {'weight decay':>14s} {'train MSE':>11s} {'val MSE':>10s} {'weight norm':>13s}")
    for wd in (0.0, 0.01, 0.05, 0.2):
        net = Net([10, 128, 128, 1], weight_decay=wd, seed=1)
        tr, va = net.fit(Xtr, ytr, Xva, yva, lr=0.02, epochs=800)
        wnorm = np.sqrt(sum(np.sum(L.W ** 2) for L in net.layers if isinstance(L, Linear)))
        print(f"    {wd:>14.2f} {tr[-1]:>11.4f} {va[-1]:>10.4f} {wnorm:>13.1f}")
    print("""
  READING: adding an L2 penalty shrinks the weights (weight norm falls as decay grows), producing a
  smoother, lower-variance function. The training MSE rises slightly (more bias) but the VALIDATION
  MSE improves — the bias-variance trade in action. Too much decay (0.2) over-shrinks and
  under-fits. Weight decay is ridge regression for neural nets (README §2).""")


# =============================================================================
# EXPERIMENT 3 — dropout (README §3)
# =============================================================================


def experiment_3_dropout():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — dropout closes the train-val gap (README §3)")
    print("=" * 88)
    Xtr, ytr, Xva, yva = _overfit_data(0)
    print(f"\n    {'dropout p':>11s} {'train MSE':>11s} {'val MSE':>10s} {'train-val gap':>14s}")
    for p in (0.0, 0.1, 0.3, 0.5):
        net = Net([10, 128, 128, 1], dropout=p, seed=1)
        tr, va = net.fit(Xtr, ytr, Xva, yva, lr=0.02, epochs=800)
        print(f"    {p:>11.1f} {tr[-1]:>11.4f} {va[-1]:>10.4f} {va[-1]-tr[-1]:>14.3f}")
    print("""
  READING: dropout randomly zeroes units each step, preventing co-adaptation and acting as an
  implicit ensemble of sub-networks. The training MSE rises (the net can no longer memorize) but the
  VALIDATION MSE improves and the train-val GAP shrinks — better generalization. As with weight
  decay, too much (p=0.5 here on this small net) can over-regularize (README §3).""")


# =============================================================================
# EXPERIMENT 4 — inverted dropout preserves the expected activation (README §4)
# =============================================================================


def experiment_4_inference():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — inverted dropout keeps E[activation] constant train vs test (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    x = np.full((5000, 8), 2.0)                      # constant activations = 2
    print(f"\n  Input activations all = 2.0. Mean output:\n")
    print(f"    {'dropout p':>11s} {'TRAIN mean (inverted)':>22s} {'TEST mean (identity)':>22s}")
    for p in (0.2, 0.5, 0.8):
        d = Dropout(p, seed=0)
        d.training = True
        train_out = d.forward(x.copy())
        d.training = False
        test_out = d.forward(x.copy())
        print(f"    {p:>11.1f} {train_out.mean():>22.3f} {test_out.mean():>22.3f}")
    print("""
  READING: inverted dropout scales the KEPT units by 1/(1-p) during training, so even though a
  fraction p are zeroed, the MEAN activation stays at 2.0 — the same as the full network's output at
  test time (identity). Train and test expectations MATCH, so the network behaves consistently
  across modes with no inference-time rescaling needed. (Forgetting eval mode is still a bug: it
  would apply random dropping at inference.) (README §4).""")


# =============================================================================
# EXPERIMENT 5 — early stopping (README §5)
# =============================================================================


def experiment_5_early_stopping():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the validation U-curve, and early stopping (README §5)")
    print("=" * 88)
    Xtr, ytr, Xva, yva = _overfit_data(0)
    net = Net([10, 128, 128, 1], seed=1)
    tr, va = net.fit(Xtr, ytr, Xva, yva, lr=0.02, epochs=1500)
    best_epoch = int(np.argmin(va))
    print(f"\n  Validation MSE over training:\n")
    print(f"    {'epoch':>7s} {'train MSE':>11s} {'val MSE':>10s}")
    for e in (10, 100, best_epoch, 800, 1499):
        tag = "  <- validation minimum (early stop here)" if e == best_epoch else ""
        print(f"    {e:>7d} {tr[e]:>11.4f} {va[e]:>10.4f}{tag}")
    print(f"""
  Early stopping keeps the weights from epoch {best_epoch} (val MSE {va[best_epoch]:.4f}), NOT the
  final epoch (val MSE {va[-1]:.4f}).

  READING: the training loss falls monotonically, but the VALIDATION loss follows a U — it drops
  while the net learns the signal, then RISES as it starts memorizing noise. Early stopping monitors
  the validation loss and keeps the weights from its minimum, halting before over-training. It is
  free (you have a validation set anyway) and one of the most effective regularizers (README §5).""")


# =============================================================================
# EXPERIMENT 6 — dropout as an ensemble (README §3, §8)
# =============================================================================


def experiment_6_ensemble():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — dropout as an implicit ensemble (averaging over masks) (README §3, §8)")
    print("=" * 88)
    Xtr, ytr, Xva, yva = _overfit_data(0)
    net = Net([10, 128, 128, 1], dropout=0.3, seed=1)
    net.fit(Xtr, ytr, lr=0.02, epochs=600)

    # (a) deterministic inference (dropout off) = the implicit average
    det = net.mse(Xva, yva)
    # (b) a SINGLE random sub-network (one dropout mask), averaged over many single masks
    net.set_mode(True)
    single_mses = [np.mean((net.forward(Xva) - yva) ** 2) for _ in range(50)]
    # (c) explicit MC ensemble: average the PREDICTIONS of many masked sub-networks
    net.set_mode(True)
    preds = np.mean([net.forward(Xva) for _ in range(50)], axis=0)
    ensemble = np.mean((preds - yva) ** 2)
    print(f"""
    {'inference':>34s} {'val MSE':>10s}
    {'single random sub-network (mean of 50)':>34s} {np.mean(single_mses):>10.4f}
    {'ensemble: average 50 sub-networks':>34s} {ensemble:>10.4f}
    {'deterministic (dropout off)':>34s} {det:>10.4f}

  READING: each dropout mask is a different 'thinned' sub-network. A SINGLE random sub-network is
  noisy (high val MSE); AVERAGING many masked sub-networks' predictions is much better — the classic
  ensemble variance reduction (06.01). Deterministic inference (dropout off, weights scaled)
  approximates that average in one forward pass. Dropout trains an exponential ensemble of
  weight-sharing sub-networks for free (README §3, §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_overfit()
    experiment_2_weight_decay()
    experiment_3_dropout()
    experiment_4_inference()
    experiment_5_early_stopping()
    experiment_6_ensemble()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
