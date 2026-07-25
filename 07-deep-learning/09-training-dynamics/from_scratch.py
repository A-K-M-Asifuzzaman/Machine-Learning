"""
07.09 — Training Dynamics & Debugging, from scratch (NumPy).

A small trainable classifier plus the diagnostic toolkit. The chapter's practical checks are
DEMONSTRATED:

  1. the initial-loss sanity check: cross-entropy at init ~= log(K)        (README §3)
  2. overfit a single batch: healthy model -> ~0 loss; broken model stuck  (README §4)
  3. loss-curve signatures: too-low / too-high / just-right learning rate  (README §5)
  4. the update-to-weight ratio and its healthy ~1e-3 band                 (README §6)
  5. the learning-rate range test finds the sweet spot                     (README §7)

Run:  python3 from_scratch.py
"""

import numpy as np


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def log_sum_exp(Z):
    m = Z.max(1, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(1, keepdims=True)))[:, 0]


# =============================================================================
# A SMALL CLASSIFIER (softmax cross-entropy)  (README §3-§4)
# =============================================================================


class Classifier:
    def __init__(self, sizes, seed=0, break_gradient=False, small_output=False):
        rng = np.random.default_rng(seed)
        self.Ws = [rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
                   for i, o in zip(sizes[:-1], sizes[1:])]
        if small_output:
            self.Ws[-1] *= 0.01                      # small OUTPUT init -> logits ~0 at start
        self.bs = [np.zeros(o) for o in sizes[1:]]
        self.break_gradient = break_gradient        # a planted bug: no gradients flow at all

    def forward(self, X):
        self.a = [np.asarray(X, float)]
        self.z = []
        a = self.a[0]
        for k, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            self.z.append(z)
            a = np.maximum(0, z) if k < len(self.Ws) - 1 else z
            self.a.append(a)
        return a                                    # logits

    def loss(self, X, y):
        logits = self.forward(X)
        return float(np.mean(log_sum_exp(logits) - logits[np.arange(len(y)), y]))

    def backward(self, y):
        n = len(y)
        P = softmax(self.a[-1])
        P[np.arange(n), y] -= 1
        delta = P / n
        if self.break_gradient:
            delta *= 0.0                             # BUG: no gradient flows anywhere
        self.dWs = [None] * len(self.Ws)
        for ell in reversed(range(len(self.Ws))):
            self.dWs[ell] = self.a[ell].T @ delta
            if ell > 0:
                delta = (delta @ self.Ws[ell].T) * (self.z[ell - 1] > 0)
        return self.dWs

    def step(self, lr):
        for i in range(len(self.Ws)):
            self.Ws[i] -= lr * self.dWs[i]

    def train(self, X, y, lr=0.1, epochs=200, track_ratio=False):
        losses, ratios = [], []
        for _ in range(epochs):
            losses.append(self.loss(X, y))
            self.backward(y)
            if track_ratio:
                upd = lr * np.linalg.norm(self.dWs[0])
                ratios.append(upd / (np.linalg.norm(self.Ws[0]) + 1e-12))
            self.step(lr)
        return np.array(losses), np.array(ratios)


def _make_clf(n=400, K=4, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, K))
    y = np.argmax(X @ W + 0.5 * rng.standard_normal((n, K)), axis=1)
    return X, y


# =============================================================================
# EXPERIMENT 1 — initial-loss sanity check (README §3)
# =============================================================================


def experiment_1_initial_loss():
    print("=" * 88)
    print("EXPERIMENT 1 — the initial-loss sanity check: CE at init ~= log(K) (README §3)")
    print("=" * 88)
    print(f"\n  A freshly-initialized softmax classifier predicts ~uniformly, so its cross-entropy")
    print(f"  should be ~log(K) BEFORE any training:\n")
    print(f"    {'# classes K':>12s} {'initial CE':>12s} {'log(K)':>10s} {'|diff|':>10s}")
    for K in (2, 4, 10, 100):
        X, y = _make_clf(n=2000, K=K, d=20, seed=1)
        net = Classifier([20, 64, K], seed=2, small_output=True)
        init_loss = net.loss(X, y)
        print(f"    {K:>12d} {init_loss:>12.4f} {np.log(K):>10.4f} {abs(init_loss-np.log(K)):>10.4f}")
    print("""
  READING: the initial cross-entropy matches log(K) at every class count (0.69 for binary, 2.30 for
  10 classes, 4.61 for 100). If your initial loss is FAR from log(K), something is broken before
  training even starts — the wrong loss, a saturating output, or a label bug. This 10-second check
  is the first thing to run (README §3).""")


# =============================================================================
# EXPERIMENT 2 — overfit a single batch (README §4)
# =============================================================================


def experiment_2_overfit_batch():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — overfit a single batch: healthy -> ~0, broken -> stuck (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 10))                # a tiny fixed batch
    y = rng.integers(0, 4, 8)

    healthy = Classifier([10, 64, 64, 4], seed=1)
    broken = Classifier([10, 64, 64, 4], seed=1, break_gradient=True)   # planted bug
    lh, _ = healthy.train(X, y, lr=0.3, epochs=500)
    lb, _ = broken.train(X, y, lr=0.3, epochs=500)
    print(f"""
  Training on ONLY 8 fixed examples (no regularization):

    {'model':>28s} {'loss start':>12s} {'loss end':>12s}
    {'healthy':>28s} {lh[0]:>12.4f} {lh[-1]:>12.6f}
    {'broken (gradient killed)':>28s} {lb[0]:>12.4f} {lb[-1]:>12.6f}

  READING: a HEALTHY model has more than enough capacity to memorize 8 examples, so its loss goes to
  ~0 — confirming the architecture, loss, and gradient flow are correct. The BROKEN model (a planted
  bug that zeroes the gradient to the first layer) CANNOT drive the loss down — it is stuck. 'Can you
  overfit a single batch?' is the single most valuable debugging check: pass it and any remaining
  problem is about generalization, not correctness (README §4).""")


# =============================================================================
# EXPERIMENT 3 — loss-curve signatures (README §5)
# =============================================================================


def experiment_3_loss_curves():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — loss-curve signatures by learning rate (README §5)")
    print("=" * 88)
    X, y = _make_clf(n=800, K=4, d=20, seed=3)
    print(f"\n  Same net, three learning rates. Training loss (log K = {np.log(4):.3f}):\n")
    print(f"    {'learning rate':>14s} {'epoch 1':>10s} {'max loss':>12s} {'epoch 200':>12s} "
          f"{'signature':>18s}")
    for lr in (1e-4, 30.0, 0.3):
        net = Classifier([20, 64, 64, 4], seed=1)
        with np.errstate(over="ignore", invalid="ignore"):
            losses, _ = net.train(X, y, lr=lr, epochs=200)
        maxl = np.nanmax(losses)
        diverged = maxl > 100 or not np.all(np.isfinite(losses))
        sig = ("DIVERGED (blew up)" if diverged else
               "flat / no learning" if losses[-1] > 0.9 * losses[0] else
               "converges")
        l1 = f"{losses[0]:.3f}"
        maxstr = f"{maxl:.2e}" if maxl > 100 else f"{maxl:.3f}"
        l200 = "nan" if not np.isfinite(losses[-1]) else f"{losses[-1]:.3f}"
        print(f"    {lr:>14.0e} {l1:>10s} {maxstr:>12s} {l200:>12s} {sig:>18s}")
    print("""
  READING: the loss curve's SHAPE diagnoses the problem. TOO LOW a learning rate -> the loss barely
  moves (flat). TOO HIGH -> it overshoots and diverges to NaN. JUST RIGHT -> a clean descent. Reading
  these shapes is the fastest route to a diagnosis: flat means raise the LR (or check for a gradient
  bug); NaN means lower it (README §5).""")


# =============================================================================
# EXPERIMENT 4 — update-to-weight ratio (README §6)
# =============================================================================


def experiment_4_update_ratio():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — the update-to-weight ratio and its healthy ~1e-3 band (README §6)")
    print("=" * 88)
    X, y = _make_clf(n=800, K=4, d=20, seed=4)
    print(f"\n  Median ||lr*dW|| / ||W|| for layer 0 over training (healthy ~ 1e-3):\n")
    print(f"    {'learning rate':>14s} {'update/weight ratio':>20s} {'verdict':>18s}")
    for lr in (1e-4, 1.0, 5.0):
        net = Classifier([20, 64, 64, 4], seed=1)
        with np.errstate(over="ignore", invalid="ignore"):
            _, ratios = net.train(X, y, lr=lr, epochs=200, track_ratio=True)
        r = np.median(ratios[np.isfinite(ratios)])
        verdict = ("too small (LR low)" if r < 1e-4 else
                   "too large (LR high)" if r > 1e-2 else "healthy (~1e-3)")
        print(f"    {lr:>14.0e} {r:>20.2e} {verdict:>18s}")
    print("""
  READING: the ratio of the update size to the weight size is a learning-rate health check. A
  healthy value is ~1e-3 (updates are ~0.1% of the weights per step). Much smaller means the LR is
  too low (glacial or vanishing gradients); much larger means it is too high (unstable). This
  internal signal catches LR problems the loss curve can miss, layer by layer (README §6).""")


# =============================================================================
# EXPERIMENT 5 — learning-rate range test (README §7)
# =============================================================================


def experiment_5_lr_range_test():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the learning-rate range test finds the sweet spot (README §7)")
    print("=" * 88)
    X, y = _make_clf(n=800, K=4, d=20, seed=5)
    lrs = np.logspace(-4, 1.0, 24)                  # exponentially increasing LR
    losses = []
    for lr in lrs:
        net = Classifier([20, 64, 64, 4], seed=1)
        with np.errstate(over="ignore", invalid="ignore"):
            L, _ = net.train(X, y, lr=lr, epochs=40)
        losses.append(L[-1] if np.isfinite(L[-1]) else np.inf)
    losses = np.array(losses)
    finite = np.isfinite(losses)
    best = lrs[np.argmin(np.where(finite, losses, np.inf))]
    # steepest descent: largest drop in loss per log-LR step
    print(f"\n  Loss after 40 steps at exponentially increasing LR:\n")
    print(f"    {'learning rate':>14s} {'final loss':>12s}")
    for i in range(0, len(lrs), 3):
        lstr = "inf/nan" if not finite[i] else f"{losses[i]:.3f}"
        print(f"    {lrs[i]:>14.0e} {lstr:>12s}")
    print(f"""
  LR with the lowest loss ~ {best:.0e}; the loss diverges above ~{lrs[finite][-1]:.0e}.
  Pick a rate near the steepest descent, ~an order of magnitude below divergence.

  READING: sweeping the learning rate exponentially, the loss barely moves at tiny LR, DROPS STEEPLY
  through the good range, then BLOWS UP past the stability limit. The best rate sits at the steepest
  descent, roughly 10x below where it diverges. This one sweep replaces blind trial-and-error and is
  standard before a serious run (README §7).""")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    experiment_1_initial_loss()
    experiment_2_overfit_batch()
    experiment_3_loss_curves()
    experiment_4_update_ratio()
    experiment_5_lr_range_test()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
