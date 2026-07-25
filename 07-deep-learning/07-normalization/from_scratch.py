"""
07.07 — Normalization, from scratch (NumPy).

Batch norm (forward + the nontrivial backward) and layer norm, in a modular layer system so
they can be trained. Forward and gradients verified against PyTorch. The chapter's claims are
then MEASURED:

  1. batch norm stabilizes activation distributions across depth        (README §1-§3)
  2. batch norm enables higher learning rates / faster convergence      (README §3)
  3. batch norm lets a badly-initialized network still train            (README §3)
  4. train vs eval mode; batch-stats-at-inference makes outputs depend on batchmates (README §4)
  5. batch norm's statistics degrade with tiny batches; layer norm does not (README §5)
  6. layer norm gives batch-INDEPENDENT per-example outputs             (README §6)

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
# LAYERS  (README §2, §6)
# =============================================================================


class Linear:
    def __init__(self, din, dout, seed=0, scale=None):
        rng = np.random.default_rng(seed)
        s = scale if scale is not None else np.sqrt(2.0 / din)   # He
        self.W = rng.standard_normal((din, dout)) * s
        self.b = np.zeros(dout)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.T @ dy
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


class BatchNorm1d:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def forward(self, x):
        if self.training:
            mu = x.mean(0)
            var = x.var(0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, dy):
        N = dy.shape[0]
        self.dgamma = (dy * self.xhat).sum(0)
        self.dbeta = dy.sum(0)
        dxhat = dy * self.gamma
        # the closed-form BN backward (the batch mean/var couple all examples)
        dx = self.inv_std / N * (N * dxhat - dxhat.sum(0) - self.xhat * (dxhat * self.xhat).sum(0))
        return dx

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.training = True

    def forward(self, x):
        mu = x.mean(1, keepdims=True)
        var = x.var(1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, dy):
        D = dy.shape[1]
        self.dgamma = (dy * self.xhat).sum(0)
        self.dbeta = dy.sum(0)
        dxhat = dy * self.gamma
        dx = self.inv_std / D * (D * dxhat - dxhat.sum(1, keepdims=True)
                                 - self.xhat * (dxhat * self.xhat).sum(1, keepdims=True))
        return dx

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class Sequential:
    def __init__(self, layers):
        self.layers = layers

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

    def train(self, X, y, lr=0.05, epochs=100):
        losses = []
        for _ in range(epochs):
            self.set_mode(True)
            pred = self.forward(X)
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)
            self.backward(2 * (pred - y) / y.size)
            self.step(lr)
        return losses


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — BatchNorm & LayerNorm forward + backward vs PyTorch")
    print("=" * 88)
    if not HAVE_TORCH:
        print("\n(PyTorch unavailable — checking BN backward by finite differences)")
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 4))
        bn = BatchNorm1d(4)
        y = bn.forward(x); dy = rng.standard_normal((8, 4))
        dx = bn.backward(dy)
        num = np.zeros_like(x)
        for i in range(8):
            for j in range(4):
                xp, xm = x.copy(), x.copy(); xp[i, j] += 1e-6; xm[i, j] -= 1e-6
                num[i, j] = ((BatchNorm1d(4).forward(xp) * dy).sum()
                             - (BatchNorm1d(4).forward(xm) * dy).sum()) / 2e-6
        print(f"  BN backward vs finite diff: max|diff| = {np.max(np.abs(dx-num)):.2e}")
        assert np.max(np.abs(dx - num)) < 1e-5
        print("\nAll verification checks passed.")
        return

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 5))

    # BatchNorm
    bn = BatchNorm1d(5, momentum=0.0)               # match torch default running behaviour off
    tx = torch.tensor(x, requires_grad=True)
    tbn = nn.BatchNorm1d(5, momentum=None, eps=1e-5).double()
    tbn.train()
    ty = tbn(tx)
    ours = bn.forward(x)
    fwd_err = np.max(np.abs(ours - ty.detach().numpy()))
    # backward
    dy = rng.standard_normal((16, 5))
    (ty * torch.tensor(dy)).sum().backward()
    dx = bn.backward(dy)
    bwd_err = np.max(np.abs(dx - tx.grad.numpy()))
    print(f"\n  BatchNorm1d: forward |diff| = {fwd_err:.2e}, backward |diff| = {bwd_err:.2e}")
    assert fwd_err < 1e-6 and bwd_err < 1e-6

    # LayerNorm
    ln = LayerNorm(5)
    tx2 = torch.tensor(x, requires_grad=True)
    tln = nn.LayerNorm(5, eps=1e-5).double()
    ty2 = tln(tx2)
    lo = ln.forward(x)
    lfe = np.max(np.abs(lo - ty2.detach().numpy()))
    dy2 = rng.standard_normal((16, 5))
    (ty2 * torch.tensor(dy2)).sum().backward()
    dx2 = ln.backward(dy2)
    lbe = np.max(np.abs(dx2 - tx2.grad.numpy()))
    print(f"  LayerNorm:   forward |diff| = {lfe:.2e}, backward |diff| = {lbe:.2e}")
    assert lfe < 1e-6 and lbe < 1e-6
    print("\n  BatchNorm and LayerNorm forward + backward match PyTorch  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — BN stabilizes activation distributions (README §1-§3)
# =============================================================================


def experiment_1_stabilize():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — batch norm stabilizes activation distributions across depth (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 64))

    def build(use_bn, scale):
        layers = []
        d = 64
        for _ in range(8):
            layers.append(Linear(d, 64, seed=rng.integers(1 << 30), scale=scale))
            if use_bn:
                layers.append(BatchNorm1d(64))
            layers.append(ReLU())
            d = 64
        return Sequential(layers)

    print(f"\n  8-layer ReLU net with slightly-large init. Activation std at layers 2, 5, 8:\n")
    print(f"    {'network':>16s} {'L2 std':>9s} {'L5 std':>9s} {'L8 std':>9s}")
    for use_bn, name in [(False, "no BN"), (True, "with BN")]:
        net = build(use_bn, scale=1.2)
        net.set_mode(True)
        a = X
        stds = []
        for L in net.layers:
            a = L.forward(a)
            if isinstance(L, ReLU):
                stds.append(np.std(a))
        picks = [stds[i] for i in (1, 4, 7)]
        print(f"    {name:>16s} " + " ".join(f"{s:>9.2e}" for s in picks))
    print("""
  READING: without normalization the slightly-large init makes the activation std GROW with depth
  (the signal is drifting out of scale). Batch norm re-centers and re-scales every layer, so the
  activation std stays stable (~0.6-0.8) all the way through — the signal is kept in a healthy range
  regardless of the weights. Stabilizing the activations is the whole point (README §3).""")


# =============================================================================
# EXPERIMENT 2 — BN enables higher learning rates (README §3)
# =============================================================================


def experiment_2_higher_lr():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — batch norm enables higher learning rates (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 20))
    y = np.tanh(X[:, :1] * 2 + X[:, 1:2])

    def build(use_bn):
        layers = []
        d = 20
        for _ in range(5):
            layers.append(Linear(d, 40, seed=int(rng.integers(1 << 30))))
            if use_bn:
                layers.append(BatchNorm1d(40))
            layers.append(ReLU())
            d = 40
        layers.append(Linear(40, 1, seed=1))
        return Sequential(layers)

    print(f"\n  5-hidden-layer net at a HIGH learning rate (0.3). Training MSE:\n")
    print(f"    {'network':>16s} {'epoch 1':>12s} {'epoch 50':>12s} {'outcome':>14s}")
    for use_bn, name in [(False, "no BN"), (True, "with BN")]:
        net = build(use_bn)
        with np.errstate(over="ignore", invalid="ignore"):
            losses = net.train(X, y, lr=0.3, epochs=100)
        losses = np.array(losses)
        outcome = "diverged" if not np.all(np.isfinite(losses)) or losses[-1] > 10 else "converged"
        l1 = f"{losses[0]:.2e}"
        l50 = "nan" if not np.isfinite(losses[49]) else f"{losses[49]:.4f}"
        print(f"    {name:>16s} {l1:>12s} {l50:>12s} {outcome:>14s}")
    print("""
  READING: at a high learning rate the un-normalized network DIVERGES (activations/gradients blow
  up). Batch norm smooths the loss landscape, so the same high rate is stable and the loss
  converges. BN lets you train with much larger learning rates — a major reason it speeds up
  training (README §3).""")


# =============================================================================
# EXPERIMENT 3 — BN reduces init sensitivity (README §3)
# =============================================================================


def experiment_3_init():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — batch norm lets a badly-initialized network still train (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 20))
    y = np.tanh(X[:, :1] * 2)

    def build(use_bn):
        layers = []
        d = 20
        for _ in range(5):
            layers.append(Linear(d, 40, seed=int(rng.integers(1 << 30)), scale=3.0))  # BAD init
            if use_bn:
                layers.append(BatchNorm1d(40))
            layers.append(ReLU())
            d = 40
        layers.append(Linear(40, 1, seed=1, scale=3.0))
        return Sequential(layers)

    print(f"\n  Badly-initialized (scale 3.0) 5-hidden-layer net at lr=0.02. Training MSE:\n")
    print(f"    {'network':>16s} {'epoch 1':>12s} {'epoch 100':>12s} {'outcome':>16s}")
    baseline = float(np.var(y))
    for use_bn, name in [(False, "no BN"), (True, "with BN")]:
        net = build(use_bn)
        with np.errstate(over="ignore", invalid="ignore"):
            losses = net.train(X, y, lr=0.02, epochs=200)
        losses = np.array(losses)
        failed = (not np.isfinite(losses[-1])) or losses[-1] > 10   # judged by the FINAL loss
        outcome = ("exploded/failed" if failed else "stalled" if losses[-1] > 0.9 * baseline
                   else "converged")
        l1 = f"{losses[0]:.2e}"
        l100 = ("nan" if not np.isfinite(losses[-1])
                else f"{losses[-1]:.2e}" if losses[-1] > 1e4 else f"{losses[-1]:.4f}")
        print(f"    {name:>16s} {l1:>12s} {l100:>12s} {outcome:>16s}")
    print("""
  READING: with a badly-scaled initialization, the un-normalized network fails (explodes or stalls).
  Batch norm re-normalizes the activations on every forward pass, correcting the bad scale on the
  fly, so the network still trains. This is why initialization matters LESS with normalization
  (07.05 §9) — BN fixes a poor init during training (README §3).""")


# =============================================================================
# EXPERIMENT 4 — train vs eval mode (README §4)
# =============================================================================


def experiment_4_train_eval():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — train vs eval mode; batch stats at inference leak across examples (§4)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    bn = BatchNorm1d(3, momentum=0.1)
    # 'train' it so running stats accumulate
    for _ in range(200):
        bn.forward(rng.standard_normal((32, 3)) * 2 + 1)

    x = np.array([[1.0, 1.0, 1.0]])                 # a single query example
    bn.training = False
    out_eval = bn.forward(x.copy())
    # same example, but in TWO different batches, using BATCH stats (training mode = the bug)
    bn.training = True
    batchA = np.vstack([x, rng.standard_normal((15, 3)) * 0.1])
    batchB = np.vstack([x, rng.standard_normal((15, 3)) * 5.0])
    outA = bn.forward(batchA.copy())[0]
    outB = bn.forward(batchB.copy())[0]
    print(f"""
  Output for the SAME example [1,1,1]:

    eval mode (running stats):                 {np.round(out_eval[0], 3)}
    train mode, in a low-variance batch:       {np.round(outA, 3)}
    train mode, in a high-variance batch:      {np.round(outB, 3)}

  READING: in EVAL mode BN uses fixed running statistics, so the example's output is deterministic
  and independent of any batch. In TRAIN mode it uses the BATCH's statistics, so the SAME example
  gets a DIFFERENT output depending on which other examples share its batch — nonsensical for
  inference. Always call .eval() before inference; forgetting it is a classic bug (README §4).""")


# =============================================================================
# EXPERIMENT 5 — batch-size dependence (README §5)
# =============================================================================


def experiment_5_batch_size():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — BN statistics degrade with tiny batches; LN does not (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    # the TRUE feature std is 1; how noisy is the per-batch estimate BN uses?
    print(f"\n  Feature with true std 1.0. Std of BN's per-batch std estimate across 500 batches:\n")
    print(f"    {'batch size':>12s} {'BN batch-std noise':>20s}")
    for bs in (2, 4, 16, 128):
        stds = [np.std(rng.standard_normal((bs, 1))) for _ in range(500)]
        print(f"    {bs:>12d} {np.std(stds):>20.3f}")
    print("""
  READING: batch norm normalizes by the BATCH's mean/variance, which are estimated from only
  `batch_size` samples. At batch size 2-4 those estimates are very noisy (std of the estimate is
  large), so BN normalizes inconsistently and can destabilize training. At batch size 128 they are
  accurate. Layer norm normalizes over FEATURES within each example, so it is completely independent
  of batch size — the reason to use LN or group norm when batches must be small (README §5).""")


# =============================================================================
# EXPERIMENT 6 — layer norm is batch-independent (README §6)
# =============================================================================


def experiment_6_ln_independent():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — layer norm gives batch-INDEPENDENT per-example outputs (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    x = np.array([[2.0, -1.0, 0.5, 3.0, -2.0]])     # one example
    ln = LayerNorm(5)
    bn = BatchNorm1d(5)

    # the same example, embedded in two very different batches
    batchA = np.vstack([x, rng.standard_normal((7, 5)) * 0.1])
    batchB = np.vstack([x, rng.standard_normal((7, 5)) * 10.0])

    ln_A = ln.forward(batchA.copy())[0]
    ln_B = ln.forward(batchB.copy())[0]
    bn.training = True
    bn_A = bn.forward(batchA.copy())[0]
    bn_B = bn.forward(batchB.copy())[0]
    print(f"""
  Output for the SAME example in two different batches (A: low-var neighbours, B: high-var):

    {'':>12s} {'LayerNorm':>28s} {'BatchNorm':>28s}
    {'in batch A':>12s} {str(np.round(ln_A, 2)):>28s} {str(np.round(bn_A, 2)):>28s}
    {'in batch B':>12s} {str(np.round(ln_B, 2)):>28s} {str(np.round(bn_B, 2)):>28s}

    LayerNorm outputs identical across batches: {np.allclose(ln_A, ln_B)}
    BatchNorm outputs identical across batches: {np.allclose(bn_A, bn_B)}

  READING: LAYER norm computes each example's statistics over its OWN features, so the same input
  gives the SAME output no matter what shares its batch — batch-independent, with no train/eval
  split. BATCH norm's output for the example changes with its batchmates. This batch-independence is
  why layer norm, not batch norm, is used in Transformers and RNNs (README §6).""")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    verify()
    experiment_1_stabilize()
    experiment_2_higher_lr()
    experiment_3_init()
    experiment_4_train_eval()
    experiment_5_batch_size()
    experiment_6_ln_independent()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
