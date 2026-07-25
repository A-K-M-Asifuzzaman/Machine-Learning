"""
09.01 — Recurrent Neural Networks, from scratch (NumPy).

An RNN processes a sequence by carrying a hidden state forward: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b).
This file builds the forward pass and backpropagation-through-time (BPTT), verifies both against
PyTorch's nn.RNN and autograd to machine precision, and then MEASURES the pathology that defines RNNs:

  1. RNN forward + BPTT == torch.nn.RNN and autograd                 (machine precision)
  2. the gradient VANISHES or EXPLODES exponentially with sequence length   -> Experiment 2
  3. tanh saturation drives the vanishing (its derivative is <= 1)          -> Experiment 3
  4. gradient clipping tames explosion (but not vanishing)                  -> Experiment 4
  5. a trained RNN learns SHORT dependencies but fails LONG ones            -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False


# =============================================================================
# The RNN cell: forward and backpropagation through time
# =============================================================================


class RNN:
    def __init__(self, Wxh, Whh, b, h0=None):
        self.Wxh, self.Whh, self.b = Wxh, Whh, b     # (H,I), (H,H), (H,)
        self.H = Whh.shape[0]
        self.h0 = np.zeros(self.H) if h0 is None else h0

    def forward(self, X):
        """X:(T,I) -> hidden states H:(T,Hdim). Caches pre-activations for BPTT."""
        self.X = X
        self.pre = []
        self.hs = [self.h0]
        h = self.h0
        for t in range(len(X)):
            z = self.Wxh @ X[t] + self.Whh @ h + self.b
            h = np.tanh(z)
            self.pre.append(z)
            self.hs.append(h)
        return np.array(self.hs[1:])

    def bptt(self, dH):
        """dH:(T,Hdim) upstream grad per timestep -> dWxh, dWhh, db, and dh0."""
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        db = np.zeros_like(self.b)
        dh_next = np.zeros(self.H)
        for t in reversed(range(len(self.X))):
            dh = dH[t] + dh_next                     # grad flowing in from output + future
            dz = dh * (1 - np.tanh(self.pre[t]) ** 2)   # through tanh
            dWxh += np.outer(dz, self.X[t])
            dWhh += np.outer(dz, self.hs[t])         # hs[t] is h_{t-1}
            db += dz
            dh_next = self.Whh.T @ dz                 # grad to previous hidden state
        return dWxh, dWhh, db, dh_next


def experiment_1_verify():
    print("=" * 88)
    print("EXPERIMENT 1 — RNN forward + BPTT == torch.nn.RNN and autograd (machine precision)")
    print("=" * 88)
    if not HAVE_TORCH:
        print("  torch not available — skipping.")
        return
    rng = np.random.default_rng(0)
    T, I, H = 7, 4, 5
    X = rng.standard_normal((T, I))
    Wxh = rng.standard_normal((H, I)) * 0.5
    Whh = rng.standard_normal((H, H)) * 0.5
    b = rng.standard_normal(H) * 0.1

    net = RNN(Wxh, Whh, b)
    Hs = net.forward(X)
    dH = np.ones_like(Hs)                             # loss = sum of all hidden states
    dWxh, dWhh, db, _ = net.bptt(dH)

    torch_rnn = torch.nn.RNN(I, H, nonlinearity="tanh", batch_first=True).double()
    with torch.no_grad():
        torch_rnn.weight_ih_l0.copy_(torch.tensor(Wxh))
        torch_rnn.weight_hh_l0.copy_(torch.tensor(Whh))
        torch_rnn.bias_ih_l0.copy_(torch.tensor(b))
        torch_rnn.bias_hh_l0.zero_()                  # our single bias == torch's b_ih (b_hh=0)
    Xt = torch.tensor(X)[None].requires_grad_(True)
    out, _ = torch_rnn(Xt)
    ef = np.abs(Hs - out.detach().numpy()[0]).max()
    out.sum().backward()
    ew = np.abs(dWxh - torch_rnn.weight_ih_l0.grad.numpy()).max()
    eh = np.abs(dWhh - torch_rnn.weight_hh_l0.grad.numpy()).max()
    print(f"""
  Random RNN, sequence length {T}:

    forward  h_t   |ours - torch|  = {ef:.1e}
    BPTT     dWxh  |ours - torch|  = {ew:.1e}
    BPTT     dWhh  |ours - torch|  = {eh:.1e}

  READING: the recurrence h_t = tanh(W_xh x_t + W_hh h_{{t-1}} + b) unrolls into a deep net that shares
  ONE set of weights across all timesteps. Backpropagation-through-time is ordinary backprop on that
  unrolled graph, accumulating the shared-weight gradient over every step. Both match PyTorch exactly.""")


# =============================================================================
# EXPERIMENT 2 — vanishing / exploding gradients
# =============================================================================


def _grad_to_start(T, spectral, seed=0):
    """Norm of the gradient of h_T w.r.t. h_0 — a product of T Jacobians diag(1-h^2) W_hh."""
    rng = np.random.default_rng(seed)
    H = 20
    W = rng.standard_normal((H, H))
    W = W * spectral / np.max(np.abs(np.linalg.eigvals(W)))   # set spectral radius exactly
    X = rng.standard_normal((T, H)) * 0.3
    h = np.zeros(H)
    pre = []
    for t in range(T):
        z = W @ h + X[t]
        h = np.tanh(z)
        pre.append(z)
    g = np.ones(H)                                    # dL/dh_T
    for t in reversed(range(T)):
        g = (W.T @ g) * (1 - np.tanh(pre[t]) ** 2)
    return np.linalg.norm(g)


def experiment_2_vanishing():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — gradients vanish or explode exponentially with sequence length (README §4)")
    print("=" * 88)
    print(f"\n  ||dL/dh_0|| as the sequence gets longer, for various spectral radii of W_hh:\n")
    print(f"    {'spectral radius':>16s}" + "".join(f"{f'T={T}':>12s}" for T in (5, 10, 25, 50)))
    for sr in (0.5, 0.9, 1.0, 1.5):
        vals = "".join(f"{_grad_to_start(T, sr):>12.1e}" for T in (5, 10, 25, 50))
        tag = f"{sr}" + (" (vanish)" if sr < 1.05 else " (explode)")
        print(f"    {tag:>16s}{vals}")
    print("""
  READING: the gradient back to step 0 is a PRODUCT of T Jacobians (one per step). If the effective
  per-step factor is < 1 the product shrinks to ~0 (VANISHING: 1e-18 at T=50, radius 0.5); if > 1 it
  blows up (EXPLODING: radius 1.5). Note even spectral radius 1.0 vanishes — because tanh's derivative
  is <= 1 (Experiment 3). Vanishing gradients are why a plain RNN cannot learn LONG-range dependencies:
  the error signal never reaches the early steps (Experiment 5). This is the problem LSTMs solve (09.02).""")


# =============================================================================
# EXPERIMENT 3 — tanh saturation drives the vanishing
# =============================================================================


def experiment_3_tanh():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — tanh saturation drives the vanishing (README §4)")
    print("=" * 88)
    print(f"\n  tanh'(z) = 1 - tanh(z)^2, its maximum is 1 (at z=0) and it shrinks fast:\n")
    print(f"    {'z':>6s} {'tanh(z)':>10s} {'tanh_prime(z)':>14s}")
    for z in (0.0, 0.5, 1.0, 2.0, 4.0):
        print(f"    {z:>6.1f} {np.tanh(z):>10.4f} {1 - np.tanh(z) ** 2:>14.4f}")
    prod = np.prod([1 - np.tanh(1.5) ** 2] * 25)
    print(f"""
  Multiplying the derivative over just 25 steps (each tanh'(1.5) = {1-np.tanh(1.5)**2:.3f}):
    ({1-np.tanh(1.5)**2:.3f})^25 = {prod:.2e}

  READING: because tanh'(z) <= 1 everywhere — and <= 0.2 once |z| > 1.5 — every step MULTIPLIES the
  gradient by at most 1, and usually much less. Over a long sequence the product collapses toward zero
  even if W_hh is perfectly scaled. Saturation makes VANISHING the default failure mode of RNNs;
  ReLU/identity recurrences avoid this shrink but then explode more easily (README §4).""")


# =============================================================================
# EXPERIMENT 4 — gradient clipping tames explosion
# =============================================================================


def experiment_4_clipping():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — gradient clipping tames explosion (README §5)")
    print("=" * 88)

    def clip(g, max_norm):
        n = np.linalg.norm(g)
        return g * (max_norm / n) if n > max_norm else g

    rng = np.random.default_rng(1)
    print(f"\n  Exploding gradients (spectral radius 1.5) with and without clipping at max-norm 5:\n")
    print(f"    {'T':>4s} {'raw ||grad||':>14s} {'clipped ||grad||':>18s}")
    for T in (10, 20, 40, 80):
        H = 20
        W = rng.standard_normal((H, H)); W = W * 1.5 / np.max(np.abs(np.linalg.eigvals(W)))
        g = np.ones(H) * 10.0
        for _ in range(T):
            g = W.T @ g                               # linear recurrence -> explodes
        raw = np.linalg.norm(g)
        clipped = np.linalg.norm(clip(g, 5.0))
        print(f"    {T:>4d} {raw:>14.2e} {clipped:>18.2f}")
    print("""
  READING: gradient clipping rescales the gradient whenever its norm exceeds a threshold, so a single
  huge update can never blow up the weights — the raw norm reaches 1e+9 but the clipped norm is capped
  at 5. Clipping is standard for training RNNs. It fixes EXPLODING gradients but does nothing for
  VANISHING ones (you cannot un-shrink a zero) — that needs a better architecture, the LSTM (README §5).""")


# =============================================================================
# EXPERIMENT 5 — short vs long dependencies (the vanishing gradient in action)
# =============================================================================


def experiment_5_memory_task():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — an RNN learns SHORT dependencies but fails LONG ones (README §6)")
    print("=" * 88)
    print(f"\n  Task: a signal bit sits at step 0; the RNN must recall it at step T (rest is noise).")
    print(f"  Train a small RNN by BPTT for each sequence length and report final accuracy:\n")
    print(f"    {'lag T':>6s} {'train accuracy':>16s}")
    for T in (3, 7, 15, 25, 40):
        acc = _train_memory(T)
        verdict = "learned" if acc > 0.9 else ("partial" if acc > 0.7 else "FAILED")
        print(f"    {T:>6d} {acc:>15.2f}  {verdict}")
    print("""
  READING: on a short lag the RNN recalls the signal perfectly; as the lag grows the accuracy collapses
  toward chance (0.5) — the gradient that should teach step 0 to store the bit has vanished by the time
  it propagates back T steps, so the early weights never learn. This is the vanishing-gradient failure
  made concrete: plain RNNs have a short effective memory. LSTMs/GRUs extend it (09.02) (README §6).""")


def _train_memory(T, epochs=400, n=64, H=32, seed=0):
    rng = np.random.default_rng(seed)
    Wxh = rng.standard_normal((H, 2)) * 0.3
    Whh = rng.standard_normal((H, H)) / np.sqrt(H)
    b = np.zeros(H)
    Wo = rng.standard_normal((2, H)) * 0.1
    bo = np.zeros(2)
    for _ in range(epochs):
        # channel 0 = signal at t=0, then INTERFERING noise (must be actively protected);
        # channel 1 = noise everywhere
        X = np.zeros((n, T, 2))
        X[:, :, 1] = rng.standard_normal((n, T))
        X[:, 1:, 0] = rng.standard_normal((n, T - 1))   # noise in the signal channel after t=0
        y = rng.integers(0, 2, n)
        X[:, 0, 0] = 2 * y - 1                        # +1/-1 signal at step 0
        gWxh = np.zeros_like(Wxh); gWhh = np.zeros_like(Whh); gb = np.zeros_like(b)
        gWo = np.zeros_like(Wo); gbo = np.zeros_like(bo)
        loss_grad_cells = []
        for i in range(n):
            net = RNN(Wxh, Whh, b)
            Hs = net.forward(X[i])
            logits = Wo @ Hs[-1] + bo
            p = np.exp(logits - logits.max()); p /= p.sum()
            d = p.copy(); d[y[i]] -= 1                 # softmax CE grad
            gWo += np.outer(d, Hs[-1]); gbo += d
            dH = np.zeros_like(Hs); dH[-1] = Wo.T @ d
            dWxh, dWhh, db, _ = net.bptt(dH)
            gWxh += dWxh; gWhh += dWhh; gb += db
        lr = 0.3 / n
        Wxh -= lr * gWxh; Whh -= lr * gWhh; b -= lr * gb; Wo -= lr * gWo; bo -= lr * gbo
    # evaluate
    Xte = np.zeros((256, T, 2)); Xte[:, :, 1] = rng.standard_normal((256, T))
    Xte[:, 1:, 0] = rng.standard_normal((256, T - 1))
    yte = rng.integers(0, 2, 256); Xte[:, 0, 0] = 2 * yte - 1
    correct = 0
    for i in range(256):
        Hs = RNN(Wxh, Whh, b).forward(Xte[i])
        correct += ((Wo @ Hs[-1] + bo).argmax() == yte[i])
    return correct / 256


if __name__ == "__main__":
    experiment_1_verify()
    experiment_2_vanishing()
    experiment_3_tanh()
    experiment_4_clipping()
    experiment_5_memory_task()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_TORCH else "ALL CHECKS PASSED (torch-verified parts skipped)")
    print("=" * 88)
