"""
07.01 — Neural Network Basics, from scratch (NumPy).

The forward pass of a multilayer perceptron, verified against PyTorch. This chapter is about
ARCHITECTURE and the forward computation; training (backprop) is 07.02 — so where a network
must be "fitted" here, we use RANDOM FEATURES (fixed random hidden layer + a least-squares
output layer), which needs no gradients and still demonstrates universality.

Claims MEASURED:
  1. a stack of LINEAR layers collapses to a single linear map (so it fails XOR)   (README §4)
  2. a nonlinearity lets the network solve XOR                                     (README §4-§5)
  3. a hand-built 2-2-1 network computes XOR; its hidden layer is linearly separable (README §5)
  4. universal approximation: a 1-hidden-layer net's error shrinks as it widens    (README §6)
  5. depth vs width: a shallow net needs far more units for an oscillatory target  (README §7)

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
# ACTIVATIONS and the MLP FORWARD PASS  (README §2-§3)
# =============================================================================


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def identity(z):
    return z


ACTS = {"relu": relu, "tanh": tanh, "identity": identity}


class MLP:
    """Forward pass only. `layers` = list of (W, b); `acts` = activation name per layer."""

    def __init__(self, layers, acts):
        self.layers = layers
        self.acts = acts

    def forward(self, X):
        a = np.asarray(X, float)
        self.pre, self.post = [], [a]
        for (W, b), act in zip(self.layers, self.acts):
            z = a @ W + b
            a = ACTS[act](z)
            self.pre.append(z)
            self.post.append(a)
        return a


def random_mlp(sizes, acts, seed=0, scale=1.0):
    """Build an MLP with random Gaussian weights (for verification / random features)."""
    rng = np.random.default_rng(seed)
    layers = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        W = rng.standard_normal((din, dout)) * scale / np.sqrt(din)
        b = rng.standard_normal(dout) * 0.1
        layers.append((W, b))
    return MLP(layers, acts)


# =============================================================================
# RANDOM FEATURES: fit the OUTPUT layer by least squares (no backprop)  (README §6)
# =============================================================================


class RandomFeatureNet:
    """Fixed random hidden layer(s), output weights fit by least squares. Demonstrates
    universal approximation without gradient descent (an 'extreme learning machine')."""

    def __init__(self, hidden_sizes, act="tanh", seed=0, scale=3.0):
        self.hidden_sizes = hidden_sizes
        self.act = act
        self.seed = seed
        self.scale = scale

    def _features(self, X):
        rng = np.random.default_rng(self.seed)
        a = np.asarray(X, float)
        d = a.shape[1]
        for h in self.hidden_sizes:
            W = rng.standard_normal((d, h)) * self.scale / np.sqrt(d)
            b = rng.standard_normal(h) * self.scale
            a = ACTS[self.act](a @ W + b)
            d = h
        return np.column_stack([a, np.ones(len(a))])      # + bias feature

    def fit(self, X, y, ridge=1e-6):
        Phi = self._features(X)
        A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
        self.w = np.linalg.solve(A, Phi.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        return self._features(X) @ self.w


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — MLP forward pass vs PyTorch")
    print("=" * 88)
    if not HAVE_TORCH:
        print("\n(PyTorch unavailable — verifying the linear-collapse identity instead)")
        # two linear layers == one: (W2 W1) x + (W2 b1 + b2)
        rng = np.random.default_rng(0)
        W1, b1 = rng.standard_normal((4, 5)), rng.standard_normal(5)
        W2, b2 = rng.standard_normal((5, 3)), rng.standard_normal(3)
        X = rng.standard_normal((10, 4))
        two = MLP([(W1, b1), (W2, b2)], ["identity", "identity"]).forward(X)
        one = X @ (W1 @ W2) + (b1 @ W2 + b2)
        print(f"  two linear layers vs one composed layer: max|diff| = {np.max(np.abs(two-one)):.2e}")
        assert np.max(np.abs(two - one)) < 1e-10
        print("  linear layers compose to a single linear map  ✓")
        print("\nAll verification checks passed.")
        return

    rng = np.random.default_rng(0)
    sizes = [6, 10, 8, 3]
    X = rng.standard_normal((20, 6)).astype(np.float64)

    # build a torch MLP and copy its weights into our MLP
    torch.manual_seed(0)
    tnet = nn.Sequential(nn.Linear(6, 10), nn.Tanh(), nn.Linear(10, 8), nn.ReLU(),
                         nn.Linear(8, 3)).double()
    layers, acts = [], []
    lin_acts = ["tanh", "relu", "identity"]
    li = 0
    for m in tnet:
        if isinstance(m, nn.Linear):
            layers.append((m.weight.detach().numpy().T.copy(),
                           m.bias.detach().numpy().copy()))
            acts.append(lin_acts[li])
            li += 1
    ours = MLP(layers, acts).forward(X)
    theirs = tnet(torch.from_numpy(X)).detach().numpy()
    print(f"""
    our forward pass vs PyTorch (same weights): max|diff| = {np.max(np.abs(ours - theirs)):.2e}
""")
    assert np.max(np.abs(ours - theirs)) < 1e-10, "forward pass must match PyTorch"
    print("  MLP forward pass matches PyTorch to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — linear layers collapse (README §4)
# =============================================================================


def experiment_1_linear_collapse():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — a stack of LINEAR layers is a single linear map (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    # deep stack of linear layers
    sizes = [2, 8, 8, 8, 1]
    net = random_mlp(sizes, ["identity"] * 4, seed=1)
    X = rng.standard_normal((100, 2))
    deep_out = net.forward(X).ravel()
    # collapse: multiply all weight matrices
    Wprod = np.eye(2)
    bprod = np.zeros(2) if False else None
    a = X.copy()
    # build the equivalent single linear map by composing
    W_eq = np.eye(2)
    b_eq = np.zeros(2)
    for (W, b) in net.layers:
        W_eq = W_eq @ W
        b_eq = b_eq @ W + b
    one_out = (X @ W_eq + b_eq).ravel()
    print(f"""
  A 5-layer network with NO activations (all identity), on 2-D inputs:

    max |deep 4-layer output - single linear map (X @ W_prod + b_prod)| = {np.max(np.abs(deep_out - one_out)):.2e}

  And a linear map cannot fit XOR. Best linear least-squares fit to XOR targets:
""")
    Xxor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    yxor = np.array([0, 1, 1, 0], float)
    Xa = np.column_stack([Xxor, np.ones(4)])
    w = np.linalg.lstsq(Xa, yxor, rcond=None)[0]
    pred = Xa @ w
    print(f"    XOR targets:     {yxor}")
    print(f"    linear predicts: {np.round(pred, 2)}   (all ~0.5 — cannot separate XOR)")
    assert np.max(np.abs(deep_out - one_out)) < 1e-9
    print("""
  READING: with no nonlinearity, composing 4 linear layers gives EXACTLY one linear map
  (W_prod = W4 W3 W2 W1). A linear map draws only straight boundaries, so it predicts ~0.5 for
  every XOR point — it cannot separate them. Stacking linear layers adds no power; the
  nonlinearity between layers is what matters (README §4).""")


# =============================================================================
# EXPERIMENT 2 — a nonlinearity solves XOR (README §4-§5)
# =============================================================================


def experiment_2_nonlinearity_xor():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — a NONLINEAR hidden layer solves XOR (README §4-§5)")
    print("=" * 88)
    Xxor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    yxor = np.array([0, 1, 1, 0], float)
    # linear model (no hidden layer)
    Xa = np.column_stack([Xxor, np.ones(4)])
    w_lin = np.linalg.lstsq(Xa, yxor, rcond=None)[0]
    acc_lin = np.mean((Xa @ w_lin >= 0.5) == yxor)
    # random-feature net: fixed random nonlinear hidden layer + least-squares output
    rf = RandomFeatureNet([20], act="tanh", seed=0, scale=3.0).fit(Xxor, yxor)
    acc_rf = np.mean((rf.predict(Xxor) >= 0.5) == yxor)
    print(f"""
    {'model':>34s} {'XOR accuracy':>14s}
    {'linear (no hidden layer)':>34s} {acc_lin:>14.2f}
    {'nonlinear hidden layer (20 units)':>34s} {acc_rf:>14.2f}

  READING: the linear model gets 2/4 = 0.5 (chance) — XOR is not linearly separable. Adding a
  nonlinear hidden layer (here fixed-random units + a fitted output, no backprop needed) lets the
  network compute nonlinear features and solve XOR perfectly. The nonlinearity is what unlocks it
  (README §4-§5).""")


# =============================================================================
# EXPERIMENT 3 — hand-built XOR network; hidden layer is separable (README §5)
# =============================================================================


def experiment_3_handbuilt_xor():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a hand-built 2-2-1 network computes XOR (README §5)")
    print("=" * 88)
    # Hidden unit 1 = OR (fires if x1+x2 >= 1); hidden unit 2 = AND (fires if x1+x2 >= 2).
    # XOR = OR AND NOT AND  ->  output = h1 - h2  (thresholded).
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])       # both hidden units sum the inputs
    b1 = np.array([-0.5, -1.5])                    # thresholds for OR and AND
    W2 = np.array([[1.0], [-1.0]])                 # h_OR - h_AND
    b2 = np.array([-0.5])
    step = lambda z: (z > 0).astype(float)

    Xxor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    yxor = np.array([0, 1, 1, 0], float)
    H = step(Xxor @ W1 + b1)                        # hidden representation
    out = step(H @ W2 + b2).ravel()

    # is the hidden representation linearly separable? (fit a linear classifier on H)
    Ha = np.column_stack([H, np.ones(4)])
    wsep = np.linalg.lstsq(Ha, yxor, rcond=None)[0]
    sep_acc = np.mean((Ha @ wsep >= 0.5) == yxor)
    print(f"""
    {'input':>10s} {'hidden (OR, AND)':>18s} {'output':>8s} {'target':>8s}""")
    for i in range(4):
        print(f"    {str(Xxor[i].astype(int)):>10s} {str(H[i].astype(int)):>18s} "
              f"{int(out[i]):>8d} {int(yxor[i]):>8d}")
    print(f"""
    network output exactly matches XOR: {np.array_equal(out, yxor)}
    hidden representation is linearly separable (linear fit on H): accuracy {sep_acc:.2f}

  READING: the hidden layer maps the 4 XOR points to (OR, AND) features. In that 2-D hidden space
  the classes ARE linearly separable — the two positive points ((0,1),(1,0)) both map to (1,0),
  the negatives to (0,0) and (1,1) — so the linear output unit solves it. Hidden layers work by
  transforming the input into a space where the problem is easy (README §5).""")


# =============================================================================
# EXPERIMENT 4 — universal approximation (README §6)
# =============================================================================


def experiment_4_universal():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — universal approximation: error shrinks as the hidden layer widens (§6)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    Xtr = np.sort(rng.uniform(-3, 3, (300, 1)), axis=0)
    f = lambda x: np.sin(2 * x[:, 0]) + 0.4 * x[:, 0]      # a wiggly target
    ytr = f(Xtr)
    Xte = np.linspace(-3, 3, 500)[:, None]
    yte = f(Xte)

    print(f"\n  Single hidden layer (fixed random units + LS output). Test MSE vs width:\n")
    print(f"    {'hidden units':>13s} {'test MSE':>10s}")
    for h in (2, 5, 20, 100, 400):
        rf = RandomFeatureNet([h], act="tanh", seed=1, scale=2.0).fit(Xtr, ytr)
        mse = np.mean((rf.predict(Xte) - yte) ** 2)
        print(f"    {h:>13d} {mse:>10.4f}")
    print("""
  READING: a network with a SINGLE hidden layer approximates the nonlinear target better and
  better as the layer widens — the test error shrinks toward zero. This is the universal
  approximation theorem in action: one hidden layer with enough units can represent any continuous
  function. (The catch — how MANY units, and whether training finds them — is Experiment 5 and
  §6-§7.)""")


# =============================================================================
# EXPERIMENT 5 — depth vs width on an oscillatory target (README §7)
# =============================================================================


def tent_fold_net(k):
    """A DEEP net computing the tent map T(x)=1-|2x-1| composed k times. Each block is a
    2-unit ReLU 'fold' that doubles the number of oscillations: T^k has 2^(k-1) humps but the
    net uses only 2k hidden units. This is the classic depth-efficiency construction (README §7).
    T(x) = 1 - relu(2x-1) - relu(1-2x)."""
    layers, acts = [], []
    for _ in range(k):
        layers.append((np.array([[2.0, -2.0]]), np.array([-1.0, 1.0])))   # 1 -> 2, relu
        acts.append("relu")
        layers.append((np.array([[-1.0], [-1.0]]), np.array([1.0])))      # 2 -> 1, identity
        acts.append("identity")
    return MLP(layers, acts)


def experiment_5_depth_vs_width():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — depth vs width: a deep FOLDING net beats a wide shallow one (README §7)")
    print("=" * 88)
    x = np.linspace(0, 1, 4000)[:, None]
    k = 5                                          # 5 fold blocks -> 2^4 = 16 oscillations
    deep = tent_fold_net(k)
    y = deep.forward(x).ravel()                    # the target: T^5 (16 oscillations)
    n_oscillations = 2 ** (k - 1)
    deep_units = 2 * k
    deep_mse = np.mean((deep.forward(x).ravel() - y) ** 2)   # trivially 0 (deep defines it)

    print(f"""
  Target: the tent map composed {k} times -> {n_oscillations} oscillations.
  The DEEP net computes it EXACTLY with {deep_units} hidden units (MSE {deep_mse:.1e}).

  A SHALLOW (single-hidden-layer) net trying to fit the SAME {n_oscillations}-oscillation function:
""")
    print(f"    {'shallow hidden units':>21s} {'MSE':>10s}")
    for h in (8, 16, 32, 64, 128, 300):
        rf = RandomFeatureNet([h], act="relu", seed=2, scale=8.0).fit(x, y)
        mse = np.mean((rf.predict(x) - y) ** 2)
        print(f"    {h:>21d} {mse:>10.4f}")
    print(f"""
  READING: the deep net folds the input {k} times, doubling oscillations per fold, so it
  represents a {n_oscillations}-oscillation function EXACTLY with just {deep_units} units. A shallow
  net needs roughly one unit per oscillation ({n_oscillations}+) and still fits it poorly with far
  more units than that. Depth is exponentially more efficient for compositional/oscillatory
  structure — the fundamental reason to go DEEP rather than merely WIDE (README §7).""")


if __name__ == "__main__":
    verify()
    experiment_1_linear_collapse()
    experiment_2_nonlinearity_xor()
    experiment_3_handbuilt_xor()
    experiment_4_universal()
    experiment_5_depth_vs_width()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
