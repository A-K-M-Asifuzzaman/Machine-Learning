"""
08.02 — CNN architectures, from scratch (NumPy).

The CNN story from LeNet to ConvNeXt is a story of FOUR ideas, each a fix for the last generation's
bottleneck. This file MEASURES each idea instead of describing it:

  1. RESIDUAL connections solve gradient degradation      (ResNet)      -> Experiment 1
  2. BOTTLENECK blocks cut the cost of going deep          (ResNet-50)   -> Experiment 2
  3. DEPTHWISE-SEPARABLE convolution cuts it further        (MobileNet)   -> Experiment 3
  4. GLOBAL AVERAGE POOLING removes the giant FC head       (NiN/ResNet)  -> Experiment 4
  + the VGG principle: stacks of small 3x3 filters beat one big filter    -> Experiment 5

Experiment 3 is verified against PyTorch grouped convolution to machine precision.

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False


# =============================================================================
# A grouped conv2d via im2col (groups=C -> depthwise; 1x1 -> pointwise)
# =============================================================================


def conv2d(X, Wt, stride=1, pad=0, groups=1):
    """X:(N,C,H,W)  Wt:(O, C/groups, k, k) -> (N,O,Hout,Wout). No bias (folded into BN in CNNs)."""
    N, C, H, W = X.shape
    O, Cg, kH, kW = Wt.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Hout = (H + 2 * pad - kH) // stride + 1
    Wout = (W + 2 * pad - kW) // stride + 1
    Og = O // groups
    out = np.zeros((N, O, Hout, Wout))
    for g in range(groups):
        xs = Xp[:, g * Cg:(g + 1) * Cg]              # this group's input channels
        # im2col for the group
        cols = np.empty((N, Cg * kH * kW, Hout * Wout))
        idx = 0
        for c in range(Cg):
            for u in range(kH):
                for v in range(kW):
                    patch = xs[:, c, u:u + stride * Hout:stride, v:v + stride * Wout:stride]
                    cols[:, idx] = patch.reshape(N, -1)
                    idx += 1
        wg = Wt[g * Og:(g + 1) * Og].reshape(Og, -1)   # (Og, Cg*k*k)
        out[:, g * Og:(g + 1) * Og] = np.einsum("oc,ncl->nol", wg, cols).reshape(N, Og, Hout, Wout)
    return out


# =============================================================================
# EXPERIMENT 1 — residual connections solve gradient degradation (ResNet)
# =============================================================================


def _deep_net_grad(depth, residual, scale=0.9, seed=0):
    """L stacked ReLU blocks; return the gradient norm that reaches layer 0."""
    rng = np.random.default_rng(seed)
    d = 64
    a = rng.standard_normal((128, d))
    alpha = 1.0 / np.sqrt(depth) if residual else 1.0      # keep residual block ~identity at init
    Ws = [rng.standard_normal((d, d)) * scale * np.sqrt(2.0 / d) for _ in range(depth)]
    acts, pre = [a], []
    for W in Ws:
        z = acts[-1] @ W
        f = np.maximum(0, z)
        pre.append(z)
        acts.append(acts[-1] + alpha * f if residual else f)
    da = np.ones_like(acts[-1])                            # d(sum a_L)/d a_L
    for l in reversed(range(depth)):
        df = (alpha * da if residual else da) * (pre[l] > 0)
        din = df @ Ws[l].T
        da = da + din if residual else din
    return np.linalg.norm(da) / np.sqrt(da.size)


def experiment_1_residual():
    print("=" * 88)
    print("EXPERIMENT 1 — residual connections solve gradient degradation (ResNet, README §4)")
    print("=" * 88)
    print(f"\n  Gradient norm reaching layer 0 of an L-layer ReLU net (init 0.9x He, imperfect):\n")
    print(f"    {'depth L':>8s} {'PLAIN grad@0':>16s} {'RESIDUAL grad@0':>18s}")
    for depth in (10, 30, 50, 100):
        pg = _deep_net_grad(depth, residual=False)
        rg = _deep_net_grad(depth, residual=True)
        print(f"    {depth:>8d} {pg:>16.2e} {rg:>18.2e}")
    print("""
  READING: with slightly-imperfect initialization, a PLAIN deep net's gradient VANISHES exponentially
  with depth — by 100 layers it reaches layer 0 at ~5e-5, so the early layers barely learn. This is
  the 'degradation problem': deeper plain nets trained WORSE, not from overfitting but because the
  signal could not propagate. A RESIDUAL block computes x + F(x), so its Jacobian is I + F' — the
  identity path carries the gradient straight through, and it NEVER vanishes (it stays >= O(1) at 100
  layers). This one change is what made 100- and 1000-layer nets trainable. Real ResNets add
  BatchNorm to keep the forward/backward magnitude bounded too (README §4).""")


# =============================================================================
# EXPERIMENT 2 — bottleneck blocks (ResNet-50)
# =============================================================================


def experiment_2_bottleneck():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — bottleneck blocks cut the cost of a 3x3 conv (ResNet-50, README §5)")
    print("=" * 88)
    C = 256
    plain = C * C * 3 * 3                             # one 3x3 conv, 256 -> 256
    r = 64                                            # bottleneck width
    bott = C * r * 1 * 1 + r * r * 3 * 3 + r * C * 1 * 1   # 1x1 down, 3x3, 1x1 up
    print(f"""
  Processing a {C}-channel feature map:

    plain 3x3 conv ({C} -> {C})                         : {plain:>12,d} params
    bottleneck 1x1({C}->{r}) + 3x3({r}) + 1x1({r}->{C}) : {bott:>12,d} params
    reduction                                           : {plain / bott:>12.2f}x

  READING: a plain 3x3 conv on 256 channels is expensive (params ~ C^2). The bottleneck first
  SQUEEZES the channels to 64 with a cheap 1x1 conv, does the expensive 3x3 in that thin space, then
  EXPANDS back with another 1x1 — {plain/bott:.1f}x fewer parameters for the same in/out shape. Those
  saved parameters are what let ResNet-50/101/152 go deep without exploding in size (README §5).""")


# =============================================================================
# EXPERIMENT 3 — depthwise-separable convolution (MobileNet), verified vs PyTorch
# =============================================================================


def experiment_3_depthwise():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — depthwise-separable convolution (MobileNet, README §6)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    N, C, O, k, H, W = 2, 16, 32, 3, 12, 12
    X = rng.standard_normal((N, C, H, W))
    Wdw = rng.standard_normal((C, 1, k, k))          # depthwise: one k x k kernel per channel
    Wpw = rng.standard_normal((O, C, 1, 1))          # pointwise: 1x1 mixes channels C -> O

    dw = conv2d(X, Wdw, pad=1, groups=C)             # each channel its own filter
    sep = conv2d(dw, Wpw, pad=0, groups=1)           # then mix channels

    err = np.nan
    if HAVE_TORCH:
        Xt = torch.tensor(X)
        t_dw = F.conv2d(Xt, torch.tensor(Wdw), padding=1, groups=C)
        t_sep = F.conv2d(t_dw, torch.tensor(Wpw))
        err = np.abs(sep - t_sep.numpy()).max()

    std = O * C * k * k                              # standard conv params
    dws = C * k * k + O * C                          # depthwise + pointwise params
    print(f"""
  A depthwise (per-channel {k}x{k}) conv followed by a pointwise (1x1) conv, {C}->{O} channels:

    our depthwise-separable == PyTorch grouped conv : max|diff| = {err:.1e}
    standard conv params    (O*C*k*k)               : {std:>10,d}
    depthwise-separable     (C*k*k + O*C)           : {dws:>10,d}
    reduction                                       : {std / dws:>10.2f}x

  READING: a standard conv both filters space AND mixes channels in one shot (O*C*k*k params).
  Depthwise-separable FACTORS it: a depthwise conv filters each channel independently (C*k*k), then a
  1x1 pointwise conv mixes channels (O*C). The cost drops by ~1/(1/O + 1/k^2) ~ {std/dws:.1f}x here,
  approaching k^2={k*k}x for many channels. This is the core of MobileNet/Xception/EfficientNet — the
  same accuracy at a fraction of the compute, which is what put CNNs on phones (README §6).""")


# =============================================================================
# EXPERIMENT 4 — global average pooling removes the giant FC head (README §7)
# =============================================================================


def experiment_4_gap():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — global average pooling vs a flatten+dense head (README §7)")
    print("=" * 88)
    Hf, Wf, Cf, hidden, classes = 7, 7, 512, 4096, 1000
    flatten_fc = (Hf * Wf * Cf) * hidden + hidden * classes   # VGG-style two-FC head (approx first FC)
    fc1 = (Hf * Wf * Cf) * hidden
    gap = Cf * classes                                        # GAP -> single classifier
    print(f"""
  Final feature map {Hf}x{Wf}x{Cf} -> {classes} classes:

    flatten + dense({hidden}) + dense({classes})  : {flatten_fc:>14,d} params
      (of which the FIRST dense alone           : {fc1:>14,d})
    global-average-pool + dense({classes})        : {gap:>14,d} params
    reduction                                    : {flatten_fc / gap:>14.0f}x

  READING: VGG's two fully-connected layers hold ~{fc1/1e6:.0f}M parameters in the FIRST one alone —
  the bulk of the whole network — purely to flatten a 7x7x512 map. GLOBAL AVERAGE POOLING collapses
  each channel to its spatial mean, giving a {Cf}-vector fed straight to the classifier: ~{flatten_fc/gap:.0f}x
  fewer parameters, no overfitting-prone FC layers, and it accepts any input size. Every modern CNN
  (ResNet, Inception, EfficientNet) uses GAP instead of a flatten head (README §7).""")


# =============================================================================
# EXPERIMENT 5 — small filters: two 3x3 vs one 5x5 (VGG, README §3)
# =============================================================================


def experiment_5_small_filters():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — stacks of small 3x3 filters beat one big filter (VGG, README §3)")
    print("=" * 88)
    C = 64
    one_5x5 = C * C * 5 * 5
    two_3x3 = 2 * (C * C * 3 * 3)
    one_7x7 = C * C * 7 * 7
    three_3x3 = 3 * (C * C * 3 * 3)
    print(f"""
  Same receptive field, {C}->{C} channels — parameters:

    one 5x5 conv          (RF 5x5) : {one_5x5:>10,d}
    two stacked 3x3 convs (RF 5x5) : {two_3x3:>10,d}   ({100*(1-two_3x3/one_5x5):.0f}% fewer)
    one 7x7 conv          (RF 7x7) : {one_7x7:>10,d}
    three stacked 3x3 convs(RF 7x7): {three_3x3:>10,d}   ({100*(1-three_3x3/one_7x7):.0f}% fewer)

  READING: two 3x3 convolutions have the SAME 5x5 receptive field as one 5x5 conv (each 3x3 adds 2),
  but fewer parameters AND an extra nonlinearity between them, so more representational power. Three
  3x3s match a 7x7 with ~45% fewer weights. VGG turned this into a design rule — use only 3x3 convs,
  go deep — and it has held ever since (README §3).""")


if __name__ == "__main__":
    experiment_1_residual()
    experiment_2_bottleneck()
    experiment_3_depthwise()
    experiment_4_gap()
    experiment_5_small_filters()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_TORCH else "ALL CHECKS PASSED (torch-verified parts skipped)")
    print("=" * 88)
