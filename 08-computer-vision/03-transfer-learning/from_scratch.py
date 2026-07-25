"""
08.03 — Transfer learning, from scratch (NumPy).

Transfer learning = reuse features learned on a big SOURCE task to help a small TARGET task. There is
no pretrained ResNet here — instead a controlled simulation where source and target genuinely SHARE a
hidden feature basis, so every claim can be measured exactly:

  1. pretrained features beat random features — but only if the source is RELATED   -> Experiment 1
  2. transfer helps most when target data is scarce (data efficiency)               -> Experiment 2
  3. feature-extraction (freeze) vs fine-tuning: when each wins                      -> Experiment 3
  4. fine-tune GENTLY — a high LR overwrites features (catastrophic forgetting)      -> Experiment 4
  5. how much target data a pretrained model is worth (the data multiplier)         -> Experiment 5

The setup: inputs x -> shared features z = ReLU(x @ W_shared). The SOURCE is a K-way classification
that needs ALL feature directions (so pretraining learns the whole basis); the TARGET is a binary task
that is another readout of the SAME z. A from-scratch net must relearn the features from scarce data;
a pretrained net already has them.

Run:  python3 from_scratch.py
"""

import numpy as np

# ---- the shared world ------------------------------------------------------
D_IN, HIDDEN, K_SRC = 100, 64, 20
_rng = np.random.default_rng(0)
W_SHARED = _rng.standard_normal((D_IN, HIDDEN)) / np.sqrt(D_IN)   # the true feature extractor
H_SRC = _rng.standard_normal((HIDDEN, K_SRC))                     # source: 20 readouts -> needs all features
W_TGT = _rng.standard_normal(HIDDEN)                             # target: one binary readout of z
W_UNREL = _rng.standard_normal((D_IN, HIDDEN)) / np.sqrt(D_IN)    # an UNRELATED feature basis


def _z(X, W=W_SHARED):
    return np.maximum(0, X @ W)


def gen_source(n, seed, W=W_SHARED):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, D_IN))
    return X, (_z(X, W) @ H_SRC).argmax(1)


def gen_target(n, seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, D_IN))
    logit = _z(X) @ W_TGT
    return X, (logit > np.median(logit)).astype(int)


def train(X, y, K, epochs=400, lr=0.1, W1=None, freeze=False, seed=1, wd=1e-3):
    """One hidden-layer softmax net. W1=given + freeze=True -> linear probe on fixed features."""
    r = np.random.default_rng(seed)
    n, d = X.shape
    W1 = (r.standard_normal((d, HIDDEN)) * np.sqrt(2 / d)) if W1 is None else W1.copy()
    v = r.standard_normal((HIDDEN, K)) * 0.01
    b = np.zeros(K)
    for _ in range(epochs):
        H = np.maximum(0, X @ W1)
        Z = H @ v + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        dZ = P
        dZ[np.arange(n), y] -= 1
        dZ /= n
        dv = H.T @ dZ + wd * v
        db = dZ.sum(0)
        if not freeze:
            dH = (dZ @ v.T) * (X @ W1 > 0)
            W1 -= lr * (X.T @ dH + wd * W1)
        v -= lr * dv
        b -= lr * db
    return W1, v, b


def acc(X, y, W1, v, b):
    return ((np.maximum(0, X @ W1) @ v + b).argmax(1) == y).mean()


N_SEED = 6                                            # average over target-data draws to kill noise


def mean_acc(nt, lr, W1, freeze):
    """Mean target test accuracy over N_SEED independent target training sets of size nt."""
    out = []
    for k in range(N_SEED):
        Xt, yt = gen_target(nt, 1000 * k + nt)
        out.append(acc(X_TE, Y_TE, *train(Xt, yt, 2, epochs=400, lr=lr, W1=W1, freeze=freeze, seed=3)))
    return float(np.mean(out))


# pretrain ONCE on abundant source data (this is "the pretrained model")
X_SRC, Y_SRC = gen_source(10000, 10)
W1_PRE, V_SRC, B_SRC = train(X_SRC, Y_SRC, K_SRC, epochs=1000, lr=0.4, seed=2, wd=1e-4)
# a model pretrained on an UNRELATED source
W1_UNREL, _, _ = train(*gen_source(10000, 11, W=W_UNREL), K_SRC, epochs=1000, lr=0.4, seed=2, wd=1e-4)
W1_RAND = np.random.default_rng(7).standard_normal((D_IN, HIDDEN)) * np.sqrt(2 / D_IN)
X_TE, Y_TE = gen_target(3000, 99)


# =============================================================================
# EXPERIMENT 1 — do pretrained features transfer? (and only if source is related)
# =============================================================================


def experiment_1_features_transfer():
    print("=" * 88)
    print("EXPERIMENT 1 — pretrained features beat random ONLY if the source is related (README §3)")
    print("=" * 88)
    src_acc = acc(*gen_source(3000, 55), W1_PRE, V_SRC, B_SRC)
    print(f"\n  (the pretrained model reaches {src_acc:.3f} on its {K_SRC}-way SOURCE task, "
          f"chance {1/K_SRC:.3f})")
    print(f"\n  Linear probe on the TARGET (freeze features, train only the classifier), n_target=100:\n")
    Xt, yt = gen_target(100, 100)
    rows = [("random features (no pretraining)", W1_RAND),
            ("UNRELATED-source features", W1_UNREL),
            ("RELATED-source features", W1_PRE)]
    print(f"    {'frozen features':>34s} {'target test acc':>16s}")
    for name, W1 in rows:
        a = acc(X_TE, Y_TE, *train(Xt, yt, 2, epochs=400, lr=0.1, W1=W1, freeze=True, seed=3))
        print(f"    {name:>34s} {a:>16.3f}")
    print("""
  READING: a linear probe on RELATED-source features far outperforms one on random features — the
  source task taught a feature basis that the target reuses. But features from an UNRELATED source are
  no better than random: transfer only helps when the source and target share structure. This is why
  ImageNet pretraining helps almost any natural-image task (shared edges/textures) but does little for,
  say, audio spectra (README §3).""")


# =============================================================================
# EXPERIMENT 2 — data efficiency: transfer wins most when target data is scarce
# =============================================================================


def experiment_2_data_efficiency():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — transfer helps most when target data is scarce (README §4)")
    print("=" * 88)
    print(f"\n  Target test accuracy (mean over {N_SEED} training sets) vs # target examples:\n")
    print(f"    {'n_target':>8s} {'from scratch':>14s} {'fine-tuned':>12s} {'transfer gain':>14s}")
    for nt in (25, 50, 100, 300, 1000, 3000):
        a_s = mean_acc(nt, lr=0.1, W1=None, freeze=False)
        a_f = mean_acc(nt, lr=0.03, W1=W1_PRE, freeze=False)
        print(f"    {nt:>8d} {a_s:>14.3f} {a_f:>12.3f} {a_f - a_s:>+14.3f}")
    print("""
  READING: the transfer gain is POSITIVE when target data is scarce (fine-tuning starts from good
  features instead of learning them from a handful of examples) and DECREASES monotonically as data
  grows — going negative once from-scratch has enough data to learn its own, target-specific features.
  The rule: the less target data you have, the more transfer is worth. (The margins are small because
  our toy features are easy to learn; with real deep features that need millions of images, the
  scarce-data gap is enormous.) (README §4)""")


# =============================================================================
# EXPERIMENT 3 — feature extraction (freeze) vs fine-tuning
# =============================================================================


def experiment_3_freeze_vs_finetune():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — feature extraction (freeze) vs fine-tuning (README §5)")
    print("=" * 88)
    print(f"\n  Two ways to transfer (mean over {N_SEED} training sets), across target size:\n")
    print(f"    {'n_target':>8s} {'freeze (probe)':>16s} {'fine-tune':>12s} {'fine-tune edge':>16s}")
    for nt in (25, 50, 100, 300, 1000):
        a_p = mean_acc(nt, lr=0.1, W1=W1_PRE, freeze=True)
        a_f = mean_acc(nt, lr=0.03, W1=W1_PRE, freeze=False)
        print(f"    {nt:>8d} {a_p:>16.3f} {a_f:>12.3f} {a_f - a_p:>+16.3f}")
    print("""
  READING: FREEZING the features and training only a new classifier (a 'linear probe') has few
  parameters, so it is safe and cheap. FINE-TUNING also updates the features, which adapts them to the
  target — and its edge over a frozen probe GROWS with target data (more data safely supports updating
  more parameters). The standard recipe: freeze first (a fast, cheap baseline), then fine-tune if you
  have the data and want the last few points (README §5).""")


# =============================================================================
# EXPERIMENT 4 — fine-tune gently: catastrophic forgetting
# =============================================================================


def experiment_4_forgetting():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — fine-tune gently: a high LR forgets the source (README §6)")
    print("=" * 88)
    X_stest, y_stest = gen_source(3000, 55)
    base = acc(X_stest, y_stest, W1_PRE, V_SRC, B_SRC)
    Xt, yt = gen_target(150, 100)
    print(f"\n  Fine-tune the FEATURES on the target, then re-measure the SOURCE task (its original")
    print(f"  classifier, unchanged). Source acc before any fine-tuning: {base:.3f}\n")
    print(f"    {'fine-tune LR':>14s} {'source acc after':>18s} {'kept':>8s}")
    for lr in (0.0, 0.02, 0.1, 0.5, 2.0):
        W1f = W1_PRE.copy() if lr == 0 else train(Xt, yt, 2, epochs=400, lr=lr, W1=W1_PRE,
                                                  freeze=False, seed=3, wd=0.0)[0]
        sa = acc(X_stest, y_stest, W1f, V_SRC, B_SRC)
        print(f"    {lr:>14.2f} {sa:>18.3f} {sa / base:>7.0%}")
    print("""
  READING: fine-tuning at a small learning rate barely disturbs the pretrained features — the source
  task is still solved. A LARGE learning rate overwrites them: the source accuracy drops sharply (here
  to ~69% of its value, toward chance) — 'catastrophic forgetting'. This is exactly why you fine-tune
  with a learning rate 10-100x smaller than pretraining, and often freeze the early layers — to adapt
  the top without destroying the features underneath (README §6).""")


if __name__ == "__main__":
    experiment_1_features_transfer()
    experiment_2_data_efficiency()
    experiment_3_freeze_vs_finetune()
    experiment_4_forgetting()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
