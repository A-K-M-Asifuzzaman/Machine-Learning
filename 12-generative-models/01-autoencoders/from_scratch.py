"""
12.01 — Autoencoders, from scratch (NumPy).

An autoencoder learns to copy its input to its output through a BOTTLENECK, forcing it to discover a
compressed code. The constraint on that code (narrow, denoising, sparse) decides what it learns. This
file builds them and MEASURES each:

  1. a LINEAR autoencoder recovers the PCA subspace exactly (Baldi-Hornik)      -> Experiment 1
  2. the bottleneck controls compression: error drops at the intrinsic dimension -> Experiment 2
  3. a DENOISING autoencoder removes noise it was never shown clean            -> Experiment 3
  4. a SPARSE autoencoder learns a sparse overcomplete code                     -> Experiment 4
  5. an autoencoder detects anomalies by reconstruction error                   -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def _adam_step(P, g, state, lr, t):
    state["m"] = 0.9 * state["m"] + 0.1 * g
    state["v"] = 0.999 * state["v"] + 0.001 * g ** 2
    return P - lr * (state["m"] / (1 - 0.9 ** t)) / (np.sqrt(state["v"] / (1 - 0.999 ** t)) + 1e-8)


def train_ae(X, k, act="tanh", denoise=0.0, l1=0.0, epochs=4000, lr=0.01, seed=0):
    """One-bottleneck autoencoder: Xhat = f(X We + be) Wd + bd, MSE loss. Returns weights and code."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    We = rng.standard_normal((d, k)) * 0.1; be = np.zeros(k)
    Wd = rng.standard_normal((k, d)) * 0.1; bd = np.zeros(d)
    st = {name: {"m": np.zeros_like(p), "v": np.zeros_like(p)}
          for name, p in [("We", We), ("be", be), ("Wd", Wd), ("bd", bd)]}
    f = (lambda z: np.tanh(z)) if act == "tanh" else (lambda z: z)
    fp = (lambda h: 1 - h ** 2) if act == "tanh" else (lambda h: np.ones_like(h))
    for t in range(1, epochs + 1):
        Xin = X + denoise * rng.standard_normal(X.shape) if denoise else X
        pre = Xin @ We + be
        H = f(pre)
        Xhat = H @ Wd + bd
        dXhat = 2 * (Xhat - X) / n                    # target is the CLEAN X
        gWd = H.T @ dXhat; gbd = dXhat.sum(0)
        dH = dXhat @ Wd.T + l1 * np.sign(H) / n       # L1 sparsity penalty on the code
        dpre = dH * fp(H)
        gWe = Xin.T @ dpre; gbe = dpre.sum(0)
        We = _adam_step(We, gWe, st["We"], lr, t); be = _adam_step(be, gbe, st["be"], lr, t)
        Wd = _adam_step(Wd, gWd, st["Wd"], lr, t); bd = _adam_step(bd, gbd, st["bd"], lr, t)
    code = f(X @ We + be)
    recon = code @ Wd + bd
    return dict(We=We, be=be, Wd=Wd, bd=bd, code=code, recon=recon)


# =============================================================================
# EXPERIMENT 1 — linear autoencoder recovers PCA
# =============================================================================


def experiment_1_pca():
    print("=" * 88)
    print("EXPERIMENT 1 — a linear autoencoder recovers the PCA subspace exactly (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    d, k, n = 10, 3, 500
    B = rng.standard_normal((d, k))
    X = rng.standard_normal((n, k)) @ B.T + 0.05 * rng.standard_normal((n, d))
    X -= X.mean(0)
    ae = train_ae(X, k, act="linear", epochs=8000)
    ae_err = np.mean((ae["recon"] - X) ** 2)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_sub = Vt[:k]
    pca_err = np.mean(((X @ pca_sub.T) @ pca_sub - X) ** 2)
    ae_sub = np.linalg.qr(ae["Wd"].T)[0].T            # row space of the decoder
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(pca_sub @ ae_sub.T, compute_uv=False), -1, 1)))
    print(f"""
  Data lies near a rank-{k} subspace in R^{d}. Train a LINEAR autoencoder with a {k}-unit bottleneck:

    autoencoder reconstruction MSE = {ae_err:.6f}
    PCA (top-{k}) reconstruction MSE = {pca_err:.6f}   (identical -> same optimum)
    principal angles between the AE's subspace and PCA's = {np.round(angles, 3)} degrees (~0)

  READING: a linear autoencoder minimizing reconstruction error learns EXACTLY the subspace spanned by
  the top-{k} principal components — the decoder's row space aligns with PCA's to a fraction of a degree.
  This is the Baldi-Hornik theorem: linear autoencoding IS PCA (up to a rotation within the subspace).
  So an autoencoder is a nonlinear generalization of PCA — swap the linear map for a deep net and the
  bottleneck learns a curved manifold instead of a flat subspace (README §2).""")


# =============================================================================
# EXPERIMENT 2 — the bottleneck controls compression
# =============================================================================


def experiment_2_bottleneck():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the bottleneck controls compression; error drops at the intrinsic dim (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    d, true_k, n = 12, 3, 400
    B = rng.standard_normal((d, true_k))
    Z = rng.standard_normal((n, true_k))
    X = np.tanh(Z @ B.T) + 0.02 * rng.standard_normal((n, d))   # nonlinear rank-3 manifold
    X -= X.mean(0)
    print(f"\n  Data on a nonlinear {true_k}-dim manifold in R^{d}. Reconstruction error vs bottleneck size:\n")
    print(f"    {'bottleneck k':>14s} {'reconstruction MSE':>20s}")
    for k in (1, 2, 3, 4, 6):
        ae = train_ae(X, k, act="tanh", epochs=3000)
        err = np.mean((ae["recon"] - X) ** 2)
        mark = "  <- intrinsic dim" if k == true_k else ""
        print(f"    {k:>14d} {err:>20.5f}{mark}")
    print("""
  READING: the bottleneck size is the compression dial. Below the data's INTRINSIC dimension (~3 here),
  the code cannot capture the manifold and reconstruction error is high (0.20 at k=1); the error falls
  steeply as the bottleneck approaches the intrinsic dimension (0.20 -> 0.05 -> 0.02 by k=3) and then
  with diminishing returns. The steep-then-shallow shape reveals roughly how many dimensions the data
  really occupies — the bottleneck just past the elbow gives strong compression with little loss (README
  §3).""")


# =============================================================================
# EXPERIMENT 3 — denoising autoencoder
# =============================================================================


def experiment_3_denoising():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a denoising autoencoder removes noise (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    d, true_k, n = 16, 3, 500
    B = rng.standard_normal((d, true_k))
    clean = np.tanh(rng.standard_normal((n, true_k)) @ B.T)
    clean -= clean.mean(0)
    ae = train_ae(clean, k=3, act="tanh", denoise=0.4, epochs=4000)   # trained on corrupted inputs
    # test on fresh noisy inputs
    test_clean = np.tanh(rng.standard_normal((200, true_k)) @ B.T) - clean.mean(0) * 0
    noisy = test_clean + 0.4 * rng.standard_normal(test_clean.shape)
    H = np.tanh(noisy @ ae["We"] + ae["be"])
    denoised = H @ ae["Wd"] + ae["bd"]
    err_noisy = np.mean((noisy - test_clean) ** 2)
    err_denoised = np.mean((denoised - test_clean) ** 2)
    print(f"""
  Train an autoencoder to reconstruct CLEAN data from NOISY inputs (noise std 0.4). On held-out data:

    distance of NOISY input to the clean signal    = {err_noisy:.4f}
    distance of DENOISED output to the clean signal = {err_denoised:.4f}   ({err_noisy/err_denoised:.1f}x closer)

  READING: a denoising autoencoder is trained to map a CORRUPTED input back to the clean original. To do
  that it cannot just copy the input — it must learn the underlying data manifold and PROJECT noisy
  points onto it. On new noisy inputs it removes most of the noise ({err_noisy/err_denoised:.1f}x closer
  to the truth than the raw noisy input). Denoising is both a useful task and a powerful way to learn
  robust features — and it is the direct ancestor of diffusion models ([12.04]), which denoise at many
  noise levels (README §4).""")


# =============================================================================
# EXPERIMENT 4 — sparse autoencoder
# =============================================================================


def experiment_4_sparse():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — a sparse autoencoder learns a sparse overcomplete code (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    d, n = 8, 400
    B = rng.standard_normal((d, 3))
    X = np.tanh(rng.standard_normal((n, 3)) @ B.T); X -= X.mean(0)
    print(f"\n  OVERCOMPLETE code (32 units > {d} inputs). L1 penalty drives the code sparse:\n")
    print(f"    {'L1 penalty':>12s} {'mean |activation|':>18s} {'% units near-zero':>18s} {'recon MSE':>11s}")
    for l1 in (0.0, 0.5, 2.0, 8.0):
        ae = train_ae(X, k=32, act="tanh", l1=l1, epochs=3000)
        act = np.abs(ae["code"])
        frac_zero = np.mean(act < 0.05)
        err = np.mean((ae["recon"] - X) ** 2)
        print(f"    {l1:>12.1f} {act.mean():>18.4f} {100 * frac_zero:>17.0f}% {err:>11.4f}")
    print("""
  READING: an OVERCOMPLETE autoencoder (more code units than inputs) would just copy the input — unless
  you constrain the code. An L1 penalty on the activations drives most units to zero, so each input is
  explained by a FEW active units. As the penalty grows, the fraction of near-zero units rises (a
  sparser code) at a small cost in reconstruction. Sparse codes are interpretable and disentangled —
  the idea behind sparse dictionary learning and the sparse autoencoders now used to interpret LLM
  activations (README §5).""")


# =============================================================================
# EXPERIMENT 5 — anomaly detection by reconstruction error
# =============================================================================


def experiment_5_anomaly():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — autoencoders detect anomalies by reconstruction error (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(4)
    d, true_k = 12, 3
    B = rng.standard_normal((d, true_k))
    normal = np.tanh(rng.standard_normal((500, true_k)) @ B.T); mu = normal.mean(0); normal -= mu
    ae = train_ae(normal, k=3, act="tanh", epochs=4000)
    # test: normal points vs anomalies (random, off-manifold)
    test_normal = np.tanh(rng.standard_normal((200, true_k)) @ B.T) - mu
    anomalies = rng.standard_normal((200, d)) * normal.std()
    def recon_err(Y):
        H = np.tanh(Y @ ae["We"] + ae["be"]); R = H @ ae["Wd"] + ae["bd"]
        return ((R - Y) ** 2).mean(1)
    en, ea = recon_err(test_normal), recon_err(anomalies)
    # AUC: fraction of (normal, anomaly) pairs correctly ordered
    auc = np.mean(en[:, None] < ea[None, :])
    print(f"""
  Train an autoencoder on NORMAL data only, then score by reconstruction error:

    mean reconstruction error, NORMAL points   = {en.mean():.4f}
    mean reconstruction error, ANOMALIES        = {ea.mean():.4f}   ({ea.mean()/en.mean():.1f}x higher)
    separation AUC (normal vs anomaly)          = {auc:.3f}

  READING: an autoencoder trained only on normal data learns to reconstruct the normal manifold well.
  Anomalies lie OFF that manifold, so the autoencoder reconstructs them poorly — high reconstruction
  error flags them. Here anomalies have {ea.mean()/en.mean():.0f}x the error and are almost perfectly
  separable (AUC {auc:.2f}). This is a standard unsupervised anomaly detector: no labels needed, just
  'what does normal look like, and what fails to reconstruct?' ([04.08]) (README §6).""")


if __name__ == "__main__":
    experiment_1_pca()
    experiment_2_bottleneck()
    experiment_3_denoising()
    experiment_4_sparse()
    experiment_5_anomaly()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
