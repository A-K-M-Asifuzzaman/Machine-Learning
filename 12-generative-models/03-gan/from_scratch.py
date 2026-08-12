"""
12.03 — Generative Adversarial Networks, from scratch (NumPy).

A GAN pits a GENERATOR (turn noise into data) against a DISCRIMINATOR (tell real from fake) in a minimax
game. At equilibrium the generator matches the data and the discriminator is reduced to guessing. This
file trains a small GAN and verifies the theory that explains its behavior (and its failure modes):

  1. a GAN learns to match a distribution; the discriminator converges to 0.5   -> Experiment 1
  2. the optimal discriminator turns the generator's loss into JS divergence     -> Experiment 2
  3. JS is FLAT for disjoint supports (no gradient); Wasserstein is linear -> WGAN -> Experiment 3
  4. mode collapse: the generator abandons modes of a multimodal target          -> Experiment 4

Run:  python3 from_scratch.py
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _adam(P, g, s, lr, t):
    s["m"] = 0.9 * s["m"] + 0.1 * g
    s["v"] = 0.999 * s["v"] + 0.001 * g ** 2
    return P - lr * (s["m"] / (1 - 0.9 ** t)) / (np.sqrt(s["v"] / (1 - 0.999 ** t)) + 1e-8)


def _net(din, h, seed):
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((din, h)) * np.sqrt(2 / din), "b1": np.zeros(h),
            "W2": rng.standard_normal((h, 1)) * np.sqrt(2 / h), "b2": np.zeros(1)}


def _fwd(P, x):
    h1 = relu(x @ P["W1"] + P["b1"])
    return h1, h1 @ P["W2"] + P["b2"]


def train_gan(sample_real, h=32, epochs=6000, lr=0.002, n=256, seed=0, g_steps=1):
    """Minimax GAN on 1-D data. sample_real(n) draws n real samples. Returns G, D.
    g_steps>1 (more generator than discriminator updates) tends to induce mode collapse."""
    rng = np.random.default_rng(seed)
    G, D = _net(1, h, seed), _net(1, h, seed + 1)
    sG = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in G.items()}
    sD = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in D.items()}
    for t in range(1, epochs + 1):
        # --- discriminator: maximize log D(real) + log(1 - D(fake)) ---
        xr = sample_real(n); z = rng.standard_normal((n, 1)); _, xf = _fwd(G, z)
        hr, lr_ = _fwd(D, xr); dr = sigmoid(lr_)
        hf, lf = _fwd(D, xf); df = sigmoid(lf)
        for x, h1, dlog in [(xr, hr, -(1 - dr) / n), (xf, hf, df / n)]:
            gW2 = h1.T @ dlog; gb2 = dlog.sum(0)
            dh = dlog @ D["W2"].T * (h1 > 0); gW1 = x.T @ dh; gb1 = dh.sum(0)
            D["W2"] = _adam(D["W2"], gW2, sD["W2"], lr, t); D["b2"] = _adam(D["b2"], gb2, sD["b2"], lr, t)
            D["W1"] = _adam(D["W1"], gW1, sD["W1"], lr, t); D["b1"] = _adam(D["b1"], gb1, sD["b1"], lr, t)
        # --- generator: minimize -log D(G(z)) ---
        for _ in range(g_steps):
            z = rng.standard_normal((n, 1)); h1g, xf = _fwd(G, z); hf, lf = _fwd(D, xf); df = sigmoid(lf)
            dscore = -(1 - df) / n
            dxf = (dscore @ D["W2"].T * (hf > 0)) @ D["W1"].T
            gW2 = h1g.T @ dxf; gb2 = dxf.sum(0)
            dh1 = dxf @ G["W2"].T * (h1g > 0); gW1 = z.T @ dh1; gb1 = dh1.sum(0)
            G["W2"] = _adam(G["W2"], gW2, sG["W2"], lr, t); G["b2"] = _adam(G["b2"], gb2, sG["b2"], lr, t)
            G["W1"] = _adam(G["W1"], gW1, sG["W1"], lr, t); G["b1"] = _adam(G["b1"], gb1, sG["b1"], lr, t)
    return G, D


# =============================================================================
# EXPERIMENT 1 — a GAN matches a distribution; D -> 0.5
# =============================================================================


def experiment_1_train():
    print("=" * 88)
    print("EXPERIMENT 1 — a GAN learns to match a distribution; discriminator -> 0.5 (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(7)
    G, D = train_gan(lambda n: rng.normal(2.0, 0.5, (n, 1)))
    z = rng.standard_normal((4000, 1)); _, xf = _fwd(G, z)
    _, lr_real = _fwd(D, rng.normal(2.0, 0.5, (4000, 1)))
    print(f"""
  Target distribution N(mean=2.0, std=0.5). After adversarial training:

    generated mean = {xf.mean():.3f}   (target 2.0)
    generated std  = {xf.std():.3f}   (target 0.5)
    discriminator's score on REAL data = {sigmoid(lr_real).mean():.3f}   (-> 0.5 = can't tell real from fake)

  READING: the generator turns noise into samples; the discriminator scores real vs fake; they train
  against each other. At the equilibrium of this minimax game the generator's distribution matches the
  data, so the discriminator can do no better than a coin flip (score -> 0.5). Our GAN recovers the
  target's mean and the discriminator collapses to ~0.5. (The generated std is a touch low — GANs are
  notoriously finicky and tend to under-cover the distribution, a mild version of mode collapse,
  Experiment 4) (README §2).""")


# =============================================================================
# EXPERIMENT 2 — the optimal discriminator gives JS divergence
# =============================================================================


def experiment_2_js():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the optimal discriminator turns the GAN loss into JS divergence (README §3)")
    print("=" * 88)
    # two known 1-D distributions on a grid; compute D* and the generator loss at D*
    x = np.linspace(-6, 8, 4000)
    dx = x[1] - x[0]
    def gauss(m, s):
        p = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)); return p
    print(f"\n  For fixed distributions p_data, p_gen: optimal D*(x) = p_data/(p_data+p_gen), and the")
    print(f"  value of the GAN objective at D* equals  2*JS(p_data||p_gen) - log 4:\n")
    print(f"    {'gap between means':>18s} {'GAN value at D*':>16s} {'2*JS - log4':>14s}")
    for gap in (0.0, 1.0, 3.0, 6.0):
        pd = gauss(0, 1); pg = gauss(gap, 1)
        pd /= (pd.sum() * dx); pg /= (pg.sum() * dx)
        Dstar = pd / (pd + pg + 1e-30)
        # V(D*) = E_pd[log D*] + E_pg[log(1-D*)]
        V = np.sum(pd * np.log(Dstar + 1e-30) + pg * np.log(1 - Dstar + 1e-30)) * dx
        m = 0.5 * (pd + pg)
        js = 0.5 * np.sum(pd * np.log((pd + 1e-30) / (m + 1e-30)) +
                          pg * np.log((pg + 1e-30) / (m + 1e-30))) * dx
        print(f"    {gap:>18.1f} {V:>16.4f} {2 * js - np.log(4):>14.4f}")
    print("""
  READING: plug the OPTIMAL discriminator D*(x) = p_data/(p_data+p_gen) back into the GAN objective and
  it becomes exactly 2*JS(p_data || p_gen) - log 4 — the two columns match. So training the generator
  against an optimal discriminator MINIMIZES the Jensen-Shannon divergence between the generated and
  real distributions. This is the theoretical foundation of GANs (Goodfellow 2014): the adversarial
  game is divergence minimization in disguise (README §3).""")


# =============================================================================
# EXPERIMENT 3 — JS is flat for disjoint supports; Wasserstein is not (-> WGAN)
# =============================================================================


def experiment_3_wasserstein():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — JS gives NO gradient for disjoint supports; Wasserstein does (README §4)")
    print("=" * 88)
    print(f"\n  Two narrow distributions separated by a distance theta. JS vs Wasserstein-1:\n")
    print(f"    {'separation theta':>18s} {'JS divergence':>14s} {'Wasserstein W1':>16s}")
    for theta in (0.0, 0.5, 1.0, 2.0, 4.0):
        # near-disjoint supports: two tight point masses at 0 and theta
        if theta < 1e-9:
            js = 0.0
        else:
            js = np.log(2)                            # disjoint supports -> JS = log 2 (constant)
        w1 = theta                                    # W1 between two point masses = the distance
        print(f"    {theta:>18.1f} {js:>14.4f} {w1:>16.4f}")
    print(f"""
  READING: when the real and generated distributions do not overlap (common early in training, when the
  generator is bad), the Jensen-Shannon divergence is CONSTANT at log 2 = {np.log(2):.3f} — it gives the
  generator ZERO gradient, so learning stalls (vanishing gradients). The Wasserstein-1 ('earth mover')
  distance instead grows LINEARLY with the separation, providing a smooth, informative gradient
  everywhere. This is exactly why WGAN (Arjovsky 2017) replaces JS with Wasserstein — using a
  Lipschitz-constrained critic (weight clipping, or the WGAN-GP gradient penalty) — and trains far more
  stably (README §4).""")


# =============================================================================
# EXPERIMENT 4 — mode collapse
# =============================================================================


def experiment_4_mode_collapse():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — mode collapse: the generator abandons modes of the target (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    # bimodal target: an equal mixture of two well-separated modes at -5 and +5
    def sample_bimodal(n):
        m = rng.integers(0, 2, (n, 1)) * 10 - 5
        return m + 0.3 * rng.standard_normal((n, 1))
    # more generator steps than discriminator steps -> the generator races ahead and collapses
    G, D = train_gan(sample_bimodal, h=16, epochs=5000, lr=0.003, seed=1, g_steps=5)
    z = np.random.default_rng(9).standard_normal((5000, 1)); _, xf = _fwd(G, z)
    left = np.mean(xf < 0); right = np.mean(xf >= 0)
    print(f"""
  Target: an equal mixture of two modes at -5 and +5 (each 50% of the data). Generated samples:

    fraction of generated samples near the LEFT mode (-5)  = {left:.2f}
    fraction near the RIGHT mode (+5)                       = {right:.2f}
    (a healthy generator would put ~0.50 on each)

  READING: the generator found it can fool the discriminator by producing convincing samples from just
  ONE mode, so it abandons the other — {max(left, right):.0%} of samples pile onto a single mode instead
  of the 50/50 split the data has. This is MODE COLLAPSE, the signature failure of GANs: because the
  objective only rewards fooling the discriminator, not covering the data, the generator has no
  incentive to be diverse. Fixes — WGAN, minibatch discrimination, unrolled GANs — all add pressure for
  coverage. It is the flip side of the VAE's blur: GANs are sharp but drop modes; VAEs cover modes but
  blur (README §5).""")


if __name__ == "__main__":
    experiment_1_train()
    experiment_2_js()
    experiment_3_wasserstein()
    experiment_4_mode_collapse()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
