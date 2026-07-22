"""
00.05 — Information Theory from Scratch
=======================================

Entropy, cross-entropy, KL, and mutual information implemented from their definitions,
plus the experiments that turn each one from a formula into a fact you can see.

Implemented here
----------------
    entropy, joint_entropy, conditional_entropy          README §3, §5
    cross_entropy                                        README §6
    kl_divergence, js_divergence                         README §7, §10
    mutual_information, information_gain                 README §9
    perplexity                                           README §13
    huffman_code                                         README §4 — entropy IS a code length
    max_entropy_discrete                                 README §11

Run it
------
    python from_scratch.py

Verifies against scipy and sklearn, then runs five experiments:
  1. Shannon's source coding theorem: H <= L < H+1, verified with real Huffman codes
  2. Cross-entropy = H(p) + KL(p||q), and why your loss cannot reach zero
  3. Forward vs reverse KL: mode-covering vs mode-seeking, fitted and shown
  4. Mutual information catches dependence that correlation is blind to
  5. Information gain is what a decision tree actually maximizes

Reference: README.md sections 3-13.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np

EPS = 1e-300          # guards log(0); see 00.06 for why this is not the real fix


# =============================================================================
# CORE QUANTITIES  (README §3, §5-§10)
# =============================================================================


def entropy(p: np.ndarray, base: float = math.e) -> float:
    """H(p) = -sum p log p.  README §3

    The convention 0 log 0 = 0 is not a hack: lim_{p->0} p log p = 0, so the limit exists
    and zero-probability outcomes contribute nothing. We implement it by masking rather
    than by adding an epsilon, which would silently bias the result.
    """
    p = np.asarray(p, dtype=float)
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log(nonzero)) / math.log(base))


def joint_entropy(p_joint: np.ndarray, base: float = math.e) -> float:
    """H(X, Y) over a joint probability table."""
    return entropy(np.asarray(p_joint, dtype=float).ravel(), base)


def conditional_entropy(p_joint: np.ndarray, base: float = math.e) -> float:
    """H(Y | X) = H(X, Y) - H(X), for a joint table with X on axis 0.  README §5

    Uses the chain rule rather than the double sum — same answer, less code, and it makes
    the identity H(X,Y) = H(X) + H(Y|X) manifest instead of something to be checked.
    """
    p_joint = np.asarray(p_joint, dtype=float)
    return joint_entropy(p_joint, base) - entropy(p_joint.sum(axis=1), base)


def cross_entropy(p: np.ndarray, q: np.ndarray, base: float = math.e) -> float:
    """H(p, q) = -sum p log q.  README §6

    The cost of encoding data drawn from p using a code built for q. Infinite if q assigns
    zero probability to something p says can happen — which is exactly why every real
    implementation clips q away from 0.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(-np.sum(p[mask] * np.log(q[mask])) / math.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = math.e) -> float:
    """D_KL(p || q) = sum p log(p/q).  README §7

    NOT symmetric, NOT a metric. Infinite when q = 0 somewhere p > 0.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])) / math.log(base))


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = math.e) -> float:
    """Jensen-Shannon divergence.  README §10

    Symmetric, always finite (m = (p+q)/2 has the union of both supports, so nothing is
    ever divided by zero), and bounded above by log 2. Its square root is a true metric.

    The boundedness is exactly the GAN problem: when p and q have disjoint support, JS is
    pinned at its maximum log 2 and its gradient is ZERO, so the generator learns nothing.
    That is what WGAN was built to fix.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def mutual_information(p_joint: np.ndarray, base: float = math.e) -> float:
    """I(X; Y) = D_KL( p(x,y) || p(x)p(y) ).  README §9

    The definition worth remembering: mutual information is the KL divergence between the
    real joint and the joint you would have if X and Y were independent. It measures how
    far from independent they are.

    Unlike correlation, I = 0 genuinely does imply independence.
    """
    p_joint = np.asarray(p_joint, dtype=float)
    p_x = p_joint.sum(axis=1, keepdims=True)
    p_y = p_joint.sum(axis=0, keepdims=True)
    return kl_divergence(p_joint.ravel(), (p_x * p_y).ravel(), base)


def information_gain(y: np.ndarray, x: np.ndarray, base: float = 2.0) -> float:
    """IG(Y, X) = H(Y) - H(Y | X) = I(Y; X).  README §9

    What a decision tree maximizes when choosing a split. "Information gain" and "mutual
    information" are the same quantity under two names — the tree literature adopted one,
    information theory the other.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    n = y.size

    def dist(values):
        counts = np.array(list(Counter(values.tolist()).values()), dtype=float)
        return counts / counts.sum()

    h_y = entropy(dist(y), base)
    h_y_given_x = 0.0
    for value in np.unique(x):
        subset = y[x == value]
        h_y_given_x += (subset.size / n) * entropy(dist(subset), base)
    return h_y - h_y_given_x


def perplexity(log_probs: np.ndarray) -> float:
    """PPL = exp(mean negative log-likelihood).  README §13

    The effective number of equally-likely options the model is choosing between at each
    step. Comparable across models ONLY under identical tokenization.
    """
    return float(np.exp(-np.mean(np.asarray(log_probs, dtype=float))))


# =============================================================================
# HUFFMAN CODING — entropy is a code length  (README §4)
# =============================================================================


def huffman_code(probabilities: dict) -> dict:
    """Build an optimal prefix code by Huffman's algorithm.

    Repeatedly merge the two least likely symbols into one node. The tree that results
    assigns short codes to frequent symbols and long codes to rare ones, and it is
    provably optimal among symbol-by-symbol prefix codes.

    Shannon's source coding theorem (README §4) says the average length L of ANY uniquely
    decodable code satisfies L >= H(X), and Huffman achieves L < H(X) + 1. Experiment 1
    verifies both bounds, including a case where the gap is nearly a full bit.
    """
    if len(probabilities) == 1:
        return {next(iter(probabilities)): "0"}

    # Heap entries carry a counter to break ties deterministically, since dicts are not
    # orderable and equal probabilities are common.
    counter = 0
    heap = []
    for symbol, p in probabilities.items():
        heapq.heappush(heap, (p, counter, {symbol: ""}))
        counter += 1

    while len(heap) > 1:
        p1, _, codes1 = heapq.heappop(heap)
        p2, _, codes2 = heapq.heappop(heap)
        merged = {s: "0" + c for s, c in codes1.items()}
        merged.update({s: "1" + c for s, c in codes2.items()})
        heapq.heappush(heap, (p1 + p2, counter, merged))
        counter += 1

    return heap[0][2]


def average_code_length(codes: dict, probabilities: dict) -> float:
    """Expected bits per symbol under a given code."""
    return sum(probabilities[s] * len(c) for s, c in codes.items())


# =============================================================================
# MAXIMUM ENTROPY  (README §11)
# =============================================================================


def max_entropy_discrete(n_outcomes: int, constraints=None,
                         n_iter: int = 5000, lr: float = 0.05) -> np.ndarray:
    """Maximum-entropy distribution over n outcomes, subject to E[f_k(x)] = c_k.

    With no constraints the answer is the uniform distribution — the honest choice when
    you know only the support. With a mean constraint on the integers you get a geometric;
    with mean and variance on the reals you would get a Gaussian (README §11).

    Solved by gradient ascent on the entropy in softmax-parameterized coordinates, with
    the constraints enforced by a quadratic penalty. Real solvers use the dual (which is
    exactly the exponential family, 00.03 §10), but this is transparent and adequate here.
    """
    theta = np.zeros(n_outcomes)

    for _ in range(n_iter):
        p = np.exp(theta - theta.max())
        p /= p.sum()

        # d/dtheta of H(p) with p = softmax(theta).
        grad = -p * (np.log(p + EPS) + entropy(p))

        if constraints:
            for f, target in constraints:
                f_values = np.asarray(f, dtype=float)
                violation = float(p @ f_values - target)
                grad -= 20.0 * violation * p * (f_values - p @ f_values)

        theta += lr * grad

    p = np.exp(theta - theta.max())
    return p / p.sum()


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<52s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 84)
    print("VERIFICATION")
    print("=" * 84)

    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.array([0.4, 0.3, 0.2, 0.1])

    print("\nCore quantities (README §3-§10)")
    try:
        from scipy.stats import entropy as scipy_entropy
        ok &= _report("entropy vs scipy.stats.entropy",
                      abs(entropy(p) - scipy_entropy(p)), 1e-12)
        ok &= _report("kl_divergence vs scipy.stats.entropy(p, q)",
                      abs(kl_divergence(p, q) - scipy_entropy(p, q)), 1e-12)
        ok &= _report("entropy in bits vs scipy base=2",
                      abs(entropy(p, base=2) - scipy_entropy(p, base=2)), 1e-12)
    except ImportError:
        print("  [SKIP]  scipy not installed")

    # Hand-computable case: p = (1/2, 1/4, 1/8, 1/8) has H = 1.75 bits exactly.
    ok &= _report("H(1/2,1/4,1/8,1/8) = 1.75 bits", abs(entropy(p, base=2) - 1.75), 1e-12)
    ok &= _report("uniform over 8 has H = 3 bits",
                  abs(entropy(np.full(8, 1 / 8), base=2) - 3.0), 1e-12)
    ok &= _report("deterministic has H = 0",
                  abs(entropy(np.array([1.0, 0.0, 0.0]))), 1e-12)
    ok &= _report("fair coin has H = 1 bit",
                  abs(entropy(np.array([0.5, 0.5]), base=2) - 1.0), 1e-12)

    print("\nIdentities (README §7.1)")
    ok &= _report("H(p,q) = H(p) + KL(p||q)",
                  abs(cross_entropy(p, q) - (entropy(p) + kl_divergence(p, q))), 1e-12)
    ok &= _report("KL(p||p) = 0", abs(kl_divergence(p, p)), 1e-12)
    ok &= _report("H(p,p) = H(p)", abs(cross_entropy(p, p) - entropy(p)), 1e-12)

    # Gibbs: KL >= 0 for arbitrary random pairs.
    worst = 0.0
    for _ in range(2000):
        a = rng.random(6); a /= a.sum()
        b = rng.random(6); b /= b.sum()
        worst = min(worst, kl_divergence(a, b))
    ok &= _report("KL >= 0 over 2000 random pairs (Gibbs)", abs(min(worst, 0.0)), 1e-12)

    # Asymmetry must be real, not an artifact.
    asymmetry = abs(kl_divergence(p, q) - kl_divergence(q, p))
    print(f"  [INFO]  {'KL(p||q) vs KL(q||p) — asymmetry is genuine':<52s}  "
          f"{kl_divergence(p, q):.4f} vs {kl_divergence(q, p):.4f}")
    ok &= asymmetry > 1e-3

    print("\nJensen-Shannon (README §10)")
    ok &= _report("JS is symmetric",
                  abs(js_divergence(p, q) - js_divergence(q, p)), 1e-12)
    ok &= _report("JS(p,p) = 0", abs(js_divergence(p, p)), 1e-12)
    disjoint_a = np.array([0.5, 0.5, 0.0, 0.0])
    disjoint_b = np.array([0.0, 0.0, 0.5, 0.5])
    ok &= _report("JS of disjoint supports = log 2 (its maximum)",
                  abs(js_divergence(disjoint_a, disjoint_b) - math.log(2)), 1e-12)
    print(f"  [INFO]  {'...while KL of the same pair is':<52s}  "
          f"{kl_divergence(disjoint_a, disjoint_b)}")

    print("\nMutual information (README §9)")
    # Independent joint: I must be exactly 0.
    px = np.array([0.3, 0.7])
    py = np.array([0.2, 0.5, 0.3])
    ok &= _report("I = 0 for an independent joint",
                  abs(mutual_information(np.outer(px, py))), 1e-12)

    # Deterministic Y = X: I must equal H(X).
    deterministic = np.array([[0.4, 0.0], [0.0, 0.6]])
    ok &= _report("I(X;Y) = H(X) when Y determines X",
                  abs(mutual_information(deterministic) - entropy(np.array([0.4, 0.6]))), 1e-12)

    # I(X;Y) = H(X) + H(Y) - H(X,Y).
    joint = rng.random((4, 5))
    joint /= joint.sum()
    identity_err = abs(mutual_information(joint)
                       - (entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0))
                          - joint_entropy(joint)))
    ok &= _report("I = H(X) + H(Y) - H(X,Y)", identity_err, 1e-12)

    # Chain rule.
    ok &= _report("H(X,Y) = H(X) + H(Y|X)",
                  abs(joint_entropy(joint)
                      - (entropy(joint.sum(axis=1)) + conditional_entropy(joint))), 1e-12)

    # Conditioning never increases entropy, on average.
    ok &= _report("H(Y|X) <= H(Y)",
                  max(0.0, conditional_entropy(joint) - entropy(joint.sum(axis=0))), 1e-12)

    print("\nInformation gain vs sklearn (README §9)")
    try:
        from sklearn.metrics import mutual_info_score
        x_disc = rng.integers(0, 4, 3000)
        y_disc = np.where(rng.random(3000) < 0.75, x_disc % 2, 1 - x_disc % 2)
        ok &= _report("information_gain vs sklearn mutual_info_score",
                      abs(information_gain(y_disc, x_disc, base=math.e)
                          - mutual_info_score(x_disc, y_disc)), 1e-10)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nHuffman coding (README §4)")
    probs = {"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.125}
    codes = huffman_code(probs)
    length = average_code_length(codes, probs)
    ok &= _report("dyadic case: L exactly equals H", abs(length - 1.75), 1e-12)

    # Prefix-freeness: no codeword may be a prefix of another, or decoding is ambiguous.
    words = sorted(codes.values())
    prefix_free = all(not words[j].startswith(words[i])
                      for i in range(len(words)) for j in range(i + 1, len(words)))
    print(f"  [{'PASS' if prefix_free else 'FAIL'}]  "
          f"{'Huffman code is prefix-free':<52s}  {codes}")
    ok &= prefix_free

    print("\nMaximum entropy (README §11)")
    p_max = max_entropy_discrete(5)
    ok &= _report("unconstrained max-entropy = uniform",
                  float(np.abs(p_max - 0.2).max()), 1e-3)

    print("\nPerplexity (README §13)")
    uniform_lp = np.full(100, math.log(1 / 50))
    ok &= _report("PPL of a uniform-over-50 model = 50",
                  abs(perplexity(uniform_lp) - 50.0), 1e-9)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_source_coding() -> None:
    """README §4: entropy is the exact lower bound on average code length."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 1 — entropy IS a code length  (README §4)")
    print("=" * 84)
    print("""
Shannon's source coding theorem says the average length L of any uniquely decodable code
satisfies H <= L < H + 1, and Huffman attains the optimum. Building actual codes and
measuring:
""")
    cases = [
        ("dyadic (1/2, 1/4, 1/8, 1/8)", {"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.125}),
        ("uniform over 4", {c: 0.25 for c in "ABCD"}),
        ("uniform over 5", {c: 0.2 for c in "ABCDE"}),
        ("very skewed", {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
        ("English letter freqs", {
            "e": .127, "t": .091, "a": .082, "o": .075, "i": .070, "n": .067,
            "s": .063, "h": .061, "r": .060, "d": .043, "l": .040, "c": .028,
            "u": .028, "m": .024, "w": .024, "f": .022, "g": .020, "y": .020,
            "p": .019, "b": .015, "v": .010, "k": .008, "j": .002, "x": .002,
            "q": .001, "z": .001}),
    ]

    print(f"  {'distribution':<26s}  {'H (bits)':>9s}  {'Huffman L':>10s}  "
          f"{'L - H':>7s}  {'fixed-length':>13s}  {'saving':>7s}")
    print("  " + "-" * 82)

    for name, probs in cases:
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}      # renormalize
        h = entropy(np.array(list(probs.values())), base=2)
        codes = huffman_code(probs)
        length = average_code_length(codes, probs)
        fixed = math.ceil(math.log2(len(probs)))
        assert length >= h - 1e-9, "source coding theorem violated"
        assert length < h + 1, "Huffman exceeded H+1"
        print(f"  {name:<26s}  {h:9.4f}  {length:10.4f}  {length - h:7.4f}  "
              f"{fixed:13d}  {(1 - length / fixed):6.1%}")

    print("""
  Every row satisfies H <= L < H + 1 — the theorem is not approximate.

  Row 1 is dyadic (all probabilities are powers of 1/2), so every code length -log2(p) is
  an integer and Huffman hits the bound exactly. Row 3 cannot: with 5 equally likely
  symbols the ideal length is log2(5) = 2.32 bits, and codes must use whole bits, so the
  gap is unavoidable rounding rather than a weakness of the algorithm.

  The last column is the practical point. Encoding English letters at their true
  frequencies costs 4.18 bits instead of the 5 bits a fixed-length code needs — a 16%
  saving, purchased entirely by matching code length to -log p. That is what "entropy is
  a code length" means operationally, and it is the same accounting a model performs when
  it assigns probabilities to tokens.""")


def experiment_cross_entropy_decomposition() -> None:
    """README §7.1: H(p,q) = H(p) + KL(p||q), so your loss has an irreducible floor."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 2 — your loss has a floor, and it is H(p)  (README §7.1)")
    print("=" * 84)
    print("""
Cross-entropy splits exactly into H(p) + KL(p||q): the data's own noise plus your model's
error. Only the second term is yours to reduce. Simulating a 3-class problem where the
true conditional is genuinely uncertain, and watching a model improve toward the floor:
""")
    p_true = np.array([0.6, 0.3, 0.1])
    h_p = cross_entropy(p_true, p_true)

    models = [
        ("uniform guess", np.array([1 / 3, 1 / 3, 1 / 3])),
        ("poor", np.array([0.4, 0.35, 0.25])),
        ("decent", np.array([0.55, 0.30, 0.15])),
        ("good", np.array([0.59, 0.30, 0.11])),
        ("PERFECT (q = p)", p_true.copy()),
        ("overconfident, right argmax", np.array([0.98, 0.01, 0.01])),
    ]

    print(f"  True distribution p = {p_true},  H(p) = {h_p:.4f} nats\n")
    print(f"  {'model q':<28s}  {'cross-entropy':>14s}  {'= H(p)':>8s}  "
          f"{'+ KL(p||q)':>11s}  {'check':>8s}")
    print("  " + "-" * 76)

    for name, q in models:
        ce = cross_entropy(p_true, q)
        kl = kl_divergence(p_true, q)
        print(f"  {name:<28s}  {ce:14.4f}  {h_p:8.4f}  {kl:11.4f}  "
              f"{abs(ce - h_p - kl):8.1e}")

    print(f"""
  The identity holds to machine precision on every row.

  Two things follow, both practically important.

  1. THE PERFECT MODEL SCORES {h_p:.4f}, NOT ZERO. When q = p exactly, KL = 0 and the loss
     equals H(p) — the data's irreducible randomness. If your training loss on genuinely
     noisy data approaches 0, you are not modelling, you are memorizing.

  2. OVERCONFIDENCE IS EXPENSIVE. The last row gets the argmax right and would score 100%
     accuracy, yet its loss is the WORST in the table — far worse than the honest uniform
     guess. Cross-entropy is a proper scoring rule: it is minimized only by reporting your
     true beliefs, which is why it produces calibrated probabilities and accuracy does
     not.""")


def experiment_forward_vs_reverse_kl() -> None:
    """README §8: mode-covering vs mode-seeking, fitted and shown."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 3 — forward vs reverse KL  (README §8)")
    print("=" * 84)
    print("""
The asymmetry of KL is not a technicality. Fitting a SINGLE Gaussian q to a BIMODAL target
p, once by minimizing forward KL(p||q) and once by reverse KL(q||p). The model is too
simple to fit p either way — what matters is HOW each objective chooses to fail.
""")
    grid = np.linspace(-10, 12, 2000)
    dx = grid[1] - grid[0]

    def gaussian_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    # Target: two well-separated modes.
    p = 0.5 * gaussian_pdf(grid, 0.0, 1.0) + 0.5 * gaussian_pdf(grid, 6.0, 1.0)
    p /= p.sum() * dx

    def fit(objective):
        best, best_params = float("inf"), None
        for mu in np.linspace(-3, 9, 200):
            for sigma in np.linspace(0.3, 6.0, 200):
                q = gaussian_pdf(grid, mu, sigma)
                q /= q.sum() * dx
                value = objective(p * dx + 1e-300, q * dx + 1e-300)
                if value < best:
                    best, best_params = value, (mu, sigma)
        return best_params, best

    (mu_f, sd_f), val_f = fit(lambda pp, qq: kl_divergence(pp, qq))   # forward: E over p
    (mu_r, sd_r), val_r = fit(lambda pp, qq: kl_divergence(qq, pp))   # reverse: E over q

    print(f"  Target p: equal mixture of N(0, 1) and N(6, 1); true modes at 0 and 6\n")
    print(f"  {'objective':<28s}  {'fitted mu':>10s}  {'fitted sigma':>13s}  {'value':>8s}")
    print("  " + "-" * 66)
    print(f"  {'forward  KL(p||q)':<28s}  {mu_f:10.3f}  {sd_f:13.3f}  {val_f:8.4f}")
    print(f"  {'reverse  KL(q||p)':<28s}  {mu_r:10.3f}  {sd_r:13.3f}  {val_r:8.4f}")

    # Density each fit places in the empty valley between the modes.
    valley = (grid > 2.0) & (grid < 4.0)
    for label, mu, sd in [("forward", mu_f, sd_f), ("reverse", mu_r, sd_r)]:
        q = gaussian_pdf(grid, mu, sd)
        q /= q.sum() * dx
        print(f"    {label} KL puts {q[valley].sum() * dx:.1%} of its mass in the empty "
              f"valley (p has {p[valley].sum() * dx:.1%})")

    print(f"""
  FORWARD KL(p||q) landed at mu = {mu_f:.2f} — squarely BETWEEN the two modes — with a wide
  sigma = {sd_f:.2f}. The expectation is taken over p, so anywhere p has mass and q does not,
  log(p/q) blows up. It is forced to cover everything, including the empty middle where the
  target has almost no mass at all. This is MODE-COVERING, and it is why an underfit
  maximum-likelihood model produces blurry, over-dispersed samples.

  REVERSE KL(q||p) landed at mu = {mu_r:.2f} — right on ONE mode — with a narrow
  sigma = {sd_r:.2f}. The expectation is over q, so regions where q is near zero contribute
  nothing regardless of what p does there. It can simply ignore the other mode. This is
  MODE-SEEKING, and it is why variational posteriors collapse.

  Same target, same model family, opposite failures — decided entirely by which way round
  the KL is written. Maximum likelihood and cross-entropy minimize the forward direction;
  the VAE's ELBO minimizes the reverse. That is a large part of why VAE samples blur while
  variational approximations under-report uncertainty.""")


def experiment_mutual_information() -> None:
    """README §9: MI sees dependence that correlation cannot."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 4 — mutual information vs correlation  (README §9)")
    print("=" * 84)
    print("""
Correlation measures LINEAR dependence only. Mutual information measures all of it. Five
relationships, each with 20,000 samples:
""")
    rng = np.random.default_rng(7)
    n = 20_000

    def mi_from_samples(a, b, bins=24):
        """Estimate I(A;B) by histogramming into a joint table.

        Binned MI is biased upward at finite n (every pair looks slightly dependent by
        chance), so the independent baseline below is the number to calibrate against —
        not zero.
        """
        joint, _, _ = np.histogram2d(a, b, bins=bins)
        joint = joint / joint.sum()
        return mutual_information(joint, base=2)

    x = rng.standard_normal(n)
    cases = [
        ("independent", x, rng.standard_normal(n)),
        ("linear: y = 2x + noise", x, 2 * x + 0.3 * rng.standard_normal(n)),
        ("quadratic: y = x^2", x, x ** 2 + 0.1 * rng.standard_normal(n)),
        ("sine: y = sin(3x)", x, np.sin(3 * x) + 0.1 * rng.standard_normal(n)),
        ("circle", np.cos(t := rng.uniform(0, 2 * np.pi, n)), np.sin(t)),
    ]

    print(f"  {'relationship':<26s}  {'|correlation|':>14s}  {'MI (bits)':>11s}  {'verdict':>28s}")
    print("  " + "-" * 84)

    for name, a, b in cases:
        corr = abs(float(np.corrcoef(a, b)[0, 1]))
        mi = mi_from_samples(a, b)
        if corr < 0.1 and mi > 0.3:
            verdict = "correlation MISSES it"
        elif corr < 0.1 and mi < 0.3:
            verdict = "genuinely independent"
        else:
            verdict = "both detect it"
        print(f"  {name:<26s}  {corr:14.4f}  {mi:11.4f}  {verdict:>28s}")

    print("""
  Rows 3-5 are the point. y = x^2, y = sin(3x), and the circle all have near-zero
  correlation — yet in every case y is an almost deterministic function of x. Correlation
  is structurally blind to any relationship that is not a straight line; mutual information
  is not.

  This is why sklearn ships mutual_info_classif alongside correlation-based filters. If you
  select features by |correlation| you will silently discard every nonlinearly informative
  one — exactly the features a tree or a neural network would have exploited.

  Caveat worth knowing: binned MI estimates are biased UPWARD at finite n, because random
  fluctuations make any two variables look slightly dependent. Calibrate against the
  independent row rather than against zero.""")


def experiment_decision_tree_split() -> None:
    """README §9: information gain is mutual information, and it is what trees maximize."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 5 — information gain is what a tree maximizes  (README §9)")
    print("=" * 84)
    print("""
A decision tree picks the feature with the highest information gain. That quantity is
mutual information under a different name. Building a dataset where we KNOW which feature
matters, and checking what the criterion says:
""")
    rng = np.random.default_rng(11)
    n = 4000

    y = (rng.random(n) < 0.5).astype(int)
    features = {
        "perfect predictor": y.copy(),
        "strong (85% agree)": np.where(rng.random(n) < 0.85, y, 1 - y),
        "weak (60% agree)": np.where(rng.random(n) < 0.60, y, 1 - y),
        "pure noise": (rng.random(n) < 0.5).astype(int),
        "constant": np.zeros(n, dtype=int),
        "high-cardinality ID": np.arange(n),         # a unique value per row
    }

    h_y = entropy(np.array([0.5, 0.5]), base=2)
    print(f"  H(Y) = {h_y:.4f} bits — the total uncertainty available to remove\n")
    print(f"  {'feature':<24s}  {'info gain':>11s}  {'H(Y|X)':>9s}  {'% removed':>10s}")
    print("  " + "-" * 60)

    for name, x in features.items():
        ig = information_gain(y, x, base=2)
        print(f"  {name:<24s}  {ig:11.4f}  {h_y - ig:9.4f}  {ig / h_y:9.1%}")

    print("""
  The ordering is exactly right: the perfect predictor removes 100% of the uncertainty, the
  strong feature most of it, noise and the constant essentially none. A tree choosing the
  largest information gain picks correctly.

  Except for the last row, and it is the important one. The unique-ID column has a PERFECT
  information gain of 1.0 — because knowing a row's ID tells you its label exactly on the
  training set. It is also completely worthless: it generalizes to nothing.

  This is the well-known bias of information gain toward high-cardinality features, and it
  is why C4.5 introduced the GAIN RATIO (dividing by the split's own entropy) and why CART
  uses Gini with binary splits. It is also why leaking an ID, a timestamp, or a row index
  into your features produces a model with a suspiciously perfect validation score — see
  02.06 on data leakage.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_source_coding()
    experiment_cross_entropy_decomposition()
    experiment_forward_vs_reverse_kl()
    experiment_mutual_information()
    experiment_decision_tree_split()

    print("\n" + "=" * 84)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 84)
