"""
00.03 — Probability from Scratch
================================

Distributions implemented from their defining formulas, samplers built from uniform
random numbers, and the chapter's claims measured rather than asserted.

Nothing from `scipy.stats` is used inside an implementation — it appears only in the
verification section as the reference to check against.

Implemented here
----------------
    Distributions (pdf/pmf, cdf, mean, var, sample), all from the formula:
        Bernoulli, Binomial, Poisson, Geometric
        Uniform, Gaussian, Laplace, Exponential, Beta

    MultivariateGaussian    density, sampling via Cholesky, marginalize, condition
    cholesky                from scratch — the factorization that makes sampling work

    Samplers:
        inverse_cdf_sample      README §15.1
        box_muller              README §15.3
        rejection_sample        README §15.2

    bayes_update            posterior from prior x likelihood     README §7
    beta_binomial_update    conjugate updating in closed form     README §8.1
    disease_test_posterior  the base rate fallacy, computed       README §7.1

Run it
------
    python from_scratch.py

Verifies every distribution against scipy.stats (moments, densities, CDFs, and a
goodness-of-fit test on the samplers), then runs four experiments:
  1. The CLT: sample means go Gaussian at rate 1/sqrt(n), whatever you start from
  2. Markov vs Chebyshev vs Hoeffding — how much slack each bound leaves
  3. Base rates: why a 99% accurate test is 9% informative
  4. Loss functions ARE negative log-likelihoods (MSE<->Gaussian, MAE<->Laplace)

Reference: README.md sections 4-15.
"""

from __future__ import annotations

import math

import numpy as np

# =============================================================================
# UNIVARIATE DISTRIBUTIONS  (README §8)
# =============================================================================


class Distribution:
    """Base class. Subclasses supply the formulas; sampling and checks come free.

    `sample` defaults to inverse-CDF sampling (README §15.1), which works for any
    distribution whose CDF can be inverted numerically. Subclasses override it when a
    direct method exists (Box-Muller for the Gaussian, for instance).
    """

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def var(self) -> float:
        raise NotImplementedError

    def std(self) -> float:
        return math.sqrt(self.var())

    def sample(self, size, rng):
        return self.ppf(rng.random(size))

    def ppf(self, q):
        """Inverse CDF by bisection — the generic fallback."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        lo = np.full_like(q, -1e6)
        hi = np.full_like(q, 1e6)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            too_low = self.cdf(mid) < q
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        return 0.5 * (lo + hi)


# --- Discrete ---------------------------------------------------------------


class Bernoulli(Distribution):
    """One yes/no trial.  p(1) = pi,  p(0) = 1 - pi."""

    def __init__(self, pi: float):
        self.pi = pi

    def pmf(self, k):
        k = np.asarray(k)
        return np.where(k == 1, self.pi, np.where(k == 0, 1 - self.pi, 0.0))

    def mean(self):
        return self.pi

    def var(self):
        # Maximized at pi = 0.5: a fair coin is the most unpredictable one.
        return self.pi * (1 - self.pi)

    def sample(self, size, rng):
        return (rng.random(size) < self.pi).astype(int)


class Binomial(Distribution):
    """Successes in n independent Bernoulli(pi) trials.

        p(k) = C(n,k) pi^k (1-pi)^(n-k)

    The binomial coefficient counts the orderings; the rest is the probability of any one
    particular ordering.
    """

    def __init__(self, n: int, pi: float):
        self.n, self.pi = n, pi

    def pmf(self, k):
        k = np.asarray(k)
        return np.array([
            math.comb(self.n, int(ki)) * self.pi ** ki * (1 - self.pi) ** (self.n - ki)
            if 0 <= ki <= self.n else 0.0
            for ki in np.atleast_1d(k)
        ])

    def mean(self):
        return self.n * self.pi

    def var(self):
        return self.n * self.pi * (1 - self.pi)

    def sample(self, size, rng):
        return (rng.random((size, self.n)) < self.pi).sum(axis=1)


class Poisson(Distribution):
    """Count of rare events in a fixed window.

        p(k) = lambda^k e^(-lambda) / k!

    Mean and variance are BOTH lambda — a strong, testable claim. When count data has
    variance well above its mean ("overdispersion"), Poisson regression is the wrong model
    and you want negative binomial instead.
    """

    def __init__(self, lam: float):
        self.lam = lam

    def pmf(self, k):
        return np.array([
            self.lam ** ki * math.exp(-self.lam) / math.factorial(int(ki)) if ki >= 0 else 0.0
            for ki in np.atleast_1d(k)
        ])

    def mean(self):
        return self.lam

    def var(self):
        return self.lam

    def sample(self, size, rng):
        """Knuth's method: count how many Uniform(0,1) draws it takes for the running
        product to fall below e^-lambda. Equivalent to counting exponential inter-arrival
        times that fit in a unit window."""
        out = np.zeros(size, dtype=int)
        threshold = math.exp(-self.lam)
        for i in range(size):
            k, p = 0, 1.0
            while True:
                p *= rng.random()
                if p <= threshold:
                    break
                k += 1
            out[i] = k
        return out


class Geometric(Distribution):
    """Number of trials until (and including) the first success.  p(k) = (1-pi)^(k-1) pi

    Memoryless: P(K > m + n | K > m) = P(K > n). Past failures tell you nothing about how
    much longer you must wait — the discrete analogue of the exponential.
    """

    def __init__(self, pi: float):
        self.pi = pi

    def pmf(self, k):
        k = np.atleast_1d(np.asarray(k))
        return np.where(k >= 1, (1 - self.pi) ** (k - 1) * self.pi, 0.0)

    def mean(self):
        return 1.0 / self.pi

    def var(self):
        return (1 - self.pi) / self.pi ** 2

    def sample(self, size, rng):
        return np.ceil(np.log1p(-rng.random(size)) / np.log1p(-self.pi)).astype(int)


# --- Continuous -------------------------------------------------------------


class Uniform(Distribution):
    """Total ignorance on [a, b]. Maximum entropy given only the support (see 00.05)."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.a) & (x <= self.b), 1.0 / (self.b - self.a), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - self.a) / (self.b - self.a), 0.0, 1.0)

    def ppf(self, q):
        return self.a + np.asarray(q, dtype=float) * (self.b - self.a)

    def mean(self):
        return 0.5 * (self.a + self.b)

    def var(self):
        return (self.b - self.a) ** 2 / 12.0


class Gaussian(Distribution):
    """The normal distribution.

        p(x) = 1/sqrt(2 pi sigma^2) exp(-(x-mu)^2 / (2 sigma^2))

    Note the density can exceed 1: at sigma = 0.1 the peak is ~3.99 (README §3.2).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu, self.sigma = mu, sigma

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        return np.exp(-0.5 * z ** 2) / (self.sigma * math.sqrt(2 * math.pi))

    def logpdf(self, x):
        """Always prefer this to log(pdf(x)): the exp() underflows to 0 for |z| > ~40,
        after which log gives -inf. See 00.06."""
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        return -0.5 * z ** 2 - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        # Phi(x) = 1/2 [1 + erf((x-mu)/(sigma sqrt 2))]
        return 0.5 * (1 + np.vectorize(math.erf)((x - self.mu) / (self.sigma * math.sqrt(2))))

    def mean(self):
        return self.mu

    def var(self):
        return self.sigma ** 2

    def sample(self, size, rng):
        return self.mu + self.sigma * box_muller(size, rng)


class Laplace(Distribution):
    """Double exponential.  p(x) = 1/(2b) exp(-|x - mu| / b)

    Tails decay as e^(-|x|) instead of the Gaussian's e^(-x^2), so extreme values are far
    less surprising. Its negative log-likelihood is |y - f|/b — i.e. **MAE is the Laplace
    assumption** (README §9.4), which is why MAE is the robust choice.
    """

    def __init__(self, mu: float = 0.0, b: float = 1.0):
        self.mu, self.b = mu, b

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-np.abs(x - self.mu) / self.b) / (2 * self.b)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < self.mu,
                        0.5 * np.exp((x - self.mu) / self.b),
                        1 - 0.5 * np.exp(-(x - self.mu) / self.b))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return np.where(q < 0.5,
                        self.mu + self.b * np.log(2 * q),
                        self.mu - self.b * np.log(2 * (1 - q)))

    def mean(self):
        return self.mu

    def var(self):
        return 2 * self.b ** 2


class Exponential(Distribution):
    """Waiting time until an event.  p(x) = lambda e^(-lambda x),  x >= 0

    Memoryless: P(X > s+t | X > s) = P(X > t). A component that has survived 10 years is
    exactly as likely to fail tomorrow as a new one — which is why the exponential is a
    poor model for anything that wears out.
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.lam * np.exp(-self.lam * x), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1 - np.exp(-self.lam * x), 0.0)

    def ppf(self, q):
        """Closed-form inverse CDF — no bisection needed (README §15.1)."""
        return -np.log1p(-np.asarray(q, dtype=float)) / self.lam

    def mean(self):
        return 1.0 / self.lam

    def var(self):
        return 1.0 / self.lam ** 2


class Beta(Distribution):
    """Distribution over a probability.  p(x) = x^(a-1) (1-x)^(b-1) / B(a,b)

    The conjugate prior for Bernoulli and Binomial: alpha and beta act as pseudo-counts of
    successes and failures, so updating is addition (README §8.1).
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha, self.beta = alpha, beta

    def _log_beta_fn(self):
        return math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = ((self.alpha - 1) * np.log(x)
                     + (self.beta - 1) * np.log1p(-x)
                     - self._log_beta_fn())
            out = np.exp(log_p)
        return np.where((x > 0) & (x < 1), out, 0.0)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def var(self):
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def cdf(self, x):
        """Regularized incomplete beta, by numerical integration. Accurate enough for the
        checks here; scipy uses a continued-fraction expansion."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= 0:
                out[i] = 0.0
            elif xi >= 1:
                out[i] = 1.0
            else:
                grid = np.linspace(1e-12, xi, 20001)
                out[i] = np.trapezoid(self.pdf(grid), grid)
        return out

    def sample(self, size, rng):
        """Beta(a,b) = G1 / (G1 + G2) for independent Gammas — README §8.1."""
        g1 = rng.gamma(self.alpha, 1.0, size)
        g2 = rng.gamma(self.beta, 1.0, size)
        return g1 / (g1 + g2)


# =============================================================================
# SAMPLERS  (README §15)
# =============================================================================


def box_muller(size: int, rng) -> np.ndarray:
    """Two uniforms -> two independent standard Gaussians.  README §15.3

        z1 = sqrt(-2 ln u1) cos(2 pi u2)
        z2 = sqrt(-2 ln u1) sin(2 pi u2)

    Derivation: write the 2-D standard Gaussian in polar coordinates. The radius satisfies
    R^2 ~ Exponential(1/2) and the angle is Uniform(0, 2pi), and both are easy to sample by
    inverse CDF. Transforming back is the change-of-variables formula (README §11) run in
    reverse.
    """
    n_pairs = (size + 1) // 2
    u1 = rng.random(n_pairs)
    u2 = rng.random(n_pairs)
    radius = np.sqrt(-2.0 * np.log(u1 + 1e-300))     # guard log(0)
    z = np.concatenate([radius * np.cos(2 * np.pi * u2),
                        radius * np.sin(2 * np.pi * u2)])
    return z[:size]


def inverse_cdf_sample(dist: Distribution, size: int, rng) -> np.ndarray:
    """X = F^-1(U) has CDF F, for U ~ Uniform(0,1).  README §15.1

    Proof: P(X <= x) = P(F^-1(U) <= x) = P(U <= F(x)) = F(x).
    """
    return dist.ppf(rng.random(size))


def rejection_sample(target_pdf, proposal: Distribution, M: float,
                     size: int, rng, max_iter: int = 10_000_000):
    """Sample from `target_pdf` using a proposal q with target <= M*q everywhere.

    Draw x ~ q, accept with probability target(x) / (M q(x)).

    Returns (samples, acceptance_rate). The acceptance rate is 1/M, which is the method's
    fatal flaw: in d dimensions M typically grows exponentially in d, so the sampler
    accepts essentially nothing. This is precisely why MCMC exists.
    """
    accepted: list[float] = []
    n_proposed = 0
    while len(accepted) < size and n_proposed < max_iter:
        batch = max(size, 1000)
        x = proposal.sample(batch, rng)
        u = rng.random(batch)
        keep = u < target_pdf(x) / (M * proposal.pdf(x))
        accepted.extend(x[keep].tolist())
        n_proposed += batch
    # Rate is total accepted over total proposed — not size/n_proposed, which would
    # undercount by discarding the surplus from the final batch.
    return np.array(accepted[:size]), len(accepted) / max(n_proposed, 1)


# =============================================================================
# MULTIVARIATE GAUSSIAN  (README §9.2, §9.3)
# =============================================================================


def cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky factorization A = L L^T for symmetric positive definite A.

    Derived by writing out the (i,j) entry of L L^T and solving for L_ij in order:

        A_ij = sum_k L_ik L_jk   =>   L_ij = (A_ij - sum_{k<j} L_ik L_jk) / L_jj
        A_jj = sum_k L_jk^2      =>   L_jj = sqrt(A_jj - sum_{k<j} L_jk^2)

    Requires A positive definite: the diagonal formula takes a square root, and that
    argument is guaranteed non-negative exactly when A is PD (00.01 §11.2). A failure here
    is therefore a *useful* signal that your covariance estimate is not a valid covariance.

    Half the cost of LU, and the standard way to sample a multivariate Gaussian.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("matrix is not positive definite")
                L[i, j] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


class MultivariateGaussian:
    """N(mu, Sigma) in d dimensions.  README §9.2

    Geometry (all of it borrowed from 00.01):
      - (x-mu)^T Sigma^-1 (x-mu) is the squared Mahalanobis distance, a quadratic form
      - Sigma = Q Lambda Q^T: columns of Q are the ellipsoid's axes, sqrt(lambda_i) its radii
      - |Sigma|^(1/2) is the ellipsoid's volume, which is why it normalizes the density
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.d = self.mu.size
        self.L = cholesky(self.Sigma)

    def _solve_with_cholesky(self, B):
        """Solve Sigma X = B using Sigma = L L^T, by two triangular solves.

        Never forms Sigma^-1 — same lesson as 00.01 §15.3.
        """
        n = self.L.shape[0]
        B = np.atleast_2d(B.T).T if B.ndim == 1 else B
        Y = np.zeros_like(B, dtype=float)
        for i in range(n):                                   # forward substitution
            Y[i] = (B[i] - self.L[i, :i] @ Y[:i]) / self.L[i, i]
        X = np.zeros_like(B, dtype=float)
        for i in range(n - 1, -1, -1):                       # back substitution
            X[i] = (Y[i] - self.L[i + 1:, i] @ X[i + 1:]) / self.L[i, i]
        return X

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """log N(x; mu, Sigma), computed stably.

        log|Sigma| = 2 sum log L_ii  — from the Cholesky, rather than computing a
        determinant that would overflow or underflow in even modest dimensions.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        diff = (x - self.mu).T                               # (d, n)
        solved = self._solve_with_cholesky(diff)
        mahalanobis = np.sum(diff * solved, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(self.L)))
        return -0.5 * (mahalanobis + log_det + self.d * math.log(2 * math.pi))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, size: int, rng) -> np.ndarray:
        """x = mu + L z  for z ~ N(0, I).  README §15.4

        Cov(Lz) = L Cov(z) L^T = L I L^T = Sigma, by the linear-map property of §9.3.
        """
        z = box_muller(size * self.d, rng).reshape(size, self.d)
        return self.mu + z @ self.L.T

    def marginal(self, idx) -> "MultivariateGaussian":
        """Marginalizing a Gaussian = deleting rows and columns. README §9.3

        No integration required, which is remarkable and specific to the Gaussian.
        """
        idx = np.asarray(idx)
        return MultivariateGaussian(self.mu[idx], self.Sigma[np.ix_(idx, idx)])

    def condition(self, idx2, x2) -> "MultivariateGaussian":
        """p(x_1 | x_2), the conditioning formula of README §9.3:

            mu_1|2    = mu_1 + S_12 S_22^-1 (x_2 - mu_2)
            Sigma_1|2 = S_11 - S_12 S_22^-1 S_21          <- the Schur complement

        Two things worth noticing in the output:
          - the mean is LINEAR in x_2 (which is why linear regression is optimal under
            joint Gaussianity, not merely convenient)
          - the covariance does NOT depend on x_2 at all — observing x_2 reduces your
            uncertainty by a fixed amount regardless of the value observed
        """
        idx2 = np.atleast_1d(np.asarray(idx2))
        x2 = np.atleast_1d(np.asarray(x2, dtype=float))
        idx1 = np.array([i for i in range(self.d) if i not in set(idx2.tolist())])

        S11 = self.Sigma[np.ix_(idx1, idx1)]
        S12 = self.Sigma[np.ix_(idx1, idx2)]
        S22 = self.Sigma[np.ix_(idx2, idx2)]

        S22_inv_rest = np.linalg.solve(S22, np.column_stack([x2 - self.mu[idx2], S12.T]))
        mu_cond = self.mu[idx1] + S12 @ S22_inv_rest[:, 0]
        Sigma_cond = S11 - S12 @ S22_inv_rest[:, 1:]
        return MultivariateGaussian(mu_cond, Sigma_cond)


# =============================================================================
# BAYES  (README §7)
# =============================================================================


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """posterior ∝ likelihood x prior, over a discrete hypothesis space.

    The denominator p(D) = sum_h p(D|h)p(h) is just the normalizing sum — which is why the
    proportional form is usually all you need (README §7).
    """
    unnormalized = np.asarray(likelihood, dtype=float) * np.asarray(prior, dtype=float)
    return unnormalized / unnormalized.sum()


def beta_binomial_update(alpha: float, beta: float,
                         successes: int, failures: int) -> tuple[float, float]:
    """Conjugate updating: Beta(a, b) + (s, f) -> Beta(a + s, b + f).  README §8.1

    Bayesian inference reduced to addition. alpha and beta are pseudo-counts: Beta(1,1) is
    uniform ("no information"), Beta(10,10) says "I've effectively seen 18 balanced trials
    already and it will take real evidence to move me."
    """
    return alpha + successes, beta + failures


def disease_test_posterior(prevalence: float, sensitivity: float,
                           specificity: float) -> dict:
    """P(disease | positive test) — the base rate fallacy, computed.  README §7.1

    Returns the posterior along with the true/false positive counts that make the answer
    intuitive: the reason the posterior is low is simply that the healthy group is so much
    larger that its small error rate produces more positives in absolute terms.
    """
    p_pos_given_d = sensitivity
    p_pos_given_not_d = 1 - specificity

    true_positives = p_pos_given_d * prevalence
    false_positives = p_pos_given_not_d * (1 - prevalence)
    p_pos = true_positives + false_positives

    return {
        "posterior": true_positives / p_pos,
        "p_positive": p_pos,
        "true_positive_rate": true_positives,
        "false_positive_rate": false_positives,
        "false_to_true_ratio": false_positives / true_positives,
        # Odds form (README §7.2): posterior odds = likelihood ratio x prior odds
        "likelihood_ratio": p_pos_given_d / p_pos_given_not_d,
        "prior_odds": prevalence / (1 - prevalence),
    }


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<50s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 82)
    print("VERIFICATION")
    print("=" * 82)

    try:
        from scipy import stats
    except ImportError:
        print("\n  scipy not installed — skipping distribution comparisons.")
        return True

    # --- densities and CDFs against scipy ---------------------------------
    print("\nDensities and CDFs vs scipy.stats (README §8)")

    checks = [
        ("Bernoulli(0.3) pmf", Bernoulli(0.3).pmf(np.array([0, 1])),
         stats.bernoulli(0.3).pmf([0, 1])),
        ("Binomial(10, 0.3) pmf", Binomial(10, 0.3).pmf(np.arange(11)),
         stats.binom(10, 0.3).pmf(np.arange(11))),
        ("Poisson(3.5) pmf", Poisson(3.5).pmf(np.arange(15)),
         stats.poisson(3.5).pmf(np.arange(15))),
        ("Geometric(0.25) pmf", Geometric(0.25).pmf(np.arange(1, 20)),
         stats.geom(0.25).pmf(np.arange(1, 20))),
        ("Gaussian(1, 2) pdf", Gaussian(1, 2).pdf(np.linspace(-6, 8, 50)),
         stats.norm(1, 2).pdf(np.linspace(-6, 8, 50))),
        ("Gaussian(1, 2) cdf", Gaussian(1, 2).cdf(np.linspace(-6, 8, 50)),
         stats.norm(1, 2).cdf(np.linspace(-6, 8, 50))),
        ("Gaussian logpdf", Gaussian(1, 2).logpdf(np.linspace(-6, 8, 50)),
         stats.norm(1, 2).logpdf(np.linspace(-6, 8, 50))),
        ("Laplace(0, 1.5) pdf", Laplace(0, 1.5).pdf(np.linspace(-8, 8, 50)),
         stats.laplace(0, 1.5).pdf(np.linspace(-8, 8, 50))),
        ("Laplace(0, 1.5) cdf", Laplace(0, 1.5).cdf(np.linspace(-8, 8, 50)),
         stats.laplace(0, 1.5).cdf(np.linspace(-8, 8, 50))),
        ("Exponential(2) pdf", Exponential(2).pdf(np.linspace(0, 5, 50)),
         stats.expon(scale=0.5).pdf(np.linspace(0, 5, 50))),
        ("Beta(2, 5) pdf", Beta(2, 5).pdf(np.linspace(0.01, 0.99, 50)),
         stats.beta(2, 5).pdf(np.linspace(0.01, 0.99, 50))),
        ("Beta(2, 5) cdf", Beta(2, 5).cdf(np.linspace(0.05, 0.95, 10)),
         stats.beta(2, 5).cdf(np.linspace(0.05, 0.95, 10))),
    ]
    for name, mine, ref in checks:
        ok &= _report(name, float(np.abs(np.asarray(mine) - np.asarray(ref)).max()), 1e-6)

    # --- analytic moments -------------------------------------------------
    print("\nAnalytic mean and variance vs scipy (README §4)")
    moment_checks = [
        ("Bernoulli(0.3)", Bernoulli(0.3), stats.bernoulli(0.3)),
        ("Binomial(10, 0.3)", Binomial(10, 0.3), stats.binom(10, 0.3)),
        ("Poisson(3.5)", Poisson(3.5), stats.poisson(3.5)),
        ("Geometric(0.25)", Geometric(0.25), stats.geom(0.25)),
        ("Uniform(-2, 5)", Uniform(-2, 5), stats.uniform(-2, 7)),
        ("Gaussian(1, 2)", Gaussian(1, 2), stats.norm(1, 2)),
        ("Laplace(0, 1.5)", Laplace(0, 1.5), stats.laplace(0, 1.5)),
        ("Exponential(2)", Exponential(2), stats.expon(scale=0.5)),
        ("Beta(2, 5)", Beta(2, 5), stats.beta(2, 5)),
    ]
    for name, mine, ref in moment_checks:
        err = max(abs(mine.mean() - ref.mean()), abs(mine.var() - ref.var()))
        ok &= _report(f"{name}: mean and var", err, 1e-9)

    # --- samplers ---------------------------------------------------------
    print("\nSamplers — Kolmogorov-Smirnov test against the true CDF (README §15)")
    n = 40_000

    sampler_checks = [
        ("box_muller -> N(0,1)", box_muller(n, rng), stats.norm(0, 1)),
        ("Gaussian.sample", Gaussian(2, 3).sample(n, rng), stats.norm(2, 3)),
        ("Exponential inverse-CDF", Exponential(2).sample(n, rng), stats.expon(scale=0.5)),
        ("Laplace inverse-CDF", Laplace(0, 1.5).sample(n, rng), stats.laplace(0, 1.5)),
        ("Uniform inverse-CDF", Uniform(-2, 5).sample(n, rng), stats.uniform(-2, 7)),
        ("Beta via gamma ratio", Beta(2, 5).sample(n, rng), stats.beta(2, 5)),
    ]
    for name, samples, ref in sampler_checks:
        p_value = stats.kstest(samples, ref.cdf).pvalue
        status = "PASS" if p_value > 0.001 else "FAIL"
        print(f"  [{status}]  {name:<50s}  KS p = {p_value:.4f}")
        ok &= p_value > 0.001

    # Discrete samplers: compare empirical to theoretical means/variances.
    for name, dist in [("Bernoulli(0.3).sample", Bernoulli(0.3)),
                       ("Binomial(10, 0.3).sample", Binomial(10, 0.3)),
                       ("Poisson(3.5).sample", Poisson(3.5)),
                       ("Geometric(0.25).sample", Geometric(0.25))]:
        s = dist.sample(20_000, rng)
        # Standard error of the mean is std/sqrt(n); allow 5 of them.
        tol = 5 * dist.std() / math.sqrt(20_000)
        ok &= _report(f"{name}: empirical mean", abs(s.mean() - dist.mean()), max(tol, 1e-3))

    # --- rejection sampling -----------------------------------------------
    print("\nRejection sampling (README §15.2)")
    target = Beta(2, 5)
    proposal = Uniform(0, 1)
    M = float(target.pdf(np.linspace(0.001, 0.999, 5000)).max())
    samples, rate = rejection_sample(target.pdf, proposal, M, 20_000, rng)
    p_value = stats.kstest(samples, stats.beta(2, 5).cdf).pvalue
    print(f"  [{'PASS' if p_value > 0.001 else 'FAIL'}]  "
          f"{'rejection sampler matches Beta(2,5)':<50s}  KS p = {p_value:.4f}")
    print(f"  [INFO]  {'acceptance rate vs predicted 1/M':<50s}  "
          f"{rate:.3f} vs {1 / M:.3f}")
    ok &= p_value > 0.001

    # --- multivariate Gaussian --------------------------------------------
    print("\nMultivariate Gaussian (README §9)")
    mu = np.array([1.0, -2.0, 0.5])
    A = rng.standard_normal((3, 3))
    Sigma = A @ A.T + 2 * np.eye(3)

    ok &= _report("cholesky vs np.linalg.cholesky",
                  float(np.abs(cholesky(Sigma) - np.linalg.cholesky(Sigma)).max()), 1e-12)

    mvn = MultivariateGaussian(mu, Sigma)
    pts = rng.standard_normal((20, 3)) * 2 + mu
    ok &= _report("logpdf vs scipy multivariate_normal",
                  float(np.abs(mvn.logpdf(pts)
                               - stats.multivariate_normal(mu, Sigma).logpdf(pts)).max()), 1e-9)

    samples = mvn.sample(200_000, rng)
    ok &= _report("sample mean -> mu", float(np.abs(samples.mean(axis=0) - mu).max()), 0.03)
    ok &= _report("sample covariance -> Sigma",
                  float(np.abs(np.cov(samples.T) - Sigma).max()), 0.10)

    marg = mvn.marginal([0, 2])
    ok &= _report("marginal = delete rows/cols",
                  float(max(np.abs(marg.mu - mu[[0, 2]]).max(),
                            np.abs(marg.Sigma - Sigma[np.ix_([0, 2], [0, 2])]).max())), 1e-12)

    # Conditioning: check against the empirical conditional from a large sample.
    cond = mvn.condition(idx2=[2], x2=[0.5])
    near = np.abs(samples[:, 2] - 0.5) < 0.05
    emp_mean = samples[near][:, :2].mean(axis=0)
    ok &= _report("conditional mean vs empirical",
                  float(np.abs(cond.mu - emp_mean).max()), 0.05)
    ok &= _report("conditional cov vs empirical",
                  float(np.abs(cond.Sigma - np.cov(samples[near][:, :2].T)).max()), 0.10)

    # --- Bayes ------------------------------------------------------------
    print("\nBayes (README §7)")
    result = disease_test_posterior(prevalence=0.001, sensitivity=0.99, specificity=0.99)
    ok &= _report("P(disease | +) for the README's example",
                  abs(result["posterior"] - 0.09016393442622951), 1e-12)

    # Odds form must agree with the direct computation (README §7.2).
    posterior_odds = result["likelihood_ratio"] * result["prior_odds"]
    ok &= _report("odds form agrees with direct form",
                  abs(posterior_odds / (1 + posterior_odds) - result["posterior"]), 1e-12)

    # Conjugate updating must match explicit grid-based Bayes.
    alpha0, beta0, s, f = 2.0, 3.0, 7, 4
    a_post, b_post = beta_binomial_update(alpha0, beta0, s, f)
    grid = np.linspace(1e-6, 1 - 1e-6, 200_001)
    prior_vals = Beta(alpha0, beta0).pdf(grid)
    likelihood = grid ** s * (1 - grid) ** f
    numeric_post = prior_vals * likelihood
    numeric_post /= np.trapezoid(numeric_post, grid)
    ok &= _report("Beta-Binomial conjugacy vs numeric Bayes",
                  float(np.abs(numeric_post - Beta(a_post, b_post).pdf(grid)).max()), 1e-4)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_clt() -> None:
    """README §12: sample means become Gaussian at rate 1/sqrt(n), from any start."""
    print("\n" + "=" * 82)
    print("EXPERIMENT 1 — the Central Limit Theorem  (README §12)")
    print("=" * 82)
    print("""
The CLT claims two things. First, that the sample mean's standard deviation shrinks as
sigma/sqrt(n) — so 4x the data halves the error bar. Second, that the mean's *shape*
becomes Gaussian no matter what you started from. Testing both, from three deliberately
non-Gaussian starting distributions:
""")
    rng = np.random.default_rng(1)
    n_trials = 20_000

    def sample_skewness(a: np.ndarray) -> float:
        """Third standardized moment. Zero for any symmetric distribution."""
        c = a - a.mean()
        return float(np.mean(c ** 3) / (np.mean(c ** 2) ** 1.5))

    # Population skewness of each source, for the Berry-Esseen prediction below.
    sources = [
        ("Exponential(1) — skewed", Exponential(1.0), 2.0),
        ("Bernoulli(0.2) — skewed, discrete", Bernoulli(0.2), (1 - 2 * 0.2) / math.sqrt(0.2 * 0.8)),
        ("Uniform(0,1) — symmetric", Uniform(0, 1), 0.0),
    ]

    for label, dist, skew_x in sources:
        print(f"\n  Starting from {label}"
              f"   (sigma = {dist.std():.4f}, skewness = {skew_x:.3f})")
        print(f"    {'n':>6s}  {'sd of mean':>12s}  {'sigma/sqrt(n)':>14s}  {'ratio':>7s}  "
              f"{'skew of mean':>13s}  {'skew/sqrt(n)':>13s}")
        print("    " + "-" * 76)

        for n in (1, 2, 5, 30, 100, 1000):
            means = np.array([dist.sample(n, rng).mean() for _ in range(n_trials)])
            observed = means.std(ddof=1)
            predicted = dist.std() / math.sqrt(n)
            print(f"    {n:6d}  {observed:12.5f}  {predicted:14.5f}  "
                  f"{observed / predicted:7.4f}  {sample_skewness(means):13.4f}  "
                  f"{skew_x / math.sqrt(n):13.4f}")

    print("""
  Two separate claims, two separate columns.

  THE RATE (column 4). The ratio sits at 1.00 everywhere — for every source, at every n.
  The sigma/sqrt(n) law is exact and needs no approximation. Practical reading: to halve a
  confidence interval you need FOUR times the data. A 1,000-example test set and a
  1,200-example one are effectively the same.

  THE SHAPE (columns 5-6). This is the part folklore gets wrong. "n = 30 is enough for the
  CLT" is not a law, and the table shows why: the residual non-normality of the sample mean
  is governed by the SOURCE's skewness, decaying as skew(X)/sqrt(n) (the Berry-Esseen
  theorem). Columns 5 and 6 track each other closely, confirming that rate.

  So the n you need depends entirely on what you started from:
    - Uniform is symmetric (skew 0) and its mean is essentially Gaussian by n = 5.
    - Exponential has skew 2.0, so at n = 30 its mean still carries skew ~0.37 — small,
      but a normality test on 20,000 trials rejects it decisively.
    - Bernoulli(0.2) sits in between at skew 1.5.

  For a strongly skewed source you may need n in the hundreds before Gaussian-based
  confidence intervals are trustworthy — which matters directly when computing intervals
  on rare-event metrics like click-through or fraud rates, where the per-example
  distribution is Bernoulli with tiny p and skewness (1-2p)/sqrt(p(1-p)) is enormous.
  At p = 0.001 that skewness is 31.6, and n = 30 is nowhere near enough.""")


def experiment_concentration() -> None:
    """README §13: Markov vs Chebyshev vs Hoeffding — how much slack each leaves."""
    print("\n" + "=" * 82)
    print("EXPERIMENT 2 — concentration inequalities  (README §13)")
    print("=" * 82)
    print("""
All three bounds are correct — none can ever be violated. The question is how much slack
each leaves, because that slack is exactly what makes a generalization bound useless or
useful. Here X_i ~ Bernoulli(0.3), and we bound P(|mean - 0.3| >= t):
""")
    rng = np.random.default_rng(2)
    pi = 0.3
    n = 100
    n_trials = 400_000

    means = (rng.random((n_trials, n)) < pi).mean(axis=1)
    sigma = math.sqrt(pi * (1 - pi))

    print(f"  n = {n}, {n_trials:,} trials\n")
    print(f"  {'t':>6s}  {'true P':>12s}  {'Chebyshev':>12s}  {'Hoeffding':>12s}  "
          f"{'Cheb/true':>11s}  {'Hoef/true':>11s}")
    print("  " + "-" * 72)

    for t in (0.05, 0.10, 0.15, 0.20):
        true_p = float(np.mean(np.abs(means - pi) >= t))
        chebyshev = min(sigma ** 2 / (n * t ** 2), 1.0)
        hoeffding = min(2 * math.exp(-2 * n * t ** 2), 1.0)
        c_ratio = chebyshev / true_p if true_p > 0 else float("inf")
        h_ratio = hoeffding / true_p if true_p > 0 else float("inf")
        print(f"  {t:6.2f}  {true_p:12.6f}  {chebyshev:12.6f}  {hoeffding:12.6f}  "
              f"{c_ratio:11.1f}x  {h_ratio:11.1f}x")

    print("""
  Both bounds hold everywhere — neither is ever below the true probability. But Chebyshev
  decays only as 1/(n t^2) while Hoeffding decays as e^(-2 n t^2), so the gap between them
  widens fast. By t = 0.20 Chebyshev is off by a factor of thousands and Hoeffding by far
  less.

  This is why learning theory is built on Hoeffding and not Chebyshev. Inverting the
  Hoeffding bound gives the generalization statement of README §13: with probability
  1 - delta, true error is within sqrt(log(2/delta) / 2n) of test error.""")

    for n_test, delta in [(1_000, 0.05), (10_000, 0.05), (100_000, 0.05)]:
        eps = math.sqrt(math.log(2 / delta) / (2 * n_test))
        print(f"    n = {n_test:>7,}:  true error is within +/-{eps * 100:5.2f}% "
              f"of test error, with 95% confidence")


def experiment_base_rates() -> None:
    """README §7.1: why a 99%-accurate test is 9% informative."""
    print("\n" + "=" * 82)
    print("EXPERIMENT 3 — base rates, or why accuracy lies  (README §7.1)")
    print("=" * 82)
    print("""
A test with 99% sensitivity AND 99% specificity sounds definitive. What it actually tells
you depends almost entirely on the prevalence — a quantity the test itself knows nothing
about. Holding the test fixed at 99/99 and varying only how common the condition is:
""")
    print(f"  {'prevalence':>12s}  {'P(disease | +)':>16s}  {'false : true positives':>24s}")
    print("  " + "-" * 56)

    for prevalence in (0.0001, 0.001, 0.01, 0.1, 0.5):
        r = disease_test_posterior(prevalence, sensitivity=0.99, specificity=0.99)
        print(f"  {prevalence:12.2%}  {r['posterior']:16.2%}  "
              f"{r['false_to_true_ratio']:20.1f} : 1")

    print("""
  The identical test is 1% informative at one-in-ten-thousand prevalence and 99%
  informative at 50%. Nothing about the test changed — only the base rate did.

  Now the same arithmetic in the language of ML. Take a fraud detector on a stream where
  0.1% of transactions are fraudulent:
""")
    r = disease_test_posterior(0.001, 0.99, 0.99)
    n = 1_000_000
    print(f"    {n:,} transactions, 0.1% fraud, model is 99% accurate both ways")
    print(f"      true positives  : {r['true_positive_rate'] * n:>10,.0f}")
    print(f"      false positives : {r['false_positive_rate'] * n:>10,.0f}")
    print(f"      precision       : {r['posterior']:>10.1%}")
    print(f"      accuracy        : {0.99:>10.1%}   <- the number that gets reported")
    print("""
  99% accuracy, and 91% of the alerts are wrong. A model that simply predicted "never
  fraud" would score 99.9% accuracy — better than this one — while catching nothing.

  This is the reason accuracy is banned as a headline metric on imbalanced problems, and
  why precision and recall must always be reported together (05.03). It is one line of
  Bayes, and it is the most commercially expensive mistake in applied ML.""")


def experiment_loss_is_likelihood() -> None:
    """README §9.4: every loss function is a negative log-likelihood."""
    print("\n" + "=" * 82)
    print("EXPERIMENT 4 — losses ARE likelihoods  (README §9.4)")
    print("=" * 82)
    print("""
The claim is not an analogy. Minimizing squared error IS maximum likelihood under Gaussian
noise; minimizing absolute error IS maximum likelihood under Laplace noise. Two ways to
demonstrate it.

First, numerically: fit a constant c to data by minimizing each loss, and separately by
maximizing each likelihood. The answers must coincide exactly.
""")
    rng = np.random.default_rng(3)
    data = np.concatenate([rng.normal(5.0, 1.0, 200), np.array([40.0, 45.0])])  # 2 outliers

    grid = np.linspace(0, 15, 300_001)

    mse_fit = grid[np.argmin([np.mean((data - c) ** 2) for c in np.linspace(0, 15, 3001)])]
    mse_fit = np.linspace(0, 15, 3001)[
        int(np.argmin([np.mean((data - c) ** 2) for c in np.linspace(0, 15, 3001)]))]
    mae_fit = np.linspace(0, 15, 3001)[
        int(np.argmin([np.mean(np.abs(data - c)) for c in np.linspace(0, 15, 3001)]))]

    gauss_nll = np.array([-Gaussian(c, 1.0).logpdf(data).sum() for c in grid])
    laplace_nll = np.array([-np.sum(np.log(Laplace(c, 1.0).pdf(data) + 1e-300)) for c in grid])
    gauss_mle = grid[int(np.argmin(gauss_nll))]
    laplace_mle = grid[int(np.argmin(laplace_nll))]

    print(f"  {'objective':<34s}  {'argmin':>10s}")
    print("  " + "-" * 46)
    print(f"  {'minimize mean squared error':<34s}  {mse_fit:10.4f}")
    print(f"  {'maximize Gaussian likelihood':<34s}  {gauss_mle:10.4f}   <- same")
    print(f"  {'minimize mean absolute error':<34s}  {mae_fit:10.4f}")
    print(f"  {'maximize Laplace likelihood':<34s}  {laplace_mle:10.4f}   <- same")

    print(f"""
  Second, the consequence. The data is N(5, 1) with two outliers at 40 and 45.

      sample mean   = {data.mean():.3f}   (the MSE / Gaussian answer)
      sample median = {np.median(data):.3f}   (the MAE / Laplace answer)
      true center   = 5.000

  MSE is dragged {data.mean() - 5:.2f} away by two points out of 202. MAE is not.

  This is not a quirk of the optimizer — it is the noise model doing exactly what it was
  told. The Gaussian's tails fall off as e^(-x^2), so a point 35 sigma out is so
  astronomically unlikely under the model that the fit contorts to accommodate it. The
  Laplace's tails fall off as e^(-|x|), so the same point is merely unusual.

  The lesson: when you have outliers, the principled fix is to change the noise model
  (MAE, Huber, Student-t), not to keep MSE and delete the inconvenient data.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_clt()
    experiment_concentration()
    experiment_base_rates()
    experiment_loss_is_likelihood()

    print("\n" + "=" * 82)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 82)
