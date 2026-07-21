# Notation

Every chapter in this repository uses the notation below. It follows the conventions of
*The Elements of Statistical Learning* and *Deep Learning* (Goodfellow et al.), with a few
disambiguations where those two disagree. When a chapter must deviate (for example, RL uses
$s, a, r$ and time series uses $t$ for time), the deviation is stated at the top of that chapter.

---

## 1. Typography

| Form | Meaning | Example |
|---|---|---|
| $a$, $\alpha$ | scalar (lowercase, italic) | learning rate $\eta$ |
| $\mathbf{a}$ | column vector (lowercase, bold) | $\mathbf{x} \in \mathbb{R}^{d}$ |
| $\mathbf{A}$ | matrix (uppercase, bold) | $\mathbf{X} \in \mathbb{R}^{n \times d}$ |
| $\mathsf{A}$ | tensor of order $\geq 3$ | a batch of images |
| $\mathcal{A}$ | set / space | $\mathcal{X}$, $\mathcal{Y}$, $\mathcal{H}$ |
| $A$ | random variable (uppercase, italic) | $Y$, $X_j$ |

**All vectors are column vectors.** A row of the design matrix is therefore written
$\mathbf{x}_i^{\top}$.

---

## 2. Data

| Symbol | Meaning |
|---|---|
| $n$ | number of training examples |
| $d$ (or $p$) | number of features / input dimension |
| $K$ | number of classes, clusters, or components |
| $\mathbf{x}_i \in \mathbb{R}^{d}$ | the $i$-th input example |
| $x_{ij}$ | feature $j$ of example $i$ |
| $y_i$ | target for example $i$ |
| $\mathbf{X} \in \mathbb{R}^{n \times d}$ | design matrix; row $i$ is $\mathbf{x}_i^{\top}$ |
| $\mathbf{y} \in \mathbb{R}^{n}$ | target vector |
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ | training set |
| $\mathcal{X}, \mathcal{Y}$ | input space, output space |
| $P(X, Y)$ or $\mathcal{P}$ | the (unknown) true data-generating distribution |

**Index convention**: $i$ indexes examples, $j$ indexes features, $k$ indexes classes or
components, $t$ indexes time or iterations, $\ell$ indexes network layers.

**Intercept**: when a bias term is folded into the weights, we write
$\tilde{\mathbf{x}} = [1, \mathbf{x}^{\top}]^{\top} \in \mathbb{R}^{d+1}$ and say so explicitly.
Otherwise the bias $b$ is kept separate.

---

## 3. Models and predictions

| Symbol | Meaning |
|---|---|
| $f$ | the true (unknown) target function |
| $\hat{f}$, $h$ | the learned model / hypothesis |
| $\mathcal{H}$ | hypothesis space |
| $\boldsymbol{\theta}$ | all model parameters, collectively |
| $\mathbf{w}$, $b$ | weight vector and bias of a linear model |
| $\hat{y}_i = \hat{f}(\mathbf{x}_i)$ | predicted value |
| $\hat{p}(y \mid \mathbf{x})$ | predicted conditional probability |
| $z$ | pre-activation ("logit"), $z = \mathbf{w}^{\top}\mathbf{x} + b$ |
| $\sigma(\cdot)$ | logistic sigmoid, $\sigma(z) = 1/(1 + e^{-z})$ |
| $\phi(\cdot)$ | feature map / basis expansion |
| $\lambda$ | regularization strength |
| $\eta$ (or $\alpha$) | learning rate |

---

## 4. Losses and risk

| Symbol | Meaning |
|---|---|
| $L(y, \hat{y})$ | pointwise loss |
| $J(\boldsymbol{\theta})$ | objective being minimized (loss + regularization) |
| $R(f) = \mathbb{E}_{(X,Y)}[L(Y, f(X))]$ | **true risk** (population, unknowable) |
| $\hat{R}_n(f) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(\mathbf{x}_i))$ | **empirical risk** (what we actually minimize) |
| $\Omega(\boldsymbol{\theta})$ | regularization penalty |
| $\nabla_{\boldsymbol{\theta}} J$ | gradient of $J$ w.r.t. $\boldsymbol{\theta}$ |
| $\mathbf{H}$ | Hessian matrix |
| $\mathbf{J}$ | Jacobian matrix |

The central tension of the whole field, in one line:
we minimize $\hat{R}_n$ but care about $R$.

---

## 5. Probability

| Symbol | Meaning |
|---|---|
| $P(A)$ | probability of event $A$ |
| $p(x)$ | probability density (continuous) or mass (discrete) function |
| $X \sim \mathcal{D}$ | $X$ is distributed according to $\mathcal{D}$ |
| $\mathbb{E}[X]$, $\mathbb{E}_{p}[X]$ | expectation (subscript names the distribution) |
| $\mathrm{Var}(X)$, $\mathrm{Cov}(X, Y)$ | variance, covariance |
| $\boldsymbol{\Sigma}$ | covariance matrix |
| $\boldsymbol{\mu}$ | mean vector |
| $\mathcal{N}(\mu, \sigma^{2})$ | Gaussian / normal distribution |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | multivariate Gaussian |
| $\mathrm{Bern}(\pi)$, $\mathrm{Bin}(n, \pi)$ | Bernoulli, binomial |
| $\mathrm{Cat}(\boldsymbol{\pi})$, $\mathrm{Mult}$ | categorical, multinomial |
| $X \perp Y \mid Z$ | $X$ conditionally independent of $Y$ given $Z$ |
| $\mathcal{L}(\boldsymbol{\theta})$ | likelihood; $\ell(\boldsymbol{\theta}) = \log \mathcal{L}(\boldsymbol{\theta})$ |
| $\mathbb{1}[\cdot]$ | indicator function (1 if true, 0 otherwise) |

---

## 6. Linear algebra

| Symbol | Meaning |
|---|---|
| $\mathbf{A}^{\top}$ | transpose |
| $\mathbf{A}^{-1}$, $\mathbf{A}^{+}$ | inverse, Moore-Penrose pseudoinverse |
| $\mathrm{tr}(\mathbf{A})$, $\det(\mathbf{A})$ | trace, determinant |
| $\mathrm{rank}(\mathbf{A})$ | rank |
| $\mathbf{I}_d$ | $d \times d$ identity |
| $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^{\top}\mathbf{v}$ | inner product |
| $\Vert \mathbf{x}\Vert _p$ | $\ell_p$ norm; $\Vert \mathbf{x}\Vert _2$ Euclidean, $\Vert \mathbf{x}\Vert _1$ Manhattan, $\Vert \mathbf{x}\Vert _0$ count of nonzeros |
| $\Vert \mathbf{A}\Vert _F$ | Frobenius norm |
| $\odot$ | elementwise (Hadamard) product |
| $\otimes$ | Kronecker product |
| $\mathbf{u} \succeq 0$ | elementwise non-negativity |
| $\mathbf{A} \succeq 0$ | $\mathbf{A}$ is positive semidefinite |
| $\lambda_i(\mathbf{A})$, $\sigma_i(\mathbf{A})$ | $i$-th eigenvalue, $i$-th singular value |
| $\kappa(\mathbf{A})$ | condition number, $\sigma_{\max}/\sigma_{\min}$ |

**Gradient layout.** This repository uses **denominator layout**: for a scalar $f$ and vector
$\mathbf{x} \in \mathbb{R}^{d}$, $\nabla_{\mathbf{x}} f \in \mathbb{R}^{d}$ has the same shape as
$\mathbf{x}$. This is the convention that makes gradient descent $\boldsymbol{\theta} \leftarrow
\boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} J$ dimensionally obvious, and it is what
every autodiff framework returns.

---

## 7. Information theory

| Symbol | Meaning |
|---|---|
| $H(X) = -\sum_x p(x)\log p(x)$ | entropy |
| $H(X, Y)$, $H(Y \mid X)$ | joint entropy, conditional entropy |
| $H(p, q) = -\sum_x p(x)\log q(x)$ | cross-entropy |
| $D_{\mathrm{KL}}(p \,\Vert \, q)$ | Kullback-Leibler divergence |
| $I(X; Y)$ | mutual information |

Logs are natural ($\ln$) unless a chapter says otherwise; units are nats. Where bits are more
natural (decision trees, coding arguments), $\log_2$ is used and stated.

---

## 8. Deep learning

| Symbol | Meaning |
|---|---|
| $\mathbf{W}^{[\ell]}$, $\mathbf{b}^{[\ell]}$ | weights and biases of layer $\ell$ |
| $\mathbf{z}^{[\ell]}$, $\mathbf{a}^{[\ell]}$ | pre-activation and activation at layer $\ell$ |
| $g(\cdot)$ | activation function |
| $L$ | number of layers |
| $B$ | mini-batch size |
| $\boldsymbol{\delta}^{[\ell]}$ | error signal $\partial J / \partial \mathbf{z}^{[\ell]}$ |

Superscript $[\ell]$ = layer index, superscript $(t)$ = iteration/time index, subscript $i$ =
example index. These three are never mixed without brackets.

---

## 9. Reinforcement learning

| Symbol | Meaning |
|---|---|
| $s \in \mathcal{S}$, $a \in \mathcal{A}$ | state, action |
| $r$, $R_t$ | reward |
| $\gamma \in [0, 1]$ | discount factor |
| $\pi(a \mid s)$ | policy |
| $V^{\pi}(s)$, $Q^{\pi}(s, a)$ | state-value, action-value function |
| $G_t = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}$ | return |

---

## 10. Common abbreviations

| Abbrev. | Expansion |
|---|---|
| i.i.d. | independent and identically distributed |
| MLE / MAP | maximum likelihood / maximum a posteriori estimation |
| ERM | empirical risk minimization |
| SGD | stochastic gradient descent |
| CV | cross-validation |
| PSD | positive semidefinite |
| w.r.t. / s.t. | with respect to / subject to |

---

## A note on rigour

Derivations in this repository state their assumptions. When a step requires an assumption
(differentiability, independence, invertibility, convexity), the assumption is named at the
point it is used — not buried. When a result holds only asymptotically or only under
conditions that real data violates, that is said plainly. The goal is that you can trust a
derivation here the same way you would trust one in a textbook: by checking it.
