# 03.07 — Support Vector Machines

> **Prerequisites**: [00.02 §13-14](../../00-mathematical-foundations/02-calculus-and-optimization/)
> (Lagrange multipliers, KKT, duality) — this chapter is where that machinery pays off;
> [03.04](../04-logistic-regression/) for the loss comparison.
> **You will be able to**: derive the dual from the primal, *derive* the existence of support
> vectors from complementary slackness rather than asserting it, explain exactly why the kernel
> trick is only visible in the dual, and choose $C$ and $\gamma$ knowing what each does.

---

## Table of contents

1. [The max-margin idea](#1-the-max-margin-idea)
2. [Functional and geometric margin](#2-functional-and-geometric-margin)
3. [The hard-margin primal](#3-the-hard-margin-primal)
4. [The dual, derived](#4-the-dual-derived)
5. [Where support vectors come from](#5-where-support-vectors-come-from)
6. [Soft margin and C](#6-soft-margin-and-c)
7. [The hinge-loss view](#7-the-hinge-loss-view)
8. [The kernel trick](#8-the-kernel-trick)
9. [Kernels and Mercer's condition](#9-kernels-and-mercers-condition)
10. [Choosing C and gamma](#10-choosing-c-and-gamma)
11. [Solving it: SMO](#11-solving-it-smo)
12. [Support vector regression](#12-support-vector-regression)
13. [Multiclass](#13-multiclass)
14. [Complexity and when to use it](#14-complexity-and-when-to-use-it)
15. [Common misconceptions](#15-common-misconceptions)

---

## 1. The max-margin idea

If the data is linearly separable, infinitely many hyperplanes separate it. Which one?

Logistic regression's answer is "the one maximizing likelihood", which — on separable data —
diverges ([03.04 §9](../04-logistic-regression/)). The SVM's answer is different and more
geometric:

> **Choose the hyperplane that is as far as possible from the nearest point of either class.**

The intuition is robustness: a boundary that sits in the widest available corridor tolerates the
most perturbation before it misclassifies anything. It also has a genuine theoretical backing —
generalization bounds that depend on the *margin* rather than on the dimension, which is why SVMs
work in very high-dimensional (even infinite-dimensional) feature spaces
([05.01](../../05-model-evaluation/01-bias-variance-and-theory/)).

---

## 2. Functional and geometric margin

For a hyperplane $\mathbf{w}^{\top}\mathbf{x}+b=0$ and labels $y_i\in\{-1,+1\}$:

$$\textbf{functional margin: } \hat{\gamma}_i = y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)
\qquad
\textbf{geometric margin: } \gamma_i = \frac{y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)}{\Vert\mathbf{w}\Vert}$$

Positive means correctly classified; larger means more confidently so.

**Why we need both.** The functional margin can be inflated arbitrarily by scaling
$(\mathbf{w},b)\to(c\mathbf{w},cb)$ — which changes nothing about the hyperplane. The geometric
margin is the actual perpendicular distance and is scale-invariant. So maximizing the functional
margin is meaningless, and maximizing the geometric margin is the real objective.

**The normalization trick.** Since the scale is free, *fix* it: require the closest points to have
functional margin exactly 1,

$$\min_i\ y_i(\mathbf{w}^{\top}\mathbf{x}_i+b) = 1$$

Then the geometric margin of the closest points is $1/\Vert\mathbf{w}\Vert$, and the width of the
whole corridor is $2/\Vert\mathbf{w}\Vert$. **Maximizing the margin becomes minimizing
$\Vert\mathbf{w}\Vert$** — which is why an objective that looks like regularization is actually a
geometric statement.

---

## 3. The hard-margin primal

$$\boxed{\;\min_{\mathbf{w},b}\ \tfrac12\Vert\mathbf{w}\Vert^{2}
\quad\text{s.t.}\quad y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\ge 1\ \ \forall i\;}$$

The $\tfrac12$ and the square are for differentiability; they do not change the minimizer.

This is a **convex quadratic program**: a convex quadratic objective with linear constraints. So
by [00.02 §6.3](../../00-mathematical-foundations/02-calculus-and-optimization/) it has a unique
global optimum, and by Slater's condition strong duality holds — both facts we will use in §4.

It is also **infeasible if the data is not separable**, which §6 fixes.

---

## 4. The dual, derived

This derivation is the payoff for [00.02 §13-14](../../00-mathematical-foundations/02-calculus-and-optimization/).
Every step is mechanical; the *result* is what makes SVMs interesting.

**Step 1 — the Lagrangian.** One multiplier $\alpha_i\ge0$ per constraint:

$$\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})
= \tfrac12\Vert\mathbf{w}\Vert^{2} - \sum_{i=1}^{n}\alpha_i\big[y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)-1\big]$$

**Step 2 — minimize over the primal variables.**

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = \mathbf{w} - \sum_i\alpha_iy_i\mathbf{x}_i = \mathbf{0}
\;\Longrightarrow\;
\boxed{\;\mathbf{w} = \sum_i \alpha_iy_i\mathbf{x}_i\;}$$

$$\frac{\partial\mathcal{L}}{\partial b} = -\sum_i \alpha_iy_i = 0
\;\Longrightarrow\;
\boxed{\;\sum_i \alpha_iy_i = 0\;}$$

The first is already remarkable: **the optimal weight vector is a linear combination of the
training points**, with the $\alpha_i$ as coefficients. Not an abstract vector — a weighted sum of
data.

**Step 3 — substitute back.** Plugging $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ into
$\mathcal{L}$, the $b$ term vanishes by the second condition, and the quadratic terms combine to
give

$$\boxed{\;\max_{\boldsymbol{\alpha}}\ \sum_{i=1}^{n}\alpha_i
- \tfrac12\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^{\top}\mathbf{x}_j
\quad\text{s.t.}\quad \alpha_i\ge0,\ \ \sum_i\alpha_iy_i=0\;}$$

**Read the two things that changed:**

1. **The dual has $n$ variables** (one per *example*), where the primal had $d$ (one per
   *feature*). When $d\gg n$ — text, genomics, kernel spaces — the dual is a far smaller problem.
2. **The data appears only as inner products $\mathbf{x}_i^{\top}\mathbf{x}_j$.** Nothing else.
   This is the observation §8 turns into the kernel trick, and it is **invisible in the primal**.

The prediction function inherits both properties:

$$f(\mathbf{x}) = \mathbf{w}^{\top}\mathbf{x}+b = \sum_i \alpha_iy_i\,\mathbf{x}_i^{\top}\mathbf{x}+b$$

---

## 5. Where support vectors come from

The KKT conditions ([00.02 §13.2](../../00-mathematical-foundations/02-calculus-and-optimization/))
include **complementary slackness**:

$$\alpha_i\big[y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)-1\big] = 0 \qquad\forall i$$

For each $i$, at least one factor must vanish. So exactly one of two things is true:

| Case | Meaning |
|---|---|
| $\alpha_i = 0$ | the constraint is slack — $y_i f(\mathbf{x}_i) > 1$, the point is strictly outside the margin |
| $y_if(\mathbf{x}_i) = 1$ | the constraint is **active** — the point lies exactly *on* the margin |

> **This is where support vectors come from, and it is a derivation, not a definition.** Points
> strictly outside the margin have $\alpha_i = 0$, so by
> $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ they contribute **nothing** to the model. Only points
> on the margin have $\alpha_i>0$. Those are the **support vectors**, and they are typically a
> small fraction of the data.
>
> The sparsity of the SVM is not a design choice or a regularization side-effect. It is a KKT
> condition. Delete every non-support-vector from your training set and refit: you get the
> *identical* model. Experiment 1 does exactly that.

This also gives a clean way to recover $b$: for any support vector,
$y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)=1$, so $b = y_i - \mathbf{w}^{\top}\mathbf{x}_i$. In
practice average over all of them for numerical stability.

---

## 6. Soft margin and C

Real data is not separable, and the hard-margin problem is then infeasible. Introduce **slack**
$\xi_i\ge0$ allowing violations, and pay for them:

$$\min_{\mathbf{w},b,\boldsymbol{\xi}}\ \tfrac12\Vert\mathbf{w}\Vert^{2} + C\sum_{i=1}^{n}\xi_i
\quad\text{s.t.}\quad y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\ge 1-\xi_i,\ \ \xi_i\ge0$$

| $\xi_i$ | Position |
|---|---|
| $0$ | correctly classified, outside the margin |
| $(0,1)$ | inside the margin, still correct |
| $1$ | exactly on the boundary |
| $>1$ | **misclassified** |

**The dual changes by exactly one character**: $\alpha_i\ge0$ becomes $0\le\alpha_i\le C$. Nothing
else. That is a striking amount of structure preserved, and it is why the same solver handles both.

The box constraint refines §5's classification into three cases:

| $\alpha_i$ | Where the point is |
|---|---|
| $0$ | outside the margin, ignored |
| $(0,C)$ | **exactly on** the margin — a "free" support vector |
| $C$ | inside the margin or misclassified — a "bounded" support vector |

**$C$ is the regularization dial, and it runs backwards from $\lambda$:**

$$\text{large } C \Rightarrow \text{few violations tolerated} \Rightarrow \text{narrow margin, complex boundary, overfitting risk}$$
$$\text{small } C \Rightarrow \text{violations cheap} \Rightarrow \text{wide margin, simple boundary, underfitting risk}$$

Comparing with [03.02](../02-regularized-linear-models/): $C \approx 1/\lambda$. Small $C$ is
*more* regularization — the same inversion as sklearn's logistic regression
([03.04 §10](../04-logistic-regression/)), and the same source of confusion.

---

## 7. The hinge-loss view

Eliminate the slack variables. The constraint $y_if_i\ge1-\xi_i$ with $\xi_i\ge0$ is tight at
$\xi_i=\max(0,1-y_if_i)$, so the problem becomes unconstrained:

$$\boxed{\;\min_{\mathbf{w},b}\ \underbrace{\sum_{i=1}^{n}\max\big(0,\ 1-y_if(\mathbf{x}_i)\big)}_{\text{hinge loss}}
+ \underbrace{\tfrac{1}{2C}\Vert\mathbf{w}\Vert^{2}}_{\ell_2\text{ penalty}}\;}$$

**The SVM is just $\ell_2$-regularized empirical risk minimization with a particular loss.** That
reframing puts it in the same family as everything else in Part 3, and makes the comparison with
logistic regression exact:

| | Hinge (SVM) | Logistic |
|---|---|---|
| Loss | $\max(0, 1-yf)$ | $\log(1+e^{-yf})$ |
| Zero loss when | $yf\ge1$ — **exactly zero** | never — always positive |
| Sparsity | ✅ points with $yf>1$ are ignored | ❌ every point contributes |
| Probabilities | ❌ not a proper scoring rule | ✅ calibrated |
| Outlier sensitivity | grows linearly | grows linearly |

> **The single difference that explains everything else**: hinge loss is *exactly zero* for
> confidently-correct points, while logistic loss is merely *small*. Zero means the gradient is
> zero means the point does not affect the solution — which is the sparsity of §5 seen from the
> loss side rather than the KKT side. Both derivations give the same fact; it is worth being able
> to produce either.
>
> It is also why the SVM gives no probabilities: hinge loss is not a proper scoring rule
> ([00.05 §6.2](../../00-mathematical-foundations/05-information-theory/)), so its output is a
> *score*, not a likelihood. `SVC(probability=True)` runs a separate logistic fit (Platt scaling)
> on those scores.

---

## 8. The kernel trick

From §4, the dual and the prediction depend on the data **only** through inner products. Suppose
we first map to a higher-dimensional space with $\phi$ and run the SVM there. Then everything
depends on $\phi(\mathbf{x}_i)^{\top}\phi(\mathbf{x}_j)$.

**The trick**: if we can compute that inner product *without* ever forming $\phi$, we get the
high-dimensional model for the price of the low-dimensional one.

$$k(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^{\top}\phi(\mathbf{z})$$

**A concrete instance.** For $\mathbf{x},\mathbf{z}\in\mathbb{R}^{2}$ take
$k(\mathbf{x},\mathbf{z}) = (\mathbf{x}^{\top}\mathbf{z})^{2}$. Expanding:

$$(x_1z_1+x_2z_2)^{2} = x_1^{2}z_1^{2} + 2x_1x_2z_1z_2 + x_2^{2}z_2^{2}
= \phi(\mathbf{x})^{\top}\phi(\mathbf{z})$$

with $\phi(\mathbf{x}) = (x_1^{2},\ \sqrt{2}x_1x_2,\ x_2^{2})$. **One multiplication and one
squaring replaces an explicit map into 3 dimensions.** For degree $p$ in $d$ dimensions, $\phi$
has $\binom{d+p}{p}$ components — at $d=100$, $p=5$ that is 96 million — and the kernel still
costs one dot product.

The RBF kernel corresponds to an **infinite-dimensional** $\phi$, which could never be formed at
all.

> **This is only possible in the dual.** The primal solves for $\mathbf{w}\in\mathbb{R}^{d}$
> explicitly, so it must know $d$. The dual solves for $\boldsymbol{\alpha}\in\mathbb{R}^{n}$ and
> never touches the feature space. That is the whole reason [00.02 §14](../../00-mathematical-foundations/02-calculus-and-optimization/)
> insisted duality was worth learning.

The **representer theorem** says this is general: for any $\ell_2$-regularized problem with a loss
depending only on $f(\mathbf{x}_i)$, the optimum has the form
$f(\cdot)=\sum_i\alpha_ik(\mathbf{x}_i,\cdot)$. Kernel ridge regression, Gaussian processes, and
smoothing splines ([03.03 §12](../03-basis-expansion/)) are all instances.

---

## 9. Kernels and Mercer's condition

| Kernel | $k(\mathbf{x},\mathbf{z})$ | Parameters |
|---|---|---|
| **Linear** | $\mathbf{x}^{\top}\mathbf{z}$ | — |
| **Polynomial** | $(\gamma\mathbf{x}^{\top}\mathbf{z}+r)^{p}$ | degree $p$, $\gamma$, $r$ |
| **RBF / Gaussian** | $\exp(-\gamma\Vert\mathbf{x}-\mathbf{z}\Vert^{2})$ | $\gamma>0$ |
| **Sigmoid** | $\tanh(\gamma\mathbf{x}^{\top}\mathbf{z}+r)$ | $\gamma$, $r$ |
| **Laplacian** | $\exp(-\gamma\Vert\mathbf{x}-\mathbf{z}\Vert_1)$ | $\gamma$ |

**Mercer's condition.** A function $k$ is a valid kernel iff the Gram matrix
$K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$ is **symmetric positive semidefinite** for every finite
sample. PSD is exactly what guarantees a $\phi$ exists — it is the same condition that makes a
covariance matrix a covariance matrix
([00.01 §11.2](../../00-mathematical-foundations/01-linear-algebra/)).

If $K$ is not PSD the dual is not concave, the QP has no unique solution, and the solver may not
converge. The sigmoid kernel is **not PSD for all parameter values**, which is why it is rarely
used despite its neural-network-flavoured motivation.

**Building new kernels.** Sums, products, positive scalings, and compositions with a positive power
series of valid kernels are valid. So $k_1+k_2$, $k_1k_2$, $ck_1$, and $\exp(k_1)$ are all kernels
— which is how the RBF is constructed from the linear one.

**Why RBF is the default.** $\Vert\mathbf{x}-\mathbf{z}\Vert^{2}$ makes it a **similarity measure
that decays with distance**, so it is local like KNN ([03.06](../06-knn/)) but smooth; it has one
parameter; and it can approximate any decision boundary. Start there.

---

## 10. Choosing C and gamma

They interact, so tune them **jointly** on a 2-D log grid — never one at a time.

| | Small | Large |
|---|---|---|
| **$C$** | wide margin, many violations tolerated → underfit | narrow margin, few violations → overfit |
| **$\gamma$** (RBF) | each point influences far → smooth, near-linear boundary | each point influences only its neighbourhood → islands around individual points → overfit |

$\gamma$ deserves the sharper warning. In $\exp(-\gamma\Vert\mathbf{x}-\mathbf{z}\Vert^{2})$,
$1/\sqrt{\gamma}$ is a length scale. Make $\gamma$ large enough and every training point becomes
its own island of influence — the model memorizes perfectly and generalizes not at all. **Large
$\gamma$ overfits regardless of $C$**, which is why grid searching $C$ alone can be badly
misleading. Experiment 5 shows the interaction.

sklearn's `gamma="scale"` default is $1/(d\cdot\mathrm{Var}(X))$, which adapts to the data's
dimension and spread and is a genuinely good starting point.

⚠️ **Standardize your features.** The RBF kernel is a function of Euclidean distance, so it
inherits every scaling pathology of [03.06 §4](../06-knn/) — one large-range feature dominates the
distance and the rest are invisible.

---

## 11. Solving it: SMO

The dual is a QP with $n$ variables and a dense $n\times n$ Gram matrix. General QP solvers are
$O(n^{3})$ in time and $O(n^{2})$ in memory, which is hopeless above ~10,000 points.

**Sequential Minimal Optimization** (Platt, 1998) exploits the structure: optimize **exactly two**
multipliers at a time, holding the rest fixed. Why two and not one? The constraint
$\sum_i\alpha_iy_i=0$ means changing one $\alpha$ alone would violate it — two is the smallest
number that can move while preserving the constraint.

With only two free variables the subproblem has a **closed-form solution**. No inner optimizer, no
line search, no matrix factorization — just clipping an analytic update to the box $[0,C]$.

The remaining art is *choosing which pair*: heuristics that pick the pair most violating the KKT
conditions converge far faster than random selection. `libsvm` — which is what
`sklearn.svm.SVC` wraps — is a refined SMO.

---

## 12. Support vector regression

Same idea, different loss: the **$\epsilon$-insensitive** loss

$$L_\epsilon(y,f) = \max\big(0,\ \lvert y-f\rvert-\epsilon\big)$$

Errors smaller than $\epsilon$ cost **nothing**. Geometrically, fit the flattest tube of width
$2\epsilon$ that contains most of the data.

The same sparsity appears for the same reason: points strictly inside the tube have $\alpha_i=0$
and are ignored; only points on or outside the tube are support vectors. $\epsilon$ controls tube
width (and therefore sparsity), $C$ controls the penalty for leaving it.

---

## 13. Multiclass

SVMs are intrinsically binary. Two standard extensions:

| | Classifiers | Cost | Notes |
|---|---|---|---|
| **One-vs-rest** | $K$ | each on all $n$ | scores not comparable across classifiers |
| **One-vs-one** | $K(K-1)/2$ | each on ~$2n/K$ | what libsvm/sklearn use |

One-vs-one trains more classifiers but each on a much smaller subset, and since SVM training is
super-linear in $n$, it is usually *faster* overall. It also avoids the class-imbalance that
one-vs-rest creates.

---

## 14. Complexity and when to use it

| Operation | Cost |
|---|---|
| Train (SMO) | $O(n^{2})$ to $O(n^{3})$ — dominated by kernel evaluations |
| Memory | $O(n^{2})$ for the Gram matrix (cached in practice) |
| Predict (linear) | $O(d)$ |
| Predict (kernel) | $O(n_{SV}\cdot d)$ — **scales with the number of support vectors** |

> **The $O(n^{2})$ training cost is why SVMs lost the large-data era.** Above roughly $10^{5}$
> samples a kernel SVM becomes impractical, and gradient boosting or a neural network will be both
> faster and more accurate. `LinearSVC` (which solves the primal with liblinear) scales to
> millions, but then you have given up the kernel.

**Use it when:**
- $n$ is moderate (thousands to tens of thousands) and $d$ is large — the dual's size is $n$, not $d$
- The boundary is nonlinear and you want it without hand-engineering features
- Text classification with a linear kernel — still an excellent, fast baseline
- You need a *sparse* model at prediction time and the support vectors are few

**Don't when:**
- $n > 10^{5}$ (kernel), or you need probabilities directly, or you need interpretability
- The data is very noisy — the margin concept degrades and $C$ becomes hard to tune

---

## 15. Common misconceptions

**"SVMs find the boundary that separates the classes."**
So does any classifier. SVMs find the boundary with the **largest margin** (§1).

**"Support vectors are the points nearest the boundary."**
They are the points **on or inside** the margin. In a soft-margin SVM, misclassified points far on
the wrong side are also support vectors, with $\alpha_i=C$ (§6).

**"The kernel trick maps data to a higher-dimensional space."**
It computes inner products *as if* you had, without ever forming the mapping. For RBF the space is
infinite-dimensional and could not be formed (§8).

**"Any similarity function can be a kernel."**
It must be PSD (Mercer's condition, §9). The sigmoid kernel is not, for many parameter values.

**"Larger $C$ means more regularization."**
Backwards: $C\approx1/\lambda$. Small $C$ is more regularization (§6).

**"SVMs output probabilities."**
They output signed distances. `probability=True` fits a separate logistic model on those scores
(§7).

**"SVMs don't overfit because they maximize the margin."**
With RBF and large $\gamma$ they overfit spectacularly (§10).

**"You should always use the RBF kernel."**
For $d\gg n$ — text, genomics — the linear kernel is usually as good and far faster. Try linear
first.

**"SVMs are obsolete."**
For $n$ in the thousands with high $d$, a well-tuned SVM is still competitive, trains in seconds,
and is far less fiddly than a network. The margin theory is also load-bearing for understanding
generalization.

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | SVM by simplified SMO with five kernels, soft margin, SVR with the $\epsilon$-tube, and explicit numerical verification of every KKT condition. Verified against sklearn |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sections used |

**Previous**: [03.06 — k-Nearest Neighbours](../06-knn/) ·
**Next**: [03.08 — Decision Trees](../08-decision-trees/)
