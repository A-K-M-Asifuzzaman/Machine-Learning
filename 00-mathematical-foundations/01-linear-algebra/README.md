# 00.01 — Linear Algebra for Machine Learning

> **Prerequisites**: high-school algebra. Nothing else.
> **You will be able to**: read any ML paper's linear algebra without stalling, derive the
> normal equations geometrically, explain PCA three different ways, and know why your matrix
> inversion blew up.

---

## Table of contents

1. [Why linear algebra is the language of ML](#1-why-linear-algebra-is-the-language-of-ml)
2. [Vectors and vector spaces](#2-vectors-and-vector-spaces)
3. [Matrices as linear maps](#3-matrices-as-linear-maps)
4. [Rank and the four fundamental subspaces](#4-rank-and-the-four-fundamental-subspaces)
5. [Norms and inner products](#5-norms-and-inner-products)
6. [Orthogonality and projection](#6-orthogonality-and-projection)
7. [Least squares, derived geometrically](#7-least-squares-derived-geometrically)
8. [Gram-Schmidt and QR](#8-gram-schmidt-and-qr)
9. [Determinant and trace](#9-determinant-and-trace)
10. [Eigenvalues and eigenvectors](#10-eigenvalues-and-eigenvectors)
11. [Symmetric matrices, quadratic forms, positive definiteness](#11-symmetric-matrices-quadratic-forms-and-positive-definiteness)
12. [The Singular Value Decomposition](#12-the-singular-value-decomposition)
13. [Low-rank approximation and the pseudoinverse](#13-low-rank-approximation-and-the-pseudoinverse)
14. [Matrix calculus](#14-matrix-calculus)
15. [Conditioning and numerical stability](#15-conditioning-and-numerical-stability)
16. [Where each concept shows up in ML](#16-where-each-concept-shows-up-in-ml)
17. [Common misconceptions](#17-common-misconceptions)

---

## 1. Why linear algebra is the language of ML

Three facts force linear algebra on us.

**Fact 1 — Data is naturally a matrix.** $n$ examples, each described by $d$ numbers, is
$\mathbf{X} \in \mathbb{R}^{n \times d}$. This isn't a convenience; it means every statement
about your dataset is a statement about a matrix. "The features are redundant" means
$\operatorname{rank}(\mathbf{X}) < d$. "Two features are duplicates" means two columns are
linearly dependent. "The data lies on a plane" means the rows live in a 2-dimensional subspace.

**Fact 2 — The simplest useful model is linear.** $\hat{y} = \mathbf{w}^{\top}\mathbf{x}$. Every
more complex model is either a composition of linear maps with nonlinearities between them (a
neural network), a linear model in a transformed space (kernel methods, basis expansion), or a
piecewise-constant approximation (trees). Understanding the linear case is not a warm-up — it is
the core, and everything else is a modification.

**Fact 3 — Hardware is built for it.** GPUs are matrix multipliers. An algorithm expressible as
dense linear algebra runs orders of magnitude faster than the same algorithm expressed as loops.
This is why ML looks the way it does: methods that vectorize won.

There is a fourth, subtler reason. **Linear algebra gives you geometry.** Once you see least
squares as a projection onto a subspace, you never need to memorize the normal equations again —
you can rederive them in one line. That transfer, from formula-memorizing to picture-seeing, is
the actual goal of this chapter.

---

## 2. Vectors and vector spaces

### 2.1 Three ways to read a vector

A vector $\mathbf{x} \in \mathbb{R}^{d}$ is an ordered list of $d$ real numbers. It admits three
readings, and fluency means switching between them without thinking:

| Reading | Interpretation | Used when |
|---|---|---|
| **List** | a record: `[age=34, income=52000, visits=7]` | thinking about data |
| **Point** | a location in $d$-dimensional space | thinking about distance, clustering |
| **Arrow** | a direction and magnitude from the origin | thinking about gradients, projections |

A gradient is an arrow. A data point is a point. A weight vector is both — it points in the
direction of steepest increase of the model output.

### 2.2 The two operations

A vector space is a set closed under two operations:

$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ \vdots \\ x_d + y_d\end{bmatrix},
\qquad
c\mathbf{x} = \begin{bmatrix} cx_1 \\ \vdots \\ cx_d \end{bmatrix}$$

"Closed" means: add two vectors from the space, you stay in the space; scale one, you stay in
the space. This is the entire definition, and it has a strong consequence — **every vector space
contains the zero vector** (take $c = 0$).

That consequence has teeth. The set of points satisfying $\mathbf{w}^{\top}\mathbf{x} = 0$ is a
vector space (a hyperplane through the origin). The set satisfying
$\mathbf{w}^{\top}\mathbf{x} = 5$ is **not** — it's an *affine* set, a shifted subspace. This is
exactly why a bias term $b$ is handled separately from the weights $\mathbf{w}$: without $b$
your decision boundary is forced through the origin.

### 2.3 Span, linear independence, basis, dimension

**Linear combination.** Given vectors $\mathbf{v}_1, \dots, \mathbf{v}_k$ and scalars
$c_1, \dots, c_k$:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_k\mathbf{v}_k$$

**Span.** $\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_k\}$ is the set of *all* linear
combinations — everything reachable from those vectors. Geometrically: one nonzero vector spans a
line; two independent vectors span a plane; $k$ independent vectors span a $k$-dimensional
subspace.

**Linear independence.** $\{\mathbf{v}_1,\dots,\mathbf{v}_k\}$ is linearly independent if

$$c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k = \mathbf{0} \;\Longrightarrow\; c_1 = \dots = c_k = 0$$

In words: **no vector in the set is a linear combination of the others** — none is redundant.

> **ML translation.** If feature columns of $\mathbf{X}$ are linearly dependent, at least one
> feature carries no information the others don't already have. This is *perfect
> multicollinearity*, and it makes $\mathbf{X}^{\top}\mathbf{X}$ singular — the normal equations
> have no unique solution. The classic cause is the **dummy variable trap**: one-hot encoding a
> $K$-category feature into $K$ columns, which always sum to the all-ones vector, making them
> dependent with the intercept. Drop one category, or regularize.

**Basis.** A linearly independent set that spans the space. Every vector then has a **unique**
representation in that basis. The standard basis of $\mathbb{R}^3$ is
$\mathbf{e}_1 = [1,0,0]^\top,\ \mathbf{e}_2 = [0,1,0]^\top,\ \mathbf{e}_3 = [0,0,1]^\top$.

**Dimension.** The number of vectors in any basis (all bases of a space have the same size).

> **This is the whole idea behind dimensionality reduction.** Your data sits in $\mathbb{R}^{d}$,
> but if it actually lies near a $k$-dimensional subspace with $k \ll d$, you can change to a
> basis adapted to the data and describe each point with $k$ numbers instead of $d$, losing
> almost nothing. PCA is precisely the algorithm that finds the best such basis. See §12.

---

## 3. Matrices as linear maps

### 3.1 The two readings of a matrix

**Reading 1 — a table of data.** $\mathbf{X} \in \mathbb{R}^{n \times d}$: rows are examples,
columns are features.

**Reading 2 — a function.** $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a *linear map*
$\mathbb{R}^{n} \to \mathbb{R}^{m}$, $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$, satisfying

$$\mathbf{A}(c\mathbf{x} + \mathbf{y}) = c\mathbf{A}\mathbf{x} + \mathbf{A}\mathbf{y}$$

Reading 2 is the one that unlocks everything. **A matrix is a verb, not a noun.**

### 3.2 Matrix-vector product: two views

$$\mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^{\top}\mathbf{x} \\ \vdots \\
\mathbf{a}_m^{\top}\mathbf{x}\end{bmatrix}
\qquad\text{(row view: a stack of inner products)}$$

$$\mathbf{A}\mathbf{x} = x_1 \mathbf{a}^{(1)} + x_2\mathbf{a}^{(2)} + \dots + x_n\mathbf{a}^{(n)}
\qquad\text{(column view: a linear combination of columns)}$$

The **column view is the important one**. It says immediately:

$$\boxed{\;\mathbf{A}\mathbf{x} \text{ always lands in } \operatorname{span}\{\text{columns of } \mathbf{A}\}\;}$$

So $\mathbf{A}\mathbf{x} = \mathbf{b}$ has a solution **if and only if** $\mathbf{b}$ lies in the
column space of $\mathbf{A}$. When it doesn't — which is the normal situation in regression,
since $n > d$ means $\mathbf{y}$ almost never lies in a $d$-dimensional column space — you can't
solve it, so you settle for the closest point you *can* reach. That is least squares (§7), and
you have just derived its existence from the column view alone.

### 3.3 Matrix multiplication is function composition

$(\mathbf{A}\mathbf{B})\mathbf{x} = \mathbf{A}(\mathbf{B}\mathbf{x})$: apply $\mathbf{B}$, then
$\mathbf{A}$. This single fact explains:

- **Why $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$**: rotating then scaling ≠ scaling then
  rotating. Function composition doesn't commute.
- **Why $(\mathbf{A}\mathbf{B})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$** and
  $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$: undoing a composition means
  undoing the last step first (socks then shoes; shoes off then socks off).
- **Why a deep network without nonlinearities is pointless**: $\mathbf{W}_3\mathbf{W}_2
  \mathbf{W}_1\mathbf{x} = \mathbf{W}\mathbf{x}$ for $\mathbf{W} = \mathbf{W}_3\mathbf{W}_2
  \mathbf{W}_1$. Any composition of linear maps is a single linear map. **The nonlinearity is
  the only reason depth buys you anything.**

**Cost.** $\mathbf{A} \in \mathbb{R}^{m\times n}$ times $\mathbf{B} \in \mathbb{R}^{n \times p}$
is $O(mnp)$. For square $n \times n$, that's $O(n^3)$ — the dominant cost in most numerical ML.
(Strassen gives $O(n^{2.807})$ and the theoretical record is ≈$O(n^{2.37})$, but neither is used
in practice: constants and numerical stability kill them.)

### 3.4 Named matrices

| Matrix | Definition | Meaning as a map |
|---|---|---|
| Identity $\mathbf{I}$ | $1$ on diagonal, $0$ elsewhere | do nothing |
| Diagonal $\mathbf{D}$ | nonzero only on diagonal | scale each axis independently |
| Orthogonal $\mathbf{Q}$ | $\mathbf{Q}^{\top}\mathbf{Q} = \mathbf{I}$ | rotation/reflection — preserves all lengths and angles |
| Symmetric $\mathbf{A} = \mathbf{A}^{\top}$ | mirror across diagonal | scale along orthogonal axes (§11) |
| Projection $\mathbf{P}$ | $\mathbf{P}^2 = \mathbf{P}$ | flatten onto a subspace; doing it twice changes nothing |
| Positive definite | $\mathbf{x}^{\top}\mathbf{A}\mathbf{x} > 0\ \forall \mathbf{x}\neq\mathbf{0}$ | a bowl — never flips a vector past 90° |

**Orthogonal matrices deserve special attention.** $\mathbf{Q}^{\top}\mathbf{Q} = \mathbf{I}$
means $\mathbf{Q}^{-1} = \mathbf{Q}^{\top}$ — the inverse is free. And

$$\|\mathbf{Q}\mathbf{x}\|_2^2 = (\mathbf{Q}\mathbf{x})^{\top}(\mathbf{Q}\mathbf{x})
= \mathbf{x}^{\top}\mathbf{Q}^{\top}\mathbf{Q}\mathbf{x} = \mathbf{x}^{\top}\mathbf{x} = \|\mathbf{x}\|_2^2$$

Lengths are preserved, so no error is ever amplified. **This is why every numerically serious
algorithm — QR, SVD, Householder — is built out of orthogonal transformations.**

---

## 4. Rank and the four fundamental subspaces

### 4.1 Rank

$$\operatorname{rank}(\mathbf{A}) = \dim(\text{column space}) = \dim(\text{row space})$$

That row rank equals column rank is not obvious and is one of the genuinely surprising theorems
of the subject. Rank is **the number of independent directions the matrix actually uses** — its
true information content.

Properties worth memorizing:

- $\operatorname{rank}(\mathbf{A}) \le \min(m, n)$. Equality = "full rank".
- $\operatorname{rank}(\mathbf{A}\mathbf{B}) \le \min(\operatorname{rank}\mathbf{A}, \operatorname{rank}\mathbf{B})$ — **multiplying can never increase rank.**
- $\operatorname{rank}(\mathbf{A}) = \operatorname{rank}(\mathbf{A}^{\top}\mathbf{A}) = \operatorname{rank}(\mathbf{A}\mathbf{A}^{\top})$
- $\operatorname{rank}(\mathbf{A}) = $ number of nonzero singular values (§12) — **the only numerically reliable way to compute it.**

> **ML translation.** Rank is why **low-rank adaptation (LoRA)** works. Fine-tuning a weight
> matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ normally updates $mn$ parameters. LoRA writes
> the update as $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ with $\mathbf{B} \in \mathbb{R}^{m\times r}$,
> $\mathbf{A} \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$. By the rank inequality above,
> $\operatorname{rank}(\Delta\mathbf{W}) \le r$, and the parameter count drops from $mn$ to
> $r(m+n)$. For $m = n = 4096, r = 8$: 16.7M → 65K parameters, a 256× reduction. The empirical
> claim LoRA makes is that the *useful* fine-tuning update genuinely is near-low-rank.

### 4.2 The four fundamental subspaces

For $\mathbf{A} \in \mathbb{R}^{m\times n}$ with $r = \operatorname{rank}(\mathbf{A})$:

| Subspace | Definition | Lives in | Dimension |
|---|---|---|---|
| **Column space** $C(\mathbf{A})$ | $\{\mathbf{A}\mathbf{x} : \mathbf{x}\in\mathbb{R}^{n}\}$ | $\mathbb{R}^{m}$ | $r$ |
| **Null space** $N(\mathbf{A})$ | $\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$ | $\mathbb{R}^{n}$ | $n - r$ |
| **Row space** $C(\mathbf{A}^{\top})$ | $\{\mathbf{A}^{\top}\mathbf{y} : \mathbf{y}\in\mathbb{R}^{m}\}$ | $\mathbb{R}^{n}$ | $r$ |
| **Left null space** $N(\mathbf{A}^{\top})$ | $\{\mathbf{y} : \mathbf{A}^{\top}\mathbf{y} = \mathbf{0}\}$ | $\mathbb{R}^{m}$ | $m - r$ |

**Rank-nullity theorem**: $\operatorname{rank}(\mathbf{A}) + \dim N(\mathbf{A}) = n$.

The deep structural fact — the **Fundamental Theorem of Linear Algebra** — is that these pair up
*orthogonally*:

$$C(\mathbf{A}^{\top}) \perp N(\mathbf{A}) \quad\text{in } \mathbb{R}^{n}, \qquad
C(\mathbf{A}) \perp N(\mathbf{A}^{\top}) \quad\text{in } \mathbb{R}^{m}$$

*Proof of the first (one line).* If $\mathbf{x} \in N(\mathbf{A})$ then $\mathbf{A}\mathbf{x} =
\mathbf{0}$, so for any $\mathbf{y}$: $(\mathbf{A}^{\top}\mathbf{y})^{\top}\mathbf{x} =
\mathbf{y}^{\top}\mathbf{A}\mathbf{x} = \mathbf{y}^{\top}\mathbf{0} = 0$. Every row-space vector
is orthogonal to every null-space vector. $\blacksquare$

**Why you should care.** The null space is the set of directions the model **cannot see**. If
$\mathbf{v} \in N(\mathbf{X})$, then $\mathbf{X}(\mathbf{w} + \mathbf{v}) = \mathbf{X}\mathbf{w}$
— adding $\mathbf{v}$ to your weights changes no prediction at all. So whenever $\mathbf{X}$ has
a nontrivial null space (guaranteed when $d > n$: more features than examples), **the optimal
weights are not unique** — there is an entire $(n-r)$-dimensional flat of equally optimal
solutions. Regularization's job is to pick one of them. Ridge picks the minimum-$\ell_2$-norm
solution; that is not an aesthetic choice but the direct consequence of adding
$\lambda\|\mathbf{w}\|_2^2$ to a problem whose loss term is flat along the null space.

---

## 5. Norms and inner products

### 5.1 Inner product

$$\langle \mathbf{x}, \mathbf{y}\rangle = \mathbf{x}^{\top}\mathbf{y} = \sum_{i=1}^{d} x_i y_i
= \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta$$

The second equality is the one to internalize: **the dot product measures alignment.**

| $\mathbf{x}^{\top}\mathbf{y}$ | Meaning |
|---|---|
| $> 0$ | same general direction ($\theta < 90°$) |
| $= 0$ | orthogonal ($\theta = 90°$) |
| $< 0$ | opposing ($\theta > 90°$) |

Every score in ML is a dot product: $\mathbf{w}^{\top}\mathbf{x}$ asks "how aligned is this
example with what I've learned to look for?" Attention's $\mathbf{q}^{\top}\mathbf{k}$ asks "how
relevant is this key to this query?" Cosine similarity in an embedding space asks the same
question with magnitude divided out.

### 5.2 Norms

A norm measures length. It must satisfy: $\|\mathbf{x}\| \ge 0$ with equality iff
$\mathbf{x} = \mathbf{0}$; $\|c\mathbf{x}\| = |c|\|\mathbf{x}\|$; and the triangle inequality
$\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$.

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^d |x_i|^p\right)^{1/p}$$

| Norm | Formula | Unit ball shape | Role in ML |
|---|---|---|---|
| $\ell_0$ | $\#\{i : x_i \neq 0\}$ | — (not a norm) | true sparsity count; NP-hard to optimize |
| $\ell_1$ | $\sum_i \lvert x_i\rvert$ | diamond | **Lasso** — produces exact zeros |
| $\ell_2$ | $\sqrt{\sum_i x_i^2}$ | circle | **Ridge**, weight decay, Euclidean distance |
| $\ell_\infty$ | $\max_i \lvert x_i \rvert$ | square | adversarial robustness budgets |

**Why $\ell_1$ gives sparsity, geometrically.** Minimizing loss subject to $\|\mathbf{w}\|_1 \le t$
means finding where the loss contours first touch the constraint ball. The $\ell_1$ ball is a
diamond — it has **corners on the axes**, and a corner on axis $j$ is a point where every other
coordinate is exactly zero. A randomly-oriented contour is far more likely to first touch a
pointy corner than a flat face. The $\ell_2$ ball is round: no corners, so the touch point
generically has all coordinates nonzero but small. That is the entire intuition behind
Lasso-vs-Ridge, and it is developed fully in [03.02](../../03-supervised-learning/02-regularized-linear-models/).

**Matrix norms.** The Frobenius norm treats a matrix as a long vector:
$\|\mathbf{A}\|_F = \sqrt{\sum_{ij} a_{ij}^2} = \sqrt{\operatorname{tr}(\mathbf{A}^{\top}\mathbf{A})}
= \sqrt{\sum_i \sigma_i^2}$. The **spectral norm** $\|\mathbf{A}\|_2 = \sigma_{\max}$ is the
maximum stretch factor the matrix applies to any vector — the quantity that controls whether
gradients explode through a layer.

---

## 6. Orthogonality and projection

### 6.1 Projecting onto a line

Project $\mathbf{b}$ onto the line spanned by $\mathbf{a}$. The projection is $\hat{\mathbf{b}} =
c\mathbf{a}$ for some scalar $c$, and the defining property is that the **error is orthogonal to
the line**:

$$\mathbf{a}^{\top}(\mathbf{b} - c\mathbf{a}) = 0
\;\Longrightarrow\; \mathbf{a}^{\top}\mathbf{b} = c\,\mathbf{a}^{\top}\mathbf{a}
\;\Longrightarrow\; c = \frac{\mathbf{a}^{\top}\mathbf{b}}{\mathbf{a}^{\top}\mathbf{a}}$$

$$\hat{\mathbf{b}} = \frac{\mathbf{a}\mathbf{a}^{\top}}{\mathbf{a}^{\top}\mathbf{a}}\mathbf{b}
\qquad\Longrightarrow\qquad
\mathbf{P} = \frac{\mathbf{a}\mathbf{a}^{\top}}{\mathbf{a}^{\top}\mathbf{a}}$$

Note $\mathbf{a}\mathbf{a}^{\top}$ is a $d\times d$ **matrix** (outer product) while
$\mathbf{a}^{\top}\mathbf{a}$ is a **scalar** (inner product). Confusing these two is the single
most common linear algebra error; check shapes every time.

### 6.2 Projecting onto a subspace

Now project $\mathbf{b} \in \mathbb{R}^{m}$ onto $C(\mathbf{A})$ for
$\mathbf{A} \in \mathbb{R}^{m \times n}$. Write $\hat{\mathbf{b}} = \mathbf{A}\hat{\mathbf{x}}$.
The error $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$ must be orthogonal to **every** column of
$\mathbf{A}$ — equivalently, it lies in the left null space:

$$\mathbf{A}^{\top}(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0}$$

$$\boxed{\;\mathbf{A}^{\top}\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^{\top}\mathbf{b}\;}
\qquad \text{(the \textbf{normal equations})}$$

If $\mathbf{A}$ has independent columns, $\mathbf{A}^{\top}\mathbf{A}$ is invertible and

$$\hat{\mathbf{x}} = (\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}\mathbf{b},
\qquad
\mathbf{P} = \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}$$

**Verify the two properties a projection must have:**

- *Idempotent*: $\mathbf{P}^2 = \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}\underbrace{\mathbf{A}^{\top}
  \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}}_{=\ \mathbf{I}}\mathbf{A}^{\top} = \mathbf{P}$ ✓
  (projecting twice = projecting once)
- *Symmetric*: $\mathbf{P}^{\top} = \mathbf{P}$ ✓ (since $\mathbf{A}^{\top}\mathbf{A}$ is symmetric)

$\mathbf{I} - \mathbf{P}$ is also a projection — onto the orthogonal complement. It maps
$\mathbf{b}$ to the residual.

---

## 7. Least squares, derived geometrically

Here is the payoff. The linear regression problem is

$$\min_{\mathbf{w}} \; \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$$

**The geometric argument, in full.** $\mathbf{X}\mathbf{w}$ ranges over $C(\mathbf{X})$, a
subspace of $\mathbb{R}^{n}$ of dimension at most $d$. Since $n > d$, the target $\mathbf{y}$
generically does **not** lie in this subspace — the system $\mathbf{X}\mathbf{w} = \mathbf{y}$ has
no solution. So we take the next best thing: the point of $C(\mathbf{X})$ closest to
$\mathbf{y}$. By §6 that is the orthogonal projection, and the projection is characterized by the
residual being orthogonal to the column space:

$$\mathbf{X}^{\top}(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0}
\quad\Longrightarrow\quad
\boxed{\;\hat{\mathbf{w}} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}\;}$$

**The calculus argument, for comparison.** Expand and differentiate:

$$
\begin{aligned}
J(\mathbf{w}) &= (\mathbf{y}-\mathbf{X}\mathbf{w})^{\top}(\mathbf{y}-\mathbf{X}\mathbf{w}) \\
&= \mathbf{y}^{\top}\mathbf{y} - 2\mathbf{w}^{\top}\mathbf{X}^{\top}\mathbf{y}
   + \mathbf{w}^{\top}\mathbf{X}^{\top}\mathbf{X}\mathbf{w} \\[4pt]
\nabla_{\mathbf{w}} J &= -2\mathbf{X}^{\top}\mathbf{y} + 2\mathbf{X}^{\top}\mathbf{X}\mathbf{w}
\;\overset{!}{=}\; \mathbf{0}
\end{aligned}
$$

Same answer. But notice what the geometric version gives you that the calculus version doesn't:

1. **It's obviously a minimum.** No need to check the Hessian — the projection is *by definition*
   the closest point.
2. **It explains what happens when $\mathbf{X}^{\top}\mathbf{X}$ is singular.** The projection
   $\hat{\mathbf{y}}$ still exists and is still unique; it's the *coordinates* $\hat{\mathbf{w}}$
   that aren't, because the columns can express $\hat{\mathbf{y}}$ in more than one way. The
   fitted values are fine; the coefficients are meaningless. **This is exactly what
   multicollinearity does to a regression.**
3. **It tells you the residual is orthogonal to every feature** — the basis of every regression
   diagnostic plot you will ever draw.

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}} = \underbrace{\mathbf{X}(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}}_{\text{the \emph{hat matrix} } \mathbf{H}}\mathbf{y}$$

$\mathbf{H}$ "puts the hat on $\mathbf{y}$". Its diagonal entries $h_{ii}$ are the **leverages** —
how much example $i$ pulls its own fitted value — and $\operatorname{tr}(\mathbf{H}) = d$, giving
the standard "effective degrees of freedom" count.

> ⚠️ **Never actually compute $(\mathbf{X}^{\top}\mathbf{X})^{-1}$.** Forming
> $\mathbf{X}^{\top}\mathbf{X}$ squares the condition number (§15), destroying precision. Use QR
> or SVD. `numpy.linalg.lstsq` and `sklearn`'s `LinearRegression` both use SVD-based solvers for
> exactly this reason. The formula is for *understanding*, not for *computing*.

---

## 8. Gram-Schmidt and QR

Given independent $\{\mathbf{a}_1,\dots,\mathbf{a}_n\}$, build an orthonormal basis spanning the
same space. The idea: take each vector, **subtract off everything already explained** by the
vectors you've orthonormalized so far, and normalize what's left.

$$
\begin{aligned}
\mathbf{u}_k &= \mathbf{a}_k - \sum_{j<k} (\mathbf{q}_j^{\top}\mathbf{a}_k)\,\mathbf{q}_j
& &\text{(remove components along previous directions)}\\
\mathbf{q}_k &= \mathbf{u}_k / \|\mathbf{u}_k\|_2 & &\text{(normalize)}
\end{aligned}
$$

Collecting the coefficients gives the **QR decomposition**:

$$\mathbf{A} = \mathbf{Q}\mathbf{R}, \qquad
\mathbf{Q}^{\top}\mathbf{Q} = \mathbf{I},\quad \mathbf{R} \text{ upper triangular}$$

**Why QR matters for regression.** Substitute $\mathbf{X} = \mathbf{Q}\mathbf{R}$ into the normal
equations:

$$\mathbf{X}^{\top}\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^{\top}\mathbf{y}
\;\Longrightarrow\;
\mathbf{R}^{\top}\underbrace{\mathbf{Q}^{\top}\mathbf{Q}}_{\mathbf{I}}\mathbf{R}\hat{\mathbf{w}}
= \mathbf{R}^{\top}\mathbf{Q}^{\top}\mathbf{y}
\;\Longrightarrow\;
\boxed{\mathbf{R}\hat{\mathbf{w}} = \mathbf{Q}^{\top}\mathbf{y}}$$

$\mathbf{R}$ is triangular, so this is solved by back-substitution in $O(d^2)$ — and crucially,
**$\mathbf{X}^{\top}\mathbf{X}$ was never formed**, so the condition number was never squared.
This is how regression is actually computed.

> **Numerical note.** Classical Gram-Schmidt as written above loses orthogonality badly in
> floating point. *Modified* Gram-Schmidt (subtract each projection immediately rather than all
> at once) is better; Householder reflections are better still and are what LAPACK uses.
> `from_scratch.py` implements both CGS and MGS so you can measure the difference yourself.

---

## 9. Determinant and trace

**Determinant** $\det(\mathbf{A})$ = the signed volume scaling factor of the map.

- $\det = 0$ ⟺ the map collapses space into a lower dimension ⟺ **not invertible** ⟺ nontrivial null space
- $\det < 0$ ⟺ orientation is flipped
- $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$ (volumes compose multiplicatively)
- $\det(\mathbf{A}) = \prod_i \lambda_i$

Determinants appear in ML in the multivariate Gaussian density (the $|\boldsymbol{\Sigma}|^{-1/2}$
normalizer is exactly the volume correction) and in normalizing flows (the change-of-variables
formula needs the Jacobian determinant — which is why flow architectures are designed to have
*triangular* Jacobians, whose determinant is just the product of the diagonal).

**Trace** $\operatorname{tr}(\mathbf{A}) = \sum_i a_{ii} = \sum_i \lambda_i$.

The property that earns its keep is **cyclic invariance**:

$$\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$$

plus the **trace trick** $\mathbf{x}^{\top}\mathbf{A}\mathbf{x} = \operatorname{tr}(\mathbf{A}\mathbf{x}\mathbf{x}^{\top})$,
which converts a scalar quadratic form into something you can push an expectation through:
$\mathbb{E}[\mathbf{x}^{\top}\mathbf{A}\mathbf{x}] = \operatorname{tr}(\mathbf{A}\,\mathbb{E}[\mathbf{x}\mathbf{x}^{\top}])$.
This shows up constantly in deriving expected losses and in Gaussian identities.

---

## 10. Eigenvalues and eigenvectors

### 10.1 Definition and meaning

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}, \qquad \mathbf{v} \neq \mathbf{0}$$

An eigenvector is a direction the matrix **does not rotate** — it only stretches it, by the
factor $\lambda$. Eigenvectors are the matrix's own natural axes. In that basis, the matrix's
complicated action becomes simple: just independent scaling.

Found from the characteristic equation $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ — though for
anything larger than $3 \times 3$ this is theoretical only; real algorithms are iterative (QR
algorithm, power iteration).

### 10.2 Eigendecomposition

If $\mathbf{A} \in \mathbb{R}^{n\times n}$ has $n$ linearly independent eigenvectors:

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$$

Read right to left, this says: *change into the eigenbasis, scale each axis, change back.*

The immediate payoff is powers:

$$\mathbf{A}^{k} = \mathbf{V}\boldsymbol{\Lambda}^{k}\mathbf{V}^{-1}$$

because all the interior $\mathbf{V}^{-1}\mathbf{V}$ pairs cancel. Raising a matrix to the 1000th
power costs one decomposition plus 1000 scalar powers.

> **This is the entire story of vanishing and exploding gradients.** Backpropagating through $T$
> steps of an RNN with recurrent matrix $\mathbf{W}$ multiplies the gradient by (roughly)
> $\mathbf{W}^{T}$, whose eigenvalues are $\lambda_i^{T}$. If $|\lambda_{\max}| < 1$ the gradient
> decays geometrically to nothing; if $|\lambda_{\max}| > 1$ it explodes. There is no stable
> middle ground for long $T$ — which is precisely why LSTMs introduce a path with multiplier
> ≈ 1 (the cell state), and why residual connections $\mathbf{x} + f(\mathbf{x})$ work: they make
> the Jacobian $\mathbf{I} + \mathbf{J}_f$, keeping eigenvalues near 1 by construction.

---

## 11. Symmetric matrices, quadratic forms, and positive definiteness

### 11.1 The spectral theorem

**Every real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors:**

$$\mathbf{A} = \mathbf{A}^{\top} \;\Longrightarrow\; \mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top},
\qquad \mathbf{Q}^{\top}\mathbf{Q} = \mathbf{I}$$

*Proof that eigenvectors of distinct eigenvalues are orthogonal.* Let $\mathbf{A}\mathbf{v}_1 =
\lambda_1\mathbf{v}_1$ and $\mathbf{A}\mathbf{v}_2 = \lambda_2\mathbf{v}_2$ with
$\lambda_1 \neq \lambda_2$. Then

$$\lambda_1 \mathbf{v}_2^{\top}\mathbf{v}_1 = \mathbf{v}_2^{\top}\mathbf{A}\mathbf{v}_1
= (\mathbf{A}\mathbf{v}_2)^{\top}\mathbf{v}_1 = \lambda_2\mathbf{v}_2^{\top}\mathbf{v}_1$$

using $\mathbf{A} = \mathbf{A}^\top$ in the middle step. So $(\lambda_1 - \lambda_2)
\mathbf{v}_2^{\top}\mathbf{v}_1 = 0$, and since $\lambda_1 \neq \lambda_2$ we get
$\mathbf{v}_2^{\top}\mathbf{v}_1 = 0$. $\blacksquare$

This matters enormously because **the matrices we care most about are symmetric**: covariance
matrices $\boldsymbol{\Sigma}$, Gram matrices $\mathbf{X}^{\top}\mathbf{X}$, kernel matrices
$\mathbf{K}$, and Hessians $\nabla^2 J$. For all of them we get a clean orthonormal
eigenbasis — a set of perpendicular natural axes — for free.

### 11.2 Quadratic forms and definiteness

$$q(\mathbf{x}) = \mathbf{x}^{\top}\mathbf{A}\mathbf{x} = \sum_{i}\sum_{j} a_{ij}x_ix_j$$

Substituting the spectral decomposition and setting $\mathbf{z} = \mathbf{Q}^{\top}\mathbf{x}$
(coordinates in the eigenbasis):

$$q(\mathbf{x}) = \mathbf{x}^{\top}\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}\mathbf{x}
= \mathbf{z}^{\top}\boldsymbol{\Lambda}\mathbf{z} = \sum_i \lambda_i z_i^2$$

**Every quadratic form is a weighted sum of squares in the eigenbasis.** The sign of $q$ is
therefore decided entirely by the signs of the eigenvalues:

| Type | Condition | Eigenvalues | Shape |
|---|---|---|---|
| Positive definite | $\mathbf{x}^{\top}\mathbf{A}\mathbf{x} > 0\ \forall\mathbf{x}\neq\mathbf{0}$ | all $\lambda_i > 0$ | bowl (unique minimum) |
| Positive semidefinite | $\ge 0$ | all $\lambda_i \ge 0$ | bowl with flat directions |
| Indefinite | both signs | mixed signs | **saddle** |
| Negative definite | $< 0$ | all $\lambda_i < 0$ | dome (unique maximum) |

**Why this is the most-used idea in optimization.** Near a critical point, a second-order Taylor
expansion gives $J(\boldsymbol{\theta} + \Delta) \approx J(\boldsymbol{\theta}) +
\tfrac{1}{2}\Delta^{\top}\mathbf{H}\Delta$. So the Hessian's eigenvalues classify the point:
all positive → local minimum; all negative → local maximum; mixed → saddle point.

In high dimensions, a random critical point having *all* $d$ eigenvalues of the same sign is
exponentially unlikely — which is the modern explanation for why deep networks are not plagued
by bad local minima but *are* slowed by saddle points. See
[07.09](../../07-deep-learning/09-training-dynamics/).

Two more consequences:

- **$\mathbf{A}^{\top}\mathbf{A}$ is always PSD**: $\mathbf{x}^{\top}\mathbf{A}^{\top}\mathbf{A}\mathbf{x}
  = \|\mathbf{A}\mathbf{x}\|_2^2 \ge 0$. Hence covariance and Gram matrices always have
  non-negative eigenvalues — variance can't be negative, and the geometry agrees.
- **Ridge regression, spectrally**: $\mathbf{X}^{\top}\mathbf{X} + \lambda\mathbf{I}$ has
  eigenvalues $\lambda_i + \lambda$. Adding $\lambda > 0$ lifts every eigenvalue away from zero,
  which is *simultaneously* why ridge always has a unique solution and why it improves
  conditioning. One line of linear algebra explains both properties.

---

## 12. The Singular Value Decomposition

The SVD is the most important matrix factorization in machine learning. Unlike
eigendecomposition, **it exists for every matrix** — square or not, singular or not.

$$\boxed{\;\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\;}$$

for $\mathbf{A}\in\mathbb{R}^{m\times n}$, where

- $\mathbf{U} \in \mathbb{R}^{m\times m}$ orthogonal — **left singular vectors**
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m\times n}$ diagonal, $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ — **singular values**
- $\mathbf{V} \in \mathbb{R}^{n\times n}$ orthogonal — **right singular vectors**

### 12.1 The geometric content

$\mathbf{A}\mathbf{x} = \mathbf{U}(\boldsymbol{\Sigma}(\mathbf{V}^{\top}\mathbf{x}))$ says:

$$\textbf{every linear map is a rotation, then an axis-aligned scaling, then another rotation.}$$

That's it. That's all a matrix can ever do. A unit sphere always maps to an ellipsoid, whose axis
directions are the columns of $\mathbf{U}$ and whose axis lengths are the $\sigma_i$.

### 12.2 Relation to eigendecomposition

$$\mathbf{A}^{\top}\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^{\top}\mathbf{U}^{\top}\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}
= \mathbf{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})\mathbf{V}^{\top}$$

$$\mathbf{A}\mathbf{A}^{\top} = \mathbf{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})\mathbf{U}^{\top}$$

So:

- $\mathbf{V}$ = eigenvectors of $\mathbf{A}^{\top}\mathbf{A}$
- $\mathbf{U}$ = eigenvectors of $\mathbf{A}\mathbf{A}^{\top}$
- $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^{\top}\mathbf{A})}$

**This proves the SVD exists** for any $\mathbf{A}$: $\mathbf{A}^{\top}\mathbf{A}$ is symmetric
PSD, so by the spectral theorem it has an orthonormal eigenbasis with non-negative eigenvalues,
and those give $\mathbf{V}$ and $\boldsymbol{\Sigma}$; $\mathbf{U}$ follows from
$\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ for $\sigma_i > 0$, extended to a full basis.

> **Do not compute the SVD this way.** Forming $\mathbf{A}^{\top}\mathbf{A}$ squares the
> condition number (§15). Golub-Kahan bidiagonalization works on $\mathbf{A}$ directly. Again:
> the identity is for understanding; the algorithm is different.

### 12.3 The SVD reveals everything about a matrix

| Quantity | From the SVD |
|---|---|
| $\operatorname{rank}(\mathbf{A})$ | number of $\sigma_i > 0$ |
| $C(\mathbf{A})$ | span of first $r$ columns of $\mathbf{U}$ |
| $N(\mathbf{A})$ | span of last $n - r$ columns of $\mathbf{V}$ |
| $C(\mathbf{A}^{\top})$ | span of first $r$ columns of $\mathbf{V}$ |
| $N(\mathbf{A}^{\top})$ | span of last $m - r$ columns of $\mathbf{U}$ |
| $\|\mathbf{A}\|_2$ | $\sigma_1$ |
| $\|\mathbf{A}\|_F$ | $\sqrt{\sum_i \sigma_i^2}$ |
| $\kappa(\mathbf{A})$ | $\sigma_1/\sigma_r$ |

All four fundamental subspaces, both norms, the rank, and the conditioning — from one
factorization. Nothing else in linear algebra is this productive.

### 12.4 SVD and PCA

Center the data ($\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{x}}^{\top}$). The sample covariance is

$$\mathbf{C} = \frac{1}{n-1}\mathbf{X}_c^{\top}\mathbf{X}_c$$

Take the SVD $\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$. Then

$$\mathbf{C} = \frac{1}{n-1}\mathbf{V}\boldsymbol{\Sigma}^{2}\mathbf{V}^{\top}$$

- Columns of $\mathbf{V}$ = **principal directions** (eigenvectors of the covariance)
- $\sigma_i^2/(n-1)$ = **variance explained** by component $i$
- $\mathbf{X}_c\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$ = **principal component scores**

**PCA is just the SVD of centered data.** The full three-derivation treatment (maximum variance,
minimum reconstruction error, and this one) is in
[04.06](../../04-unsupervised-learning/06-linear-dim-reduction/).

---

## 13. Low-rank approximation and the pseudoinverse

### 13.1 Eckart-Young-Mirsky

Write the SVD as a sum of rank-1 pieces:

$$\mathbf{A} = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^{\top}$$

Truncate to the top $k$ terms: $\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^{\top}$.

**Theorem (Eckart-Young-Mirsky).** $\mathbf{A}_k$ is the *best possible* rank-$k$ approximation
to $\mathbf{A}$, in both the spectral and Frobenius norms:

$$\min_{\operatorname{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A}-\mathbf{A}_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$$

This is a remarkable guarantee: a *greedy* truncation is *globally* optimal. It is why the SVD
underlies image compression, latent semantic analysis, matrix-factorization recommenders,
model compression, and the low-rank hypothesis behind LoRA.

The error tells you how many components to keep: plot the **cumulative explained variance**
$\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$ and stop where it plateaus.

### 13.2 The Moore-Penrose pseudoinverse

$$\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^{\top},
\qquad \boldsymbol{\Sigma}^{+} = \operatorname{diag}(1/\sigma_1, \dots, 1/\sigma_r, 0, \dots, 0)$$

Invert the nonzero singular values; leave the zeros alone. Then $\hat{\mathbf{x}} =
\mathbf{A}^{+}\mathbf{b}$ gives:

- the exact solution, if one exists and is unique;
- the least-squares solution, if the system is overdetermined;
- **the minimum-norm least-squares solution**, if it is underdetermined.

That last case is the important one. When $d > n$ (more features than examples) there are
infinitely many perfect fits; the pseudoinverse silently returns the one with smallest
$\|\mathbf{w}\|_2$ — an implicit ridge-like regularization, arising purely from the geometry.
This connects directly to why over-parameterized networks trained by gradient descent generalize:
GD from zero initialization converges to a minimum-norm solution too.

---

## 14. Matrix calculus

Denominator layout throughout (see [notation](../../docs/notation.md)): $\nabla_{\mathbf{x}} f$
has the same shape as $\mathbf{x}$.

### 14.1 The identities you need

| $f$ | $\nabla_{\mathbf{x}} f$ | Note |
|---|---|---|
| $\mathbf{a}^{\top}\mathbf{x}$ | $\mathbf{a}$ | linear |
| $\mathbf{x}^{\top}\mathbf{A}\mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^{\top})\mathbf{x}$ | $= 2\mathbf{A}\mathbf{x}$ if $\mathbf{A}$ symmetric |
| $\|\mathbf{x}\|_2^2$ | $2\mathbf{x}$ | special case, $\mathbf{A}=\mathbf{I}$ |
| $\|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2^2$ | $2\mathbf{A}^{\top}(\mathbf{A}\mathbf{x}-\mathbf{b})$ | **least squares** |
| $\log\det(\mathbf{X})$ | $\mathbf{X}^{-\top}$ | w.r.t. matrix $\mathbf{X}$ |
| $\operatorname{tr}(\mathbf{A}\mathbf{X})$ | $\mathbf{A}^{\top}$ | w.r.t. matrix $\mathbf{X}$ |

### 14.2 Derivation of the quadratic form gradient

Don't memorize — derive. Work in components:

$$q(\mathbf{x}) = \mathbf{x}^{\top}\mathbf{A}\mathbf{x} = \sum_{i}\sum_{j}a_{ij}x_ix_j$$

Differentiate w.r.t. $x_k$. The variable $x_k$ appears in terms where $i = k$, where $j = k$, and
once where both:

$$\frac{\partial q}{\partial x_k} = \sum_{j}a_{kj}x_j + \sum_{i}a_{ik}x_i
= (\mathbf{A}\mathbf{x})_k + (\mathbf{A}^{\top}\mathbf{x})_k$$

Stacking over $k$:

$$\nabla_{\mathbf{x}}\,\mathbf{x}^{\top}\mathbf{A}\mathbf{x} = (\mathbf{A}+\mathbf{A}^{\top})\mathbf{x}$$

For symmetric $\mathbf{A}$, this is $2\mathbf{A}\mathbf{x}$ — the exact matrix analogue of
$\frac{d}{dx}ax^2 = 2ax$, which is a good sanity check to keep.

### 14.3 The chain rule, in matrix form

For $\mathbf{y} = f(\mathbf{x})$ with $\mathbf{x}\in\mathbb{R}^{n}, \mathbf{y}\in\mathbb{R}^{m}$,
the Jacobian is $\mathbf{J} \in \mathbb{R}^{m\times n}$, $J_{ij} = \partial y_i/\partial x_j$.
For a composition $\mathbf{z} = g(f(\mathbf{x}))$:

$$\mathbf{J}_{\mathbf{z}/\mathbf{x}} = \mathbf{J}_{\mathbf{z}/\mathbf{y}}\,\mathbf{J}_{\mathbf{y}/\mathbf{x}}$$

Jacobians multiply, in reverse order of application. **This is backpropagation.** A neural
network is a composition $f_L \circ \dots \circ f_1$, and the gradient of a scalar loss is a
product of Jacobians accumulated right-to-left. Reverse-mode is chosen over forward-mode because
the output is a *scalar*: a $1\times n$ row vector times a matrix is far cheaper than propagating
full $n\times n$ Jacobians forward. Full derivation in
[07.02](../../07-deep-learning/02-backpropagation/).

**Shape-checking rule.** Before trusting any matrix derivative, verify the shapes compose. A
gradient w.r.t. $\mathbf{W} \in \mathbb{R}^{m\times n}$ must itself be $m \times n$. This one
check catches the overwhelming majority of matrix calculus errors.

---

## 15. Conditioning and numerical stability

### 15.1 The condition number

$$\kappa(\mathbf{A}) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

It bounds how much a relative input perturbation can be amplified in the output:

$$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A})\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$$

| $\kappa$ | Interpretation |
|---|---|
| $\approx 1$ | perfectly conditioned (orthogonal matrices have $\kappa = 1$) |
| $10^3$ | fine |
| $10^8$ | you have lost about half your double-precision digits |
| $>10^{16}$ | numerically singular in float64 |

**Rule of thumb**: with $\kappa \approx 10^{k}$, expect to lose about $k$ digits of accuracy.
Double precision starts with ~16.

### 15.2 Why $\mathbf{X}^{\top}\mathbf{X}$ is a trap

$$\kappa(\mathbf{X}^{\top}\mathbf{X}) = \kappa(\mathbf{X})^{2}$$

because the singular values get squared. A design matrix with a merely uncomfortable
$\kappa(\mathbf{X}) = 10^{8}$ produces a Gram matrix with $\kappa = 10^{16}$ — **numerically
singular**. The normal equations destroy your precision before you start.

This single identity is the reason:

- `LinearRegression` uses SVD, not the closed form
- QR is the standard for least squares
- feature scaling matters so much: features on wildly different scales inflate $\kappa(\mathbf{X})$ directly
- ridge helps numerically as well as statistically: $\kappa(\mathbf{X}^{\top}\mathbf{X}+\lambda\mathbf{I})
  = (\sigma_1^2+\lambda)/(\sigma_r^2+\lambda)$, which is strictly smaller than
  $\sigma_1^2/\sigma_r^2$ for $\lambda>0$

### 15.3 Practical rules

1. **Never** call `inv()`. To solve $\mathbf{A}\mathbf{x}=\mathbf{b}$, use `np.linalg.solve`
   (LU) or `np.linalg.lstsq` (SVD). Explicit inversion is both slower and less accurate.
2. **Standardize your features.** It is a conditioning fix, not just a convention.
3. **Check $\kappa$ when regression coefficients look insane.** Huge coefficients of opposite
   signs on correlated features is the signature of ill-conditioning.
4. **Use log-space for products of probabilities** — see
   [00.06 Numerical methods](../06-numerical-methods/) for log-sum-exp and stable softmax.

---

## 16. Where each concept shows up in ML

| Concept | Appears in |
|---|---|
| Dot product | every linear model, attention scores, cosine similarity, kernels |
| Matrix multiplication | every forward pass; the reason GPUs exist |
| Column space | which targets a linear model can represent |
| Null space | non-identifiability, why multicollinearity breaks coefficients |
| Rank | multicollinearity, LoRA, matrix-factorization recommenders |
| Projection | least squares, Gram-Schmidt, the hat matrix, residual diagnostics |
| $\ell_1$ / $\ell_2$ norms | Lasso / Ridge, weight decay, gradient clipping |
| Orthogonality | PCA components, QR, orthogonal initialization |
| Eigenvalues | PCA, spectral clustering, PageRank, vanishing/exploding gradients |
| PSD matrices | covariance, kernels (Mercer's condition), convexity of the Hessian |
| Quadratic forms | second-order optimization, Gaussian densities, Newton's method |
| **SVD** | PCA, LSA, pseudoinverse, low-rank compression, whitening, LoRA |
| Jacobians | backpropagation, normalizing flows, influence functions |
| Condition number | numerical stability, why we scale features, why ridge stabilizes |

---

## 17. Common misconceptions

**"$\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$ if the shapes work."**
No. Matrix multiplication is function composition, which does not commute. Both products can
exist and be different — or have different shapes entirely.

**"The inverse always exists."**
Only for square, full-rank matrices. And even then, you should not compute it — see §15.3.

**"Eigenvectors are always orthogonal."**
Only guaranteed for **symmetric** matrices (§11.1). A general matrix can have wildly non-orthogonal
eigenvectors, or too few of them to form a basis. Singular vectors, by contrast, are *always*
orthogonal — one of several reasons the SVD is more robust than eigendecomposition.

**"Eigendecomposition and SVD are the same thing."**
They coincide only for symmetric PSD matrices. Eigendecomposition needs a square matrix and may
not exist; the SVD always exists, for any shape.

**"More features can only help."**
Adding a linearly dependent feature adds zero information and makes $\mathbf{X}^{\top}\mathbf{X}$
singular. Adding a nearly-dependent one leaves it invertible but catastrophically ill-conditioned,
which is worse — you get a numerically-garbage answer instead of an error message.

**"$\mathbf{X}^{\top}\mathbf{X}$ being invertible means we're fine."**
Invertibility is binary; conditioning is continuous. $\kappa = 10^{15}$ is technically invertible
and practically useless.

**"Rank is computed by row reduction."**
Correct in exact arithmetic, meaningless in floating point — a "zero" pivot is never exactly zero.
Real rank determination counts singular values above a tolerance, which is what
`np.linalg.matrix_rank` does.

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | Gram-Schmidt (classical + modified), QR, power iteration, eigendecomposition, SVD via eigendecomposition, PCA, projections, pseudoinverse, condition-number experiments — all NumPy-primitive, all verified against LAPACK |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sections used |

**Next**: [00.02 — Calculus & Optimization](../02-calculus-and-optimization/) picks up where §14
stops and builds the full optimization toolkit.
