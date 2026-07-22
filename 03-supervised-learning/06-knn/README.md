# 03.06 — k-Nearest Neighbours

> **Prerequisites**: [00.01 §5](../../00-mathematical-foundations/01-linear-algebra/) (norms),
> [00.03](../../00-mathematical-foundations/03-probability/) (Bayes error).
> **You will be able to**: explain why *every* point is equidistant in high dimensions, predict
> when a KD-tree will help and when it degenerates to brute force, choose $k$ from the
> bias-variance tradeoff rather than folklore, and state the one theoretical guarantee KNN has.

---

## Table of contents

1. [The idea](#1-the-idea)
2. [The algorithm](#2-the-algorithm)
3. [Distance metrics](#3-distance-metrics)
4. [Scaling is not optional](#4-scaling-is-not-optional)
5. [Choosing k](#5-choosing-k)
6. [Weighted KNN](#6-weighted-knn)
7. [KNN for regression](#7-knn-for-regression)
8. [The curse of dimensionality](#8-the-curse-of-dimensionality)
9. [The one theoretical guarantee](#9-the-one-theoretical-guarantee)
10. [Making it fast](#10-making-it-fast)
11. [Complexity](#11-complexity)
12. [When to use it](#12-when-to-use-it)
13. [Common misconceptions](#13-common-misconceptions)

---

## 1. The idea

> **Similar inputs have similar outputs.** To predict for a new point, find the $k$ most similar
> training points and let them vote.

That is the entire model. There are no parameters, no optimization, no loss function. KNN is the
purest form of **instance-based** or **lazy** learning: it does no work at training time and all
of it at prediction time — the exact inverse of every other model in this part.

This makes it a useful reference point. It is the simplest thing that could possibly work, it is
**non-parametric** (its complexity grows with the data rather than being fixed in advance), and it
is universally consistent — given enough data it converges to the Bayes optimal classifier
([00.03 §2](../../00-mathematical-foundations/03-probability/)). The catch is what "enough data"
means in high dimensions, which is §8 and the real content of this chapter.

---

## 2. The algorithm

**Training**: store the data. That's it.

**Prediction** for a query $\mathbf{x}$:

1. Compute $d(\mathbf{x}, \mathbf{x}_i)$ for all $n$ training points
2. Take the $k$ smallest
3. **Classification**: majority vote. **Regression**: mean (or median)

$$\hat{y} = \frac{1}{k}\sum_{i\in N_k(\mathbf{x})}y_i
\qquad\text{or}\qquad
\hat{y} = \arg\max_c\sum_{i\in N_k(\mathbf{x})}\mathbb{1}[y_i = c]$$

Two details that matter:

- **Break ties deterministically.** With $k$ even and two classes, votes can tie. Use odd $k$ for
  binary problems, or break by nearest-neighbour distance.
- **The decision boundary is piecewise linear.** For $k=1$ it is exactly the boundary of the
  **Voronoi diagram** of the training points — each point owns the region closer to it than to any
  other. Larger $k$ smooths this.

---

## 3. Distance metrics

The metric *is* the model. Everything KNN believes about similarity lives here.

| Metric | Formula | Use for |
|---|---|---|
| **Euclidean** ($\ell_2$) | $\sqrt{\sum_j (x_j-z_j)^{2}}$ | continuous, comparable scales |
| **Manhattan** ($\ell_1$) | $\sum_j \lvert x_j-z_j\rvert$ | high dimensions; more robust to outliers |
| **Minkowski** ($\ell_p$) | $(\sum_j \lvert x_j-z_j\rvert^{p})^{1/p}$ | the general family |
| **Chebyshev** ($\ell_\infty$) | $\max_j \lvert x_j-z_j\rvert$ | when the worst coordinate matters |
| **Cosine** | $1 - \frac{\mathbf{x}^{\top}\mathbf{z}}{\Vert\mathbf{x}\Vert\Vert\mathbf{z}\Vert}$ | text, embeddings — magnitude irrelevant |
| **Hamming** | $\#\{j : x_j \ne z_j\}$ | categorical, binary strings |
| **Mahalanobis** | $\sqrt{(\mathbf{x}-\mathbf{z})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{z})}$ | correlated features — accounts for covariance |

**Mahalanobis deserves attention.** Euclidean distance treats all directions as equivalent;
Mahalanobis whitens by the covariance first
([00.03 §9.2](../../00-mathematical-foundations/03-probability/)), so it measures distance in
units of standard deviations along each principal direction. It is what you want when features are
correlated — and note that Euclidean distance on standardized data is exactly Mahalanobis with a
*diagonal* covariance, the same diagonal-vs-full choice as
[03.05 §9](../05-generative-classifiers/).

**Cosine vs Euclidean.** For $\ell_2$-normalized vectors they are monotonically related
($\Vert\mathbf{x}-\mathbf{z}\Vert^{2} = 2 - 2\mathbf{x}^{\top}\mathbf{z}$), so the neighbour
*ordering* is identical. They differ only when magnitudes vary — which is exactly when you would
choose cosine, e.g. documents of different lengths.

---

## 4. Scaling is not optional

Euclidean distance sums squared differences across features. A feature measured in dollars
(range $10^{5}$) and one in years (range $10^{1}$) do not contribute equally — the dollar feature
contributes $10^{8}$ times more to the squared distance. **KNN then ignores every other feature.**

$$d^{2}(\mathbf{x},\mathbf{z}) = \underbrace{(x_{\text{salary}}-z_{\text{salary}})^{2}}_{\sim 10^{8}} + \underbrace{(x_{\text{age}}-z_{\text{age}})^{2}}_{\sim 10^{2}}$$

Always standardize:

$$x_j \leftarrow \frac{x_j - \bar{x}_j}{s_j} \qquad\text{or}\qquad x_j\leftarrow\frac{x_j-\min_j}{\max_j-\min_j}$$

⚠️ **Fit the scaler on the training fold only** — computing means and standard deviations on the
full dataset leaks test information ([02.06](../../02-data/06-data-leakage/)). Use
`Pipeline(StandardScaler(), KNeighborsClassifier())`.

Note this is the same lesson as [03.02 §10](../02-regularized-linear-models/), for a different
reason: there the *penalty* was not scale-invariant; here the *distance* is not.

---

## 5. Choosing k

$k$ is the bias-variance dial, and it runs the opposite way to most hyperparameters:

| $k$ | Bias | Variance | Boundary | Behaviour |
|---|---|---|---|---|
| 1 | lowest | highest | maximally jagged | fits every point, including noise |
| moderate | balanced | balanced | smooth | usually best |
| $n$ | highest | lowest | none | predicts the global majority always |

> **Small $k$ = complex model.** This trips people up, because for most hyperparameters a larger
> value means more capacity. Here the *effective number of parameters* is roughly $n/k$: with
> $k=1$ you have $n$ effective parameters, with $k=n$ you have one. That is why $1$-NN has zero
> training error and terrible test error — the textbook signature of overfitting.

Choose $k$ by cross-validation ([05.04](../../05-model-evaluation/04-cross-validation/)). The
common $k=\sqrt{n}$ heuristic is a starting point, not an answer.

⚠️ **Training error is meaningless for KNN.** With $k=1$, every training point is its own nearest
neighbour, so training accuracy is exactly 100% regardless of how bad the model is. Never tune $k$
on training error — it will always pick $k=1$.

---

## 6. Weighted KNN

Plain KNN gives the 1st and $k$-th neighbours equal votes even when one is ten times closer.
Weighting by inverse distance fixes it:

$$w_i = \frac{1}{d(\mathbf{x},\mathbf{x}_i) + \epsilon}
\qquad\Longrightarrow\qquad
\hat{y} = \frac{\sum_i w_iy_i}{\sum_i w_i}$$

The $\epsilon$ guards against division by zero when the query coincides with a training point.

Weighting makes the model **less sensitive to $k$** — distant neighbours contribute little, so
adding more of them changes the prediction less. If you are unsure about $k$, weighting is a cheap
hedge.

The limiting case of distance weighting over *all* points is **kernel regression**
(Nadaraya-Watson), $\hat{y} = \sum_i K_h(\mathbf{x},\mathbf{x}_i)y_i / \sum_i K_h(\mathbf{x},\mathbf{x}_i)$
— KNN with a hard cutoff replaced by a smooth one.

---

## 7. KNN for regression

Average the neighbours instead of voting. Two properties worth knowing:

**It cannot extrapolate.** Predictions are always averages of observed $y$ values, so
$\hat{y}$ is bounded by $[\min y_i, \max y_i]$ — always. If the true relationship keeps rising
beyond your data, KNN flattens out. Compare
[03.03 §7](../03-basis-expansion/), where at least the natural spline extended the trend
linearly.

**The fit is piecewise constant.** Predictions jump as the neighbour set changes, giving a step
function. Distance weighting smooths this considerably, which is a second reason to prefer it for
regression.

---

## 8. The curse of dimensionality

This is why KNN, which is theoretically excellent, fails in practice on high-dimensional data.
Three distinct problems, all consequences of geometry.

### 8.1 Neighbourhoods stop being local

To capture a fraction $r$ of the data in a $d$-dimensional unit hypercube, you need a
sub-cube of edge length

$$e_d(r) = r^{1/d}$$

| $d$ | Edge length to capture 1% | To capture 10% |
|---|---|---|
| 1 | 0.010 | 0.100 |
| 2 | 0.100 | 0.316 |
| 10 | 0.631 | 0.794 |
| 50 | 0.912 | 0.955 |
| 100 | 0.955 | 0.977 |

At $d=100$, capturing your nearest **1%** of neighbours requires a box spanning **95.5% of the
range of every feature**. Those points are not "nearby" in any meaningful sense — the word
"neighbourhood" has stopped meaning anything.

### 8.2 Distances concentrate

The deeper problem. For i.i.d. features, as $d\to\infty$:

$$\frac{d_{\max}(\mathbf{x}) - d_{\min}(\mathbf{x})}{d_{\min}(\mathbf{x})} \longrightarrow 0$$

**The nearest and farthest points become equidistant.** Distance stops discriminating at all, so
"nearest neighbour" becomes arbitrary — determined by noise rather than by similarity.

The reason is a variance argument: $d^{2}(\mathbf{x},\mathbf{z}) = \sum_j (x_j-z_j)^{2}$ is a sum
of $d$ i.i.d. terms. Its mean grows as $d$ while its standard deviation grows only as $\sqrt{d}$,
so the *relative* spread shrinks as $1/\sqrt{d}$. Every pairwise distance converges to the same
value.

Experiment 1 measures both effects directly. The practical threshold arrives sooner than people
expect: by $d\approx 20$ the contrast has already degraded substantially.

### 8.3 Everything is on the boundary

The volume of a $d$-ball inscribed in a unit cube, relative to the cube, is

$$\frac{V_{\text{ball}}}{V_{\text{cube}}} = \frac{\pi^{d/2}}{2^{d}\,\Gamma(d/2+1)} \longrightarrow 0$$

At $d=10$ it is 0.0025; at $d=20$, $2.5\times10^{-8}$. **Almost all the volume of a high-dimensional
cube is in its corners**, so almost every point is near a boundary and few points have neighbours
"all around" them. Every prediction becomes an extrapolation.

### 8.4 What actually saves KNN

The curse is about *intrinsic* dimension, not the number of columns. Real data often lies near a
low-dimensional manifold — images of faces have thousands of pixels but far fewer degrees of
freedom. KNN works on such data despite a large nominal $d$.

The practical responses:

| Response | Why it helps |
|---|---|
| Dimensionality reduction (PCA, UMAP) | reduces $d$ toward the intrinsic dimension |
| **Learned embeddings** | a network maps inputs to a space where $\ell_2$ *means* similarity |
| Feature selection | removes irrelevant dimensions, which are pure noise in the distance |
| Metric learning | learns $\boldsymbol{\Sigma}$ for a Mahalanobis distance from labels |

> **This is why modern nearest-neighbour search works so well.** Vector databases and RAG
> ([11.08](../../11-transformers-and-llms/08-rag-and-agents/)) run KNN in 768- or 1536-dimensional
> embedding spaces and it is enormously effective — because those embeddings were *trained* so
> that distance encodes semantic similarity. KNN on raw high-dimensional features fails; KNN on a
> learned representation is state of the art. The algorithm did not change; the metric did.

---

## 9. The one theoretical guarantee

**Cover & Hart (1967).** As $n\to\infty$, the 1-nearest-neighbour error rate $R_{1NN}$ satisfies

$$R^{*} \le R_{1NN} \le 2R^{*}(1-R^{*}) \le 2R^{*}$$

where $R^{*}$ is the Bayes error ([00.03](../../00-mathematical-foundations/03-probability/)).

**1-NN is asymptotically at most twice as bad as the optimal classifier** — using no model, no
training, and no assumptions. That is a remarkable result and worth knowing: it says the
information needed to classify is largely contained in the nearest example.

Two refinements:

- For $k\to\infty$ with $k/n\to 0$, KNN is **universally consistent**: $R_{kNN}\to R^{*}$. It
  reaches the optimum exactly.
- The bound is tight at both ends: $R^{*}=0$ gives $R_{1NN}=0$, and small $R^{*}$ gives
  $R_{1NN}\approx 2R^{*}$.

⚠️ The theorem is **asymptotic**, and the rate is exponentially slow in $d$ — which is §8 restated.
"Enough data" for $d=100$ exceeds the number of atoms available.

---

## 10. Making it fast

Brute-force prediction is $O(nd)$ per query. Three families of acceleration:

### 10.1 Space-partitioning trees

**KD-tree**: recursively split on the median of one axis at a time. Query prunes branches whose
bounding box is farther than the current $k$-th best.

**Ball tree**: partition into nested hyperspheres. Better than KD-trees when $d$ is moderate or
the metric is non-Euclidean.

> **Both degenerate to brute force as $d$ grows** — and §8.2 explains exactly why. Pruning works by
> proving a whole region is farther than the current best; when all distances concentrate, no
> region can be excluded and the tree visits everything. **The rule of thumb is $d \lesssim 20$**,
> which is not a tuning issue but a consequence of distance concentration. Experiment 4 measures
> the crossover. sklearn's `algorithm="auto"` silently switches to brute force in high dimensions
> for this reason.

### 10.2 Approximate nearest neighbours

Give up exactness for speed:

| Method | Idea |
|---|---|
| **LSH** | hash so that nearby points collide with high probability |
| **HNSW** | a navigable small-world graph; greedy descent through hierarchical layers |
| **IVF-PQ** | cluster, then compress vectors with product quantization |
| **Annoy** | a forest of random projection trees |

**HNSW is the current default** for vector search — it scales to billions of vectors with ~99%
recall in milliseconds, and it is what FAISS, Qdrant, Milvus, and pgvector are built on.

### 10.3 Reduce the data

**Condensing** (keep only points near the boundary) and **editing** (remove misclassified points)
shrink the stored set without much accuracy loss. Rarely used now that ANN search is cheap.

---

## 11. Complexity

| | Time | Memory |
|---|---|---|
| Train (brute) | $O(1)$ | $O(nd)$ — stores everything |
| Train (KD-tree) | $O(dn\log n)$ | $O(nd)$ |
| Query (brute) | $O(nd)$ | $O(1)$ |
| Query (KD-tree, low $d$) | $O(d\log n)$ | $O(1)$ |
| Query (KD-tree, high $d$) | $O(nd)$ — degenerate | $O(1)$ |
| Query (HNSW) | $O(\log n)$ approximate | $O(nd)$ + graph |

**The memory cost is the real constraint.** KNN stores the entire training set forever. A
million-row, 100-feature dataset in float32 is 400 MB that must be resident at inference time —
against a logistic regression's 400 bytes.

---

## 12. When to use it

**Use it when:**
- The decision boundary is genuinely irregular and you have plenty of data
- $d$ is low, or you have a **learned embedding** where distance means something (§8.4)
- You need a baseline in five minutes
- You need **retrieval**, not just classification — "show me similar cases" is KNN's native
  question and is often more valuable than a label
- New classes appear constantly (recommendation, face recognition) — adding a class means adding
  points, with no retraining

**Don't when:**
- $d$ is high and the features are raw (§8)
- Inference must be fast or memory is constrained (§11)
- You need interpretable coefficients or feature importances
- The data is noisy — KNN has no averaging mechanism beyond $k$

> **KNN's second life.** As a *classifier* it has largely been superseded. As a **retrieval
> engine over learned embeddings** it is more central to production ML than ever: semantic search,
> RAG, recommendation candidate generation, deduplication, and few-shot classification are all
> nearest-neighbour lookups. Learning the chapter is worth it for that reason alone.

---

## 13. Common misconceptions

**"KNN has no training phase, so it's fast."**
Training is free; *prediction* is expensive, and the model is the entire dataset (§11).

**"Larger $k$ means a more complex model."**
Backwards. Small $k$ = complex, large $k$ = simple (§5).

**"$k=1$ overfits because it's sensitive to noise."**
True, and it is worse than that: its training error is *identically zero* by construction, so
training error cannot be used to select $k$ at all (§5).

**"KNN is non-parametric, so it makes no assumptions."**
It assumes that your distance metric encodes similarity. That is a strong assumption, and §4 and
§8 are both about it being violated.

**"Scaling is good practice."**
It is mandatory. Without it, the largest-range feature is the only one that matters (§4).

**"KD-trees make KNN fast."**
Below $d\approx 20$. Above it, they are slower than brute force because of the traversal overhead
(§10.1).

**"KNN doesn't work in high dimensions."**
Not on *raw* high-dimensional features. On learned embeddings of the same nominal dimension it
works extremely well (§8.4).

**"1-NN is a weak baseline."**
Asymptotically it is within a factor of 2 of the Bayes optimum (§9).

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | KNN classifier and regressor, seven distance metrics, uniform and distance weighting, a KD-tree with proper pruning, and Mahalanobis distance — verified against sklearn, with experiments measuring distance concentration, the $k$ tradeoff, scaling, KD-tree degeneration, and the Cover-Hart bound |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sections used |

**Previous**: [03.05 — Generative Classifiers](../05-generative-classifiers/) ·
**Next**: [03.07 — Support Vector Machines](../07-svm/)
