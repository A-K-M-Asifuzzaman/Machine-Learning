# 03.06 — Exercises: k-Nearest Neighbours

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that for $k=1$ the decision boundary is exactly the boundary of the Voronoi diagram
of the training points.

**D2.** Prove that $k$-NN's training accuracy is identically 1 when $k=1$ (with distinct points),
and explain the consequence for hyperparameter selection.

**D3.** Show that the effective number of parameters of $k$-NN is roughly $n/k$. Use this to
explain why *small* $k$ means a *complex* model.

**D4.** Show that for $\ell_2$-normalized vectors, cosine distance and Euclidean distance induce
the same neighbour ordering: $\Vert\mathbf{x}-\mathbf{z}\Vert^{2} = 2-2\mathbf{x}^{\top}\mathbf{z}$.
When do they differ?

**D5.** Show that Mahalanobis distance equals Euclidean distance after whitening by
$\boldsymbol{\Sigma}^{-1/2}$. What does this say about the relationship between standardizing
features and using Mahalanobis distance?

**D6.** Derive $e_d(r)=r^{1/d}$, the edge length of a sub-cube capturing a fraction $r$ of a unit
hypercube. Evaluate at $d=100$, $r=0.01$ and interpret.

**D7.** *(Distance concentration.)* For $\mathbf{x},\mathbf{z}$ with i.i.d. coordinates, show
$\mathbb{E}[d^{2}]$ grows as $O(d)$ while $\mathrm{sd}(d^{2})$ grows as $O(\sqrt{d})$. Conclude
that the relative spread shrinks as $O(1/\sqrt{d})$.

**D8.** Derive the ratio of the volume of the inscribed $d$-ball to the unit cube, and show it
tends to 0. What does this imply about where the points are?

**D9.** State the Cover-Hart theorem. Show the bound $2R^{*}(1-R^{*})$ is tightest at $R^{*}=0$
and loosest as $R^{*}\to 0.5$.

**D10.** Explain why $k$-NN with $k\to\infty$ and $k/n\to 0$ is universally consistent, but
$k$ fixed is not.

**D11.** Prove that $k$-NN regression cannot extrapolate: show $\hat{y}\in[\min y_i,\max y_i]$
always.

**D12.** Explain, in terms of the pruning test, why a KD-tree degenerates to brute force as $d$
grows. Connect it explicitly to D7.

**D13.** Show that inverse-distance weighting makes predictions less sensitive to $k$ than uniform
weighting.

**D14.** Show that the limit of distance-weighted KNN as $k\to n$ with a kernel weight is the
Nadaraya-Watson estimator.

---

## Tier 2 — Implementation

**I1.** Implement `pairwise_distances` for Euclidean using the $\Vert x\Vert^2 - 2x^\top z + \Vert z\Vert^2$
expansion. Verify against scipy, and find an input where the un-clipped version returns `nan`.

**I2.** Implement the classifier and regressor with uniform and distance weighting. Verify
`predict_proba` against sklearn to $10^{-10}$ for several $k$.

**I3.** Implement a KD-tree with correct pruning. Verify it returns *exactly* the brute-force
neighbours, then instrument it to count visited nodes.

**I4.** Reproduce Experiment 4. Plot the fraction of the tree visited against $d$ and find where
it reaches 100%.

**I5.** Reproduce Experiment 1(b). Confirm the $1/\sqrt{d}$ law by plotting $\mathrm{sd}/\mathrm{mean}$
against $1/\sqrt{d}$ and checking for a straight line.

**I6.** Reproduce Experiment 3. Then repeat with Manhattan distance — is it more or less sensitive
to the unscaled feature, and why?

**I7.** Reproduce Experiment 2. Add 5-fold cross-validation and confirm CV selects a $k$ near the
test optimum, while training error selects $k=1$.

**I8.** Implement Mahalanobis KNN. On data with strongly correlated features, compare it against
Euclidean on standardized features and explain the difference.

**I9.** *(The escape.)* Take a high-dimensional dataset where raw KNN fails. Apply PCA, then UMAP,
then (if available) a pretrained embedding. Measure KNN accuracy in each space and relate the
result to intrinsic dimension.

**I10.** Implement locality-sensitive hashing for cosine similarity (random hyperplane hashing).
Measure recall against exact search as a function of the number of hash bits.

**I11.** Compare `faiss` or `hnswlib` against exact search on 100k vectors: measure recall@10 and
query time. At what recall does the speedup become worthwhile?

**I12.** *(Retrieval, not classification.)* Build a nearest-neighbour index over sentence
embeddings and use it for semantic search. This is KNN's most important modern use
([11.08](../../11-transformers-and-llms/08-rag-and-agents/)).

**I13.** Implement condensed nearest neighbour (keep only points needed to classify the rest
correctly). Measure the reduction in stored points and the accuracy cost.

---

## Tier 3 — Interview

**Q1.** How does KNN work? What happens at training time?

**Q2.** Does a larger $k$ make the model more or less complex? Explain.

**Q3.** Why can't you choose $k$ by training error?

**Q4.** Why must you scale features for KNN?

**Q5.** What is the curse of dimensionality? Give three distinct consequences.

**Q6.** Why do all points become equidistant in high dimensions?

**Q7.** Why do KD-trees stop helping above ~20 dimensions?

**Q8.** How good is 1-NN, theoretically?

**Q9.** Your KNN model works on 5 features and fails on 500. Diagnose and propose fixes.

**Q10.** Why does KNN work so well over embeddings but not over raw high-dimensional features?

**Q11.** Can KNN regression predict a value outside the training range?

**Q12.** What is the memory cost of a deployed KNN model?

**Q13.** When would you use cosine distance over Euclidean?

**Q14.** What is approximate nearest neighbour search, and when is the approximation acceptable?

**Q15.** Is KNN still relevant in production ML?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain why small $k$ is the complex model
- [ ] State three separate consequences of high dimensionality and derive at least one
- [ ] Predict whether a KD-tree will help, from $d$ alone
- [ ] Explain the Cover-Hart guarantee *and* why it does not save you in practice
- [ ] Say precisely why KNN over embeddings succeeds where KNN over raw features fails
- [ ] Recognize that KNN's modern importance is retrieval, not classification
