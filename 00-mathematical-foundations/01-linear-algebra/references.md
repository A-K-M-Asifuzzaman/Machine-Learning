# 00.01 — References: Linear Algebra

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-4 | Vector spaces, matrices, four subspaces | Strang, *Linear Algebra and Learning from Data*, Ch. 1; Deisenroth et al., *MML*, Ch. 2 |
| §4.2 | Fundamental Theorem of Linear Algebra | Strang, *Introduction to Linear Algebra*, §3.5 |
| §5 | Norms and inner products | Deisenroth et al., *MML*, Ch. 3; Boyd & Vandenberghe, *Convex Optimization*, App. A |
| §6-7 | Projection and least squares | Strang, *Introduction to Linear Algebra*, Ch. 4; Hastie et al., *ESL*, §3.2 |
| §7 | Hat matrix, leverage, degrees of freedom | Hastie et al., *ESL*, §3.2, §7.6 |
| §8 | Gram-Schmidt, QR, stability | Trefethen & Bau, *Numerical Linear Algebra*, Lectures 7-10, 16 |
| §9 | Determinant, trace, trace trick | Petersen & Pedersen, *The Matrix Cookbook*, §1-2 |
| §10-11 | Eigendecomposition, spectral theorem | Deisenroth et al., *MML*, Ch. 4; Strang, *ILA*, Ch. 6 |
| §11.2 | Quadratic forms, definiteness, saddle points | Boyd & Vandenberghe, §A.5; Goodfellow et al., *Deep Learning*, §4.3 |
| §12 | SVD | Trefethen & Bau, Lectures 4-5; Strang, *LALFD*, Ch. 1.8-1.9 |
| §12.4 | PCA via SVD | Bishop, *PRML*, §12.1; Hastie et al., *ESL*, §14.5 |
| §13.1 | Eckart-Young-Mirsky | Trefethen & Bau, Lecture 5, Theorem 5.8 |
| §13.2 | Moore-Penrose pseudoinverse | Golub & Van Loan, *Matrix Computations*, §5.5 |
| §14 | Matrix calculus | Petersen & Pedersen, *The Matrix Cookbook*, §2; Deisenroth et al., *MML*, Ch. 5 |
| §15 | Conditioning, stability | Trefethen & Bau, Lectures 12-15; Higham, *Accuracy and Stability of Numerical Algorithms*, Ch. 1-3 |

---

## Books

**Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.**
The best book on this material, and short. If you read one thing after this chapter, read
Lectures 4-5 (SVD) and 12-15 (conditioning and stability). It is where §8, §12, and §15 come from.

**Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge.**
Strang writes linear algebra as it is actually used in machine learning. The four-subspaces
framing of §4 is his.
Free lectures: [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) and
[MIT 18.065](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/).

**Deisenroth, M. P., Faisal, A. A. & Ong, C. S. (2020). *Mathematics for Machine Learning*.
Cambridge.** — free at <https://mml-book.github.io/>
Exactly the math ML needs. Chapters 2-5 cover this chapter's material with ML motivation attached.
The ideal companion volume.

**Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. JHU Press.**
The reference work. Consult for algorithm details — §5.5 (pseudoinverse), §8.4 (Jacobi
eigenvalue method, used in `from_scratch.py`), §5.2 (Householder QR).

**Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.**
The definitive treatment of floating-point error analysis. The backing for every claim in §15.

**Petersen, K. B. & Pedersen, M. S. (2012). *The Matrix Cookbook*.** — free at
<https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html>
A lookup table of matrix identities. Not for reading; for keeping open during derivations.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*,
2nd ed. Springer.** — free at <https://hastie.su.domains/ElemStatLearn/>
§3.2 for least squares, §3.4.1 for the ridge SVD decomposition used in exercise D9.

**Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge.** — free at
<https://web.stanford.edu/~boyd/cvxbook/>
Appendix A for the linear algebra background to convexity; §A.5 for quadratic forms.

---

## Papers

**Eckart, C. & Young, G. (1936).** "The approximation of one matrix by another of lower rank."
*Psychometrika* 1(3), 211-218. — the original low-rank approximation theorem (§13.1).

**Mirsky, L. (1960).** "Symmetric gauge functions and unitarily invariant norms."
*Quarterly Journal of Mathematics* 11(1), 50-59. — the generalization to all unitarily invariant
norms.

**Golub, G. & Kahan, W. (1965).** "Calculating the singular values and pseudo-inverse of a
matrix." *SIAM J. Numerical Analysis* 2(2), 205-224. — the algorithm LAPACK actually uses, and
the reason §12.2's construction is a derivation rather than an implementation.

**Hu, E. J. et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models."
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — the rank argument of §4.1 applied to
LLM fine-tuning; the basis of exercise I9.

**Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011).** "Finding structure with randomness:
probabilistic algorithms for constructing approximate matrix decompositions." *SIAM Review*
53(2), 217-288. — randomized SVD, how truncated SVD is computed at scale.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [LAPACK](https://www.netlib.org/lapack/) | `dgeqrf` (Householder QR), `dgesdd` (divide-and-conquer SVD), `dsyev` (symmetric eigen) |
| [`numpy.linalg`](https://github.com/numpy/numpy/blob/main/numpy/linalg/_linalg.py) | the thin Python layer over LAPACK — worth reading to see which routine each call maps to |
| [`scipy.linalg`](https://docs.scipy.org/doc/scipy/reference/linalg.html) | more control than NumPy: `lstsq` with driver choice, `qr` with pivoting |
| [scikit-learn `_base.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_base.py) | `LinearRegression.fit` — see it dispatch to `scipy.linalg.lstsq`, not a closed-form inverse (§15.3) |

---

## Visual and interactive

- **3Blue1Brown, *Essence of Linear Algebra*** — <https://www.3blue1brown.com/topics/linear-algebra>
  Watch this first if the geometry of §3 and §10 does not click. Fifteen short videos; the best
  visual intuition for linear maps and eigenvectors that exists.
- **Immersive Linear Algebra** — <http://immersivemath.com/ila/> — a fully interactive textbook.
- **Strang's *A 2020 Vision of Linear Algebra*** — <https://ocw.mit.edu/resources/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/>
  Six short lectures reorganizing the subject around matrix factorizations.

---

## A note on what this chapter deliberately omits

Left out because they are not load-bearing for machine learning: general vector spaces over
arbitrary fields, Jordan normal form, abstract dual spaces, and most of determinant theory
(cofactor expansion, Cramer's rule — correct but computationally useless).

Deferred to later chapters rather than omitted:

- **Kernel methods and RKHS** → [03.07 SVM](../../03-supervised-learning/07-svm/)
- **Matrix calculus for backprop** → [07.02 Backpropagation](../../07-deep-learning/02-backpropagation/)
- **Spectral graph theory, graph Laplacians** → [04.05 Spectral clustering](../../04-unsupervised-learning/05-spectral-clustering/) and [14.01](../../14-graph-ml/01-graph-fundamentals/)
- **Tensor operations, einsum** → [01.01 NumPy](../../01-python-for-ml/01-numpy/)
