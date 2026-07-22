# 00.06 — References: Numerical Methods

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-3 | IEEE 754, machine epsilon | Goldberg (1991); IEEE 754-2019 standard |
| §4 | Catastrophic cancellation | Higham, *Accuracy and Stability*, Ch. 1-2 |
| §5-6 | Overflow, underflow, log space | Murphy, *PML*, §2.5.4; Bishop, *PRML*, §4.3 |
| §7-8 | Log-sum-exp, stable softmax | Blanchard, Higham & Higham (2021) |
| §9 | Stable sigmoid and BCE | PyTorch `BCEWithLogitsLoss` source; Bishop §4.3.2 |
| §10 | Welford's algorithm | Welford (1962); Chan, Golub & LeVeque (1983); Knuth, *TAOCP* Vol. 2 §4.2.2 |
| §11 | Kahan summation | Kahan (1965); Higham (1993) |
| §12 | Conditioning | Trefethen & Bau, *Numerical Linear Algebra*, Lectures 12-15 |
| §13 | Gradient checking | Nocedal & Wright, *Numerical Optimization*, §8.1; CS231n notes |
| §14 | Mixed precision, bfloat16 | Micikevicius et al. (2018); Kalamkar et al. (2019) |
| §15 | Determinism | PyTorch reproducibility docs; Nagarajan et al. (2019) |

---

## Books

**Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.**
The definitive reference for everything in this chapter. Chapters 1-4 cover floating point,
cancellation, and summation error; Chapter 4 specifically analyses summation methods including
Kahan. Dense but authoritative — the book to check when you need a *bound*, not a rule of thumb.

**Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.**
Lectures 12-15 on conditioning and stability are the source for §12, and the clearest exposition
of "backward stable" that exists. Short.

**Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*,
3rd ed. Addison-Wesley.**
§4.2 is the classic treatment of floating-point arithmetic. §4.2.2 contains Welford's algorithm,
which is where most people first meet it.

**Overton, M. L. (2001). *Numerical Computing with IEEE Floating Point Arithmetic*. SIAM.**
A short, focused book on the standard itself. Good if §2 left you wanting the bit-level detail.

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>
§4.1 ("Overflow and Underflow") is the deep-learning-specific version of §5-§8 here, and uses
softmax as its worked example.

---

## Papers and articles

### Floating point
- **Goldberg, D. (1991).** "What Every Computer Scientist Should Know About Floating-Point
  Arithmetic." *ACM Computing Surveys* 23(1), 5-48. — free and widely mirrored. **The single most
  useful thing to read after this chapter.** Everything in §2-§4 is developed properly there.
- **Kahan, W. (1965).** "Further remarks on reducing truncation errors." *Communications of the
  ACM* 8(1), 40. — compensated summation, in one page.
- **Higham, N. J. (1993).** "The accuracy of floating point summation." *SIAM J. Scientific
  Computing* 14(4), 783-799. — the comparison of naive, pairwise, and compensated summation that
  Experiment 5 reproduces.

### Stable computation of ML primitives
- **Blanchard, P., Higham, D. J. & Higham, N. J. (2021).** "Accurately computing the log-sum-exp
  and softmax functions." *IMA Journal of Numerical Analysis* 41(4), 2311-2330. — the rigorous
  error analysis of the shift trick in §7-§8, including the low-precision case. Directly relevant
  to why the naive version fails where it does.
- **Welford, B. P. (1962).** "Note on a method for calculating corrected sums of squares and
  products." *Technometrics* 4(3), 419-420. — two pages.
- **Chan, T. F., Golub, G. H. & LeVeque, R. J. (1983).** "Algorithms for computing the sample
  variance: Analysis and recommendations." *The American Statistician* 37(3), 242-247. — the
  systematic comparison behind §10's table, including the parallel/mergeable variant.

### Low precision and mixed-precision training
- **Micikevicius, P. et al. (2018).** "Mixed Precision Training." *ICLR*.
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) — the paper that introduced loss scaling and
  the float32 master-weight copy. §14 follows it.
- **Kalamkar, D. et al. (2019).** "A Study of BFLOAT16 for Deep Learning Training."
  [arXiv:1905.12322](https://arxiv.org/abs/1905.12322) — why the range/precision trade favours
  bfloat16.
- **Wang, N. et al. (2018).** "Training Deep Neural Networks with 8-bit Floating Point Numbers."
  *NeurIPS*.
- **Dettmers, T. et al. (2022).** "LLM.int8(): 8-bit Matrix Multiplication for Transformers at
  Scale." [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — where quantization takes over from
  floating point.

### Reproducibility
- **Nagarajan, P. et al. (2019).** "Deterministic Implementations for Reproducibility in Deep
  Reinforcement Learning." [arXiv:1809.05676](https://arxiv.org/abs/1809.05676)
- **PyTorch Reproducibility docs** — <https://pytorch.org/docs/stable/notes/randomness.html> — the
  authoritative list of what must be set, and what it costs.

---

## Practical references

| Resource | Why |
|---|---|
| [Float Exposed](https://float.exposed/) | interactive bit-level float inspector — set a bit, see the value |
| [IEEE-754 Floating Point Converter](https://www.h-schmidt.net/FloatConverter/IEEE754.html) | the same, for float32 |
| [`0.30000000000000004.com`](https://0.30000000000000004.com/) | the `0.1 + 0.2` problem across 50+ languages |
| [CS231n: Gradient checks](https://cs231n.github.io/neural-networks-3/#gradcheck) | the practical gradient-checking guide §13 follows |
| [NumPy floating point docs](https://numpy.org/doc/stable/user/basics.types.html) | dtype ranges and `finfo` |

---

## Reference implementations

Read these; they are short and they are exactly the material of this chapter in production form:

| Source | What to look at |
|---|---|
| [`scipy.special.logsumexp`](https://github.com/scipy/scipy/blob/main/scipy/special/_logsumexp.py) | the shift trick, plus careful handling of `-inf` and weights |
| [`torch.nn.functional.binary_cross_entropy_with_logits`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp) | the C++ kernel implementing §9's formula |
| [`torch.nn.CrossEntropyLoss`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp) | note the fusion of `log_softmax` and `nll_loss` |
| [`torch.amp`](https://github.com/pytorch/pytorch/tree/main/torch/amp) | `GradScaler` — loss scaling and inf-checking from §14 |
| [`numpy.add.reduce`](https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/loops_utils.h.src) | pairwise summation, why `np.sum` beats a Python loop for free |
| [`sklearn.utils.extmath`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py) | `_incremental_mean_and_var` — Welford in production, used by `StandardScaler` for partial fits |
| [`torch.nn.BatchNorm`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp) | how running statistics are accumulated, and in what precision |

---

## Deferred to later chapters

- **Conditioning of the design matrix, QR vs normal equations** → [00.01 §15](../01-linear-algebra/) (already covered)
- **Learning-rate stability threshold and divergence** → [00.02 §7.1](../02-calculus-and-optimization/) (already covered)
- **Feature scaling as a numerical intervention** → [02.04](../../02-data/04-scaling-and-transformation/)
- **Gradient clipping, normalization layers** → [07.07](../../07-deep-learning/07-normalization/), [07.09](../../07-deep-learning/09-training-dynamics/)
- **Quantization, pruning, distillation, ONNX** → [19.04](../../19-mlops/04-efficiency/)
- **Distributed training and reduction order** → [19.04](../../19-mlops/04-efficiency/)
