# 07.02 — References: Backpropagation

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§4 | The algorithm, the four equations | Rumelhart et al. (1986); Nielsen (2015) Ch. 2 |
| §5 | Reverse-mode autodiff | Baydin et al. (2018); Griewank & Walther (2008) |
| §6 | Gradient checking | Goodfellow et al. §6.5; CS231n notes |
| §7 | Forward vs reverse mode | Baydin et al. (2018) |
| §8 | Vanishing/exploding gradients | Hochreiter (1991); Bengio et al. (1994) |
| §9 | Autograd frameworks | Paszke et al. (2019, PyTorch) |

---

## Books

**Nielsen, M. (2015). *Neural Networks and Deep Learning*, Chapter 2.** — free at
<http://neuralnetworksanddeeplearning.com/chap2.html>. **The clearest derivation of the four
backpropagation equations** (§4), which this chapter follows. Read it.

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. §6.5 "Back-Propagation and Other Differentiation Algorithms"
covers the computational-graph view (§5), gradient checking (§6), and the general autodiff framing.

**Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic
Differentiation*, 2nd ed.** The definitive reference on automatic differentiation — forward vs reverse
mode (§7), and the memory/compute tradeoffs.

---

## Papers

- **Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986).** "Learning representations by
  back-propagating errors." *Nature* 323, 533-536. — **the paper that popularized backpropagation** for
  neural networks (§1-§4).
- **Werbos, P. J. (1974).** *Beyond Regression* (PhD thesis). — an earlier derivation of backprop in a
  general setting; the algorithm predates its 1986 fame.
- **Baydin, A. G., Pearlmutter, B. A., Radul, A. A. & Siskind, J. M. (2018).** "Automatic
  Differentiation in Machine Learning: a Survey." *JMLR* 18. — **the reference survey on autodiff**
  (§5, §7): forward vs reverse mode, VJPs, and how frameworks implement it. Free at
  <https://arxiv.org/abs/1502.05767>.
- **Hochreiter, S. (1991).** *Untersuchungen zu dynamischen neuronalen Netzen* (diploma thesis). — the
  first analysis of the **vanishing-gradient problem** (§8).
- **Bengio, Y., Simard, P. & Frasconi, P. (1994).** "Learning long-term dependencies with gradient
  descent is difficult." *IEEE TNN* 5(2). — the classic analysis of vanishing/exploding gradients in
  deep/recurrent nets (§8).
- **Paszke, A. et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library."
  *NeurIPS*. — how a modern autograd framework builds and traverses the tape (§9). Free at
  <https://arxiv.org/abs/1912.01703>.

---

## Lecture notes & tutorials

- **Stanford CS231n — "Backpropagation, Intuitions"** (<https://cs231n.github.io/optimization-2/>).
  The best intuition-building notes: backprop as local gradient flow on the computational graph, with
  worked examples and gradient-checking practice.
- **Karpathy, A. — "The spelled-out intro to neural networks and backpropagation: building micrograd"**
  (<https://www.youtube.com/watch?v=VMj-3S1tku0>). Builds a reverse-mode autograd engine from scratch,
  the clearest hands-on path to §5.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [micrograd](https://github.com/karpathy/micrograd) | a ~150-line reverse-mode autograd engine + MLP; the minimal backprop implementation |
| [PyTorch autograd](https://github.com/pytorch/pytorch/tree/main/torch/autograd) | the production tape-based reverse-mode engine our gradients are verified against |
| [JAX `grad` / `vjp` / `jvp`](https://github.com/google/jax) | functional autodiff exposing both reverse (`vjp`) and forward (`jvp`) modes (§7) |
| [`torch.autograd.gradcheck`](https://github.com/pytorch/pytorch/blob/main/torch/autograd/gradcheck.py) | PyTorch's built-in gradient checker (§6) |

---

## Deferred to later chapters

- **Neural network basics — the forward pass and computational graph** → [07.01](../01-neural-network-basics/)
- **Activations — the $\sigma'$ that shapes the backprop gradient** → [07.03](../03-activations/)
- **Initialization — keeping the product of Jacobians near 1** → [07.05](../05-initialization/)
- **Optimizers — using the gradients backprop computes** → [07.06](../06-optimizers/)
- **Normalization & residuals — architectural fixes for gradient flow** → [07.07](../07-normalization/)
- **Backprop through time (RNNs) and through attention** → [09.xx / 11.xx]
