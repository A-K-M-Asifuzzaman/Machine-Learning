# 09.01 — References: Recurrent Neural Networks

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | The recurrence, unrolling | Elman (1990); Goodfellow et al. Ch. 10 |
| §3 | BPTT | Werbos (1990); Rumelhart et al. (1986) |
| §4 | Vanishing/exploding gradients | Bengio et al. (1994); Hochreiter (1991); Pascanu et al. (2013) |
| §5 | Gradient clipping | Pascanu et al. (2013) |
| §6 | Long-range dependency failure | Bengio et al. (1994); Hochreiter & Schmidhuber (1997) |
| §7 | Bidirectional / architectures | Schuster & Paliwal (1997) |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — **Chapter 10 "Sequence
Modeling: Recurrent and Recursive Nets"** is the reference: the recurrence, BPTT, the vanishing/
exploding analysis (§10.7), and clipping. Free at <https://www.deeplearningbook.org/>.

---

## Foundational papers

- **Elman, J. L. (1990).** "Finding Structure in Time." *Cognitive Science*. — the **simple recurrent
  network** (§1). <https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1>.
- **Werbos, P. (1990).** "Backpropagation Through Time: What It Does and How to Do It." *Proc. IEEE*. —
  the **BPTT** algorithm (§3).
- **Rumelhart, D., Hinton, G. & Williams, R. (1986).** "Learning representations by back-propagating
  errors." *Nature*. — backpropagation, the basis of BPTT (§3).

## The gradient-flow problem

- **Hochreiter, S. (1991).** *Untersuchungen zu dynamischen neuronalen Netzen* (diploma thesis). — the
  first identification of the **vanishing-gradient** problem (§4).
- **Bengio, Y., Simard, P. & Frasconi, P. (1994).** "Learning Long-Term Dependencies with Gradient
  Descent is Difficult." *IEEE TNN*. — the definitive analysis: why the Jacobian product vanishes or
  explodes and why long-range learning fails (§4, §6). Free at
  <https://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf>.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013).** "On the difficulty of training Recurrent Neural
  Networks." *ICML*. — the spectral-radius condition and **gradient clipping** (§4-§5). Free at
  <https://arxiv.org/abs/1211.5063>.

## Architectures and applications

- **Schuster, M. & Paliwal, K. (1997).** "Bidirectional Recurrent Neural Networks." *IEEE TSP*. —
  bidirectional RNNs (§7).
- **Le, Q., Jaitly, N. & Hinton, G. (2015).** "A Simple Way to Initialize Recurrent Networks of
  Rectified Linear Units" (**IRNN**). — identity-initialized ReLU RNNs (§8). <https://arxiv.org/abs/1504.00941>.
- **Karpathy, A. (2015).** "The Unreasonable Effectiveness of Recurrent Neural Networks." — the classic
  intuition-building blog, char-RNN language modeling. <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.RNN`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py) | the reference cell this chapter verifies against |
| [`torch.nn.utils.clip_grad_norm_`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | gradient clipping (§5) |
| [min-char-rnn (Karpathy)](https://gist.github.com/karpathy/d4dee566867f8291f086) | a 100-line char-RNN with hand-written BPTT |

---

## Deferred to later chapters

- **LSTM & GRU — the gated fix for vanishing gradients** → [09.02](../02-lstm-gru/)
- **Seq2seq and attention** → [09.03](../03-seq2seq-and-attention/)
- **Backpropagation in general** → [07.02](../../07-deep-learning/02-backpropagation/)
- **Transformers — sequence modeling without recurrence** → [Part 11](../../11-transformers-and-llms/)
