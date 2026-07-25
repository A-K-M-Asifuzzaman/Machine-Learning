# 09.02 — References: LSTM & GRU

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | LSTM cell, gates | Hochreiter & Schmidhuber (1997); Gers et al. (2000) |
| §3 | GRU cell | Cho et al. (2014) |
| §4 | Constant error carousel, gradient flow | Hochreiter & Schmidhuber (1997) |
| §5 | Long-range learning | Hochreiter & Schmidhuber (1997); Greff et al. (2017) |
| §6 | Forget-gate bias initialization | Gers et al. (2000); Jozefowicz et al. (2015) |
| §7 | LSTM vs GRU comparison | Chung et al. (2014); Greff et al. (2017) |

---

## The foundational papers

- **Hochreiter, S. & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation* 9(8). —
  the **LSTM**: the cell state, gates, and the "constant error carousel" that solves vanishing
  gradients (§1-§5). <https://www.bioinf.jku.at/publications/older/2604.pdf>.
- **Gers, F., Schmidhuber, J. & Cummins, F. (2000).** "Learning to Forget: Continual Prediction with
  LSTM." *Neural Computation*. — added the **forget gate** (not in the 1997 version) and its
  initialization (§2, §6).
- **Cho, K. et al. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical
  Machine Translation." *EMNLP*. — introduced the **GRU** (§3). <https://arxiv.org/abs/1406.1078>.

## Comparisons and analysis

- **Chung, J. et al. (2014).** "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
  Modeling." — **LSTM vs GRU**; comparable performance (§7). <https://arxiv.org/abs/1412.3555>.
- **Greff, K. et al. (2017).** "LSTM: A Search Space Odyssey." *IEEE TNNLS*. — a large ablation of LSTM
  components; the forget gate and output activation matter most (§5-§7). <https://arxiv.org/abs/1503.04069>.
- **Jozefowicz, R., Zaremba, W. & Sutskever, I. (2015).** "An Empirical Exploration of Recurrent
  Network Architectures." *ICML*. — searches RNN architectures; confirms the value of **positive
  forget-bias initialization** (§6). <https://proceedings.mlr.press/v37/jozefowicz15.html>.

## Textbook

- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, §10.10** — "The Long Short-Term
  Memory and Other Gated RNNs." Free at <https://www.deeplearningbook.org/>.
- **Olah, C. (2015).** "Understanding LSTM Networks." — the classic visual walkthrough of the gates
  (§2-§3). <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.LSTM`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py) | the reference cell (gate order i,f,g,o) verified against here |
| [`torch.nn.GRU`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py) | the reference GRU (gate order r,z,n) verified against here |
| [cuDNN RNN](https://docs.nvidia.com/deeplearning/cudnn/latest/) | the fused, fast LSTM/GRU kernels |

---

## Deferred to later chapters

- **The vanishing-gradient problem these cells solve** → [09.01](../01-rnn/)
- **Seq2seq and attention** → [09.03](../03-seq2seq-and-attention/)
- **Residual connections (the same gradient-path idea for depth)** → [08.02](../../08-computer-vision/02-cnn-architectures/)
- **Transformers** → [Part 11](../../11-transformers-and-llms/)
