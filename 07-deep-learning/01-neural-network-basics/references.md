# 07.01 — References: Neural Network Basics

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | MLP, forward pass | Goodfellow, Bengio & Courville, Ch. 6 |
| §4-§5 | Nonlinearity, representation | Rumelhart, Hinton & Williams (1986) |
| §6 | Universal approximation | Cybenko (1989); Hornik (1991) |
| §7 | Depth vs width | Telgarsky (2016); Montúfar et al. (2014) |
| §8 | Computational graph | Goodfellow et al. §6.5 |

---

## Books

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>. **Chapter 6 "Deep Feedforward Networks" is the reference** for
this chapter: the MLP, the forward pass, why nonlinearity, universal approximation (§6.4.1), and the
depth argument (§6.4.1). The standard graduate text.

**Nielsen, M. (2015). *Neural Networks and Deep Learning*.** — free at
<http://neuralnetworksanddeeplearning.com/>. Chapters 1-2 are the clearest gentle introduction to MLPs
and the forward pass, with an interactive universal-approximation demo (§6).

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** Chapter 5 "Neural Networks" gives
the classical statistical treatment of feedforward networks.

---

## Papers

- **Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986).** "Learning representations by
  back-propagating errors." *Nature* 323, 533-536. — **the paper that revived neural networks**:
  multilayer networks, learned internal representations (§5), and backprop ([07.02](../02-backpropagation/)).
- **Cybenko, G. (1989).** "Approximation by superpositions of a sigmoidal function." *Mathematics of
  Control, Signals and Systems* 2(4), 303-314. — **the universal approximation theorem** for sigmoidal
  activations (§6).
- **Hornik, K. (1991).** "Approximation capabilities of multilayer feedforward networks." *Neural
  Networks* 4(2), 251-257. — generalizes universal approximation to any non-polynomial activation (§6).
- **Telgarsky, M. (2016).** "Benefits of depth in neural networks." *COLT*. — **the depth-separation
  result** (§7): functions a deep net represents compactly that a shallow net needs exponentially many
  units for; the folding argument. Free at <https://arxiv.org/abs/1602.04485>.
- **Montúfar, G., Pascanu, R., Cho, K. & Bengio, Y. (2014).** "On the Number of Linear Regions of Deep
  Neural Networks." *NeurIPS*. — deep ReLU networks have exponentially more linear regions than shallow
  ones (§7). Free at <https://arxiv.org/abs/1402.1869>.
- **Minsky, M. & Papert, S. (1969).** *Perceptrons*. — the book whose XOR critique (§1) stalled neural
  networks until the 1986 backprop revival.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [PyTorch `nn.Linear` / `nn.Sequential`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py) | the dense layer and the forward pass our code is verified against |
| [`sklearn.neural_network.MLPClassifier`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_multilayer_perceptron.py) | a from-scratch (NumPy) MLP with backprop — the next chapter's target |
| [micrograd](https://github.com/karpathy/micrograd) | Karpathy's tiny autograd + MLP; the clearest minimal implementation of forward + backward |

---

## Deferred to later chapters

- **Backpropagation — computing the gradients to train these networks** → [07.02](../02-backpropagation/)
- **Activation functions — which nonlinearity and why** → [07.03](../03-activations/)
- **The perceptron and XOR — the starting point** → [03.09](../../03-supervised-learning/09-perceptron/)
- **Kernels — the other "map to an easy space" idea** → [03.07](../../03-supervised-learning/07-svm/)
- **Capacity and overfitting — the cost of universality** → [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)
- **CNNs, RNNs, Transformers — architectures beyond the MLP** → [Parts 8-11]
