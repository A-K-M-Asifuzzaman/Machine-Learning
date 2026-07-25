# 07.09 — References: Training Dynamics & Debugging

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | A recipe for training neural nets | Karpathy (2019) |
| §3 | Initial-loss sanity check ($\log K$) | Karpathy, CS231n notes |
| §4 | Overfit a single batch | Karpathy (2019) |
| §5 | Loss-curve signatures | CS231n "Learning" notes |
| §6 | Gradient / update statistics, the $10^{-3}$ ratio | CS231n; Bottou (2012) |
| §7 | Learning-rate range test | Smith (2017) |
| §8 | Gradient checking, common bugs | Deep Learning §11; CS231n |
| §9 | Monitoring, reproducibility | PyTorch reproducibility docs |
| §10 | Misconceptions | Karpathy (2019) |

---

## The core reads

**Karpathy, A. (2019). "A Recipe for Training Neural Networks."** — the single best practical guide to
this chapter: "neural net training is a leaky abstraction," become one with the data, the fixed-seed
baseline, overfit-then-regularize, and the specific sanity checks (initial loss $= \log K$, overfit
one batch, human baseline). Free at <https://karpathy.github.io/2019/04/25/recipe/>.

**Stanford CS231n — "Convolutional Neural Networks for Visual Recognition."** — the notes on
[optimization / learning](https://cs231n.github.io/neural-networks-3/) are the reference for the
babysitting checklist: loss-curve shapes, the update-to-weight ratio (~$10^{-3}$), gradient/activation
histograms, and gradient checking with *relative* error. Free at <https://cs231n.github.io/>.

---

## Papers and primary sources

- **Smith, L. N. (2017).** "Cyclical Learning Rates for Training Neural Networks." *WACV*. — introduces
  the **LR range test** (§7): sweep the learning rate up and read the loss to find the bounds. Free at
  <https://arxiv.org/abs/1506.01186>.
- **Bottou, L. (2012).** "Stochastic Gradient Descent Tricks." In *Neural Networks: Tricks of the
  Trade*. — practical SGD diagnostics and learning-rate selection (§6). Free at
  <https://leon.bottou.org/papers/bottou-tricks-2012>.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — **§11 "Practical
  Methodology"** (default baselines, whether to gather more data, hyperparameter search) and **§8.2**
  on gradient checking. Free at <https://www.deeplearningbook.org/>.
- **Smith, S. L. et al. (2018).** "Don't Decay the Learning Rate, Increase the Batch Size." *ICLR*. —
  the learning-rate / batch-size relationship behind the batch-size sensitivity in the §8 bug table.
  Free at <https://arxiv.org/abs/1711.00489>.

---

## Tooling and reproducibility

| Source | What to look at |
|---|---|
| [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) | seeding, deterministic algorithms, and what still varies across GPU runs (§9) |
| [Weights & Biases](https://docs.wandb.ai/) / [TensorBoard](https://www.tensorflow.org/tensorboard) | logging loss, gradient norms, and update ratios during training (§9) |
| [`torch.autograd.gradcheck`](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) | reference finite-difference gradient checker (§8) |

---

## Deferred to later chapters

- **Optimizers and learning-rate schedules in depth** → [07.06](../06-optimizers/)
- **Initialization and why bad init looks like a training bug** → [07.05](../05-initialization/)
- **Normalization for training stability** → [07.07](../07-normalization/)
- **Regularization once the model can overfit** → [07.08](../08-regularization/)
- **Metrics that back the loss (the "actual task metric")** → [05.02](../../05-model-evaluation/02-regression-metrics/)–[05.03](../../05-model-evaluation/03-classification-metrics/)
- **Experiment tracking / MLOps at scale** → [19.xx]
