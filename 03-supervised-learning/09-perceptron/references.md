# 03.09 — References: The Perceptron

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | The perceptron and its rule | Rosenblatt (1958, 1962) |
| §5 | Convergence theorem | Novikoff (1962); Block (1962) |
| §7 | XOR, representational limits | Minsky & Papert (1969) |
| §7 | Backpropagation resolution | Rumelhart, Hinton & Williams (1986) |
| §8 | Loss comparison | Hastie et al., *ESL*, §4.5; Shalev-Shwartz & Ben-David, Ch. 9 |
| §9 | Pocket / averaged / kernel variants | Gallant (1990); Freund & Schapire (1999) |
| §10 | ADALINE, the delta rule | Widrow & Hoff (1960) |
| §11 | The perceptron as a neuron | Goodfellow et al., *Deep Learning*, §6.1 |

---

## Books

**Minsky, M. & Papert, S. (1969). *Perceptrons*. MIT Press.**
The book that ended the first neural-network era. Its formal results on what single-layer
perceptrons cannot represent (§7) are correct and important; the damage came from the broader,
pessimistic reading. The 1988 expanded edition adds a preface reflecting on the aftermath. Worth
reading to see how a technically sound result reshaped a field.

**Rosenblatt, F. (1962). *Principles of Neurodynamics: Perceptrons and the Theory of Brain
Mechanisms*. Spartan Books.**
The full development of the perceptron and its convergence theory, by its inventor.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§4.5.1 ("Rosenblatt's Perceptron Learning Algorithm") places it among the linear classifiers and
makes the connection to the separating-hyperplane methods of §8 explicit.

**Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning*.** — free at
<https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/>
Chapter 9 gives the modern learning-theory treatment: the perceptron as an online algorithm, the
mistake bound, and its relationship to margins and to the SVM.

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*.** — free at
<https://www.deeplearningbook.org/>
§1.2.1 tells the two-winter history, and §6.1 opens the feedforward-network chapter with exactly
the XOR example of §7 — the direct continuation of this chapter into
[Part 7](../../07-deep-learning/).

---

## Papers

### Foundational
- **Rosenblatt, F. (1958).** "The Perceptron: A Probabilistic Model for Information Storage and
  Organization in the Brain." *Psychological Review* 65(6), 386-408. — the original, and one of
  the most consequential papers in the field's history.
- **Novikoff, A. B. (1962).** "On convergence proofs on perceptrons." *Symposium on the
  Mathematical Theory of Automata* 12, 615-622. — **the convergence theorem of §5.** The proof in
  this chapter follows Novikoff's.
- **Block, H. D. (1962).** "The Perceptron: A Model for Brain Functioning." *Reviews of Modern
  Physics* 34(1), 123-135. — an independent convergence analysis.

### The winter and the thaw
- **Minsky, M. & Papert, S. (1969).** *Perceptrons* — see Books.
- **Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986).** "Learning representations by
  back-propagating errors." *Nature* 323, 533-536. — **the paper that ended the first AI winter**
  by supplying a way to train multilayer networks, resolving §7. The foundation of
  [07.02](../../07-deep-learning/02-backpropagation/).

### Variants
- **Widrow, B. & Hoff, M. E. (1960).** "Adaptive Switching Circuits." *IRE WESCON Convention
  Record*. — **ADALINE and the delta rule** (§10); the true start of the gradient-descent lineage.
- **Gallant, S. I. (1990).** "Perceptron-based learning algorithms." *IEEE Trans. Neural Networks*
  1(2), 179-191. — the **pocket algorithm** for non-separable data (§9).
- **Freund, Y. & Schapire, R. E. (1999).** "Large Margin Classification Using the Perceptron
  Algorithm." *Machine Learning* 37(3), 277-296. — the **voted / averaged perceptron** (§9),
  with generalization bounds showing it is competitive with SVMs. Experiment 4 reproduces its
  central finding.
- **Collins, M. (2002).** "Discriminative Training Methods for Hidden Markov Models: Theory and
  Experiments with Perceptron Algorithms." *EMNLP*. — the **structured perceptron**, which kept
  the averaged perceptron a serious tool in NLP for over a decade.
- **Crammer, K. et al. (2006).** "Online Passive-Aggressive Algorithms." *JMLR* 7, 551-585. —
  the modern margin-based online descendants of the perceptron.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `Perceptron`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_perceptron.py) | a thin wrapper over `SGDClassifier` with the perceptron loss — a direct demonstration that the perceptron rule *is* SGD on that loss (§8, §10) |
| [scikit-learn `SGDClassifier`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_stochastic_gradient.py) | `loss="perceptron"` vs `"hinge"` vs `"log_loss"` — the three losses of §8 selectable by one argument |

> **A telling detail.** sklearn does not have a bespoke perceptron solver. `Perceptron` is
> literally `SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1, penalty=None)`.
> The perceptron, logistic regression, and the linear SVM are, in sklearn, the *same* optimizer
> with three different loss functions — which is precisely the point of §8, made concrete in the
> source.

---

## Historical and popular accounts

- **Olazaran, M. (1996).** "A Sociological Study of the Official History of the Perceptrons
  Controversy." *Social Studies of Science* 26(3), 611-659. — a careful account of what Minsky and
  Papert actually claimed versus how it was received, and how much the "winter" owes to each.
- **Nielsen, M. *Neural Networks and Deep Learning*, Ch. 1.** — free at
  <http://neuralnetworksanddeeplearning.com/> — the clearest modern bridge from the perceptron to
  the sigmoid neuron and backprop, and a good companion to §11.

---

## Deferred to later chapters

- **Backpropagation — training the multilayer network XOR needs** → [07.02](../../07-deep-learning/02-backpropagation/)
- **Neural network basics — the perceptron generalized** → [07.01](../../07-deep-learning/01-neural-network-basics/)
- **Activation functions — the smooth replacements for the step** → [07.03](../../07-deep-learning/03-activations/)
- **Initialization — why the hidden units need to break symmetry** → [07.05](../../07-deep-learning/05-initialization/)
- **Weight averaging (SWA, EMA) — the averaged perceptron's descendants** → [07.06](../../07-deep-learning/06-optimizers/)
- **The kernel trick in full** → [03.07 §8](../07-svm/)
- **Hinge, logistic, and perceptron losses compared** → [07.04](../../07-deep-learning/04-loss-functions/)
