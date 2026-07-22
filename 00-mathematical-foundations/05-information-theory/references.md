# 00.05 — References: Information Theory

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Axioms forcing $-\log p$ | Shannon (1948) §6; Cover & Thomas, *Elements of Information Theory*, Ch. 2 |
| §3 | Entropy and its properties | Cover & Thomas §2.1-2.2; MacKay, *ITILA*, Ch. 2 |
| §4 | Source coding theorem, Huffman | Cover & Thomas Ch. 5; MacKay Ch. 4-5 |
| §5 | Joint/conditional entropy, chain rule | Cover & Thomas §2.2, §2.5 |
| §6 | Cross-entropy as the classification loss | Goodfellow et al., *Deep Learning*, §3.13, §5.5 |
| §7 | KL divergence, Gibbs' inequality | Cover & Thomas §2.3, §2.6 |
| §8 | Forward vs reverse KL | Bishop, *PRML*, §10.1.2 — the definitive treatment, with figures |
| §9 | Mutual information | Cover & Thomas §2.4; Kraskov et al. (2004) for estimation |
| §10 | Jensen-Shannon | Lin (1991); Goodfellow et al. (2014) §4 for the GAN connection |
| §11 | Maximum entropy | Jaynes (1957); Cover & Thomas Ch. 12 |
| §12 | Minimum description length | Rissanen (1978); Grünwald, *The Minimum Description Length Principle* |
| §13 | Perplexity | Jurafsky & Martin, *SLP3*, Ch. 3 |

---

## Books

**Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.**
The standard reference, and the source for most of this chapter. Chapter 2 covers entropy through
mutual information; Chapter 5 is source coding; Chapter 12 is maximum entropy. Clear, rigorous,
and the proofs are short enough to actually read.

**MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge.**
— free at <https://www.inference.org.uk/mackay/itila/>
Idiosyncratic and brilliant. MacKay's insight — that inference, compression, and learning are the
same activity — is the thesis of §1 and §12 of this chapter. Chapters 1-6 are the best available
introduction to coding; the whole book is worth reading once, and it is free.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.** — free
**§10.1.2 is the definitive treatment of forward vs reverse KL**, with the figures that made the
mode-covering/mode-seeking distinction standard vocabulary. Read it alongside §8 here.

**Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.** — free at
<https://www.deeplearningbook.org/>
§3.13 for information theory as used in deep learning; §5.5 for maximum likelihood as KL
minimization.

**Grünwald, P. D. (2007). *The Minimum Description Length Principle*. MIT Press.**
The comprehensive treatment of §12. Long; the introduction alone conveys the idea.

**Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, 3rd ed. draft.** — free at
<https://web.stanford.edu/~jurafsky/slp3/>
Chapter 3 for perplexity, including the tokenization caveat of §13.

---

## Papers

### Foundational
- **Shannon, C. E. (1948).** "A Mathematical Theory of Communication." *Bell System Technical
  Journal* 27, 379-423 and 623-656. — the paper that created the field. Genuinely readable;
  §6 contains the axiomatic derivation of entropy used in §2 here.
- **Kullback, S. & Leibler, R. A. (1951).** "On Information and Sufficiency." *Annals of
  Mathematical Statistics* 22(1), 79-86.
- **Jaynes, E. T. (1957).** "Information Theory and Statistical Mechanics." *Physical Review*
  106(4), 620-630. — the maximum entropy principle of §11.
- **Rissanen, J. (1978).** "Modeling by shortest data description." *Automatica* 14(5), 465-471.
  — MDL.
- **Lin, J. (1991).** "Divergence measures based on the Shannon entropy." *IEEE Trans.
  Information Theory* 37(1), 145-151. — Jensen-Shannon.

### In machine learning
- **Goodfellow, I. et al. (2014).** "Generative Adversarial Networks." *NeurIPS*.
  [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) — §4 derives the JS-divergence objective
  discussed in §10.
- **Arjovsky, M., Chintala, S. & Bottou, L. (2017).** "Wasserstein GAN." *ICML*.
  [arXiv:1701.07875](https://arxiv.org/abs/1701.07875) — the diagnosis that JS gives zero gradient
  on disjoint supports, and the replacement. Read §2-3 immediately after §10 here.
- **Kingma, D. P. & Welling, M. (2014).** "Auto-Encoding Variational Bayes." *ICLR*.
  [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) — the ELBO, whose KL term is the reverse
  direction of §8.
- **Hinton, G., Vinyals, O. & Dean, J. (2015).** "Distilling the Knowledge in a Neural Network."
  [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) — "dark knowledge", §14.
- **Tishby, N. & Zaslavsky, N. (2015).** "Deep Learning and the Information Bottleneck Principle."
  [arXiv:1503.02406](https://arxiv.org/abs/1503.02406) — influential and contested; see Saxe et al.
  (2018), "On the Information Bottleneck Theory of Deep Learning", for the rebuttal. Worth reading
  both.
- **Oord, A. van den, Li, Y. & Vinyals, O. (2018).** "Representation Learning with Contrastive
  Predictive Coding." [arXiv:1807.03748](https://arxiv.org/abs/1807.03748) — InfoNCE as a mutual
  information lower bound.
- **Poole, B. et al. (2019).** "On Variational Bounds of Mutual Information." *ICML*.
  [arXiv:1905.06922](https://arxiv.org/abs/1905.06922) — why estimating MI in high dimensions is
  much harder than it looks. Important caveat to §9.
- **Kraskov, A., Stögbauer, H. & Grassberger, P. (2004).** "Estimating mutual information."
  *Physical Review E* 69, 066138. — the KSG estimator that `sklearn.feature_selection.mutual_info_*`
  implements, and a better choice than the binning used in `from_scratch.py`.

---

## Courses and lectures

| Course | Institution | Link |
|---|---|---|
| Information Theory, Pattern Recognition and Neural Networks (MacKay) | Cambridge | <https://www.inference.org.uk/itprnn_lectures/> |
| 6.441 Information Theory | MIT | <https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/> |
| EE376A Information Theory | Stanford | <https://web.stanford.edu/class/ee376a/> |

**MacKay's own lecture videos** are the recommendation here — the same material as his book,
delivered with unusual clarity about *why* each quantity is defined the way it is.

---

## Visual explanations

- **Colah, C. (2015).** "Visual Information Theory." <https://colah.github.io/posts/2015-09-Visual-Information/>
  The single best visual introduction to entropy, cross-entropy, and KL divergence. If §3-§7 did
  not click, read this next — it makes the "code length" picture concrete with diagrams.
- **Three-way relationship diagrams** for $H(X), H(Y), H(X\mid Y), I(X;Y)$ — the Venn-style
  picture in Cover & Thomas Fig. 2.2 is worth memorizing.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`scipy.stats.entropy`](https://github.com/scipy/scipy/blob/main/scipy/stats/_entropy.py) | handles both entropy and KL in one function; note the zero-handling |
| [`sklearn.feature_selection`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py) | the KSG nearest-neighbour MI estimator — much better than binning |
| [`torch.nn.CrossEntropyLoss`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py) | note it fuses log-softmax with NLL for numerical stability (see [00.06](../06-numerical-methods/)) |
| [`torch.nn.KLDivLoss`](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) | note the argument-order trap: it expects log-probabilities first |

---

## Deferred to later chapters

- **Log-sum-exp, stable softmax and cross-entropy** → [00.06](../06-numerical-methods/)
- **Information gain, gain ratio, Gini in real trees** → [03.08](../../03-supervised-learning/08-decision-trees/)
- **Calibration and proper scoring rules** → [05.06](../../05-model-evaluation/06-calibration/)
- **Label smoothing** → [07.08](../../07-deep-learning/08-regularization/)
- **The ELBO derived in full** → [12.02](../../12-generative-models/02-vae/)
- **JS divergence, mode collapse, WGAN** → [12.03](../../12-generative-models/03-gan/)
- **Perplexity and LM evaluation** → [11.02](../../11-transformers-and-llms/02-pretraining/)
- **Knowledge distillation** → [19.04](../../19-mlops/04-efficiency/)
- **InfoNCE and contrastive learning** → [08.05](../../08-computer-vision/05-vision-transformers/)
