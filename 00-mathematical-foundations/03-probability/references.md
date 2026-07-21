# 00.03 — References: Probability

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Aleatoric vs epistemic uncertainty | Kendall & Gal (2017); Hüllermeier & Waegeman (2021) |
| §2 | Axioms, conditional probability, chain rule | Blitzstein & Hwang, *Introduction to Probability*, Ch. 1-2 |
| §3-4 | Random variables, expectation, variance | Wasserman, *All of Statistics*, Ch. 2-3 |
| §4.3 | Ensemble variance formula | Hastie et al., *ESL*, §15.2 |
| §5-6 | Joint/conditional, conditional independence | Bishop, *PRML*, §1.2, §8.2 |
| §7 | Bayes, base rates, MAP | Murphy, *PML: An Introduction*, Ch. 2, §4.5 |
| §7.1 | Base rate fallacy | Kahneman & Tversky (1973); Gigerenzer & Hoffrage (1995) |
| §8 | Distribution zoo, conjugacy | Murphy, *PML*, Ch. 2; Blitzstein & Hwang, Ch. 3-5 |
| §9 | The Gaussian, conditioning formula | Bishop, *PRML*, §2.3 — the definitive treatment |
| §9.4 | Losses as negative log-likelihoods | Murphy, *PML*, §4.2; Bishop, *PRML*, §3.1 |
| §10 | Exponential family | Bishop, *PRML*, §2.4; Wainwright & Jordan (2008) |
| §11 | Change of variables | Murphy, *PML*, §2.8; Papamakarios et al. (2021) |
| §12 | LLN, CLT, Berry-Esseen | Wasserman, *All of Statistics*, §5.3-5.4 |
| §13 | Concentration inequalities | Boucheron, Lugosi & Massart (2013); Shalev-Shwartz & Ben-David, App. B |
| §14 | Jensen's inequality | Cover & Thomas, *Elements of Information Theory*, §2.6 |
| §15 | Sampling methods | MacKay, *ITILA*, Ch. 29; Bishop, *PRML*, Ch. 11 |

---

## Books

**Blitzstein, J. K. & Hwang, J. (2019). *Introduction to Probability*, 2nd ed. CRC Press.**
— free at <https://projects.iq.harvard.edu/stat110/home>
The best *first* probability book, by a distance. Harvard's Stat 110 lectures are on YouTube and
are exceptional. Start here if §2-§8 of this chapter felt fast.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.** — free at
<https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>
**§2.3 on the Gaussian is the single best 25 pages written on the subject** — the marginal and
conditional formulas of §9.3, derived carefully with all the linear algebra shown. Chapter 2 also
covers the exponential family and conjugate priors.

**Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.** — free at
<https://probml.github.io/pml-book/>
The modern reference. Chapter 2 covers this entire chapter with ML framing throughout; §4.5 for
MAP-as-regularization.

**Wasserman, L. (2004). *All of Statistics*. Springer.**
Fast and dense — probability in Chapters 1-5 with no wasted words. The right book if you want the
results without the pedagogy.

**MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge.**
— free at <https://www.inference.org.uk/mackay/itila/>
Chapter 29 on Monte Carlo methods is the source for §15, and the clearest explanation of why
rejection sampling collapses in high dimensions.

**Boucheron, S., Lugosi, G. & Massart, P. (2013). *Concentration Inequalities: A Nonasymptotic
Theory of Independence*. Oxford.**
The reference for §13. Deep water; consult rather than read.

**Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge.**
The philosophical case for probability-as-extended-logic. Opinionated and worth reading once,
especially §1-4 on why the Bayesian view is not merely a preference.

---

## Papers

### Uncertainty
- **Kendall, A. & Gal, Y. (2017).** "What Uncertainties Do We Need in Bayesian Deep Learning for
  Computer Vision?" *NeurIPS*. [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) — the
  aleatoric/epistemic split of §1, made operational.
- **Hüllermeier, E. & Waegeman, W. (2021).** "Aleatoric and epistemic uncertainty in machine
  learning: an introduction to concepts and methods." *Machine Learning* 110, 457-506.

### Base rates
- **Kahneman, D. & Tversky, A. (1973).** "On the psychology of prediction." *Psychological Review*
  80(4), 237-251. — the original base rate neglect result.
- **Gigerenzer, G. & Hoffrage, U. (1995).** "How to improve Bayesian reasoning without
  instruction: frequency formats." *Psychological Review* 102(4), 684-704. — the finding that
  presenting the problem in *counts* rather than percentages dramatically improves accuracy. This
  is why §7.1 shows the 100,000-person table.
- **Eddy, D. M. (1982).** "Probabilistic reasoning in clinical medicine." In *Judgment Under
  Uncertainty*. — the study finding most physicians answer the §7.1 question wrong.

### Exponential family and generative models
- **Wainwright, M. J. & Jordan, M. I. (2008).** "Graphical Models, Exponential Families, and
  Variational Inference." *Foundations and Trends in ML* 1(1-2). — free; the definitive treatment.
- **Kingma, D. P. & Welling, M. (2014).** "Auto-Encoding Variational Bayes." *ICLR*.
  [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) — the reparameterization trick of §11.
- **Papamakarios, G. et al. (2021).** "Normalizing Flows for Probabilistic Modeling and
  Inference." *JMLR* 22(57), 1-64. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762) — the
  change-of-variables formula of §11 turned into an architecture.

### Limit theorems
- **Berry, A. C. (1941)** and **Esseen, C.-G. (1942).** The Berry-Esseen theorem quantifying the
  CLT's rate (§12.1). Modern statement in Wasserman §5.4 or Durrett, *Probability: Theory and
  Examples*, §3.4.

---

## Courses and lectures

| Course | Institution | Link |
|---|---|---|
| Stat 110 — Probability | Harvard (Blitzstein) | <https://projects.iq.harvard.edu/stat110/home> |
| 6.041 — Probabilistic Systems Analysis | MIT | <https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/> |
| CS109 — Probability for Computer Scientists | Stanford | <https://web.stanford.edu/class/cs109/> |
| Seeing Theory — visual probability | Brown | <https://seeing-theory.brown.edu/> |

**Seeing Theory** is worth a special mention: interactive visualizations of expectation, the CLT,
and Bayesian inference. Fifteen minutes there will do more for §12 than an hour of reading.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`scipy.stats`](https://github.com/scipy/scipy/tree/main/scipy/stats) | `_continuous_distns.py` — how ~100 distributions are implemented, and the numerical care taken in each `_pdf`/`_cdf` |
| [`numpy.random.Generator`](https://github.com/numpy/numpy/tree/main/numpy/random) | `_generator.pyx` — the Ziggurat algorithm for Gaussians (faster than Box-Muller) |
| [`torch.distributions`](https://github.com/pytorch/pytorch/tree/main/torch/distributions) | how `rsample()` implements the reparameterization trick of §11, and which distributions support it |
| [`pyro`](https://github.com/pyro-ppl/pyro) / [`numpyro`](https://github.com/pyro-ppl/numpyro) | probabilistic programming — Bayes as a language construct |

---

## Deferred to later chapters

- **Entropy, KL divergence, mutual information** → [00.05](../05-information-theory/)
- **Estimators, MLE properties, confidence intervals, bootstrap** → [00.04](../04-statistics-and-inference/)
- **Log-sum-exp, stable softmax, Welford's algorithm** → [00.06](../06-numerical-methods/)
- **Naive Bayes, LDA/QDA in full** → [03.05](../../03-supervised-learning/05-generative-classifiers/)
- **Gaussian mixtures and EM** → [04.04](../../04-unsupervised-learning/04-gaussian-mixtures/)
- **MCMC, variational inference, the ELBO** → [12.02](../../12-generative-models/02-vae/)
- **Calibration — whether a predicted 0.9 means anything** → [05.06](../../05-model-evaluation/06-calibration/)
- **Thompson sampling and conjugate bandits** → [13.05](../../13-reinforcement-learning/05-bandits/)
