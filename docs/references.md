# References — annotated bibliography

Every source this repository draws on, with an honest note on **what each one is actually good
for**. A bibliography without that note is just a list; the point here is to tell you *when to
reach for which book*.

Sources marked **(free)** are legally free from the authors or publishers.

> **On usage:** the prose in this repository is written from scratch. These works are cited for
> the ideas, derivations, and framings they contribute. No text is reproduced verbatim. Where a
> derivation closely follows a specific treatment, the chapter's `references.md` names the exact
> section.

---

## 1. Core machine learning textbooks

### The Elements of Statistical Learning — Hastie, Tibshirani, Friedman (2nd ed., 2009) **(free)**
<https://hastie.su.domains/ElemStatLearn/>

The statistical backbone of this repository. Unmatched on linear methods, regularization,
basis expansion, trees, boosting, and the bias-variance decomposition.
**Reach for it when**: you want the statistician's view and full rigour.
**Warning**: it is dense and assumes real linear algebra and statistics. Not a first book.
Key chapters used here: 3 (linear regression), 4 (classification), 7 (model assessment),
9 (additive models & trees), 10 (boosting), 12 (SVM), 14 (unsupervised).

### An Introduction to Statistical Learning — James, Witten, Hastie, Tibshirani (2nd ed., 2021) **(free)**
<https://www.statlearning.com/>

ESL with the hard math removed and the intuition kept. Python edition available (ISLP).
**Reach for it when**: ESL loses you and you need the same idea explained gently first.

### Pattern Recognition and Machine Learning — Bishop (2006) **(free)**
<https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>

The Bayesian counterweight to ESL. Best available treatment of the EM algorithm, mixture
models, graphical models, variational inference, and the exponential family.
**Reach for it when**: you want probabilistic derivations done properly. Ch. 9 (EM) and
Ch. 12 (latent variables) are used heavily in Part 4 of this repo.

### Probabilistic Machine Learning: An Introduction / Advanced Topics — Murphy (2022 / 2023) **(free)**
<https://probml.github.io/pml-book/>

The modern, comprehensive reference — effectively PRML updated for the deep learning era.
Enormous. **Reach for it when**: you need a topic covered that the older books predate,
or you want the current probabilistic framing of something.
Companion code: <https://github.com/probml/pyprobml>

### Understanding Machine Learning: From Theory to Algorithms — Shalev-Shwartz & Ben-David (2014) **(free)**
<https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/>

The theory book. PAC learning, VC dimension, Rademacher complexity, why learning is possible
at all. **Reach for it when**: you want to know *why* generalization works, not just that it does.
Used in Part 5.01.

### Machine Learning: A Probabilistic Perspective — Murphy (2012)

The predecessor to the PML books. Still the best single-volume reference for many classical
probabilistic models. Superseded for most purposes by PML.

### Mathematics for Machine Learning — Deisenroth, Faisal, Ong (2020) **(free)**
<https://mml-book.github.io/>

Exactly the math you need for ML and nothing more — linear algebra, analytic geometry, matrix
decompositions, vector calculus, probability, optimization, then four ML applications.
**Reach for it when**: starting Part 0 of this repository. This is the ideal companion.

---

## 2. Deep learning

### Deep Learning — Goodfellow, Bengio, Courville (2016) **(free)**
<https://www.deeplearningbook.org/>

The canonical text. Part I (math) and Part II (modern practical DL) remain excellent;
Part III (research) has aged. **Reach for it when**: you want the definitive treatment of
optimization for deep models, regularization, and representation learning. Ch. 6-8 used
throughout Part 7.

### Understanding Deep Learning — Simon Prince (2023) **(free)**
<https://udlbook.github.io/udlbook/>

The best *current* deep learning textbook. Clear, beautifully illustrated, covers transformers,
diffusion, and modern RL — everything Goodfellow predates. Free PDF plus notebooks.
**Reach for it when**: you want a modern, visual, rigorous explanation of anything in Parts 7-13.

### Dive into Deep Learning — Zhang, Lipton, Li, Smola **(free)**
<https://d2l.ai/>

Interactive: every concept comes with runnable PyTorch/NumPy code. Adopted by 500+ universities.
**Reach for it when**: you want to *run* the idea immediately. The from-scratch philosophy of
this repository is closest to D2L's.

### Deep Learning: Foundations and Concepts — Bishop & Bishop (2024)
<https://www.bishopbook.com/>

Bishop's rewrite for the deep learning era. Strong on transformers and diffusion from a
probabilistic angle.

### The Deep Learning Tuning Playbook — Google Research **(free)**
<https://github.com/google-research/tuning_playbook>

Not a textbook — a distilled practitioner's guide to actually making training work.
**Reach for it when**: your model trains but not well. Used in Part 7.09.

### Neural Networks: Zero to Hero — Andrej Karpathy **(free)**
<https://karpathy.ai/zero-to-hero.html>

Video series building backprop, then a GPT, from nothing. The single best resource for
*understanding by building*. Companion repos: `micrograd`, `makemore`, `nanoGPT`.

---

## 3. Mathematics

### Convex Optimization — Boyd & Vandenberghe (2004) **(free)**
<https://web.stanford.edu/~boyd/cvxbook/>

Definitive on convexity, duality, KKT conditions. Used in Part 0.02 and the SVM chapter.

### Numerical Optimization — Nocedal & Wright (2nd ed., 2006)

The reference for the algorithms themselves: line search, trust region, quasi-Newton, L-BFGS.

### Linear Algebra and Learning from Data — Gilbert Strang (2019)

Strang writes linear algebra the way it is actually used in ML. His MIT 18.06 lectures are free
and are the recommended first exposure.

### The Matrix Cookbook — Petersen & Pedersen **(free)**
<https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html>

Not for reading — for looking up matrix derivative identities mid-derivation. Keep it open.

### Information Theory, Inference, and Learning Algorithms — MacKay (2003) **(free)**
<https://www.inference.org.uk/mackay/itila/>

Entropy, coding, and inference unified. Idiosyncratic and brilliant. Used in Part 0.05.

### All of Statistics — Wasserman (2004)

Fast, dense coverage of everything a computer scientist needs from statistics.

### Bayesian Data Analysis — Gelman et al. (3rd ed.) **(free)**
<http://www.stat.columbia.edu/~gelman/book/>

The Bayesian reference. Used in Part 0.04.

---

## 4. Domain-specific

| Domain | Book | Notes |
|---|---|---|
| **RL** | Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) **(free)** — <http://incompleteideas.net/book/the-book-2nd.html> | The RL book. Nothing else is close for fundamentals. Part 13 follows its structure. |
| **NLP** | Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft) **(free)** — <https://web.stanford.edu/~jurafsky/slp3/> | Continuously updated, now includes LLMs. Part 10 follows it. |
| **Time series** | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.) **(free)** — <https://otexts.com/fpp3/> | The best forecasting text, free and practical. Part 15. |
| **Interpretability** | Molnar, *Interpretable Machine Learning* **(free)** — <https://christophm.github.io/interpretable-ml-book/> | Comprehensive on PDP/ICE/ALE/LIME/SHAP. Part 17 follows it. |
| **Info retrieval / RAG** | Manning, Raghavan, Schütze, *Introduction to Information Retrieval* **(free)** — <https://nlp.stanford.edu/IR-book/> | Foundations that RAG systems rediscovered. |
| **Gaussian processes** | Rasmussen & Williams, *Gaussian Processes for Machine Learning* **(free)** — <http://gaussianprocess.org/gpml/> | Definitive. |
| **Graphical models** | Koller & Friedman, *Probabilistic Graphical Models* | Exhaustive; use as reference, not cover-to-cover. |
| **Recommenders** | Aggarwal, *Recommender Systems: The Textbook* | Broad coverage of the classical methods. |

---

## 5. Practice, engineering, and systems

### Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Géron (3rd ed., 2022)
Companion code **(free)**: <https://github.com/ageron/handson-ml3>

The best applied ML book. If this repository's theory chapters are the "why", Géron is the "how".

### Machine Learning with PyTorch and Scikit-Learn — Raschka, Liu, Mirjalili (2022)
Companion code **(free)**: <https://github.com/rasbt/machine-learning-book>

Excellent middle ground: real derivations *and* clean code.

### Designing Machine Learning Systems — Chip Huyen (2022)

The production reference: data engineering, feature stores, deployment, monitoring, drift.
Part 19 and Part 20 follow its framing.
Free companion: <https://github.com/chiphuyen/machine-learning-systems-design>

### Machine Learning Engineering — Andriy Burkov **(free to read online)**
<http://www.mlebook.com/>

Short, practical, checklist-driven view of the ML lifecycle.

### Approaching (Almost) Any Machine Learning Problem — Abhishek Thakur **(free)**
<https://github.com/abhishekkrthakur/approachingalmost>

A competitive practitioner's playbook: cross-validation, feature engineering, tuning.

---

## 6. Courses

| Course | Institution | Link |
|---|---|---|
| CS229 — Machine Learning | Stanford | <https://cs229.stanford.edu/> |
| CS231n — CNNs for Visual Recognition | Stanford | <https://cs231n.stanford.edu/> |
| CS224n — NLP with Deep Learning | Stanford | <https://web.stanford.edu/class/cs224n/> |
| CS234 — Reinforcement Learning | Stanford | <https://web.stanford.edu/class/cs234/> |
| CS285 — Deep Reinforcement Learning | Berkeley | <https://rail.eecs.berkeley.edu/deeprlcourse/> |
| 6.S191 — Introduction to Deep Learning | MIT | <http://introtodeeplearning.com/> |
| 18.06 — Linear Algebra (Strang) | MIT | <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/> |
| Practical Deep Learning for Coders | fast.ai | <https://course.fast.ai/> |
| ML / DL Specializations — Andrew Ng | DeepLearning.AI | <https://www.deeplearning.ai/> |
| NLP & Deep RL Courses | Hugging Face | <https://huggingface.co/learn> |

CS229's lecture notes and the **CS229 cheatsheets** (<https://stanford.edu/~shervine/teaching/cs-229/>)
are the most efficient revision material that exists for classical ML.

---

## 7. Reference implementations studied

Reading good source code is a skill. These are the codebases worth reading:

| Repo | Why read it |
|---|---|
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | The gold standard for API design and numerically careful classical ML. Read `linear_model/_coordinate_descent.pyx`, `tree/_splitter.pyx`. |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | Autograd engine internals (`torch/csrc/autograd/`). |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | Reverse-mode autodiff in ~150 lines. Read this before reading PyTorch's. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | A complete, readable, trainable GPT. The reference for Part 11. |
| [karpathy/minGPT](https://github.com/karpathy/minGPT) | The pedagogical predecessor to nanoGPT. |
| [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) | Every algorithm with runnable code, from scratch and framework versions. |
| [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) | Clean NumPy implementations of most classical algorithms. |
| [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | 60+ papers implemented with line-by-line annotations. |
| [huggingface/transformers](https://github.com/huggingface/transformers) | How production transformer code is actually organized. |
| [probml/pyprobml](https://github.com/probml/pyprobml) | Code for Murphy's PML books. |
| [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook) | Distilled empirical wisdom on training. |

---

## 8. Interview preparation

| Resource | Link |
|---|---|
| Machine Learning Interviews (Alireza Dirafzoon) | <https://github.com/alirezadir/Machine-Learning-Interviews> |
| ML Interview Guide (Khang Pham) | <https://github.com/khangich/machine-learning-interview> |
| ML Systems Design (Chip Huyen) | <https://github.com/chiphuyen/machine-learning-systems-design> |
| Introduction to ML Interviews Book (Chip Huyen) **(free)** | <https://huyenchip.com/ml-interviews-book/> |
| CS229 Cheatsheets (Shervine Amidi) | <https://stanford.edu/~shervine/teaching/cs-229/> |

---

## 9. Staying current

- **Papers with Code** — <https://paperswithcode.com/> — SOTA leaderboards linked to implementations
- **arXiv Sanity / alphaXiv** — filtered arXiv browsing
- **ML Papers of the Week** — <https://github.com/dair-ai/ML-Papers-of-the-Week>
- **The Batch** (DeepLearning.AI), **Import AI** (Jack Clark) — weekly newsletters
- **Distill.pub** — <https://distill.pub/> — archived but the finest visual explanations ever published on ML
- **Lil'Log** (Lilian Weng) — <https://lilianweng.github.io/> — the best long-form technical blog in ML

---

## 10. Per-chapter citations

Every chapter carries its own `references.md` naming the **exact** sections used, so a claim in
this repository can always be traced back to a source you can check. If you find a chapter
making a claim without a traceable citation, that is a bug — please open an issue.
