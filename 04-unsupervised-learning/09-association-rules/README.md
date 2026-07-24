# 04.09 — Association Rule Mining

> **Prerequisites**: [00.03](../../00-mathematical-foundations/03-probability/) (conditional
> probability, independence). No heavy math — this chapter is about counting cleverly.
> **You will be able to**: define support, confidence, and lift, run the Apriori algorithm using the
> anti-monotone pruning that makes it tractable, generate rules, and avoid the confidence trap that
> makes obvious co-occurrences look like insights.

---

## Table of contents

1. [Market-basket analysis](#1-market-basket-analysis)
2. [Itemsets and support](#2-itemsets-and-support)
3. [Association rules: confidence and lift](#3-association-rules-confidence-and-lift)
4. [The Apriori principle](#4-the-apriori-principle)
5. [The Apriori algorithm](#5-the-apriori-algorithm)
6. [Generating rules from itemsets](#6-generating-rules-from-itemsets)
7. [The confidence trap and why lift matters](#7-the-confidence-trap-and-why-lift-matters)
8. [FP-Growth and scaling](#8-fp-growth-and-scaling)
9. [Practical issues](#9-practical-issues)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Market-basket analysis

Association rule mining finds **co-occurrence patterns** in transactional data: baskets of items,
each transaction a set. The canonical example is the supermarket — "customers who buy bread and
butter also buy milk" — but the same structure appears everywhere: products bought together (cross-
selling), web pages visited together, symptoms occurring together, genes expressed together. The
output is a set of human-readable **if-then rules**: *if a basket contains {bread, butter}, then it
likely also contains {milk}*.

Unlike clustering or dimensionality reduction, association mining is about **discrete co-occurrence**,
not geometry, and it produces *interpretable rules* rather than a model. The challenge is
combinatorial: with $d$ items there are $2^d$ possible itemsets, so naive enumeration is hopeless. The
**Apriori** algorithm's clever pruning (§4) is what makes it tractable, and it is one of the most
elegant ideas in data mining.

---

## 2. Itemsets and support

An **itemset** is a set of items (e.g. {bread, milk}). Its **support** is the fraction of transactions
that contain *all* of its items:

$$
\mathrm{support}(X) = \frac{\text{number of transactions containing } X}{\text{total transactions}}.
$$

Support measures how *frequent* a combination is. An itemset is **frequent** if its support meets a
minimum threshold `min_support`. The first goal of association mining is to find **all frequent
itemsets** — the combinations common enough to be worth reasoning about. This is the hard part
(exponentially many candidates), and it is what Apriori solves; turning frequent itemsets into rules
(§6) is comparatively easy.

`min_support` is the key knob: set it too high and you miss interesting patterns; too low and you drown
in combinations (and compute) (§9).

---

## 3. Association rules: confidence and lift

An **association rule** $X \Rightarrow Y$ (with $X, Y$ disjoint itemsets) says "baskets with $X$ tend
to have $Y$." Three metrics grade it:

- **Support** of the rule = $\mathrm{support}(X \cup Y)$ — how often the whole pattern occurs. Filters
  out rare, unreliable rules.
- **Confidence** = $\dfrac{\mathrm{support}(X \cup Y)}{\mathrm{support}(X)} = \hat{P}(Y \mid X)$ — of the
  baskets with $X$, what fraction also have $Y$. This is the rule's "reliability."
- **Lift** = $\dfrac{\mathrm{confidence}(X \Rightarrow Y)}{\mathrm{support}(Y)} = \dfrac{\hat P(Y\mid X)}{\hat P(Y)}$
  — how much *more* likely $Y$ is given $X$ than at baseline. **Lift > 1** means $X$ and $Y$ are
  *positively* associated (buying $X$ makes $Y$ more likely); **lift = 1** means independent (no real
  association); **lift < 1** means negatively associated.

Confidence alone is seductive but misleading (§7); **lift is the metric that tells you whether a rule
is a genuine association or just an artifact of a common item**. A good rule has adequate support
(reliable), high confidence (holds often), and lift meaningfully above 1 (a real, non-trivial pattern).

---

## 4. The Apriori principle

The insight that makes frequent-itemset mining feasible is the **Apriori principle** (anti-monotonicity
/ downward closure):

> **If an itemset is infrequent, then every superset of it is also infrequent.**

The reason is immediate: adding items to a set can only *reduce* (never increase) the number of
transactions that contain all of it, so $\mathrm{support}(X \cup \lbrace i\rbrace) \le \mathrm{support}(X)$.
If $X$ already fails the support threshold, no superset can pass it.

This single fact enables massive pruning. Instead of counting all $2^d$ itemsets, we build frequent
itemsets **level by level** — first frequent singletons, then pairs, then triples — and at each level
we only consider candidates whose *sub*sets are all already frequent. The moment an itemset is found
infrequent, its entire lattice of supersets is eliminated without ever being counted. Experiment 1
measures how many candidates the anti-monotone pruning removes.

---

## 5. The Apriori algorithm

**Input**: transactions, `min_support`.

1. **$L_1$**: count each single item's support; keep the frequent ones.
2. **For $k = 2, 3, \dots$** until no frequent itemsets remain:
   1. **Candidate generation.** Form candidate $k$-itemsets by joining pairs of frequent
      $(k{-}1)$-itemsets that share $k{-}2$ items.
   2. **Prune (Apriori principle, §4).** Discard any candidate that has a $(k{-}1)$-subset which is
      *not* frequent — it cannot be frequent, so do not even count it.
   3. **Count.** Scan the transactions and count the surviving candidates' supports.
   4. **$L_k$**: keep those meeting `min_support`.
3. **Output** the union of all $L_k$ — every frequent itemset.

The prune step (2.2) is where the Apriori principle pays off: it avoids counting the vast majority of
candidates. `from_scratch.py` implements this and verifies the frequent itemsets exactly against
brute-force enumeration (every subset counted) — identical results, far less work.

---

## 6. Generating rules from itemsets

Once you have the frequent itemsets, generating rules is straightforward. For each frequent itemset
$Z$, consider every way to split it into a non-empty antecedent $X$ and consequent $Y = Z \setminus X$,
and compute the rule $X \Rightarrow Y$'s confidence $\mathrm{support}(Z)/\mathrm{support}(X)$. Keep
rules meeting a **`min_confidence`** threshold, and report their lift.

There is an anti-monotone shortcut here too: if $X \Rightarrow Y$ fails the confidence threshold, then
moving items from $X$ to $Y$ (a "weaker" antecedent) also fails, so those rules can be pruned. The
practical output is a ranked list of rules — usually sorted by **lift** (most surprising / strongest
associations first), filtered to adequate support and confidence.

---

## 7. The confidence trap and why lift matters

This is the most important practical lesson. **High confidence does not mean a useful rule.** Suppose
90% of *all* baskets contain milk. Then *any* rule $X \Rightarrow \text{milk}$ has confidence around
90% — including {pickles} ⇒ milk, {batteries} ⇒ milk, nonsense ⇒ milk — simply because milk is
everywhere. The high confidence reflects milk's popularity, not any association with $X$.

**Lift exposes this.** For those rules, lift $= 0.90 / 0.90 = 1.0$ — exactly independent, no real
association. A rule is only interesting when lift is meaningfully **above 1**: {diapers} ⇒ {beer} with
lift 3 means beer is 3× more likely in diaper baskets than baseline — a genuine, actionable pattern.
Experiment 2 constructs a rule with 90% confidence but lift 1.0 (a common consequent, no association)
alongside a rule with lower confidence but lift 3 (a real association), showing that **confidence
ranks them backwards and lift ranks them correctly**.

The rule: **never rank association rules by confidence alone** — a rule needs support (reliable),
confidence (holds), *and* lift > 1 (real). Lift, or a related independence-corrected measure
(leverage, conviction), is what separates insight from triviality.

---

## 8. FP-Growth and scaling

Apriori's weakness is that it makes **multiple passes** over the data (one per level) and can generate
enormous numbers of candidates on dense data. **FP-Growth** (Frequent Pattern Growth) avoids both:

- It compresses the transactions into an **FP-tree** — a prefix tree of items ordered by frequency,
  so shared prefixes are stored once.
- It mines frequent patterns **recursively** from the tree via "conditional" sub-trees, **without ever
  generating candidate itemsets** and with only two data passes.

FP-Growth is typically much faster than Apriori on large or dense datasets and is the default in most
libraries. Apriori remains the clearer *conceptual* algorithm — its anti-monotone pruning (§4) is the
idea to understand — while FP-Growth is the one to *run* at scale. **Eclat** (using vertical
transaction-ID lists and set intersections) is a third common approach.

---

## 9. Practical issues

- **Threshold selection.** `min_support` and `min_confidence` control both the number of rules and the
  compute. Too strict and you find nothing; too loose and you get a combinatorial explosion of rules
  and runtime. Tune to get a manageable number of rules.
- **Combinatorial explosion.** On dense data (many items per transaction, low support threshold), the
  number of frequent itemsets and candidates can blow up; this is Apriori's main limitation (§8).
- **Redundant rules.** Many rules are logical consequences of others (if {A,B}⇒C, then subsets/related
  rules abound). Mining **closed** or **maximal** frequent itemsets, or ranking by lift, controls
  redundancy.
- **Spurious rules.** With many items you will find high-lift rules by chance; validate on held-out
  data or apply statistical significance tests. Correlation is not causation — a rule is an
  association, not a mechanism.

---

## 10. Common misconceptions

**"High confidence means a strong, useful rule."**
No — if the consequent is common, confidence is high for *any* antecedent (§7). Check **lift**: only
lift > 1 indicates a real association.

**"Association rules imply causation."**
They are co-occurrence patterns, not causal claims (§9). {diapers}⇒{beer} does not mean diapers cause
beer purchases.

**"Apriori counts all possible itemsets."**
Its whole point is *not* to — the anti-monotone principle prunes the supersets of infrequent itemsets
without counting them (§4–§5).

**"Lower `min_support` is always better (finds more)."**
It finds more, but the number of itemsets and the compute can explode combinatorially (§9). There is a
practical floor.

**"Apriori is the algorithm to use."**
It is the clearest to *understand*, but **FP-Growth** is usually faster in practice and is the library
default (§8).

**"A rule that holds in the data will hold in the future."**
With many items, high-lift rules appear by chance; validate before acting (§9).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the Apriori algorithm (frequent-itemset mining with
  anti-monotone pruning + rule generation with support/confidence/lift) in pure Python/NumPy, verified
  against brute-force enumeration (identical frequent itemsets, far fewer candidates counted). Five
  experiments: (1) the anti-monotone pruning removing most candidates; (2) **the confidence trap** —
  confidence ranks a trivial rule above a real one, lift fixes it; (3) `min_support` and the
  combinatorial explosion; (4) lift detecting negative association (lift < 1); (5) recovering a planted
  strong rule at high lift.
- **[exercises.md](exercises.md)** — prove the Apriori principle, derive lift, implement rule
  generation, reproduce every experiment.
- **[references.md](references.md)** — Agrawal & Srikant (Apriori), Han et al. (FP-Growth).

**This completes Part 4 — Unsupervised Learning.** From clustering (k-means, hierarchical, DBSCAN,
GMM, spectral) through dimensionality reduction (PCA, manifold learning) to anomaly detection and
association rules, you can now find structure in unlabeled data. **Next**: the labeled counterpart
resumes with [Part 7 — Deep Learning](../../07-deep-learning/), or fill in
[Part 2 — Data](../../02-data/) and [Part 1 — Python](../../01-python-for-ml/).
