# 04.09 — References: Association Rule Mining

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§3 | Rules, support, confidence, lift | Agrawal, Imieliński & Swami (1993); Tan et al. (2005) |
| §4-§5 | Apriori principle & algorithm | Agrawal & Srikant (1994) |
| §7 | Lift, interestingness measures | Brin et al. (1997); Tan, Kumar & Srivastava (2004) |
| §8 | FP-Growth, Eclat | Han, Pei & Yin (2000); Zaki (2000) |
| §9 | Closed/maximal itemsets, redundancy | Pasquier et al. (1999) |

---

## Books

**Tan, P.-N., Steinbach, M. & Kumar, V. (2005). *Introduction to Data Mining*.** — **the standard
textbook treatment.** Chapter 6 "Association Analysis" covers support/confidence/lift (§2-§3), the
Apriori algorithm and principle (§4-§5), FP-Growth (§8), and interestingness measures / the confidence
trap (§7) in depth. The reference for this chapter.

**Han, J., Kamber, M. & Pei, J. (2011). *Data Mining: Concepts and Techniques*, 3rd ed.** Chapters 6-7
on frequent patterns and associations, by the authors of FP-Growth; excellent on the algorithms (§5,
§8) and pattern evaluation (§7).

**Aggarwal, C. C. (2015). *Data Mining: The Textbook*.** Chapter 4-5 on frequent pattern mining, with
the modern algorithmic variants.

---

## Papers

- **Agrawal, R., Imieliński, T. & Swami, A. (1993).** "Mining Association Rules between Sets of Items
  in Large Databases." *SIGMOD*. — **the paper that introduced association rule mining** and
  support/confidence (§1-§3).
- **Agrawal, R. & Srikant, R. (1994).** "Fast Algorithms for Mining Association Rules." *VLDB*. —
  **the Apriori algorithm and the anti-monotone principle** (§4-§5). One of the most-cited data-mining
  papers. Free at <https://www.vldb.org/conf/1994/P487.PDF>.
- **Brin, S., Motwani, R., Ullman, J. D. & Tsur, S. (1997).** "Dynamic Itemset Counting and Implication
  Rules for Market Basket Data." *SIGMOD*. — introduces **lift** (as "interest") and the case against
  confidence alone (§7).
- **Han, J., Pei, J. & Yin, Y. (2000).** "Mining Frequent Patterns without Candidate Generation."
  *SIGMOD*. — **FP-Growth** (§8): the FP-tree and candidate-free mining. Free at
  <https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf>.
- **Zaki, M. J. (2000).** "Scalable Algorithms for Association Mining." *IEEE TKDE* 12(3). — **Eclat**
  (§8): vertical tid-lists and set intersection.
- **Pasquier, N., Bastide, Y., Taouil, R. & Lakhal, L. (1999).** "Discovering Frequent Closed Itemsets
  for Association Rules." *ICDT*. — **closed frequent itemsets** for reducing redundancy (§9).
- **Tan, P.-N., Kumar, V. & Srivastava, J. (2004).** "Selecting the right objective measure for
  association analysis." *Information Systems* 29(4). — a systematic comparison of interestingness
  measures (lift, leverage, conviction, etc.) (§7).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`mlxtend.frequent_patterns`](https://github.com/rasbt/mlxtend/tree/master/mlxtend/frequent_patterns) | `apriori`, `fpgrowth`, `association_rules` (support/confidence/lift/leverage/conviction) — the standard Python implementation |
| [`efficient-apriori`](https://github.com/tommyod/Efficient-Apriori) | a fast, clean pure-Python Apriori |
| [SPMF](https://www.philippe-fournier-viger.com/spmf/) | a large Java library of frequent-pattern and rule-mining algorithms (Apriori, FP-Growth, Eclat, closed/maximal, and many more) |

---

## Deferred to later chapters

- **Recommender systems — the applied cousin of market-basket analysis** → [16.xx recommenders]
- **Sequential pattern mining (order matters)** → [15.xx time series]
- **Graph / subgraph mining** → [14.xx graph ML]
- **The probability foundations (conditional probability, independence)** → [00.03](../../00-mathematical-foundations/03-probability/)
