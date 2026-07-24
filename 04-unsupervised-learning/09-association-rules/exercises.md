# 04.09 — Exercises: Association Rule Mining

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Define support, confidence, and lift for a rule $X \Rightarrow Y$, and express each in terms
of probabilities.

**D2.** Prove the Apriori principle: $\mathrm{support}(X \cup \lbrace i\rbrace) \le \mathrm{support}(X)$,
and hence a superset of an infrequent itemset is infrequent.

**D3.** Show that lift $= 1$ corresponds to independence of $X$ and $Y$, lift $> 1$ to positive
association, lift $< 1$ to negative.

**D4.** Explain the confidence trap: show that if $\hat P(Y)$ is large, confidence of $X\Rightarrow Y$
is large for *any* $X$, and that lift removes this artifact.

**D5.** Derive the candidate-generation and pruning steps of Apriori and explain why pruning uses all
$(k{-}1)$-subsets.

**D6.** Explain the anti-monotone shortcut in *rule* generation: if $X\Rightarrow Y$ fails confidence,
which related rules can be pruned?

**D7.** Define leverage and conviction and explain what each adds over lift.

**D8.** Explain closed and maximal frequent itemsets and how they reduce redundancy.

**D9.** Explain why FP-Growth avoids candidate generation and needs only two data passes.

**D10.** Explain why high-lift rules can be spurious with many items, and how to guard against it.

---

## Tier 2 — Implementation

**I1.** Implement Apriori (frequent-itemset mining with anti-monotone pruning). Verify the frequent
itemsets exactly against brute-force enumeration.

**I2.** Implement rule generation with support, confidence, and lift. Reproduce Experiment 2's
confidence trap.

**I3.** Reproduce Experiment 1: count candidates per level and show the pruning removing most of them.

**I4.** Reproduce Experiment 3: sweep `min_support` and show the number of frequent itemsets growing.

**I5.** Reproduce Experiment 4: build substitute items and show lift < 1.

**I6.** Reproduce Experiment 5: plant a strong rule and recover it, ranked by lift.

**I7.** Implement FP-Growth (FP-tree + recursive conditional mining) and compare its speed to Apriori
on dense data.

**I8.** Add leverage and conviction to the rule output and compare the rankings to lift.

**I9.** Mine closed / maximal frequent itemsets and show the reduction in redundant rules.

**I10.** *(Real data.)* Run Apriori on a real market-basket dataset (e.g. Online Retail or Groceries)
and report the top-10 rules by lift.

---

## Tier 3 — Interview

**Q1.** What is association rule mining?

**Q2.** Define support, confidence, and lift.

**Q3.** Why isn't high confidence enough?

**Q4.** What is the Apriori principle and why does it matter?

**Q5.** Walk through the Apriori algorithm.

**Q6.** What does lift < 1 mean?

**Q7.** Apriori vs FP-Growth — which and why?

**Q8.** How do you choose `min_support` and `min_confidence`?

**Q9.** Do association rules imply causation?

**Q10.** How do you avoid redundant or spurious rules?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Compute support, confidence, and lift by hand
- [ ] Prove and apply the Apriori principle
- [ ] Explain and avoid the confidence trap using lift
- [ ] Implement Apriori and verify against brute force
- [ ] Interpret lift > 1, = 1, < 1 correctly
- [ ] Choose Apriori vs FP-Growth and set thresholds sensibly
