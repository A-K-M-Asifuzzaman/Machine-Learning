# 10.04 — Exercises: NLP Tasks & Metrics

Three tiers. **Reasoning** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Reasoning

**D1.** List the five NLP task shapes and give two example tasks, the output head, and the metric for
each.

**D2.** Explain the BIO tagging scheme and how entity spans are recovered from per-token tags. What does
`I-PER` after `O` mean?

**D3.** Show, with an example, why token accuracy overstates NER quality and why entity-level F1 is the
honest metric.

**D4.** Define SQuAD Exact-Match and token-F1. Give a case where they diverge and explain why both are
reported.

**D5.** Derive BLEU: clipped n-gram precision, the geometric mean over 1–4-grams, and the brevity
penalty. Explain what the penalty guards against.

**D6.** Derive ROUGE-N (recall) and ROUGE-L (LCS). Explain why summarization is recall-oriented while
translation is precision-oriented.

**D7.** Explain why n-gram metrics (BLEU/ROUGE) correlate imperfectly with human judgment, and what
BERTScore / LLM-as-judge add.

**D8.** Derive micro-F1 and macro-F1 and show why they diverge under class imbalance.

**D9.** Explain the NLI task (entailment/contradiction/neutral) and how the two sentences are encoded
jointly.

**D10.** Given a new NLP problem, describe how you would (a) frame it as one of the five shapes and
(b) choose its metric.

---

## Tier 2 — Implementation

**I1.** Implement `spans_from_bio` and entity-level precision/recall/F1; reproduce Experiment 1.

**I2.** Implement SQuAD EM and token-F1 with answer normalization; reproduce Experiment 2.

**I3.** Implement BLEU (clipped precision + brevity penalty) and verify against `nltk`
(Experiment 3).

**I4.** Implement ROUGE-N and ROUGE-L (LCS via dynamic programming); reproduce Experiment 4.

**I5.** Implement micro and macro F1 and reproduce Experiment 5 on imbalanced classes.

**I6.** Add corpus-level BLEU (aggregate counts across sentences) and compare to averaging sentence
BLEU.

**I7.** Implement BERTScore (cosine similarity of contextual embeddings) and compare its ranking to
ROUGE on a few summaries.

**I8.** Build a small NER tagger (features + a per-token classifier) and evaluate it with your
entity-F1.

**I9.** Build a text classifier and report accuracy, micro-F1, and macro-F1; show how they differ on an
imbalanced split.

**I10.** *(Judgment.)* Design a small LLM-as-judge rubric for summarization and compare its rankings to
ROUGE.

---

## Tier 3 — Interview

**Q1.** What are the main NLP task types?

**Q2.** How is NER evaluated, and why not token accuracy?

**Q3.** What is the BIO tagging scheme?

**Q4.** What metrics does SQuAD use and why both?

**Q5.** How does BLEU work, and what is the brevity penalty for?

**Q6.** What is ROUGE and why is summarization recall-oriented?

**Q7.** What are the limitations of BLEU/ROUGE?

**Q8.** What is the difference between micro-F1 and macro-F1?

**Q9.** When would you report macro-F1 over accuracy?

**Q10.** How would you evaluate an open-ended generation task today?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Map a new problem to one of the five task shapes
- [ ] Implement entity-level F1 and explain why token accuracy misleads
- [ ] Implement SQuAD EM and token-F1
- [ ] Implement and explain BLEU and ROUGE
- [ ] Explain micro vs macro F1 under imbalance
- [ ] Choose the correct metric for a given task
- [ ] State the limits of n-gram metrics and the alternatives
