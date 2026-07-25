# 10.04 — NLP Tasks & Metrics

> **Nearly every NLP problem is one of five task shapes — and each has a metric that will lie to you if
> you pick the wrong one.** Classify a sequence, tag every token, extract a span, generate text,
> compare a pair: that taxonomy covers sentiment, NER, question answering, translation, summarization,
> and inference. This chapter maps the tasks to their formulations and builds the standard metrics from
> scratch — because on real NLP the metric is where models are won and lost.

A model architecture ([Part 11](../../11-transformers-and-llms/)) is reusable across tasks; what changes is
the **output head** and the **metric**. Understanding the task shapes and their metrics is what lets you
frame a new problem correctly and evaluate it honestly.

## Table of contents

1. [The five task shapes](#1-the-five-task-shapes)
2. [Sequence classification](#2-sequence-classification)
3. [Token classification: NER and POS](#3-token-classification-ner-and-pos)
4. [Span extraction: question answering](#4-span-extraction-question-answering)
5. [Generation: translation and summarization](#5-generation-translation-and-summarization)
6. [Pairwise tasks and imbalanced metrics](#6-pairwise-tasks-and-imbalanced-metrics)
7. [Choosing the right metric](#7-choosing-the-right-metric)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The five task shapes

| Shape | Input → output | Tasks | Head | Metric |
|---|---|---|---|---|
| **Sequence classification** | text → 1 label | sentiment, topic, spam | pooled + linear | accuracy, F1 |
| **Token classification** | text → 1 label/token | NER, POS, chunking | per-token linear | entity-level F1 |
| **Span extraction** | (question, context) → span | extractive QA | start/end pointers | EM, token-F1 |
| **Sequence generation** | text → text | translation, summarization | autoregressive decoder | BLEU, ROUGE |
| **Pairwise** | (text A, text B) → label | NLI, similarity, paraphrase | joint encode + linear | accuracy, F1 |

All ride on the same backbone; the head and the loss change. The metrics are where the subtlety lives.

## 2. Sequence classification

The simplest shape: map a whole sequence to one label. **Sentiment** (positive/negative),
**topic classification**, **spam detection**, **language identification**. The head pools the token
representations (a `[CLS]` token or a mean) into one vector and applies a linear classifier. Metrics are
the standard classification metrics ([05.03](../../05-model-evaluation/03-classification-metrics/)):
accuracy when balanced, **F1** when not (§6).

## 3. Token classification: NER and POS

Assign a label to **every token**: **named-entity recognition** (person/location/organization),
**part-of-speech tagging**, chunking. Multi-token entities are encoded with the **BIO scheme** —
`B-PER` begins a person entity, `I-PER` continues it, `O` is outside — so spans are recoverable from
per-token tags.

The metric matters enormously here. Token accuracy is **misleading** because a near-miss span is a
*wrong entity*. Experiment 1 tags "Barack Obama visited New York" but mislabels "Obama" as `O`:

| Metric | Value |
|---|:--:|
| token accuracy | 0.800 (4/5 tokens correct) |
| **entity-level F1** | **0.500** |

Token accuracy looks great (80%), but the model got only **1 of 2 entities** right — "Barack" is not
"Barack Obama". Entity-level F1 counts a prediction correct only if the **entire span and type match**,
so it drops to 0.50. NER is *always* scored at the entity level (the `seqeval`/CoNLL standard), never
token accuracy, because partial spans are failures.

## 4. Span extraction: question answering

**Extractive QA** (SQuAD) gives a question and a context paragraph and asks for the **answer span**
within the context — the head predicts a start and an end token. Two metrics, always reported together:

- **Exact Match (EM)** — 1 if the predicted string equals the gold answer (after normalization), else 0.
- **Token-F1** — the token overlap F1 between prediction and gold, giving **partial credit**.

Experiment 2 (gold = "the Eiffel Tower"):

| Prediction | EM | F1 |
|---|:--:|:--:|
| the Eiffel Tower | 1 | 1.00 |
| Eiffel Tower (dropped article) | 0 | 0.80 |
| the Eiffel Tower in Paris (extra) | 0 | 0.75 |
| the Louvre (wrong) | 0 | 0.40 |

EM is all-or-nothing and would score the essentially-correct "Eiffel Tower" as 0; token-F1 rewards the
overlap. Reporting only EM massively understates a model, so SQuAD reports both. (Generative /
open-book QA instead uses generation metrics or human/LLM judgment.)

## 5. Generation: translation and summarization

When the output is free text, exact match is hopeless (many valid outputs), so metrics measure
**n-gram overlap** with reference texts — and the direction of overlap distinguishes the two tasks:

- **BLEU** (translation) — n-gram **precision**: what fraction of the *hypothesis's* n-grams appear in
  the reference, as a geometric mean over 1–4-grams, times a **brevity penalty**. The penalty is
  essential: without it a system could score high by emitting a tiny high-precision fragment.
  Experiment 3 (verified equal to `nltk`):

  | Hypothesis | BLEU |
  |---|:--:|
  | perfect | 1.0000 |
  | one word off | 0.5969 |
  | "the cat" (too short, gamed) | **0.0000** |
  | reordered | 0.6148 |

  The brevity penalty crushes the too-short output to 0.

- **ROUGE** (summarization) — n-gram **recall**: what fraction of the *reference's* n-grams the summary
  covers (the mirror of BLEU). Summarization is recall-oriented because a good summary must **cover**
  the key content. ROUGE-1/2 use unigram/bigram overlap; **ROUGE-L** uses the longest common
  subsequence. Experiment 4:

  | Summary | ROUGE-1 | ROUGE-2 | ROUGE-L |
  |---|:--:|:--:|:--:|
  | full match | 1.00 | 1.00 | 1.00 |
  | good summary | 0.78 | 0.50 | 0.78 |
  | misses content | 0.22 | 0.00 | 0.22 |

**Both are flawed** — they reward surface overlap, not meaning ("great" vs "excellent" scores 0), and
correlate only loosely with human judgment. Modern evaluation adds embedding-based metrics (BERTScore),
and increasingly **LLM-as-judge** and human evaluation.

## 6. Pairwise tasks and imbalanced metrics

**Pairwise** tasks take two texts: **natural language inference** (does A entail, contradict, or stay
neutral to B?), **semantic similarity**, **paraphrase detection**. The two texts are encoded jointly
(concatenated with a separator) and classified — NLI is the classic 3-way accuracy task (SNLI, MNLI).

Most NLP classification is **imbalanced**, which makes the choice of F1 averaging decisive. Experiment 5
scores a model that always predicts the majority class (900 A's, 50 B's, 50 C's):

| Averaging | F1 |
|---|:--:|
| micro-F1 (pool all predictions) | **0.90** |
| macro-F1 (average per-class) | **0.32** |

Micro-F1 pools everything, so the 900 easy A's swamp the rare classes and it looks excellent (0.90).
Macro-F1 averages the per-class F1s equally, so failing on B and C (F1 = 0) crushes it to 0.32. **On
imbalanced tasks, report macro-F1** — it refuses to hide failure on the classes you care about.

## 7. Choosing the right metric

The through-line of this chapter: **the metric encodes what you actually want.**

- Balanced classification → accuracy; imbalanced → **macro-F1** (§6).
- NER/tagging → **entity-level F1**, never token accuracy (§3).
- Extractive QA → **EM and token-F1** together (§4).
- Translation → **BLEU** (precision + brevity); summarization → **ROUGE** (recall) (§5).
- Anything generative, for real quality → add **human or LLM-as-judge** evaluation; n-gram metrics are
  a proxy, not the goal.

Optimizing a proxy metric that diverges from your true objective (token accuracy for NER, EM alone for
QA, BLEU alone for open generation) is one of the most common and costly mistakes in applied NLP.

## 8. Common misconceptions

- **"High token accuracy means good NER."** No — a near-miss span is a wrong entity; use entity-level
  F1 (§3).
- **"BLEU/ROUGE measure quality."** They measure n-gram overlap, which correlates loosely with quality
  and ignores meaning and fluency (§5).
- **"Exact Match is the QA metric."** EM alone understates near-correct answers; report token-F1 too
  (§4).
- **"Accuracy is fine for classification."** Only when balanced; on imbalanced tasks it (and micro-F1)
  hides minority-class failure — use macro-F1 (§6).
- **"One backbone can't do all these tasks."** It can — the same transformer with different heads and
  losses handles every shape (§1); that is the premise of Part 11.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the standard NLP metrics in Python. Five experiments:
  (1) NER entity-level F1 vs the misleading token accuracy; (2) SQuAD EM vs token-F1; (3) BLEU verified
  equal to `nltk`, including the brevity-penalty guard; (4) ROUGE-N and ROUGE-L; (5) micro vs macro F1
  on imbalanced classes.
- **[exercises.md](exercises.md)** — implement each metric, reason about task formulations and metric
  pitfalls.
- **[references.md](references.md)** — the benchmark and metric papers (SQuAD, GLUE, BLEU, ROUGE, CoNLL).

## Where this leads

- **The transformers that solve these tasks** → [Part 11](../../11-transformers-and-llms/)
- **Classification metrics in general** → [05.03](../../05-model-evaluation/03-classification-metrics/)
- **Embeddings and features feeding these tasks** → [10.02](../02-classical-representations/), [10.03](../03-word-embeddings/)
- **Decoding for generation tasks** → [11.07](../../11-transformers-and-llms/07-inference/)
- **Evaluating LLMs (benchmarks, LLM-as-judge)** → [11.08](../../11-transformers-and-llms/08-rag-and-agents/)
