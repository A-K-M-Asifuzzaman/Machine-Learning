# 11.02 — Pretraining Paradigms

> **The transformer block is fixed; what makes BERT, GPT, and T5 different is one attention mask and one
> self-supervised objective.** Pretraining is the idea that turned NLP upside down: instead of training
> a model per task on scarce labels, train *one* model on the whole internet with a task that needs no
> labels — predict masked words, or the next word — then adapt it. This chapter builds the three
> paradigms, shows that they differ only in masking and objective, and measures why each suits the tasks
> it does.

Before 2018, each NLP task trained from scratch on its own labeled data ([10.04](../../10-nlp/04-nlp-tasks/)).
Pretraining ([08.03](../../08-computer-vision/03-transfer-learning/) for the vision analogue) changed the
recipe to **pretrain once, fine-tune everywhere** — and self-supervision made the pretraining data
effectively unlimited.

## Table of contents

1. [Self-supervision: labels for free](#1-self-supervision-labels-for-free)
2. [Three architectures, one block](#2-three-architectures-one-block)
3. [BERT: masked language modeling](#3-bert-masked-language-modeling)
4. [GPT: causal language modeling](#4-gpt-causal-language-modeling)
5. [T5: span corruption and text-to-text](#5-t5-span-corruption-and-text-to-text)
6. [Perplexity](#6-perplexity)
7. [Why decoder-only won](#7-why-decoder-only-won)
8. [Common misconceptions](#8-common-misconceptions)

## 1. Self-supervision: labels for free

The breakthrough insight: **text is its own supervision.** Hide part of a sentence and predict it from
the rest — no human annotation needed, so the training set is the entire web. This is *self-supervised*
learning: the label is derived from the input. Experiment 3 shows three ways to carve a label out of the
sentence "the cat sat on the mat":

| Objective | Input | Target |
|---|---|---|
| MLM (BERT) | the `[MASK]` sat on the `[MASK]` | predict positions 1, 5 → cat, mat |
| Causal LM (GPT) | the cat sat on the | cat sat on the mat (shift by one) |
| Span corruption (T5) | the `<X>` on `<Y>` | `<X>` cat sat `<Y>` the mat |

All three need zero labels, which is exactly why pretraining scaled to internet-sized corpora.

## 2. Three architectures, one block

The *only* architectural difference between the paradigms is the **attention mask**. Experiment 1 counts
how many positions each token can attend to in a length-6 sequence:

| Model | Visible-per-position | Meaning |
|---|---|---|
| **Encoder (BERT)** | [6, 6, 6, 6, 6, 6] | bidirectional — every token sees all |
| **Decoder (GPT)** | [1, 2, 3, 4, 5, 6] | causal — token $i$ sees only 0..$i$ |
| **Prefix-LM (T5-ish)** | [3, 3, 3, 4, 5, 6] | bidirectional over the input, causal over the output |

- **Encoder-only** (BERT) — no mask; full bidirectional context. Best for **understanding**
  (classification, NER, QA). Cannot generate.
- **Decoder-only** (GPT) — causal mask; each token predicts the next from the left. Best for
  **generation**. This is the LLM architecture.
- **Encoder–decoder** (T5, BART, original transformer) — a bidirectional encoder feeds a causal decoder
  via cross-attention. Natural for **seq2seq** (translation, summarization).

Same block ([11.01](../01-transformer/)), three models — the mask is the fork in the road.

## 3. BERT: masked language modeling

BERT (Devlin et al., 2018) is an **encoder** pretrained by **masked language modeling (MLM)**: mask 15%
of tokens and predict them from *bidirectional* context. There is a subtlety in the corruption, which
Experiment 2 reproduces on 10,000 tokens (14.7% chosen; of those 80% → `[MASK]`, 11% → random, 9% →
unchanged):

| Of the 15% chosen | Action | Why |
|---|---|---|
| 80% | replace with `[MASK]` | the standard fill-in-the-blank |
| 10% | replace with a **random** word | force robust representations |
| 10% | leave **unchanged** (still predicted) | close the train/test gap |

The 80/10/10 split exists because `[MASK]` **never appears during fine-tuning**. If the model only ever
saw `[MASK]`, it would build features that vanish downstream; the random/keep tokens force it to
represent *every* token well. (Original BERT also had a Next-Sentence-Prediction objective, later found
unnecessary and dropped by RoBERTa.) MLM's payoff — bidirectional context — is exactly why BERT
dominates understanding tasks (§7, Experiment 5).

## 4. GPT: causal language modeling

GPT (Radford et al., 2018) is a **decoder** pretrained by **causal language modeling (CLM)**: predict
the next token given all previous tokens, with a causal mask so the model can't cheat by looking ahead.
The objective is simply

$$
\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t}),
$$

the cross-entropy of next-token prediction over the corpus. This one objective is deceptively powerful:
to predict the next token well, the model must learn grammar, facts, reasoning, and style — and because
it is *generative*, the same pretrained model can produce text, which is what makes it a chatbot after
alignment ([11.06](../06-alignment/)). Every modern LLM (GPT-4, Llama, Claude) is a causal decoder.

## 5. T5: span corruption and text-to-text

T5 (Raffel et al., 2020) is an **encoder–decoder** that frames **every** task as text-to-text — input
text in, output text out, including classification ("cola sentence: … → acceptable"). It pretrains with
**span corruption**: drop random *spans* (not single tokens), replace each with a sentinel, and have the
decoder generate the dropped spans (Experiment 3). This unifies BERT's masking with GPT's generation and
handles seq2seq tasks natively. BART is a close relative (corrupt then reconstruct the full text).

## 6. Perplexity

The intrinsic metric for a language model is **perplexity** — the exponential of the cross-entropy loss,
interpretable as the number of equally-likely words the model is "choosing between" per token.
Experiment 4 (vocabulary 100):

| Model assigns p(correct word) | Perplexity |
|:--:|:--:|
| 1.000 (perfect) | 1.0 |
| 0.500 | 2.0 |
| 0.100 | 10.0 |
| 0.010 (uniform over 100) | 100.0 |

$$
\text{PPL} = \exp\!\Big(\underbrace{-\tfrac{1}{N}\textstyle\sum_t \log P(x_t \mid x_{<t})}_{\text{cross-entropy}}\Big).
$$

A perfect model has perplexity 1; a model that guesses uniformly over $V$ words has perplexity $V$ (as
confused as a $V$-way coin flip). Halving perplexity means the model is twice as certain. Language-model
progress (GPT-2 → GPT-3) was largely a perplexity race, and perplexity is what the scaling laws
([11.04](../04-scaling-and-architecture/)) predict from model and data size.

## 7. Why decoder-only won

BERT's bidirectionality is a real advantage for *understanding*. Experiment 5 masks the subject in "the
`[SUBJ]` barked loudly", where the *following* verb identifies it:

| Context available | Accuracy |
|---|:--:|
| left-only (causal) "the ___" | 0.349 (≈ chance) |
| both-sides (bidirectional) "the ___ barked" | **1.000** |

So why do LLMs use decoder-only? Three reasons: (1) **generation** requires causality — you can't
generate the future you're conditioning on; (2) causal LM gives a training signal at **every** position
(predict each next token), where MLM only trains on the 15% masked, making CLM more sample-efficient at
scale; (3) with enough scale, **in-context learning** (§ next chapters) lets a decoder do understanding
tasks *without* fine-tuning, erasing BERT's edge. The field converged on decoder-only LLMs — but encoders
remain the right tool for embeddings and pure-classification pipelines.

## 8. Common misconceptions

- **"BERT and GPT are different architectures."** They are the *same* transformer block; the difference
  is the attention mask and the objective (§2).
- **"Pretraining needs labeled data."** It is *self*-supervised — the label comes from the text itself
  (§1).
- **"MLM just inserts [MASK]."** 20% of chosen tokens are random-or-kept to close the train/test gap
  (§3).
- **"Lower loss and lower perplexity are different metrics."** Perplexity *is* $\exp(\text{loss})$ — a
  monotone rescaling for interpretability (§6).
- **"Decoder-only won because it's better at everything."** It won for generation, per-token training
  signal, and in-context learning at scale; encoders can still beat it on fixed understanding tasks
  (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the masks, objectives, and metrics in Python. Five
  experiments: (1) encoder/decoder/prefix mask patterns; (2) BERT's 15% + 80/10/10 MLM corruption;
  (3) MLM vs causal-LM vs span-corruption labels on one sentence; (4) perplexity from cross-entropy;
  (5) bidirectional context solving a fill-in-the-blank that causal cannot.
- **[exercises.md](exercises.md)** — derive the objectives, implement MLM corruption, reason about
  architecture choice.
- **[references.md](references.md)** — BERT, GPT, T5, RoBERTa, and the pretraining literature.

## Where this leads

- **The transformer block underneath** → [11.01](../01-transformer/)
- **Efficient attention for long-context pretraining** → [11.03](../03-efficient-attention/)
- **Scaling laws that predict pretraining loss** → [11.04](../04-scaling-and-architecture/)
- **Fine-tuning and adapting the pretrained model** → [11.05](../05-adaptation/)
- **The transfer-learning analogue in vision** → [08.03](../../08-computer-vision/03-transfer-learning/)
