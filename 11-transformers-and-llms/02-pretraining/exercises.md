# 11.02 — Exercises: Pretraining Paradigms

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain self-supervision: how a label is derived from unlabeled text, and why this made
pretraining scale to the internet.

**D2.** Write the attention masks for encoder, decoder, and prefix-LM and describe what each token can
attend to.

**D3.** Write the MLM objective and explain the 80/10/10 corruption split — what each part fixes.

**D4.** Write the causal-LM objective $-\sum_t \log P(x_t\mid x_{<t})$ and explain why it provides a
training signal at every position.

**D5.** Describe T5's span-corruption objective and how text-to-text unifies classification and
generation.

**D6.** Derive perplexity $\text{PPL} = \exp(\text{cross-entropy})$ and interpret it as an effective
branching factor.

**D7.** Explain why bidirectional context helps understanding tasks but makes generation impossible.

**D8.** Give three reasons decoder-only architectures came to dominate LLMs.

**D9.** Explain why NSP was dropped (RoBERTa) and what else RoBERTa changed.

**D10.** Compare encoder, decoder, and encoder-decoder on: generation ability, understanding ability,
and training sample-efficiency.

---

## Tier 2 — Implementation

**I1.** Reproduce Experiment 1: build the three attention masks and count visible positions.

**I2.** Implement MLM corruption (15% + 80/10/10) and reproduce Experiment 2's statistics.

**I3.** Reproduce Experiment 3: produce MLM, causal-LM, and span-corruption inputs/targets for a
sentence.

**I4.** Implement perplexity from a model's next-token distribution; reproduce Experiment 4.

**I5.** Reproduce Experiment 5: show bidirectional context solves a fill-in-the-blank that causal
cannot.

**I6.** Train a tiny decoder-only transformer with the causal-LM objective on a small corpus; measure
perplexity dropping.

**I7.** Train a tiny encoder with MLM on the same corpus; probe its representations on a classification
task.

**I8.** Implement a T5-style span-corruption data pipeline and encoder-decoder forward pass.

**I9.** Fine-tune a pretrained BERT and a pretrained GPT on a text-classification task and compare.

**I10.** *(Analysis.)* Compare sample efficiency of MLM (15% of tokens) vs causal LM (every token) by
measuring loss vs number of tokens seen.

---

## Tier 3 — Interview

**Q1.** What is self-supervised pretraining and why did it change NLP?

**Q2.** What is the difference between BERT, GPT, and T5?

**Q3.** What is masked language modeling, and why the 80/10/10 split?

**Q4.** What is causal language modeling?

**Q5.** Why can't BERT generate text?

**Q6.** What is perplexity and what does it measure?

**Q7.** What is span corruption / text-to-text?

**Q8.** When would you use an encoder vs a decoder vs an encoder-decoder?

**Q9.** Why do modern LLMs use decoder-only architectures?

**Q10.** What is the pretrain-then-fine-tune paradigm?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain self-supervision and why it scales
- [ ] Write the three attention masks and their consequences
- [ ] Derive MLM, causal-LM, and span-corruption objectives
- [ ] Explain the MLM 80/10/10 corruption split
- [ ] Compute and interpret perplexity
- [ ] Choose an architecture for a given task
- [ ] Explain why decoder-only won for LLMs
