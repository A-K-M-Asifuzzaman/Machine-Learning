# 10.01 — Exercises: Text Preprocessing & Tokenization

Three tiers. **Derivation/reasoning** (paper), **implementation** (code), **interview** (explain out
loud).

---

## Tier 1 — Derivation & reasoning

**D1.** Describe the full preprocessing pipeline (normalization → numericalization) and what each step
does. Which steps are unnecessary for subword models, and why?

**D2.** Compare word, character, and subword tokenization on three axes: vocabulary size, sequence
length, and OOV behavior. State when each is preferable.

**D3.** Write the BPE training algorithm in pseudocode and argue it terminates and is deterministic
(given a tie-break rule).

**D4.** Explain why BPE can represent *any* string and therefore needs no `<UNK>` token. What is the
worst-case tokenization of a novel string?

**D5.** Contrast BPE (frequency-based merges) with WordPiece (likelihood-based merges). Write the
WordPiece merge criterion.

**D6.** Explain the Unigram LM tokenizer: how it starts large and prunes, and why it can produce
multiple segmentations with probabilities.

**D7.** Explain byte-level BPE and why operating on bytes guarantees no unknown token for any input.

**D8.** Derive the ~4-characters-per-token rule of thumb for English and explain why it varies by
language.

**D9.** Explain how tokenization can cause LLM arithmetic errors, and what digit-splitting fixes.

**D10.** Explain why swapping a pretrained model's tokenizer requires retraining.

---

## Tier 2 — Implementation

**I1.** Implement BPE training (`get_pair_counts`, `merge`, `train`) and reproduce the textbook merges
(Experiment 1).

**I2.** Implement BPE encoding (apply merges by rank) and decoding; verify round-trip on unseen words
(Experiment 2).

**I3.** Reproduce Experiment 3: measure vocab size and sequence length for word/char/subword on a
corpus.

**I4.** Reproduce Experiment 4: sweep the merge count and plot vocab size and tokens/word.

**I5.** Compare your BPE to Hugging Face `tokenizers` trained on the same corpus; explain any
differences (tie-breaking, pre-tokenization).

**I6.** Implement WordPiece training (likelihood-based merges) and compare its vocabulary to BPE's.

**I7.** Use `tiktoken` / `sentencepiece` to tokenize a paragraph in several languages; measure
tokens-per-character per language and discuss fairness.

**I8.** Implement byte-level pre-processing (map text to bytes) and run BPE on bytes; confirm it handles
emoji and non-ASCII.

**I9.** Implement subword regularization (sample among Unigram segmentations) and discuss its use as
data augmentation.

**I10.** *(Analysis.)* Measure how many tokens common number formats ("2024", "3.14", "1,000,000") take
in GPT-2 vs a digit-splitting tokenizer.

---

## Tier 3 — Interview

**Q1.** Why don't we just tokenize on words?

**Q2.** What is Byte-Pair Encoding and how does it work?

**Q3.** How does BPE handle a word it has never seen?

**Q4.** What is the trade-off between vocabulary size and sequence length?

**Q5.** What is the difference between BPE and WordPiece?

**Q6.** What is byte-level BPE and why is it used?

**Q7.** What does SentencePiece add over BPE?

**Q8.** Why can tokenization make LLMs bad at arithmetic?

**Q9.** Roughly how many characters are in a token, and why does it vary by language?

**Q10.** Can you change a model's tokenizer after training?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Describe the preprocessing pipeline end to end
- [ ] Compare word/char/subword tokenization on vocab, length, and OOV
- [ ] Implement BPE training and encoding from scratch
- [ ] Explain the open-vocabulary (no-UNK) guarantee
- [ ] Distinguish BPE, WordPiece, Unigram, and SentencePiece
- [ ] Explain byte-level BPE
- [ ] Reason about token counts, cost, and cross-language fairness
