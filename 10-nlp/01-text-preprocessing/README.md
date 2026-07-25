# 10.01 — Text Preprocessing & Tokenization

> **A language model never sees text — it sees integers.** Tokenization is the bridge: the choice of
> how to chop a string into a finite vocabulary of tokens. Too coarse (whole words) and you get a
> huge vocabulary that still chokes on any unseen word; too fine (characters) and sequences become
> punishingly long. **Byte-Pair Encoding** finds the middle — a bounded vocabulary of subwords that can
> spell *any* string — and it is why every modern LLM tokenizes the way it does. This chapter builds
> BPE from scratch and measures the trade-off it resolves.

Everything downstream — embeddings ([10.03](../03-word-embeddings/)), transformers
([Part 11](../../11-transformers-llms/)) — operates on token IDs. Tokenization decides what the model
*can* represent and how long its sequences are, so it is the quiet foundation of all of NLP.

## Table of contents

1. [The preprocessing pipeline](#1-the-preprocessing-pipeline)
2. [The granularity question](#2-the-granularity-question)
3. [Byte-Pair Encoding](#3-byte-pair-encoding)
4. [No unknown token: the open vocabulary](#4-no-unknown-token-the-open-vocabulary)
5. [WordPiece, Unigram, SentencePiece](#5-wordpiece-unigram-sentencepiece)
6. [Byte-level BPE and real tokenizers](#6-byte-level-bpe-and-real-tokenizers)
7. [Practical notes](#7-practical-notes)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The preprocessing pipeline

Turning raw text into model input is a short pipeline:

1. **Normalization** — Unicode normalization (NFC/NFKC), optional lowercasing, accent stripping,
   whitespace cleanup. Modern tokenizers normalize *lightly* — aggressive lowercasing loses
   information ("US" vs "us").
2. **Pre-tokenization** — a rough split (on whitespace/punctuation) that bounds where tokens can begin.
3. **Tokenization** — map the pieces to a finite vocabulary of tokens (the subject of this chapter).
4. **Numericalization** — replace each token with its integer ID.
5. **Special tokens** — add `[CLS]`, `[SEP]`, `<bos>`, `<eos>`, `<pad>` as the model requires.

Classical NLP also did **stemming** and **lemmatization** (reducing "running"→"run") — useful for
bag-of-words ([10.02](../02-classical-representations/)) but unnecessary for subword models, which see
the shared substrings anyway.

## 2. The granularity question

The central design choice is the *unit*. Experiment 3 measures all three on a 600-word corpus:

| Tokenization | Vocab size | Sequence length | Tokens/word |
|---|---:|---:|---:|
| word-level | 170 | 600 | 1.00 |
| character-level | 22 | 3,689 | 6.15 |
| **subword (BPE)** | **43** | **1,483** | **2.47** |

The trade-off is stark:

- **Word-level** — shortest sequences (1 token/word), but a **huge, open-ended vocabulary** and the
  fatal **out-of-vocabulary (OOV)** problem: any word not in the vocab becomes a single `<UNK>`,
  destroying its content.
- **Character-level** — a **tiny vocabulary** (~ the alphabet) and no OOV, but **very long sequences**
  (6× here), which are slow and hard to model (long-range dependencies, [09.01](../../09-sequence-models/01-rnn/)).
- **Subword** — engineered to sit between: a **bounded vocabulary** (43 ≪ 170) *and* a **modest
  sequence length** (between char and word). The best of both, which is why it won.

## 3. Byte-Pair Encoding

BPE (Sennrich et al., 2016, adapting a 1994 compression algorithm) *learns* the subword vocabulary
from data. The algorithm:

1. Start with every word as a sequence of **characters** (plus an end-of-word marker `</w>`).
2. Count all adjacent symbol **pairs** across the corpus, weighted by word frequency.
3. **Merge** the most frequent pair into a single new symbol.
4. Repeat for a fixed number of merges.

Experiment 1 runs it on `{low, lower, newest, widest}` and reproduces the canonical bottom-up
construction:

$$
\texttt{s}+\texttt{t}\to\texttt{st}, \quad \texttt{e}+\texttt{st}\to\texttt{est}, \quad \texttt{est}+\texttt{</w>}\to\texttt{est</w>}, \quad \dots, \quad \texttt{ne}+\texttt{west</w>}\to\texttt{newest</w>}.
$$

Frequent substrings (common endings, then whole frequent words) become single tokens; rare strings
stay split into pieces. **The ordered list of merges *is* the tokenizer** — to encode a new word, apply
the learned merges in the order they were discovered. The **number of merges is a dial**: Experiment 4
shows tokens/word falling from 7.1 (characters) → 2.9 (50 merges) → 1.1 (200 merges) as the vocabulary
grows from 23 → 154 tokens. Real tokenizers pick a target vocabulary size (GPT-2: 50,257) balancing a
manageable embedding table against short sequences.

## 4. No unknown token: the open vocabulary

The property that made subwords universal: **BPE can represent any string.** Experiment 2 trains on
`{low, lower, lowest, newest, slow, slower, wider, widest}` and tokenizes words never seen:

| Unseen word | BPE tokens | Round-trips? |
|---|---|:--:|
| slowest | `s low est_` | ✓ |
| newer | `ne w er_` | ✓ |
| wildest | `wi l d est_` | ✓ |

A word-level tokenizer would emit `<UNK>` for each and lose them entirely. BPE instead falls back to
the subwords — and, in the worst case, individual characters — it *does* know, so **every string is
representable and decoding is exact** (all round-trip). This open-vocabulary guarantee is why subword
tokenization replaced word-level everywhere.

## 5. WordPiece, Unigram, SentencePiece

BPE has close relatives that share the subword idea with different merge criteria:

- **WordPiece** (BERT) — like BPE, but merges the pair that most increases the *likelihood* of the
  training data (a pointwise-mutual-information-style score), not raw frequency. Marks continuations
  with `##`.
- **Unigram LM** (used in SentencePiece) — starts with a large vocabulary and *prunes* tokens to
  maximize corpus likelihood under a unigram model; can produce *multiple* segmentations with
  probabilities (useful for regularization via subword sampling).
- **SentencePiece** — a *framework* (not an algorithm) that runs BPE or Unigram **directly on raw
  text** including spaces (encoded as `▁`), so it is language-agnostic and needs no pre-tokenization —
  essential for languages without spaces (Chinese, Japanese).

All three produce a bounded subword vocabulary with no OOV; they differ mainly in the merge/scoring
rule.

## 6. Byte-level BPE and real tokenizers

GPT-2 and successors use **byte-level BPE**: BPE run over raw **bytes** (256 base symbols) rather than
Unicode characters. This guarantees *any* byte sequence — any language, emoji, code, corrupted text —
is representable with **no unknown token ever**, at the cost of some multi-byte characters splitting
into several tokens. Experiment 5 shows GPT-2 (`tiktoken`, vocab 50,257):

| Word | # tokens | Subwords |
|---|:--:|---|
| hello | 1 | `hello` |
| tokenization | 2 | `token`, `ization` |
| antidisestablishmentarianism | 5 | `ant`, `idis`, `establishment`, `arian`, `ism` |
| supercalifragilistic | 6 | `super`, `cal`, `if`, `rag`, `il`, `istic` |

Common words are single tokens; rarer words split into meaningful morphemes; novel strings split
further, down to bytes. This is exactly the from-scratch BPE above, trained on a large corpus over
bytes. **Token counts matter practically:** API pricing, context windows, and even model behavior are
measured in tokens, and a rough rule is **~4 characters ≈ 1 token** for English.

## 7. Practical notes

- **Match the tokenizer to the model.** A pretrained model's embeddings are tied to *its* tokenizer;
  you cannot swap tokenizers without retraining.
- **Vocabulary size** trades embedding-table size and sequence length. 30k–100k is typical; multilingual
  models go larger.
- **Numbers and code tokenize badly.** Digits often split oddly ("2024" → several tokens), a known
  source of arithmetic errors in LLMs; some newer tokenizers split digits individually on purpose.
- **Whitespace is information.** Byte-level tokenizers attach leading spaces to tokens (` the` ≠ `the`),
  so tokenization is position-sensitive.
- **Tokenization affects fairness and cost.** Languages underrepresented in the training corpus tokenize
  into *more* tokens, making them slower and more expensive to process.

## 8. Common misconceptions

- **"Tokens are words."** They are subwords; a long word can be several tokens and a space can belong to
  a token (§6).
- **"Bigger vocabulary is always better."** It shortens sequences but enlarges the embedding table and
  worsens rare-token statistics; it is a trade-off (§3–§4).
- **"Character models avoid all problems."** They avoid OOV but pay with very long sequences (§2).
- **"Stemming/lemmatization is needed."** Only for classical bag-of-words; subword models see shared
  substrings already (§1).
- **"The tokenizer doesn't affect the model much."** It sets the vocabulary, sequence lengths, cost, and
  even arithmetic ability — it is a first-class design choice (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — BPE training and encoding in pure Python, plus a comparison
  to GPT-2 via `tiktoken`. Five experiments: (1) BPE learns the textbook merges; (2) it round-trips and
  needs no `<UNK>`; (3) the word/char/subword vocab-vs-length trade-off; (4) the merge-count compression
  dial; (5) GPT-2 splitting rare words into subwords.
- **[exercises.md](exercises.md)** — implement BPE/WordPiece, analyze the trade-offs, reason about
  byte-level tokenization.
- **[references.md](references.md)** — the BPE, WordPiece, Unigram, and SentencePiece papers.

## Where this leads

- **Classical representations built on tokens (BoW, TF-IDF)** → [10.02](../02-classical-representations/)
- **Turning tokens into vectors (embeddings)** → [10.03](../03-word-embeddings/)
- **The transformers that consume tokens** → [Part 11](../../11-transformers-llms/)
- **The long-sequence cost that motivates subwords** → [09.01](../../09-sequence-models/01-rnn/)
- **NLP tasks and their metrics** → [10.04](../04-nlp-tasks/)
