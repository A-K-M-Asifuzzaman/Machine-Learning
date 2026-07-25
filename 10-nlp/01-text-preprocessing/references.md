# 10.01 — References: Text Preprocessing & Tokenization

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Preprocessing, normalization | Jurafsky & Martin, ch. 2 |
| §2 | Granularity trade-off | Sennrich et al. (2016) |
| §3-§4 | Byte-Pair Encoding | Sennrich et al. (2016); Gage (1994) |
| §5 | WordPiece | Schuster & Nakajima (2012); Wu et al. (2016) |
| §5 | Unigram LM, SentencePiece | Kudo (2018); Kudo & Richardson (2018) |
| §6 | Byte-level BPE | Radford et al. (2019, GPT-2) |

---

## The tokenization papers

- **Sennrich, R., Haddow, B. & Birch, A. (2016).** "Neural Machine Translation of Rare Words with
  Subword Units." *ACL*. — introduced **BPE for NLP**; the algorithm, the open-vocabulary property, and
  the rare-word motivation (§2-§4). <https://arxiv.org/abs/1508.07909>.
- **Gage, P. (1994).** "A New Algorithm for Data Compression." *C Users Journal*. — the original **BPE**
  compression algorithm that Sennrich adapted (§3).
- **Schuster, M. & Nakajima, K. (2012).** "Japanese and Korean Voice Search." *ICASSP*. — the
  **WordPiece** algorithm (§5).
- **Kudo, T. (2018).** "Subword Regularization: Improving NMT Models with Multiple Subword Candidates."
  *ACL*. — the **Unigram LM** tokenizer and subword sampling (§5). <https://arxiv.org/abs/1804.10959>.
- **Kudo, T. & Richardson, J. (2018).** "SentencePiece: A simple and language independent subword
  tokenizer." *EMNLP (demo)*. — **SentencePiece** on raw text (§5). <https://arxiv.org/abs/1808.06226>.
- **Radford, A. et al. (2019).** "Language Models are Unsupervised Multitask Learners" (**GPT-2**). —
  **byte-level BPE**, vocabulary 50,257 (§6).
  <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>.

---

## Textbook

- **Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed. draft).** — **Chapter 2**
  covers normalization, tokenization, and BPE in depth. Free at <https://web.stanford.edu/~jurafsky/slp3/>.

---

## Tools and reference implementations

| Source | What to look at |
|---|---|
| [`tiktoken`](https://github.com/openai/tiktoken) | OpenAI's byte-level BPE tokenizers (GPT-2/3/4); used in Experiment 5 |
| [Hugging Face `tokenizers`](https://github.com/huggingface/tokenizers) | fast BPE / WordPiece / Unigram trainers and encoders |
| [`sentencepiece`](https://github.com/google/sentencepiece) | BPE and Unigram on raw text |
| [The Tokenizer Playground](https://platform.openai.com/tokenizer) | visualize how text tokenizes |
| [Karpathy, minBPE](https://github.com/karpathy/minbpe) | a clean from-scratch BPE, and his BPE video |

---

## Deferred to later chapters

- **Classical representations over tokens (BoW, TF-IDF)** → [10.02](../02-classical-representations/)
- **Embeddings — tokens to vectors** → [10.03](../03-word-embeddings/)
- **Transformers that consume tokens** → [Part 11](../../11-transformers-llms/)
- **Sequence length and cost** → [09.01](../../09-sequence-models/01-rnn/), [08.05](../../08-computer-vision/05-vision-transformers/)
