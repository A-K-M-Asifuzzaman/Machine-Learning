# 11.02 — References: Pretraining Paradigms

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Self-supervised pretraining | Peters et al. (2018, ELMo); Howard & Ruder (2018) |
| §2-§3 | Encoder, MLM | Devlin et al. (2019, BERT); Liu et al. (2019, RoBERTa) |
| §2, §4 | Decoder, causal LM | Radford et al. (2018, 2019, GPT/GPT-2) |
| §2, §5 | Encoder-decoder, span corruption | Raffel et al. (2020, T5); Lewis et al. (2020, BART) |
| §6 | Perplexity | Jurafsky & Martin ch. 3 |
| §7 | In-context learning, decoder-only | Brown et al. (2020, GPT-3) |

---

## The paradigm papers

- **Devlin, J. et al. (2019).** "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding." *NAACL*. — the **encoder / MLM** paradigm and the 80/10/10 corruption (§2-§3).
  <https://arxiv.org/abs/1810.04805>.
- **Liu, Y. et al. (2019).** "RoBERTa: A Robustly Optimized BERT Pretraining Approach." — drops NSP,
  trains longer/bigger; the tuned BERT recipe (§3). <https://arxiv.org/abs/1907.11692>.
- **Radford, A. et al. (2018).** "Improving Language Understanding by Generative Pre-Training"
  (**GPT**). — the **decoder / causal-LM** paradigm (§4).
  <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>.
- **Radford, A. et al. (2019).** "Language Models are Unsupervised Multitask Learners" (**GPT-2**). —
  scaling causal LM; zero-shot task transfer (§4, §7).
- **Raffel, C. et al. (2020).** "Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer" (**T5**). *JMLR*. — **span corruption** and text-to-text (§5).
  <https://arxiv.org/abs/1910.10683>.
- **Lewis, M. et al. (2020).** "BART: Denoising Sequence-to-Sequence Pre-training." *ACL*. — corrupt-
  then-reconstruct encoder-decoder pretraining (§5). <https://arxiv.org/abs/1910.13461>.

## Context and scaling

- **Peters, M. et al. (2018).** "Deep contextualized word representations" (**ELMo**). *NAACL*. — early
  contextual pretraining (§1). <https://arxiv.org/abs/1802.05365>.
- **Howard, J. & Ruder, S. (2018).** "Universal Language Model Fine-tuning (ULMFiT)." *ACL*. — the
  pretrain-then-fine-tune recipe for NLP (§1). <https://arxiv.org/abs/1801.06146>.
- **Brown, T. et al. (2020).** "Language Models are Few-Shot Learners" (**GPT-3**). *NeurIPS*. —
  in-context learning; the case for decoder-only at scale (§7). <https://arxiv.org/abs/2005.14165>.

## Textbook

- **Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed. draft).** — language modeling,
  perplexity (ch. 3), and pretraining. Free at <https://web.stanford.edu/~jurafsky/slp3/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | BERT / GPT / T5 model code and pretrained weights |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | causal-LM pretraining from scratch |
| [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer) | T5 span corruption |

---

## Deferred to later chapters

- **The transformer block** → [11.01](../01-transformer/)
- **Efficient / long-context attention** → [11.03](../03-efficient-attention/)
- **Scaling laws** → [11.04](../04-scaling-and-architecture/)
- **Fine-tuning, LoRA, instruction tuning** → [11.05](../05-adaptation/)
- **Alignment (RLHF/DPO)** → [11.06](../06-alignment/)
