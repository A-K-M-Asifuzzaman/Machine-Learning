# 10.04 — References: NLP Tasks & Metrics

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2, §6 | Classification, benchmarks | Wang et al. (2018, GLUE); Wang et al. (2019, SuperGLUE) |
| §3 | NER, BIO tagging, entity F1 | Tjong Kim Sang & De Meulder (2003, CoNLL) |
| §4 | Extractive QA, EM/F1 | Rajpurkar et al. (2016, SQuAD) |
| §5 | BLEU | Papineni et al. (2002) |
| §5 | ROUGE | Lin (2004) |
| §5 | BERTScore | Zhang et al. (2020) |
| §6 | NLI | Bowman et al. (2015, SNLI); Williams et al. (2018, MNLI) |

---

## Task and benchmark papers

- **Tjong Kim Sang, E. & De Meulder, F. (2003).** "Introduction to the CoNLL-2003 Shared Task:
  Language-Independent Named Entity Recognition." — the **NER** benchmark and **entity-level F1**
  scoring (§3). <https://aclanthology.org/W03-0419/>.
- **Rajpurkar, P. et al. (2016).** "SQuAD: 100,000+ Questions for Machine Comprehension of Text."
  *EMNLP*. — **extractive QA** and the **EM / token-F1** metrics (§4). <https://arxiv.org/abs/1606.05250>.
- **Bowman, S. et al. (2015).** "A large annotated corpus for learning natural language inference"
  (**SNLI**). *EMNLP*. — the **NLI** task (§6). <https://arxiv.org/abs/1508.05326>.
- **Williams, A., Nangia, N. & Bowman, S. (2018).** "A Broad-Coverage Challenge Corpus for Sentence
  Understanding through Inference" (**MultiNLI**). *NAACL*. <https://arxiv.org/abs/1704.05426>.
- **Wang, A. et al. (2018).** "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language
  Understanding." — the standard **NLU benchmark suite** (§2, §6). <https://arxiv.org/abs/1804.07461>.
- **Wang, A. et al. (2019).** "SuperGLUE." — the harder successor. <https://arxiv.org/abs/1905.00537>.

## Metric papers

- **Papineni, K. et al. (2002).** "BLEU: a Method for Automatic Evaluation of Machine Translation."
  *ACL*. — **BLEU** (§5). <https://aclanthology.org/P02-1040/>.
- **Lin, C.-Y. (2004).** "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop*. —
  **ROUGE-N and ROUGE-L** (§5). <https://aclanthology.org/W04-1013/>.
- **Zhang, T. et al. (2020).** "BERTScore: Evaluating Text Generation with BERT." *ICLR*. — an
  embedding-based alternative to n-gram overlap (§5). <https://arxiv.org/abs/1904.09675>.
- **Zheng, L. et al. (2023).** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." — the
  **LLM-as-judge** evaluation paradigm (§5, §7). <https://arxiv.org/abs/2306.05685>.

---

## Textbook

- **Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed. draft).** — chapters on
  sequence labeling, NER, QA, MT, and evaluation. Free at <https://web.stanford.edu/~jurafsky/slp3/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`seqeval`](https://github.com/chakki-works/seqeval) | the standard entity-level NER F1 (§3) |
| [`sacrebleu`](https://github.com/mjpost/sacrebleu) | the reproducible standard BLEU (§5) |
| [`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge) | Google's ROUGE (§5) |
| [`nltk.translate.bleu_score`](https://www.nltk.org/api/nltk.translate.bleu_score.html) | BLEU, verified against here |
| [Hugging Face `evaluate`](https://github.com/huggingface/evaluate) | a unified metrics library |

---

## Deferred to later chapters

- **The transformers solving these tasks** → [Part 11](../../11-transformers-and-llms/)
- **General classification metrics** → [05.03](../../05-model-evaluation/03-classification-metrics/)
- **Decoding for generation** → [11.07](../../11-transformers-and-llms/07-inference/)
- **LLM evaluation and safety** → [11.08](../../11-transformers-and-llms/08-rag-and-agents/)
