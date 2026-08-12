# 11.08 — References: RAG & Agents

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | RAG | Lewis et al. (2020) |
| §2 | Dense retrieval | Karpukhin et al. (2020, DPR) |
| §4 | ANN (IVF, HNSW, PQ) | Johnson et al. (2019, FAISS); Malkov & Yashunin (2018, HNSW) |
| §5 | Cross-encoder reranking | Nogueira & Cho (2019) |
| §6 | Agents, tool use | Yao et al. (2023, ReAct); Schick et al. (2023, Toolformer) |
| §7 | RAG evaluation | Es et al. (2023, RAGAS) |

---

## RAG and retrieval

- **Lewis, P. et al. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
  *NeurIPS*. — the RAG framework (§1). <https://arxiv.org/abs/2005.11401>.
- **Karpukhin, V. et al. (2020).** "Dense Passage Retrieval for Open-Domain Question Answering."
  *EMNLP*. — **DPR**, dense bi-encoder retrieval (§2). <https://arxiv.org/abs/2004.04906>.
- **Nogueira, R. & Cho, K. (2019).** "Passage Re-ranking with BERT." — cross-encoder **reranking** (§5).
  <https://arxiv.org/abs/1901.04085>.
- **Robertson, S. & Zaragoza, H. (2009).** "The Probabilistic Relevance Framework: BM25 and Beyond." —
  the keyword baseline for hybrid search (§2).

## Vector search

- **Johnson, J., Douze, M. & Jégou, H. (2019).** "Billion-scale similarity search with GPUs" (**FAISS**).
  *IEEE Big Data*. — IVF, PQ, and the vector-search toolkit (§4). <https://arxiv.org/abs/1702.08734>.
- **Malkov, Y. & Yashunin, D. (2018).** "Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs" (**HNSW**). *IEEE TPAMI*. (§4).
  <https://arxiv.org/abs/1603.09320>.

## Agents and tools

- **Yao, S. et al. (2023).** "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*. — the
  think-act-observe loop (§6). <https://arxiv.org/abs/2210.03629>.
- **Schick, T. et al. (2023).** "Toolformer: Language Models Can Teach Themselves to Use Tools."
  *NeurIPS*. — tool/function calling (§6). <https://arxiv.org/abs/2302.04761>.
- **Wei, J. et al. (2022).** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."
  *NeurIPS*. — the reasoning that agents build on (§6). <https://arxiv.org/abs/2201.11903>.

## Evaluation

- **Es, S. et al. (2023).** "RAGAS: Automated Evaluation of Retrieval Augmented Generation." — faithfulness
  and answer-relevance metrics (§7). <https://arxiv.org/abs/2309.15217>.
- **Liu, N. et al. (2023).** "Lost in the Middle." — long-context retrieval quality (§3, §7).
  <https://arxiv.org/abs/2307.03172>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [FAISS](https://github.com/facebookresearch/faiss) | IVF / HNSW / PQ vector search |
| [LangChain](https://github.com/langchain-ai/langchain) / [LlamaIndex](https://github.com/run-llama/llama_index) | RAG and agent orchestration |
| [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) | bi-encoders and cross-encoder rerankers |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG evaluation |

---

## Deferred to later chapters

- **Inference and serving** → [11.07](../07-inference/)
- **Embeddings** → [10.03](../../10-nlp/03-word-embeddings/)
- **Retrieval metrics** → [10.04](../../10-nlp/04-nlp-tasks/)
- **Production MLOps & system design** → [Part 19](../../19-mlops/), [Part 20](../../20-ml-system-design/)
