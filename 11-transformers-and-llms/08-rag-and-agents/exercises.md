# 11.08 — Exercises: RAG & Agents

Three tiers. **Reasoning** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Reasoning

**D1.** Describe the full RAG pipeline (embed → chunk → index → retrieve → rerank → generate) and what
each stage contributes.

**D2.** Explain why RAG beats fine-tuning for fresh/private knowledge, and its cost advantage.

**D3.** Contrast semantic (dense) and keyword (BM25) retrieval; explain why hybrid search wins.

**D4.** Explain the chunk-size trade-off (context vs dilution) and how overlap helps.

**D5.** Explain approximate nearest neighbor and the recall/speed trade-off; contrast IVF and HNSW.

**D6.** Explain the two-stage retriever/reranker design: bi-encoder vs cross-encoder, and why both are
needed.

**D7.** Describe the ReAct loop (think-act-observe) and how tool use overcomes the LLM's hard limits.

**D8.** Explain how RAG is a special case of tool use.

**D9.** Explain how to evaluate RAG on retrieval and on faithfulness separately.

**D10.** Describe the failure modes of agentic systems (compounding errors) and mitigations.

---

## Tier 2 — Implementation

**I1.** Build a semantic retriever (embed + cosine) and show it beats keyword search on synonyms
(Experiment 1).

**I2.** Implement a two-stage retriever + reranker; reproduce Experiment 2's precision gain.

**I3.** Reproduce Experiment 3: measure retrieval quality vs chunk size and find the optimum.

**I4.** Implement an IVF ANN index (k-means cells + probe) and reproduce Experiment 4's recall/speed
curve.

**I5.** Implement a ReAct agent loop with a calculator and a search tool (Experiment 5).

**I6.** Build an end-to-end RAG system over a real document set and answer questions with citations.

**I7.** Implement hybrid search (dense + BM25) and compare to each alone.

**I8.** Implement HNSW (or use FAISS) and compare recall/speed to your IVF.

**I9.** Add chunk overlap and measure its effect on retrieval of boundary-spanning answers.

**I10.** *(Eval.)* Implement RAGAS-style faithfulness/answer-relevance metrics with an LLM judge and
evaluate your RAG system.

---

## Tier 3 — Interview

**Q1.** What is RAG and what problems does it solve?

**Q2.** When would you use RAG vs fine-tuning?

**Q3.** How does semantic search work, and why not just keywords?

**Q4.** How do you choose chunk size?

**Q5.** What is approximate nearest neighbor and its trade-off?

**Q6.** What is a reranker and why use a two-stage retriever?

**Q7.** What is an agent, and what is the ReAct loop?

**Q8.** How is RAG related to tool use?

**Q9.** How do you evaluate a RAG system?

**Q10.** What are the main failure modes of agents?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Describe and build the RAG pipeline end to end
- [ ] Explain semantic vs keyword vs hybrid retrieval
- [ ] Choose chunk size and explain the trade-off
- [ ] Implement an ANN index and reason about recall/speed
- [ ] Build a two-stage retriever + reranker
- [ ] Implement a ReAct tool-use agent loop
- [ ] Evaluate retrieval and generation separately
