# 11.08 — RAG & Agents

> **An LLM's knowledge is frozen at training time and its context is finite — so give it a memory and
> hands.** Retrieval-Augmented Generation (RAG) fetches relevant documents and puts them in the prompt,
> grounding answers in current, private, or citeable sources. Agents go further, letting the model call
> **tools** — search, code, calculators, APIs — in a think-act-observe loop. This chapter builds the
> retrieval stack (embed → chunk → index → retrieve → rerank) and the agent loop from scratch, and
> measures what each part buys.

Two hard limits of a bare LLM: it can't know anything after its training cutoff or anything private, and
it can't reliably do exact operations (arithmetic, lookups, actions). RAG addresses the first; tool-using
agents address the second. Both are *inference-time* systems built around the model
([11.07](../07-inference/)), not new training.

## Table of contents

1. [Why RAG](#1-why-rag)
2. [Semantic search](#2-semantic-search)
3. [Chunking](#3-chunking)
4. [Vector databases and ANN](#4-vector-databases-and-ann)
5. [Reranking](#5-reranking)
6. [Agents and tool use](#6-agents-and-tool-use)
7. [Evaluation](#7-evaluation)
8. [Common misconceptions](#8-common-misconceptions)

## 1. Why RAG

The RAG pipeline: **embed** a corpus of documents into vectors, **index** them, and at query time
**retrieve** the most relevant chunks, **rerank** them, and **stuff** the top ones into the prompt as
context. The model then answers *grounded* in the retrieved text. This buys four things a bare LLM lacks:

- **Fresh knowledge** — retrieve from a live, updatable index instead of retraining.
- **Private knowledge** — your documents, never in the model's training data.
- **Citations** — the answer points to its sources, reducing hallucination.
- **Cost** — updating an index is far cheaper than fine-tuning.

The whole thing hinges on **retrieval quality**: garbage retrieved is garbage generated.

## 2. Semantic search

Retrieval uses **dense embeddings** ([10.03](../../10-nlp/03-word-embeddings/)), not keyword matching,
so it finds documents by *meaning*. Experiment 1 queries "star nebula" (space terms in **no** document):

| Document | Keyword sim | Semantic sim |
|---|:--:|:--:|
| cosmos galaxy orbit | 0.00 | **1.00** |
| recipe cook kitchen | 0.00 | 0.00 |
| planet orbit cosmos | 0.00 | **1.00** |

Keyword search scores **0** on the relevant space documents — it cannot see that "star" and "cosmos" are
the same topic. Semantic search embeds query and documents into a meaning space where synonyms land
close, retrieving the right documents by content. (In practice **hybrid** search — dense embeddings *plus*
keyword/BM25 — wins, because keywords catch exact terms like names and codes that embeddings blur.)

## 3. Chunking

Documents are split into **chunks** before embedding, and chunk size is a real knob. Experiment 3 hides
an answer that needs three aspects spread across sentences 12–14, then retrieves the best chunk at each
size:

| Chunk size | Best-chunk similarity |
|:--:|:--:|
| 1 (sentence) | 0.933 |
| **3 (the span)** | **1.000** |
| 5 | 0.952 |
| 10 | 0.926 |
| 30 (whole doc) | 0.906 |

Too **small** and a single sentence misses the surrounding context (and the answer can split across
chunks); too **large** and the relevant sentences are averaged with irrelevant ones, **diluting** the
signal. The best retrieval comes at a chunk size close to the answer span — big enough to hold the
context, small enough not to drown it. Tuning chunk size (and adding overlap between chunks) is one of
the highest-leverage RAG decisions.

## 4. Vector databases and ANN

At scale (millions–billions of vectors), comparing a query to *every* embedding is too slow. **Vector
databases** use **approximate nearest neighbor (ANN)** search. Experiment 4 builds an **IVF** index
(cluster the vectors with k-means, search only the nearest few clusters):

| Probe cells | Comparisons | Speedup | Recall@10 |
|:--:|:--:|:--:|:--:|
| 1 | 71 | 84× | 0.70 |
| 4 | 280 | 21× | 0.90 |
| 8 | 428 | 14× | **1.00** |

Probing few cells is fast but may miss some true neighbors; probing more recovers them — the classic
**recall/speed trade-off**. Production indexes (**HNSW**, **IVF-PQ** in FAISS/ScaNN) reach ~0.95+ recall
at 10–100× speedup and are the engine under every vector database (Pinecone, Weaviate, pgvector, …).

## 5. Reranking

A single retriever must be both fast (to scan millions) and accurate (to rank well) — conflicting goals.
The standard solution is **two-stage** retrieval. Experiment 2 (200 documents):

| k | Bi-encoder only | + Reranker |
|:--:|:--:|:--:|
| 3 | 0.33 | **0.67** |
| 5 | 0.20 | **0.60** |
| 10 | 0.30 | **0.40** |

- **Stage 1 — bi-encoder** (retriever): embed query and documents *separately*, compare vectors. Fast
  (documents pre-embedded) but scores noisily.
- **Stage 2 — cross-encoder** (reranker): feed query *and* document together through a transformer for a
  precise relevance score. Far more accurate but too expensive for the whole corpus.

Retrieve a shortlist cheaply, then rerank it accurately — precision@k jumps (0.33 → 0.67 at k=3) without
the cross-encoder ever touching the full corpus. This two-stage pattern is standard in search and RAG.

## 6. Agents and tool use

An **agent** is an LLM in a loop with **tools**. The **ReAct** pattern (Yao et al., 2023): the model
**thinks** about what it needs, **acts** by calling a tool, **observes** the result, and repeats until
done. Experiment 5 solves `(47 × 89) + (123 × 4)` — which LLMs do unreliably — by delegating to a
calculator:

```
step 1:  THINK 'I need 47 * 89'    -> ACT calculator('47 * 89')   -> OBSERVE 4183
step 2:  THINK 'I need 123 * 4'    -> ACT calculator('123 * 4')   -> OBSERVE 492
step 3:  THINK 'I need 4183 + 492' -> ACT calculator('4183 + 492')-> OBSERVE 4675   ✓
```

Tools overcome the model's hard limits — exact computation, fresh information, real-world actions —
by delegating them to reliable external systems, with the LLM as the **orchestrator** deciding which
tool to call and how to combine results. **Function/tool calling** (the model emits a structured call
the runtime executes) is the API-level version. **RAG is the special case where the only tool is
"retrieve documents."** Agentic systems chain many such steps (plan, search, code, reflect), which is
powerful but compounds errors and cost — reliability and evaluation are the hard parts.

## 7. Evaluation

RAG and agents need evaluation on *two* axes:

- **Retrieval** — did we fetch the right context? Measured by recall@k, precision@k, MRR, nDCG
  ([10.04](../../10-nlp/04-nlp-tasks/)) against labeled relevant documents.
- **Generation / faithfulness** — is the answer *grounded* in the retrieved context (not hallucinated),
  and does it actually answer the question? Measured by faithfulness/groundedness and answer-relevance,
  increasingly via **LLM-as-judge** (RAGAS, and human eval).

For agents, add **task success rate**, number of steps, and cost. The recurring lesson: a great
generator with bad retrieval fails, so **evaluate the pieces separately** before the whole.

## 8. Common misconceptions

- **"RAG fine-tunes the model."** RAG changes the *prompt* (retrieved context) at inference; no weights
  change (§1).
- **"Semantic search replaces keywords."** Hybrid (dense + BM25) beats either alone — keywords catch
  exact terms embeddings blur (§2).
- **"Bigger chunks are better."** They dilute relevance; there is an optimum near the answer span (§3).
- **"ANN is exact."** It trades recall for speed; you tune the operating point (§4).
- **"Agents are just prompting."** The think-act-observe loop with real tools is a system; its failure
  mode is compounding errors, which is why evaluation and guardrails matter (§6–§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the RAG stack and agent loop in NumPy. Five experiments:
  (1) semantic search finds space docs keyword search scores 0 on; (2) a reranker lifts precision@3 from
  0.33 to 0.67; (3) chunk size has an optimum at the answer span; (4) IVF ANN's recall/speed trade-off
  (0.70 @ 84× to 1.0 @ 14×); (5) a ReAct tool-use loop solving arithmetic the model can't.
- **[exercises.md](exercises.md)** — build a retriever, a reranker, an ANN index, and an agent loop;
  reason about chunking and evaluation.
- **[references.md](references.md)** — RAG, dense retrieval, FAISS/HNSW, ReAct, and RAG-evaluation papers.

## Where this leads

- **Inference and serving underneath RAG** → [11.07](../07-inference/)
- **The embeddings retrieval is built on** → [10.03](../../10-nlp/03-word-embeddings/)
- **Retrieval metrics (recall/precision/nDCG)** → [10.04](../../10-nlp/04-nlp-tasks/)
- **The transformer that generates the answer** → [11.01](../01-transformer/)
- **Production deployment of these systems** → [Part 19](../../19-mlops/), [Part 20](../../20-ml-system-design/)
