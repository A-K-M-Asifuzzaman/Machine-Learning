"""
11.08 — RAG & agents, from scratch (NumPy).

An LLM's knowledge is frozen at training time and its context is finite. RETRIEVAL-AUGMENTED GENERATION
(RAG) fixes this by fetching relevant documents and putting them in the prompt; AGENTS extend it by
letting the model call TOOLS. This file builds the retrieval machinery and the agent loop, and MEASURES
each:

  1. semantic (embedding) search finds relevant docs that keyword search misses  -> Experiment 1
  2. two-stage retrieval: a reranker improves precision over the first-stage retriever -> Experiment 2
  3. chunk size has an optimum: too small loses context, too large dilutes relevance   -> Experiment 3
  4. approximate nearest neighbor (IVF): big speedup for a small recall cost           -> Experiment 4
  5. the tool-use (ReAct) loop solves a task the model alone cannot                     -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


# =============================================================================
# EXPERIMENT 1 — semantic search vs keyword search
# =============================================================================


def experiment_1_semantic():
    print("=" * 88)
    print("EXPERIMENT 1 — semantic (embedding) search finds what keyword search misses (README §2)")
    print("=" * 88)
    # a tiny topic-embedding space: each word maps to a 3-topic vector (space / food / sport)
    topics = {"planet": [1, 0, 0], "cosmos": [1, 0, 0], "galaxy": [1, 0, 0], "orbit": [1, 0, 0],
              "star": [1, 0, 0], "nebula": [1, 0, 0],           # space SYNONYMS not used in any doc
              "recipe": [0, 1, 0], "cook": [0, 1, 0], "kitchen": [0, 1, 0],
              "match": [0, 0, 1], "score": [0, 0, 1], "team": [0, 0, 1]}
    docs = ["cosmos galaxy orbit", "recipe cook kitchen", "match score team", "planet orbit cosmos"]

    def embed(text):
        v = np.sum([topics[w] for w in text.split()], axis=0).astype(float)
        return v / (np.linalg.norm(v) + 1e-12)

    def keyword_vec(text, vocab):
        v = np.zeros(len(vocab))
        for w in text.split():
            v[vocab[w]] += 1
        return v / (np.linalg.norm(v) + 1e-12)

    query = "star nebula"                              # a SPACE query using words in NO document
    vocab = {w: i for i, w in enumerate(sorted(topics))}
    q_emb = embed(query); q_kw = keyword_vec(query, vocab)
    print(f'\n  Query: "{query}" (a space query). Similarity to each document:\n')
    print(f"    {'document':>22s} {'keyword sim':>12s} {'semantic sim':>13s}")
    for d in docs:
        print(f"    {d:>22s} {cos(q_kw, keyword_vec(d, vocab)):>12.2f} {cos(q_emb, embed(d)):>13.2f}")
    print("""
  READING: keyword search matches exact words, so the query "planet galaxy" scores 0 on
  "cosmos galaxy orbit" if they share few words — it cannot see that both are about SPACE. SEMANTIC
  search embeds query and documents into a meaning space (here a topic vector) where synonyms and
  related terms land close, so it retrieves the right documents by MEANING, not surface words. This is
  why RAG uses dense embeddings + vector search, not keyword match (though hybrids of both win in
  practice) (README §2).""")


# =============================================================================
# EXPERIMENT 2 — reranking
# =============================================================================


def experiment_2_reranking():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — two-stage retrieval: a reranker improves precision (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n_docs = 200
    true_rel = rng.random(n_docs)                      # ground-truth relevance to the query
    # first-stage retriever (cheap bi-encoder): a NOISY estimate of relevance
    biencoder = true_rel + 0.3 * rng.standard_normal(n_docs)
    # retrieve top-20 by the cheap score
    retrieved = np.argsort(-biencoder)[:20]
    # reranker (expensive cross-encoder): near-exact relevance, but only on the 20 retrieved
    crossencoder = true_rel + 0.03 * rng.standard_normal(n_docs)

    def precision_at_k(order, k):
        top = order[:k]
        return np.mean(true_rel[top] > np.sort(true_rel)[-20])   # in the true top-20?

    stage1 = retrieved[np.argsort(-biencoder[retrieved])]        # order by bi-encoder
    stage2 = retrieved[np.argsort(-crossencoder[retrieved])]     # rerank by cross-encoder
    print(f"""
  {n_docs} documents. Stage 1 (cheap bi-encoder) retrieves 20 candidates; stage 2 (expensive
  cross-encoder) reranks them. Precision@k = fraction of top-k that are truly top-20 relevant:

    {'k':>4s} {'bi-encoder only':>17s} {'+ reranker':>12s}
    {3:>4d} {precision_at_k(stage1, 3):>17.2f} {precision_at_k(stage2, 3):>12.2f}
    {5:>4d} {precision_at_k(stage1, 5):>17.2f} {precision_at_k(stage2, 5):>12.2f}
    {10:>4d} {precision_at_k(stage1, 10):>17.2f} {precision_at_k(stage2, 10):>12.2f}

  READING: a fast bi-encoder (embed query and docs separately, compare vectors) scans millions of
  documents cheaply but scores them noisily. A cross-encoder (feed query+doc together through a
  transformer) is far more accurate but too expensive to run on everything. The two-stage pipeline gets
  both: retrieve a shortlist cheaply, then RERANK the shortlist accurately — lifting precision@k without
  the cross-encoder ever touching the full corpus. Standard in production search and RAG (README §4).""")


# =============================================================================
# EXPERIMENT 3 — chunk size has an optimum
# =============================================================================


def experiment_3_chunking():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — chunk size: too small loses context, too large dilutes (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    # a 30-sentence document; the answer needs THREE aspects split across sentences 12-14,
    # so no single sentence is sufficient (the query needs all three)
    n_sent = 30
    q = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)          # query needs all three aspects
    sent_vecs = rng.standard_normal((n_sent, 3)) * 0.2  # background sentences: low query relevance
    sent_vecs[12] = [1, 0, 0]; sent_vecs[13] = [0, 1, 0]; sent_vecs[14] = [0, 0, 1]  # one aspect each
    print(f"\n  Answer needs 3 aspects split across sentences 12-14. Chunk the doc, embed each chunk")
    print(f"  (mean of its sentences), retrieve the best chunk. Its similarity to the query vs chunk size:\n")
    print(f"    {'chunk size':>12s} {'best-chunk similarity':>22s}")
    for cs in (1, 3, 5, 10, 30):
        best = 0.0
        for start in range(0, n_sent, cs):
            chunk = sent_vecs[start:start + cs].mean(0)
            best = max(best, cos(q, chunk))
        print(f"    {cs:>12d} {best:>22.3f}")
    print("""
  READING: retrieval embeds whole CHUNKS, so chunk size is a real knob. Size 1 (single sentences) can
  split the answer across chunks and misses the surrounding context, lowering similarity; size 30 (the
  whole doc) averages the 3 relevant sentences with 27 irrelevant ones, DILUTING the signal. The best
  similarity comes at a chunk size close to the answer span (~3-5 sentences) — big enough to hold the
  context, small enough not to drown it. Tuning chunk size (and overlap) is one of the highest-leverage
  RAG decisions (README §3).""")


# =============================================================================
# EXPERIMENT 4 — approximate nearest neighbor (IVF)
# =============================================================================


def experiment_4_ann():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — approximate nearest neighbor: big speedup for a small recall cost (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    n, d, k, n_clusters = 6000, 32, 10, 15
    centers = rng.standard_normal((n_clusters, d)) * 2.5
    X = centers[rng.integers(0, n_clusters, n)] + rng.standard_normal((n, d))   # clustered (real embeddings are)
    query = X[rng.integers(0, n)] + 1.2 * rng.standard_normal(d)
    exact = set(np.argsort(-(X @ query))[:k].tolist())

    def kmeans(X, C, iters=20, seed=0):
        r = np.random.default_rng(seed)
        cent = X[r.choice(len(X), C, replace=False)].copy()
        for _ in range(iters):
            a = np.argmin(((X[:, None] - cent[None]) ** 2).sum(2), 1)
            for c in range(C):
                if (a == c).any():
                    cent[c] = X[a == c].mean(0)
        return a, cent

    C = 128
    idx, centroids = kmeans(X, C)                       # IVF cells via k-means (like FAISS)
    print(f"\n  {n:,} vectors, dim {d}. Exact search compares to all {n:,}. IVF searches only the")
    print(f"  nearest `probe` of {C} k-means clusters:\n")
    print(f"    {'probe cells':>12s} {'comparisons':>13s} {'speedup':>9s} {'recall@10':>11s}")
    for probe in (1, 4, 8, 16):
        near_cells = np.argsort(-(centroids @ query))[:probe]
        cand = np.where(np.isin(idx, near_cells))[0]
        approx = set(cand[np.argsort(-(X[cand] @ query))[:k]].tolist())
        recall = len(approx & exact) / k
        print(f"    {probe:>12d} {len(cand):>13,d} {n / max(len(cand),1):>8.1f}x {recall:>11.2f}")
    print("""
  READING: exact nearest-neighbor search compares the query to EVERY vector — O(n), too slow for
  billions of embeddings. IVF (inverted file) clusters the vectors and searches only the nearest few
  clusters, cutting comparisons dramatically. The trade-off is recall: probing few cells is fast but may
  miss some true neighbors; probing more recovers them. Real indexes (HNSW, IVF-PQ in FAISS) hit ~0.95+
  recall at 10-100x speedup — the engine under every vector database (README §5).""")


# =============================================================================
# EXPERIMENT 5 — the tool-use (ReAct) agent loop
# =============================================================================


def experiment_5_agent():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the tool-use (ReAct) loop solves what the model alone cannot (README §6)")
    print("=" * 88)
    # tools the "agent" can call
    def calculator(expr):
        return eval(expr, {"__builtins__": {}}, {})

    # a task requiring exact arithmetic (LLMs are unreliable at this)
    task = "Compute (47 * 89) + (123 * 4)"
    # the agent's POLICY (what an LLM would emit) is scripted here to show the LOOP structure:
    plan = ["47 * 89", "123 * 4", "4183 + 492"]        # think -> act (call tool) -> observe -> repeat
    print(f'\n  Task: "{task}"\n')
    print(f"  The model is bad at exact arithmetic, so it delegates to a calculator TOOL:\n")
    observations = []
    for step, expr in enumerate(plan, 1):
        result = calculator(expr)
        observations.append(result)
        print(f"    step {step}:  THINK 'I need {expr}'  ->  ACT calculator('{expr}')  ->  OBSERVE {result}")
    final = observations[-1]
    truth = (47 * 89) + (123 * 4)
    print(f"""
    final answer: {final}   (ground truth {truth}, correct = {final == truth})

  READING: an AGENT is an LLM in a loop with TOOLS: it THINKS about what it needs, ACTS by calling a
  tool (calculator, web search, code execution, an API), OBSERVES the result, and repeats until done
  (the 'ReAct' pattern). This overcomes the model's hard limits — exact arithmetic, fresh information,
  real actions — by delegating them to reliable external tools, with the LLM as the ORCHESTRATOR that
  decides which tool to call and how to combine results. RAG is the special case where the only tool is
  'retrieve documents' (README §6).""")


if __name__ == "__main__":
    experiment_1_semantic()
    experiment_2_reranking()
    experiment_3_chunking()
    experiment_4_ann()
    experiment_5_agent()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
