# 11.07 — Inference & Serving

> **Training an LLM is a one-time cost; serving it is forever — and it has its own algorithms.** How you
> turn logits into tokens (decoding), how you go faster without changing a single output (speculative
> decoding), and how you trade a request's latency for the server's throughput (batching) determine
> whether a model is usable and affordable. This chapter builds those algorithms from scratch, including
> the beautiful result that **speculative decoding is exact** — it produces text identically distributed
> to plain sampling, just faster.

Inference is where models meet users. It is dominated by two facts: generation is **autoregressive**
(one token at a time, [11.03](../03-efficient-attention/)) and **memory-bandwidth bound** (each token
requires reading all the weights). Every technique here works around one of those.

## Table of contents

1. [Prefill and decode](#1-prefill-and-decode)
2. [Decoding strategies](#2-decoding-strategies)
3. [Speculative decoding](#3-speculative-decoding)
4. [Batching: throughput vs latency](#4-batching-throughput-vs-latency)
5. [The serving stack](#5-the-serving-stack)
6. [Metrics that matter](#6-metrics-that-matter)
7. [Common misconceptions](#7-common-misconceptions)

## 1. Prefill and decode

Generation has two phases with opposite characteristics:

- **Prefill** — process the whole prompt at once. All positions run in parallel, so it is
  **compute-bound** and fast per token. Fills the KV cache ([11.03 §2](../03-efficient-attention/)).
- **Decode** — generate one token, append to the cache, repeat. Each step reads *all* the weights to
  produce *one* token, so it is **memory-bandwidth-bound** and the dominant cost of long generations.

The asymmetry drives everything: prefill is cheap and parallel; decode is the bottleneck, which is why
speculative decoding (§3) and batching (§4) target it.

## 2. Decoding strategies

The model outputs a probability distribution over the next token; the **decoder** decides how to pick
one. Experiment 1 measures how each strategy reshapes a 20-token distribution:

| Strategy | Entropy (bits) | Tokens with p > 0.001 |
|---|:--:|:--:|
| temperature T=0.5 | 1.84 | 11 |
| temperature T=1.0 | 3.10 | 19 |
| temperature T=2.0 | 3.91 | 20 |
| top-k (k=3) | 1.52 | 3 |
| top-p (p=0.9) | 2.66 | 9 |
| greedy (argmax) | 0.00 | 1 |

- **Greedy / beam** — take the most probable token(s). Best for tasks with one right answer
  (translation, extraction), but repetitive and dull for open-ended text.
- **Temperature** — scale the logits by $1/T$: $T < 1$ sharpens (more deterministic), $T > 1$ flattens
  (more diverse).
- **Top-k** — sample only from the $k$ most-likely tokens.
- **Top-p (nucleus)** — sample from the smallest set of tokens whose cumulative probability exceeds $p$;
  adapts to how peaked the distribution is (keeps few tokens when confident, more when uncertain).

The open-ended-generation standard is **top-p (~0.9–0.95) with moderate temperature** — it truncates the
unreliable long tail while preserving diversity.

## 3. Speculative decoding

The elegant idea: use a small, fast **draft** model to *guess* several tokens, and the big **target**
model to *check* them all in **one** forward pass (which costs the same as generating a single token).
The subtlety is doing this **without changing the output distribution**. The accept/reject rule
(Leviathan et al., 2023): the draft proposes $x \sim q$; accept it with probability

$$
\min\!\left(1, \frac{p(x)}{q(x)}\right),
$$

and on rejection, resample from the normalized residual $\frac{\max(0,\,p - q)}{\sum \max(0,\,p - q)}$.
Experiment 2 confirms this produces samples distributed **exactly** as the target $p$ (empirical error
0.0019 over 200k draws, → 0 as $N$ grows):

| | token distribution |
|---|---|
| target $p$ | [0.108, 0.220, 0.105, 0.009, 0.250, …] |
| speculative | [0.108, 0.222, 0.106, 0.009, 0.249, …] |

**It is not an approximation** — the generated text is identical in distribution to plain sampling. The
speedup is the number of tokens accepted per target pass. Experiment 3 (draft proposes 4 tokens):

| Draft–target agreement | Accept rate | Tokens/step | Speedup |
|---|:--:|:--:|:--:|
| high (similar models) | 0.89 | 4.0 | 4.0× |
| medium | 0.66 | 2.6 | 2.6× |
| low (very different) | 0.34 | 1.5 | 1.5× |

When the draft agrees with the target, 3–4 tokens land per verification — a 2–4× speedup with **zero**
quality change. This is standard in production serving.

## 4. Batching: throughput vs latency

Because decode reads all weights to make one token, the weight-read dominates — and it can be
**amortized across a batch**. Experiment 4 models a step as `weight_read (10 ms, shared) + compute
(0.5 ms/request)`:

| Batch size | Step time | Latency/req | Throughput (req/s) |
|:--:|:--:|:--:|:--:|
| 1 | 10.5 ms | 10.5 ms | 95 |
| 16 | 18.0 ms | 18.0 ms | 889 |
| 64 | 42.0 ms | 42.0 ms | 1,524 |
| 256 | 138.0 ms | 138.0 ms | **1,855** |

The one weight-read serves the whole batch, so **throughput rises steeply** (95 → 1,855 req/s) while
**per-request latency grows only modestly** (10 → 138 ms). This is why servers batch aggressively — but
there is a ceiling (compute eventually dominates), and interactive use accepts lower throughput for low
latency. **Continuous batching** (adding/removing requests mid-flight, as in vLLM) keeps the batch full
without making requests wait for each other.

## 5. The serving stack

Production LLM serving combines:

- **KV cache + PagedAttention** — the cache ([11.03](../03-efficient-attention/)) grows per request;
  PagedAttention (vLLM) manages it in non-contiguous "pages" like OS virtual memory, eliminating
  fragmentation and enabling continuous batching.
- **Quantization** — int8/int4 weights ([11.05 §6](../05-adaptation/)) cut memory and bandwidth,
  speeding up the memory-bound decode.
- **GQA + FlashAttention** — smaller cache, faster attention ([11.03](../03-efficient-attention/)).
- **Speculative decoding** — §3.
- **Tensor/pipeline parallelism** — split a model too big for one GPU across many.

## 6. Metrics that matter

Serving is measured by more than "tokens/sec":

- **TTFT (time to first token)** — dominated by prefill; what the user feels as responsiveness.
- **TPOT / ITL (time per output token / inter-token latency)** — the decode speed; sets how fast text
  streams.
- **Throughput (tokens or requests/sec)** — the server's total capacity; sets cost per token.
- **Latency vs throughput** — the fundamental trade (§4); batching, quantization, and speculative
  decoding all move this frontier.

Cost per token — the number that decides whether an application is viable — is set by throughput on a
given GPU, which is why all of the above matter.

## 7. Common misconceptions

- **"Speculative decoding approximates the model."** It is provably exact — same output distribution,
  just faster (§3).
- **"Lower temperature is always better."** Greedy is best for one-answer tasks but repetitive for
  open-ended generation; sampling (top-p) is preferred there (§2).
- **"Bigger batches are always better."** They raise throughput but also latency, and hit a compute
  ceiling (§4).
- **"Inference is just a forward pass."** The autoregressive, memory-bound decode loop — with KV cache,
  batching, and speculative decoding — is a whole engineering discipline (§1, §5).
- **"Faster = smaller model."** Speculative decoding, batching, and quantization speed up the *same*
  model with no quality loss (§3–§5).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the inference algorithms in NumPy. Four experiments:
  (1) greedy / temperature / top-k / top-p reshaping the distribution; (2) speculative decoding is exact
  (empirical distribution matches the target to 0.002); (3) its speedup grows with draft–target
  agreement (up to 4×); (4) batching's throughput-vs-latency trade-off.
- **[exercises.md](exercises.md)** — implement the decoders and speculative sampling, derive its
  correctness, model the serving trade-offs.
- **[references.md](references.md)** — speculative decoding, PagedAttention/vLLM, and serving papers.

## Where this leads

- **Efficient attention & the KV cache** → [11.03](../03-efficient-attention/)
- **Quantization for inference** → [11.05](../05-adaptation/)
- **RAG and agents built on top of inference** → [11.08](../08-rag-and-agents/)
- **Decoding for seq2seq (beam search)** → [09.03](../../09-sequence-models/03-seq2seq-and-attention/)
- **MLOps and production serving** → [Part 19](../../19-mlops/)
