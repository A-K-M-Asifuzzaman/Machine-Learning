# 11.07 — Exercises: Inference & Serving

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Contrast prefill and decode: which is compute-bound, which is memory-bound, and why decode
dominates long generations.

**D2.** Define greedy, temperature, top-k, and top-p decoding and how each reshapes the distribution.

**D3.** Explain why top-p adapts to the distribution's shape better than top-k.

**D4.** Derive speculative decoding's correctness: show the accept-with-prob-$\min(1, p/q)$ + residual
resample scheme produces samples exactly from $p$.

**D5.** Derive the expected number of accepted tokens per verification as a function of the acceptance
rate $\alpha$ and the draft length $\gamma$.

**D6.** Model decode as memory-bound and show why batching amortizes the weight-read, raising throughput
with modest latency cost.

**D7.** Explain the throughput-vs-latency trade-off and where the compute ceiling appears.

**D8.** Explain PagedAttention and how it eliminates KV-cache fragmentation.

**D9.** Define TTFT, TPOT, and throughput, and map each to a user-facing quality.

**D10.** Explain how quantization speeds up the memory-bound decode phase.

---

## Tier 2 — Implementation

**I1.** Implement greedy, temperature, top-k, and top-p decoders; reproduce Experiment 1.

**I2.** Implement speculative sampling and verify its output matches the target distribution
(Experiment 2).

**I3.** Reproduce Experiment 3: measure acceptance rate and speedup vs draft/target agreement.

**I4.** Reproduce Experiment 4: model batching's throughput/latency trade-off.

**I5.** Implement multi-token speculative decoding with a draft that proposes $\gamma$ tokens and a
target that verifies them in one pass.

**I6.** Implement a KV cache and continuous batching for a toy autoregressive model.

**I7.** Add repetition penalty / no-repeat-ngram to your decoder and observe the effect.

**I8.** Measure real TTFT and TPOT for a small model at several batch sizes.

**I9.** Implement min-p or typical sampling and compare to top-p.

**I10.** *(Roofline.)* Build a roofline model of decode and predict the batch size where compute
overtakes memory bandwidth.

---

## Tier 3 — Interview

**Q1.** What is the difference between prefill and decode?

**Q2.** Why is decode memory-bandwidth bound?

**Q3.** What are the main decoding strategies and when do you use each?

**Q4.** What is the difference between top-k and top-p?

**Q5.** How does speculative decoding work?

**Q6.** Is speculative decoding an approximation? Prove your answer.

**Q7.** What determines speculative decoding's speedup?

**Q8.** Why does batching help LLM serving, and what's the trade-off?

**Q9.** What is continuous batching / PagedAttention?

**Q10.** What metrics matter for LLM serving?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain prefill vs decode and their cost profiles
- [ ] Implement all four decoding strategies
- [ ] Implement speculative decoding and prove it is exact
- [ ] Explain the speedup's dependence on draft/target agreement
- [ ] Model the batching throughput/latency trade-off
- [ ] Describe the production serving stack (KV cache, PagedAttention, quantization)
- [ ] Map serving metrics (TTFT/TPOT/throughput) to user experience
