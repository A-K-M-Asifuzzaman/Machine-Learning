# 09.02 — Exercises: LSTM & GRU

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the LSTM equations and label each gate's role. Show the cell update
$c_t = f_t \odot c_{t-1} + i_t \odot g_t$ is nearly linear in $c_{t-1}$.

**D2.** Derive $\partial c_t/\partial c_{t-1} = f_t$ and hence $\partial c_T/\partial c_0 = \prod_t f_t$.
Explain why this avoids the plain RNN's vanishing gradient.

**D3.** Derive the full LSTM backward pass (gradients w.r.t. each gate pre-activation, the cell state,
and the weights).

**D4.** Write the GRU equations and show the update gate $z$ interpolates between keeping and writing —
the same "keep + add" structure as the LSTM cell.

**D5.** Explain why initializing the forget-gate bias positive (~1) improves long-range learning, using
the $\prod_t f_t$ result.

**D6.** Compare the LSTM cell's gradient path to a residual connection's $I + F'$ Jacobian and state the
analogy precisely.

**D7.** Count the parameters of an LSTM and a GRU with input size $I$ and hidden size $H$; show the GRU
has ~25% fewer.

**D8.** Explain why gates fix vanishing but not exploding gradients, and why clipping is still used.

**D9.** Derive the GRU backward pass for the update and reset gates.

**D10.** Explain the fundamental limitation LSTMs/GRUs still have (sequential computation, finite
effective memory) that motivates attention.

---

## Tier 2 — Implementation

**I1.** Implement the LSTM forward pass; verify against `torch.nn.LSTM` (Experiment 1).

**I2.** Implement the LSTM backward pass; verify against PyTorch autograd (Experiment 1).

**I3.** Implement the GRU forward pass; verify against `torch.nn.GRU` (Experiment 2).

**I4.** Reproduce Experiment 3: measure the cell-state gradient vs sequence length for RNN vs LSTM.

**I5.** Reproduce Experiment 4: train an LSTM on the long-lag recall task and show it beats the RNN.

**I6.** Reproduce Experiment 5: vary the forget-gate bias and measure the memory decay / gradient.

**I7.** Implement the GRU backward pass and verify against autograd.

**I8.** Train an LSTM and a GRU character-language model and compare quality and speed.

**I9.** Ablate the forget-gate bias initialization (0 vs 1 vs 2) on a long-dependency task.

**I10.** *(Vectorize.)* Implement a batched LSTM (state $(n, H)$) and confirm it matches the
per-sequence version, then compare training speed.

---

## Tier 3 — Interview

**Q1.** Why do LSTMs solve the vanishing-gradient problem?

**Q2.** What are the three LSTM gates and what does each do?

**Q3.** What is the cell state, and how is it different from the hidden state?

**Q4.** Why is the cell update "additive," and why does that matter?

**Q5.** What is a GRU and how does it differ from an LSTM?

**Q6.** Why initialize the forget-gate bias to a positive value?

**Q7.** Do gates fix exploding gradients too?

**Q8.** How does the LSTM's gradient path relate to residual connections?

**Q9.** When would you choose a GRU over an LSTM?

**Q10.** What limitation do LSTMs still have that transformers remove?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Write the LSTM and GRU equations from memory
- [ ] Derive $\partial c_T/\partial c_0 = \prod_t f_t$ and explain the constant error carousel
- [ ] Implement both cells and verify against PyTorch
- [ ] Explain the residual-connection analogy
- [ ] Justify positive forget-bias initialization
- [ ] Choose between LSTM and GRU with reasons
- [ ] State what limitation motivates attention
