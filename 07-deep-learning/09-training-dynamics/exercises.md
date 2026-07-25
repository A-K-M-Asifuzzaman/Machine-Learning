# 07.09 — Exercises: Training Dynamics & Debugging

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that a freshly-initialized softmax classifier that predicts a near-uniform distribution
has cross-entropy $\approx \log K$. Start from $L = -\log \hat p_{y}$ and set every
$\hat p_k \approx 1/K$.

**D2.** You observe an initial loss of $2\log K$ instead of $\log K$. Give two concrete bugs that
produce a loss roughly a constant factor too high, and one that produces a loss *below* $\log K$ at
step 0.

**D3.** Explain why "can you overfit a single batch?" is a test of *optimization and wiring*, not of
generalization. What does passing it rule out, and what does it leave open?

**D4.** Derive the healthy range for the update-to-weight ratio $\lVert\eta\,\Delta W\rVert/\lVert
W\rVert$. If a weight is $O(1)$ and you want it to move by $\sim 0.1\%$ per step, what does that imply
about $\eta\lVert\Delta W\rVert$? Why is a ratio of $1$ catastrophic and $10^{-6}$ useless?

**D5.** Sketch the three canonical loss-curve shapes (flat, diverging, clean descent) and state the
learning-rate action each one calls for. Add the shape for "healthy then plateaus" and say what it
means.

**D6.** In the learning-rate range test, explain why the *steepest-descent* point — not the
minimum-loss point — is the rate to pick, and why the chosen rate sits about an order of magnitude
below where the loss diverges.

**D7.** A network trains fine at batch size 32 but diverges at batch size 512 with the *same* learning
rate. Explain using the relationship between gradient-noise scale, batch size, and the stable
learning-rate ceiling.

**D8.** Explain why setting all seeds is *necessary but not sufficient* for bit-exact reproducibility,
and name two nondeterminism sources that survive seeding (GPU reductions, data-loader ordering, …).

**D9.** Gradient checking: derive the two-sided finite-difference estimate
$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon e_i) - L(\theta - \epsilon
e_i)}{2\epsilon}$ and explain why the *relative* error, not the absolute error, is the right thing to
threshold.

**D10.** Given a train loss that falls smoothly while the validation loss turns up, state precisely
which knob (capacity, regularization, learning rate, data) you would move first and why.

---

## Tier 2 — Implementation

**I1.** Reproduce Experiment 1: build a softmax classifier and confirm the initial cross-entropy
matches $\log K$ for $K \in \{2, 4, 10, 100\}$. Then deliberately break it (apply softmax twice) and
show the initial loss moves off $\log K$.

**I2.** Reproduce Experiment 2: overfit a single fixed batch of 8 examples to ~0 loss. Then plant the
"gradient killed" bug and confirm the loss stays stuck at its initial value.

**I3.** Reproduce Experiment 3: run the same network at three learning rates and label each curve
flat / diverged / converges from the loss trace alone.

**I4.** Reproduce Experiment 4: track the per-layer update-to-weight ratio during training and confirm
the healthy band is $\sim 10^{-3}$, with too-small and too-large rates on either side.

**I5.** Reproduce Experiment 5: implement the learning-rate range test (loss after a few steps at
exponentially increasing LR) and read off the sweet spot and the divergence point.

**I6.** Implement two-sided gradient checking and use it to verify one layer's analytic gradient to a
relative error below $10^{-6}$. Then introduce an off-by-one in the backward pass and watch the check
fail.

**I7.** Build a tiny "training monitor": log train/val loss, gradient norm, and the update-to-weight
ratio every $n$ steps, and print a one-line health verdict.

**I8.** Take a network that silently fails (learning rate 100x too high) and use *only* your monitor
from I7 to diagnose it — no changing the code, just reading the signals.

**I9.** *(Bug hunt.)* You are given four small training scripts, each with one planted bug (wrong loss
reduction, labels shuffled, LR too high, a detached gradient). Diagnose each from its symptom using
the §8 table.

**I10.** *(Reproducibility.)* Make a training run bit-exact reproducible: seed NumPy, control data
order, and confirm two runs produce identical loss traces. Then identify one change that breaks
determinism.

---

## Tier 3 — Interview

**Q1.** What is the very first thing you check when a model won't train?

**Q2.** Why should the initial loss be about $\log K$ for a $K$-class classifier?

**Q3.** What does "overfit a single batch" tell you, and what does it *not* tell you?

**Q4.** Your loss is flat. Walk me through your diagnosis.

**Q5.** Your loss goes to NaN after a few steps. What happened and what do you change?

**Q6.** What is the update-to-weight ratio and what is a healthy value?

**Q7.** How do you pick a learning rate without blind trial-and-error?

**Q8.** Training is fine but validation loss climbs. What do you do?

**Q9.** How do you make a training run reproducible, and why is it hard?

**Q10.** Name three bugs that produce a wrong-but-not-crashing training run and how you'd tell them
apart.

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Run the initial-loss sanity check and know what "off" means
- [ ] Overfit a single batch and know what passing/failing implies
- [ ] Read flat / diverging / clean-descent from a loss curve and act on each
- [ ] Interpret the update-to-weight ratio as a learning-rate health check
- [ ] Run a learning-rate range test and pick a rate from it
- [ ] Gradient-check a layer to machine precision
- [ ] Diagnose a silent-failure run from monitored signals alone
- [ ] Make a run reproducible and name what still breaks determinism
