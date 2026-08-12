# 13.01 — Exercises: MDPs & Dynamic Programming

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Define an MDP and state the Markov property. Why is it a "sufficient summary"?

**D2.** Define the return $G_t$ and the value functions $V^\pi$ and $Q^\pi$. Relate them.

**D3.** Derive the Bellman expectation equation for $V^\pi$ and write it as the linear system
$V^\pi = R_\pi + \gamma P_\pi V^\pi$.

**D4.** Derive the Bellman optimality equation for $V^*$ and $Q^*$.

**D5.** Prove the Bellman operator is a $\gamma$-contraction in the max-norm.

**D6.** Use the contraction + Banach fixed-point theorem to prove value iteration converges to a unique
$V^*$.

**D7.** State and prove the policy-improvement theorem.

**D8.** Show policy iteration converges in finitely many steps.

**D9.** Explain how $\gamma$ can change the optimal policy, with the near/far-reward example.

**D10.** Explain what changes when the model $(P, R)$ is unknown — the leap from DP to RL.

---

## Tier 2 — Implementation

**I1.** Build a gridworld MDP ($P$, $R$) and implement iterative policy evaluation; verify against the
exact linear solve (Experiment 1).

**I2.** Implement value iteration; verify Bellman optimality and visualize the policy (Experiment 2).

**I3.** Implement policy iteration; verify it matches value iteration (Experiment 3).

**I4.** Reproduce Experiment 4: measure the error ratio per sweep and confirm it $\to \gamma$.

**I5.** Reproduce Experiment 5: show $\gamma$ changing the optimal action on a near/far-reward chain.

**I6.** Add stochastic transitions (slip probability) and re-run all algorithms.

**I7.** Implement asynchronous / in-place value iteration and compare convergence.

**I8.** Implement generalized policy iteration with $k$ evaluation sweeps per improvement and study the
$k$ trade-off.

**I9.** Solve a larger MDP (e.g. FrozenLake) with DP and compare to a known solution.

**I10.** *(Horizon.)* Sweep $\gamma$ and plot the optimal policy's behavior vs planning horizon.

---

## Tier 3 — Interview

**Q1.** What is a Markov Decision Process?

**Q2.** What is the Markov property?

**Q3.** What is the difference between $V$ and $Q$?

**Q4.** What is the Bellman equation?

**Q5.** How does value iteration work?

**Q6.** How does policy iteration differ from value iteration?

**Q7.** Why do these algorithms converge?

**Q8.** What does the discount factor do?

**Q9.** What is the difference between prediction and control?

**Q10.** What changes when you don't know the model?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Define an MDP and the value functions
- [ ] Derive the Bellman expectation and optimality equations
- [ ] Implement policy evaluation, value iteration, and policy iteration
- [ ] Prove the contraction and convergence
- [ ] Explain the role of the discount factor
- [ ] Distinguish prediction from control
- [ ] Explain the DP → RL transition
