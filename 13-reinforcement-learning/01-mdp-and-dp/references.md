# 13.01 — References: MDPs & Dynamic Programming

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | MDPs, Bellman equations | Bellman (1957); Sutton & Barto Ch. 3 |
| §3-§4 | Value/policy iteration | Sutton & Barto Ch. 4; Howard (1960) |
| §5 | Contraction, convergence | Puterman (1994); Bertsekas |
| §6 | Discounting | Sutton & Barto §3.3 |

---

## The reference text

**Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).** — THE textbook.
**Chapter 3** (finite MDPs, returns, value functions, Bellman equations) and **Chapter 4** (dynamic
programming: policy evaluation, policy iteration, value iteration, GPI). Free at
<http://incompleteideas.net/book/the-book-2nd.html>.

---

## Foundational

- **Bellman, R. (1957). *Dynamic Programming*.** Princeton University Press. — the Bellman equation and
  the principle of optimality (§2).
- **Howard, R. (1960). *Dynamic Programming and Markov Processes*.** — **policy iteration** (§4).
- **Puterman, M. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*.** — the
  rigorous treatment: contraction mappings, convergence proofs (§5).
- **Bertsekas, D. *Dynamic Programming and Optimal Control*.** — the optimization-theoretic view.

---

## Courses

- **David Silver, "RL Course" (DeepMind/UCL).** — lectures 1–3 cover MDPs and DP.
  <https://www.davidsilver.uk/teaching/>.
- **Stanford CS234, "Reinforcement Learning."** <https://web.stanford.edu/class/cs234/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | standard RL environments (FrozenLake, etc.) |
| [Sutton & Barto code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) | worked examples for every chapter |

---

## Deferred to later chapters

- **Model-free RL (MC, TD, Q-learning)** → [13.02](../02-model-free/)
- **Deep RL (DQN)** → [13.03](../03-deep-rl/)
- **Policy gradients** → [13.04](../04-policy-gradients/)
- **Bandits (the one-state MDP)** → [13.05](../05-bandits/)
- **RLHF** → [11.06](../../11-transformers-and-llms/06-alignment/)
