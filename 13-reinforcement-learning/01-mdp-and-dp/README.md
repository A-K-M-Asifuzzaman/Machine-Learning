# 13.01 — MDPs & Dynamic Programming

> **Reinforcement learning is the science of making sequences of decisions, and it all rests on one
> equation.** Formalize the world as a Markov Decision Process — states, actions, transitions, rewards,
> and a discount — and the value of every state is defined *recursively* by the Bellman equation: the
> value of *here* is the immediate reward plus the discounted value of *where you go next*. When you
> know the world's dynamics, dynamic programming solves that equation exactly, giving the optimal policy.
> This chapter is the mathematical bedrock of everything in Part 13.

Supervised learning ([Part 3](../../03-supervised-learning/)) learns from labeled examples;
reinforcement learning learns from *reward* by *interacting*. There are no labels — only a scalar signal
that may be delayed, so the agent must figure out which actions led to good outcomes (the **credit
assignment** problem). The MDP is the framework, and the Bellman equation is the tool.

## Table of contents

1. [The Markov Decision Process](#1-the-markov-decision-process)
2. [Value functions and the Bellman equation](#2-value-functions-and-the-bellman-equation)
3. [Value iteration](#3-value-iteration)
4. [Policy iteration](#4-policy-iteration)
5. [Why it converges: contraction](#5-why-it-converges-contraction)
6. [The discount factor](#6-the-discount-factor)
7. [From DP to RL](#7-from-dp-to-rl)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The Markov Decision Process

An MDP is a tuple $(S, A, P, R, \gamma)$: **states** $S$, **actions** $A$, **transition** probabilities
$P(s' \mid s, a)$, **reward** $R(s, a)$, and a **discount** $\gamma \in [0, 1)$. The defining **Markov
property**: the next state depends only on the current state and action, not the full history — the
present is a sufficient summary. The agent follows a **policy** $\pi(a \mid s)$ and seeks to maximize the
expected **return** — the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k\, R_{t+k}.
$$

The discount $\gamma$ makes the infinite sum finite and encodes how far ahead the agent plans (§6).

## 2. Value functions and the Bellman equation

The **state-value** $V^\pi(s)$ is the expected return starting from $s$ under $\pi$; the **action-value**
$Q^\pi(s,a)$ starts by taking $a$. Both satisfy a *recursive* **Bellman expectation equation** — value
here = reward now + discounted value next:

$$
V^\pi(s) = \sum_a \pi(a\mid s)\Big[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\,V^\pi(s')\Big].
$$

This is a linear system, $V^\pi = R_\pi + \gamma P_\pi V^\pi$, solvable directly. But iterating it —
**policy evaluation** — also converges to the same answer. Experiment 1 evaluates the uniform-random
policy on a 4×4 gridworld (goal at corner, −1 per step, $\gamma = 0.9$):

```
V^pi:   -9.4  -9.2  -9.0  -8.8
        -9.2  -9.0  -8.5  -8.0
        -9.0  -8.5  -7.4  -5.8
        -8.8  -8.0  -5.8   0.0   (goal)
```

The iterative and exact solutions agree to $7\times10^{-10}$. States near the goal have higher (less
negative) value — the value function encodes "how good is it to be here." This is **prediction**:
evaluating a *fixed* policy.

## 3. Value iteration

**Control** — finding the *best* policy — uses the **Bellman optimality equation**, which replaces the
policy average with a **max**:

$$
V^*(s) = \max_a \Big[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\,V^*(s')\Big].
$$

**Value iteration** applies this update until convergence, then acts greedily. Experiment 2 (same
gridworld) converges in 7 sweeps to $V^*$ satisfying Bellman optimality exactly, and the greedy policy
sends every state along a shortest path to the goal:

```
optimal policy:   >  >  >  v
                  >  >  >  v
                  >  >  >  v
                  >  >  >  G
```

Value iteration folds evaluation and improvement into a single max operation.

## 4. Policy iteration

**Policy iteration** does the two steps explicitly: (1) **evaluate** the current policy (§2), then
(2) **improve** it by acting greedily w.r.t. its value. The **policy improvement theorem** guarantees each
improvement is no worse, and since there are finitely many deterministic policies, it converges to the
optimum — usually in very few steps. Experiment 3 reaches the **same** $V^*$ and optimal policy as value
iteration in just **3 improvement steps**. Value iteration and policy iteration are two schedules of the
same underlying Bellman updates — one takes many cheap sweeps, the other few expensive ones.

## 5. Why it converges: contraction

Both methods work because the Bellman operator $T$ is a **$\gamma$-contraction** in the max-norm:

$$
\lVert T V - T U \rVert_\infty \le \gamma\,\lVert V - U \rVert_\infty.
$$

Each application shrinks the distance to the fixed point by a factor $\gamma$, so iteration converges
**geometrically to a unique fixed point** from any start (Banach fixed-point theorem). Experiment 4
measures the error ratio per sweep — it converges to **0.89 ≈ $\gamma = 0.9$**:

| Sweep | $\lVert V_k - V^\pi\rVert_\infty$ | Ratio |
|:--:|:--:|:--:|
| 1 | 8.36 | 0.893 |
| 4 | 5.92 | 0.890 |
| 8 | 3.67 | 0.886 |

**The discount $\gamma < 1$ is precisely what makes the operator a contraction** — with $\gamma = 1$
convergence can fail. This is why DP works at all.

## 6. The discount factor

$\gamma$ is not just a math convenience — it sets **how far ahead the agent plans**. Experiment 5 puts a
small reward (+1) one step to the left and a large reward (+10) six steps to the right, and finds the
optimal first move:

| $\gamma$ | Best action |
|:--:|:--:|
| 0.50 | **LEFT** (grab the small reward) |
| 0.70 | RIGHT (go for the big reward) |
| 0.90 | RIGHT |
| 0.99 | RIGHT |

A **myopic** agent (small $\gamma$) values the immediate small reward more than a large reward six steps
away ($10\gamma^6$ is tiny), so it goes left. A **far-sighted** agent ($\gamma \to 1$) barely discounts
the future, so the big reward dominates. Changing $\gamma$ **changes the optimal behavior** — it is a
first-class part of the problem definition.

## 7. From DP to RL

Dynamic programming needs the **model** — the transition probabilities $P$ and rewards $R$. Real agents
usually **don't have it**. That is the leap from DP to reinforcement learning proper: learn to act
optimally *without* knowing the dynamics, by **sampling** experience instead of computing expectations.
The Bellman equations still hold — but now their expectations are estimated from data:

- **Model-free prediction/control** ([13.02](../02-model-free/)) — Monte Carlo and temporal-difference
  learning replace the known $P$ with sampled transitions (Q-learning, SARSA).
- **Deep RL** ([13.03](../03-deep-rl/)) — use a neural network to approximate $V$/$Q$ over huge state
  spaces (DQN).
- **Policy gradients** ([13.04](../04-policy-gradients/)) — optimize the policy directly.

Every one of these is the Bellman equation, approximated. This chapter is the exact case that anchors
them all.

## 8. Common misconceptions

- **"RL needs a reward at every step."** Rewards can be sparse and delayed; the value function propagates
  them backward through the Bellman equation (§2).
- **"The value function is the reward."** It is the *expected discounted return* — cumulative future
  reward, not the immediate one (§2).
- **"Value iteration and policy iteration give different answers."** They converge to the same $V^*$ and
  optimal policy (§3–§4).
- **"$\gamma$ is a minor hyperparameter."** It defines the planning horizon and can change the optimal
  policy (§6).
- **"DP is real RL."** DP assumes a known model; RL's defining challenge is learning without one (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — an MDP and the DP algorithms in NumPy. Five experiments:
  (1) policy evaluation matches the exact linear solve; (2) value iteration finds the optimal policy;
  (3) policy iteration reaches the same optimum in 3 steps; (4) the Bellman operator's error ratio
  converges to $\gamma$; (5) the discount trades near vs far reward.
- **[exercises.md](exercises.md)** — derive the Bellman equations, prove the contraction, implement the
  three algorithms.
- **[references.md](references.md)** — Bellman, Sutton & Barto, and the DP literature.

## Where this leads

- **Model-free RL — learning without the dynamics** → [13.02](../02-model-free/)
- **Deep RL — function approximation** → [13.03](../03-deep-rl/)
- **Policy gradients** → [13.04](../04-policy-gradients/)
- **RLHF, which optimizes a KL-regularized MDP** → [11.06](../../11-transformers-and-llms/06-alignment/)
- **Probability and expectation** → [00.03](../../00-mathematical-foundations/03-probability/)
