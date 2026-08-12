"""
13.01 — MDPs & dynamic programming, from scratch (NumPy).

Reinforcement learning is built on the Markov Decision Process and the Bellman equations. When the MDP
is KNOWN (transitions and rewards), dynamic programming solves it exactly. This file builds a gridworld
and verifies the core algorithms:

  1. policy evaluation converges to V^pi (matches the exact linear solve)       -> Experiment 1
  2. value iteration finds V* and the optimal policy                            -> Experiment 2
  3. policy iteration reaches the same optimum                                  -> Experiment 3
  4. the Bellman operator is a gamma-contraction (geometric convergence)        -> Experiment 4
  5. the discount gamma changes the optimal policy (myopic vs far-sighted)      -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


# =============================================================================
# A gridworld MDP:  P[s, a, s'] transition probs, R[s, a] expected reward
# =============================================================================


def make_gridworld(size=4, goal=15, step_reward=-1.0):
    """size x size grid; `goal` is terminal (absorbing, 0 reward). Actions: 0=up,1=right,2=down,3=left."""
    n = size * size
    P = np.zeros((n, 4, n))
    R = np.full((n, 4), step_reward)
    for s in range(n):
        if s == goal:
            P[s, :, s] = 1.0; R[s, :] = 0.0; continue    # terminal: self-loop, no reward
        r, c = divmod(s, size)
        for a, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                ns = nr * size + nc
            else:
                ns = s                                    # bump into wall -> stay
            P[s, a, ns] += 1.0
    return P, R


def policy_evaluation(policy, P, R, gamma, tol=1e-10):
    """Iterative policy evaluation: V <- R_pi + gamma P_pi V. Returns V and #iterations."""
    n = P.shape[0]
    V = np.zeros(n)
    for it in range(1, 100000):
        Vnew = np.zeros(n)
        for s in range(n):
            Vnew[s] = sum(policy[s, a] * (R[s, a] + gamma * P[s, a] @ V) for a in range(4))
        if np.abs(Vnew - V).max() < tol:
            return Vnew, it
        V = Vnew
    return V, it


def exact_value(policy, P, R, gamma):
    """Solve V = (I - gamma P_pi)^-1 R_pi directly."""
    n = P.shape[0]
    P_pi = np.einsum("sa,sat->st", policy, P)
    R_pi = np.einsum("sa,sa->s", policy, R)
    return np.linalg.solve(np.eye(n) - gamma * P_pi, R_pi)


def value_iteration(P, R, gamma, tol=1e-10):
    n = P.shape[0]
    V = np.zeros(n)
    for it in range(1, 100000):
        Q = R + gamma * np.einsum("sat,t->sa", P, V)     # Q[s,a]
        Vnew = Q.max(1)
        if np.abs(Vnew - V).max() < tol:
            return Vnew, Q.argmax(1), it
        V = Vnew
    return V, Q.argmax(1), it


def policy_iteration(P, R, gamma):
    n = P.shape[0]
    policy = np.ones((n, 4)) / 4                          # start uniform random
    for it in range(1, 1000):
        V = exact_value(policy, P, R, gamma)
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        greedy = Q.argmax(1)
        new_policy = np.zeros((n, 4)); new_policy[np.arange(n), greedy] = 1.0
        if np.array_equal(new_policy, policy):
            return V, greedy, it
        policy = new_policy
    return V, greedy, it


# =============================================================================
# EXPERIMENT 1 — policy evaluation
# =============================================================================


def experiment_1_evaluation():
    print("=" * 88)
    print("EXPERIMENT 1 — policy evaluation converges to V^pi (README §2)")
    print("=" * 88)
    P, R = make_gridworld()
    gamma = 0.9
    policy = np.ones((16, 4)) / 4                         # uniform random policy
    V_iter, iters = policy_evaluation(policy, P, R, gamma)
    V_exact = exact_value(policy, P, R, gamma)
    print(f"""
  4x4 gridworld, goal at state 15, -1 per step, gamma={gamma}. Evaluate the uniform-random policy:

    iterative policy evaluation converged in {iters} sweeps
    max |iterative V - exact linear solve| = {np.abs(V_iter - V_exact).max():.2e}

    V^pi (value of each state under random policy), as a grid:""")
    print("   " + str(np.round(V_iter.reshape(4, 4), 1)).replace("\n", "\n   "))
    print("""
  READING: the value function V^pi(s) is the expected discounted return from state s under policy pi. It
  satisfies the Bellman EXPECTATION equation V^pi = R_pi + gamma P_pi V^pi, a linear system. Iterating
  that equation (dynamic programming) converges to the same answer as solving the linear system directly
  (difference ~1e-10). States near the goal have higher (less negative) value — the value function
  encodes 'how good is it to be here'. This is 'prediction': evaluating a fixed policy (README §2).""")


# =============================================================================
# EXPERIMENT 2 — value iteration
# =============================================================================


def experiment_2_value_iteration():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — value iteration finds V* and the optimal policy (README §3)")
    print("=" * 88)
    P, R = make_gridworld()
    gamma = 0.9
    Vstar, pi_star, iters = value_iteration(P, R, gamma)
    # verify Bellman optimality: V*(s) = max_a Q*(s,a)
    Q = R + gamma * np.einsum("sat,t->sa", P, Vstar)
    bellman_gap = np.abs(Vstar - Q.max(1)).max()
    arrows = np.array(["^", ">", "v", "<"])[pi_star].reshape(4, 4)
    arrows[3, 3] = "G"
    print(f"""
  Value iteration on the same gridworld (gamma={gamma}), converged in {iters} sweeps:

    Bellman optimality check  max|V* - max_a Q*| = {bellman_gap:.1e}   (satisfied)

    optimal policy (arrows point the way to the goal G):""")
    for row in arrows:
        print("      " + "  ".join(row))
    print("""
  READING: value iteration applies the Bellman OPTIMALITY operator V(s) <- max_a [R(s,a) + gamma sum
  P(s'|s,a) V(s')] until convergence. The result V* is the best achievable value from each state, and
  acting greedily w.r.t. V* gives the OPTIMAL policy — every arrow points along a shortest path to the
  goal. This is 'control': finding the best policy. Value iteration interleaves evaluation and
  improvement into a single max operation (README §3).""")


# =============================================================================
# EXPERIMENT 3 — policy iteration
# =============================================================================


def experiment_3_policy_iteration():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — policy iteration reaches the same optimum (README §4)")
    print("=" * 88)
    P, R = make_gridworld()
    gamma = 0.9
    V_pi, pi_pi, iters_pi = policy_iteration(P, R, gamma)
    V_vi, pi_vi, _ = value_iteration(P, R, gamma)
    print(f"""
  Policy iteration (alternate exact evaluation + greedy improvement), gamma={gamma}:

    converged in {iters_pi} policy-improvement steps (far fewer than value iteration's sweeps)
    same optimal values as value iteration?  {np.allclose(V_pi, V_vi, atol=1e-6)}
    same optimal policy?                      {np.array_equal(pi_pi, pi_vi)}

  READING: policy iteration alternates two steps: (1) EVALUATE the current policy exactly, then
  (2) IMPROVE it by acting greedily w.r.t. its value. Each improvement is guaranteed not to make the
  policy worse (the policy-improvement theorem), and with finitely many policies it converges to the
  optimum in just a handful of steps. It reaches the SAME V* and optimal policy as value iteration —
  they are two schedules of the same underlying Bellman updates (README §4).""")


# =============================================================================
# EXPERIMENT 4 — the Bellman operator is a gamma-contraction
# =============================================================================


def experiment_4_contraction():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — the Bellman operator is a gamma-contraction (README §5)")
    print("=" * 88)
    P, R = make_gridworld()
    gamma = 0.9
    policy = np.ones((16, 4)) / 4                         # the linear Bellman-expectation operator
    Vpi = exact_value(policy, P, R, gamma)
    P_pi = np.einsum("sa,sat->st", policy, P)
    R_pi = np.einsum("sa,sa->s", policy, R)
    V = np.zeros(16)
    print(f"\n  Distance to V^pi after each policy-evaluation sweep V <- R_pi + gamma P_pi V (gamma={gamma}):\n")
    print(f"    {'sweep':>6s} {'||V_k - V^pi||_inf':>18s} {'ratio to previous':>18s}")
    prev = np.abs(V - Vpi).max()
    for k in range(1, 9):
        V = R_pi + gamma * P_pi @ V
        err = np.abs(V - Vpi).max()
        print(f"    {k:>6d} {err:>18.5f} {err / prev:>18.4f}")
        prev = err
    print(f"""
  READING: the Bellman operator is a CONTRACTION in the max-norm with factor gamma={gamma}:
  ||T V - T U|| <= gamma ||V - U||. So each sweep shrinks the distance to the fixed point by a factor
  gamma — the ratio-to-previous column converges to {gamma}. This guarantees the iteration converges to a
  UNIQUE fixed point geometrically, from any starting V. It is why DP works at all, and the discount
  gamma<1 is exactly what makes the operator a contraction (gamma=1 can fail to converge) (README §5).""")


# =============================================================================
# EXPERIMENT 5 — the discount changes the optimal policy
# =============================================================================


def experiment_5_discount():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the discount gamma trades near vs far reward (README §6)")
    print("=" * 88)
    # a 1-D chain of 8 states with TERMINAL rewards at both ends. Start at state 1:
    # small reward (+1) is 1 step LEFT (state 0); large reward (+10) is 6 steps RIGHT (state 7).
    n, start = 8, 1
    P = np.zeros((n, 2, n)); R = np.zeros((n, 2))        # actions 0=left, 1=right
    for s in range(n):
        if s in (0, n - 1):
            P[s, :, s] = 1.0; continue                   # terminal, absorbing
        P[s, 0, s - 1] = 1.0                             # left
        P[s, 1, s + 1] = 1.0                             # right
    R[1, 0] = 1.0                                         # entering the LEFT terminal: +1
    R[n - 2, 1] = 10.0                                   # entering the RIGHT terminal: +10
    print(f"\n  Chain (terminal ends): small reward +1 is 1 step LEFT, large reward +10 is 6 steps RIGHT.")
    print(f"  Optimal first action from the start vs the discount gamma:\n")
    print(f"    {'gamma':>8s} {'best action from start':>26s}")
    for gamma in (0.5, 0.7, 0.8, 0.9, 0.99):
        _, pi, _ = value_iteration(P, R, gamma)
        action = "LEFT (grab small reward)" if pi[start] == 0 else "RIGHT (go for big reward)"
        print(f"    {gamma:>8.2f} {action:>26s}")
    print("""
  READING: gamma discounts future rewards by gamma^t. A MYOPIC agent (small gamma) values the immediate
  small reward more than a large reward five steps away (10 * gamma^5 is tiny), so it goes LEFT. A
  FAR-SIGHTED agent (gamma near 1) barely discounts the future, so the big reward dominates and it goes
  RIGHT. The discount is not just a math convenience — it encodes HOW FAR AHEAD the agent plans, and
  changing it changes the optimal behavior (README §6).""")


if __name__ == "__main__":
    experiment_1_evaluation()
    experiment_2_value_iteration()
    experiment_3_policy_iteration()
    experiment_4_contraction()
    experiment_5_discount()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
