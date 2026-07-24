"""
04.09 — Association Rule Mining (Apriori), from scratch (pure Python + NumPy).

Frequent-itemset mining with the anti-monotone (Apriori) pruning, plus rule generation with
support / confidence / lift. Verified against brute-force enumeration (every subset counted).
Then the chapter's claims are MEASURED:

  1. the anti-monotone principle prunes most candidates                    (README §4-§5)
  2. THE CONFIDENCE TRAP: confidence ranks a trivial rule above a real one; lift fixes it (§7)
  3. min_support controls the combinatorial explosion of itemsets          (README §9)
  4. lift < 1 detects NEGATIVE association                                 (README §3)
  5. a planted strong rule is recovered at high lift                       (README §6)

Run:  python3 from_scratch.py
"""

from itertools import combinations, chain
import numpy as np


# =============================================================================
# APRIORI  (README §5)
# =============================================================================


def _support_counts(transactions, candidates):
    """Count how many transactions contain each candidate itemset."""
    counts = {c: 0 for c in candidates}
    for t in transactions:
        ts = set(t)
        for c in candidates:
            if c.issubset(ts):
                counts[c] += 1
    return counts


def apriori(transactions, min_support=0.2, _track=None):
    """Return {frozenset: support} for all frequent itemsets. If `_track` is a dict, record
    the number of candidates counted per level (to measure pruning)."""
    n = len(transactions)
    items = sorted({i for t in transactions for i in t})
    # L1: frequent singletons
    L1 = {frozenset([i]): 0 for i in items}
    for t in transactions:
        for i in set(t):
            L1[frozenset([i])] += 1
    freq = {k: v / n for k, v in L1.items() if v / n >= min_support}
    if _track is not None:
        _track[1] = len(items)
    all_freq = dict(freq)

    k = 2
    current = list(freq.keys())
    while current:
        # candidate generation: join frequent (k-1)-itemsets sharing k-2 items
        cand = set()
        for a in range(len(current)):
            for b in range(a + 1, len(current)):
                union = current[a] | current[b]
                if len(union) == k:
                    cand.add(union)
        # PRUNE: drop candidates with any infrequent (k-1)-subset (anti-monotone, README §4)
        pruned = [c for c in cand
                  if all(frozenset(s) in all_freq for s in combinations(c, k - 1))]
        if _track is not None:
            _track[k] = len(pruned)
        if not pruned:
            break
        counts = _support_counts(transactions, pruned)
        Lk = {c: v / n for c, v in counts.items() if v / n >= min_support}
        all_freq.update(Lk)
        current = list(Lk.keys())
        k += 1
    return all_freq


def brute_force_frequent(transactions, min_support=0.2):
    """Count EVERY possible itemset (for verification). Exact but exponential."""
    n = len(transactions)
    items = sorted({i for t in transactions for i in t})
    freq = {}
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            s = frozenset(combo)
            sup = sum(1 for t in transactions if s.issubset(set(t))) / n
            if sup >= min_support:
                freq[s] = sup
    return freq


# =============================================================================
# RULE GENERATION  (README §6)
# =============================================================================


def generate_rules(freq_itemsets, min_confidence=0.5):
    """Return list of (X, Y, support, confidence, lift) for rules X => Y."""
    rules = []
    n_support = freq_itemsets                      # {frozenset: support}
    for Z, supZ in freq_itemsets.items():
        if len(Z) < 2:
            continue
        items = list(Z)
        # every non-empty proper subset X as antecedent
        for r in range(1, len(items)):
            for X in combinations(items, r):
                X = frozenset(X)
                Y = Z - X
                supX = n_support.get(X)
                supY = n_support.get(Y)
                if supX is None:
                    continue
                conf = supZ / supX
                if conf >= min_confidence:
                    lift = conf / supY if supY else float("nan")
                    rules.append((X, Y, supZ, conf, lift))
    return rules


# =============================================================================
# VERIFICATION
# =============================================================================


def _fmt(s):
    return "{" + ",".join(sorted(s)) + "}"


def _random_transactions(n=400, n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    items = [f"i{j}" for j in range(n_items)]
    txns = []
    for _ in range(n):
        size = rng.integers(2, 6)
        txns.append(list(rng.choice(items, size, replace=False)))
    return txns


def verify():
    print("=" * 88)
    print("VERIFICATION — Apriori vs brute-force enumeration")
    print("=" * 88)
    txns = _random_transactions(400, 12, seed=0)
    for ms in (0.1, 0.2, 0.05):
        track = {}
        ap = apriori(txns, min_support=ms, _track=track)
        bf = brute_force_frequent(txns, min_support=ms)
        same = set(ap.keys()) == set(bf.keys())
        max_sup_diff = max((abs(ap[k] - bf[k]) for k in ap), default=0.0)
        print(f"\n  min_support={ms}: Apriori found {len(ap)} frequent itemsets, "
              f"brute force {len(bf)}")
        print(f"    identical set of itemsets: {same}, max support diff: {max_sup_diff:.1e}")
        assert same, "Apriori must find exactly the brute-force frequent itemsets"
    print("\n  Apriori's frequent itemsets exactly match brute-force enumeration  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — anti-monotone pruning (README §4-§5)
# =============================================================================


def experiment_1_pruning():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the anti-monotone principle prunes most candidates (README §4-§5)")
    print("=" * 88)
    txns = _random_transactions(500, 15, seed=1)
    ms = 0.05
    track = {}
    apriori(txns, min_support=ms, _track=track)
    n_items = 15
    # brute force would consider all C(15,k) itemsets at each level
    from math import comb
    print(f"\n  15 items, min_support={ms}. Candidates counted at each level:\n")
    print(f"    {'level k':>8s} {'brute-force C(15,k)':>20s} {'Apriori candidates':>20s}")
    total_bf, total_ap = 0, 0
    for k in sorted(track):
        bf = comb(n_items, k)
        total_bf += bf
        total_ap += track[k]
        print(f"    {k:>8d} {bf:>20d} {track[k]:>20d}")
    print(f"""
    total itemsets considered: brute force {total_bf}, Apriori {total_ap}
    -> Apriori counted {100*(1-total_ap/total_bf):.0f}% fewer candidates

  READING: at each level Apriori only forms candidates whose subsets are ALL frequent, so the
  supersets of every infrequent itemset are never generated. It counts far fewer candidates than
  the brute-force 2^d lattice — the anti-monotone principle is what makes frequent-itemset mining
  tractable (README §4).""")


# =============================================================================
# EXPERIMENT 2 — the confidence trap (README §7)
# =============================================================================


def experiment_2_confidence_trap():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the confidence trap: confidence misleads, lift corrects (README §7)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n = 2000
    txns = []
    for _ in range(n):
        t = []
        # MILK is in 90% of baskets (very common) — independent of everything
        if rng.uniform() < 0.9:
            t.append("milk")
        # PICKLES in 20%, INDEPENDENT of milk (no real association)
        if rng.uniform() < 0.2:
            t.append("pickles")
        # DIAPERS in 20%; BEER strongly associated with diapers (real pattern)
        has_diapers = rng.uniform() < 0.2
        if has_diapers:
            t.append("diapers")
            if rng.uniform() < 0.6:                 # 60% of diaper baskets have beer
                t.append("beer")
        elif rng.uniform() < 0.1:                    # 10% of others have beer
            t.append("beer")
        txns.append(t)

    freq = apriori(txns, min_support=0.02)
    rules = generate_rules(freq, min_confidence=0.0)

    def find(x, y):
        X, Y = frozenset([x]), frozenset([y])
        for r in rules:
            if r[0] == X and r[1] == Y:
                return r
        return None

    r_trivial = find("pickles", "milk")             # high confidence (milk common), lift ~1
    r_real = find("diapers", "beer")                # lower confidence, high lift
    print(f"""
    {'rule':>22s} {'confidence':>12s} {'lift':>8s}
    {'pickles => milk':>22s} {r_trivial[3]:>12.2f} {r_trivial[4]:>8.2f}
    {'diapers => beer':>22s} {r_real[3]:>12.2f} {r_real[4]:>8.2f}

  READING: 'pickles => milk' has HIGHER confidence ({r_trivial[3]:.2f}) than 'diapers => beer'
  ({r_real[3]:.2f}) — but only because milk is in 90% of ALL baskets, so ANY rule predicting milk
  looks confident. Its LIFT is ~1.0 (independent, no real association). 'diapers => beer' has lift
  {r_real[4]:.1f} — beer is {r_real[4]:.1f}x more likely in diaper baskets — a GENUINE pattern.
  Confidence ranks them BACKWARDS; lift ranks them correctly. Never rank rules by confidence alone
  (README §7).""")


# =============================================================================
# EXPERIMENT 3 — min_support and combinatorial explosion (README §9)
# =============================================================================


def experiment_3_explosion():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — min_support controls the combinatorial explosion (README §9)")
    print("=" * 88)
    txns = _random_transactions(600, 18, seed=2)
    print(f"\n    {'min_support':>12s} {'# frequent itemsets':>20s} {'max itemset size':>18s}")
    for ms in (0.30, 0.15, 0.08, 0.04, 0.02):
        freq = apriori(txns, min_support=ms)
        max_size = max((len(k) for k in freq), default=0)
        print(f"    {ms:>12.2f} {len(freq):>20d} {max_size:>18d}")
    print("""
  READING: lowering min_support finds more (and larger) frequent itemsets, but the count grows
  fast — at low support the number of itemsets (and the compute) can explode combinatorially. The
  support threshold is the main control on both output size and runtime; set it to get a
  manageable number of patterns (README §9).""")


# =============================================================================
# EXPERIMENT 4 — lift detects negative association (README §3)
# =============================================================================


def experiment_4_negative():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — lift < 1 detects NEGATIVE association (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n = 2000
    txns = []
    for _ in range(n):
        t = []
        # coke and pepsi are SUBSTITUTES — rarely bought together (negative association)
        buys_coke = rng.uniform() < 0.4
        if buys_coke:
            t.append("coke")
            if rng.uniform() < 0.05:                # only 5% of coke buyers also buy pepsi
                t.append("pepsi")
        else:
            if rng.uniform() < 0.4:                 # 40% of non-coke buyers buy pepsi
                t.append("pepsi")
        if rng.uniform() < 0.5:
            t.append("chips")
        txns.append(t)

    freq = apriori(txns, min_support=0.01)
    rules = generate_rules(freq, min_confidence=0.0)
    for r in rules:
        if r[0] == frozenset(["coke"]) and r[1] == frozenset(["pepsi"]):
            coke_pepsi = r
    print(f"""
  Coke and pepsi are substitutes (people buy one OR the other):

    rule: coke => pepsi
    confidence = {coke_pepsi[3]:.3f}
    lift       = {coke_pepsi[4]:.3f}   (< 1: NEGATIVE association)

  READING: lift below 1 means the items appear together LESS than if independent — a negative
  association. Here buying coke makes pepsi {1/coke_pepsi[4]:.1f}x LESS likely (they are
  substitutes). Support and confidence cannot express this (confidence is just a conditional
  probability); only lift reveals whether the association is positive, none, or negative (README §3).""")


# =============================================================================
# EXPERIMENT 5 — recover a planted rule (README §6)
# =============================================================================


def experiment_5_planted():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — a planted strong rule is recovered at high lift (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    n = 3000
    items = [f"x{j}" for j in range(10)]
    txns = []
    for _ in range(n):
        t = list(rng.choice(items, rng.integers(1, 4), replace=False))
        # PLANT: whenever {x0, x1} appear together, add x2 with 80% probability
        if "x0" in t and "x1" in t and rng.uniform() < 0.8:
            t.append("x2")
        txns.append(list(set(t)))

    freq = apriori(txns, min_support=0.01)
    rules = generate_rules(freq, min_confidence=0.3)
    # find the planted rule {x0,x1} => {x2}
    planted = [r for r in rules if r[0] == frozenset(["x0", "x1"]) and r[1] == frozenset(["x2"])]
    top = sorted(rules, key=lambda r: -r[4])[:5]
    print(f"\n  Planted: baskets with {{x0, x1}} get x2 80% of the time. Top rules by lift:\n")
    print(f"    {'rule':>20s} {'support':>9s} {'confidence':>12s} {'lift':>7s}")
    for X, Y, sup, conf, lift in top:
        print(f"    {_fmt(X) + ' => ' + _fmt(Y):>20s} {sup:>9.3f} {conf:>12.2f} {lift:>7.2f}")
    if planted:
        p = planted[0]
        print(f"""
  The planted rule {{x0,x1}} => {{x2}} is recovered: confidence {p[3]:.2f}, lift {p[4]:.2f}.

  READING: Apriori finds the planted association and rule generation surfaces it with high
  confidence (~0.8, matching the plant) and lift well above 1 (x2 is far more likely given
  {{x0,x1}} than at baseline). Ranking rules by lift puts the genuine, strongest associations on
  top — the intended output of association mining (README §6).""")


if __name__ == "__main__":
    verify()
    experiment_1_pruning()
    experiment_2_confidence_trap()
    experiment_3_explosion()
    experiment_4_negative()
    experiment_5_planted()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
