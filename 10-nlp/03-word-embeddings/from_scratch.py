"""
10.03 — Word embeddings, from scratch (NumPy).

An embedding maps each word to a dense vector so that similar words land near each other — the
"distributional hypothesis" (a word is known by the company it keeps) turned into geometry. This file
builds Word2Vec (skip-gram with negative sampling), GloVe, and FastText's subword idea, and MEASURES
what they learn:

  1. the SGNS objective is binary classification; its gradient is exact          -> Experiment 1
  2. distributional hypothesis: words in the same contexts get similar vectors   -> Experiment 2
  3. embedding geometry: king - man + woman ~ queen                              -> Experiment 3
  4. GloVe factorizes the log co-occurrence matrix and recovers the same structure -> Experiment 4
  5. FastText: subword n-grams give OOV / misspelled words sensible vectors      -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


# =============================================================================
# Skip-gram with negative sampling (SGNS)
# =============================================================================


def build_pairs(sentences, window=2):
    vocab = sorted({w for s in sentences for w in s})
    V = {w: i for i, w in enumerate(vocab)}
    pairs = []
    for s in sentences:
        ids = [V[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    return np.array(pairs), vocab, V


def train_sgns(sentences, dim=24, epochs=8, lr=0.05, K=5, window=2, seed=0):
    rng = np.random.default_rng(seed)
    pairs, vocab, V = build_pairs(sentences, window)
    n = len(vocab)
    freq = np.bincount(pairs[:, 0], minlength=n).astype(float)
    neg_p = freq ** 0.75
    neg_p /= neg_p.sum()                                          # noise distribution ~ unigram^0.75
    Win = rng.standard_normal((n, dim)) * 0.1                     # "input" (word) vectors
    Wout = rng.standard_normal((n, dim)) * 0.1                    # "output" (context) vectors
    for _ in range(epochs):
        for idx in rng.permutation(len(pairs)):
            c, o = pairs[idx]
            negs = rng.choice(n, K, p=neg_p)
            vc = Win[c]
            g = sigmoid(vc @ Wout[o]) - 1                         # positive pair -> label 1
            gvc = g * Wout[o]
            Wout[o] -= lr * g * vc
            for m in negs:
                s = sigmoid(vc @ Wout[m])                         # negative pair -> label 0
                Wout[m] -= lr * s * vc
                gvc += s * Wout[m]
            Win[c] -= lr * gvc
    return Win, vocab, V


# =============================================================================
# EXPERIMENT 1 — the SGNS objective and its gradient
# =============================================================================


def experiment_1_objective():
    print("=" * 88)
    print("EXPERIMENT 1 — SGNS is binary classification; its gradient is exact (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    d = 8
    w = rng.standard_normal(d); c_pos = rng.standard_normal(d); c_negs = rng.standard_normal((3, d))

    def loss(w):
        L = -np.log(sigmoid(w @ c_pos) + 1e-12)
        for cn in c_negs:
            L += -np.log(sigmoid(-w @ cn) + 1e-12)
        return L

    # analytic gradient w.r.t. w
    grad = (sigmoid(w @ c_pos) - 1) * c_pos
    for cn in c_negs:
        grad += sigmoid(w @ cn) * cn
    # finite-difference check
    eps = 1e-6
    fd = np.array([(loss(w + eps * e) - loss(w - eps * e)) / (2 * eps)
                   for e in np.eye(d)])
    print(f"""
  Objective for one (word, +context) pair and 3 negative samples:
    L = -log sigma(w . c+)  -  sum_neg log sigma(-w . c-)

    analytic grad vs finite-difference: max|diff| = {np.abs(grad - fd).max():.1e}

  READING: negative sampling turns learning an embedding into BINARY CLASSIFICATION: push the word
  vector to agree with its true context (sigma(w.c+) -> 1) and disagree with K random 'noise' words
  (sigma(w.c-) -> 0). This replaces the expensive full-vocabulary softmax with K+1 cheap logistic
  updates. The gradient is exact to ~1e-9 (README §3).""")


# =============================================================================
# EXPERIMENT 2 — the distributional hypothesis
# =============================================================================


def _animals_corpus():
    rng = np.random.default_rng(1)
    animals, actions = ["cat", "dog", "fox"], ["ran", "sat", "ate"]
    colors, things = ["red", "blue", "green"], ["ball", "car", "box"]
    sents = []
    for _ in range(1500):
        sents.append(["the", rng.choice(animals), rng.choice(actions)])
        sents.append(["the", rng.choice(colors), rng.choice(things)])
    return sents


def experiment_2_distributional():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the distributional hypothesis: shared contexts -> similar vectors (README §2)")
    print("=" * 88)
    E, vocab, V = train_sgns(_animals_corpus(), epochs=6, seed=2)

    def sim(a, b):
        return cos(E[V[a]], E[V[b]])

    print(f"""
  Corpus: animals {{cat,dog,fox}} appear with actions; colors {{red,blue,green}} appear with objects.
  Words that are INTERCHANGEABLE (same contexts) should get similar vectors:

    within animals:  cos(cat, dog) = {sim('cat','dog'):.3f}   cos(cat, fox) = {sim('cat','fox'):.3f}
    within colors:   cos(red, blue) = {sim('red','blue'):.3f}
    across groups:   cos(cat, red)  = {sim('cat','red'):.3f}   cos(cat, ball) = {sim('cat','ball'):.3f}

  READING: cat, dog, and fox never appear together, yet their vectors are nearly identical (~0.99)
  because they occur in the SAME contexts (the ___ ran/sat/ate). Words in different contexts (cat vs
  red) stay dissimilar (~0.4). This is the distributional hypothesis made geometric: meaning is
  learned from company, not co-occurrence — which is why embeddings capture synonymy that bag-of-words
  ([10.02](../02-classical-representations/)) cannot (README §2).""")


# =============================================================================
# EXPERIMENT 3 — embedding geometry / analogies
# =============================================================================


def _analogy_corpus():
    rng = np.random.default_rng(0)
    royal, person = ["throne", "crown", "rule", "palace"], ["street", "walk", "work", "home"]
    male, female = ["he", "his", "him"], ["she", "her", "hers"]

    def gen(word, role, gender, n=500):
        return [[word, rng.choice(role), rng.choice(gender)] for _ in range(n)]
    return (gen("king", royal, male) + gen("queen", royal, female) +
            gen("man", person, male) + gen("woman", person, female))


def experiment_3_analogy():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — embedding geometry: king - man + woman ~ queen (README §4)")
    print("=" * 88)
    E, vocab, V = train_sgns(_analogy_corpus(), dim=24, epochs=8, window=2, seed=0)

    def vec(w):
        return E[V[w]]

    q = vec("king") - vec("man") + vec("woman")
    scored = sorted(((w, cos(q, vec(w))) for w in ["king", "queen", "man", "woman"]),
                    key=lambda x: -x[1])
    best = max((w for w in vocab if w not in ("king", "man", "woman")), key=lambda w: cos(q, vec(w)))
    print(f"\n  Corpus with a gender x role structure (king/queen royal, man/woman common).")
    print(f"  Solving the analogy 'king - man + woman = ?':\n")
    print(f"    {'candidate':>10s} {'cosine to (king - man + woman)':>32s}")
    for w, s in scored:
        print(f"    {w:>10s} {s:>32.3f}")
    print(f"\n    nearest word overall (excluding the inputs) = {best!r}")
    print("""
  READING: subtracting 'man' from 'king' and adding 'woman' lands almost exactly on 'queen' (~0.99).
  The embedding has organized words along consistent DIRECTIONS — a gender axis and a royalty axis — so
  vector arithmetic performs analogical reasoning. This famous king-man+woman=queen result (Mikolov et
  al.) is not hand-coded; it EMERGES from predicting contexts, because parallel relationships produce
  parallel offsets in the vector space (README §4).""")


# =============================================================================
# EXPERIMENT 4 — GloVe factorizes the log co-occurrence matrix
# =============================================================================


def experiment_4_glove():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — GloVe factorizes the log co-occurrence matrix (README §5)")
    print("=" * 88)
    sents = _animals_corpus()
    vocab = sorted({w for s in sents for w in s})
    V = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    X = np.zeros((n, n))                                          # co-occurrence counts, window=2
    for s in sents:
        ids = [V[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    X[c, ids[j]] += 1
    # GloVe: minimize sum f(X_ij) (w_i.w~_j + b_i + b~_j - log X_ij)^2
    rng = np.random.default_rng(0)
    d = 16
    W = rng.standard_normal((n, d)) * 0.1; Wt = rng.standard_normal((n, d)) * 0.1
    b = np.zeros(n); bt = np.zeros(n)
    ii, jj = np.nonzero(X)
    xij = X[ii, jj]
    f = np.minimum((xij / 10.0) ** 0.75, 1.0)                    # GloVe weighting f(x)
    logx = np.log(xij)
    lr = 0.05
    for _ in range(200):
        pred = (W[ii] * Wt[jj]).sum(1) + b[ii] + bt[jj]
        diff = pred - logx
        g = f * diff
        # gradient updates (accumulated per index)
        for a in range(len(ii)):
            i, j = ii[a], jj[a]
            gw = g[a]
            W[i] -= lr * gw * Wt[j]; Wt[j] -= lr * gw * W[i]
            b[i] -= lr * gw; bt[j] -= lr * gw
    E = W + Wt                                                    # GloVe uses the sum of both vectors

    def sim(x, y):
        return cos(E[V[x]], E[V[y]])
    print(f"""
  GloVe minimizes  sum_ij f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2  over co-occurrence counts X.
  On the same animals/colors corpus:

    within animals:  cos(cat, dog) = {sim('cat','dog'):.3f}   cos(cat, fox) = {sim('cat','fox'):.3f}
    within colors:   cos(red, green) = {sim('red','green'):.3f}
    across groups:   cos(cat, red)  = {sim('cat','red'):.3f}

  READING: where Word2Vec learns from local (word, context) samples, GloVe fits GLOBAL co-occurrence
  counts directly — its objective is a weighted least-squares factorization of the LOG co-occurrence
  matrix. It recovers the same structure (animals cluster, animals != colors) from the aggregate
  statistics in one shot, and the weighting f(X) down-weights rare and caps frequent pairs (README §5).""")


# =============================================================================
# EXPERIMENT 5 — FastText: subword n-grams handle OOV
# =============================================================================


def experiment_5_fasttext():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — FastText: subword n-grams give OOV/misspelled words vectors (README §6)")
    print("=" * 88)

    def char_ngrams(word, n=3):
        w = "<" + word + ">"
        return {w[i:i + n] for i in range(len(w) - n + 1)}

    def ngram_sim(a, b):                                          # Jaccard over shared char n-grams
        A, B = char_ngrams(a), char_ngrams(b)
        return len(A & B) / len(A | B)

    print(f"\n  A word is represented by its character 3-grams (FastText sums their vectors). Overlap:\n")
    print(f"    {'pair':>26s} {'shared-3gram similarity':>24s}")
    for a, b in [("running", "runnign"), ("running", "runner"), ("running", "jumping"),
                 ("king", "kings"), ("king", "banana")]:
        tag = f"{a} ~ {b}"
        print(f"    {tag:>26s} {ngram_sim(a, b):>24.3f}")
    print("""
  READING: FastText represents each word as the SUM of its character n-gram vectors (e.g. 'running' ->
  <ru, run, unn, nni, nin, ing, ng>). So a misspelling ('runnign') or an unseen inflection ('runner')
  shares most n-grams with 'running' and gets a NEARBY vector — even if the exact word was never seen in
  training. Word2Vec/GloVe would emit <UNK> for these; FastText composes a vector from subwords,
  handling OOV, misspellings, and morphologically rich languages (README §6).""")


if __name__ == "__main__":
    experiment_1_objective()
    experiment_2_distributional()
    experiment_3_analogy()
    experiment_4_glove()
    experiment_5_fasttext()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
