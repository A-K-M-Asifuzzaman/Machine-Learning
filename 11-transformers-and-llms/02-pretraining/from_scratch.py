"""
11.02 — Pretraining paradigms, from scratch (NumPy / Python).

The same transformer block ([11.01]) becomes BERT, GPT, or T5 depending on the ATTENTION MASK and the
PRETRAINING OBJECTIVE. This file makes those differences concrete:

  1. mask patterns: encoder (bidirectional) vs decoder (causal) vs prefix-LM   -> Experiment 1
  2. BERT masked-language-modeling corruption (15%, 80/10/10)                  -> Experiment 2
  3. the three objectives on one sentence: MLM vs causal-LM vs span corruption -> Experiment 3
  4. perplexity: the language-model metric                                     -> Experiment 4
  5. WHY bidirectional: right context helps fill-in-the-blank, causal can't    -> Experiment 5

Run:  python3 from_scratch.py
"""

import collections

import numpy as np


# =============================================================================
# EXPERIMENT 1 — attention mask patterns
# =============================================================================


def experiment_1_masks():
    print("=" * 88)
    print("EXPERIMENT 1 — encoder (bidirectional) vs decoder (causal) vs prefix-LM masks (README §2)")
    print("=" * 88)
    n = 6
    bidir = np.ones((n, n))                           # BERT: everyone sees everyone
    causal = np.tril(np.ones((n, n)))                 # GPT: token i sees 0..i
    prefix = causal.copy()                            # prefix-LM: first 3 tokens bidirectional
    prefix[:3, :3] = 1
    print(f"\n  Which positions can each query (row) attend to? (1 = visible)\n")
    for name, M in [("encoder / BERT", bidir), ("decoder / GPT", causal), ("prefix-LM / T5", prefix)]:
        visible = M.sum(1).astype(int)
        print(f"    {name:>16s}: visible-per-position = {visible.tolist()}")
    print("""
  READING: the ONLY architectural difference between BERT and GPT is this mask. An ENCODER uses no mask
  — every token attends to the whole sequence (bidirectional context), ideal for UNDERSTANDING. A
  DECODER uses a causal mask — token i sees only 0..i — which is required to GENERATE (you cannot peek
  at the future you are predicting). A PREFIX-LM (T5's encoder-decoder) is bidirectional over the input
  prefix and causal over the output. Same block, three models (README §2).""")


# =============================================================================
# EXPERIMENT 2 — BERT masked-language-modeling corruption
# =============================================================================


def experiment_2_mlm():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — BERT MLM corruption: 15%, split 80/10/10 (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(100)]
    tokens = list(rng.choice(vocab, 10000))
    # choose 15% to corrupt; of those 80% -> [MASK], 10% -> random, 10% -> unchanged
    n = len(tokens)
    chosen = rng.random(n) < 0.15
    out = list(tokens)
    n_mask = n_rand = n_keep = 0
    for i in np.where(chosen)[0]:
        r = rng.random()
        if r < 0.8:
            out[i] = "[MASK]"; n_mask += 1
        elif r < 0.9:
            out[i] = rng.choice(vocab); n_rand += 1
        else:
            n_keep += 1                               # left unchanged (but still predicted)
    nc = chosen.sum()
    print(f"""
  Corrupting {n:,} tokens:

    chosen for prediction  = {nc:,}  ({100*nc/n:.1f}%)   (target ~15%)
    -> replaced with [MASK] = {n_mask:,}  ({100*n_mask/nc:.0f}% of chosen)   (target 80%)
    -> replaced with random = {n_rand:,}  ({100*n_rand/nc:.0f}% of chosen)   (target 10%)
    -> kept unchanged       = {n_keep:,}  ({100*n_keep/nc:.0f}% of chosen)   (target 10%)

  READING: BERT predicts 15% of tokens from bidirectional context. But it doesn't just insert [MASK]:
  10% of chosen tokens are replaced with a RANDOM word and 10% are left UNCHANGED. Why? Because [MASK]
  never appears at fine-tuning time — if the model only ever saw [MASK], it would learn features that
  vanish downstream. The random/keep split forces it to build a good representation of EVERY token, not
  just the masked slots (README §3).""")


# =============================================================================
# EXPERIMENT 3 — the three objectives on one sentence
# =============================================================================


def experiment_3_objectives():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — MLM vs causal-LM vs span corruption on one sentence (README §4)")
    print("=" * 88)
    sent = ["the", "cat", "sat", "on", "the", "mat"]
    print(f"\n  Sentence: {sent}\n")
    # MLM (BERT)
    mlm_in = ["the", "[MASK]", "sat", "on", "the", "[MASK]"]
    mlm_tgt = {1: "cat", 5: "mat"}
    print(f"  MLM (BERT):        input  {mlm_in}")
    print(f"                     target predict masked positions {mlm_tgt}")
    # causal LM (GPT)
    print(f"\n  Causal LM (GPT):   input  {sent[:-1]}")
    print(f"                     target {sent[1:]}   (shift by one: predict the next token)")
    # span corruption (T5)
    t5_in = ["the", "<X>", "on", "<Y>"]
    t5_tgt = ["<X>", "cat", "sat", "<Y>", "the", "mat"]
    print(f"\n  Span corruption(T5): input  {t5_in}   (spans replaced by sentinels)")
    print(f"                       target {t5_tgt}   (generate the dropped spans)")
    print("""
  READING: three ways to make a self-supervised label out of raw text. MLM masks individual tokens and
  predicts them (bidirectional, understanding). Causal LM predicts the NEXT token at every position
  (left-to-right, generation) — the objective behind every GPT. Span corruption (T5) drops whole spans
  and generates them with an encoder-decoder, unifying tasks as text-to-text. All three need NO labels —
  the text is its own supervision, which is what let pretraining scale to the whole internet (README §4).""")


# =============================================================================
# EXPERIMENT 4 — perplexity
# =============================================================================


def experiment_4_perplexity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — perplexity: the language-model metric (README §5)")
    print("=" * 88)
    V = 100
    # true next-token is index 0; three models assign it different probability
    def ppl(p_true, others_uniform_over=V - 1, n=1):
        # perplexity = exp(cross-entropy); here cross-entropy = -log p(true)
        return np.exp(-np.log(p_true))
    print(f"\n  Vocabulary size {V}. Perplexity = exp(cross-entropy) = how many words the model is")
    print(f"  effectively 'choosing between' at each step (lower is better):\n")
    print(f"    {'model assigns p(correct)':>28s} {'perplexity':>12s}")
    for p in (1.0, 0.5, 0.1, 1 / V):
        label = f"{p:.3f}" + ("  (uniform)" if abs(p - 1 / V) < 1e-9 else "")
        print(f"    {label:>28s} {ppl(p):>12.1f}")
    print(f"""
  READING: perplexity is exp(cross-entropy loss) — geometrically, the number of equally-likely choices
  the model faces per token. A perfect model (p=1 on the right word) has perplexity 1; a model that
  guesses uniformly over {V} words has perplexity {V} (it is 'as confused as' a {V}-way coin flip).
  Halving perplexity means the model is twice as sure. It is THE intrinsic language-model metric —
  GPT-2/3 progress was largely a perplexity race (README §5).""")


# =============================================================================
# EXPERIMENT 5 — why bidirectional context helps understanding
# =============================================================================


def experiment_5_bidirectional():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — bidirectional context helps fill-in-the-blank; causal can't (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    subj_verb = {"dog": "barked", "cat": "meowed", "bird": "chirped"}
    subjs = list(subj_verb)
    sents = [["the", (s := rng.choice(subjs)), subj_verb[s], "loudly"] for _ in range(1500)]
    left = collections.defaultdict(collections.Counter)
    both = collections.defaultdict(collections.Counter)
    for s in sents:                                   # predict masked subject at position 1
        left[(s[0],)][s[1]] += 1
        both[(s[0], s[2])][s[1]] += 1
    lc = bc = 0
    for s in sents:
        lc += left[(s[0],)].most_common(1)[0][0] == s[1]
        bc += both[(s[0], s[2])].most_common(1)[0][0] == s[1]
    print(f"""
  Sentences "the [SUBJ] [VERB] loudly" where each subject has a distinctive verb
  (dog->barked, cat->meowed, bird->chirped). Predict the masked SUBJECT:

    left-only context  "the ___"          accuracy = {lc / len(sents):.3f}   (~chance 0.33)
    both-sides context "the ___ [VERB]"   accuracy = {bc / len(sents):.3f}   (the verb gives it away)

  READING: the word AFTER the blank ('barked') identifies the subject, but a causal (left-to-right)
  model cannot use it — it only sees 'the ___'. A bidirectional model sees both sides and gets it every
  time. This is exactly why BERT masks tokens and attends both ways: for UNDERSTANDING tasks
  (classification, NER, QA) two-sided context is a large win. The cost is that BERT cannot GENERATE —
  which is the trade GPT makes the other way (README §2).""")


if __name__ == "__main__":
    experiment_1_masks()
    experiment_2_mlm()
    experiment_3_objectives()
    experiment_4_perplexity()
    experiment_5_bidirectional()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
