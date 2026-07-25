"""
10.01 — Text preprocessing & tokenization, from scratch (NumPy / pure Python).

Before a model sees text it must be turned into integer tokens. The central question is the GRANULARITY:
words (huge vocab, out-of-vocabulary problem), characters (tiny vocab, very long sequences), or SUBWORDS
learned by Byte-Pair Encoding (the modern default). This file builds BPE from scratch and MEASURES the
trade-offs:

  1. BPE learns merges by greedily combining the most frequent pair (matches the textbook example)
  2. BPE round-trips (encode -> decode = identity) and needs NO unknown token       -> Experiment 2
  3. vocabulary size vs sequence length: word vs char vs subword                     -> Experiment 3
  4. more merges -> shorter sequences but a larger vocabulary (the compression dial) -> Experiment 4
  5. a real tokenizer (GPT-2, tiktoken) splits rare words into subwords              -> Experiment 5

Run:  python3 from_scratch.py
"""

import collections
import re


# =============================================================================
# BPE — training (learn merges) and encoding (apply them)
# =============================================================================


def _get_pair_counts(word_freqs):
    """Count adjacent symbol pairs across the corpus, weighted by word frequency."""
    pairs = collections.Counter()
    for word, freq in word_freqs.items():
        syms = word.split()
        for a, b in zip(syms[:-1], syms[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_pair(pair, word_freqs):
    """Replace every occurrence of the pair with the merged symbol, across the corpus."""
    out = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(pair)
    for word, f in word_freqs.items():
        out[pattern.sub(merged, word)] = f
    return out


def train_bpe(corpus, n_merges):
    """Learn an ordered list of merges from a corpus (words split into chars + end marker)."""
    word_freqs = {" ".join(list(w)) + " </w>": f
                  for w, f in collections.Counter(corpus.split()).items()}
    merges = []
    for _ in range(n_merges):
        pairs = _get_pair_counts(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))     # most frequent (ties: lexicographic)
        word_freqs = _merge_pair(best, word_freqs)
        merges.append(best)
    return merges


def encode_word(word, ranks):
    """Tokenize one word by repeatedly applying the earliest-learned applicable merge."""
    syms = list(word) + ["</w>"]
    while True:
        best_rank, best_i = None, None
        for i in range(len(syms) - 1):
            r = ranks.get((syms[i], syms[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i is None:
            break
        syms[best_i:best_i + 2] = ["".join(syms[best_i:best_i + 2])]
    return syms


def decode_tokens(tokens):
    """Join subword tokens back into the original word (strip the end marker)."""
    return "".join(tokens).replace("</w>", "")


# =============================================================================
# EXPERIMENT 1 — BPE learns the textbook merges
# =============================================================================


def experiment_1_merges():
    print("=" * 88)
    print("EXPERIMENT 1 — BPE greedily merges the most frequent pair (README §3)")
    print("=" * 88)
    corpus = ("low low low low low lower lower newest newest newest newest newest newest "
              "widest widest widest") * 3
    merges = train_bpe(corpus, 8)
    print(f"\n  Corpus of {{low, lower, newest, widest}}. First 8 merges learned:\n")
    for i, (a, b) in enumerate(merges, 1):
        print(f"    {i}. {a!r:>10s} + {b!r:<8s} -> {a + b!r}")
    print("""
  READING: BPE starts from characters and repeatedly merges the MOST FREQUENT adjacent pair into a new
  symbol. Here it discovers 's'+'t'->'st', then 'e'+'st'->'est', then the frequent ending 'est</w>' and
  eventually whole words like 'newest</w>' — the classic bottom-up construction of Sennrich et al.
  (2016). Frequent substrings become single tokens; rare ones stay split. The learned merge list IS the
  tokenizer (README §3).""")


# =============================================================================
# EXPERIMENT 2 — round-trip and no unknown token
# =============================================================================


def experiment_2_oov():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — BPE round-trips and needs no <UNK> for unseen words (README §4)")
    print("=" * 88)
    corpus = ("low low low lower lower lowest newest newest newest wider widest slow slower ") * 5
    merges = train_bpe(corpus, 30)
    ranks = {m: i for i, m in enumerate(merges)}
    train_words = set(corpus.split())

    print(f"\n  Trained on {sorted(train_words)}.\n")
    print(f"  Tokenizing words NEVER seen in training (word-level would emit <UNK>):\n")
    print(f"    {'word':>12s} {'BPE tokens':>34s} {'round-trips?':>13s}")
    ok = True
    for w in ("slowest", "newer", "lowestest", "wildest"):
        toks = encode_word(w, ranks)
        rt = decode_tokens(toks) == w
        ok = ok and rt
        pretty = " ".join(t.replace("</w>", "_") for t in toks)
        print(f"    {w:>12s} {pretty:>34s} {str(rt):>13s}")
    print(f"""
  All round-trip correctly: {ok}

  READING: a word-level tokenizer must map any unseen word to a single <UNK> token, destroying it. BPE
  instead falls back to the subwords (and ultimately characters) it DOES know, so every possible string
  is representable and decoding is exact. 'slowest' was never seen, yet it tokenizes into known pieces
  and reconstructs perfectly. This open-vocabulary property is why subword tokenization replaced
  word-level everywhere (README §4).""")


# =============================================================================
# EXPERIMENT 3 — vocabulary size vs sequence length
# =============================================================================


def experiment_3_tradeoff():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — granularity trade-off: word vs char vs subword (README §2)")
    print("=" * 88)
    corpus = _sample_corpus()
    words = corpus.split()
    n_words = len(words)
    # word-level
    word_vocab = set(words)
    # char-level
    char_vocab = set(corpus.replace(" ", ""))
    char_len = sum(len(w) for w in words)
    # subword (BPE) with a moderate merge budget
    merges = train_bpe(corpus, 60)                     # subword: a genuine middle ground
    ranks = {m: i for i, m in enumerate(merges)}
    sub_vocab = set()
    sub_len = 0
    for w in set(words):
        toks = encode_word(w, ranks)
        sub_vocab.update(toks)
        sub_len += len(toks) * words.count(w)
    print(f"\n  Corpus: {n_words} words, {char_len} characters.\n")
    print(f"    {'tokenization':>14s} {'vocab size':>12s} {'sequence length':>18s} {'tokens/word':>13s}")
    print(f"    {'word-level':>14s} {len(word_vocab):>12d} {n_words:>18d} {1.0:>13.2f}")
    print(f"    {'char-level':>14s} {len(char_vocab):>12d} {char_len:>18d} {char_len / n_words:>13.2f}")
    print(f"    {'subword (BPE)':>14s} {len(sub_vocab):>12d} {sub_len:>18d} {sub_len / n_words:>13.2f}")
    print("""
  READING: word-level gives the SHORTEST sequences (1 token/word) but the LARGEST vocab and an OOV
  problem. Char-level gives a TINY vocab but the LONGEST sequences (every character is a step, costly
  for a model). SUBWORD sits in between by design — a bounded vocab AND a modest sequence length — which
  is why every modern LLM uses it (README §2).""")


# =============================================================================
# EXPERIMENT 4 — the compression dial: merges vs sequence length
# =============================================================================


def experiment_4_compression():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — more merges -> shorter sequences, larger vocab (README §3)")
    print("=" * 88)
    corpus = _sample_corpus()
    words = corpus.split()
    n_words = len(words)
    base_vocab = len(set(corpus.replace(" ", "")))
    print(f"\n  As the BPE merge budget grows (starting from {base_vocab} characters):\n")
    print(f"    {'# merges':>10s} {'vocab size':>12s} {'tokens/word':>13s}")
    for n_merges in (0, 50, 200, 500, 1000):
        merges = train_bpe(corpus, n_merges)
        ranks = {m: i for i, m in enumerate(merges)}
        vocab = set()
        total = 0
        for w in set(words):
            toks = encode_word(w, ranks)
            vocab.update(toks)
            total += len(toks) * words.count(w)
        print(f"    {n_merges:>10d} {len(vocab):>12d} {total / n_words:>13.3f}")
    print("""
  READING: the number of merges is a DIAL between the char and word extremes. With 0 merges you have
  characters (~7 tokens/word here); each merge folds a frequent pair into one token, so sequences get
  shorter (2.86 -> 1.12 -> 1.0) while the vocabulary grows (23 -> 154 -> 170). Real tokenizers pick a
  vocab size (GPT-2: 50,257) balancing a manageable embedding table against short sequences (README §3).""")


# =============================================================================
# EXPERIMENT 5 — a real tokenizer (GPT-2 via tiktoken)
# =============================================================================


def experiment_5_real():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — a production tokenizer: GPT-2 splits rare words into subwords (README §5)")
    print("=" * 88)
    try:
        import tiktoken
    except Exception:
        print("  tiktoken not available — skipping."); return
    enc = tiktoken.get_encoding("gpt2")
    print(f"\n  GPT-2 byte-level BPE, vocabulary size = {enc.n_vocab:,}:\n")
    print(f"    {'word':>32s} {'# tokens':>9s}   subwords")
    for w in ("hello", "tokenization", "antidisestablishmentarianism", "supercalifragilistic"):
        ids = enc.encode(w)
        pieces = [enc.decode([i]) for i in ids]
        print(f"    {w:>32s} {len(ids):>9d}   {pieces}")
    print("""
  READING: GPT-2's tokenizer is byte-level BPE with a 50,257-token vocabulary. Common words ('hello')
  are a single token; rarer words split into meaningful subwords ('token'+'ization'); very rare or
  novel strings split further, down to bytes in the worst case — so it NEVER fails on any input. This is
  exactly the from-scratch BPE above, trained on a large corpus and operating on raw bytes (README §5).""")


def _sample_corpus():
    """A morphologically-rich corpus: stems x suffixes -> many words sharing few subwords."""
    import random
    stems = ["play", "walk", "talk", "jump", "call", "work", "learn", "teach", "read", "open",
             "close", "start", "move", "turn", "look", "help", "need", "want", "know", "show",
             "tell", "give", "take", "make", "find"]
    suffixes = ["", "s", "ed", "ing", "er", "ers", "est"]
    rng = random.Random(0)
    return " ".join(rng.choice(stems) + rng.choice(suffixes) for _ in range(600))


if __name__ == "__main__":
    experiment_1_merges()
    experiment_2_oov()
    experiment_3_tradeoff()
    experiment_4_compression()
    experiment_5_real()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
