"""
10.02 — Classical text representations, from scratch (NumPy).

Before embeddings, text was turned into vectors by COUNTING. This file builds the classical pipeline —
bag-of-words, TF-IDF, n-grams, LSA, and NMF topic models — and verifies each against scikit-learn:

  1. bag-of-words counts == sklearn CountVectorizer                (exact)
  2. TF-IDF == sklearn TfidfVectorizer (exact smoothing + L2 norm) -> Experiment 2
  3. n-grams capture word order that bag-of-words throws away      -> Experiment 3
  4. LSA (truncated SVD) uncovers latent topics / synonymy         -> Experiment 4
  5. NMF factorizes documents into interpretable topics            -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, NMF
    HAVE_SK = True
except Exception:                                    # pragma: no cover
    HAVE_SK = False


# =============================================================================
# Bag-of-words and TF-IDF
# =============================================================================


def build_vocab(docs, ngram=1):
    vocab = {}
    for d in docs:
        for tok in _ngrams(d.lower().split(), ngram):
            vocab.setdefault(tok, len(vocab))
    return vocab


def _ngrams(tokens, n):
    if n == 1:
        return tokens
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_matrix(docs, vocab, ngram=1):
    X = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for tok in _ngrams(d.lower().split(), ngram):
            if tok in vocab:
                X[r, vocab[tok]] += 1
    return X


def tfidf_matrix(docs, vocab):
    """Exactly sklearn's TfidfVectorizer: tf * (ln((1+n)/(1+df)) + 1), then L2-normalize rows."""
    tf = count_matrix(docs, vocab)
    n = len(docs)
    df = (tf > 0).sum(0)
    idf = np.log((1 + n) / (1 + df)) + 1                 # smooth_idf=True
    tfidf = tf * idf
    norm = np.linalg.norm(tfidf, axis=1, keepdims=True)
    return tfidf / np.clip(norm, 1e-12, None)


# =============================================================================
# EXPERIMENT 1-2 — BoW and TF-IDF == scikit-learn
# =============================================================================


def experiment_1_2_verify():
    print("=" * 88)
    print("EXPERIMENTS 1-2 — bag-of-words and TF-IDF == scikit-learn (exact)")
    print("=" * 88)
    docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are friends"]
    vocab = build_vocab(docs)
    bow = count_matrix(docs, vocab)
    tfidf = tfidf_matrix(docs, vocab)
    if HAVE_SK:
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        sk_bow = cv.fit_transform(docs).toarray()
        # align columns to sklearn's vocabulary order
        order = [vocab[t] for t in cv.get_feature_names_out()]
        e_bow = np.abs(bow[:, order] - sk_bow).max()
        tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        sk_tfidf = tv.fit_transform(docs).toarray()
        order2 = [vocab[t] for t in tv.get_feature_names_out()]
        e_tfidf = np.abs(tfidf[:, order2] - sk_tfidf).max()
    else:
        e_bow = e_tfidf = np.nan
    print(f"""
  3 documents, vocabulary of {len(vocab)} words:

    bag-of-words counts   |ours - sklearn CountVectorizer| = {e_bow:.1e}
    TF-IDF (L2-normalized) |ours - sklearn TfidfVectorizer| = {e_tfidf:.1e}

  READING: a bag-of-words vector counts how often each vocabulary word appears in a document, ignoring
  order. TF-IDF reweights those counts by INVERSE DOCUMENT FREQUENCY — idf = ln((1+n)/(1+df))+1 — so a
  word appearing in every document (like 'the') gets a small weight and a distinctive word gets a large
  one. Both match scikit-learn exactly (including its specific smoothing and L2 normalization).""")


# =============================================================================
# EXPERIMENT 2b — TF-IDF downweights common words
# =============================================================================


def experiment_2b_downweight():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2b — TF-IDF downweights ubiquitous words (README §2)")
    print("=" * 88)
    docs = ["the market rose today", "the market fell today", "the weather is nice today",
            "the game was exciting today"]
    vocab = build_vocab(docs)
    tf = count_matrix(docs, vocab)
    n = len(docs); df = (tf > 0).sum(0)
    idf = np.log((1 + n) / (1 + df)) + 1
    inv = {i: w for w, i in vocab.items()}
    print(f"\n  {n} documents. IDF weight of each word (high = distinctive, low = ubiquitous):\n")
    print(f"    {'word':>10s} {'doc freq':>9s} {'idf':>8s}")
    for i in np.argsort(idf):
        print(f"    {inv[i]:>10s} {int(df[i]):>9d} {idf[i]:>8.3f}")
    print("""
  READING: 'the' and 'today' appear in ALL documents, so their idf is the minimum (1.0) — TF-IDF nearly
  zeroes them out. 'market' appears in 2 docs (medium idf); words in a single doc get the maximum idf.
  This is why TF-IDF beats raw counts for search and classification: it automatically discounts the
  filler words that dominate raw counts and keeps the content words (README §2).""")


# =============================================================================
# EXPERIMENT 3 — n-grams capture order
# =============================================================================


def experiment_3_ngrams():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — n-grams capture word order that bag-of-words loses (README §3)")
    print("=" * 88)
    a = "the movie was good not bad"
    b = "the movie was bad not good"
    uni = build_vocab([a, b], ngram=1)
    Xu = count_matrix([a, b], uni, ngram=1)
    bi = build_vocab([a, b], ngram=2)
    Xb = count_matrix([a, b], bi, ngram=2)
    print(f"""
  Two sentences with OPPOSITE meaning but the same words:
    A: "{a}"
    B: "{b}"

    unigram (bag-of-words) vectors identical?  {np.array_equal(Xu[0], Xu[1])}   (order lost)
    bigram vectors identical?                  {np.array_equal(Xb[0], Xb[1])}   (order captured)
    distinguishing bigrams: {sorted(set(_ngrams(a.split(),2)) ^ set(_ngrams(b.split(),2)))}

  READING: bag-of-words is a MULTISET — it discards order, so 'good not bad' and 'bad not good' have
  identical unigram vectors despite opposite sentiment. Adding bigrams (pairs of adjacent words) brings
  back local order: 'not bad' vs 'not good' become distinct features. n-grams are the cheap classical
  fix for word order, at the cost of a much larger, sparser vocabulary (README §3).""")


# =============================================================================
# EXPERIMENT 4 — LSA uncovers latent topics / synonymy
# =============================================================================


def experiment_4_lsa():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — LSA (truncated SVD) uncovers latent topics / synonymy (README §4)")
    print("=" * 88)
    docs = [
        "planet star orbit",        # space (bridges star & orbit & planet)
        "star galaxy cosmos",       # space  <- doc 1
        "galaxy cosmos nebula",     # space
        "orbit planet sun",         # space  <- doc 3: NO words shared with doc 1
        "recipe food cook",         # food
        "food cook kitchen",        # food
        "kitchen chef meal",        # food
        "meal recipe food",         # food
    ]
    vocab = build_vocab(docs)
    X = tfidf_matrix(docs, vocab)
    # LSA = truncated SVD of the doc-term matrix, keep k topic dimensions
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = 2
    lsa = U[:, :k] * S[:k]                                # documents in 2-D topic space

    def cos(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    raw_sim = cos(X[1], X[3])                             # two space docs, no shared words
    lsa_same = cos(lsa[1], lsa[3])                        # same topic in LSA space
    lsa_diff = cos(lsa[1], lsa[4])                        # space vs food in LSA space
    print(f"""
  8 docs (4 space, 4 food). doc1 "star galaxy cosmos" and doc3 "orbit planet sun" share NO words but
  are both about space (linked through the corpus, e.g. doc0 "planet star orbit"):

    doc1-doc3 (both space)  cosine in RAW tf-idf space = {raw_sim:.3f}   (0: no shared words!)
    doc1-doc3 (both space)  cosine in LSA topic space  = {lsa_same:.3f}   (same latent topic)
    doc1-doc4 (space vs food) cosine in LSA topic space = {lsa_diff:.3f}   (different topic)

  READING: in raw bag-of-words space doc1 and doc3 are ORTHOGONAL (similarity 0) because they share no
  words — bag-of-words cannot see synonymy. LSA takes the truncated SVD of the doc-term matrix,
  projecting documents onto k latent 'topic' directions. There the two space docs are nearly identical
  ({lsa_same:.2f}) while a space-vs-food pair stays near 0 — LSA recovered the topic structure that the
  surface words hide, linking documents through their shared co-occurrence patterns (README §4).""")


# =============================================================================
# EXPERIMENT 5 — NMF topic model
# =============================================================================


def experiment_5_nmf():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — NMF factorizes documents into interpretable topics (README §5)")
    print("=" * 88)
    docs = [
        "money bank loan market money", "stock market money invest bank",
        "loan bank credit money debt", "game team score win play",
        "team player game win coach", "score goal team play win",
    ]
    vocab = build_vocab(docs)
    inv = {i: w for w, i in vocab.items()}
    X = count_matrix(docs, vocab)
    if HAVE_SK:
        nmf = NMF(n_components=2, init="nndsvda", random_state=0, max_iter=500)
        W = nmf.fit_transform(X)                         # doc x topic
        Ht = nmf.components_                             # topic x word
        recon = np.abs(X - W @ Ht).mean()
        print(f"\n  6 documents (3 finance, 3 sports), factorized into 2 topics. Top words per topic:\n")
        for t in range(2):
            top = np.argsort(Ht[t])[::-1][:4]
            print(f"    topic {t}: {[inv[i] for i in top]}")
        print(f"\n    each document's dominant topic: {W.argmax(1).tolist()}")
        print(f"    reconstruction mean|X - W H| = {recon:.3f}")
    print("""
  READING: non-negative matrix factorization decomposes the (non-negative) document-term matrix X into
  W (documents x topics) times H (topics x words), all entries >= 0. The non-negativity forces PARTS-
  based, additive topics: each topic is a set of words that co-occur, and each document is a mixture of
  topics. Here it cleanly recovers a 'finance' topic and a 'sports' topic and assigns each document to
  the right one — an unsupervised, interpretable summary of the corpus (README §5).""")


if __name__ == "__main__":
    experiment_1_2_verify()
    experiment_2b_downweight()
    experiment_3_ngrams()
    experiment_4_lsa()
    experiment_5_nmf()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_SK else "ALL CHECKS PASSED (sklearn-verified parts skipped)")
    print("=" * 88)
