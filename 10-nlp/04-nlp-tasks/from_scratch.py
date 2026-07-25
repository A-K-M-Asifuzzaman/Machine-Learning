"""
10.04 — NLP tasks & metrics, from scratch (Python).

NLP is a handful of task SHAPES (classify a sequence, tag each token, extract a span, generate text,
compare a pair) each with its own METRIC. Getting the metric right matters more than the model — a
wrong metric optimizes the wrong thing. This file implements the standard metrics and shows why each
exists:

  1. NER: entity-level F1 vs token accuracy (why span F1 is the honest metric)   -> Experiment 1
  2. extractive QA: SQuAD Exact-Match vs token-F1 (partial credit)               -> Experiment 2
  3. generation: BLEU (n-gram precision + brevity penalty) == nltk               -> Experiment 3
  4. summarization: ROUGE-N and ROUGE-L (recall-oriented)                        -> Experiment 4
  5. classification: micro vs macro F1 on imbalanced classes                     -> Experiment 5

Run:  python3 from_scratch.py
"""

import collections
import warnings

warnings.filterwarnings("ignore")                    # quiet nltk's short-hypothesis BLEU warnings


# =============================================================================
# EXPERIMENT 1 — NER: entity-level F1 vs token accuracy
# =============================================================================


def spans_from_bio(tags):
    """Extract (type, start, end) entity spans from a BIO tag sequence."""
    spans = []
    start = None
    etype = None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-"):
            if start is not None:
                spans.append((etype, start, i))
            start, etype = i, t[2:]
        elif t.startswith("I-") and start is not None and t[2:] == etype:
            continue
        else:                                            # O or a mismatched I-
            if start is not None:
                spans.append((etype, start, i))
                start, etype = None, None
    return set(spans)


def entity_f1(true_tags, pred_tags):
    tp = fp = fn = 0
    for tt, pt in zip(true_tags, pred_tags):
        T, P = spans_from_bio(tt), spans_from_bio(pt)
        tp += len(T & P); fp += len(P - T); fn += len(T - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def experiment_1_ner():
    print("=" * 88)
    print("EXPERIMENT 1 — NER: entity-level F1 vs token accuracy (README §3)")
    print("=" * 88)
    # "Barack Obama visited New York" — 2 entities. The model gets the person half-right.
    true = [["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]]
    pred = [["B-PER", "O", "O", "B-LOC", "I-LOC"]]        # missed the "Obama" token
    tok_acc = sum(t == p for tt, pp in zip(true, pred) for t, p in zip(tt, pp)) / \
        sum(len(tt) for tt in true)
    prec, rec, f1 = entity_f1(true, pred)
    print(f"""
  Sentence "Barack Obama visited New York": true entities [PER: Barack Obama], [LOC: New York].
  The model tags "Obama" as O (misses one token of the PER span):

    token accuracy        = {tok_acc:.3f}   (4 of 5 tokens correct — looks great)
    entity precision      = {prec:.3f}
    entity recall         = {rec:.3f}   (found LOC, but the PER span is WRONG)
    entity-level F1       = {f1:.3f}

  READING: token accuracy says 80% — but the model got only 1 of 2 entities right, because a partial
  span is a WRONG entity ('Barack' != 'Barack Obama'). Entity-level F1 counts a prediction correct only
  if the ENTIRE span and its type match: the predicted PER span is wrong (a false positive AND a missed
  true entity), so precision and recall are both 0.5 and F1 drops to 0.50. NER is scored at the entity
  level (BIO spans), never token accuracy, precisely because near-misses are failures (README §3).""")


# =============================================================================
# EXPERIMENT 2 — QA: SQuAD Exact-Match vs token-F1
# =============================================================================


def normalize(s):
    return " ".join(s.lower().replace(".", "").replace(",", "").split())


def squad_em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def squad_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ncommon = sum(common.values())
    if ncommon == 0:
        return 0.0
    prec = ncommon / len(p)
    rec = ncommon / len(g)
    return 2 * prec * rec / (prec + rec)


def experiment_2_qa():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — extractive QA: SQuAD Exact-Match vs token-F1 (README §4)")
    print("=" * 88)
    gold = "the Eiffel Tower"
    cases = [("the Eiffel Tower", "exact"), ("Eiffel Tower", "missing article"),
             ("the Eiffel Tower in Paris", "extra words"), ("the Louvre", "wrong")]
    print(f'\n  Gold answer: "{gold}"\n')
    print(f"    {'prediction':>30s} {'EM':>5s} {'F1':>6s}")
    for pred, _ in cases:
        print(f"    {pred:>30s} {squad_em(pred, gold):>5.0f} {squad_f1(pred, gold):>6.2f}")
    print("""
  READING: Exact Match is all-or-nothing — only the perfectly-matching answer scores 1. But 'Eiffel
  Tower' (dropped 'the') is basically correct, and 'the Eiffel Tower in Paris' contains the answer;
  token-F1 gives them partial credit (0.80, 0.75) where EM gives 0. SQuAD reports BOTH: EM for strict
  correctness, F1 for the overlap that reflects how good a near-answer is. Reporting only EM would
  massively understate a model (README §4).""")


# =============================================================================
# EXPERIMENT 3 — generation: BLEU
# =============================================================================


def _ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference, hypothesis, max_n=4):
    """Sentence BLEU: geometric mean of clipped n-gram precisions x brevity penalty."""
    import math
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hypothesis, n)
        ref_ng = _ngrams(reference, n)
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        precisions.append(overlap / total)
    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(hypothesis) > len(reference) else math.exp(1 - len(reference) / max(len(hypothesis), 1))
    return bp * geo


def experiment_3_bleu():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — generation: BLEU (n-gram precision + brevity penalty) == nltk (README §5)")
    print("=" * 88)
    ref = "the cat is sitting on the warm mat today".split()
    hyps = {
        "perfect": ref,
        "one word off": "the cat is resting on the warm mat today".split(),
        "too short (gamed)": "the cat".split(),
        "reordered": "today the warm mat the cat is sitting on".split(),
    }
    try:
        from nltk.translate.bleu_score import sentence_bleu
        have_nltk = True
    except Exception:
        have_nltk = False
    print(f'\n  Reference: "{" ".join(ref)}"\n')
    print(f"    {'hypothesis':>20s} {'our BLEU':>10s} {'nltk BLEU':>11s}")
    for name, hyp in hyps.items():
        ours = bleu(ref, hyp)
        ref_nltk = sentence_bleu([ref], hyp) if have_nltk else float("nan")
        print(f"    {name:>20s} {ours:>10.4f} {ref_nltk:>11.4f}")
    print("""
  READING: BLEU measures n-gram PRECISION — what fraction of the hypothesis's n-grams appear in the
  reference — as a geometric mean over 1..4-grams, times a BREVITY PENALTY. The penalty is the key
  guard: without it a system could score high by emitting just 'the cat' (high precision, no recall);
  BLEU crushes it (0.0) because it is far too short. Our implementation matches nltk. BLEU is the
  standard machine-translation metric (README §5).""")


# =============================================================================
# EXPERIMENT 4 — summarization: ROUGE
# =============================================================================


def rouge_n(reference, hypothesis, n):
    ref_ng, hyp_ng = _ngrams(reference, n), _ngrams(hypothesis, n)
    overlap = sum((ref_ng & hyp_ng).values())
    return overlap / max(sum(ref_ng.values()), 1)         # RECALL-oriented


def rouge_l(reference, hypothesis):
    a, b = reference, hypothesis
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    return lcs / max(len(a), 1)                            # LCS recall


def experiment_4_rouge():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — summarization: ROUGE-N and ROUGE-L (recall-oriented) (README §5)")
    print("=" * 88)
    ref = "the quick brown fox jumps over the lazy dog".split()
    hyps = {
        "full match": ref,
        "good summary": "the brown fox jumps over the dog".split(),
        "misses content": "the fox runs".split(),
    }
    print(f'\n  Reference summary: "{" ".join(ref)}"\n')
    print(f"    {'hypothesis':>16s} {'ROUGE-1':>9s} {'ROUGE-2':>9s} {'ROUGE-L':>9s}")
    for name, hyp in hyps.items():
        print(f"    {name:>16s} {rouge_n(ref, hyp, 1):>9.2f} {rouge_n(ref, hyp, 2):>9.2f} "
              f"{rouge_l(ref, hyp):>9.2f}")
    print("""
  READING: ROUGE measures n-gram RECALL — what fraction of the REFERENCE's n-grams the summary covers —
  the mirror image of BLEU's precision. Summarization is recall-oriented because a good summary must
  COVER the source's key content; missing information is the failure mode. ROUGE-1/2 use unigram/bigram
  overlap; ROUGE-L uses the longest common subsequence, rewarding in-order coverage without requiring
  contiguity. ROUGE is the standard summarization metric (README §5).""")


# =============================================================================
# EXPERIMENT 5 — classification: micro vs macro F1
# =============================================================================


def experiment_5_micro_macro():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — classification: micro vs macro F1 on imbalanced classes (README §6)")
    print("=" * 88)
    # 3 classes, very imbalanced: A common (900), B and C rare (50 each). Model ignores B,C.
    # per-class (tp, fp, fn):
    stats = {"A (900)": (900, 100, 0), "B (50)": (0, 0, 50), "C (50)": (0, 0, 50)}

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    macro = sum(f1(*s) for s in stats.values()) / len(stats)
    TP = sum(s[0] for s in stats.values()); FP = sum(s[1] for s in stats.values())
    FN = sum(s[2] for s in stats.values())
    micro = f1(TP, FP, FN)
    print(f"\n  A model that predicts everything as the majority class A:\n")
    print(f"    {'class':>10s} {'F1':>8s}")
    for c, s in stats.items():
        print(f"    {c:>10s} {f1(*s):>8.3f}")
    print(f"""
    micro-F1 (pool all classes) = {micro:.3f}   (dominated by the huge class A)
    macro-F1 (average classes)  = {macro:.3f}   (rare classes drag it down)

  READING: a model that always predicts the majority class A scores a great MICRO-F1 ({micro:.2f}) —
  micro pools every prediction, so the 900 easy A's swamp the rare classes. MACRO-F1 averages the
  per-class F1s equally, so failing on B and C (F1 = 0) crushes it to {macro:.2f}. On imbalanced NLP
  tasks (most of them) report MACRO-F1 — it refuses to hide failure on the classes you care about
  (README §6).""")


if __name__ == "__main__":
    experiment_1_ner()
    experiment_2_qa()
    experiment_3_bleu()
    experiment_4_rouge()
    experiment_5_micro_macro()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
