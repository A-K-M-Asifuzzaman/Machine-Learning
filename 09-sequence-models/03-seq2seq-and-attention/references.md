# 09.03 — References: Seq2seq & Attention

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | Encoder–decoder, the bottleneck | Sutskever et al. (2014); Cho et al. (2014) |
| §3-§4 | Attention, additive scoring, alignment | Bahdanau et al. (2015) |
| §3 | Dot / multiplicative scoring | Luong et al. (2015) |
| §5 | Scaled dot-product | Vaswani et al. (2017) |
| §6 | Beam search, length normalization | Wu et al. (2016) |
| §7 | Attention → transformer | Vaswani et al. (2017) |

---

## The core papers

- **Sutskever, I., Vinyals, O. & Le, Q. (2014).** "Sequence to Sequence Learning with Neural Networks."
  *NeurIPS*. — the **encoder–decoder** for neural machine translation; the fixed-context design (§1-§2).
  <https://arxiv.org/abs/1409.3215>.
- **Cho, K. et al. (2014).** "Learning Phrase Representations using RNN Encoder–Decoder." *EMNLP*. — the
  encoder–decoder framing (and the GRU) (§1). <https://arxiv.org/abs/1406.1078>.
- **Bahdanau, D., Cho, K. & Bengio, Y. (2015).** "Neural Machine Translation by Jointly Learning to
  Align and Translate." *ICLR*. — **attention** (additive), removing the bottleneck, and interpretable
  alignments (§2-§4). <https://arxiv.org/abs/1409.0473>.
- **Luong, M.-T., Pham, H. & Manning, C. (2015).** "Effective Approaches to Attention-based Neural
  Machine Translation." *EMNLP*. — **dot-product / multiplicative** attention and global vs local
  attention (§3). <https://arxiv.org/abs/1508.04025>.
- **Vaswani, A. et al. (2017).** "Attention Is All You Need." *NeurIPS*. — **scaled dot-product
  attention** ($1/\sqrt{d}$) and the transformer that drops recurrence (§5, §7).
  <https://arxiv.org/abs/1706.03762>.

## Decoding

- **Wu, Y. et al. (2016).** "Google's Neural Machine Translation System." — **beam search** with length
  normalization and coverage penalties in a production NMT system (§6). <https://arxiv.org/abs/1609.08144>.
- **Freitag, M. & Al-Onaizan, Y. (2017).** "Beam Search Strategies for Neural Machine Translation." —
  beam-width effects and the length-bias problem (§6). <https://arxiv.org/abs/1702.01806>.

## Textbook

- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, §10.4 & §12.4.3** —
  encoder–decoder and attention. Free at <https://www.deeplearningbook.org/>.
- **Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed. draft), ch. on MT & attention.**
  <https://web.stanford.edu/~jurafsky/slp3/>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [PyTorch seq2seq+attention tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) | a full Bahdanau-attention translator |
| [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) | production seq2seq with beam search |
| [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | the scaled dot-product op (§5) |

---

## Deferred to later chapters

- **Self-attention and the transformer** → [11.01](../../11-transformers-llms/01-attention/), [11.02](../../11-transformers-llms/02-transformer-architecture/)
- **The RNN encoders/decoders** → [09.01](../01-rnn/), [09.02](../02-lstm-gru/)
- **Positional encodings** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **LLM decoding strategies (top-k, nucleus sampling)** → [11.06](../../11-transformers-llms/06-decoding-generation/)
- **Machine translation and NLP tasks** → [Part 10](../../10-nlp/)
