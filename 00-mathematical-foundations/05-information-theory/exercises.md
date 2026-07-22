# 00.05 — Exercises: Information Theory

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that the three axioms in §2 ($S(1)=0$, decreasing, additive over independent events)
force $S(p) = -\log p$ up to the choice of base.

**D2.** Prove $H(X)\le\log K$ for a variable with $K$ outcomes, with equality iff uniform.
*Hint*: Jensen, or a Lagrange multiplier on $\sum_x p(x) = 1$.

**D3.** Compute $H$ for $p = (1/2, 1/4, 1/8, 1/8)$ by hand, then construct its Huffman code and
verify the average length equals $H$ exactly. Why is this case exact when others are not?

**D4.** Prove the chain rule $H(X,Y) = H(X) + H(Y\mid X)$ directly from the definitions.

**D5.** Prove $H(Y\mid X)\le H(Y)$. Then give a concrete example where a *particular* observation
increases your uncertainty, and explain why this does not contradict the inequality.

**D6.** Derive $H(p,q) = H(p) + D_{\mathrm{KL}}(p\Vert q)$. State the two practical consequences
for a training loss curve.

**D7.** Prove $D_{\mathrm{KL}}(p\Vert q)\ge 0$ using Jensen's inequality. Identify exactly where
the convexity of $-\log$ is used.

**D8.** Show that for a one-hot target, cross-entropy reduces to $-\log q(c)$, and hence that
cross-entropy minimization is maximum likelihood.

**D9.** Give an explicit $p, q$ pair for which $D_{\mathrm{KL}}(p\Vert q)$ is finite but
$D_{\mathrm{KL}}(q\Vert p) = \infty$. What structural property of the two distributions causes it?

**D10.** Explain, using the direction of the expectation, why forward KL is mode-covering and
reverse KL is mode-seeking. Then predict what each would do fitting a Gaussian to a *three*-mode
target, and check your prediction with `from_scratch.py`.

**D11.** Prove $I(X;Y) = H(X) - H(X\mid Y) = H(X)+H(Y)-H(X,Y)$ and that all three forms agree.

**D12.** Show $I(X;Y) = D_{\mathrm{KL}}(p(x,y)\Vert p(x)p(y))$, and use this to prove $I \ge 0$
with equality iff independence.

**D13.** Prove that Jensen-Shannon divergence is bounded above by $\log 2$. Then explain the
consequence for GAN training when $p$ and $q$ have disjoint support.

**D14.** Show that the maximum-entropy distribution on $\mathbb{R}$ with fixed mean and variance
is the Gaussian. *Hint*: Lagrange multipliers on the normalization, mean, and variance
constraints.

**D15.** Show that maximizing entropy subject to matching feature expectations yields an
exponential-family distribution, and that for a classification setting this is exactly softmax
regression.

**D16.** Show that BIC $= -2\ell + k\log n$ can be read as a description length, identifying which
term encodes the model and which the data.

**D17.** Explain why perplexity is not comparable across tokenizers. What quantity *is* comparable?

---

## Tier 2 — Implementation

**I1.** Implement `entropy`, `cross_entropy`, `kl_divergence`. Verify the decomposition
$H(p,q) = H(p)+D_{\mathrm{KL}}$ to machine precision on random distributions.

**I2.** Implement Huffman coding. Reproduce Experiment 1 and verify $H \le L < H+1$ on 20 random
distributions. Find the distribution shape that maximizes the gap $L - H$.

**I3.** Verify $D_{\mathrm{KL}} \ge 0$ empirically over 10,000 random pairs. Then find the pair
maximizing the asymmetry $|D(p\Vert q) - D(q\Vert p)|$ for fixed support size.

**I4.** Reproduce Experiment 3. Then extend it: fit a **two**-component mixture to a three-mode
target under both KL directions and describe what each does.

**I5.** Implement mutual information from a joint histogram. Reproduce Experiment 4, then
investigate the estimator's bias: plot estimated MI for two *independent* variables against $n$
and against the number of bins. What does this tell you about trusting small-sample MI?

**I6.** Implement information gain. Reproduce Experiment 5, then implement the **gain ratio**
(C4.5's fix) and confirm it demotes the high-cardinality ID feature.

**I7.** Train a small classifier and plot the training cross-entropy alongside an estimate of
$H(p)$ for your data (e.g. by fitting the label distribution given features). Show the loss
approaching a floor rather than zero.

**I8.** Implement label smoothing and explain it information-theoretically: what target
distribution are you now cross-entropy-matching, and what does that do to $H(p)$ and the minimum
achievable loss?

**I9.** Implement knowledge distillation loss $D_{\mathrm{KL}}(p_{\text{teacher}}^{T}\Vert p_{\text{student}}^{T})$
with temperature $T$. Show empirically that the teacher's full distribution carries more
information per example than its argmax.

**I10.** Implement perplexity for a character-level bigram model on any text. Then build a
word-level model on the same text and demonstrate that their perplexities are not comparable,
while their bits-per-character are.

**I11.** Implement the InfoNCE loss and show it is a lower bound on mutual information between
two views of the same data.

---

## Tier 3 — Interview

**Q1.** What is entropy, in one sentence, without saying "disorder"?

**Q2.** Why is cross-entropy the loss for classification? Give at least two independent
justifications.

**Q3.** Is KL divergence a distance? What properties does it fail?

**Q4.** What is the difference between forward and reverse KL, and where does each appear in ML?

**Q5.** Why do VAE samples tend to look blurry and GAN samples sharp but less diverse?

**Q6.** Your training cross-entropy plateaus at 0.4 and won't go lower. Is the model broken?

**Q7.** A model achieves 100% accuracy but a terrible cross-entropy. How?

**Q8.** What does mutual information tell you that correlation doesn't? Give an example.

**Q9.** What does a decision tree actually maximize when it chooses a split?

**Q10.** Why does information gain prefer high-cardinality features, and what are two fixes?

**Q11.** What does perplexity 20 mean? When can you compare two models' perplexities?

**Q12.** Why is the Gaussian the maximum-entropy distribution for a given mean and variance, and
why does that matter?

**Q13.** Explain knowledge distillation in information-theoretic terms. Why does the teacher's
soft output help more than the hard label?

**Q14.** Why does RLHF include a KL penalty against the reference model?

**Q15.** What happens to cross-entropy if your model outputs exactly zero for the true class, and
how do real implementations avoid it?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive $-\log p$ from the axioms rather than quoting it
- [ ] Explain entropy as a code length, with a worked Huffman example
- [ ] Give three independent derivations of cross-entropy as the classification loss
- [ ] Predict the qualitative failure mode of a model from which KL direction it minimizes
- [ ] Explain why your loss has a nonzero floor, and what that floor is
- [ ] Use mutual information to find dependence that correlation misses
- [ ] Recognize information gain as mutual information, and name its bias
