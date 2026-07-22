# 00.06 — Exercises: Numerical Methods

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write 0.1 in binary. Show it is infinitely repeating, and compute the exact float64 value
stored. Then explain precisely why `0.1 + 0.2 != 0.3`.

**D2.** Derive machine epsilon for float64, float32, float16, and bfloat16 from their significand
widths. Then compute, for each, the smallest $x$ such that $1 + x \ne 1$.

**D3.** Show that subtracting two numbers agreeing to $k$ significant digits amplifies relative
error by roughly $10^{k}$.

**D4.** Derive the numerically stable form of the quadratic formula for the root that suffers
cancellation. Verify it on $x^{2}+10^{8}x+1=0$.

**D5.** Prove the log-sum-exp identity $\log\sum_i e^{x_i} = c + \log\sum_i e^{x_i-c}$ for any $c$.
Then explain why $c = \max_i x_i$ is optimal — what does each of the three properties in §7 buy?

**D6.** Prove that softmax is exactly shift-invariant. Why does this make the stable version an
*identity* rather than an approximation?

**D7.** Derive $\log\mathrm{softmax}(\mathbf{x})_i = x_i - \mathrm{LSE}(\mathbf{x})$ and explain
what is lost by computing `log(softmax(x))` instead.

**D8.** Derive the stable binary cross-entropy form
$L = \max(z,0) - zy + \log(1+e^{-|z|})$ from $-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$.

**D9.** Show that for large positive $z$ and $y=0$, the stable BCE reduces to $z$ — and that its
derivative with respect to $z$ is $\sigma(z) - y$, which stays bounded. Contrast with the naive
form's `inf`.

**D10.** Prove Welford's update is algebraically equivalent to the two-pass formula. *Hint*: show
$M_{2,n} = \sum_{i\le n}(x_i - \mu_n)^{2}$ by induction.

**D11.** Explain why Welford uses the **updated** mean $\mu_n$ in the $M_2$ update rather than
$\mu_{n-1}$. What breaks if you use the old one?

**D12.** Explain why Kahan's `c = (t - sum) - y` is not algebraically zero in floating point, and
what quantity it recovers. Why would an aggressive compiler optimization break it?

**D13.** Derive the optimal step size for a central-difference gradient check by minimizing
$h^{2} + \varepsilon/h$ over $h$. Show the answer is $O(\varepsilon^{1/3})$.

**D14.** Compare float16 and bfloat16 for representing a gradient of $3\times10^{-8}$ and an
activation of $10^{5}$. Which format survives each, and why does that make bfloat16 the better
choice for deep learning?

**D15.** Explain why floating-point addition is not associative, and give a three-number
counterexample. What does this imply for reproducibility across GPUs?

---

## Tier 2 — Implementation

**I1.** Implement `logsumexp`. Verify it matches the naive version for small inputs and returns a
finite answer for `[1000, 1001, 1002]`. Compare against `scipy.special.logsumexp`.

**I2.** Implement `stable_softmax` and `log_softmax`. Find, by bisection, the exact logit at which
naive softmax returns `nan` in float64 and in float32.

**I3.** Implement `bce_with_logits`. Find the logit at which naive BCE becomes `inf` in float64
and float32. Verify your version matches `torch.nn.BCEWithLogitsLoss` to machine precision.

**I4.** Implement all three variance algorithms. Reproduce Experiment 4 and find, by searching
seeds and offsets, a case where the naive formula returns a **negative** variance.

**I5.** Implement Welford's algorithm in a streaming class with `update(x)` and `finalize()`.
Verify it against `np.var` on 1 million values, then extend it to compute a running covariance.

**I6.** Implement Kahan summation. Reproduce Experiment 5 including the pathological case, and
measure the crossover $n$ at which naive summation's error exceeds $10^{-10}$ relative.

**I7.** Implement gradient checking. Reproduce the U-shaped curve of Experiment 6 in both float64
and float32, and compare the optimal $h$ and the achievable accuracy in each.

**I8.** *(Break something on purpose.)* Train a small logistic regression using naive
sigmoid + BCE, with a learning rate large enough that logits grow. Find the step at which it
produces `nan`. Then swap in `bce_with_logits` and show it trains through.

**I9.** Simulate float16 training: take a float32 model, round the weights and the update to
float16 at each step, and show training stalls. Then add a float32 master copy and show it
recovers.

**I10.** Implement loss scaling: multiply the loss by $2^{15}$, compute gradients in float16,
unscale, and skip the step if any gradient is non-finite. Measure how often steps are skipped.

**I11.** Take any dataset with a large-offset feature (timestamps work well). Compute its variance
with all three algorithms and standardize with each. Show the naive route produces `nan` or
`inf` downstream.

**I12.** Write a `assert_finite` hook that registers on every module of a PyTorch model and
reports the first layer to emit a non-finite value. Test it by deliberately injecting a `log(0)`.

---

## Tier 3 — Interview

**Q1.** Why is `0.1 + 0.2 != 0.3`? How should you compare floats?

**Q2.** Your loss became `NaN` at step 5,000 after training normally. Walk through your diagnosis.

**Q3.** What is catastrophic cancellation? Give an ML example.

**Q4.** Why does `nn.CrossEntropyLoss` take logits rather than probabilities?

**Q5.** What does log-sum-exp do and why does every framework have it?

**Q6.** Why can't you just compute `log(softmax(x))`?

**Q7.** Why does `BCEWithLogitsLoss` exist when `Sigmoid` and `BCELoss` already do?

**Q8.** What is wrong with `Var(X) = E[X²] − E[X]²`? When exactly does it bite?

**Q9.** What is Welford's algorithm and where would you find it in a deep learning library?

**Q10.** What is the difference between float16 and bfloat16? Which would you pick for training
and why?

**Q11.** Why does mixed-precision training keep a float32 copy of the weights?

**Q12.** What is loss scaling and which format needs it?

**Q13.** Why is your GPU training run not bitwise reproducible, and what would it cost to make it
so?

**Q14.** How do you check a hand-derived gradient? What $h$ would you use, and why not smaller?

**Q15.** Why should you never call `np.linalg.inv`?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain a `NaN` loss from first principles rather than by trial and error
- [ ] Write a numerically stable softmax and cross-entropy without looking them up
- [ ] Spot the three or four expressions in any codebase that are cancellation traps
- [ ] Say why `log1p`, `expm1`, `logaddexp`, and `hypot` exist
- [ ] Choose between float16 and bfloat16 and defend the choice
- [ ] Debug a wrong gradient with a properly-tuned finite-difference check
- [ ] Recognize that "correct math" and "correct code" are different claims
