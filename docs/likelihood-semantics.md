# Likelihood semantics

LM-IDNet deliberately separates the parameter-dependent likelihood kernel used
for model fitting from the complete probability used to score a count window.
The two quantities must not be confused.

For a count vector

\[
\mathbf{x}=(x_1,\ldots,x_K), \qquad N=\sum_{k=1}^{K}x_k,
\]

and fitted parameters

\[
\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_K), \qquad
\alpha_0=\sum_{k=1}^{K}\alpha_k,
\]

the likelihood backend calculates only the parameter-dependent kernel:

\[
K(\mathbf{x};\boldsymbol{\alpha})
=
\log\Gamma(\alpha_0)
-\log\Gamma(\alpha_0+N)
+\sum_{k=1}^{K}
\left[
\log\Gamma(\alpha_k+x_k)-\log\Gamma(\alpha_k)
\right].
\]

For a matrix of independent count windows, the backend returns the sum of this
kernel over all rows.

## Backend contract

Both configured backends have the same semantics:

- `lm` is the production native Languasco--Migliardi implementation. It
  evaluates paired log-gamma differences with the configured
  `precision_digits` value;
- `scipy` is the reference implementation using `scipy.special.gammaln`;
- neither backend includes the multinomial coefficient.

The LM adapter sends the complete matrix to the bundled C library in one call.
The native implementation reuses setup that is constant for that call while
still evaluating and summing each row independently. Its project-specific
changes cover integration, validation, zero counts, memory lifecycle, and the
matrix boundary; they do not change the underlying LM approximation. The
original licence and attribution remain in
[`native/LICENSE.`](../src/lm_idnet/algorithms/native/LICENSE.).

The multinomial coefficient is excluded during parameter estimation because it
depends only on the observed counts. It is constant while alpha is updated, so
it cannot change the fitted alpha, concentration, or psi, and it cancels from
likelihood differences used for convergence.

Consequently, the initial and final likelihoods recorded by training are kernel
values. They are fitting diagnostics and must not be compared directly with
calibration thresholds or anomaly scores.

## Complete window probability

The complete Dirichlet-multinomial log-probability additionally contains the
logarithm of the multinomial coefficient:

\[
C(\mathbf{x})
=
\log\frac{N!}{\prod_{k=1}^{K}x_k!}
=
\log\Gamma(N+1)
-\sum_{k=1}^{K}\log\Gamma(x_k+1).
\]

Therefore, the complete raw score for one window is:

\[
\log P(\mathbf{x}\mid\boldsymbol{\alpha})
=
C(\mathbf{x})+K(\mathbf{x};\boldsymbol{\alpha}).
\]

The coefficient is calculated outside the likelihood backend by
[`log_multinomial_coefficient()`](../src/lm_idnet/algorithms/dirichlet_multinomial.py).
The calibration and scoring stages must both combine it with the kernel using
this same complete formula; otherwise, their values would be on different
scales.

| Pipeline operation | Kernel | Multinomial coefficient |
|---|---:|---:|
| Update alpha | No | No |
| Check fitting convergence | Yes | No |
| Calculate concentration and psi | No | No |
| Calibrate an anomaly threshold | Yes | Yes |
| Score a new window | Yes | Yes |

The implementation contract is defined by
[`algorithms/log_likelihood.py`](../src/lm_idnet/algorithms/log_likelihood.py)
and its numerical tests in
[`tests/test_likelihood.py`](../tests/test_likelihood.py).

## Joint detector score

New calibration artifacts keep the DM model unchanged and add two upper count
tails: total packets and ARP packets. Total volume complements a model conditioned
on the packet total; ARP volume exposes a small category that can be masked by
dominant traffic. Both counts already exist in processed windows.

Fit-only medians and IQRs put the DM log-probability, negative `log1p(total)`, and
negative `log1p(ARP)` on comparable scales. Constant fit features use unit scale.
The minimum of these three scores is calibrated once at the configured quantile
on normal calibration windows. This accounts for their correlation without
label-tuned weights or separate extreme-tail budget allocations. The quantile is
an empirical calibration target, not a guarantee under future traffic drift.

Events retain the raw DM value in `composition_score`. Their `feature_margins`
give signed distances from the calibrated cutoffs; `score` is the minimum,
`threshold` is zero, and `severity` is `max(0, -score)`. Negative margins alert.
The threshold artifact's `score_iqr` stores the fit reference scale for the DM;
each `count_tails` entry stores its log-count cutoff and reference scale.

Adaptive modes update DM parameters/thresholds using DM scores, while keeping
the original DM and count tails as fixed detection references. Conservative
learning also excludes windows near either count cutoff. This prevents updates
from absorbing volume evidence or suppressing an original DM alert. It can also
retain an original false alert after benign drift; replacing that trusted
reference requires a new normal-data fit/calibration. Legacy learning retains
its historical admission and update policy, with the same fixed detection
references when a new calibration artifact is supplied.

Threshold artifacts without `count_tails` retain the earlier DM-only behavior.
Run `calibrate` and prepare a new detector state to enable the combined detector;
existing snapshots retain their embedded calibration.
