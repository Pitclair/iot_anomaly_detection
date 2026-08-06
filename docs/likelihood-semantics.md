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

- `scipy` is the temporary implementation of the kernel using
  `scipy.special.gammaln`;
- `lm` will be the Languasco-Migliardi implementation of the same kernel;
- neither backend includes the multinomial coefficient.

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
