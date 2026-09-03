# Professor-style LM architecture experiment

Date: 2026-09-03 (UTC)

## Verdict

The professor-style aggregate convergence check does make C-wrapped LM faster
than SciPy on this machine, without worsening the fitted model:

- across 84 paired paper-style fits, LM used 12.2% less likelihood time and
  8.8% less full wall time;
- on the current pooled 846-row workload, one aggregate LM call took 8.44 us
  versus 9.72 us for SciPy, and a literal professor-style fit was 3.1% faster;
- LM and SciPy differed by at most 1.45e-7 relatively in fitted alpha, while
  goodness-of-fit differences were at most 6.86e-11;
- forecast differences were at most 2.84e-13, roughly nine orders of magnitude
  smaller than the variation between the three fixed data samples.

So LM holds up on runtime under that workload, but not as a meaningfully more
accurate estimator. The apparent accuracy advantage is numerical dust caused by
slightly different stopping iterations.

The aggregate architecture should not replace LM-IDNet's matrix likelihood. It
checks convergence with the likelihood of one pooled count vector while the
fixed-point update still optimizes independent count windows. That hybrid is
not the likelihood of either model. A small native optimization tested here
removes almost all of LM's matrix penalty while preserving the correct
independent-window semantics.

## What was run

The reproducible experiment is
[`scripts/lm_professor_experiment.py`](../scripts/lm_professor_experiment.py).
Its complete machine-readable output is
[`lm_professor_experiment.json`](../reports/D-LinkDayCam5/lm_professor_experiment.json).

It follows the paper-shaped protocol as closely as the available processed data
allows:

- separate models for October 8, 9, and 12--16;
- random 10%, 15%, 20%, and 25% samples;
- three deterministic repeats, with the exact same input given to both
  backends;
- the professor's moment-based alpha initialization;
- row-wise Minka digamma updates;
- one aggregate count vector for the convergence likelihood;
- two likelihood calls per iteration, matching the professor's implementation;
- six LM digits, tolerance 2e-9, and at most 100,000 iterations;
- 1,000 Dirichlet samples from each of the seven 25% daily models, averaged and
  compared with actual October 19, 21, and 22 category proportions.

The local processed captures contain 191/144/144/79/144/144/144 training
windows rather than the paper's 196/143/143/78/143/143/143. This is a controlled
comparison of backends and architectures on LM-IDNet data, not an exact
reproduction of the paper's tables.

Environment: Python 3.12.14, NumPy 2.5.1, SciPy 1.18.0, Linux x86-64. The
optimized native probe was compiled with `-O3` in the existing local
`lm-idnet:latest` image and loaded into the same Python process as the other
backends.

## Paper-style modeling results

Each time below is the mean total for fitting all seven training days. A row
contains 21 fits per backend: seven days times three samples.

| Sample | Converged | SciPy wall | LM wall | LM wall reduction | SciPy likelihood | LM likelihood | LM likelihood reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 17/21 | 4.8933 s | 4.4593 s | 8.9% | 3.5746 s | 3.1396 s | 12.2% |
| 15% | 20/21 | 2.6517 s | 2.4187 s | 8.8% | 1.9094 s | 1.6766 s | 12.2% |
| 20% | 21/21 | 2.3125 s | 2.1064 s | 8.9% | 1.6485 s | 1.4447 s | 12.4% |
| 25% | 21/21 | 1.3957 s | 1.2799 s | 8.3% | 0.9779 s | 0.8599 s | 12.1% |
| All 84 paired fits | 79/84 | 33.7594 s | 30.7930 s | 8.8% | 24.3311 s | 21.3625 s | 12.2% |

The two backends had exactly the same five non-converging samples: four at 10%
and one at 15%. These small samples drive concentration upward toward the
multinomial boundary (`psi -> 0`) instead of reaching a finite overdispersed
optimum. Raising the iteration cap delays rather than fixes that condition. It
is a sampling/estimator issue, not an LM failure.

Across the 84 pairs, LM stopped earlier 25 times, at the same iteration 24
times, and later 35 times. The largest difference was 91 iterations. This is
only 0.09% of the configured cap and produced a maximum relative alpha
difference of 1.45e-7.

At the paper's selected 25% sample size, mean fit quality was:

| Metric | SciPy | LM | LM minus SciPy |
|---|---:|---:|---:|
| TVD | 0.0143270133477 | 0.0143270133455 | -2.21e-12 |
| KL divergence | 0.00150580503491 | 0.00150580503410 | -8.11e-13 |
| Euclidean distance | 0.0166562677851 | 0.0166562677826 | -2.44e-12 |
| MSE | 0.000152414911185 | 0.000152414911109 | -7.63e-14 |

LM's averages happen to be lower, but the differences are too small to support
an accuracy claim. Individual fits also do not consistently favor LM: depending
on the metric, LM was lower in 33--35 pairs, equal in 24, and higher in 25--27.

## Forecast results

The same random stream was reset for LM and SciPy so this test measures backend
effects rather than Monte Carlo noise.

| Test day | Mean Brier, SciPy | Mean Brier, LM | Mean JSD, SciPy | Mean JSD, LM |
|---|---:|---:|---:|---:|
| Oct 19 | 5.556197909e-5 | 5.556197902e-5 | 3.324157281e-5 | 3.324157276e-5 |
| Oct 21 | 1.686972823e-4 | 1.686972822e-4 | 1.060210323e-4 | 1.060210322e-4 |
| Oct 22 | 2.309953807e-4 | 2.309953805e-4 | 1.489322600e-4 | 1.489322598e-4 |

LM was numerically lower in all nine paired day/repeat comparisons, but its
largest advantage was only 2.84e-13 for Brier and 1.91e-13 for JSD. By contrast,
changing the deterministic training sample changed Brier by 1.08e-4 to 2.26e-4
depending on the test day. The sample choice matters about a billion times more
than the backend here.

These values cannot be used as an exact reproduction of the paper because the
window counts differ and the paper does not specify enough sampling detail to
reconstruct its exact random realizations. They do establish that the LM-versus-
SciPy forecast ranking is not practically meaningful when inputs and randomness
are controlled.

## Direct architecture comparison on the 846-row fit

Using the 846-by-4 training matrix and the professor's initialization, before
the native optimization was installed:

| Operation | SciPy | LM | LM / SciPy |
|---|---:|---:|---:|
| Correct row-wise likelihood call | 49.92 us | 114.34 us | 2.290 |
| Professor-style aggregate likelihood call | 9.72 us | 8.44 us | 0.868 |
| Literal aggregate-convergence full fit | 0.16425 s | 0.15909 s | 0.969 |

Nine interleaved timing batches were used. Aggregation therefore reverses the
microbenchmark and makes the complete fit slightly faster with LM.

Changing only the stopping likelihood from row-wise to aggregate changed alpha
by at most 2.18e-5 relatively. Recalibrating and rescoring the development set
with that alpha shifted the threshold by 1.63e-4 and individual scores by at
most 6.18e-4, but still produced three anomalies and zero decision mismatches
across all 270 windows.

This is why the aggregate experiment looks safe on this dataset. It does not
make the hybrid objective mathematically correct, and a close decision boundary
on another dataset could still flip.

## The overlooked native cost

The previous matrix adapter did more than iterate over rows. For every row it:

1. frees and reallocates four native work arrays;
2. recomputes the adaptive Euler--Maclaurin order and horizontal shift;
3. then evaluates that row's likelihood.

The second step depends on alpha, category probabilities, requested precision,
and category count. It does **not** depend on a row's observed counts, so doing
it 846 times per call is redundant.

The initial disposable C variant initialized the arrays once and moved that
adaptive search outside the row loop. The optimization has now been installed
in the runtime source and shared library. The old and optimized LM libraries
were bit-for-bit identical in 1,000 seeded stress cases; optimized LM differed
from SciPy by at most 6.71e-8.

| Correct row-wise operation | SciPy | Previous LM | Optimized LM |
|---|---:|---:|---:|
| One 846-by-4 likelihood | 49.96 us | 114.68 us | 55.46 us |
| Complete model fit, median of 7 | 0.18129 s | 0.30410 s | 0.19297 s |

This removes about 51.5% of the LM likelihood-call time and 35.9% of its
full-fit time. The production full fit is 6.4% slower than SciPy, within the
paper's 10% equivalence criterion, while keeping the model contract intact.
The implementation and verification are recorded as Change 006 in
[`lm_backend_change_record.tex`](lm_backend_change_record.tex).

## Where LM actually adds numerical value

At the current fitted concentration, neither backend is numerically stressed.
The aggregate kernel was checked against a 100-decimal-digit `mpmath` reference
using the current count totals and category proportions:

| Concentration | `psi` | SciPy absolute error | LM absolute error |
|---:|---:|---:|---:|
| 122.22 (current) | 8.18e-3 | 3.07e-10 | 3.91e-10 |
| 1e8 | 1e-8 | 2.50e-7 | 1.50e-8 |
| 1e10 | 1e-10 | 2.83e-5 | 5.65e-8 |
| 1e12 | 1e-12 | 2.02e-3 | 5.58e-5 |
| 1e14 | 1e-14 | 3.37e-1 | 1.03e-3 |

LM's real advantage appears near the multinomial limit, where subtracting two
large ordinary `gammaln` values loses precision. At the current `psi = 0.00818`,
SciPy is already accurate and there is no accuracy left for LM to improve.

## Architectural conclusions

Three facts explain all observed behavior:

1. **LM is only the convergence backend.** Alpha updates use SciPy's digamma in
   both implementations. With the same initial alpha and counts, LM and SciPy
   follow the same alpha sequence; only floating-point differences choose a
   slightly different stopping iteration.
2. **The professor implementation is hybrid.** It calculates update terms from
   every independent row but tests convergence with `K(sum(rows); alpha)` rather
   than `sum(K(row; alpha))`. These are different functions, not equivalent
   rearrangements.
3. **The current LM slowdown is mostly redundant native setup.** Aggregation
   hides that cost by reducing 846 evaluations to one, but computing invariant
   setup once achieves nearly the same speed without changing the statistical
   model.

## Recommendation

Do not adopt aggregate convergence in the production anomaly detector. Keep it
as a named paper-reproduction experiment.

The native invariant-setup optimization has now been applied while retaining
the row-wise likelihood contract. It reduces the LM/SciPy full-fit ratio from
about 1.67 to 1.064, preserves all numerical results, and retains LM's large
accuracy advantage for genuinely low-`psi` workloads.

Separately, replace the synthetic forecast scaffold with held-out daily data if
forecasting is meant to support a thesis claim. That change evaluates the model;
switching likelihood backends does not.
