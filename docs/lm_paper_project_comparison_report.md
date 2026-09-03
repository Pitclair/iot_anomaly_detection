# LM algorithm, the ALM IoT paper, and LM-IDNet: reproduction and comparison

Date of investigation: 2026-09-03 (UTC)

## Executive conclusion

The results make sense once three different experiments are separated:

1. The paper studies **traffic modeling and forecasting** and compares LM mainly
   with Yu--Shaw (YS).
2. The supplied `lm-algorithm` capsule defaults to a **different, later
   benchmark**: a five-category `IoT.csv` sample and C-wrapped LM versus
   `math.lgamma` and SciPy. It does not reproduce the paper's Camera 5 tables or
   run YS through its normal entry point.
3. LM-IDNet currently studies **anomaly scoring** and compares C-wrapped LM with
   SciPy on a pooled 846-by-4 training matrix. That is a useful backend
   equivalence/performance test, but it is not a reproduction of the paper's
   forecast experiment.

The strongest result in the current project is reassuring: LM and SciPy produce
nearly identical fitted parameters, thresholds, scores, and all 270 anomaly
decisions. There is no evidence here of an LM numerical-correctness bug.

The main mistaken assumption would be to conclude that the paper promises that
LM must make the current project faster or more accurate. It does not. The
paper's large speed advantage is principally against YS, on different data and
with different likelihood-call semantics. In the current implementation, LM is
about 1.67 times slower than SciPy for a complete model fit, although it is
about 10% faster for one single-window score.

There are two concrete project issues:

- The accepted Camera 5 model is presently fitted with the **SciPy backend**, not
  LM (`"log_likelihood_backend": "scipy"`). LM is exercised only by the backend
  benchmark unless the configuration is changed and the downstream artifacts
  are regenerated.
- The current `forecast` stage is a scaffold. It generates synthetic "actual"
  observations from the same fitted probabilities, ignores the supplied dataset
  argument, and does not evaluate held-out Camera 5 traffic. Its Brier Score and
  JSD therefore cannot support or contradict the paper's forecasting claim.

## Sources examined

- Professor's paper: [`ALM_IoT-CR.pdf`](../resouces/ALM_IoT-CR.pdf)
- Capsule instructions: [`lm-algorithm/REPRODUCING.md`](../lm-algorithm/REPRODUCING.md)
- Capsule entry point: [`executor_prod.py`](../lm-algorithm/code/executor_prod.py)
- Capsule benchmark: [`LogL-global-v5.py`](../lm-algorithm/code/LogL-global-v5.py)
- Older instrumented implementation containing YS:
  [`LogL-global-v5-instrumented-SYNCED-ULANG.py`](../lm-algorithm/code/LogL-global-v5-instrumented-SYNCED-ULANG.py)
- LM-IDNet configuration: [`d_link_day_cam5.json`](../configs/d_link_day_cam5.json)
- Existing project benchmark artifact:
  [`lm_backend_benchmark.json`](../reports/D-LinkDayCam5/lm_backend_benchmark.json)
- LM-IDNet benchmark implementation:
  [`benchmark_stage.py`](../src/lm_idnet/evaluation/benchmark_stage.py)
- LM-IDNet forecast implementation:
  [`forecasting_stage.py`](../src/lm_idnet/models/forecasting_stage.py)
- Likelihood definition used by LM-IDNet:
  [`likelihood-semantics.md`](likelihood-semantics.md)

No repository code was edited and no commit was created. The professor's code
was copied to disposable directories under `/tmp` for execution. The only
repository change made by this investigation is this requested report.

## 1. What LM is doing

LM is not itself the anomaly detector or the forecast model. It is a numerical
method for accurately evaluating paired log-gamma differences inside a
Dirichlet--multinomial (DM) log-likelihood. Minka's fixed-point iteration uses
likelihood differences to decide when the fitted Dirichlet parameter vector
`alpha` has converged.

This distinction matters:

- **Estimator:** Minka fixed-point iteration updates `alpha`.
- **Numerical backend:** LM, YS, `math.lgamma`, or SciPy evaluates the
  likelihood used in the convergence check.
- **Application:** the paper turns fitted models into forecasts; LM-IDNet turns
  a fitted model into calibrated anomaly scores.

A numerical backend can affect convergence, runtime, and fitted values when
ordinary subtraction loses precision. It does not, by itself, define a better
forecast or anomaly detector.

## 2. What the paper claims

### Dataset and protocol

The paper uses D-Link Camera 5 traffic aggregated into 10-minute vectors with
four categories: TCP, UDP, SSDP, and ARP.

- Modeling/training days: October 8, 9, 12, 13, 14, 15, and 16, 2020.
- Forecast/testing days: October 19, 21, and 22, 2020.
- Each training day is modeled separately using random 10%, 15%, 20%, and 25%
  samples.
- Precision is six mantissa digits, `K = 4`, and the convergence tolerance is
  `2e-9`.
- The environment was Python 3.10.2, NumPy 1.22.2, mpmath 1.2.1, and SciPy
  1.8.0 on an Intel Core i5 with 8 GB RAM.
- Tests were repeated three times and averaged.

The paper defines a dataset as overdispersed when `psi < 0.05`. All ten daily
sub-datasets meet that rule.

### Modeling result

The paper's primary comparison is LM versus YS:

- LM and YS generally require similar iteration counts.
- About 65% of the shown YS tests encounter the YS delta-barrier problem and
  require a bypass to return an estimate.
- LM is much faster than YS. The highlighted October 13, 20% sample takes about
  6 seconds with LM and about 8 minutes with YS, approximately an 80-times
  speedup.
- Excluding that extreme point, the reported average LM speedup over YS is 14;
  the lowest reported speedup is 9.11.

For LM model fit, Table III reports these ranges across the seven training
days:

| Metric | Paper range |
|---|---:|
| TVD | 0.004393 to 0.053707 |
| KL divergence | 0.000073 to 0.012963 |
| Euclidean distance | 0.005111 to 0.062229 |
| MSE | 0.000007 to 0.000984 |

The chi-squared and Z tests produce 14 test outcomes in Table IV; 13 pass and
the October 8 chi-squared test fails. The table appears to label October 16
twice, with one of those rows presumably intended to be October 15.

### Forecast result

The paper samples each of the seven fitted daily models 1,000 times and
averages the samples into a forecast model. LM, YS, and standard
`math.lgamma` (`lgam-std`) are then compared against the three held-out daily
models.

At the selected 25% training-sample size, Table V reports:

| Technique | Modeling time (s) |
|---|---:|
| `lgam-std` | 10.923 |
| LM | 11.083 |
| YS | 264.075 |

Thus LM is about 1.5% slower than `lgam-std`, not faster, but YS is about 23.8
times slower than LM. The paper describes LM and `lgam-std` as having no
significant speed difference because they are within 10%.

Table VI reports:

| Test day | LM BS | `lgam-std` BS | YS BS | LM JSD | `lgam-std` JSD | YS JSD |
|---|---:|---:|---:|---:|---:|---:|
| Oct 19 | 1.11e-5 | 1.36e-5 | 1.24e-5 | 1.65e-5 | 1.99e-5 | 1.78e-5 |
| Oct 21 | 5.09e-5 | 5.31e-5 | **4.98e-5** | 6.70e-5 | 7.13e-5 | 1.97e-4 |
| Oct 22 | 6.30e-5 | 6.72e-5 | 6.34e-5 | 9.09e-5 | 9.74e-5 | 1.48e-4 |

LM has the best JSD on all three days and the best BS on two of three days.
The absolute differences are small, and the paper does not report confidence
intervals for these forecast-score differences. “LM is best” is therefore a
result for this experiment, not a universal property of using LM.

## 3. What the supplied `lm-algorithm` folder actually runs

The capsule metadata describes a nine-dataset benchmark of LM,
`math.lgamma`, and SciPy. Its default production executor is not configured to
reproduce the paper's Camera 5 daily experiment:

- It selects `IoT.csv`, which has 3,420 rows and **five** columns.
- It takes an unseeded random 5% sample: 171 rows.
- It uses six-digit precision and obtains tolerance
  `10^(-8) / (5 + 1) = 1.6666666666666667e-9`.
- It runs standard `math.lgamma`, SciPy log-gamma, and **C-wrapped** LM.
- It computes goodness-of-fit statistics only from the last fit, LM.

This differs from the paper's four-category, per-day Camera 5 data and its
10--25% samples. It is a later LM benchmark capsule, not the paper's complete
forecast reproduction package.

There are also reproducibility limitations:

- The sampler has no seed, so each run uses different rows.
- `random.sample(range(0, len(lines)-1), ...)` permanently excludes the final
  CSV row from sampling.
- Printed `time` is the accumulated time inside the two likelihood evaluations
  per iteration, not wall-clock time for the full model fit. It excludes the
  fixed-point digamma updates and surrounding Python work.
- The production executor does not use `check=True` for child processes. A
  child can fail while the outer executor still exits successfully; this
  happened during the first literal launch when the requested `python3.11`
  command was absent.

### Execution environment

The original code was run unchanged from a temporary copy. The machine did not
provide a command named `python3.11`, so a temporary wrapper invoked the
project virtual environment's Python. This is an environment deviation from
the archived Docker image:

- Python 3.12.14
- NumPy 2.5.1
- mpmath 1.4.1
- SciPy 1.18.0

The bundled native `LM_time_lib.so` loaded successfully. All three production
runs completed.

### Three production runs

| Run | Math time (ms) / iterations | SciPy time (ms) / iterations | C-LM time (ms) / iterations | LM `psi` |
|---|---:|---:|---:|---:|
| 1 | 1.713 / 496 | 2.343 / 493 | 0.834 / 496 | 0.0791834 |
| 2 | 1.776 / 521 | 2.493 / 520 | 0.896 / 524 | 0.0728382 |
| 3 | 1.834 / 549 | 2.501 / 548 | 0.822 / 546 | 0.0708702 |
| Mean | 1.774 / 522.0 | 2.446 / 520.3 | 0.851 / 522.0 | 0.0742973 |

On this particular likelihood-only timer, C-wrapped LM was about 2.09 times
faster than `math.lgamma` and 2.87 times faster than SciPy.

The random-sample LM goodness-of-fit results varied as follows:

| Metric | Observed range across three runs |
|---|---:|
| TVD | 0.048319 to 0.055994 |
| KL divergence | 0.014800 to 0.019743 |
| Euclidean distance | 0.055437 to 0.063010 |
| MSE | 0.000615 to 0.000794 |
| Chi-squared test | rejected in all three runs |
| Z test | accepted in all three runs |

These numbers do not reproduce Table III. Some overlap its ranges, while all
three KL values are above the paper's maximum. That is not evidence against the
paper because this run used a different dataset, category count, sample ratio,
and software environment.

### Supplementary YS check

The older instrumented file contains implementations of YS and pure-Python LM,
but the YS invocation is inside a triple-quoted comment and is not called by
the shipped run path. In the disposable copy only, that existing block was
enabled to test the paper's main comparator on Run 1's sample.

- C-wrapped LM likelihood time: 0.745 ms, 496 iterations.
- Pure-Python LM likelihood time: 0.716 s, 496 iterations.
- YS immediately printed `ERROR from _init_a: delta-mesh not applicable`.
- YS produced no further output and did not finish within the capsule's
  recommended five-minute cutoff, so it was interrupted.

This qualitatively supports the paper's claim that YS is barrier-prone and can
be drastically slower. It is not a numerical reproduction of the paper's
9.11x, 14x, or 80x speedups because YS did not complete and the input was not a
paper sub-dataset. It also shows why “LM performance” must name the
implementation: on this sample, the optional pure-Python LM path was hundreds
of times slower than the standard log-gamma calls, while C-wrapped LM was
faster.

## 4. How LM behaves in the current project

The current project adapts the same LM C core but adds validation, memory
handling, zero-count support, and a matrix entry point. Unlike the professor's
default script, which first aggregates all sampled rows into one count vector,
LM-IDNet evaluates the likelihood independently for every matrix row and sums
the row results. This matches the likelihood contract documented in
`likelihood-semantics.md`, but it creates a very different benchmark workload.

The fresh benchmark used:

- D-Link Camera 5;
- six pooled fit captures and an 846-by-4 count matrix;
- precision 6 and tolerance `2e-9`;
- 270 development-test windows;
- Python 3.12.14, NumPy 2.5.1, and SciPy 1.18.0;
- five full-fit measurements, with medians reported;
- temporary artifacts only, leaving accepted project artifacts unchanged.

### Fresh LM-IDNet benchmark

| Operation | SciPy | LM | LM / SciPy | Interpretation |
|---|---:|---:|---:|---|
| Full 846-by-4 model fit | 0.183641 s | 0.307252 s | 1.673 | LM is 67% slower |
| One 846-by-4 likelihood | 50.212 us | 114.816 us | 2.287 | LM is 2.29x slower |
| One complete window score | 13.525 us | 12.139 us | 0.898 | LM is about 10% faster |

The likely reason is structural rather than contradictory: SciPy evaluates the
whole matrix with vectorized native operations, while the adapted LM matrix
entry point loops over 846 rows and initializes per-row native state. That
overhead hurts the matrix and full-fit cases but is small for a single window.
The professor's capsule instead reduces a matrix to one aggregate count vector
before the timed likelihood call.

### Numerical and detector agreement

| Comparison | Fresh result |
|---|---:|
| SciPy fit iterations | 1,879 |
| LM fit iterations | 1,882 |
| Maximum relative `alpha` difference | 3.857e-7 |
| Likelihood absolute difference at common `alpha` | 2.675e-7 |
| Threshold absolute difference | 2.886e-6 |
| Median score absolute difference | 3.153e-7 |
| Maximum score absolute difference | 1.092e-5 |
| Anomalies from SciPy / LM | 3 / 3 |
| Decision mismatches across 270 windows | **0** |

This is strong practical agreement at the configured precision. For the
current Camera 5 workload, LM is an accurate substitute for SciPy, but not a
full-fit speed improvement.

The existing committed benchmark artifact records almost the same timings and
exactly the same numerical comparisons under Python 3.11.16 and SciPy 1.17.1,
so the conclusion is stable across these two local software environments.

The focused native and benchmark checks also passed: **9 tests passed**.

## 5. Paper dataset versus project dataset construction

The project uses the same device, dates, 10-minute concept, and four protocol
categories, but it does not reproduce the paper's row counts or split.

| Day | Paper instances | Project observed windows | Paper `psi` | Project daily `psi` |
|---|---:|---:|---:|---:|
| Oct 8 | 196 | 191 | 0.004493 | 0.014719 |
| Oct 9 | 143 | 144 | 0.004342 | 0.005184 |
| Oct 12 | 143 | 144 | 0.003408 | 0.003178 |
| Oct 13 | 78 | 79 | 0.003327 | 0.004805 |
| Oct 14 | 143 | 144 | 0.008205 | 0.004832 |
| Oct 15 | 143 | 144 | 0.004914 | 0.005104 |
| Oct 16 | 143 | 144 | 0.006185 | 0.005472 |
| Oct 19 | 124 | 126 | 0.008430 | 0.005989 |
| Oct 21 | 143 | 144 | 0.004099 | 0.004198 |
| Oct 22 | 143 | 144 | 0.007148 | 0.004676 |

The exact `psi` values therefore should not be expected to match. Different
window boundary/filter rules or source extraction are already visible in the
instance-count differences. Nevertheless, every project daily `psi` is below
0.05, and the pooled project model has `psi = 0.00818188`. The paper's central
qualitative premise—this traffic is overdispersed—is reproduced.

The split is intentionally different too:

- The paper trains separate models on October 8, 9, and 12--16, then forecasts
  October 19, 21, and 22.
- LM-IDNet pools October 8--13, including weekend days October 10 and 11, into
  one fit; uses October 14--15 for threshold calibration; scores October 16 and
  19 as development test; and reserves October 21--22 as final test.

That split is defensible for anomaly-detector development because it separates
fitting, calibration, development testing, and final testing. It is simply not
the paper's forecast protocol.

## 6. The current forecast output is not comparable to the paper

Running the current forecast function prints:

```text
Forecasting: horizon=24h, BrierScore=0.007774, JSD=0.099738
```

Those values must not be put beside Table VI as if they measured the same
thing. The implementation:

1. reads the fitted `alpha`;
2. derives its mean category probabilities;
3. generates 24 synthetic multinomial observations from those same
   probabilities using a fixed total of 100;
4. compares those synthetic observations back to their generating mean;
5. never reads the dataset path passed to the function.

This is a deterministic smoke demonstration of metric plumbing, not held-out
forecast validation. It is circular by construction and cannot demonstrate
next-day forecasting.

There is also a metric aggregation problem: `brier_score` averages across
hours, while `js_divergence` sums relative entropy across the entire 24-by-4
array without dividing by the number of hours. The comment says the metrics are
averaged per hour, but JSD is not. Its scale therefore grows with the horizon
and is not directly comparable with a per-distribution JSD.

## 7. Do the results add up?

### They agree on these points

- Camera 5 traffic is overdispersed under the stated `psi < 0.05` rule.
- LM can evaluate the required likelihood accurately enough to drive Minka
  convergence.
- LM and standard log-gamma methods can produce almost identical fitted models
  on numerically manageable data.
- YS can hit its delta barrier and can be dramatically slower.
- A C implementation changes the performance conclusion substantially compared
  with pure Python.

### They do not measure the same claims

- Paper: LM versus YS modeling and actual held-out traffic forecasting.
- Capsule default: C-LM versus `math.lgamma` and SciPy on random samples from
  generic datasets.
- LM-IDNet benchmark: C-LM versus vectorized SciPy for pooled fitting,
  calibration, and anomaly scoring.
- LM-IDNet forecast scaffold: synthetic self-comparison, not a held-out
  forecast.

Therefore, the current project does **not contradict** the paper when LM is
slower than SciPy for fitting. The paper's main speed claim is LM versus YS,
and its Table V already shows LM slightly slower than standard `lgamma`.

Likewise, the paper's better BS/JSD does not imply that LM-IDNet should find
more anomalies than SciPy. On the current workload the backends make identical
decisions, which is what should be expected when both accurately evaluate the
same likelihood.

## 8. What should be claimed in the project now

A defensible thesis statement based on the evidence is:

> LM-IDNet integrates a hardened C-wrapped Languasco--Migliardi likelihood
> backend. On the D-Link Camera 5 workload it agrees with SciPy to small
> numerical tolerances and produces identical anomaly decisions. It is about
> 10% faster for isolated single-window scoring but about 67% slower for the
> complete fitted workload in the current row-wise matrix adapter. These
> results validate backend correctness; they do not reproduce the forecasting
> comparison in ALM_IoT-CR.

Avoid claiming any of the following from current evidence:

- that the project reproduces the paper's 14x average speedup;
- that LM improves anomaly-detection accuracy;
- that the current Brier/JSD output validates three-day forecasting;
- that the accepted project model currently uses LM.

## 9. Minimum work needed for a true paper comparison

If the goal is to reproduce the paper rather than only build an anomaly
detector, the smallest valid experiment is:

1. Make preprocessing reproduce the paper's ten daily instance counts, or
   document precisely why it cannot.
2. Use the paper's seven training days, four categories, separate daily models,
   10/15/20/25% samples, six-digit precision, and `2e-9` tolerance.
3. Fix and record random seeds, run each setting three times, and preserve both
   central values and variation.
4. Enable a reviewed YS path and record delta-barrier failures rather than
   silently bypassing them.
5. Build the seven-model, 1,000-sample forecast described in the paper.
6. Evaluate against actual October 19, 21, and 22 traffic with clearly defined
   per-day BS and JSD.
7. Keep the anomaly-detector experiment separate: pooled fit, normal-only
   calibration, labeled held-out scoring, and detector metrics answer a
   different research question.

Before that work, the current backend benchmark is already sufficient for its
narrow purpose: LM is integrated correctly and yields the same detector
decisions as SciPy, but it is not presently the faster fitting backend.

