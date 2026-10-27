# LM-IDNet discussion presentation

Presentation date: **20 October 2026**. Present `lm_idnet_discussion.pdf`.
Slides 1–9 are the main talk within the **12-minute presentation slot**;
R1–R8 follow “Thank you” and are for
questions only. The device profiles are in R2 and the window-duration study is
in R6–R7. The visual format reuses
the existing 16:9 burgundy Beamer deck in `docs/slides/`. All numerical
comparisons use vector graphs; there are no slide tables or animated reveals.

The speaking allocations below total 10 minutes 10 seconds, leaving 110 seconds
for pointing at graphs, pauses and transitions. Practise against the cumulative
times. The script is a speaking
guide; the clock, rather than a fixed reading speed, determines the pace.
There is no dependency on a completed dissertation document.

## 1 — LM-IDNet · 00:00–00:45 · 45 seconds

Good morning, everyone. I am Fatemeh Mahvari. This work was supervised by
Professor Mauro Migliardi and co-supervised by Professor Alessandro Languasco.

Every ten minutes, an IoT device leaves behind four simple numbers: its TCP,
ordinary UDP, SSDP and ARP packet counts. My question was whether those four
numbers could provide a compact and understandable warning of abnormal
behaviour.

On two labelled device traces, LM-IDNet alerts on eight of nine fully observed
attack intervals, seven before the annotated attack ends. The thesis contribution
is the complete reproducible path from packet captures to explainable decisions,
together with validation of the numerical method used for its likelihood.

## 2 — What exists and where this thesis fits · 00:45–01:45 · 60 seconds

IoT anomaly detection is already an active research and commercial field.
Research systems such as N-BaIoT, Kitsune and DÏoT use richer neural, ensemble
or federated representations. Commercial platforms such as Microsoft Defender
for IoT and Palo Alto Device Security add asset discovery, broad telemetry,
threat intelligence and operational response.

Their headline metrics are not directly comparable with this thesis because
their datasets, features and evaluation protocols differ. LM-IDNet occupies a
narrower niche: four explicit counts, a device-specific normal model, an
inspectable decision margin and reproducible state changes. Its value is
compactness and transparency, not a claim to replace a commercial platform.

## 3 — Four counts, three signals, one decision · 01:45–03:00 · 75 seconds

The statistical core is a Dirichlet–multinomial model. A fixed multinomial
assumes the same protocol probabilities in every window. The observed normal
traffic varies much more than that assumption permits: the Pearson dispersion
ratios are 218.96 for Chromecast and 86.84 for Samsung, where a value near one
would support the fixed multinomial. The DM model allows the normal mixture to
vary between windows.

The probability model and its numerical implementation are separate. The DM
defines what is calculated; the Languasco–Migliardi method calculates its
likelihood. Minka's fixed-point method estimates the parameters. I also
acknowledge Professor Sherenaz Al-Haj Baddar who provided scientific support
during this project.

For anomaly detection, composition alone is incomplete. The model conditions
on the total packet count, so the same mixture can appear at a very different
volume, and a small ARP change can be hidden by dominant traffic. LM-IDNet
therefore combines the DM score with total-count and ARP checks. Separate
benign traffic calibrates the joint decision, and conservative refitting checks
candidate changes before activation.

## 4 — Where LM adds numerical value · 03:00–04:25 · 85 seconds

This graph compares LM and SciPy with a 100-decimal-digit reference. At the
current D-Link concentration, both errors are around ten to the minus ten, so
the data are numerically easy for both backends. The advantage appears as the
concentration approaches the multinomial boundary. At ten to the ten, LM's
absolute error is about five hundred times lower; at ten to the fourteen it is
about three hundred and twenty-seven times lower.

Across 1,300 tested development windows, the small numerical differences cause
no binary decision mismatch. This validates the integration but does not mean
that LM increases attack recall: both backends calculate the same statistical
objective.

The runtime comparison also depends on the software phase. Routine live
operation scores one completed window at a time. Here LM takes 12.04
microseconds versus 13.39 for SciPy, about 10.1 percent faster. Initial training
and occasional periodic refits use the matrix and complete-fitting paths, where
SciPy is faster. The live saving is real but only about 1.35 microseconds, so the
ten-minute collection interval, not backend computation, dominates alert delay.

## 5 — Datasets and chronological partitions · 04:25–05:25 · 60 seconds

To interpret the results, the data roles must remain separate. Both labelled
devices have earlier normal captures for fitting, a separate normal calibration period, later development traffic and configured final captures.

The arrows show that order. The widths do not represent elapsed time. The
development data supports comparisons, while the final replay uses saved
normal-only starting models and thresholds with fixed settings. Static
detection and conservative periodic refitting each start independently from
the same pair for their device.

Labels enter only the offline metrics. A window is scored before it can
influence learning.

These traces are useful but imperfect. Chromecast's fitted reciprocal
concentration is 0.0356 and passes the papers' empirical threshold of 0.05;
Samsung's is 0.0575 and does not. That convention is not a formal test of
overdispersion: both training sets show strong extra-multinomial variation.
A deliberately selected low-psi dataset would test LM's numerical advantage
more directly, but better attack detection would still need to be measured.

D-Link supports the numerical tests, but it has no attack ground truth here.

## 6 — Final detection results by device · 05:25–07:10 · 105 seconds

Each graph compares static detection with conservative adaptive periodic
refitting, using ten-minute windows on the final partitions. Grey bars show
static results and burgundy bars show periodic-refit results. The left panel
is Chromecast and the right panel is Samsung; the devices remain separate.

For Chromecast, precision is 57.14 percent, recall is 66.67 percent and F1 is
61.54 percent. Its 432 final windows contain six attack-labelled windows. Four
are detected and two are missed. Three of 426 benign windows produce false
alerts, giving a false-positive rate of about 0.70 percent.

For the Samsung camera, precision is 100 percent, recall is 83.33 percent and
F1 is 90.91 percent. Its 576 final windows contain twelve attack-labelled
windows. Ten are detected and two are missed. None of its 564 benign windows
produces a false alert in this recording.

For each device, the paired bars have the same values. Conservative periodic
refitting makes the same decisions as static detection because no candidate
update is activated in these final runs.

*Point to the two groups separately. Do not read every number twice. The
attack-window denominators are six for Chromecast and twelve for Samsung.*

## 7 — Attack coverage and detection delay · 07:10–08:25 · 75 seconds

Window metrics do not directly tell us how many separate attacks received an
alert. In static mode, Chromecast detects two of its three fully observed
official attack intervals, both by the annotated end. Samsung detects all six
of its intervals, with five alerted by the annotated end.

Each row aligns an attack to its onset. The pale bar shows its actual duration
and the diamond is the first alert, emitted at the close of an overlapping
window. One Samsung alert arrives after its attack has ended. The Chromecast
row marked missed is a low-rate SYN reflection attack.

Among detected attacks, median delay is about 6.9 minutes for Chromecast and
7.5 minutes for Samsung. Those medians exclude the missed attack. The
ten-minute collection interval is part of response time, even when computation
after the window arrives is fast.

## 8 — What this thesis adds and what remains · 08:25–09:55 · 90 seconds

These measurements use adaptive periodic-refit mode with the conservative
policy and the D-Link Camera 5 starting model and threshold. They come from
the recorded raw-DM supplied-window runtime in a local x86-64 Docker simulation
limited to one CPU and 512 MiB of memory. The inputs are an accelerated
synthetic fixture rather than live traffic from that camera.

Across twenty thousand inputs, sixteen thousand windows are scored and four
thousand marked missing are skipped. Sampled process memory stays around
69 MiB. The worst observed processing time is 5.079 seconds, including a
rejected refit. No update is promoted in this trial.

The thesis therefore contributes three connected things: a compact four-count
detector, traceable attack evidence, and operational software with guarded
state, restart and rollback. The native likelihood comparison adds an explicit
numerical correctness boundary rather than treating the score as a black box.

The evidence has clear limits. Chromecast has only six final attack windows
and Samsung has twelve. Attacks that preserve the observed count pattern can
remain invisible. The runtime uses synthetic supplied windows on x86-64, not a
physical gateway or live packet collector. Useful conservative adaptation is
still unproven on these traces.

The central outcome is a compact, reproducible detector whose decisions,
numerical core and state changes can all be inspected.

## 9 — Thank you · 09:55–10:10 · 15 seconds

Thank you. I welcome your questions.

*Stop here. Open reference slides only when a question needs them.*

## Reference slides — outside the 12-minute talk

- **R1 — Model and decision:** distinguish complete DM log-probability from the
  fitting kernel. Fit-only scaling precedes joint benign calibration. Event
  severity is a margin, not an attack probability. Original references remain
  active during adaptation; this can also preserve false alerts under drift.
- **R2 — Device traffic profiles:** the expected protocol shares come from
  the saved normal-only models. They demonstrate why a shared feature
  vocabulary does not imply one shared reference distribution.
- **R3 — Native Languasco–Migliardi implementation:** fitting retains a
  separate likelihood contribution for every window. The graphs show only
  this method's recorded D-Link timings: 192.974 ms for complete fitting,
  55.457 microseconds for a matrix kernel, and 12.044 microseconds for a
  single-window score. The two graphs use different time units. The record
  uses an 846-by-4 fitting matrix and a 270-window scoring workload, with five
  timing batches and warm-up. These are historical observations on that
  workload, not new performance measurements.
- **R4 — Confusion counts by device:** interval coverage is 2/3 for Chromecast
  and 6/6 for Samsung; window recall is 4/6 and 10/12. A single attack can
  overlap multiple positive windows. Chromecast has TP 4, FP 3, FN 2 and
  TN 423; Samsung has TP 10, FP 0, FN 2 and TN 564. Keep the metrics separate.
- **R5 — Guards:** the 1.25 concentration ratio, 0.05 proportion movement and
  two-original-IQR threshold limit apply to individual updates. They do not
  prove immunity to contamination or bounded cumulative drift. Candidate
  likelihood is evaluated on its fitting buffer rather than an independent
  candidate holdout. Activation is tested separately from useful adaptation.
- **R6 — Window duration and attack coverage:** this separate static
  development study compares 10/20/30/60-minute windows over matched traffic
  coverage within each device. Each duration gets a fresh normal fit and
  calibration under the study's common split. For Chromecast, detected
  attacks are 10/18, 9/18, 10/18 and 11/18; alerts by attack end are 9/18,
  4/18, 3/18 and 1/18. For Samsung, detections are 28/36, 28/36, 35/36 and
  34/36; alerts by attack end are 23/36, 13/36, 9/36 and 5/36. The delivered
  setting remains ten minutes for earlier response and the validated starting
  pairs. These development results are separate from the selected detector's
  final evaluation.
- **R7 — Window duration, F1 and delay:** at 10/20/30/60 minutes, Chromecast
  development F1 is 61.22%, 62.50%, 66.67% and 71.43%; Samsung development F1
  is 78.33%, 84.09%, 91.43% and 85.11%. Longer windows change the label
  granularity as well as detection behaviour. Samsung's study normal split differs from the
  reference split so hourly calibration still meets the 30-observation guard.
  The same adjusted split is used at every duration. Delay is conditional on
  detection. Longer-window adaptive results are a separate concern: at thirty
  minutes Samsung's later-sequence threshold adaptation has 66 false alerts,
  versus zero for static detection; longer-window gains do not establish safe
  adaptation.
- **R8 — Sources:** scientific references are retained from the repository's
  bibliography. The paths below identify the exact project evidence.

## Numerical sources and claim boundaries

- **Slide 2:** [N-BaIoT](https://arxiv.org/abs/1805.03409),
  [Kitsune](https://arxiv.org/abs/1802.09089),
  [DÏoT](https://arxiv.org/abs/1804.07474),
  [Microsoft Defender for IoT](https://learn.microsoft.com/en-us/azure/defender-for-iot/organizations/overview),
  and [Palo Alto Device Security](https://docs.paloaltonetworks.com/iot/getting-started/iot-security-solution)
  supply representative architectural context, not common-benchmark results.
- **R2:** [Chromecast model](../../artifacts/final_evaluation/UNSW-Chromecast/model.json)
  and [Samsung model](../../artifacts/final_evaluation/UNSW-Samsung-Camera/model.json).
  Plot values are `100 * alpha[k] / sum(alpha)`, not observed attack proportions.
- **Slides 3 / R1:** [scoring description](../full_project_report/sections/multi_signal_research.tex)
  and [scientific basis](../full_project_report/sections/scientific_basis.tex).
  Pearson dispersion values come from
  [the overdispersion analysis](../understanding_overdispersion.tex).
- **Slides 3 / 8:** [conservative adaptation](../full_project_report/sections/conservative_adaptation.tex),
  [streaming runtime](../full_project_report/sections/streaming_runtime.tex),
  and [persistent state](../full_project_report/sections/persistent_state.tex).
- **Slide 4 / R3:** [retained D-Link timings](../../reports/D-LinkDayCam5/lm_backend_benchmark.json),
  [three-dataset backend checks](../../reports/detection_backend_checks.json),
  and the [high-precision probe](../lm_professor_architecture_experiment_report.md).
  The stress curve changes concentration while retaining current count totals
  and category proportions; it is not the current D-Link fit.
- **Slide 5:** [original dataset roles](../full_project_report/sections/datasets.tex)
  and [final protocol](../../reports/final_evaluation/protocol.json). Chromecast
  final dates are 25–27 October 2018; Samsung final dates are 5–8 June 2018.
  These are configured later partitions with fixed settings for the final
  replay. Earlier retrospective audits exist; do not describe these data as an
  untouched or independently locked external test set.
- **Slides 6–7 / R4:** [final results](../../reports/final_evaluation/results.json),
  [saved final events](../../artifacts/final_evaluation/), and
  [recorded verification](../../reports/final_evaluation/verification.json).
  Delays use the first overlapping alert's window end. The plotted C1–C3 and
  S1–S6 identifiers are chronological within device; their original attack
  names and timestamps are in `interval_results` in the results JSON.
- **Slide 6 / R5:** [Chromecast refit attempts](../../artifacts/final_evaluation/UNSW-Chromecast/periodic_refit_updates.json),
  [Samsung refit attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/periodic_refit_updates.json),
  and [guard definitions](../full_project_report/sections/conservative_adaptation.tex).
- **R6–R7:** [duration study results](../../reports/window_size/results.json)
  and [study protocol and interpretation](../full_project_report/sections/window_size_experiments.tex).
  Only static development rows supply the plotted duration comparisons. Fresh
  study fits and matched coverage make these distinct from the saved reference
  detector's original development or final metrics.
- **Slide 8:** [20,000-input simulation](../../reports/edge/simulation-20000.json)
  and [measurement scope](../../reports/edge/simulation.md). The plotted `/proc`
  sampled maximum is 68.92 MiB; the separately recorded `getrusage` process
  maximum is 68.62 MiB. The measured simulation predates the final evaluation
  and establishes resource behaviour for its recorded runtime and synthetic
  workload, not a performance measurement of every final-method configuration.
  Its configuration explicitly records `periodic_refit`, the `conservative`
  policy, the `D-LinkDayCam5` reference and raw DM scoring; this is the earlier
  recorded runtime, rather than a resource measurement of the later joint
  composition/volume/ARP detector configuration. The input windows are
  synthetic and are not a real camera capture replay.
- **R8:** [project bibliography](../full_project_report/bibliography.tex).
  No literature or performance results were newly measured for these slides.

## Rebuild

Run from the repository root, using the existing Docker LaTeX setup:

```sh
python docs/discussion_slides/prepare_evidence.py
./scripts/latex.sh docs/discussion_slides/lm_idnet_discussion.tex
cp out/lm_idnet_discussion.pdf docs/discussion_slides/lm_idnet_discussion.pdf
```

`prepare_evidence.py` uses only Python's standard library. It regenerates
`evidence.tex` from retained JSON and asserts the final counts, agreement
between modes, update totals and common attack coverage. The TeX source
reuses `../slides/anomaly_detector.png`; compile from this repository.

Compilation and slide inspection validate the presentation. They do not rerun
the detector's experiments or establish a measured speaking duration.
