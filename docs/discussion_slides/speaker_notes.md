# LM-IDNet discussion presentation

Presentation date: **20 October 2026**. Present `lm_idnet_discussion.pdf`.
Slides 1–10 are the **12-minute talk**; R1–R7 follow “Thank you” and are for
questions only. The window-duration study is in R5–R6. The visual format reuses
the existing 16:9 burgundy Beamer deck in `docs/slides/`. All numerical
comparisons use vector graphs; there are no slide tables or animated reveals.

The time allocations below total 720 seconds. Practise against the cumulative
times and leave time for pointing at the graphs. The script is a speaking
guide; the clock, rather than a fixed reading speed, determines the pace.
There is no dependency on a completed dissertation document.

## 1 — LM-IDNet · 00:00–00:30 · 30 seconds

Good morning, everyone. I am Fatemeh Mahvari. This work was supervised by
Professor Mauro Migliardi and co-supervised by Professor Alessandro Languasco.

LM-IDNet is a study about an IoT anomaly-detection system using the
Languasco–Migliardi (LM) method. I developed the complete path from packet
captures to detection events, evaluated each device on labelled attacks, and
measured the adaptive periodic-refit runtime in a deployment simulation.

## 2 — Device traffic profiles · 00:30–01:25 · 55 seconds

The starting point is that different devices have different normal traffic.
Here, the bars show the expected protocol proportions in the two fitted models.
Chromecast is dominated by ordinary UDP, while the Samsung camera has a much
larger SSDP component. These are fitted normal means, not attack measurements.

Each device therefore receives its own model. Every ten minutes, we keep just
four mutually exclusive counts: TCP, ordinary UDP, SSDP (separately) and ARP.

The model, following unsupervised learning, learns from a trusted benign
period. This keeps the representation compact and makes an alert
understandable in terms of the traffic that changed.

*Point to the different UDP and SSDP bars; do not read every proportion.*

## 3 — Detection model and conservative adaptation · 01:25–03:00 · 95 seconds

The statistical core is a Dirichlet–multinomial model. It represents both the
normal protocol mixture and the variation we expect between windows. A window
with a low probability under that model provides evidence of an anomaly.

For anomaly detection, composition alone is incomplete. The model conditions
on the total packet count, so the same mixture can appear at a very different
volume. Also, a small ARP change can be hidden by dominant traffic. I therefore
retain two cheap count signals: total packets and ARP packets.

The three signals are scaled from fitting data and calibrated jointly on a
separate benign period. A negative margin from any resulting cutoff triggers
an alert. The event also records which evaluated signals support that decision.

To compute the DM likelihood, the Languasco–Migliardi method is used.
Parameter estimation follows Minka's fixed-point updates.

For adaptation, I use conservative periodic refitting. It admits only
sufficiently safe windows and checks candidate changes before activating them.

## 4 — Datasets and chronological partitions · 03:00–04:05 · 65 seconds

To interpret the results, the data roles must remain separate. Both labelled
devices have earlier normal captures for fitting, a separate normal calibration period, later development traffic and configured final captures.

The arrows show that order. The widths do not represent elapsed time. The
development data supports comparisons, while the final replay uses saved
normal-only starting models and thresholds with fixed settings. Static
detection, threshold adaptation and periodic refitting each start independently from the same pair for their device.

Labels enter only the offline metrics. A window is scored before it can
influence learning.

D-Link supports numerical testing, but it has no attack ground truth here, so
its alerts cannot establish detection accuracy.

## 5 — Final detection results by device · 04:05–06:05 · 120 seconds

These graphs show the final results for static mode, using ten-minute windows.
Each device is evaluated separately with its own fitted model and calibration.

For Chromecast, precision is 57.14 percent, recall is 66.67 percent and F1 is
61.54 percent. Its 432 final windows contain six attack-labelled windows. Four
are detected and two are missed. Three of 426 benign windows produce false
alerts, giving a false-positive rate of about 0.70 percent.

For the Samsung camera, precision is 100 percent, recall is 83.33 percent and
F1 is 90.91 percent. Its 576 final windows contain twelve attack-labelled
windows. Ten are detected and two are missed. None of its 564 benign windows
produces a false alert in this recording.

The conservative adaptive-threshold and periodic-refit modes make the same
decisions for each device because no update is activated in these final runs.
Slide 7 shows those update outcomes.

*Point to the two groups separately. The attack-window denominators are six
for Chromecast and twelve for Samsung; keep conclusions device-specific.*

## 6 — Attack coverage and detection delay · 06:05–07:40 · 95 seconds

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

## 7 — Conservative adaptation results · 07:40–09:10 · 90 seconds

The equal static and adaptive results have a concrete explanation: no
conservative candidate update is activated for either device.

Chromecast has three adaptive-threshold attempts; all exceed the permitted
threshold movement. Its periodic-refit run has three attempts: two exceed the
concentration-change limit and one has too few accepted windows.

Samsung has four adaptive-threshold attempts; all exceed the threshold limit.
Its four periodic-refit attempts include three rejected for concentration
movement and one for too few accepted windows.

A refit also has a matching threshold record; the graph counts each scheduled
attempt once and keeps the devices separate.

The policy scores before learning, keeps original detection references and
activates only a valid complete pair. Those mechanisms preserve the starting
detector in this replay.

This is evidence of the guards' behaviour. It does not establish a useful
conservative learning gain on these traces. Controlled tests exercise
successful activation separately; a benefit under real benign drift still
needs evidence.

## 8 — Edge simulation: adaptive periodic refitting · 09:10–10:25 · 75 seconds

These measurements use adaptive periodic-refit mode with the conservative
policy and the D-Link Camera 5 starting model and threshold. They come from
the recorded raw-DM supplied-window runtime in a local x86-64 Docker simulation
limited to one CPU and 512 MiB of memory. The inputs are an accelerated
synthetic fixture rather than live traffic from that camera.

Across twenty thousand inputs, sixteen thousand windows are scored and four
thousand marked missing are skipped. Sampled process memory stays around
69 MiB. The worst observed processing time is 5.079 seconds, including a
rejected refit. No update is promoted in this trial.

These measurements show observed processing headroom relative to the
ten-minute interval. They do not remove collection delay, prove indefinite
memory behaviour or establish physical gateway performance. The completed
software supports bounded operation, restart and rollback; physical hardware
and ARM measurements remain the next deployment validation step.

## 9 — Achievements, limitations and future work · 10:25–11:45 · 80 seconds

The work delivers three connected achievements.

First, a compact statistical detector: four counts, a normal-only DM model,
native likelihood integration and explainable complementary count signals.

Second, traceable final evidence for each device: static F1 of 61.54 percent
for Chromecast and 90.91 percent for Samsung. Chromecast detects two of three
attack intervals; Samsung detects six of six. Full event and update records
explain the results, including why conservative adaptive decisions match
static decisions in these runs.

Third, operational software: one batch and streaming detection path, guarded
state changes, restart and rollback, and a measured constrained simulation.

The evidence has clear limits. Chromecast has only six final attack windows
and Samsung has twelve. Attacks that preserve the observed count pattern can
remain invisible. Useful conservative adaptation is still unproven on these
traces.

The next steps are broader device and attack coverage, longer evaluation under benign drift, and physical gateway validation. The central outcome is a
complete, reproducible system whose measured tradeoffs can guide that work.

## 10 — Thank you · 11:45–12:00 · 15 seconds

Thank you. I welcome your questions.

*Stop here. Open reference slides only when a question needs them.*

## Reference slides — outside the 12-minute talk

- **R1 — Model and decision:** distinguish complete DM log-probability from the
  fitting kernel. Fit-only scaling precedes joint benign calibration. Event
  severity is a margin, not an attack probability. Original references remain
  active during adaptation; this can also preserve false alerts under drift.
- **R2 — Native Languasco–Migliardi implementation:** fitting retains a
  separate likelihood contribution for every window. The graphs show only
  this method's recorded D-Link timings: 192.974 ms for complete fitting,
  55.457 microseconds for a matrix kernel, and 12.044 microseconds for a
  single-window score. The two graphs use different time units. The record
  uses an 846-by-4 fitting matrix and a 270-window scoring workload, with five
  timing batches and warm-up. These are historical observations on that
  workload, not new performance measurements.
- **R3 — Confusion counts by device:** interval coverage is 2/3 for Chromecast
  and 6/6 for Samsung; window recall is 4/6 and 10/12. A single attack can
  overlap multiple positive windows. Chromecast has TP 4, FP 3, FN 2 and
  TN 423; Samsung has TP 10, FP 0, FN 2 and TN 564. Keep the metrics separate.
- **R4 — Guards:** the 1.25 concentration ratio, 0.05 proportion movement and
  two-original-IQR threshold limit apply to individual updates. They do not
  prove immunity to contamination or bounded cumulative drift. Candidate
  likelihood is evaluated on its fitting buffer rather than an independent
  candidate holdout. Activation is tested separately from useful adaptation.
- **R5 — Window duration and attack coverage:** this separate static
  development study compares 10/20/30/60-minute windows over matched traffic
  coverage within each device. Each duration gets a fresh normal fit and
  calibration under the study's common split. For Chromecast, detected
  attacks are 10/18, 9/18, 10/18 and 11/18; alerts by attack end are 9/18,
  4/18, 3/18 and 1/18. For Samsung, detections are 28/36, 28/36, 35/36 and
  34/36; alerts by attack end are 23/36, 13/36, 9/36 and 5/36. The delivered
  setting remains ten minutes for earlier response and the validated starting
  pairs. These development results are separate from the selected detector's
  final evaluation.
- **R6 — Window duration, F1 and delay:** at 10/20/30/60 minutes, Chromecast
  development F1 is 61.22%, 62.50%, 66.67% and 71.43%; Samsung development F1
  is 78.33%, 84.09%, 91.43% and 85.11%. Longer windows change the label
  granularity as well as detection behaviour. Samsung's study normal split differs from the
  reference split so hourly calibration still meets the 30-observation guard.
  The same adjusted split is used at every duration. Delay is conditional on
  detection. Longer-window adaptive results are a separate concern: at thirty
  minutes Samsung's later-sequence threshold adaptation has 66 false alerts,
  versus zero for static detection; longer-window gains do not establish safe
  adaptation.
- **R7 — Sources:** scientific references are retained from the repository's
  bibliography. The paths below identify the exact project evidence.

## Numerical sources and claim boundaries

- **Slide 2:** [Chromecast model](../../artifacts/final_evaluation/UNSW-Chromecast/model.json)
  and [Samsung model](../../artifacts/final_evaluation/UNSW-Samsung-Camera/model.json).
  Plot values are `100 * alpha[k] / sum(alpha)`, not observed attack proportions.
- **Slides 3 / R1:** [scoring description](../full_project_report/sections/multi_signal_research.tex)
  and [scientific basis](../full_project_report/sections/scientific_basis.tex).
- **Slides 3 / 9:** [conservative adaptation](../full_project_report/sections/conservative_adaptation.tex),
  [streaming runtime](../full_project_report/sections/streaming_runtime.tex),
  and [persistent state](../full_project_report/sections/persistent_state.tex).
- **R2:** [retained D-Link timings](../../reports/D-LinkDayCam5/lm_backend_benchmark.json).
  Only the native method's absolute timings are plotted.
- **Slide 4:** [original dataset roles](../full_project_report/sections/datasets.tex)
  and [final protocol](../../reports/final_evaluation/protocol.json). Chromecast
  final dates are 25–27 October 2018; Samsung final dates are 5–8 June 2018.
  These are configured later partitions with fixed settings for the final
  replay. Earlier retrospective audits exist; do not describe these data as an
  untouched or independently locked external test set.
- **Slides 5–6 / R3:** [final results](../../reports/final_evaluation/results.json),
  [saved final events](../../artifacts/final_evaluation/), and
  [recorded verification](../../reports/final_evaluation/verification.json).
  Delays use the first overlapping alert's window end. The plotted C1–C3 and
  S1–S6 identifiers are chronological within device; their original attack
  names and timestamps are in `interval_results` in the results JSON.
- **Slide 7 / R4:** [Chromecast threshold attempts](../../artifacts/final_evaluation/UNSW-Chromecast/adaptive_threshold_updates.json),
  [Chromecast refit attempts](../../artifacts/final_evaluation/UNSW-Chromecast/periodic_refit_updates.json),
  [Samsung threshold attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/adaptive_threshold_updates.json),
  [Samsung refit attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/periodic_refit_updates.json),
  and [guard definitions](../full_project_report/sections/conservative_adaptation.tex).
- **R5–R6:** [duration study results](../../reports/window_size/results.json)
  and [study protocol and interpretation](../full_project_report/sections/window_size_experiments.tex).
  Only static development rows supply the plotted duration comparisons. Fresh
  study fits and matched coverage make these distinct from the saved reference
  detector's original development or final metrics.
- **Slides 8–9:** [20,000-input simulation](../../reports/edge/simulation-20000.json)
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
- **R7:** [project bibliography](../full_project_report/bibliography.tex).
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
