# LM-IDNet discussion presentation

Present `lm_idnet_discussion.pdf`. Slides 1–12 are the **12-minute talk**;
R1–R6 follow “Thank you” and are for questions only. The visual format reuses
the existing 16:9 burgundy Beamer deck in `docs/slides/`. All numerical
comparisons use vector graphs; there are no slide tables or animated reveals.

The time allocations below total 720 seconds. Practise against the cumulative
times and leave time for pointing at the graphs. The script is a speaking
guide; the clock, rather than a fixed reading speed, determines the pace.
There is no dependency on a completed dissertation document.

## 1 — LM-IDNet · 00:00–00:30 · 30 seconds

Good morning. I am Fatemeh Mahvari, and this work was supervised by Professor
Mauro Migliardi.

LM-IDNet asks whether a small model of normal IoT traffic can provide useful,
explainable anomaly detection. I developed the complete path from packet
captures to detection events, evaluated it on labelled attacks, and measured
the software in a constrained deployment simulation.

I will focus on what the system delivers and what the results establish.

## 2 — Device profiles · 00:30–01:25 · 55 seconds

The starting point is that different devices have different normal traffic.
Here, the bars show the expected protocol proportions in the two fitted models.
Chromecast is dominated by ordinary UDP, while the Samsung camera has a much
larger SSDP component. These are fitted normal means, not attack measurements.

Each device therefore receives its own model. Every ten minutes, we keep just
four mutually exclusive counts: TCP, ordinary UDP, SSDP and ARP. SSDP is removed
from ordinary UDP so a packet is not counted twice.

The model learns from a trusted benign period without attack examples or
signatures. This keeps the representation compact and makes an alert
understandable in terms of the traffic that changed.

*Point to the different UDP and SSDP bars; do not read every proportion.*

## 3 — Method · 01:25–02:40 · 75 seconds

The statistical core is a Dirichlet–multinomial model. It represents both the
normal protocol mixture and the variation we expect between windows. A window
with a low probability under that model provides evidence of an anomaly.

Composition alone is incomplete. The model conditions on the total packet
count, so the same mixture can appear at a very different volume. Also, a small
ARP change can be hidden by dominant traffic. I therefore retain two cheap
count signals: total packets and ARP packets, both already available in the
same four-count vector.

The three signals are scaled from fitting data and calibrated jointly on a
separate benign period. A negative margin from any resulting cutoff triggers
an alert. The event records which evaluated signals support that decision.

The native LM backend computes the DM likelihood. It supplies the numerical
calculation; parameter estimation still follows Minka's fixed-point updates.

## 4 — Delivered system · 02:40–03:30 · 50 seconds

My contribution also covers the engineering needed to make the method usable.
The same detector supports capture-based batch experiments and a streaming
command that receives completed count windows. Events retain their decisions,
feature margins and source fingerprints.

The native integration preserves a separate likelihood contribution for every
window. In the retained D-Link comparison, LM and SciPy disagree on none of
270 decisions, with a maximum absolute score difference of about one times
ten to the minus five.

Accepted state is stored as a complete model and threshold pair. The runtime
supports guarded activation, restoration after restart and rollback.

*Do not claim that LM always runs faster; the timing breakdown is R2.*

## 5 — Evaluation protocol · 03:30–04:35 · 65 seconds

To interpret the results, the data roles must remain separate. Both labelled
devices have earlier normal captures for fitting, a separate normal calibration
period, later development traffic and configured final captures.

The arrows show that order. The widths do not represent elapsed time. The
development data supports comparisons, while the final replay uses saved
normal-only starting models and thresholds with fixed settings. Static
detection, threshold adaptation and periodic refitting each start independently
from the same pair for their device.

During that replay, labels enter only the offline metrics. A window is scored
before it can influence learning. All 1,008 final windows are observed and
matched to labels; none is silently removed.

D-Link supports numerical testing, but it has no attack ground truth here, so
its alerts cannot establish detection accuracy.

## 6 — Final results · 04:35–06:00 · 85 seconds

The main result is 80 percent pooled F1 across 1,008 final windows. Pooled
precision is 82.35 percent and recall is 77.78 percent. These values come from
summing confusion counts across the devices, rather than averaging their F1.

The device results differ. Chromecast reaches 61.54 percent F1, with four true
positive windows, two misses and three false alerts. Samsung reaches 90.91
percent F1, with ten true positives, two misses and no false alerts in this
final sequence. The zero false-alert count applies to this recording.

Across both devices, 14 of 18 attack-labelled windows are flagged. Three of
990 benign windows are flagged, giving a measured false-positive rate of
0.30 percent. The one-percent calibration quantile is a normal-reference
setting, not a guarantee about future false positives.

All three selected modes make these same decisions. I will explain why after
looking at attack coverage and response time.

*Point to pooled F1, then the two device F1 bars. Emphasise the denominator of
only 18 attack windows; avoid presenting this as broad cross-device validation.*

## 7 — Incidents and response time · 06:00–07:05 · 65 seconds

Window metrics do not directly tell us how many separate attacks received an
alert. The final recordings contain nine fully observed official attack
intervals. Eight receive an alert; seven receive one by the annotated end.

Each row aligns an attack to its onset. The pale bar shows its actual duration
and the diamond is the first alert, emitted at the close of an overlapping
window. One Samsung alert arrives after its attack has ended. The Chromecast
row marked missed is a low-rate SYN reflection attack.

Among detected attacks, median delay is about 6.9 minutes for Chromecast and
7.5 minutes for Samsung. Those medians exclude the missed attack. The
ten-minute collection interval is part of response time, even when computation
after the window arrives is fast.

## 8 — Adaptation · 07:05–08:05 · 60 seconds

The equal static and adaptive results have a concrete explanation: none of
14 scheduled final update attempts is activated.

Seven threshold candidates exceed the permitted threshold movement. Five
refit candidates exceed the permitted concentration change, and two attempts
have too few accepted windows. A refit also has a matching threshold record;
the graph counts each scheduled attempt once.

The policy scores before learning, keeps original detection references and
activates only a valid complete pair. Those mechanisms preserve the starting
detector in this replay.

This is evidence of the guards' behaviour. It does not establish a useful
conservative learning gain on these traces. Controlled tests exercise
successful activation separately; a benefit under real benign drift still
needs evidence.

## 9 — Window duration · 08:05–09:20 · 75 seconds

The practical design choice is how long to aggregate traffic. A separate
development study compares ten, twenty, thirty and sixty minutes over matched
traffic coverage and the same 54 fully observed attacks. Each duration gets a
fresh normal fit and calibration under a common split for the study.

The burgundy line counts attacks eventually detected; the dark dashed line
counts alerts arriving by attack end. At ten minutes, 38 attacks are detected
and 32 receive an alert by the end. At thirty minutes, coverage improves to
45 attacks, but only twelve receive an alert by the end. At one hour, only six
are alerted in time.

Thirty minutes has the strongest pooled static window F1 in this study, but
coarser windows also change the labels' meaning. I retain ten minutes for
earlier response and the validated starting pairs. Thirty-minute static
operation is a candidate when the use case accepts longer delay.

*Keep this development study separate from the selected detector's final result.*

## 10 — Resource measurements · 09:20–10:25 · 65 seconds

I measured the supplied-window runtime in a local x86-64 Docker simulation
limited to one CPU and 512 MiB of memory. This uses an accelerated synthetic
fixture rather than a live packet collector.

Across twenty thousand inputs, sixteen thousand windows are scored and four
thousand marked missing are skipped. Sampled process memory stays around
69 MiB. The worst observed processing time is 5.079 seconds, including a
rejected refit. No update is promoted in this trial.

These measurements show observed processing headroom relative to the
ten-minute interval. They do not remove collection delay, prove indefinite
memory behaviour or establish physical gateway performance. The completed
software supports bounded operation, restart and rollback; physical hardware
and ARM measurements remain the next deployment validation step.

## 11 — Achievements and limits · 10:25–11:45 · 80 seconds

The work delivers three connected achievements.

First, a compact statistical detector: four counts, a normal-only DM model,
native likelihood integration and explainable complementary count signals.

Second, traceable final evidence: 80 percent pooled F1 over 1,008 windows,
eight of nine attacks detected, with full event and update records. The
evaluation also exposes device differences and the response-time cost of
aggregation.

Third, operational software: one batch and streaming detection path, guarded
state changes, restart and rollback, and a measured constrained simulation.

The evidence has clear limits. There are only two labelled devices and
18 final attack windows. Attacks that preserve the observed count pattern can
remain invisible. Useful conservative adaptation is still unproven on these
traces.

The next steps are broader device and attack coverage, longer evaluation under
benign drift, and physical gateway validation. The central outcome is a
complete, reproducible system whose measured tradeoffs can guide that work.

## 12 — Thank you · 11:45–12:00 · 15 seconds

Thank you. LM-IDNet turns a small normal-traffic model into a working detector
with explainable decisions and measurable results. I welcome your questions.

*Stop here. Open reference slides only when a question needs them.*

## Reference slides — outside the 12-minute talk

- **R1 — Model and decision:** distinguish complete DM log-probability from the
  fitting kernel. Fit-only scaling precedes joint benign calibration. Event
  severity is a margin, not an attack probability. Original references remain
  active during adaptation; this can also preserve false alerts under drift.
- **R2 — Native backend:** ratios above one mean LM takes longer. Retained
  medians are 192.974 versus 181.286 ms for complete fitting; 55.457 versus
  49.961 microseconds for a matrix kernel; and 12.044 versus 13.394
  microseconds for a single-window score. These are historical observations on
  one workload, not a fresh benchmark or a detection improvement.
- **R3 — Confusion counts:** pooled F1 uses summed counts. Interval coverage is
  2/3 for Chromecast and 6/6 for Samsung; window recall is 4/6 and 10/12. A
  single attack can overlap multiple positive windows.
- **R4 — Guards:** the 1.25 concentration ratio, 0.05 proportion movement and
  two-original-IQR threshold limit apply to individual updates. They do not
  prove immunity to contamination or bounded cumulative drift. Candidate
  likelihood is evaluated on its fitting buffer rather than an independent
  candidate holdout. Activation is tested separately from useful adaptation.
- **R5 — Duration:** pooled development F1 is 73.37%, 78.33%, 85.11% and 81.97%
  at 10/20/30/60 minutes. Samsung's study normal split differs from the
  reference split so hourly calibration still meets the 30-observation guard.
  The same adjusted split is used at every duration. Delay is conditional on
  detection. Longer-window adaptive results are a separate concern: at thirty
  minutes Samsung's later-sequence threshold adaptation has 66 false alerts,
  versus zero for static detection; longer-window gains do not establish safe
  adaptation.
- **R6 — Sources:** scientific references are retained from the repository's
  bibliography. The paths below identify the exact project evidence.

## Numerical sources and claim boundaries

- **Slide 2:** [Chromecast model](../../artifacts/final_evaluation/UNSW-Chromecast/model.json)
  and [Samsung model](../../artifacts/final_evaluation/UNSW-Samsung-Camera/model.json).
  Plot values are `100 * alpha[k] / sum(alpha)`, not observed attack proportions.
- **Slides 3 / R1:** [scoring description](../full_project_report/sections/multi_signal_research.tex)
  and [scientific basis](../full_project_report/sections/scientific_basis.tex).
- **Slides 4 / R2:** [retained D-Link benchmark](../../reports/D-LinkDayCam5/lm_backend_benchmark.json),
  [streaming runtime](../full_project_report/sections/streaming_runtime.tex),
  and [persistent state](../full_project_report/sections/persistent_state.tex).
- **Slide 5:** [original dataset roles](../full_project_report/sections/datasets.tex)
  and [final protocol](../../reports/final_evaluation/protocol.json). Chromecast
  final dates are 25–27 October 2018; Samsung final dates are 5–8 June 2018.
  These are configured later partitions with fixed settings for the final
  replay. Earlier retrospective audits exist; do not describe these data as an
  untouched or independently locked external test set.
- **Slides 6–7 / R3:** [final results](../../reports/final_evaluation/results.json),
  [saved final events](../../artifacts/final_evaluation/), and
  [recorded verification](../../reports/final_evaluation/verification.json).
  Delays use the first overlapping alert's window end. The plotted C1–C3 and
  S1–S6 identifiers are chronological within device; their original attack
  names and timestamps are in `interval_results` in the results JSON.
- **Slide 8 / R4:** [Chromecast threshold attempts](../../artifacts/final_evaluation/UNSW-Chromecast/adaptive_threshold_updates.json),
  [Chromecast refit attempts](../../artifacts/final_evaluation/UNSW-Chromecast/periodic_refit_updates.json),
  [Samsung threshold attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/adaptive_threshold_updates.json),
  [Samsung refit attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/periodic_refit_updates.json),
  and [guard definitions](../full_project_report/sections/conservative_adaptation.tex).
- **Slides 9 / R5:** [duration study results](../../reports/window_size/results.json)
  and [study protocol and interpretation](../full_project_report/sections/window_size_experiments.tex).
  Only static development rows supply the plotted duration comparisons. Fresh
  study fits and matched coverage make these distinct from the saved reference
  detector's original development or final metrics.
- **Slides 10–11:** [20,000-input simulation](../../reports/edge/simulation-20000.json)
  and [measurement scope](../../reports/edge/simulation.md). The plotted `/proc`
  sampled maximum is 68.92 MiB; the separately recorded `getrusage` process
  maximum is 68.62 MiB. The measured simulation predates the final evaluation
  and establishes resource behaviour for its recorded runtime and synthetic
  workload, not a performance measurement of every final-method configuration.
- **R6:** [project bibliography](../full_project_report/bibliography.tex).
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
