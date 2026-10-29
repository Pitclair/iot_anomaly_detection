# LM-IDNet discussion presentation

Present `lm_idnet_discussion.pdf`. Slides 1–10 are the main talk within the **12-minute presentation slot**; R1–R8 follow “Thank you” and are for questions only. The device profiles are in R2 and the window-duration study is in R6–R7. The visual format reuses the existing 16:9 burgundy Beamer deck in `docs/slides/`. All numerical comparisons use vector graphs; there are no slide tables or animated reveals.

The speaking allocations below total 10 minutes 40 seconds, leaving 80 seconds for pointing at graphs, pauses and transitions. Practise against the cumulative times. The script is a speaking guide; the clock, rather than a fixed reading speed, determines the pace. There is no dependency on a completed dissertation document.

## 1 — LM-IDNet · 00:00–00:45 · 45 seconds

Good morning, everyone. I am Fatemeh Mahvari and i am going to present the work that is done for my master's thesis. This work was supervised by Professor Migliardi and co-supervised by Professor Languasco, with special thanks to Professor Al-haj Baddar for her scientific support.

..
Every ten minutes, an IoT device leaves behind four simple numbers: its TCP, ordinary UDP, SSDP and ARP packet counts. This thesis studies how those counts can support a device-speccific and understandable warning of abnormal traffic behaviour.

to give you an overview, On two labelled device traces, LM-IDNet alerts on eight of nine fully observed attack intervals, seven before the annotated attack ends. I will present the statistical model, the numerical comparison behind its likelihood, and the measured detection and runtime results.

## 2 — Context: IoT anomaly-detection approaches · 00:45–01:45 · 60 seconds

IoT anomaly detection is already an active research and commercial field. there are Research systems such as N-BaIoT, Kitsune and DÏoT that use richer neural, ensemble or federated representations. Commercial platforms such as Microsoft Defender for IoT and Palo Alto Device Security add asset discovery, broad telemetry, threat intelligence and operational response.

Their headline metrics are not directly comparable with this study because their datasets, features and evaluation protocols differ.

what LM-IDNet does is that instead it proposes a device-specific detector built from four explicit counts and a Dirichlet–multinomial model. It combines composition, total-volume and ARP signals, then calibrates an inspectable decision margin. Conservative periodic refitting can propose a new model, but guarded validation decides whether it is allowed to replace the active state.

## 3 — Four counts, three signals, one decision · 01:45–03:00 · 75 seconds

the modeling works such that Each ten-minute window becomes four protocol counts. to detect abnormal behavior, The three branches score protocol composition, total traffic volume and ARP activity. Separate benign traffic calibrates the three margins, and crossing any cutoff produces an alert with evidence about which signal changed.

the main part, protocol composition, is done with Dirichlet–multinomial model using Minka's fixed-point method estimates the parameters

The Languasco–Migliardi algorithm evaluates the likelihood through stable log-gamma differences. Direct subtraction of large ordinary log-gamma values can lose precision near the multinomial limit. Concentration describes how tightly traffic stays around its average protocol mix, and As concentration grows model approaches the multinomial limit. LM is used to control that numerical cancellation.


## 4 — Datasets and chronological partitions · 03:00–04:00 · 60 seconds

To interpret the results, the data roles must remain separate. Both labelled devices have earlier normal captures for fitting, a separate normal calibration period, later development traffic and configured final captures.

The arrows show that order. The widths do not represent elapsed time. The development data supports comparisons, while the final replay uses saved normal-only starting models and thresholds with fixed settings. Static detection and conservative periodic refitting each start independently from the same pair for their device.

Labels enter only the offline metrics. A window is scored before it can influence learning.

The reference baseline is the 2025 ICNC ALM IoT study by Al-Haj Baddar, Languasco and Migliardi. It uses the empirical convention psi below 0.05; the same convention also appears in their later TPAMI paper. Chromecast's fitted value is 0.0356 and meets that convention; Samsung's is 0.0575 and does not. This convention is reported for comparison and is not a formal overdispersion test. Both training sets still show strong extra-multinomial variation through their Pearson dispersion diagnostics.

A deliberately selected low-psi dataset would test LM's numerical behaviour more directly.

there is also the D-Link dataset that supports the numerical tests, but it has no attack ground truth here.

## 5 — Numerical comparison: LM and SciPy · 04:00–05:25 · 85 seconds

This graph compares LM and SciPy with a 100-decimal-digit reference. At the current D-Link concentration, both errors are around ten to the minus ten, so the data are numerically easy for both backends. The advantage appears as the concentration approaches the multinomial boundary. At ten to the ten, LM's absolute error is about five hundred times lower; at ten to the fourteen it is about three hundred and twenty-seven times lower.

The error reduction is the main result of this comparison. Across the current 1,300 development windows there are zero binary decision differences, which is supporting compatibility evidence rather than the central claim.

The runtime comparison also depends on the software phase. Routine live operation scores one completed window at a time. Here LM runs about 10.1 percent faster than scipy. Initial training and occasional periodic refits take 193 milliseconds with LM versus 181 with SciPy, so the LM complete fit is 6.4 percent slower on this workload. The live saving is only about 1.35 microseconds, so the ten-minute collection interval, not backend computation, dominates alert delay.
## 6 — Final detection results by device · 05:25–07:10 · 105 seconds

Each graph compares static detection with conservative adaptive periodic refitting, using ten-minute windows on the final partitions. Grey bars show static results and burgundy bars show periodic-refit results. The left panel is Chromecast and the right panel is Samsung; the devices remain separate.

For Chromecast, precision is 57.14 percent, recall is 66.67 percent and F1 is 61.54 percent. Its 432 final windows contain six attack-labelled windows. Four are detected and two are missed. Three of 426 benign windows produce false alerts, giving a false-positive rate of about 0.70 percent.

For the Samsung camera, precision is 100 percent, recall is 83.33 percent and F1 is 90.91 percent. Its 576 final windows contain twelve attack-labelled windows. Ten are detected and two are missed. None of its 564 benign windows produces a false alert in this recording.

For each device, the paired bars have the same values. Conservative periodic refitting makes the same decisions as static detection because no candidate update is activated in these final runs.

*Point to the two groups separately. Do not read every number twice. The attack-window denominators are six for Chromecast and twelve for Samsung.*

## 7 — Attack coverage and detection delay · 07:10–08:25 · 75 seconds

Window metrics do not directly tell us how many separate attacks received an alert. In static mode, Chromecast detects two of its three fully observed official attack intervals, both by the annotated end. Samsung detects all six of its intervals, with five alerted by the annotated end.

Each row aligns an attack to its onset. The pale bar shows its actual duration and the diamond is the first alert, emitted at the close of an overlapping window. One Samsung alert arrives after its attack has ended. The Chromecast row marked missed is a low-rate SYN reflection attack.

Among detected attacks, median delay is about 6.9 minutes for Chromecast and 7.5 minutes for Samsung. Those medians exclude the missed attack. The ten-minute collection interval is part of response time, even when computation after the window arrives is fast.

## 8 — Edge architecture and simulation · 08:25–09:30 · 65 seconds

The batch and streaming paths use the same detector. Each validated window is scored before any learning step, and its event is emitted immediately. Only high-confidence evidence enters the bounded buffer. A candidate model must pass fitting and movement guards before its complete state is saved and activated, so any accepted change can affect only later windows and supports rollback.

The recorded edge trial uses conservative periodic-refit mode in an x86-64 Docker simulation limited to one CPU and 512 MiB. Across twenty thousand synthetic inputs, sixteen thousand windows are scored and four thousand marked missing are skipped. Sampled memory remains near 69 MiB. The worst observed window takes 5.079 seconds, including a rejected refit. No candidate update is promoted in this trial.

These are constrained-runtime measurements, not physical-gateway or live packet-capture results.

## 9 — Summary: results and limitations · 09:30–10:25 · 55 seconds

The measured results show that four protocol counts can support three inspectable signals. Eight of nine attack intervals receive an alert, seven by the annotated end. The numerical probe shows substantially lower LM absolute error in the high-concentration stress regime, and the streaming trial remains near 69 MiB while preserving guarded state and rollback.

The scope remains limited. There are only two labelled devices and eighteen final attack windows. Attacks that preserve the observed count pattern can be missed. The edge workload is synthetic x86-64 rather than a physical gateway, and useful conservative adaptation has not yet been demonstrated on these traces. The conclusion is therefore about this prototype and measured setting, not universal detector superiority.

## 10 — Thank you · 10:25–10:40 · 15 seconds

Thank you. I welcome your questions.

*Stop here. Open reference slides only when a question needs them.*

## Reference slides — outside the 12-minute talk

- **R1 — Model and decision:** distinguish complete DM log-probability from the fitting kernel. Fit-only scaling precedes joint benign calibration. Event severity is a margin, not an attack probability. Original references remain active during adaptation; this can also preserve false alerts under drift.
- **R2 — Device traffic profiles:** the expected protocol shares come from the saved normal-only models. They demonstrate why a shared feature vocabulary does not imply one shared reference distribution.
- **R3 — Native Languasco–Migliardi implementation:** fitting retains a separate likelihood contribution for every window. The graphs show only this method's recorded D-Link timings: 192.974 ms for complete fitting, 55.457 microseconds for a matrix kernel, and 12.044 microseconds for a single-window score. The two graphs use different time units. The record uses an 846-by-4 fitting matrix and a 270-window scoring workload, with five timing batches and warm-up. These are historical observations on that workload, not new performance measurements.
- **R4 — Confusion counts by device:** interval coverage is 2/3 for Chromecast and 6/6 for Samsung; window recall is 4/6 and 10/12. A single attack can overlap multiple positive windows. Chromecast has TP 4, FP 3, FN 2 and TN 423; Samsung has TP 10, FP 0, FN 2 and TN 564. Keep the metrics separate.
- **R5 — Guards:** the 1.25 concentration ratio, 0.05 proportion movement and two-original-IQR threshold limit apply to individual updates. They do not prove immunity to contamination or bounded cumulative drift. Candidate likelihood is evaluated on its fitting buffer rather than an independent candidate holdout. Activation is tested separately from useful adaptation.
- **R6 — Window duration and attack coverage:** this separate static development study compares 10/20/30/60-minute windows over matched traffic coverage within each device. Each duration gets a fresh normal fit and calibration under the study's common split. For Chromecast, detected attacks are 10/18, 9/18, 10/18 and 11/18; alerts by attack end are 9/18, 4/18, 3/18 and 1/18. For Samsung, detections are 28/36, 28/36, 35/36 and 34/36; alerts by attack end are 23/36, 13/36, 9/36 and 5/36. The delivered setting remains ten minutes for earlier response and the validated starting pairs. These development results are separate from the selected detector's final evaluation.
- **R7 — Window duration, F1 and delay:** at 10/20/30/60 minutes, Chromecast development F1 is 61.22%, 62.50%, 66.67% and 71.43%; Samsung development F1 is 78.33%, 84.09%, 91.43% and 85.11%. Longer windows change the label granularity as well as detection behaviour. Samsung's study normal split differs from the reference split so hourly calibration still meets the 30-observation guard. The same adjusted split is used at every duration. Delay is conditional on detection. Longer-window adaptive results are a separate concern: at thirty minutes Samsung's later-sequence threshold adaptation has 66 false alerts, versus zero for static detection; longer-window gains do not establish safe adaptation.
- **R8 — Sources:** scientific references are retained from the repository's bibliography. The slide links the public project repository containing the source, protocols, results and reproducibility records.

## Numerical sources and claim boundaries

- **Slide 2:** [N-BaIoT](https://arxiv.org/abs/1805.03409), [Kitsune](https://arxiv.org/abs/1802.09089), [DÏoT](https://arxiv.org/abs/1804.07474), [Microsoft Defender for IoT](https://learn.microsoft.com/en-us/azure/defender-for-iot/organizations/overview), and [Palo Alto Device Security](https://docs.paloaltonetworks.com/iot/getting-started/iot-security-solution) supply representative architectural context, not common-benchmark results.
- **R2:** [Chromecast model](../../artifacts/final_evaluation/UNSW-Chromecast/model.json) and [Samsung model](../../artifacts/final_evaluation/UNSW-Samsung-Camera/model.json). Plot values are `100 * alpha[k] / sum(alpha)`, not observed attack proportions.
- **Slides 3 / R1:** [scoring description](../full_project_report/sections/multi_signal_research.tex) and [scientific basis](../full_project_report/sections/scientific_basis.tex). Pearson dispersion values come from [the overdispersion analysis](../understanding_overdispersion.tex).
- **Slides 3 / 8:** [conservative adaptation](../full_project_report/sections/conservative_adaptation.tex), [streaming runtime](../full_project_report/sections/streaming_runtime.tex), and [persistent state](../full_project_report/sections/persistent_state.tex).
- **Slide 4:** [original dataset roles](../full_project_report/sections/datasets.tex) and [final protocol](../../reports/final_evaluation/protocol.json). Chromecast final dates are 25–27 October 2018; Samsung final dates are 5–8 June 2018. These are configured later partitions with fixed settings for the final replay. Earlier retrospective audits exist; do not describe these data as an untouched or independently locked external test set.
- **Slide 5 / R3:** [retained D-Link timings](../../reports/D-LinkDayCam5/lm_backend_benchmark.json), [three-dataset backend checks](../../reports/detection_backend_checks.json), and the [high-precision probe](../lm_professor_architecture_experiment_report.md). The stress curve changes concentration while retaining current count totals and category proportions; it is not the current D-Link fit.
- **Slides 6–7 / R4:** [final results](../../reports/final_evaluation/results.json), [saved final events](../../artifacts/final_evaluation/), and [recorded verification](../../reports/final_evaluation/verification.json). Delays use the first overlapping alert's window end. The plotted C1–C3 and S1–S6 identifiers are chronological within device; their original attack names and timestamps are in `interval_results` in the results JSON.
- **Slide 6 / R5:** [Chromecast refit attempts](../../artifacts/final_evaluation/UNSW-Chromecast/periodic_refit_updates.json), [Samsung refit attempts](../../artifacts/final_evaluation/UNSW-Samsung-Camera/periodic_refit_updates.json), and [guard definitions](../full_project_report/sections/conservative_adaptation.tex).
- **R6–R7:** [duration study results](../../reports/window_size/results.json) and [study protocol and interpretation](../full_project_report/sections/window_size_experiments.tex). Only static development rows supply the plotted duration comparisons. Fresh study fits and matched coverage make these distinct from the saved reference detector's original development or final metrics.
- **Slide 8:** [20,000-input simulation](../../reports/edge/simulation-20000.json) and [measurement scope](../../reports/edge/simulation.md). The plotted `/proc` sampled maximum is 68.92 MiB; the separately recorded `getrusage` process maximum is 68.62 MiB. The measured simulation predates the final evaluation and establishes resource behaviour for its recorded runtime and synthetic workload, not a performance measurement of every final-method configuration. Its configuration explicitly records `periodic_refit`, the `conservative` policy, the `D-LinkDayCam5` reference and raw DM scoring; this is the earlier recorded runtime, rather than a resource measurement of the later joint composition/volume/ARP detector configuration. The input windows are synthetic and are not a real camera capture replay.
- **R8:** [project bibliography](../full_project_report/bibliography.tex) and [public repository](https://github.com/Pitclair/iot_anomaly_detection). No literature or performance results were newly measured for these slides.

## Rebuild

Run from the repository root, using the existing Docker LaTeX setup:

```sh
python docs/discussion_slides/prepare_evidence.py
./scripts/latex.sh docs/discussion_slides/lm_idnet_discussion.tex
cp out/lm_idnet_discussion.pdf docs/discussion_slides/lm_idnet_discussion.pdf
```

`prepare_evidence.py` uses only Python's standard library. It regenerates `evidence.tex` from retained JSON and asserts the final counts, agreement between modes, update totals and common attack coverage. The TeX source reuses `../slides/anomaly_detector.png`; compile from this repository.

Compilation and slide inspection validate the presentation. They do not rerun the detector's experiments or establish a measured speaking duration.
