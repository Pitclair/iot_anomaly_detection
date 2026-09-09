# LM-IDNet Project Presentation — Speaker Notes

These notes follow the current slide order in
`lm_idnet_project_presentation.tex`. Slides 1–36 form the main presentation.
Slides 37–42 are backup material and should be opened only when a question calls
for the additional detail.

The notes are written as a speaking guide rather than text that must be read
word for word. The transitions at the end of each section are there to make the
presentation feel like one argument instead of a sequence of unrelated slides.

## Slide 1 — LM-IDNet

Good morning. My name is Fatemeh Mahvari, and this project was completed under
the supervision of Professor Mauro Migliardi.

LM-IDNet is an unsupervised anomaly-detection system for IoT network traffic.
The project began from a statistical method for accurately computing the
Dirichlet–multinomial likelihood, then developed into a complete experimental
pipeline: packet-capture processing, model fitting, threshold calibration,
static and adaptive scoring, labelled evaluation, and reproducible experiment
records.

The detector image is a visual reference for the presentation’s monitoring
theme. The scientific content begins with the traffic data and the statistical
model.

Transition: I will first show how the presentation follows the structure of the
full project report.

## Slide 2 — Presentation Map

The presentation follows the same logical order as the report. I will begin
with the problem and statistical basis, then show the working implementation
and the way the datasets are assigned to different experimental roles.

The second half focuses on evidence. I will compare the static and adaptive
systems, explain the temporal-evaluation work, discuss the additional detector
signals that were tested, and finish with the remaining limitations and the
completion plan.

This order is important because the results only make sense after the data
roles, score definition, and adaptation rules are clear.

Transition: I will start with the shortest possible description of what the
system actually does.

## Slide 3 — Executive Summary: What the Project Is

The detector learns normal behaviour separately for each IoT device. It does
not require examples of attacks during training. Each packet capture is reduced
to consecutive windows containing four mutually exclusive counts: TCP, ordinary
UDP, SSDP, and ARP.

The fitted Dirichlet–multinomial model represents both the expected protocol
mixture and the ordinary variation around that mixture. A later window receives
a log-probability score. A threshold learned from a separate benign calibration
period converts that score into an alert decision.

The diagram on the right shows the complete path: raw captures become windows,
the windows support the model, the model produces alerts, and every stage leaves
an audit record. Static and adaptive configurations use this same path, which
makes their comparison controlled.

Transition: That complete path exists today, but not every research claim is at
the same level of maturity.

## Slide 4 — Executive Summary: Current Project Position

I separate the project into three evidence levels.

The working reference system contains the complete operational path: capture
processing, fitting, calibration, static scoring, adaptive thresholding, guarded
periodic refitting, explanations, and labelled development evaluation.

The additional experimental work addresses two known limitations. The temporal
work checks actual timestamps, repeated windows, several chronological
boundaries, and attack episodes. The detector experiments test traffic volume,
connection behaviour, ARP activity, and direction because protocol composition
alone misses some attacks.

The final claims remain open. The combined detector still needs stronger
adaptation safeguards, one frozen configuration, an untouched labelled
evaluation, and measurement on the intended edge hardware. Throughout the
presentation I will distinguish existing evidence from those remaining claims.

Transition: The need for those safeguards comes directly from the behaviour of
real IoT traffic.

## Slide 5 — Research Problem: Normal IoT Traffic Changes

IoT devices are attractive for anomaly detection because each device usually
performs a narrow set of tasks. At the same time, their normal traffic is not
constant. Routines change, cloud services change, firmware is updated, and the
network itself introduces variation.

A completely fixed detector can therefore become stale and produce false
alerts. An adaptive detector creates the opposite danger: if it learns from the
wrong observations, an attack can become part of the new normal.

The signal illustration distinguishes those cases conceptually. Normal
variation and benign drift should remain acceptable, while an attack should
produce a sufficiently unlikely score. The research problem is not merely to
detect change; it is to detect harmful change while controlling how the model is
allowed to evolve.

Transition: To keep that decision lightweight and interpretable, the baseline
starts with protocol counts.

## Slide 6 — Scientific Basis: Traffic as Counts

Each observation is a 10-minute vector containing the number of TCP, ordinary
UDP, SSDP, and ARP packets associated with one device. SSDP is removed from the
ordinary UDP count, so a packet is not counted in two categories.

This representation has three advantages. It is compact, it has a direct
network interpretation, and it can be produced without retaining packet
payloads. The expected profile can be explained as a mixture of familiar
protocol categories.

The implementation also distinguishes two cases that would otherwise both look
like zeros. A genuinely silent device produces a valid zero-count window.
Missing recording coverage provides no evidence and is excluded. That
distinction prevents capture gaps from being interpreted as device behaviour.

Transition: The next question is which probability distribution can represent
the variation in these count vectors.

## Slide 7 — Scientific Basis: Dirichlet–Multinomial Model

A standard multinomial model assumes one fixed protocol-probability vector.
That is often too rigid for IoT traffic because ordinary activity can be
bursty or can alternate between several operating modes.

The Dirichlet–multinomial adds overdispersion. The ratio of each alpha parameter
to the total concentration gives the expected protocol mixture. The total
concentration describes how tightly windows are expected to remain near that
mixture. A larger concentration means a more tightly concentrated model; its
inverse is reported as the dispersion indicator.

The formula on the right is the complete log-probability of one count vector.
The detector uses that value as its score. A score below the independently
calibrated threshold is anomalous. Absolute thresholds are specific to their
device and model, so they should not be ranked across devices.

Transition: Computing this likelihood accurately is the numerical role of the
specialised backend.

## Slide 8 — Scientific Basis: Specialised Likelihood Method

The Languasco–Migliardi method evaluates close log-Gamma differences inside the
Dirichlet–multinomial likelihood. Model estimation itself uses Minka’s
fixed-point equations. This separation matters: the specialised method supplies
the numerical likelihood used for convergence checks and scoring, while the
fixed-point estimator updates alpha.

The system retains SciPy as an independent reference. This allows direct
agreement and runtime tests instead of assuming that the native implementation
is correct.

The D-Link Camera 5 captures provide a workload that matches the supplied IoT
study. They are appropriate for numerical benchmarking, but there are no attack
labels for those captures in this repository. For that reason, D-Link supports
runtime and numerical evidence, not precision, recall, or F1 claims.

Transition: With the scientific roles clear, I can now show how they are
implemented as a reproducible system.

## Slide 9 — Implementation: Working System Architecture

The system is intentionally divided into explicit stages. Packet captures and a
device policy produce validated count windows. Earlier windows fit the normal
model, a separate benign period calibrates the threshold, and later windows are
scored by either the static or an adaptive mode. Events, audits, and evaluation
reports are written at the end.

Configuration files specify the device, protocol order, dataset roles,
estimator, numerical backend, calibration policy, adaptation mode, and random
seed. The outputs are structured JSON rather than hidden in program memory.

This architecture allows a detector variant to be selected by configuration
and run from the shell. It also means that the systems being compared share the
same preprocessing and scoring implementation.

Transition: Reproducibility depends on rejecting invalid configurations before
an experiment starts.

## Slide 10 — Implementation: Configuration and Controls

The configuration is strict by design. Unknown fields, invalid quantiles,
reversed partitions, duplicate capture assignments, inconsistent category
counts, and impossible adaptation-buffer settings are rejected immediately.
This prevents a misspelled field from silently changing an experiment.

Models, thresholds, and datasets also carry fingerprints. A threshold is tied
to the exact model that produced its calibration scores. An apparently
compatible filename is not enough to combine artefacts.

The status display on the right summarizes this operational sequence:
configuration, data, model, threshold, stream, and audit. Each stage must be
valid before the next stage is trusted. Dry runs and stable error codes make
the same commands usable both interactively and in scripts.

Transition: Those controls begin before model code touches a packet capture.

## Slide 11 — Implementation: Capture Processing

The processor can inventory, fully validate, and fingerprint source captures.
It isolates one device using Ethernet or ARP hardware addresses before asking
Scapy to perform expensive protocol decoding.

That early filter produced a large practical improvement on a 1.48-gigabyte
mixed-device capture: processing fell from 280.0 seconds to 12.5 seconds while
the resulting processed JSON remained byte-identical.

Retained timestamps are normalized to UTC and placed into epoch-aligned,
half-open windows. Provenance records preserve capture identifiers, hashes,
coverage ranges, exclusions, incomplete days, labels, and experimental roles.
Attack annotations remain separate from the count data, so preprocessing does
not turn a labelled dataset into a supervised training input.

Transition: After preprocessing, each statistical stage has one declared job.

## Slide 12 — Implementation: Fit, Calibrate, Score, Explain, Evaluate

Fitting uses only the declared earlier benign captures and stores the fitted
alpha vector, convergence state, likelihood, runtime, data identity, and time
range.

Calibration uses a different benign period. It scores those windows and stores
the lower-tail threshold, interquartile range, score distribution, and model
fingerprint. This prevents the fitting data from also deciding the alert
boundary.

Scoring produces one event for every non-missing development window. Each event
contains the counts, score, decision, severity, expected protocol profile,
category residuals, and the model identity. Evaluation later performs an exact
time-aligned join with the separate labels.

The residuals are intentionally modest explanations: they say which protocol
counts were high or low relative to expectation. They do not claim to identify
an attack family.

Transition: The scoring path also required careful integration of the native
likelihood code.

## Slide 13 — Implementation: Native Likelihood Integration

The supplied numerical source remains preserved as a reference. A separate
production copy is wrapped for safe use inside the application.

The safety changes include exact handling of zero counts, checked files and
allocations, dimension validation, cleanup after successful and failed calls,
and conversion of native failures into Python exceptions instead of process
termination.

The production changes add a matrix entry point and reuse setup across all rows
in a scoring matrix. The adapter converts the fitted alpha representation and
protects the numerical code’s legacy global state. Importantly, the detector
still sums the likelihoods of independent windows; it does not replace them
with one pooled count vector.

Transition: The same scoring implementation is then reused by all three
adaptation modes.

## Slide 14 — Adaptation Modes: Shared Scoring Path

There are three comparable systems.

The static system never changes its model or threshold. Threshold-only
adaptation keeps alpha fixed but periodically recalculates the lower score
quantile and interquartile range. Periodic refitting admits apparently safe
windows into a buffer and may promote a new model and threshold.

The ordering shown on the left is a core safeguard. The current window is
scored and its decision is recorded before it can update any buffer or affect
future state. This avoids using the updated state to rewrite the decision that
triggered the update.

The original saved model and threshold are never overwritten. Every adaptive
run writes separate events and an update audit, so the static result remains a
permanent control.

Transition: The first adaptive system is deliberately simple and acts as a
baseline for threshold drift.

## Slide 15 — Adaptation Mode: Threshold-Only

Threshold-only adaptation keeps the original alpha vector and expected protocol
profile fixed. After scoring each non-missing window, it appends the score to a
bounded recent-score buffer. At the configured update interval, it recalculates
the same lower quantile and interquartile range used during calibration.

A candidate with a non-positive interquartile range is rejected because it
cannot support the existing severity calculation.

The main limitation is deliberate: every recent score enters the buffer,
including anomalous scores. This mode therefore answers a narrow experimental
question—whether adapting only the score boundary improves the detector. It is
not presented as a safe online-learning policy.

Transition: Periodic refitting adds a separate admission rule and stricter
promotion conditions.

## Slide 16 — Adaptation Mode: Guarded Periodic Refitting

Every candidate window is judged by the unchanged original detector. It enters
the refit buffer only when its original score is above the original threshold
plus a safety margin measured in original interquartile ranges.

Once enough accepted windows are available, the existing fixed-point estimator
fits a candidate alpha vector. Four artificial one-count rows—one for each
protocol—prevent a temporarily absent category from receiving an invalid zero
alpha.

Promotion is atomic. The candidate must converge, improve likelihood on the
accepted buffer, and produce a positive recent-score interquartile range. The
alpha vector, expected profile, fingerprint, threshold, and IQR change together.
If a check fails, the old model remains active and the audit records the reason.

These checks make the mechanism reproducible and fail-closed, but later results
will show that they are not sufficient to prevent attack contamination.

Transition: Before discussing results, I will state what evidence the
implementation itself produces.

## Slide 17 — Implementation: Validation and Generated Evidence

At the report boundary, the default automated suite passes 261 tests. Ten slow,
packet-capture, or benchmark checks are excluded by the default markers rather
than silently counted as passes.

The tests cover configuration, timestamps, classification, aggregation,
schemas, failure handling, fitting, calibration, scoring, evaluation, numerical
agreement, and adaptive update ordering. They also check candidate promotion,
rejection, and output fingerprints.

Experiments write models, thresholds, events, evaluations, adaptation audits,
capture records, benchmarks, and UTC logs to separate locations. This means
that the evidence shown later can be traced to concrete outputs instead of
reconstructed from console messages.

Transition: The next two slides explain which data may support each type of
claim.

## Slide 18 — Dataset Protocol: Roles and Composition

The three datasets have different scientific roles.

D-Link Camera 5 supplies 846 fitting windows, 288 calibration windows, 270
development windows, and 288 declared final windows. It is the numerical
reference workload and does not provide attack-labelled detection metrics.

UNSW Chromecast supplies 867 fitting windows and 576 benign calibration
windows. Its development period contains 474 windows, including 32 attack
windows. The declared final period contains 432 windows, including six labelled
attacks. The October 24 source is excluded because the published capture is
truncated.

UNSW Samsung camera supplies 432 fitting windows, 81 calibration windows, 556
development windows with 70 attack windows, and 576 declared final windows with
12 attack windows. Complete mixed-device captures are filtered to the selected
device before aggregation.

Transition: The counts alone are not enough; the temporal order of those roles
must also be protected.

## Slide 19 — Dataset Protocol: Chronology and Leakage

The intended sequence is earlier benign fitting data, later benign calibration
data, still-later development traffic, and finally sealed test data.

Attack labels do not fit alpha, choose thresholds, or decide which windows enter
adaptive buffers. Development labels are consulted only after a run has
produced all of its decisions. Complete captures remain intact instead of
randomly mixing their windows.

This policy is stronger than random sampling, but it is not a complete temporal
evaluation. Adjacent windows remain correlated, one boundary represents only
one version of history, and a date embedded in a filename does not prove that
the actual processed time ranges are disjoint. Those limitations motivate the
later temporal-fold work.

Transition: With the protocol defined, I can now compare the detector results.

## Slide 20 — Static Result: Precision and Recall

The static detector is the fixed scientific control.

For Chromecast, it produces 15 true positives, 440 true negatives, two false
positives, and 17 false negatives. Precision is 88.24 percent, but recall is
only 46.88 percent, giving an F1 score of 61.22 percent.

For Samsung, it produces 48 true positives, 482 true negatives, four false
positives, and 22 false negatives. Precision is 92.31 percent, recall is 68.57
percent, and F1 is 78.69 percent.

The shared pattern is high precision with incomplete recall. Accuracy is also
high, but benign windows are much more common than attack windows, so accuracy
can conceal missed attacks. D-Link is not shown as a detection result because
it has no attack ground truth.

Transition: The adaptation experiment asks whether changing future detector
state improves that baseline.

## Slide 21 — Adaptation Result: Mode Comparison

This graph compares F1 under four systems using the same initial artefacts and
development streams.

Threshold-only adaptation makes both datasets worse: Chromecast falls from
61.22 to 55.32 percent, and Samsung falls from 78.69 to 54.00 percent. The first
periodic configuration, with a 144-window update and 144 accepted windows
required for refitting, recovers one Chromecast attack window but does not
change the Samsung decisions.

The selected 288/240 periodic configuration raises Chromecast F1 to 80.70
percent. Samsung reaches 77.31 percent, close to its 78.69 percent static
result. This is promising development evidence, not a final result, because the
cadence was selected after inspecting these development streams.

Transition: The next two graphs show why update cadence cannot be treated as an
innocent implementation detail.

## Slide 22 — Adaptation Result: Cadence — Chromecast

The dashed line is the static F1 score. The lighter curve is threshold-only
adaptation, and the burgundy curve is periodic refitting.

For Chromecast, most intervals do not improve the static detector. The major
change occurs at an update interval of 288 windows, where periodic refitting
reaches 80.70 percent. An interval of 336 windows is also strong, but later
intervals fall back to the poorer threshold-only outcome.

The explanation is positional. The useful refit occurs before a later group of
attacks, while a later update cannot affect those earlier decisions. The graph
therefore shows sensitivity to the order of events, not a universal optimum at
288.

Transition: Samsung shows the complementary failure—an update that occurs too
early is harmful.

## Slide 23 — Adaptation Result: Cadence — Samsung Camera

Samsung performance improves gradually as the first update is delayed. At 144
windows, both adaptive systems have an F1 score of 54.00 percent. By 288
windows, they reach 77.31 percent, and at 336 windows they slightly exceed the
static F1.

This pattern indicates that the early update is the main source of damage.
Delaying the update preserves the original detector through that vulnerable
part of the stream.

The similar threshold-only and periodic curves also show that a successful
refit does not necessarily change decisions. Depending on when it happens, the
threshold movement can dominate the observed effect.

These curves justify keeping 288/240 as a development candidate, but they also
show why a single stream cannot establish a general cadence.

Transition: Performance alone is not enough; the adaptive data itself must be
audited for safety.

## Slide 24 — Adaptation Safety: Attack Contamination

After each run, labels were used only for an audit: to count how many known
attack windows had entered the supposedly safe refit buffer. The detector did
not use those labels during admission or updating.

For Chromecast, the buffer is clean at the first update and reaches only 1.04
percent contamination later. For Samsung, contamination is 9.13 percent at the
first selected update: 22 attack windows among 241 accepted windows. It remains
6.94 percent later.

Increasing the safety margin on the same composition score did not solve the
Samsung problem. Those attacks already look ordinary when the detector observes
only protocol proportions. This points to an independent rate or volume check
rather than another threshold on the same information.

Transition: The second safety audit asks how far the fitted model is allowed to
move.

## Slide 25 — Adaptation Safety: Model Movement

This graph reports the final concentration divided by the initial
concentration. Chromecast changes moderately: a factor of 1.28 in the first
periodic experiment and 1.23 in the selected configuration.

Samsung is very different. Its concentration increases by a factor of 11.87 in
the first experiment and 9.58 in the selected configuration. The model becomes
much more concentrated, even though the fit converged and improved likelihood
on the accepted buffer.

This is evidence that convergence and buffer likelihood are necessary but not
sufficient promotion checks. A safer design should either blend old and
candidate alpha values or reject a candidate when total concentration moves
beyond a predeclared limit.

Transition: The final result for the working system concerns numerical runtime
and agreement.

## Slide 26 — Numerical Result: Workload-Dependent Runtime

The graph normalizes native runtime by SciPy runtime. A ratio below one means
the native backend is faster; a ratio above one means SciPy is faster.

Across all three datasets, the native backend is slower for a complete model fit
and for evaluating the full training matrix. It is slightly faster for scoring
one window, which is the repeated online operation. The correct claim is
therefore workload-dependent performance, not universal acceleration.

Numerical agreement is strong. On the D-Link production benchmark, the maximum
relative alpha difference is \(3.86 \times 10^{-7}\), the maximum absolute score
difference is \(1.09 \times 10^{-5}\), and there are zero decision mismatches
across 270 windows.

Transition: The static and adaptive results still use one main chronological
boundary, so the next work examines temporal stability.

## Slide 27 — Temporal Evaluation: Why One Split Is Not Enough

Ten-minute windows are not independent trials. A cloud upload, scanning burst,
or attack can continue through several adjacent windows.

The measured lag-one autocorrelation is about 0.86 for Chromecast packet count.
Score autocorrelation is about 0.39 for Chromecast and 0.42 for Samsung. This
does not prove that the count model is invalid, but it means that hundreds of
adjacent windows do not represent hundreds of independent operating periods.

The same issue affects labels. Six adjacent attack windows may represent one
continuing incident. Evaluation should therefore report attack-episode recall,
false-alert episodes per day, detection delay, and performance by capture in
addition to window-level metrics.

Transition: The temporal design repeats the experiment at several points in
history while keeping final data outside every fold.

## Slide 28 — Temporal Evaluation: Fold Design and Evidence

Each temporal fold uses earlier history for fitting, a later benign block for
calibration, and a still-later capture for evaluation. Fitting history expands
as the boundary moves forward. Actual UTC ranges are checked globally, and an
unexplained overlap stops the run.

This work found 144 repeated D-Link window identities in the current fitting
sources. That overlap must be resolved before fitting the final model.

The labelled temporal folds detect seven of ten Chromecast attack episodes and
23 of 28 Samsung episodes. Episode recall is therefore 0.70 and 0.82,
respectively. False-alert episode rates are 0.91 and 0.78 per day.

Two benign Chromecast folds each contain 288 windows but produce 12 and two
false alerts. A one-percent calibration quantile does not guarantee the same
future false-alert rate across operating periods.

Transition: Time is one limitation; the other is the information contained in
the four protocol counts.

## Slide 29 — Detector Experiment: Composition Gap

The static model evaluates the relative mixture of TCP, UDP, SSDP, and ARP.
That question is useful, but it does not measure whether the total traffic
volume or direction is ordinary.

One missed Chromecast UDP flood illustrates the gap. Its packet rate was
abnormal, but the relative protocol composition remained close to normal. Other
missed attacks changed total bytes, TCP connection attempts, ARP activity, or
incoming and outgoing traffic.

The additional detector experiment therefore retains protocol composition as
the baseline and adds the smallest interpretable signals tied to observed
misses: packet count, byte count, TCP SYN count, ARP count, and direction.

Transition: The next graph shows the measured effect of adding those signals
one at a time.

## Slide 30 — Detector Experiment: Chromecast Development

The initial Dirichlet–multinomial baseline has an F1 score of 61.22 percent.
Adding a packet-count check raises it to 64.00 percent by finding the high-rate
UDP flood without a new false alert.

The larger improvement, to 86.15 percent, comes from the group of traffic
checks for bytes, TCP SYN packets, and direction. An ARP-count check raises F1
to 89.55 percent. Correctly handling silent windows raises it to 90.91 percent.
Finally, modelling incoming and outgoing counts with a two-category
Dirichlet–multinomial raises it to 92.54 percent.

Each accepted change addresses an observed failure. The experiment does not add
a general classifier or an opaque learned feature layer.

Transition: A second device shows which parts of that improvement transfer and
which still need calibration.

## Slide 31 — Detector Experiment: Cross-Device Evidence

The latest experimental detector reaches 92.54 percent F1 and 99.70 percent
ROC–AUC on Chromecast development data.

It is then applied to Amazon Echo without device-specific retuning. Echo reaches
69.23 percent F1 and 95.93 percent ROC–AUC. It detects every labelled ARP attack
window, but it misses several low-rate UDP windows and produces more benign
alerts. This is useful transfer evidence, but it also shows that the additional
feature thresholds need cross-device calibration.

An earlier frozen six-signal detector was evaluated once on Chromecast final
data and achieved 66.67 percent F1. That dataset cannot provide unbiased
evidence for later direction and score-ranking refinements because its labels
have already been examined.

Transition: Failed ideas are also recorded, so the proposed combined detector
does not accumulate features without evidence.

## Slide 32 — Detector Experiment: Rejected and Retained Changes

Adjacent-window byte change recovered one Echo attack window but added false
alerts across devices. Mean packet-size tails were unstable. Feature
corroboration rules suppressed attacks that were correctly visible in only one
signal. Automatic promotion of an adapted candidate also failed a later-traffic
safety check.

Those variants remain documented but are not part of the proposed detector.

The strongest candidates to carry forward are packet count, byte count, TCP SYN
count, ARP count, direction modelling, attack-episode reporting, and the
detector-health audit. Score normalization is optional: it helps create one
ranking across signals, but it enlarges the threshold artefact without changing
the present alert decisions.

Transition: Together, the positive and negative results define what the project
can honestly claim today.

## Slide 33 — Interpretation: Strengths and Limits

The strongest contribution is the evidence chain. Source bytes, device
identity, window policy, model, threshold, event, explanation, and evaluation
remain connected through explicit artefacts and fingerprints. Fitting and
adaptation do not require attack labels, and the numerical implementation is
checked against an independent reference.

The static detector establishes a high-precision baseline, while the additional
signals show a clear path to recovering composition-preserving attacks. The
temporal and adaptation audits expose failure modes that a single pooled F1
score would hide.

The evidence does not yet support calling the system a finished edge-security
product. Generality across devices, poisoning resistance, stability across
time and window length, final refined-system performance, and target-hardware
cost all remain open. The current forecasting command is also synthetic metric
scaffolding, not empirical forecasting evidence.

Transition: Those limits lead to four concrete completion priorities.

## Slide 34 — Completion Plan: Four Priorities

First, enforce temporal integrity in the common system: order actual timestamps,
reject unexplained overlap, preserve complete captures, and report capture and
episode metrics.

Second, compare detector versions without replacing the control. The
composition-only static detector remains the named baseline. A separate hybrid
configuration can add the smallest supported signals. The same temporal
development protocol should compare several window lengths.

Third, strengthen adaptation one safeguard at a time: use only gate-accepted
scores for threshold updates, compare refitting with a fixed threshold, limit
alpha movement, add an independent volume gate, and judge promotion on data the
candidate did not fit.

Fourth, freeze the architecture, settings, data identity, and evaluation rules.
Then run one genuinely untouched labelled evaluation and benchmark the selected
system on the intended edge hardware.

Transition: I will finish by stating the project’s current position in three
sentences.

## Slide 35 — Conclusion: Current Position

First, LM-IDNet is a working end-to-end research system. It validates captures,
builds device-specific windows, fits and calibrates a statistical model, scores
traffic, supports static and adaptive operation, explains alerts, and writes
auditable evidence.

Second, the results identify both value and failure modes. Static detection is
precise but incomplete. The selected adaptive configuration strongly improves
one development stream, but the contamination and concentration audits show why
it is not final. Temporal and multi-signal experiments address different,
measured limitations.

Third, the remaining experiment is now focused: combine only the supported
controls and signals, freeze one common detector, and evaluate it on untouched
data. The contribution so far is not only the best score; it is the transparent
record of what was built, what improved, and what failed.

Transition: I am ready for questions on the model, implementation, experiments,
or remaining evaluation.

## Slide 36 — Questions and Discussion

Invite questions and leave this slide visible.

If the question concerns exact adaptive confusion matrices, use slides 37 and
38. If it concerns the selected settings or commands, use slides 39–41. If it
concerns where the evidence is stored, use slide 42.

Do not introduce a new claim during discussion. Distinguish clearly between the
working reference system, additional experimental evidence, and the evaluation
that remains open.

---

# Backup Slides

## Slide 37 — Backup: Chromecast Adaptation Metrics

Use this table when someone asks whether the F1 improvement came from more true
positives or from more false alerts.

The selected 288/240 periodic system increases true positives from 15 to 23,
reduces false negatives from 17 to nine, and keeps false positives fixed at two.
Precision therefore rises to 92.00 percent while recall rises to 71.88 percent.
The weaker 144-window variants reduce recall.

## Slide 38 — Backup: Samsung Adaptation Metrics

Use this table for the corresponding Samsung question.

The selected periodic system produces 46 true positives, 483 true negatives,
three false positives, and 24 false negatives. Compared with static, it removes
one false positive but misses two additional attack windows. Its F1 score of
77.31 percent is therefore close to, but slightly below, the static 78.69
percent.

## Slide 39 — Backup: Selected Adaptation Configuration

The selected development configuration uses periodic refitting, a buffer of 288
usable observations, an update interval of 288 windows, a minimum of 240
accepted windows, and a safety margin of one original IQR.

Missing windows are skipped rather than inserted into the buffer. Both selected
development runs promote one candidate model. Emphasize that these settings are
an experimental candidate selected from the cadence study, not frozen final
settings.

## Slide 40 — Backup: Reproduction Commands

These commands show the explicit working-system stages for Chromecast:
preprocessing, training, calibration, scoring, and evaluation. Samsung uses its
corresponding device configuration.

The adaptation experiment runner executes the stored comparison configurations
and resolves the repository root internally, so it works from another current
directory or when invoked by absolute path.

## Slide 41 — Backup: One Selected Adaptive Run

Use these commands when asked how to reproduce only the selected Chromecast
periodic-refit result rather than the full experiment set.

The configuration directory and filename are one continuous path at the shell.
They are broken across lines on the slide only for readability. Run `adapt`
first to generate events and the update audit, then run `evaluate` to align
those events with the development labels.

## Slide 42 — Backup: Evidence Map

This slide identifies the primary records behind the presentation. The full
project report provides the complete architecture and interpretation. The
adaptation report contains the cadence sweep and safety analysis. The numerical
change record documents the native integration.

The per-device evaluation JSON files contain the static confusion matrices.
The experiment-report directory contains adaptive events and audits. The
temporal architecture review explains timestamp integrity, folds, and
episode-level evaluation. Use this map to point to evidence rather than relying
on memory during detailed questions.
