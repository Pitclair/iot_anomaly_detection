# Prepared detector input and one-window processing

`WindowDetector` provides the shared detector used by batch scoring and the
streaming `run` command. It loads the backend and detector state at construction, then
returns one window's event and update attempts from `process()`. Both adaptation
policies and all three detector modes retain their existing rules. Gateway
packaging and hardware validation remain planned work.

The input contract identifies the device, its MAC, the ordered categories, the
trained window duration, and `device-mac-bidirectional-ssdp-exclusive-v1`:

- Include packets whose Ethernet source/destination or ARP sender/target MAC
  matches the device. Count both directions.
- Count TCP, ordinary UDP, SSDP, and ARP. A UDP packet with source or destination
  port 1900 is SSDP and is excluded from ordinary UDP. Ignore unsupported protocols.
- Use UTC, epoch-aligned, left-closed windows of the trained duration. The current
  configurations use ten minutes. Preserve the model's category order.
- Supply non-negative integer counts. Booleans, numeric strings, fractions, and
  non-finite values are invalid. Each count and the packet total must be at most
  `2**63 - 2`, allowing the numerical implementation to add one without overflow.
- An `observed-silent` window has four zeros and is scored. A `missing` window has
  `counts: null` and is neither scored nor learned from.

The model stores this contract and the fingerprint binds it to its threshold.
Calibration and scoring check device identity and duration as well as category
order. Models and state snapshots now use format version 2. Old models without
the contract and older saved-state directories must be regenerated; changing a
configuration cannot establish how an old model was trained. The bundled model
pairs were regenerated from their configured processed data with unchanged alpha
values and numeric thresholds. Existing historical experiment reports are retained.

## Prepare state on the development machine

The bundled D-Link example already has a conservative periodic-refit state in
`artifacts/edge-initial-state`. To use that example, copy it to a fresh writable
directory and load `configs/edge_example.json`:

```sh
test ! -e artifacts/edge-state && cp -R artifacts/edge-initial-state artifacts/edge-state
```

For your own settings, prepare a fresh state as follows.

Train and calibrate with the intended research configuration. For an adaptive
detector, set its mode, policy, and limits in that configuration first. The
prepared state's settings must match the edge configuration exactly.

```sh
lm-idnet train --config configs/d_link_day_cam5.json
lm-idnet calibrate --config configs/d_link_day_cam5.json
```

Create a fresh state directory and its edge configuration with the existing
state API. Keep any old directory separately for its original software revision.
Run this from the project root in the installed Python environment:

```python
from pathlib import Path
from lm_idnet.artifacts import detector_state_lock, load_detector_state
from lm_idnet.config import DetectorInputConfig, EdgeConfig, load_config

research = load_config("configs/d_link_day_cam5.json")
research.adaptation.mode = "periodic_refit"
directory = Path("artifacts/edge-state")
with detector_state_lock(directory):
    state = load_detector_state(directory, research)

edge = EdgeConfig.model_validate({
    **research.model_dump(include={
        "precision_digits", "estimator", "calibration", "adaptation",
    }),
    "ingest": research.ingest.model_dump(
        include=set(DetectorInputConfig.model_fields),
    ),
    "outputs": research.outputs.model_dump(
        include={"model_path", "threshold_path"},
    ),
    "state_dir": directory,
})
Path("configs/edge_local.json").write_text(edge.model_dump_json(indent=2) + "\n")
```

Initialisation uses the calibration captures for the conservative volume guard
when needed. The resulting snapshot embeds the baseline model, matching
threshold, original safety references, settings, and volume limit. Training and
calibration capture IDs remain inside those artefacts as provenance.

Copy the complete prepared state directory and edge configuration after releasing
the lock. Adjust `state_dir` for the gateway. The example
[`edge_example.json`](../configs/edge_example.json) uses the D-Link configuration
with conservative periodic refitting enabled. Other configurations should be
generated from their research settings using the same snippet. Edge configuration has no raw or
processed data roots and no capture partitions. Its model/threshold paths allow
`load_model_threshold(edge)` to validate a standalone pair; loading prepared state
uses the embedded pair and does not require those files.

## Stream windows through the CLI

After copying the bundled initial state to `artifacts/edge-state`, the installed
command accepts completed count windows without research data or capture lists:

```sh
lm-idnet run --config configs/edge_example.json \
  --state-dir artifacts/edge-state <<'JSONL'
{"device_id":"D-LinkDayCam5","source_id":"local-collector","window":{"start_utc":"2026-10-15T12:00:00Z","end_utc":"2026-10-15T12:10:00Z","categories":["tcp","udp","ssdp","arp"],"counts":[100,20,3,2],"state":"observed"}}
JSONL
```

Keep each record on one physical line. To use a local collector, pipe its JSON
lines to the same command. `--state-dir` is optional when the edge configuration
already names the correct directory. Use `configs/edge_local.json` for your own
prepared detector. `--dry-run` checks that configuration without consuming input
or creating detector state, and prints the normal readiness status.

During normal execution, standard output contains one existing anomaly event per
scored window, with its score, threshold, decision, severity, expected profile,
and category residuals. Each event is flushed immediately, even while input stays
open. Status messages and accepted/rejected update attempts go to a rotating log:
`logs/lm_idnet_runtime.log`, 5 MiB per file with two backups. Override its path with
`LM_IDNET_LOG_PATH`; use global `--verbose` to also see informational logs on
standard error. The output consumer owns event retention.

The reader allows at most 65,536 bytes per line, including its ending. It checks
UTF-8 JSON, required fields, integer counts, the device, category order, duration,
and UTC window grid before scoring. Timezone-aware offsets are normalised to UTC.
Duplicate, backward, or overlapping windows stop the command. Missing records
advance the coverage boundary while producing no score or update. Gaps do not
create artificial zero-count windows. Blank, malformed, incomplete, non-finite,
or oversized records stop with `input line N` on standard error and a nonzero
exit status. Previously emitted events remain usable. End of input exits cleanly,
and a complete final record does not require a newline.

The command holds the existing state lock for its lifetime, reuses accepted-state
loading and atomic updates, and releases the lock when input ends or validation
fails. Input and output run synchronously: a slow consumer stops further input
processing instead of growing a queue. Normal service-signal shutdown and edge
configuration support in the rollback CLI remain separate lifecycle work.

## Process a window through the Python API

Use `configs/edge_local.json` below when you prepared your own settings.

```python
from lm_idnet.artifacts import detector_state_lock
from lm_idnet.config import load_edge_config
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.schemas import WindowRecord

config = load_edge_config("configs/edge_example.json")
window = WindowRecord(
    start_utc="2026-10-15T12:00:00Z",
    end_utc="2026-10-15T12:10:00Z",
    categories=config.ingest.categories,
    counts=(100, 20, 3, 2),
    state="observed",
)
with detector_state_lock(config.state_dir):
    detector = WindowDetector(config)
    event, threshold_attempt, refit_attempt = detector.process(
        window, device_id=config.ingest.device_id, capture_id="local-collector",
    )
    print(event)
```

For successive windows, construct the detector once and call `process()` inside
the same lock. The source identifier need not contain a date. The event describes
the detector used to score that window; any accepted update affects the next
window. State is saved before the new pair becomes active. Pre-commit failures
reject the update; a failure after the active pointer has switched stops the
caller so it can reload the committed pair.

The return values contain only this window's event and attempts, with `None` for
work that did not occur. Consume or write them immediately. The detector retains
bounded count/score deques and counters, without accumulating result or attempt
lists. The batch adapter collects its usual full JSON reports separately.
The existing saved-state helpers retain the active/previous versions and the
last 100 audit entries.

The Python API checks device, category order, and duration; its `WindowRecord`
input validates counts and timestamps. The streaming command additionally checks
window ordering and the UTC grid while bounding its JSON-line reader. State
restores the accepted pair, while learning buffers
and counters restart empty. It does not restore an input cursor or prevent
learning from a replayed window after restart. Stop any running writer before
using the existing rollback API.
