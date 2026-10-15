# LM-IDNet configuration

LM-IDNet reads one JSON file and validates the entire file before importing or
running pipeline stages. The default file is
[`configs/d_link_day_cam5.json`](../configs/d_link_day_cam5.json), and the typed schema is defined
in [`src/lm_idnet/config.py`](../src/lm_idnet/config.py).

## Sections

- `precision_digits`: positive numerical precision setting.
- `ingest`: raw/processed paths, dataset name, column names, window length,
  ordered protocol categories, and the fit/calibration/development/final
  capture partitions.
  Required `device_id` and `device_mac` fields identify the device. The ID is
  retained in processed metadata and anomaly events; only the stable MAC
  address selects packets from mixed-device PCAPs.
  `allowed_duplicate_captures` documents exact, justified groups of captures
  that are intentionally byte-identical; it is empty by default.
- `seeds`: independent non-negative seeds for simulation, model fitting,
  bootstrap procedures, and other randomized algorithms.
- `estimator`: category count, positive initial alpha concentration, positive
  convergence tolerance, positive iteration limit, and the selected
  likelihood-kernel backend. `lm` selects the bundled native
  Languasco--Migliardi implementation and is used by the versioned research
  configurations. `scipy` selects the reference implementation based on
  `scipy.special.gammaln`. Both implement the same row-wise kernel and neither
  includes the multinomial coefficient; the complete formulas and pipeline
  ownership are documented in
  [`likelihood-semantics.md`](likelihood-semantics.md). The top-level
  `precision_digits` value controls LM evaluation in training, diagnostics,
  calibration, and scoring; it is accepted but not needed by SciPy.
- `calibration`: lower-tail quantile, required positive sample count, and
  `raw` (default) or per-packet `normalized` anomaly score.
  Compare them by running `calibrate` and `score` once per value with distinct
  threshold and event output paths.
- `adaptation`: `static`, `adaptive_threshold`, or `periodic_refit` detector
  mode, with a `policy` of `conservative` (default) or `legacy`.
  The following accepted-buffer rules describe the conservative policy.
  `buffer_size` bounds one buffer of accepted count windows, and
  `update_every` sets the number of scored windows between update attempts.
  `minimum_refit_windows` is the required number of high-confidence windows
  for a model refit. A window must clear the threshold plus `safe_margin`
  times the score IQR under both the active detector and the original static
  detector. `safe_margin` defaults to 1.0. Missing windows are skipped; genuine
  silent windows are scored and can be accepted.
  Both adaptive modes derive thresholds exclusively from accepted windows.
  Rejected refits keep the entire active model/threshold/IQR pair unchanged.
- `outputs`: paths for model, threshold, event, and report artifacts.
- `forecast`: positive forecast horizon.

### Conservative update limits

These settings have defaults, so existing configurations and saved model,
threshold, and processed JSON files remain usable. Conservative runs additionally
need the existing processed benign calibration files; original PCAPs are not
required. Limits are fixed for a run and reported in `adaptation.json`.

| Adaptation setting | Default | Rule |
|---|---|---|
| `volume_quantile` | 0.99 | Reject learning windows above this quantile of benign calibration packet totals. Linear interpolation is used. 1.0 selects the observed maximum. |
| `max_concentration_ratio` | 1.25 | Candidate alpha concentration must stay between active concentration / 1.25 and active concentration * 1.25. |
| `max_category_shift` | 0.05 | Each category's expected proportion may move by at most five percentage points per update. |
| `max_threshold_shift_iqr` | 2.0 | Absolute threshold movement per update may not exceed two **original calibration** score IQRs. |

The defaults are conservative software limits, not empirically tuned device
settings. They can reject otherwise valid changes. Repeated allowed changes
can accumulate, while admission remains tied to the original static detector.
They reduce contamination risk without guaranteeing that all attacks are excluded.

### Earlier adaptation policy

Set `adaptation.policy` to `legacy` to reproduce the pre-Step-2 methods:

```json
"adaptation": {
  "mode": "periodic_refit",
  "policy": "legacy",
  "buffer_size": 288,
  "update_every": 288,
  "minimum_refit_windows": 240,
  "safe_margin": 1.0
}
```

`adaptive_threshold` estimates its threshold and IQR from all recent scores,
including flagged windows. `periodic_refit` admits fitting windows using only
the original static detector's confidence gate. It retains smoothing,
convergence, finite positive parameter, and likelihood checks. A successful
refit recalculates its threshold on all recent count windows; a failed refit
can still update the threshold using all recent scores. Every window is scored
before an update can affect subsequent windows. No labels are used.

The volume and movement limits above do not apply to legacy runs and appear
as `null` in their audit reports. Legacy runs do not load calibration counts.
This policy preserves the earlier contamination risk for research comparisons.
The six ready-to-run configurations under `configs/experiments/legacy/` retain
the earlier experiment settings and write to separate `legacy` directories.
They reuse the saved initial model and threshold. Select a policy through the
same configuration in both the CLI and Python:

```python
from lm_idnet.config import load_config
from lm_idnet.models.scoring_stage import adapt_windows

config = load_config("configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json")
result = adapt_windows(config)
assert result["adaptation_policy"] == "legacy"
```

## Validation behavior

Configuration validation is intentionally strict:

- Unknown fields are rejected. This catches misspellings instead of silently
  ignoring them.
- Required sections and fields cannot be omitted.
- Category names are stripped and converted to lowercase once at the
  configuration boundary.
- `ingest.categories` is the sole source of feature-column order. The current
  paper taxonomy is fixed to `tcp`, `udp`, `ssdp`, and `arp`, but those names
  may be configured in any order and that order is preserved throughout the
  pipeline.
- Categories must remain unique after normalization and must contain exactly
  the fixed paper taxonomy.
- `estimator.categories_k` must equal the number of configured categories.
- Every temporal partition must be non-empty, internally chronological, and
  disjoint from every other partition.
- Partition boundaries must move forward in the order fit, calibration,
  development-test, and final-test.
- Window sizes and count-like settings must be positive.
- `ingest.window_minutes` accepts any positive integer. The versioned experiment
  configuration uses 10 minutes; this is a default research choice, not a
  software whitelist.
- Seeds must be non-negative.
- Calibration quantiles must satisfy \(0 < q < 1\).
- Non-static buffers must be large enough for the configured minimum
  calibration sample count. In `periodic_refit` mode,
  `minimum_refit_windows` cannot exceed `buffer_size`.
- Adaptation margins and movement limits must be finite. `volume_quantile`
  must be in `(0, 1]`, `max_concentration_ratio` must be at least 1,
  `max_category_shift` must be in `[0, 1]`, and `max_threshold_shift_iqr`
  must be non-negative.
- `adaptation.policy` accepts only `conservative` or `legacy`. Omission keeps
  the conservative policy.

Validation errors identify the offending field and prevent the command from
starting. No partially validated dictionary is passed to the application.

## Loading configuration in Python

```python
from lm_idnet.config import load_config

config = load_config("configs/d_link_day_cam5.json")
print(config.ingest.window_minutes)
print(config.outputs.model_path)
```

The result is an `AppConfig` instance, so settings are accessed through typed
attributes rather than nested dictionary `.get()` calls.

## Checking a change

After editing the JSON file, run:

```sh
python -c "from lm_idnet.config import load_config; print(load_config('configs/d_link_day_cam5.json').model_dump_json(indent=2))"
python -m pytest tests/test_config.py
```

The normalized configuration is snapshot-tested in
`tests/snapshots/normalized_config.json`. An intentional change to normalized
defaults or structure requires reviewing and updating that snapshot.
