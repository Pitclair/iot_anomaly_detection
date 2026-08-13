# LM-IDNet configuration

LM-IDNet reads one JSON file and validates the entire file before importing or
running pipeline stages. The default file is
[`configs/config.json`](../configs/config.json), and the typed schema is defined
in [`src/lm_idnet/config.py`](../src/lm_idnet/config.py).

## Sections

- `precision_digits`: positive numerical precision setting.
- `ingest`: raw/processed paths, dataset name, column names, window length,
  ordered protocol categories, and the fit/calibration/development/final
  capture partitions.
  Required `device_name` and `device_mac` fields identify the device. The name
  is retained in processed metadata and anomaly events; only the stable MAC
  address selects packets from mixed-device PCAPs.
  `allowed_duplicate_captures` documents exact, justified groups of captures
  that are intentionally byte-identical; it is empty by default.
- `seeds`: independent non-negative seeds for simulation, model fitting,
  bootstrap procedures, and other randomized algorithms.
- `estimator`: category count, positive initial alpha concentration, positive
  convergence tolerance, positive iteration limit, and the selected
  likelihood-kernel backend. `scipy` is the working pipeline backend; `lm`
  selects the placeholder for the future Languasco-Migliardi implementation
  and fails until it is implemented. Neither backend includes the multinomial
  coefficient; the complete formulas and pipeline ownership are documented in
  [`likelihood-semantics.md`](likelihood-semantics.md).
- `calibration`: lower-tail quantile, required positive sample count, and
  `raw` (default) or per-packet `normalized` anomaly score.
  Compare them by running `calibrate` and `score` once per value with distinct
  threshold and event output paths.
- `adaptation`: whether adaptation is enabled, its non-negative safety margin,
  and positive buffer capacity. Adaptation remains disabled by default.
- `outputs`: paths for model, threshold, event, and report artifacts.
- `forecast`: positive forecast horizon.

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

Validation errors identify the offending field and prevent the command from
starting. No partially validated dictionary is passed to the application.

## Loading configuration in Python

```python
from lm_idnet.config import load_config

config = load_config("configs/config.json")
print(config.ingest.window_minutes)
print(config.outputs.model_path)
```

The result is an `AppConfig` instance, so settings are accessed through typed
attributes rather than nested dictionary `.get()` calls.

## Checking a change

After editing the JSON file, run:

```sh
python -c "from lm_idnet.config import load_config; print(load_config('configs/config.json').model_dump_json(indent=2))"
python -m pytest tests/test_config.py
```

The normalized configuration is snapshot-tested in
`tests/snapshots/normalized_config.json`. An intentional change to normalized
defaults or structure requires reviewing and updating that snapshot.
