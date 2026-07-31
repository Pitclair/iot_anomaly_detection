# Public synthetic smoke fixture

The fixture is `tests/data/smoke/synthetic_packet_counts.json`. It is a tiny
synthetic event stream used to check integration and reproducibility without
requiring private research captures.

It is:

- explicitly synthetic;
- payload-free;
- free of IP addresses, MAC addresses, hostnames, and real device identifiers;
- dedicated to the public domain under SPDX identifier `CC0-1.0`;
- small enough to run in the normal test suite.

## Known expected windows

It declares three consecutive ten-minute UTC windows:

| Window | TCP | UDP | SSDP | ARP | State |
|---|---:|---:|---:|---:|---|
| 00:00--00:10 | 2 | 1 | 0 | 0 | observed |
| 00:10--00:20 | -- | -- | -- | -- | missing |
| 00:20--00:30 | 0 | 0 | 1 | 2 | observed |

Events on a boundary enter the later window because the reference aggregation
uses half-open intervals `[start, end)`. Integration tests compare every field
of the produced windows with the exact expected JSON.

The fixture declares the middle interval as a capture discontinuity. Its count
vector is `null`, so unavailable observation cannot be mistaken for observed
silence or passed to model fitting as a row of zeros.

## Reproducibility smoke experiment

Run the infrastructure harness directly:

```sh
python -m lm_idnet.smoke \
  --fixture tests/data/smoke/synthetic_packet_counts.json \
  --config configs/config.json \
  --output /tmp/smoke_report.json
```

The report contains:

- exact aggregated windows;
- named simulation and model-fitting seeds;
- deterministic toy profile parameters;
- one deterministic injected count anomaly;
- a deterministic count-change metric;
- the fixture checksum;
- declared floating-point comparison tolerance.

Tests launch this command in two separate Python processes and require identical
normalized reports, model parameters, injections, and metrics.

The toy profile is intentionally labeled `infrastructure_smoke_experiment` with
`scientific_claim: none`. It checks that data, seeds, subprocess execution, and
reporting connect reproducibly. It is not the verified Dirichlet--Multinomial
estimator, anomaly score, or thesis evaluation; those belong to P2--P5.
