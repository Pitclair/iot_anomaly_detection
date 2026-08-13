# Dataset fingerprints

The dataset fingerprint identifies both the configured source bytes and the
policy that turns those bytes into modeling features.

Generate it with:

```sh
lm-idnet preprocess \
  --config configs/d_link_day_cam5.json \
  --fingerprint-only
```

The default output is `reports/D-LinkDayCam5/dataset_fingerprint.json`. Use
`--fingerprint-output <path>` to override it.

## Hashed content

The version identifier covers:

- the ordered capture list;
- each capture's SHA-256 checksum and byte size;
- each capture's frozen partition assignment;
- the ordered traffic-category taxonomy;
- configured timestamp and protocol column names;
- window duration;
- half-open window interval policy `[start, end)`;
- UTC timestamp policy;
- the complete fit/calibration/development/final partition definition.

Absolute repository paths are excluded. Files are represented relative to the
dataset directory so an unchanged clone produces the same identifier on a
different machine.

The manifest contains a component hash for files, taxonomy, window policy, and
partitions, plus one overall identifier:

```text
sha256:<64 hexadecimal characters>
```

This makes it possible to distinguish whether a change came from source bytes,
category semantics, windowing, or experimental partitioning.

## Reproducibility and safety

The manifest contains no generation timestamp, so unchanged inputs produce
byte-identical JSON. Hashing is streamed with bounded memory. File size and
modification time are checked before and after hashing; a source that changes
during the operation is rejected.

Missing or unreadable configured captures also fail rather than producing a
partial dataset identity.

A fingerprint identifies data and policy. It does not assert that preprocessing
is scientifically correct; the later processing acceptance gates establish
that separately.
