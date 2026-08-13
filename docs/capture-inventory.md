# Capture inventory

The capture inventory is the first data-reproducibility check. It compares the
capture identifiers required by configuration with the files actually present
under `ingest.raw_root/ingest.dataset_folder`.

Run it without preprocessing any packets:

```sh
lm-idnet preprocess \
  --config configs/d_link_day_cam5.json \
  --inventory-only
```

The default report is
`reports/D-LinkDayCam5/capture_inventory.json`. Override it with:

```sh
lm-idnet preprocess \
  --config configs/d_link_day_cam5.json \
  --inventory-only \
  --inventory-output reports/D-LinkDayCam5/another_inventory.json
```

## Information recorded

Every configured capture receives an entry, including missing files. Each entry
contains:

- configured capture identifier;
- date extracted from the identifier, where available;
- expected filesystem path;
- whether the path is a present regular file;
- byte size for present readable files;
- SHA-256 checksum;
- packet count for complete classic-PCAP files;
- parser status and an explanatory message where parsing failed.

The classic-PCAP counter streams records and does not retain packet payloads.
PCAPNG and unknown formats are currently recorded as `unsupported_format`, not
silently skipped. Complete support and deeper timestamp/corruption validation
belong to the subsequent capture-integrity task.

## Acceptance behavior

The report is always written before acceptance is enforced. The command fails
with a data-validation exit code when:

- any configured required capture is missing; or
- two or more captures have the same SHA-256 checksum without an explicit
  explanation.

This fail-after-report behavior leaves evidence describing why the gate failed.

## Explaining an intentional duplicate

Duplicates are rejected by default. A legitimate duplicate must be documented
in the configuration using the exact set of capture identifiers and a reason:

```json
{
  "ingest": {
    "allowed_duplicate_captures": [
      {
        "capture_ids": [
          "camera-2020-10-08",
          "camera-2020-10-09"
        ],
        "reason": "Provider supplied the same capture under two date labels"
      }
    ]
  }
}
```

The configuration rejects unknown capture identifiers, repeated identifiers,
blank reasons, and capture identifiers appearing in multiple explanation
groups. An explanation does not hide the duplicate: the report still records
the shared checksum, affected identifiers, and reason.
