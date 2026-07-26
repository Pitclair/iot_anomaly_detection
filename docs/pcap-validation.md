# PCAP integrity validation

Capture validation is separate from packet classification and preprocessing. It
checks whether every configured source can be parsed completely and trusted as
structurally valid input.

Run:

```sh
lm-idnet preprocess \
  --config configs/config.json \
  --validate-captures-only
```

The default report is `reports/capture_validation.json`. Use
`--validation-output <path>` to select another location.

## Validation performed

The validator streams classic PCAP files record by record and checks:

- complete 24-byte global header;
- recognized microsecond or nanosecond PCAP byte order;
- supported PCAP version 2.4;
- positive snapshot length;
- complete 16-byte packet headers;
- timestamp fractions within the declared resolution;
- captured lengths that do not exceed snapshot or original lengths;
- complete packet data matching each declared captured length;
- successful parsing through the physical end of the file.

Packet payloads are read only to advance and verify record boundaries; they are
not retained in the validation report.

## Recoverable warnings

These conditions do not prevent a complete parse:

- `out_of_order_timestamp`: a record is earlier than its predecessor;
- `packet_snaplen_truncation`: fewer packet bytes were stored than existed on
  the original wire;
- `zero_length_packet`: a record stores no packet bytes.

Warnings are aggregated by code. The report stores the total occurrence count
and up to ten example record numbers, keeping report size bounded.

Captures with warnings have status `warning`, `complete_to_eof: true`, and are
accepted by this structural gate. Later temporal and data-quality tasks decide
whether particular warnings require exclusion.

## Fatal corruption

Fatal conditions include:

- missing or unreadable files;
- unsupported formats or PCAP versions;
- incomplete global, record, or packet data;
- invalid timestamp fractions;
- impossible captured/original/snapshot length relationships.

The first fatal defect records its code, message, byte offset, and packet record
number where applicable. The packet count then represents only complete records
before the defect and is not treated as a successful source count.

The JSON report is written atomically before the command exits with a
data-validation error. This preserves evidence for rejected captures.

## Reproducibility

Each report includes a `validation_fingerprint` calculated from capture results
and the summary, excluding the run timestamp. Revalidating unchanged sources
must produce identical:

- per-capture packet counts;
- warnings and fatal classifications;
- summary counts;
- validation fingerprint.

## Quarantined regression fixture

`tests/data/quarantine/truncated_capture.hex` describes a deliberately
truncated classic-PCAP fixture. The test materializes it outside normal source
data, copies it into an isolated quarantine directory, and verifies the fatal
classification `truncated_packet_data`. It is never processed as benign input.

PCAPNG is currently classified as `unsupported_format`. Adding PCAPNG support
requires a separate complete parser or a pinned external validation tool; it
must not be silently interpreted as classic PCAP.
