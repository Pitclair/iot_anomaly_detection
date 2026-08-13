# UNSW Chromecast dataset integration

This document is the living preparation record and task checklist for adding the
Chromecast traces from the UNSW IoT Analytics Attack Traces dataset to LM-IDNet.
Update it whenever a preparation or integration decision changes.

## Source and selection

- Source: [UNSW IoT Analytics Attack Traces](https://iotanalytics.unsw.edu.au/attack-data.html)
- License stated by the publisher: MIT-0
- Source format: daily mixed-device PCAPNG captures, attack annotations, and
  `attackinfo.xlsx` device metadata
- Chromecast annotation source: the publisher's
  [`annotations.zip`](https://iotanalytics.unsw.edu.au/anomaly-data/annotations.zip),
  file `f4f5d88f0a3c.csv`
- Chromecast annotation SHA-256:
  `e9b4d9d5374b8c6dcad2ac8e5947cfa44079d0cc35bda8e46bd1be16822d8655`
- Selected device: Chromecast
- Ethernet MAC address: `f4:f5:d8:8f:0a:3c`
- Selected benign captures: October 10–19, 2018
- Selected mixed attack-and-benign captures: October 20–23 and 25–27, 2018

The PCAPs contain traffic from several devices. They remain unmodified as source
data; device filtering is performed when preparing LM-IDNet count windows.
Ground-truth attack annotations are used only during evaluation.

## Preparation decisions

1. Downloaded captures are staged under `unsw_iot_attack_pcaps/benign` and
   `unsw_iot_attack_pcaps/mixed`.
2. Staged filenames use ISO dates (`YYYY-MM-DD.pcap`) before any prepared raw-data
   view is created.
3. October 24 is excluded because the publisher's file contains a truncated final
   PCAPNG block. Redownloading does not repair it: the local MD5 matched the
   publisher's ETag (`1c013392187b045a87e2774fbd6f7098`).
4. The planned prepared dataset name is `UNSW-Chromecast` so both the source and
   selected device remain visible.
5. The existing ten-minute, UTC, epoch-aligned, half-open window policy is reused.
6. Processed count windows and evaluation labels remain separate artifacts.
7. Label capture membership and window bounds come from packet timestamps, not
   the nominal PCAP filename date. A window is positive when an annotation
   satisfies `attack_start < window_end and attack_end > window_start`.
8. `attack_overlap_seconds` is the union of all annotated overlap within the
   window, so overlapping annotations cannot produce more than 600 seconds.
9. `device_id` remains the human-readable artifact identity. Packet selection
   uses only `device_mac`; IP addresses are deliberately excluded because they
   may change across captures.

## Label preparation result

The official Chromecast file contains 27 attack intervals. The retained captures
match 21 intervals and produce 2,349 labeled windows, of which 38 are positive:

| Partition/capture | Positive windows |
| --- | ---: |
| Development test, `2018-10-23` | 32 |
| Final test, `2018-10-25` | 6 |
| All other retained captures | 0 |

The six unmatched annotations are three SSDP and three TCP SYN-reflection
intervals from the deliberately removed October 24 capture. The October 21 source
capture is structurally readable but ends after approximately seven hours; its
label file therefore contains 42 windows rather than a full day. Evaluation must
use only windows for which both a score and label exist.

## Frozen temporal partition plan

| Partition | Capture dates |
| --- | --- |
| Fit | 2018-10-10 through 2018-10-15 |
| Calibration | 2018-10-16 through 2018-10-19 |
| Development test | 2018-10-20 through 2018-10-23 |
| Final test | 2018-10-25 through 2018-10-27 |

## Integration checklist

- [x] Identify the available October devices and select Chromecast.
- [x] Record the dataset source, device identity, preparation rules, and partition plan.
- [x] Remove the excluded October 24 staged capture.
- [x] Rename all retained staged captures to `YYYY-MM-DD.pcap`.
- [x] Update the downloader so a rerun reproduces the retained ISO-named files.
- [x] Identify and validate the D-Link camera MAC address.
- [x] Add required device-ID and MAC ingest settings.
- [x] Filter Ethernet and ARP hardware addresses for the configured device MAC.
- [x] Add focused MAC-filtering tests, including ARP hardware-address matching.
- [x] Create the `UNSW-Chromecast` prepared raw-data view without duplicating PCAP data.
- [x] Add an isolated `configs/unsw_chromecast.json` configuration and namespaced outputs.
- [ ] Validate and fingerprint all configured source captures.
- [ ] Preprocess the retained captures and review per-capture coverage and counts.
- [x] Normalize the Chromecast attack annotations into separate window-label JSON files.
- [ ] Train on the fit partition and calibrate on the calibration partition.
- [ ] Score and evaluate the development-test partition.
- [ ] Record any development-driven decisions and freeze the experiment.
- [ ] Score and evaluate the locked final-test partition.

## Preparation log

| Date (UTC) | Change |
| --- | --- |
| 2026-08-13 | Created this record; selected Chromecast; excluded October 24; froze the initial partition plan. |
| 2026-08-13 | Removed the staged October 24 file, renamed 17 retained captures to ISO dates, and updated the downloader to reproduce that selection. |
| 2026-08-13 | Identified D-Link MAC `b0:c5:54:42:8f:88`. |
| 2026-08-13 | Added MAC-based filtering, its tests and fingerprint policy, the isolated Chromecast config, and a 17-link prepared raw-data view. |
| 2026-08-13 | Preserved the official Chromecast CSV and generated 17 separate label files plus a provenance manifest: 2,349 windows, 38 positive, 21/27 intervals retained. |
| 2026-08-13 | Retained human-readable device names and removed IP selectors; device traffic is now selected only by its stable MAC address. |
| 2026-08-13 | Restored the `device_id` configuration name for backward compatibility. |
