# UNSW Chromecast dataset integration

This document is the living preparation record and task checklist for adding the
Chromecast traces from the UNSW IoT Analytics Attack Traces dataset to LM-IDNet.
Update it whenever a preparation or integration decision changes.

## Source and selection

- Source: [UNSW IoT Analytics Attack Traces](https://iotanalytics.unsw.edu.au/attack-data.html)
- License stated by the publisher: MIT-0
- Source format: daily mixed-device PCAPNG captures, attack annotations, and
  `attackinfo.xlsx` device metadata
- Selected device: Chromecast
- Ethernet MAC address: `f4:f5:d8:8f:0a:3c`
- IPv4 address recorded in `attackinfo.xlsx`: `192.168.1.119`
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

## Frozen temporal partition plan

| Partition | Capture dates |
| --- | --- |
| Fit | 2018-10-10 through 2018-10-15 |
| Calibration | 2018-10-16 through 2018-10-19 |
| Development test | 2018-10-20 through 2018-10-23 |
| Final test | 2018-10-25 through 2018-10-27 |

## Integration checklist

- [x] Identify the available October devices and select Chromecast.
- [x] Record the dataset source, device addresses, preparation rules, and partition plan.
- [x] Remove the excluded October 24 staged capture.
- [x] Rename all retained staged captures to `YYYY-MM-DD.pcap`.
- [x] Update the downloader so a rerun reproduces the retained ISO-named files.
- [x] Identify and validate the D-Link camera MAC and IPv4 addresses.
- [x] Add optional MAC and IPv4 ingest settings used for packet filtering.
- [x] Filter Ethernet, IPv4, and relevant ARP traffic for the configured device.
- [x] Add focused filtering tests, including broadcast ARP addressed to Chromecast.
- [x] Create the `UNSW-Chromecast` prepared raw-data view without duplicating PCAP data.
- [x] Add an isolated `configs/unsw_chromecast.json` configuration and namespaced outputs.
- [ ] Validate and fingerprint all configured source captures.
- [ ] Preprocess the retained captures and review per-capture coverage and counts.
- [ ] Normalize the Chromecast attack annotations into separate window-label JSON files.
- [ ] Train on the fit partition and calibrate on the calibration partition.
- [ ] Score and evaluate the development-test partition.
- [ ] Record any development-driven decisions and freeze the experiment.
- [ ] Score and evaluate the locked final-test partition.

## Preparation log

| Date (UTC) | Change |
| --- | --- |
| 2026-08-13 | Created this record; selected Chromecast; excluded October 24; froze the initial partition plan. |
| 2026-08-13 | Removed the staged October 24 file, renamed 17 retained captures to ISO dates, and updated the downloader to reproduce that selection. |
| 2026-08-13 | Identified D-Link MAC `b0:c5:54:42:8f:88`; its configured captures use IPv4 `192.170.11.211` and later `192.170.11.210`, requiring an IP list. |
| 2026-08-13 | Added address-based filtering, its tests and fingerprint policy, the isolated Chromecast config, and a 17-link prepared raw-data view. |
