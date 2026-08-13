# UNSW IoT attack PCAP preparation

Run `./download_pcaps.sh` to download the selected October daily traces into
`benign/` and `mixed/`. Downloads resume when rerun. Staged filenames use ISO
dates; October 24 is intentionally excluded.

These source PCAPs contain multiple devices. The selected Chromecast is filtered
by its MAC/IP addresses during project preprocessing. See
[`docs/unsw-chromecast-dataset.md`](../docs/unsw-chromecast-dataset.md) for the
preparation record and integration checklist.

Run `./prepare_dataset.sh` after downloading to create the flat
`data/raw/UNSW-Chromecast` input view using symbolic links. No PCAP bytes are
duplicated.

Run `./download_annotations.sh`, then
`python prepare_labels.py` from the repository root to preserve the official
Chromecast annotations and generate separate ten-minute label JSON files.
