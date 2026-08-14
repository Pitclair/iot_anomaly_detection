# UNSW IoT attack PCAP preparation

Run `./download_pcaps.sh` for Chromecast or
`./download_pcaps.sh samsung-camera` for Samsung Camera. Downloads resume when
rerun. Captures are separated into `benign/` and `mixed/` (the publisher's
attack-and-benign traces), and staged filenames use ISO dates. Chromecast's
October 24 capture is intentionally excluded.

These source PCAPs contain multiple devices. Each selected device is filtered by
its MAC address during project preprocessing. See the dataset records for
[Chromecast](../docs/unsw-chromecast-dataset.md) and
[Samsung Camera](../docs/unsw-samsung-camera-dataset.md).

Run `./prepare_dataset.sh` after downloading to create the flat
`data/raw/UNSW-Chromecast` input view using symbolic links. No PCAP bytes are
duplicated. Pass `samsung-camera` to create `data/raw/UNSW-Samsung-Camera`.

Run `./download_annotations.sh`, then
`python prepare_labels.py` from the repository root to preserve the official
Chromecast annotations and generate separate ten-minute label JSON files.
For Samsung Camera, pass its config, annotation, and label output explicitly:

```sh
python unsw_iot_attack_pcaps/prepare_labels.py \
  --config configs/unsw_samsung_camera.json \
  --annotations unsw_iot_attack_pcaps/annotations/00166cab6b88.csv \
  --output data/labels/UNSW-Samsung-Camera
```
