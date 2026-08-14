# UNSW Samsung Camera dataset integration

- Source: [UNSW IoT Analytics Attack Traces](https://iotanalytics.unsw.edu.au/attack-data.html)
- Device: Samsung Camera
- Ethernet MAC address: `00:16:6c:ab:6b:88`
- Annotation: `annotations/00166cab6b88.csv`
- Annotation SHA-256: `618bb6637c8d388aacc7091d336343a7ac3b2d6af2d07b311c0686bbcb6a450e`
- Benign captures: May 28–31, 2018
- Attack-and-benign captures: June 1–8, 2018

The source PCAPs contain several devices. Samsung Camera traffic is selected by
MAC address during preprocessing, and the source bytes remain under
`unsw_iot_attack_pcaps/Samsung camera/`. The flat
`data/raw/UNSW-Samsung-Camera` view contains symbolic links, so captures are not
duplicated.

| Partition | Captures |
| --- | --- |
| Fit | May 28–30 |
| Calibration | May 31 |
| Development test | June 1–4 |
| Final test | June 5–8 |

The 42 official attack intervals all match retained captures. Label preparation
produces 1,645 ten-minute windows, including 82 attack-positive windows. May 31
contains 81 device windows and June 1 contains 124; every other retained capture
contains 144.
