# Simulated gateway measurements

This verifies the user-requested local simulation: **Linux/amd64, one CPU quota,
512 MiB RAM, no network, read-only root filesystem**. Docker is already used by
the project, making this the simplest deployable target without new hardware or
a packet collector. These results do not establish physical gateway or ARM performance.

The host is an AMD Ryzen 7 9700X running AlmaLinux 10.2 and Docker 29.7.2. The
image runs Debian 13 and Python 3.11.17. Exact image ID, package/native-library
hashes, dependency versions, settings, and resource samples are recorded in
[the environment](simulation-environment.json), [short replay](simulation-2000.json),
and [long replay](simulation-20000.json). The measured image is `lm-idnet:edge`.

Measurements used the working tree based on commit `8365870`, before the final
implementation commit. Its dirty-tree status is explicit. The recorded installed
Python-package hash matches the completed source; the native binary and
measurement script have separate hashes. Later documentation/test/report edits
do not change that measured runtime. A new build can label its own source revision.

## Workload and results

The synthetic five-record fixture repeats ordinary counts, silence, missing
coverage, and a large unusual count. Timestamps advance monotonically; missing
windows are not scored. Both trials start from an isolated copy of the supplied
conservative periodic-refit state, using the normal 288-window buffer, update
interval 144, and estimator limit 100,000 iterations. Admission/movement limits
were not loosened. Output passes through the real streaming serializer/flush and
Docker's bounded `local` log driver; the host stdout consumer discards its copy.

| Measurement | 2,000 input windows | 20,000 input windows |
| --- | ---: | ---: |
| Scored windows | 1,600 | 16,000 |
| Replay wall time | 42.82 s | 479.76 s |
| Throughput including refits and synthetic producer | 37.37 windows/s | 33.35 windows/s |
| Mean window processing, including updates | 26.75 ms | 29.97 ms |
| Mean processing without a refit attempt | 0.047 ms | 0.049 ms |
| Worst window processing | 4.449 s | 5.079 s |
| Actual estimator calls | 10 | 110 |
| Worst estimator call | 4.430 s | 5.069 s |
| Peak process RSS (`getrusage`) | 67.88 MiB | 68.62 MiB |
| Highest sampled RSS (`/proc`) | 67.63 MiB | 68.92 MiB |
| Retained detector-state files at end | 6.78 KiB | 31.25 KiB |
| Application logs at end | 10.84 KiB | 109.55 KiB |

Linux RSS counters use slightly different accounting; both are retained rather
than treating them as interchangeable. The long replay's sampled RSS rose from
67.08 to 68.92 MiB and its final two samples were equal. History reached its
100-record cap; the longer run retained one active snapshot and no extra model
versions because no candidate was promoted. This is finite-run evidence,
supported by explicit bounded buffers/retention and forced-rotation checks.

Launching the container through `run-edge.sh` and receiving its first event took
0.595, 0.439, and 0.436 seconds in three separate starts. The benchmark's internal
initialisation clock was 0.157 seconds in the long trial; it excludes Docker and
interpreter launch. Continuous replay used about 99.3% of its one-CPU allowance.
The highest sampled RSS occupied about 13.5% of the 512-MiB limit. Worst observed
window processing used 0.85% of the 600-second window interval and was below the
service's 30-second stop allowance. These are observations, not worst-case guarantees.

The ten-minute collection wait is separate from processing latency. Accelerated
replay is not ten-minute real-time detection. Docker reports the image size as
145,782,693 bytes; CPU/RSS values describe the container process and exclude the
Docker daemon's host resource use.

## Adaptation and operational checks

The long trial made 111 refit attempts: one lacked enough accepted windows and
110 candidate fits did not converge. **No update was promoted.** This synthetic
replay demonstrates guarded rejection and its cost, not useful learned change.
Separate controlled correctness checks exercise accepted/rejected candidates
without relaxing the default movement or admission limits. The historical legacy
experiments remain available for their original labelled traffic comparisons.

The production image starts offline with native LM available and neither gcc nor
pytest installed. Checks compare streaming and batch events/state for raw and
normalised scores, both policies, and all three modes. Signal tests interrupt an
idle reader and request shutdown during atomic activation. ENOSPC tests preserve
the prior pair. Edge-config rollback restores the previous accepted snapshot.

The Compose check supplies windows through a held-open FIFO, injects a failing
input record, and confirms restart into the accepted pair. It then confirms an
explicit stop exits zero and stays stopped. Docker output rotation is forced
with 1-KiB/two-file test limits, and application rotation is checked separately.
The production log limits are 5 MiB/current-plus-two application files and
5 MB/three Docker log files. The long replay retained 16,000 events (11,720,000
payload bytes) within that allowance; this is exported payload size, not total
on-disk size including Docker metadata/compression.

Use [the deployment guide](../../docs/edge-deployment.md) for exact commands.
Physical gateway performance, real-time packet capture, durable event delivery,
and input-cursor persistence are outside this agreed simulation.
