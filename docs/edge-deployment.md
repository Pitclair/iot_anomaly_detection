# Run the simulated gateway

The current target is a local **Linux/amd64 Docker container with one CPU and
512 MiB of RAM**, processing supplied ten-minute count windows. This is the
simulation requested for the thesis. Docker already belongs to this project,
so it needs no additional machine, packet collector, or service framework.
Its limits make resource checks repeatable. They do not reproduce an embedded
processor's speed or power use; physical gateway and ARM validation remain future work.

The [measurement report](../reports/edge/simulation.md) identifies the host,
image, dependencies, configuration, and observed limits. The supplied example
uses conservative periodic refitting; both adaptation policies remain available.

## Build and prepare once

From the project root, with Docker installed:

```sh
docker build --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" --tag lm-idnet:edge .
test ! -e artifacts/edge-state && cp -R artifacts/edge-initial-state artifacts/edge-state
mkdir -p logs/edge
```

Reuse `edge-state` if it already contains the accepted detector. Keep the shipped
initial state unchanged. For different model/settings, follow
[state preparation](edge-detector.md#prepare-state-on-the-development-machine).
The build compiles LM and installs its coefficient files and normal dependencies.
The final image contains neither gcc nor pytest. Startup never compiles or
downloads. Configuration and the root filesystem are read-only; copied state and
logs are writable using your numeric user/group IDs. Networking is disabled.

## Replay and service commands

```sh
./scripts/run-edge.sh < tests/data/edge/windows.jsonl
```

The synthetic, payload-free fixture emits four JSON events, including silence
and an alert; its missing window produces no score. It demonstrates the interface,
not detection accuracy. See the [input/output contract](edge-detector.md#stream-windows-through-the-cli).
The runner uses the already-built image. `LM_IDNET_IMAGE_NAME`,
`LM_IDNET_STATE_DIR`, and `LM_IDNET_LOG_DIR` select another image or host directory.

For an unattended simulation:

```sh
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
docker compose -p lm-idnet-edge -f compose.edge.yaml up --detach
docker compose -p lm-idnet-edge -f compose.edge.yaml logs --no-log-prefix detector
docker compose -p lm-idnet-edge -f compose.edge.yaml ps --all
docker compose -p lm-idnet-edge -f compose.edge.yaml stop
```

The producer is the finite prepared JSON-lines file. Normal EOF exits with zero,
leaving the container stopped. `on-failure:3` retries failures up to three times,
without a replay loop after normal completion. An explicit stop stays stopped.
Use the same Compose command with `start` to replay again. `LM_IDNET_INPUT` can
select another prepared file or a collector-owned FIFO. The collector owns
connection recovery and input positions; packet capture is outside this simulation.

SIGINT/SIGTERM stop an idle reader immediately. During scoring/saving, shutdown
waits for that detector operation to finish, preserving atomic activation. The
service allows 30 seconds before Docker's final kill. Slow output consumers pause
input reads rather than growing a queue. Input/output errors exit nonzero.

## Restart and roll back

Stop any writer first, then restore the previous accepted pair and start again:

```sh
docker compose -p lm-idnet-edge -f compose.edge.yaml stop
./scripts/run-edge.sh rollback --config /app/configs/edge_example.json --state-dir /state
docker compose -p lm-idnet-edge -f compose.edge.yaml start
```

Rollback requires a previous accepted version and refuses a locked directory.
Its CLI accepts either the edge or research configuration. Restart restores the
accepted model, matching threshold/IQR, original safety references, and count
limit without research captures. Learning buffers and counters reset.

Saved state does not include an input cursor or guarantee event delivery. A stop
received during an update can leave it committed before its event is delivered.
A retry of a file replays already processed windows and can learn from them again.
Use a fresh state copy for an independent experiment. Durable acknowledgements
and replay prevention are outside this simulation.

## Storage limits and failures

| Data | Retention |
| --- | --- |
| Application `/logs/runtime.log` | 5 MiB per file, current plus two backups |
| Docker events and stderr (`local` driver) | 5 MB per file, at most three files, plus metadata |
| Accepted state | Active/previous snapshots; last 100 audit records |
| Temporary files | 16 MiB tmpfs |

Docker's bounded log store is also the unattended event destination. Removing
the container removes those events. Export a finite replay for a thesis report;
do not append an unlimited stream to a file. The image remains fixed at runtime.

A failed state write before activation rejects the update and keeps the valid
pair. A failure after pointer activation stops the process so restart validates
the committed pair. Audit/log writes are best effort and can be missing after
storage failures. Docker storage failures can also lose event delivery: inspect
daemon errors and recover space before restarting. These limits bound retention,
without promising durable delivery.

## Verify and measure

```sh
python -m pytest -q
LM_IDNET_EDGE_IMAGE=lm-idnet:edge python -m pytest -q tests/test_edge_deployment.py
```

The Docker checks use the prebuilt image without downloads. They verify offline
native loading, state across container removal, failure restart with a held-open
FIFO, explicit stop, resource limits, and forced output rotation. Other tests
cover raw/normalised parity, both policies, controlled accepted/rejected updates,
ENOSPC, signal deferral, malformed input, and rollback.

For an isolated resource measurement:

```sh
trial_dir=$(mktemp -d)
cp -R artifacts/edge-initial-state "$trial_dir/state"
mkdir -p "$trial_dir/logs" reports/edge
docker run --rm --pull never --network none --read-only \
  --memory 512m --cpus 1 --pids-limit 64 \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m --user "$(id -u):$(id -g)" \
  --log-driver local --log-opt max-size=5m --log-opt max-file=3 \
  --env LM_IDNET_LOG_PATH=/logs/runtime.log \
  --volume "$PWD/configs:/app/configs:ro" \
  --volume "$PWD/scripts/benchmark_edge.py:/measure.py:ro" \
  --volume "$PWD/tests/data/edge/windows.jsonl:/fixture.jsonl:ro" \
  --volume "$trial_dir/state:/state" --volume "$trial_dir/logs:/logs" \
  --volume "$PWD/reports/edge:/reports" --entrypoint python lm-idnet:edge \
  /measure.py --config /app/configs/edge_example.json --state-dir /state \
  --fixture /fixture.jsonl --windows 20000 --output /reports/replay.json > /dev/null
```

Inspect `reports/edge/replay.json`, then remove the temporary `trial_dir` you
created. Use a fresh state copy per trial. The script advances timestamps and
keeps counters and at most ten resource samples. It measures line availability
through serialization/output flush, actual estimator calls, CPU, peak RSS, and
state/log sizes. Total throughput also includes synthetic producer work.
It runs as fast as possible; waiting ten minutes to collect a window is a
separate delay. Failed/rejected refits still contribute their full processing cost.

Synthetic performance and correctness checks are separate from labelled traffic
results. All conservative candidates can be rejected in this replay; that does
not demonstrate useful adaptation. The legacy thesis experiments remain
available through their existing explicit policy/configuration.
