# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 6,000 inspected frames, 374 training images, two model
versions. Client and server on the same machine (arm64, 12 cores). Reproduce with `make demo`, then `make load-test` (the quality read);
the defects read was run by hand with `core.loadtest` against the first run's id, which changes with every demo.

## The edge: inspection time per frame, measured inside the stream job

Every frame of the demo's stream is timed around `Model.inspect` (patch extraction, PCA, nearest-neighbour search against the memory
bank, and on a reject the type classifier, the box and the confidence). The budget at 30 FPS is 33.3 ms a frame.

| Stretch | Frames | p50 | p95 | p99 | Worst | Over 33.3 ms | Edge busy at 30 FPS | Capacity |
|---|---|---|---|---|---|---|---|---|
| First 100 s, version 1 (38% rejected) | 3,000 | 1.02 ms | 11.16 ms | 18.39 ms | 97.93 ms | 9 | 12.7% | about 236 FPS |
| Second 100 s, versions 1 and 2 (60% rejected) | 3,000 | 7.42 ms | 10.14 ms | 17.18 ms | 87.29 ms | 16 | 16.8% | about 179 FPS |

A pass costs about 1 ms; a reject about 7 ms, almost all of it the 200-tree random forest naming the defect type. The p50 moves with
the reject rate for that reason. 9 and 16 frames went over the budget, with spikes to 98 ms that look like garbage collection or the
container being scheduled out. A real edge would need a frame buffer of a few frames, or the classifier moved off the critical path
(decide pass/reject first, name the type afterwards). Held-out lines (`python -m qi.evaluate`, outside Docker, 2,700 frames): p50 0.69 ms,
p99 7.77 ms.

The capacity column is frames over busy time on one core. It is not a claim about any real edge device: 64 × 64 grayscale frames are far
smaller than a line camera's.

## Reads: line quality by stretch and model, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 2504 | 250.4 req/s | 3.3 ms | 7.6 ms | 12.2 ms | 0.00% |
| 8 | 2509 | 250.6 req/s | 28.1 ms | 51.2 ms | 77.7 ms | 0.00% |
| 32 | 2350 | 233.2 req/s | 111.8 ms | 153.8 ms | 204.6 ms | 0.00% |

## Reads: the twelve most anomalous rejects of a run, with images and anomaly maps

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 537 | 53.6 req/s | 15.1 ms | 38.4 ms | 58.3 ms | 0.00% |
| 2 | 830 | 83.0 req/s | 20.3 ms | 39.6 ms | 68.8 ms | 0.00% |
| 8 | 1358 | 135.2 req/s | 55.6 ms | 77.4 ms | 88.2 ms | 0.00% |

## Compute (one core)

| Work | Time |
|---|---|
| Render a 64 × 64 frame | about 0.2 ms |
| Train a version: PCA, memory bank (9 shifts × 200 boards, greedy k-centre to 320 per position), threshold, classifier | 4–7 s |
| The demo end to end | about 1 minute |
| Evaluation: nine held-out lines and the supplier-change study | about 4 minutes |

Model artifacts are 7.5 MB (version 1) and 10.5 MB (version 2), mostly the memory bank: 225 positions × 320 (or 448) vectors × 25 floats.

Not measured: real camera resolution (a 2,048 × 2,048 frame has about 1,000 times as many patches), a GPU, or more than one camera.
