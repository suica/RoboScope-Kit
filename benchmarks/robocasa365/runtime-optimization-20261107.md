# RoboCasa365 small-scale runtime study — 2026-11-07

This study uses the integration machine, source/checkpoint pins and dependencies
recorded in [integration validation](validation-20261107.md), with Kit 0.8.0b32.
Trials retain native five-slot batches, pretrain scenes, constructor streams,
FP32, 100 denoising steps, eight-action chunks, full task horizons and K=1/G=1.
They are partial evaluations, not official 2,500-rollout scores.

## Task selection

The official DP submission publishes split averages rather than per-task rates.
Its v1.0.0 results cannot identify a 20–80% task in this v1.0.1 environment.
We screened five native slots per task without shortening task horizons.

| Task | Official horizon | Successes | Physical steps |
|---|---:|---:|---:|
| CloseToasterOvenDoor | 450 | 0/5 | 2,250 |
| TurnOnMicrowave | 450 | 0/5 | 2,250 |
| **TurnOnElectricKettle** | **450** | **1/5 (20%)** | **2,250** |

Use TurnOnElectricKettle for subsequent small experiments. The 20% value is a
five-episode sample rate after task screening, not an unbiased population-rate
estimate or a full-suite score. Its reference run had complete unique coverage,
no worker failures, OOMs or retries, and 57 native batch-five inference calls.

## Detailed baseline attribution

The baseline group was instrumented for diagnosis. Model loading took 14.438 s,
runtime preparation 0.944 s, evaluation 171.130 s and helper end-to-end 188.806 s.
The worker task took 170.888 s. The following top-level worker phases are sequential
within this single worker; percentages use its task time.

| Phase | Calls / work | Wall time | Fraction |
|---|---|---:|---:|
| Environment construction | Five constructor streams, including native warm resets | 25.448 s | 14.9% |
| Evaluated reset | Five unseeded continuation resets | 26.436 s | 15.5% |
| Simulation, controllers, observations and rendering | 2,250 physical steps | 79.347 s | 46.4% |
| Policy round trips | 57 batches of five slots | 38.764 s | 22.7% |
| Observation history assembly | 57 batches | 0.059 s | 0.03% |
| Action conversion | 2,250 actions | 0.0066 s | 0.004% |
| Policy reset/end calls | Five episodes | 0.0016 s | 0.001% |
| First-batch/audit evidence writes | NPZ and per-batch hashes | 0.202 s | 0.12% |
| Audit computations | Complete inputs/actions/physical transitions | 0.277 s | 0.16% |
| Environment close | Five slots | 0.308 s | 0.18% |
| Video encoding | Disabled in both variants | 0 s | 0% |

Camera render API wall time was **3.922 s over 6,750 calls**, nested inside the
79.347 s step phase. It is only **2.3% of evaluation time**. The residual step
host time is 75.425 s, including physics, controllers, state updates, success
checks and observation conversion; it is not isolated pure-physics time.

The separate Policy diagnostic wrapper measured:

| Nested server phase | Count | Wall/event time |
|---|---:|---:|
| All `act_batch` calls | 57 | 35.571 s |
| First batch | 1 | 1.413 s |
| Later batches | 56 | 0.610 s mean; 0.606 s median |
| Observation preprocessing | 285 observations | 1.043 s |
| CPU CLIP on language-cache misses | Five observations | 0.600 s |
| Cached-language preprocessing | 280 observations | 0.443 s |
| `predict_action` Python wall | 57 | 34.171 s |
| Prediction CUDA stream event intervals | 57, all resolved | 34.172 s |

The event intervals include GPU stream idle gaps caused by host dispatch and
overlap Python execution. They are not summed GPU kernel execution time and
cannot be added to server or RPC wall time. The 3.193 s RPC-minus-server residual
also includes protocol work, transfers and waits; it is not isolated serialization.
No additional CUDA synchronization, warmup, inference, RNG or precision changes
were introduced by the diagnostic wrapper.

Mean sampled GPU utilization was 7.44%, peak VRAM 6,233 MiB and mean CPU usage
1.078 cores. These aggregate samples do not establish a kernel-level bottleneck.

## Current acceleration experiment

The optional consumed-frame mode reduces intermediate RGB rendering. Baseline
rendering updates three cameras on every control step, while inference consumes
the last two consecutive physical frames at each eight-action chunk boundary.
The patch returns the native cached camera sample on discarded steps while
leaving `Observable.update` and its sampling clock intact. Disabling/re-enabling
an observable would reset its sampling phase and is deliberately avoided.

All physical substeps, controller updates, state observations, native success
checks and the group termination barrier remain. Required frames retain native
sampling, including partial final chunks and odd-step video frames. Unsupported
camera processing is rejected. The committed builder applies reviewed upstream,
patch and resulting-file hashes; unrecognized source modifications are rejected.

| Diagnostic comparison | Baseline: all frames | Candidate: consumed frames |
|---|---:|---:|
| Successes / episodes | 1/5 | 1/5 |
| Physical steps | 2,250 | 2,250 |
| Real camera calls | 6,750 | 1,710 |
| Reused camera samples | 0 | 5,040 |
| Camera API wall time | 3.922 s | 1.078 s |
| Evaluation wall time | 171.130 s | 171.737 s |

This is **not a valid speedup pair**. Initial physics, metadata and observations
already differed in slots 1 and 3 before the first action, although constructor
seeds and reset counts matched. Slots 0, 2 and 4 had identical initial features
and actions; only slot 4 retained identical complete rollout digests. The full
comparison rejected 244 checks and intentionally emitted no throughput ratio.
Equal aggregate success counts alone do not prove unchanged behavior.

Baseline `all` rendering remains the default. No acceleration benefit or unchanged
full-suite accuracy is claimed. Given the small render-time share, this experiment
is not expanded into a large timing sweep; investigate native step host cost and
scene repeatability before another acceleration change.

## Small-scale iteration

Use one full-horizon native group per variant first. Compare initial scenarios,
all consumed features/actions, every physical state/control/success transition,
terminal steps, and complete unique records. Repeat timing runs with audit and
diagnostic tools disabled only after these checks pass. Expand seeds/tasks later,
then retune K/G separately from the fixed-K/G runtime result.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /autodl-fs/data/robocasa365/control/bin/python \
  /root/autodl-tmp/robocasa365-20261107/kit/examples/robocasa365_diffusion/run_group_smoke.py \
  --task TurnOnElectricKettle --groups 1 --render-mode all \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --policy-config /workspace/evostudio-policy.json \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --output-dir /results/kettle-small-baseline
```

Raw logs, captures, diagnostic programs and comparison outputs stay outside Git
under `/autodl-fs/data/robocasa365/evidence` and in the task's local evidence folder:
`screen-TurnOnElectricKettle-01`, `audit-kettle-consumed-01`,
`policy-diagnostic-30844.json` and `kettle-render-audit-comparison.json`.
