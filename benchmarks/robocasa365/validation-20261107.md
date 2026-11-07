# RoboCasa365 integration validation — 2026-11-07

**Later correction:** multi-slot hard resets exposed an upstream EGL resource
cleanup bug that could corrupt later step images in the first four slots.
Initial captures/first-action comparisons were unaffected, but the old five-slot
smoke score is not a corrected-renderer baseline. See the
[large rendering correctness study](rendering-correctness-20261107.md) and the
source-pinned owning-context cleanup patch in Kit 0.8.0b33.

The simulator, official multi-task Diffusion Policy, SDK lifecycle and real
evaluation loop run in the adaptation container. This is integration readiness
evidence, not a completed 50-task evaluation, a leaderboard score reproduction,
or a measured runtime speedup. Both smoke workloads completed with **zero task
successes**; successful execution must not be confused with task success.

## Revisions and machine

- Kit: `0.8.0b31`, implementation commit
  [`f82276a5879c894af2dc674dbc64df8476b6018e`](https://github.com/suica/RoboScope-Kit/commit/f82276a5879c894af2dc674dbc64df8476b6018e),
  based on upstream `44fa1adb934456c40df809a7baad27152182488d`.
  Its Git tree is `7e62dd7fea3d9776d7cb5f03a7c8363354d730c9`.
- RoboCasa `1.0.1`: `3443cdf3dd20f9ae8ed41adba442059b506891d0`.
- RoboSuite `1.5.2`: `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.
- Simulator: Python `3.11.13`, MuJoCo `3.3.1`, NumPy `2.2.5`, CPU torch `2.5.1+cpu`.
- Model: official `pretrain_human300` epoch-0500 Diffusion Policy; source
  `41212698b6f481ed92a55f0d5f1778ec56bea417`, robomimic fork
  `7dc18461c4fc497095f1044f4c40cd44fff62e6a`.
- Inference: Python `3.11.15`, torch `2.6.0+cu124`, CUDA runtime `12.4`,
  torchvision `0.21.0`, NumPy `1.26.4`, diffusers `0.11.1`, transformers `4.46.3`.
  Direct dependencies and their validated closure are committed in the example.
- Checkpoint and CLIP revisions/hashes are in
  [`model-inputs.json`](../../examples/robocasa365_diffusion/model-inputs.json).
  Both real weight hashes were verified before loading.
- One NVIDIA GeForce RTX 4090, reported memory `49,140 MiB`, driver `580.105.08`;
  container CPU quota 12 cores, memory limit 96 GiB. MPS was disabled.

The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
all 50 tasks, pretrain scenes, the v1.0.1 horizons and native constructor streams.
The older `robocasa/tabletop` GR1 suite was not changed.

## Installed paths

Persistent root: `/autodl-fs/data/robocasa365`.

| Path | Purpose |
|---|---|
| `downloads/` | Six original asset ZIPs and four verified source archives |
| `models/official-dp/` | `model.ckpt` and local `clip/` weights/tokenizer |
| `kit/` | Source export and reproducible build/Policy scripts |
| `runtime`, `policy`, `control` | Links to the working simulator, inference and SDK environments |
| `evidence/` | Real run results, captures and comparison outputs |
| `/workspace/user_policy.py` | Link to the installed official Policy example |
| `/workspace/evostudio-policy.json` | Current Kit process contract, **schema version 2** |
| `/root/startbenchmark.sh` | Delegates to the persistent launcher when no platform entrypoint is supplied |

Working environments and extracted assets are on the container SSD under
`/root/autodl-tmp/robocasa365-20261107`. The shared filesystem initially exhausted
its 200,000-file quota despite free bytes. Only three regenerable old pip/uv
cache directories were removed, freeing about 12 GB and 21,000 inode entries;
RoboTwin assets, model weights and evaluation results were retained. Keeping
the 135,345 ZIP entries on the SSD, with persistent original ZIPs and restoration
scripts, follows the existing benchmark archive deployment pattern.

## Completed checks

| Check | Result |
|---|---|
| Six asset archives | Expected sizes, frozen SHA256 and full ZIP CRC verification passed |
| Fresh simulator build | Complete build from a new prefix, including all six assets and final self-test; 180 s with cached downloads |
| Fresh inference build | New `policy-clean` prefix with the committed dependency lock; isolated real model-class imports passed |
| Real checkpoint load | `/workspace/user_policy.py` loaded verified weights and returned capabilities in 7.68 s |
| `/root/startbenchmark.sh --self-test` | Passed simulator imports and the frozen 50-task/four-suite registry |
| Existing pytest | **188 passed** on Linux; no tests added or changed |
| Ruff | Source, existing tests, RoboTwin scripts and both new RoboCasa365 directories passed |
| Build scripts | `bash -n` and ShellCheck passed |

The source installer explicitly exposes the pinned Diffusion Policy namespace:
its upstream `find_packages()` metadata alone does not install the namespace.
The model requirements include the fork's directly imported `six` dependency.
Both changes are reproduced by the committed installer. The two Lightwheel
ZIPs contain an extra top-level attribution README; the asset lock explicitly
excludes that metadata from the runtime root, preserving the original archives.

## Real simulation results

Both runs used `CloseBlenderLid`, pretrain scenes, full 900-step horizons, eight
actions per inference, the last two consecutive physical observations, FP32,
100 denoising steps, no videos and fixed K=1/G=1. The five-slot run used constructor
seeds `1111111` through `1111115`; rollout labels were `100000` through `100004`.

| Metric | CLI Quick Test | Native five-slot group smoke |
|---|---:|---:|
| Completed / expected episodes | 1 / 1 | 5 / 5 |
| Task successes | **0 / 1** | **0 / 5** |
| Actual steps | 900 | 4,500 |
| Episode termination | Official horizon | Official horizon in all five slots |
| Worker failures / retries / OOM | 0 / 0 / 0 | 0 / 0 / 0 |
| Missing / duplicate records | 0 / 0 | 0 / 0 |
| Policy loading | 9.499 s | 9.274 s |
| Evaluation phase | 129.556 s | 286.788 s |
| SDK runtime elapsed | 131.295 s | 288.459 s |
| Helper end-to-end wall time | Not separately recorded | 297.881 s |
| Sampled mean GPU utilization | 13.71% | 8.08% |
| Peak GPU memory | 2,717 MiB | 6,144 MiB |
| Mean CPU core usage | 1.075 | 1.146 |

Both outputs report `state: complete`, complete coverage and `full_suite: false`.
The standard `failures` counters describe unsuccessful task episodes; the
`recovery.worker_failures` lists and OOM counters are empty. This demonstrates
working evaluation plumbing, not good policy performance. These two workloads
have different native batch sizes and scenario coverage: their timings must not
be used as a before/after acceleration comparison.

## Native first-batch comparison

For **both batch size 1 and batch size 5**, the checker independently loaded the
upstream workspace/EMA model and exercised the hash-verified upstream observation
processing and language encoder on the captured real observations.

- CPU and CUDA model-loader RNG states: exactly equal.
- Three RGB inputs, relative end-effector position/quaternion, gripper state and
  CLIP language embedding: exactly equal.
- Adapted actions versus native actions: exactly equal.
- Actions captured during the real run versus native actions: exactly equal.
- Maximum absolute action difference: **0.0** in both comparisons.

This comparison covers preprocessing and the first action chunk. It does not
compare entire native physics rollouts, all later stochastic actions, all tasks,
or full-suite success rates. `native_equivalence_verified` therefore remains
false in evaluation results. No inference precision, sampler, normalization,
trained weights or action conversion was optimized.

## Worker phase measurements

The single-slot smoke recorded the following inclusive worker timers:

| Phase | Time |
|---|---:|
| Environment construction | 10.214 s |
| Evaluated-episode reset | 6.668 s |
| Policy round trips | 65.053 s |
| Simulation, observations and rendering | 47.020 s |
| Video writes | 0 s |
| Environment close | 0.154 s |

The two large hot-loop categories are Policy round trips and simulation/rendering.
The round-trip timer includes serialization, CPU preprocessing, GPU inference
and waits; it is not isolated GPU execution time. GPU utilization combines
rendering and inference. Nsight was not installed, so these measurements do not
claim a verified `--profile` capture or kernel-level bottleneck attribution.
Independent workers and the five shared episode intervals must not be summed
as elapsed time. Environment preparation and model downloads are excluded from
all evaluation measurements above.

## Reproduction and evidence

In the configured container:

```bash
/root/startbenchmark.sh --self-test --output-dir /results/robocasa365-self-test
/root/startbenchmark.sh --quick-test --output-dir /results/robocasa365-quick

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /autodl-fs/data/robocasa365/control/bin/python \
  /autodl-fs/data/robocasa365/kit/examples/robocasa365_diffusion/run_group_smoke.py \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --policy-config /workspace/evostudio-policy.json \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --output-dir /results/robocasa365-native-group
```

Use new output directories. Quick Test selects K=G=1 itself and rejects explicit
concurrency arguments. The current Kit requires Policy configuration schema 2;
the older adaptation prompt's schema-1 example is incompatible.

Evidence remains outside Git under the persistent root:

- `evidence/quick-test-04/`: results, actual first-batch NPZ and worker records.
- `evidence/group-smoke-01/`: five-slot results and coverage validation.
- `evidence/native-first-batch-04.json` and
  `evidence/native-first-batch-group-01.json`: exact native comparisons.
- Working-prefix `evidence/clean-runtime-proof/` and `clean-runtime-build.log`:
  complete fresh-build pins, source provenance, dependency inventory and self-test.

Raw evidence is also downloaded outside the checkout in the task's local
`evidence/` directory. Weights, assets, caches, raw logs and NPZ captures are not
committed to the repository.

## Remaining acceptance work

The complete target50 matrix (2,500 rollouts), a matched full native evaluator
comparison, repeated fixed-K/G baseline/optimization runs, autotune and the
requested 2x completed-episode throughput target remain **unmeasured**. Before
optimizing, match simulator dependencies and scenario fingerprints, retain the
full horizons and output settings, and compare per-task/per-split successes,
actual steps, coverage and errors. The smoke scores cannot establish unchanged
full-suite accuracy, and no runtime acceleration is claimed by this integration.
