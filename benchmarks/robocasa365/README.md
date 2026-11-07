# RoboCasa365 runtime

This runtime is independent of `benchmarks/robocasa`, which owns GR1 Tabletop.
The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
the 50-task multi-task workload in **pretrain scenes**, with 50 rollouts per
task (2,500 records), updated task horizons and native five-slot groups.

## Build and run

Use Linux x86_64, Python 3.11, uv 0.11.14, Git, NVIDIA EGL libraries and a CUDA
GPU. Install the [official Diffusion Policy example](../../examples/robocasa365_diffusion/README.md)
in its separate inference environment. The simulator uses CPU torch for
imported playback utilities and excludes interactive device and training tools.
Allow space for six asset ZIPs (about 10.99 GB), their extracted files,
source checkouts, environments and model weights. Also check the filesystem's
**file count quota** before extraction.

```bash
python -m pip install ./packages/roboscope-kit
KIT_PYTHON="$(command -v python)" \
  ROBOSCOPE_ROBOCASA365_ASSETS_DIR=/autodl-fs/data/robocasa365/downloads \
  bash benchmarks/robocasa365/build-runtime.sh /autodl-fs/data/robocasa365/runtime
```

`ROBOSCOPE_ROBOCASA365_PYTHON` can select an existing absolute Python 3.11
executable instead of downloading one; it must remain available after building.
`ROBOSCOPE_PYPI_INDEX` selects a reachable package mirror without changing pins.
`--environment-only` prepares imports without installing a manifest or claiming
a usable asset-backed runtime. Normal builds verify all asset hashes and finish
with the Kit self-test. Downloads use GET: Box may reject HEAD for valid links.
An existing verified ZIP cache can be transferred from another machine.

Both build profiles include an owning-EGL-context cleanup fix: destroying an old
renderer must bind its own GL context before freeing context-local GPU resources.
Without that bind, sequential resets of five environments in one process can
destroy another slot's framebuffers and produce corrupted observations.
Marker-2 cleanup also restores the interrupted EGL API/context/surfaces after
delayed destruction. Per-context cleanup no longer releases another renderer's
thread binding. Both profiles include these corrections.

For strict rendering experiments, build a **separate** prefix with
`--paired-validation`. This adds stable Counter reset-region deduplication and
forwards the intended RNG to stack-base fixtures, equally in both experiment
arms. It retains candidate membership and does not add environment RNG draws,
but stable ordering changes the previous identity-dependent seed-to-scene map.
The generated manifest declares `paired-stable-regions-v1`, and results are
always `full_suite: false`; this is a common validation setup, not an official
target50 score. Cross-profile reuse of an existing prefix is rejected.

**Consumed-frame rendering is experimental; `all` remains the default.** The
registered full 50-pair trial completed with 14/50 successes (28%) in each arm,
zero discordant successes, and identical steps/termination after all physical
reset/provenance checks passed. This is no observed task-success change in that
sample, not an absolute or full-suite accuracy guarantee. Secondary input/action/
physics audits still failed, and a multi-group `all`/`all` control also diverged;
optimization causality and speedup remain unproven. The
[study](rendering-correctness-20261107.md) retains the earlier failed/incomplete
attempts and record-loss limits alongside the registered result.

Configure the example with absolute interpreter and source paths, and a model
directory containing `model.ckpt` and `clip/`. In an adaptation container the
configuration can be `/workspace/evostudio-policy.json`.

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --self-test --output-dir /results/robocasa365-self-test

python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 --suite target50 \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --policy-config /workspace/evostudio-policy.json \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --quick-test --output-dir /results/robocasa365-quick-test
```

The self-test verifies imports and the frozen registry, without creating an
environment. The real Quick Test runs one complete episode of the first task,
retaining its official horizon. It uses one slot and is a smoke test.
For the full matrix remove `--quick-test` and set `--concurrency K --replicas G`.
Each formal worker manages **five** native environments. Use fresh output
directories, verify all records and compare with the pinned native evaluator.
The Policy README provides a captured-input native action comparison helper.

## Performance evidence

Establish the unoptimized Kit baseline first. Keep checkpoint, precision,
task matrix, scenario streams, horizons, native slot count, K/G and output
settings fixed for paired comparisons. Report formal completed episodes/h and
actual steps/s separately from preparation, loading, tuning, cold start and
end-to-end time. Repeat timing runs three times when practical.

Worker results separate construction, resets, simulation/observation, Policy
round trips, video writes and cleanup. These inclusive timings overlap between
workers; do not add them, or the five shared group intervals, as wall time.
`--profile` captures the first native group in the worker, excluding the
independent Policy Server's GPU execution. Profile inference separately when
needed and measure formal throughput without profiling.

A self-test, smoke episode or first-action comparison does not establish full
native equivalence, unchanged full-suite success rate or a 2x speedup. Record
measured results and their limits here before making those claims.

## Validation record

[2026-11-07 integration validation](validation-20261107.md) records the fresh builds,
real one-slot and five-slot smoke runs, exact native first-batch comparison,
worker phase measurements and remaining full-suite acceptance work.

The [small-scale runtime study](runtime-optimization-20261107.md) preserves
earlier diagnostics; its multi-slot baselines are superseded by the renderer
correction. The [final study](rendering-correctness-20261107.md) records the
complete baseline, failed candidate/control comparisons, partial-record loss,
descriptive timings and canonical default-profile operational check. Use the
example's `--task` and `--groups` for bounded iteration, and retain full records
outside the live queue before automatic retry cleanup can remove them.
