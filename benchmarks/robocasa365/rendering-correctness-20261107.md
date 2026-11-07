# RoboCasa365 rendering correctness study — 2026-11-07

Kit **0.8.0b33** provides a corrected renderer baseline and an isolated common
setup for testing consumed-frame rendering. A full five-slot diagnostic matched
all physical trajectories, actions and Policy RNG states, with 3/5 successes in
each arm. Its few unequal border pixels were outside the checkpoint's verified
fixed crop; **full-image equality still failed**. A check on the final runtime
and the replacement **50-episode-per-arm** evaluation remain pending. No speedup
or unchanged full-suite accuracy claim is accepted.

## Frozen workload

The selected task is **TurnOnElectricKettle**, with ten sequential groups of five
native slots per arm. The official v1.0.1 horizon remains **450 physical steps**;
the native group success barrier and termination checks remain active. Episodes
are not shortened for this experiment. Actual terminal steps must come from
completed records, since the native barrier can end a group before its horizon.

An earlier five-slot screening reported 1/5 successes and motivated this task
selection. That screening preceded the renderer correction below. Its 20% rate
is **not** a corrected-renderer baseline or a population success-rate estimate.

| Setting | Both arms |
|---|---|
| Machine | RTX 4090, reported 49,140 MiB; driver 580.105.08; 12-core container quota; 96 GiB memory limit |
| Kit | 0.8.0b33 |
| Simulator | RoboCasa 1.0.1, `3443cdf3dd20f9ae8ed41adba442059b506891d0`; RoboSuite 1.5.2, `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| Runtime | Python 3.11.13, MuJoCo 3.3.1, NumPy 2.2.5, EGL |
| Model | Official `pretrain_human300` epoch-0500 Diffusion Policy; pinned checkpoint and CLIP hashes |
| Inference | FP32, 100 denoising steps, five-row batches, eight actions per chunk |
| Inputs | Three 256×256 RGB cameras, relative end-effector position/quaternion, gripper state and language; last two consecutive physical frames |
| Scenarios | `pretrain`; constructor seeds `1111111 + slot`; sequential unseeded continuation resets |
| Planned identifiers | `TurnOnElectricKettle-rollout-100000` through `100049` |
| Scheduling | K=1/G=1 per arm; MPS disabled; concurrent correctness arms share the machine |
| Outputs | Videos disabled; trajectory, initial-state, Policy and diagnostic timing audits enabled |
| Runtime profile | `paired-stable-regions-v1`; always `full_suite: false` |

Frozen hashes are in [source-inputs.json](source-inputs.json),
[runtime-patches.json](runtime-patches.json), [asset-inputs.json](asset-inputs.json)
and [model-inputs.json](../../examples/robocasa365_diffusion/model-inputs.json).
The [integration record](validation-20261107.md) lists the separate inference
environment and model-source pins. The older GR1 `robocasa/tabletop` runtime is
separate.

## Renderer baseline erratum

Sequential hard resets exposed a context ownership error in upstream renderer
cleanup. Deleting an old renderer's GPU resources without binding its owning EGL
context could free resources belonging to another slot. Initial images, physical
states and first actions could match while subsequent images in the first four
slots were corrupted; the last-created slot could remain unaffected.

The reviewed [owning-context patch](patches/owning-egl-context-cleanup.patch)
binds the old renderer's context before destruction and rebinds the new context
after replacement. Both build profiles and both experiment arms include this
fix, verified by original, patch and resulting-file hashes. The worker rejects
a renderer missing its cleanup marker.

The final **marker-2** cleanup also preserves the interrupted EGL client API,
display, context and read/draw surfaces around delayed destruction. Per-context
cleanup no longer calls the thread-wide `eglReleaseThread`; initialization
retains its release, and final thread release runs at process exit before display
termination. A previously destroyed owner is never rebound. Both profiles and
both arms receive the
[EGL preservation patch](patches/preserve-egl-thread-state.patch). CPU mocks cover
foreign/owner/unbound contexts, nested finalizers, exceptions and restoration
failure. **This known restoration defect has not been established as the cause
of the small border-pixel differences described below.**

Earlier multi-slot scores and timing attribution in the
[integration record](validation-20261107.md) and
[small-scale study](runtime-optimization-20261107.md) are superseded diagnostics.
They must not serve as corrected baseline scores, render-cost estimates or a
before/after speed comparison. The earlier single-slot fixed-action prefix did
not exercise the five-slot cleanup failure either. This study starts with a
fresh corrected runtime.

## Common paired setup and its protocol limit

The optional `--paired-validation` build creates a separate prefix and applies
two setup fixes **equally to both arms**:

- Counter reset-region deduplication uses XML insertion order through
  `list(dict.fromkeys(valid_geoms))`. The former set used object-identity order,
  which `PYTHONHASHSEED` does not stabilize.
- Stack-base construction passes `rng=self.rng` to `initialize_fixture`. That
  helper overwrites the config's RNG with its explicit argument. Previously the
  missing argument caused the Box to retain an entropy-seeded RNG. The Box only
  assigns the supplied RNG and adds no environment RNG draws; explicit box size
  also avoids random size sampling.

Stable ordering preserves candidate membership but **changes the previous
identity-dependent seed-to-scene mapping**. Equal numeric seeds do not make this
the official upstream scenario realization. Results are validation evidence on
a common setup, not an official target50 score or native leaderboard
reproduction. The profile always reports `full_suite: false`. Cross-profile
prefix reuse and changed selected-patch provenance are rejected; the canonical
runtime is not patched in place.

The earlier `aa-01` A/A control was rejected: equal initial model, integration
state and inputs did not prevent unequal fixture RNG signatures and later drift.
After the common setup and renderer fixes, **`aa-02` passed the strict full-episode
comparison in all five slots**. This validates repeatability for that control
workload before the final marker-2 update. The final runtime and larger rendering
comparison require fresh checks; an earlier pass is not transferred to changed
runtime code.

## Render change and audit coverage

`all` renders each camera at every physical step. `consumed` retains the final
two consecutive frames of each action chunk, including partial final chunks,
and returns cached RGB on discarded intermediate steps. Sensor updates and
sampling clocks, physics, controllers, state observations, success checks and
termination continue at their native frequency. The model, normalization,
precision, diffusion noise stream and inference schedule are unchanged.
Default evaluation remains `all`.

The second arm's `--pair-reference-output` guard requires exact initial evidence
for each group **before Policy reset/inference**. A mismatch persists as
`pairing-rejected.json`; a retry cannot turn that rejection into acceptance.
The guard does not restore state, reseed, drop or replace unmatched episodes.

| Layer | Compared evidence |
|---|---|
| Workload | Source/profile/task/scenario plan, native slots/order, IDs, horizon, steps, termination, unique coverage, worker failures/retries/OOM |
| Initial state/model | Scenario metadata, consumed raw features, `mjSTATE_INTEGRATION`, public primitive compiled-model fields plus `mjOption`/`mjStatistic` |
| Python state | Located environment/robot/fixture/placement-sampler RNGs and sharing; declared controller/buffer/fixture fields; sensor schedules and non-image caches |
| Each physical step | Integration-state digest and physical state/control/success/done digest |
| Each inference batch | Full worker/prepared input and action digests; separate verified effective-input digests; row identities/order; Policy CPU/CUDA RNG before and after inference |
| Completion | Per-episode success/terminal step, four complete trajectory digests, final covered RNG/controller/sensor signatures and coverage descriptions |

`mjSTATE_INTEGRATION` includes warmstart, control and other integration inputs
missing from `sim.get_state().flatten()`. The compiled-model signature omits only
RGBA rows of currently invisible collision-group-zero geometry, whose random
colors can change raw XML without changing visible images or physics. Collision
geometry, materials, textures and all other covered fields remain checked.

Signatures declare their limits: they are not raw MuJoCo memory hashes or a
complete Python clone. Discarded image caches are excluded from sensor schedule
signatures. The strict report retains every full-image/prepared-input mismatch.
A separate model-specific behavioral report requires exact **effective** inputs,
actions/RNG, covered trajectories, outcomes, termination, provenance and coverage;
it does not relabel a failed full-image report as a strict pass.

Effective inputs are verified against the **actual loaded evaluation modules**,
not inferred from a nominal crop setting. The audit rejects unknown encoder,
normalizer or crop source hashes, training mode, unsupported shapes/randomizers,
multiple crops or positional encoding. For this checkpoint, each RGB path uses
one deterministic 224×224 center crop from 256×256: zero-based rows and columns
`[16:240]`, with the upper bound excluded. All non-image prepared inputs remain
included unchanged. The saved normalizer is the verified elementwise affine
implementation, so equality on the selected CPU values implies equality after
that same normalization and fixed selection. No inference crop is added or
changed. The wrapper retains full `prepared_sha256` alongside the separate
`effective_input_sha256`; no pixel tolerance is used. It adds no model calls,
RNG draws, GPU copies or explicit CUDA synchronization.

## Frozen attempt and results

`fixed50-01` planned 50 episodes per arm, but was stopped after the first reported
prepared-input difference in **slot 3, batch 16, physical step 128**. Initial
group checks had passed. Model metadata, Policy RNG signatures and actions still
matched through 20 captured calls; those matches do not excuse unequal inputs.
The failed attempt is retained. It produced no completed episode records and
cannot support a 50-episode success-rate comparison.

| Check | `aa-02`: `all` / `all` | `fixed50-01`: `all` / `consumed` |
|---|---|---|
| Planned episodes per arm | 5 | 50 |
| Completed episode coverage | Five per arm; strict comparison passed | No completed episode records; stopped early |
| Initial group guard | Passed | Passed for the reached initial group |
| Full recorded comparison | Passed for all five episodes | Rejected: prepared-input mismatch |
| Success-rate comparison | Repeatability control only | Unavailable |
| Unchanged rendering behavior | Not tested by A/A | Not established |
| Speedup ratio | Not measured | Suppressed |
| `full_suite` | `false` | `false` |

### Complete five-slot diagnostic and crop investigation

`probe-render-01` ran both modes for **450 steps in every slot**: 2,250 physical
steps and 57 five-row Policy batches per arm. All captured integration, control,
qpos and qvel arrays were exactly equal, including the initial state and every
transition. All 57 batches' actions and Policy RNG states matched exactly. Both
arms recorded **3/5 successes**.

Only three prepared rows differed: slot 3 at inference steps 128, 136 and 144.
Each difference was in the **first history frame** of the left camera, from
physical steps **127, 135 and 143** respectively. There were **4, 6 and 7 RGB
channel values** differing by one uint8 level, within zero-based rows 97–101 and
columns 240–243. These columns lie outside `[16:240]`. The maximum difference
after conversion to float32 `[0,1]` was approximately `1/255`.

The real checkpoint was loaded on CPU to validate the encoder/crop/normalizer
contract without inference calls. Separately, **105 checks of captured arrays
through the actual evaluation crop path were exact**, with no failed crop checks.
This explains why those saved border differences do not alter the verified
model-consumed crop; it does not establish identical 256×256 rendered images or
explain their rendering cause. The strict raw/prepared-image report remains
**false**. Observed equality of physics/actions/RNG/outcomes and the verified
effective-input criterion are reported separately.

### Current acceptance state

| Attempt | Scope / result |
|---|---|
| `fixed50-01` | Rejected prepared-input mismatch; stopped without completed episode records |
| `probe-render-01` | Five full episodes per arm; physics/actions/RNG and outcomes exact; full images differ outside verified crop |
| `ab-02` | Cancelled after an upload timeout; no evaluation evidence accepted |
| `ab-03` | Native five-slot check running on the final source and marker-2 runtime; result pending |
| Replacement 50-episode pair | Pending; no completed 50-episode accuracy result |

**A replacement 50-episode result remains pending.**
Its completed report must include successes/rates, actual steps and termination
distribution, missing/duplicate records, worker failures/retries/OOM, camera
render/reuse calls, Policy batch counts and every exact audit result. Do not
retain only matching episodes or present partial records as 50 completed trials.

A future behavioral pass would establish no observed effective-input, behavior
or success change on those paired episodes of this fixed-crop model, task and
profile. A strict full-image pass is a separate stronger result. Neither proves
unchanged general success probability, full-suite equivalence or native score
equivalence. Unequal effective inputs, physics, actions/RNG or outcomes cannot
be excused by a similar aggregate rate. Shared five-slot streams and Policy
batches also require care before treating outcomes as independent Bernoulli
samples.

## Available timing fields and their limits

The concurrent audited arms compete for CPU/GPU resources, and the reference
guard can wait. **No throughput, render-time, step-rate or end-to-end speedup
ratio is accepted from this correctness experiment.** Shared-device GPU samples
cannot be attributed to one arm. The following fields exist in actual run
records; incomplete-run values must remain labelled partial diagnostics.

| Field | Meaning |
|---|---|
| `phase_timings.policy_load_s` | Policy-process startup/load phase; not isolated weight I/O or GPU upload |
| `phase_timings.runtime_prepare_s` | Runtime preparation outside evaluated worker execution |
| `phase_timings.evaluation_s` | Evaluation wall time including worker lifecycle and audited work |
| `result.elapsed_s`, `end_to_end_wall_s` | Runtime interval and helper interval respectively; helper includes loading/preparation but excludes earlier builds/downloads |
| `environment_create_s` | Five constructors, native initialization/warm resets and rendering setup |
| `reset_s` | Evaluated continuation resets plus initial fingerprints |
| `environment_step_s` | Gym steps: physics, controllers, native observations/conversion and rendering |
| `camera_render_s` | Nested host wall time in render API calls; not isolated GPU kernel execution |
| `inference_wait_s` | Policy RPC including protocol, preprocessing, model work, copies and server audit overhead |
| `observation_build_s`, `action_conversion_s` | Worker history assembly and native action conversion |
| `policy_lifecycle_s`, `evidence_write_s`, `audit_s` | Explicitly timed requests, writes and audit computations; not a complete partition of audit overhead |
| `video_write_s`, `environment_close_s` | Video writing and cleanup; videos disabled here |
| `task_elapsed_s` | Worker construction through cleanup, including uncategorized work and waiting |
| `physical_steps`, `fresh_observation_steps`, `*_calls` | Work counters, not seconds |
| Per-record `group_elapsed_s`, `group_reset_s` | One shared five-slot interval copied to five records |
| Resource/recovery fields | CPU use/throttling, sampled GPU utilization/peak memory and worker/OOM counters |

Use unique group intervals for latency distributions. Do not sum their five
record copies or add nested camera time to step time. Initial/final signatures,
queue publishing, server audit flushing and reference waiting are not all
isolated by named worker timers. Nsight capture and separate Policy GPU kernel
attribution remain unverified. Earlier CUDA stream events include host-dispatch
idle gaps and are not summed kernel execution time.

After correctness passes, performance requires a separate sequential experiment
on idle hardware with audit/diagnostic flags disabled, fixed K/G, repeated full
workloads and separate cold-start/evaluation reporting. Retuning K/G belongs to
a later experiment. No inference optimization is included here.

## Reproduction and evidence

Build a fresh isolated prefix using committed locks and verified caches:

```bash
KIT_PYTHON=/autodl-fs/data/robocasa365/control/bin/python \
  ROBOSCOPE_ROBOCASA365_PYTHON=/opt/evostudio/control-runtime/python/bin/python3 \
  ROBOSCOPE_ROBOCASA365_ASSETS_DIR=/autodl-fs/data/robocasa365/downloads \
  ROBOSCOPE_ROBOCASA365_SOURCE_CACHE=/autodl-fs/data/robocasa365/downloads \
  bash benchmarks/robocasa365/build-runtime.sh \
  /root/autodl-tmp/robocasa365-20261107/runtime-paired-v3 --paired-validation
```

Each arm uses a schema-2 config launching the committed `audit_policy.py` with
its own fresh absolute `ROBOCASA365_POLICY_AUDIT` path. Follow the
[example](../../examples/robocasa365_diffusion/README.md#native-five-slot-smoke)
with `--task TurnOnElectricKettle --groups 10 --audit-trajectories
--diagnostic-timings`, the same paired manifest/checkpoint and fresh outputs.
Arm A uses `--render-mode all`; arm B uses `--render-mode consumed` and
`--pair-reference-output <absolute-arm-A-output>`. A/A controls use `all` twice.
The rejected `fixed50-01` output paths must not be reused.

Launch commands, runtime/source provenance, logs, initial evidence, worker
records, Policy audits and comparison outputs remain outside Git under
`/autodl-fs/data/robocasa365/large-pair/` and the task's local evidence directory.
Keep `aa-01`, `aa-02`, `fixed50-01`, `probe-render-01` and later attempts distinct.
Diagnostic summaries include `probe-render-01-summary.json`,
`probe-render-01-arrays.json`, `checkpoint-crop.json` and
`effective-contract-check.json`. Rejected/cancelled attempts remain identified
as such. The completed report must add the final comparison artifact link.
Weights, assets, raw arrays and bulk logs are not committed.
