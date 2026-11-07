# Official RoboCasa365 Diffusion Policy

This example serves the officially published `pretrain_human300` multi-task
Diffusion Policy through the RoboScope Kit lifecycle. It serves the independent
**`robocasa365`** benchmark. The existing `robocasa/tabletop` suite uses GR1,
29-dimensional actions and a different model interface.

The checkpoint was trained on RoboCasa365 human demonstrations across 300 tasks.
It is linked in the [official multi-task documentation](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/docs/benchmarking/multitask_learning.md).
This example performs no training or model inference optimization.

## Pinned inputs

| Input | Revision |
| --- | --- |
| Diffusion Policy source | `41212698b6f481ed92a55f0d5f1778ec56bea417` |
| RoboCasa robomimic fork (`robocasa-dp`) | `7dc18461c4fc497095f1044f4c40cd44fff62e6a` |
| Official checkpoint repository | `c484448aba1a9b60a04c9b0ca117241518ea69f3` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

`model-inputs.json` records source archive and weight hashes, checkpoint size and
download locations. The installer uses canonical commit-pinned codeload archives,
without requiring Git or inheriting host Git mirror rewrites. It verifies archives
before extraction, writes `.roboscope-source.json` provenance, and checks existing
source files against the archive on repeated builds. An existing unverified source
directory is rejected rather than overwritten. The downloader and `load()` verify
both weight files. The checkpoint is about
1.70 GB; the full pinned CLIP weights are about 1.71 GB. Allow additional space
for caches and CUDA wheels. The official inference guidance recommends at least
8 GB GPU memory; five simulator slots need additional memory.

Use Linux x86_64, Python 3.11 and a CUDA GPU. Keep the inference and simulator
environments separate. The official model needs the RoboCasa robomimic fork's
language-conditioned visual encoder, which ARISE robomimic does not provide.
The installer uses `--no-deps` for that source because its package metadata
specifies older training dependencies. Direct inference dependencies are in `requirements.txt`; `requirements-lock.txt`
freezes the complete validated inference dependency closure used by the installer.
The pinned Diffusion Policy tree uses implicit namespace packages while its
`setup.py` uses `find_packages()`. The installer therefore writes a `.pth` entry
for the verified source root into the inference environment and checks the actual
model imports with Python's isolated mode. Imports do not depend on the working
directory or `PYTHONPATH`.

## Install and download

From the repository root, with `uv` available:

```bash
bash examples/robocasa365_diffusion/install-policy.sh /sim/robocasa365-policy

HF_HOME=/sim/robocasa365-policy/huggingface \
  /sim/robocasa365-policy/env/bin/python \
  examples/robocasa365_diffusion/download_model.py /models/robocasa365-diffusion
```

Set `ROBOSCOPE_POLICY_PYTHON` to an existing Python 3.11 executable to avoid
downloading an interpreter. `ROBOSCOPE_PYPI_INDEX` selects a reachable package
index; versions remain pinned. The CUDA wheel index is selected explicitly.
Set `ROBOSCOPE_POLICY_SOURCE_ARCHIVES_DIR` to keep the two verified source archives
in persistent storage. Pre-seed that directory with
`diffusion_policy-<revision>.tar.gz` and `robomimic-<revision>.tar.gz` when the build
host cannot access codeload directly. The same committed hashes are required for
cached and downloaded archives. `install_sources.py` can also run directly with
any Python 3.11 interpreter before installing GPU dependencies.

This creates `model.ckpt` and a local `clip/` encoder directory. Pass the model
directory as `--checkpoint-source`. `load()` uses the provided
`SessionContext.checkpoint_path`; it also accepts a checkpoint file with `clip/`
as its sibling. The MetaWorld StarVLA checkpoint is incompatible and is rejected.

Replace paths in `roboscope-policy.json` with your absolute Python, repository
and working-directory paths. In an EvoStudio adaptation container,
`/workspace/user_policy.py` can symlink to this example's `policy.py`, preserving
access to its adjacent metadata and downloader module. Set that entrypoint in
`/workspace/evostudio-policy.json`.

The installer installs the local Kit package; reinstall it after modifying Kit.
Build the simulator separately with `benchmarks/robocasa365/build-runtime.sh`.

## Observation and action semantics

The checkpoint config is authoritative. The Policy verifies its exact feature
shapes, delta actions, two observation frames and eight-action chunk. It selects
`ema_model` when the checkpoint requests EMA and loads the full state dictionary,
including observation/action normalization. Precision, 100 diffusion sampling
steps and trained model parameters are unchanged.

| Worker feature | Model input |
| --- | --- |
| `images.agentview_right` | `robot0_agentview_right_image` |
| `images.agentview_left` | `robot0_agentview_left_image` |
| `images.wrist` | `robot0_eye_in_hand_image` |
| `state.end_effector_position_relative` | `robot0_base_to_eef_pos` |
| `state.end_effector_rotation_relative` | `robot0_base_to_eef_quat` |
| `state.gripper_qpos` | `robot0_gripper_qpos` |
| `language.instruction` | `lang_emb` (CPU CLIP projection) |

Images arrive as `(2, 256, 256, 3)` uint8. The native gym wrapper already flipped
them vertically. The Policy applies robomimic `process_frame` to convert to CHW
float32 and divide by 255. State histories have float32 shape `(2, 3)`, `(2, 4)`
and `(2, 2)`. History contains the **last two consecutive simulation steps**;
the initial frame is duplicated. Previous inference-boundary frames would be
incorrect inputs. Temporal history is owned and reset by the worker.

The CPU language encoder uses the pinned local CLIP weights, upstream
`padding="max_length"` tokenization and 768-dimensional `text_embeds` projection.
It creates one embedding per episode, matching the fork's `LangEncoder` without
a network download during evaluation.

Native `predict_action` normalizes observations, samples diffusion and
denormalizes actions. The Policy returns its eight 12-dimensional actions as
float32 without clipping or scheduler changes. The worker applies native action
conversion and scoring.

`reset()` clears per-episode language state. The checkpoint's native model reset
is a no-op because it has no recurrent hidden state. Each task owns a torch RNG
stream initialized from checkpoint training seed 42. Batches within a task use
that stream in native slot order; interleaved tasks cannot consume one another's
diffusion noise. The worker's first-slot/first-episode reset restarts the stream
for each new workload or retry, so autotune probes cannot consume formal-run
noise. CLIP initialization and encoding do not consume these streams.

## Native five-slot smoke

The CLI Quick Test uses one environment. To exercise the formal batch-five path
on one task without running all 2,500 episodes, use the public runtime helper:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python examples/robocasa365_diffusion/run_group_smoke.py \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --policy-config /workspace/evostudio-policy.json \
  --checkpoint-source /models/robocasa365-diffusion \
  --output-dir /results/robocasa365-group-smoke
```

By default it runs the first target50 task in all five native slots. Use
`--task <official-task-name> --groups 1` to select five full-horizon episodes;
`--groups 2` continues the same five constructor/reset streams for ten episodes.
One to ten groups are supported. K=1/G=1 and videos remain fixed. This makes a
small workload explicit without shortening horizons, padding records or changing
the formal suite. Outputs record `full_suite: false` and the selected workload.
The captured first batch can be passed to `compare_native.py` in the same way as
a single-slot capture.

`--render-mode all` is the default baseline. The optional `--render-mode consumed`
requires rebuilding the simulator with the committed RGB sensor patch. It renders
the final two consecutive physical frames of every action chunk and reuses only
discarded intermediate camera samples. Sensor clocks, physics, controllers and
success checks still execute at their native frequency. Partial final chunks keep
their own last two frames. A Policy must declare stride eight without intermediate
observation updates; depth, segmentation and custom camera processing are rejected.
The benchmark worker also preserves the native odd-step video frames when video
logging is enabled. Neither the model nor its denoising schedule is changed.

For a correctness run, add `--audit-trajectories`: records contain independent
digests of all consumed features/actions and every physical state/control/success
transition; per-batch hashes are saved outside Git in the run's evidence folder.
`--diagnostic-timings` adds render API wall-time and call counters. Camera API time
is a subset of environment-step time, not a separate additive phase or isolated
GPU kernel time. Disable both diagnostic flags in repeated timing comparisons.

## Native comparison

Run the pinned upstream `eval_robocasa.py` with the same simulator, checkpoint,
`--task_set atomic_seen composite_seen composite_unseen`, `--split pretrain`,
`--num_rollouts 50` and `--num_envs 5`. This covers 50 tasks and 2,500 rollouts.
Smaller subsets must be labelled smoke/partial evaluations.

The Kit formal factory preserves five native slots per task group. Its
`--quick-test` uses one slot and is a smoke test. Native batch size and ordering
affect diffusion noise, so a one-slot smoke result cannot establish native score
equivalence. The native evaluator's `100000+i` labels are ineffective seeds:
its `env.seed(seed)` call is commented out. Constructor seed `1111111+slot` and
sequential resets govern scenes. Compare scenario metadata and initial-state
digests; matching rollout labels alone is insufficient.

Report native-versus-Kit correctness separately from Kit runtime speedups. Keep
model inputs, scheduler, precision, native slot count, task horizons and output
settings fixed. Changes to this Policy's inference are outside simulator/runtime
speedup claims.

For a first-batch input/action parity check, install `native-requirements.txt`
into the inference environment with the pinned model sources.
Use a real worker capture, with the same device and runtime as its producing run:

```bash
python examples/robocasa365_diffusion/compare_native.py \
  --checkpoint /models/robocasa365-diffusion \
  --batch /results/evidence/worker-000/CloseBlenderLid-first-batch.npz \
  --output /results/evidence/native-first-batch-parity.json
```

The helper invokes the upstream workspace loader and language encoder. It checks
the pinned observation wrapper file's SHA-256 and executes its `process_obs`
method directly, skipping unrelated simulator imports. No simulator installation
or environment construction is needed for this capture-only check. It compares
each loader's independently captured RNG state, every prepared feature, and the
first action chunk exactly. This does not replace a full native rollout and
per-task/per-split score comparison. The additional gym/AV packages in the native
requirements are for running the complete upstream evaluator with the simulator.
