# RoboCasa365 rendering correctness and timing — 2026-11-07

**Consumed-frame rendering remains experimental and disabled by default.** The
registered `fixed50-03` trial completed **50 episodes per arm**, with **14/50
successes (28%) in both arms**, identical paired success outcomes, steps and
termination reasons. All registered physical reset conditions and provenance
checks passed. This establishes **no observed paired task-success change in
this recorded sample**, not unchanged population accuracy or exact behavior.
Strict/effective/trajectory audits still failed, and the earlier `all`/`all`
control also diverged. No speedup or absolute no-impact claim is made. Earlier
failed studies and their lost-record limits remain documented below.

## Frozen workload and revisions

This task retains its official v1.0.1 **450-step horizon**, native five-slot
batches, group success barrier and termination checks. Fifty episodes mean ten
sequential groups. Labels are identifiers, not reset seeds: constructor streams
use `1111111 + slot`, followed by unseeded continuation resets. The study is
always `full_suite: false`, separate from the official 50-task/2,500-rollout suite.
The corrected 28% baseline sample rate satisfies the requested 20–80% task
selection range; it does not establish a population success probability.

| Setting | Fixed configuration |
|---|---|
| Kit | 0.8.0b33; published fork commit `425e9fd5e84f1c63c3e50d64f2bb316a5f483c64`, tree `489a7646ea75f69d45455917e917d32bd17105ce` |
| Machine | RTX 4090, reported 49,140 MiB; driver 580.105.08; 12-core container quota; 96 GiB memory limit |
| Simulator | RoboCasa 1.0.1 `3443cdf3dd20f9ae8ed41adba442059b506891d0`; RoboSuite 1.5.2 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| Runtime | Python 3.11.13, MuJoCo 3.3.1, NumPy 2.2.5, EGL |
| Policy | Official `pretrain_human300` epoch-0500 DP/local CLIP; FP32, 100 denoising steps, five-row batches, eight-action chunks |
| Inputs | Three 256×256 RGB cameras, relative end-effector position/quaternion, gripper state and language; last two consecutive physical observations |
| Scenarios | `pretrain`; common `paired-stable-regions-v1` validation profile |
| Scheduling/output | K=1/G=1 per arm, MPS disabled; concurrent arms; audits/timers enabled, videos disabled |

Hashes are in [source-inputs.json](source-inputs.json),
[runtime-patches.json](runtime-patches.json), [asset-inputs.json](asset-inputs.json)
and [model-inputs.json](../../examples/robocasa365_diffusion/model-inputs.json).
The [integration record](validation-20261107.md) lists inference dependencies.
GR1 `robocasa/tabletop` remains separate.
Reproduction uses the [build instructions](README.md#build-and-run) and
[native group helper](../../examples/robocasa365_diffusion/README.md#native-five-slot-smoke):
`--task TurnOnElectricKettle --groups 10` for the large attempt, or `--groups 2`
with `all` in both arms for the final control, retaining the audited settings above.

## Corrections and experimental scope

The pinned renderer could free another slot's resources during sequential hard
resets unless cleanup first bound its owning GL context. The reviewed
[cleanup patch](patches/owning-egl-context-cleanup.patch) corrects ownership and
rebinds a new renderer after replacement. Marker 2 also
[preserves interrupted EGL bindings](patches/preserve-egl-thread-state.patch):
API, display, context and read/draw surfaces. Per-context destruction no longer
releases the whole thread; initialization retains its release, and process-exit
cleanup releases the thread before display termination. Destroyed contexts are
never rebound. Both profiles include these fixes. They have **not** been
established as the cause or cure of the pixel differences or later drift.

Earlier b31/b32 multi-slot scores and timing attribution in
[integration validation](validation-20261107.md) and the
[small-scale study](runtime-optimization-20261107.md) are superseded diagnostics.
Matching initial inputs had not excluded later corrupted images. The old
single-slot replay did not exercise the five-slot cleanup lifecycle.

A fresh `--paired-validation` prefix adds stable Counter reset-region ordering
and intended RNG forwarding for stack-base fixtures, equally in both arms.
RNG forwarding adds no environment RNG draws. Stable ordering preserves
candidate membership but **changes seed-to-scene mapping**; this is not the
upstream official seeded realization. The profile always reports
`full_suite: false`, and cross-profile prefix reuse is rejected.

The tested acceleration renders the last two consecutive frames of each action
chunk, including partial chunks, and reuses discarded intermediate RGB samples.
Sensor clocks, physics, controllers, state observations, success checks and
termination still execute. No model, precision, sampler, normalization, K/G or
autotune optimization was included. Default evaluation remains `all`.

## Correctness outcomes

| Run | Retained/completed evidence | Outcome |
|---|---|---|
| `fixed50-02/a`, `all` | 50 unique original records; 22,500 steps; 570 batches | 14/50 successes (28%); no worker failure/retry/OOM; 67,500 render calls, zero reuse |
| `fixed50-02/b`, `consumed` | 35 episodes observed live; 15,750 steps; 399 batches | 11/35 successes (31.4%) in a **diagnostic prefix only**; no terminal result; full 35 partial records lost |
| `aa-03`, `all` / `all` | 10 unique original records per arm; 4,500 steps and 114 batches each | 4/10 successes each; no worker failure/retry/OOM; strict and behavioral checks failed |

### Incomplete candidate and retention failure

The first effective-input/action difference occurred at **global batch 60,
second-group step 24, slots 0 and 2**. The preceding steps 0/8/16 of that group
matched. Across **399 shared batches / 1,995 rows**, saved Policy audits contain
**676 unequal effective-input rows and 676 unequal action rows across 333
batches**, plus 683 unequal full prepared rows. All shared CPU/CUDA RNG
signatures matched. These are not solely outside-crop differences.

Live comparison reported physical/integration/action trajectory hash differences
in **15 episodes** before failure. Candidate terminal hashes were subsequently
lost, so that live observation **cannot now be revalidated against all original
records**. It is not a complete retrospective per-episode comparison.

At **group 07, slot 2**, the initial guard rejected only
`initial_observation_sha256`. Covered compiled model, integration-state, RNG,
controller and sensor signatures matched. Actual initial arrays were not saved
for this rejection: **the differing feature, camera, pixel and crop location are
unknown**. The earlier border-pixel explanation cannot be assigned to group 07.

The SDK automatically retried once; the sticky rejection correctly refused the
retry before Policy inference. B ended with **two worker failures and one
automatic retry**. There was no manual restart, reseed, record replacement or
guard relaxation. SDK requeue cleanup deleted its partial WorkResult; retry
publication may also have overwritten group-00 initial evidence. The monitor
had retained summaries/counts, not the full 35 candidate records. Surviving
Policy/group audits and logs are preserved. Missing records/hashes have not
been fabricated or reconstructed.

The 11/35 prefix is not a formal score or a valid accuracy comparison with
14/50. B has no complete final coverage validation, and the full-run comparator
could not finish without its terminal result/WorkRecords. Available effective
input/action/physics evidence already fails acceptance. A full 50/50 accuracy
claim or paired confidence interval is unavailable.

### Final all/all control

`aa-03` used the frozen marker-2 runtime and `all` in both arms. First-group
inputs matched and both initial-group checks passed. Effective-input/actions
first diverged at **batch 60 / step 24, slot 0**; slot 2 joined at **batch 61 /
step 32**. There were **107 unequal effective/prepared rows**, **54 unequal
action batches**, and **two episodes** with unequal physical/integration/action
and final controller/sensor signatures. RNG signatures remained equal. Both
arms completed 10 episodes with 4 successes. Original records are retained;
the final comparator reports `pass: false` and
`behavioral_equivalence.pass: false`.

Thus multi-episode repeatability fails even within the baseline. A/B drift cannot
be attributed solely to consumed rendering. Equal success counts do not prove
unchanged behavior; failed comparisons also do not prove the optimization
caused an accuracy change. **Neither no-impact nor speedup is claimed.**

## Registered completed outcome trial: fixed50-03

The method was registered **before launch**, with plan SHA256
`4d94fd45333f78351eaa8863770f8f637f3b23ea6a9693b2c8ab9e7647053241`.
The source, checkpoint, FP32/100-step inference, 50 scenario labels/constructor
streams, task, full horizons, native ordering, K/G and modes were unchanged.
Only the existing optional `--pair-reference-output` raw-observation blocking
guard was omitted; no SDK evaluation code or scoring rule changed. Raw inputs
and all secondary hashes were still recorded. Old failures were retained;
there was no reseeding, selective replacement, dropped mismatch or manual retry.

This method follows the failed multi-group A/A control: bitwise trajectories
are not a repeatable baseline criterion. The independent **post-run** validator
required every pair's initial state, scenario metadata, full integration state,
compiled-model signature, located RNG, controller and sensor signature, plus
all coverage descriptions, to match. It also checked frozen source/runtime,
Policy metadata, task/slot/seed/protocol provenance, counters, full unique
terminal coverage and zero recovery. All **50 pairs** passed those requirements;
all 10 initial files and all 50 **raw initial observation hashes also matched**.
Nothing was restored or selectively matched after the run.

| Primary outcome / validation | `all` baseline | `consumed` candidate |
|---|---:|---:|
| Completed unique records | 50/50 | 50/50 |
| Task successes | 14/50 (28%) | 14/50 (28%) |
| Actual physical steps | 22,500 | 22,500 |
| Policy batches | 570 | 570 |
| Worker failures / retries / OOM | 0 / 0 / 0 | 0 / 0 / 0 |
| Real camera calls | 67,500 | 17,100 |
| Reused camera samples | 0 | 50,400 |
| Steps with fresh RGB | 22,500 | 5,700 |
| `full_suite` | `false` | `false` |

The paired table has **14 both-success, 36 both-failure, zero baseline-only and
zero candidate-only successes**. There were **zero step-count differences and
zero termination-reason differences**. Every episode used 450 steps. Model RNG
reset and batch signatures had zero differences. `primary-outcome.json` reports
`criterion_valid: true`, `no_observed_paired_success_change: true` and **zero
validation issues**. Original full WorkResults were backed up with content hashes
at every five-record increment, independently of retry-managed queue files.

The separate audits remain **`strict_pass: false` and behavioral pass false**:
19 episodes differ in physical/integration/action digests, 22 in observation
digests, 911 effective-input/action rows across 490 batches, and 920 full
prepared rows across 494 batches. The first effective/action difference is
batch 60, slot 0, step 24. These failures were not masked or converted into exact
passes by the outcome criterion. The render-call reduction is measured work
avoidance, not a measured end-to-end speedup.

The supported conclusion is limited to these 50 recorded pairs on one task and
the common diagnostic scene mapping. Equal sampled outcomes do not prove
population equivalence, arbitrary-Policy/full-suite accuracy or future no-impact.
Only one registered 50-pair trial was run. A non-significant test would not
supply such a guarantee either. Because baseline A/A trajectories also diverge,
secondary A/B drift cannot be attributed solely to consumed rendering. Default
rendering remains `all`; concurrent audited timings support no speedup claim.

## Audit coverage and bounded crop evidence

The audit covers workload/provenance, initial scenario/compiled-model
signatures, full MuJoCo `mjSTATE_INTEGRATION`, declared controller/fixture fields,
located RNG states/sharing, sensor schedules, physical transitions, prepared
inputs, actions, Policy RNG, outcomes and termination. Coverage is declared;
this is not a complete Python clone or raw MuJoCo memory dump. Only RGBA rows
of currently hidden collision-group-zero geometry are omitted from model
signatures, because upstream randomizes these invisible colors.

Earlier `probe-render-01` completed five 450-step episodes per arm with physical
arrays and 57 action/RNG batches exact and 3/5 successes each. Three slot-3
left-camera first-history frames, physical steps 127/135/143, differed by
4/6/7 uint8 channel values of one level at rows 97–101, columns 240–243. Those
saved differences were outside the actual 224×224 center crop `[16:240,16:240]`;
105 captured crop checks were exact. This one-group finding did not extend to
the multi-episode failures.

The Policy audit preserves full prepared hashes and separately hashes effective
CPU inputs after checking the actual loaded eval encoder, crop and elementwise
normalizer sources/configuration. Unknown paths fail closed; all non-image
inputs remain included. No inference crop, model call, RNG draw, GPU copy or
synchronization is added. Full-image mismatches stay strict failures. The
separate model-specific behavioral criterion requires exact effective inputs,
actions/RNG, covered trajectories, outcomes and coverage. Both criteria failed
in the final two-group control and the registered 50-pair trial; task outcomes
are assessed separately above.

## Descriptive baseline timing

`baseline50-timing.json` derives these intervals from the earlier completed
`fixed50-02/a` final result and
original WorkResult. **This was not an idle-machine performance baseline.**
The arms shared resources until B exited after 35 observed episodes; CPU build
and snapshot work also overlapped. Audits/timers remained enabled. No comparative
speed ratio is calculated.

| Top-level helper phase | Seconds | Scope |
|---|---:|---|
| End-to-end | 1,583.614 | Whole helper interval; excludes prior builds/downloads |
| Checkpoint preparation | 0.038 | Path/preparation bookkeeping |
| Autotune placeholder | 0.038 | No tuning; fixed K/G |
| Policy process/load/readiness | 11.214 | Imports, verification, model/CLIP loading and readiness |
| Runtime preparation | 0.841 | Before evaluated execution |
| Evaluation | 1,569.717 | Worker lifecycle and evaluated/audited work |
| Other helper work | 1.767 | Remaining interval |

Model-only loading, first-inference latency and server GPU execution were **not
measured separately**. The runtime window was 1,571.806 s, with descriptive rates
114.52 completed episodes/h and 14.315 steps/s. These are not production
throughput measurements.

| Worker category | Seconds | Evaluation share / meaning |
|---|---:|---|
| Environment construction | 26.723 | 1.7%; five constructors, including native warm resets |
| Evaluated resets/initial evidence | 248.004 | 15.8%; includes fingerprints and initial audits |
| Native simulation/observations | 808.320 | 51.5%; physics, controllers, fixtures, success checks, conversion/rendering |
| Policy round trips | 464.856 | 29.6%; protocol, preprocessing, GPU work, copies/server audit writes |
| History assembly | 0.641 | Worker observation construction |
| Action conversion | 0.081 | Native converter |
| Policy lifecycle | 4.635 | Reset/end requests and server work |
| Explicit worker audits | 3.700 | Excludes initial audits inside reset and some final signature work |
| Evidence writes | 3.822 | Timed worker evidence output |
| Video writes | 0.000 | Disabled |
| Environment cleanup | 0.516 | Five environments |
| Unattributed worker interval | 7.632 | Untimed final signatures, queue/Python work |
| Worker task total | 1,568.932 | Construction through cleanup |
| Evaluation outside worker task | 0.786 | Enclosing runtime overhead |

Camera API wall time was **89.876 s**, nested inside native steps, over **67,500
calls**. The remaining step interval was 718.444 s, **not isolated physics**.
API host wall time, measured without extra GPU synchronization, is not summed
GPU kernel time. Never add camera time to step time or worker time to evaluation
time. Policy-server audit flushes are nested in RPC/lifecycle timings.

The first group's reset-plus-rollout interval was **167.142 s**. The later nine
had median **148.765 s**, p90 **161.989 s**, range 137.880–173.150 s. These are
unique native-group intervals, not 50 independent episode latencies: five
records repeat each group's duration. They exclude some post-group evidence
work. Later groups are **not steady-state benchmark measurements**, because
contention changed and background work overlapped.

| Resource / measurement | Available evidence |
|---|---|
| Sampled mean GPU utilization | 15.89%, shared-device observation |
| Peak GPU memory | 13,789 MiB, shared-device observation |
| Average GPU memory | Not measured |
| Mean container CPU use | 2.360 cores, including overlapping work |
| CPU throttle-time ratio | 0.170% |
| Isolated rendering/inference GPU kernel time | Not measured |
| Nsight hardware capture | Unverified; worker timers are separate |
| B formal timing/resources | Unavailable as a completed result |

For the **completed registered trial**, `timing-breakdown.json` records the
following descriptive intervals. These arms also ran concurrently with audits
and backups; the shorter arm's exit changed contention for the other's tail.
**No speedup ratio is inferred from these values.**

| Registered-trial interval | `all` seconds | `consumed` seconds |
|---|---:|---:|
| End-to-end | 1,565.523 | 1,499.135 |
| Evaluation | 1,551.117 | 1,483.572 |
| Policy process/load | 12.010 | 11.962 |
| Runtime preparation | 0.839 | 0.830 |
| Environment construction | 26.107 | 25.524 |
| Resets/initial evidence | 250.095 | 253.381 |
| Native simulation/observations | 809.042 | 746.329 |
| Camera API, nested in native steps | 91.079 | 23.800 |
| Policy round trips | 454.708 | 446.341 |
| Environment cleanup | 0.360 | 0.529 |

Native steps, Policy RPC and resets dominate these recorded intervals. Further
work should quantify baseline variability and retain full failure records, then
isolate stepping/reset host cost and server inference. Report gains only after
sequential, uninstrumented, repeated fixed-K/G trials on idle hardware; retune
K/G separately. No inference or scheduling improvement is included here.

## Canonical deployment and evidence

The canonical **default `all`** b33 runtime passed self-test. Its 248 source files
match the frozen Git tree byte for byte; the tested Python/runtime sources match
as well. Each installed SDK environment (control/inference) has **88 verified
Python files** matching that commit. The canonical `default-b33-fast` operational
run, using the original Policy without audits, completed **3/5 successes and
2,250 physical steps**, with no worker errors. Helper end-to-end was 201.173 s,
evaluation 187.052 s, Policy startup/load 11.598 s and preparation 0.832 s.
This default-profile smoke is distinct from the paired diagnostic study; its
different instrumentation/profile prevents a speed comparison with A50.

The canonical runtime archive was verified by size and SHA256:

- `tested-runtime.tar.zst`: **14,366,336,823 bytes**.
- SHA256: `ff22463d073661987d39026922366d41445a658463731f1fd86732e0c367b3cd`.

The archive keeps runtime/environments/source; original source/asset archives
and external model weights remain in persistent storage separately.

Original A50 records/results/timings and surviving failed-B evidence are outside
Git in local `evidence/large-pair/fixed50-02-preserved/`.
`final-monitor-summary.json` and `preservation.json` identify available/lost data.
Full A/A10 records, comparator and Policy audits are in
`evidence/large-pair/aa-03-preserved/`. The complete registered trial is in
`evidence/large-pair/fixed50-03-preserved/`, including `primary-outcome`,
`final-statistics`, `timing-breakdown` JSON/Markdown, original results/Policy
audits, source inventories and all content-addressed WorkResult snapshots.
`baseline50-timing.json` records source
hashes and interval definitions. The verified raw-evidence attachment is
`robocasa365-correctness-b33-evidence.tar.gz` (**13,196,968 bytes**), retained at
`/autodl-fs/data/robocasa365/large-pair/` and delivered separately from Git.
Its SHA256 is
`58919d014917ac8be288b0195a133c00f86a02b33becb4701d440773b8edaf2e`.
The bundle includes the registered plan, original records and analysis code;
its 164 file payloads were checked against the preserved files. Weights, assets,
bulk logs and arrays are not committed. Missing candidate records from the
earlier failed attempt are explicitly absent, not rebuilt from baseline outcomes.
