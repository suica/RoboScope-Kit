# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 evaluation with native five-slot reset and termination groups.

Every task retains its constructor RNG streams across sequential rollouts. The
Policy receives the last two physical-step observations, with the initial frame
duplicated. Official Gym converters own image orientation and controller logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient, profiling
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkQueue, WorkResult

from .protocol import (
    ACTION_DIMENSION,
    ACTION_STEPS,
    CONSTRUCTOR_SEED,
    EPISODES_PER_TASK,
    IMAGE_KEYS,
    NATIVE_ENVS,
    OBSERVATION_STEPS,
    ROLLOUT_LABEL_BASE,
    SCENE_SPLIT,
    STATE_DIMENSIONS,
    TASKS_BY_NAME,
    Task,
    validate_upstream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--render-mode", choices=("all", "consumed"), default="all")
    parser.add_argument("--audit-trajectories", action="store_true")
    parser.add_argument("--diagnostic-timings", action="store_true")
    parser.add_argument(
        "--runtime-profile", choices=("official", "paired-stable-regions-v1"), default="official"
    )
    parser.add_argument("--pair-reference-output", type=Path)
    parser.add_argument("--pair-reference-timeout-s", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def create_environment(task: Task, slot: int):
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401 -- registers pinned Gym environments

    return gym.make(
        f"robocasa/{task.name}", split=SCENE_SPLIT, seed=CONSTRUCTOR_SEED + slot,
        camera_widths=256, camera_heights=256,
    )


def episode_id(task: str, index: int) -> str:
    return f"{task}-rollout-{ROLLOUT_LABEL_BASE + index}"


def environment_range(name: str):
    """Use the pinned Kit's NVTX range only under an active profiler injection."""
    return profiling._range(name) if profiling.ENABLED else nullcontext()


def build_observation(identifier: str, step: int, history: deque) -> Observation:
    if len(history) != OBSERVATION_STEPS:
        raise RuntimeError("RoboCasa365 requires two consecutive observation frames")
    features: dict[str, Any] = {}
    for feature, key in IMAGE_KEYS.items():
        value = np.stack([np.asarray(frame[key]) for frame in history])
        if value.shape != (2, 256, 256, 3) or value.dtype != np.uint8:
            raise RuntimeError(f"RoboCasa365 image {key}: invalid {value.shape}/{value.dtype}")
        features[feature] = np.ascontiguousarray(value)
    for key, dimension in STATE_DIMENSIONS.items():
        value = np.stack([np.asarray(frame[f"state.{key}"]) for frame in history])
        if value.shape != (2, dimension) or not np.isfinite(value).all():
            raise RuntimeError(f"RoboCasa365 state {key}: invalid values or shape {value.shape}")
        features[f"state.{key}"] = np.ascontiguousarray(value, dtype=np.float32)
    instruction = history[-1]["annotation.human.task_description"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("RoboCasa365 instruction is missing or blank")
    features["language.instruction"] = instruction
    return Observation(identifier, step, features, time.time())


def validate_actions(value: np.ndarray) -> np.ndarray:
    actions = np.asarray(value)
    if actions.shape != (ACTION_STEPS, ACTION_DIMENSION) or not np.isfinite(actions).all():
        raise RuntimeError(f"RoboCasa365 requires finite (8, 12) actions; received {actions.shape}")
    return actions


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported scenario metadata type {type(value).__name__}")


def reset_fingerprint(environment, observation: Observation, *, complete: bool = False) -> dict[str, Any]:
    native = environment.unwrapped.env
    state = np.asarray(native.sim.get_state().flatten(), dtype=np.float64)
    meta = json.dumps(native.get_ep_meta(), sort_keys=True, default=_json_default, separators=(",", ":"))
    features = hashlib.sha256()
    for key, value in sorted(observation.features.items()):
        features.update(key.encode())
        features.update(value.encode() if isinstance(value, str) else value.tobytes())
    result = {
        "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "scenario_metadata_sha256": hashlib.sha256(meta.encode()).hexdigest(),
        "initial_observation_sha256": features.hexdigest(),
    }
    if complete:
        from .audit import initial_signatures
        from .audit_state import native_control_signature, rng_signature, sensor_schedule_signature

        result.update(initial_signatures(environment))
        for key, signature in (
            ("initial_rng", rng_signature(native)),
            ("initial_controller", native_control_signature(native)),
            ("initial_sensor_schedule", sensor_schedule_signature(native)),
        ):
            result[f"{key}_sha256"] = signature["sha256"]
            result[f"{key}_coverage"] = signature["coverage"]
    return result


def save_first_batch(args: argparse.Namespace, task: str, observations: tuple, chunks: tuple) -> None:
    path = args.output_dir / "evidence" / f"worker-{args.worker_id:03d}" / f"{task}-first-batch.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        key: np.stack([observation.features[key] for observation in observations])
        for key in observations[0].features
    }
    values["actions"] = np.stack([chunk.actions for chunk in chunks])
    values["task_name"] = np.asarray(task)
    values["episode_ids"] = np.asarray([observation.episode_id for observation in observations])
    np.savez(path, **values)


def feature_digest(observation: Observation) -> bytes:
    digest = hashlib.sha256()
    for key, value in sorted(observation.features.items()):
        digest.update(key.encode())
        digest.update(value.encode() if isinstance(value, str) else value.tobytes())
    return digest.digest()


def configure_rendering(environment, args: argparse.Namespace) -> None:
    from robocasa.models.fixtures.counter import Counter

    native = environment.unwrapped.env
    if getattr(native.sim._render_context_offscreen, "_roboscope_owning_context_cleanup", None) != 2:
        raise RuntimeError("rebuild RoboCasa365 runtime with the owning-context cleanup patch")
    profile = getattr(args, "runtime_profile", "official")
    stable_regions = getattr(Counter, "_roboscope_reset_region_order", None)
    if (profile == "paired-stable-regions-v1") != (stable_regions == "xml-insertion-v1"):
        raise RuntimeError("RoboCasa365 runtime profile differs from the installed reset-region patch")
    if (args.render_mode == "consumed" or args.diagnostic_timings) and (
        getattr(native, "_roboscope_camera_gate_version", None) != 1
    ):
        raise RuntimeError("rebuild RoboCasa365 runtime with the committed camera patch")
    if args.render_mode == "consumed":
        if not native.ignore_done or native.viewer_get_obs:
            raise RuntimeError("consumed rendering requires the pinned headless ignore_done runtime")
        if any(native.camera_depths) or any(native.camera_segmentations):
            raise RuntimeError("consumed rendering supports RGB without depth or segmentation")
        if native.sim.model._model.nplugin:
            raise RuntimeError("consumed rendering requires a model without visualization plugins")
    native._roboscope_measure_camera_render = args.diagnostic_timings


def camera_counters(environments: list) -> dict[str, float]:
    return {
        key: sum(float(getattr(environment.unwrapped.env, attribute, 0)) for environment in environments)
        for key, attribute in (
            ("camera_render_s", "_roboscope_camera_render_s"),
            ("camera_render_calls", "_roboscope_camera_render_calls"),
            ("camera_reuse_calls", "_roboscope_camera_reuse_calls"),
        )
    }


def run_group(
    args: argparse.Namespace, task: Task, environments: list,
    stream_episode: int, client: PolicyClient,
) -> tuple[list[EvaluationRecord], dict[str, float]]:
    from robocasa.utils.env_utils import convert_action

    count = len(environments)
    indices = [stream_episode * count + slot for slot in range(count)]
    identifiers = [episode_id(task.name, index) for index in indices]
    histories: list[deque] = []
    fingerprints = []
    reset_started = time.perf_counter()
    with environment_range("environment.reset"):
        for slot, environment in enumerate(environments):
            rng_before = None
            if args.audit_trajectories:
                from .audit_state import rng_signature

                rng_before = rng_signature(environment.unwrapped.env)["sha256"]
            # Native labels are not reset seeds: continue the constructor stream.
            native, _ = environment.reset()
            history = deque((native, native), maxlen=OBSERVATION_STEPS)
            histories.append(history)
            fingerprints.append(reset_fingerprint(
                environment, build_observation(identifiers[slot], 0, history),
                complete=args.audit_trajectories,
            ))
            if rng_before is not None:
                fingerprints[-1]["rng_before_reset_sha256"] = rng_before
    reset_s = time.perf_counter() - reset_started
    if args.audit_trajectories:
        from .pairing import publish_and_check_initial

        publish_and_check_initial(args, task.name, stream_episode, identifiers, fingerprints)
    cameras_before = camera_counters(environments)
    started = time.perf_counter()
    steps = [0] * count
    successes = [False] * count
    dones = [False] * count
    recent_success = [deque(maxlen=OBSERVATION_STEPS) for _ in environments]
    inference_s = 0.0
    stepping_s = 0.0
    video_s = 0.0
    observation_s = 0.0
    conversion_s = 0.0
    lifecycle_s = 0.0
    evidence_s = 0.0
    audit_s = 0.0
    inference_calls = 0
    fresh_observation_steps = 0
    hashes = [
        {key: hashlib.sha256() for key in (
            "physical_state", "integration_state", "policy_observations", "policy_actions",
        )}
        for _ in environments
    ] if args.audit_trajectories else []
    audit_batches = []
    termination = "horizon"
    with ExitStack() as stack:
        writers = []
        for identifier in identifiers:
            if args.save_videos:
                import imageio.v2 as imageio

                path = args.output_dir / "videos" / f"worker-{args.worker_id:03d}" / f"{identifier}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(
                    path, fps=10, codec="libx264", quality=None, ffmpeg_params=["-crf", "22"]
                )
                stack.callback(writer.close)
                writers.append(writer)
            else:
                writers.append(None)
        reset_ids: list[str] = []
        if profiling.ENABLED:
            profiling.episode_start()
        try:
            for slot, identifier in enumerate(identifiers):
                context = EpisodeContext(identifier, task.name, {
                    "episode_index": indices[slot], "slot": slot,
                    "stream_episode": stream_episode, "constructor_seed": CONSTRUCTOR_SEED + slot,
                    "scene_split": SCENE_SPLIT, "native_envs": count,
                })
                lifecycle_started = time.perf_counter()
                if profiling.ENABLED:
                    # PolicyClient's automatic range assumes serial episodes.
                    # Keep one capture open for the entire native group instead.
                    client._request("reset", {"context": context.to_dict()})
                else:
                    client.reset(context)
                lifecycle_s += time.perf_counter() - lifecycle_started
                reset_ids.append(identifier)
            while not all(dones):
                observation_started = time.perf_counter()
                observations = tuple(
                    build_observation(identifier, steps[slot], histories[slot])
                    for slot, identifier in enumerate(identifiers)
                )
                observation_s += time.perf_counter() - observation_started
                inference_started = time.perf_counter()
                # Native inference includes done slots; their wrapper ignores actions.
                chunks = client.act_batch(observations)
                inference_s += time.perf_counter() - inference_started
                inference_calls += 1
                if len(chunks) != count:
                    raise RuntimeError("RoboCasa365 Policy returned an incomplete action batch")
                if stream_episode == 0 and not any(steps):
                    evidence_started = time.perf_counter()
                    save_first_batch(args, task.name, observations, chunks)
                    evidence_s += time.perf_counter() - evidence_started
                if args.audit_trajectories:
                    audit_started = time.perf_counter()
                    batch = []
                    for slot, (observation, chunk) in enumerate(zip(observations, chunks, strict=True)):
                        digest = feature_digest(observation)
                        hashes[slot]["policy_observations"].update(digest)
                        actions_digest = hashlib.sha256(np.asarray(chunk.actions).tobytes()).digest()
                        hashes[slot]["policy_actions"].update(actions_digest)
                        batch.append({
                            "episode_id": observation.episode_id, "step": observation.step,
                            "observation_sha256": digest.hex(), "actions_sha256": actions_digest.hex(),
                        })
                    audit_batches.append(batch)
                    audit_s += time.perf_counter() - audit_started
                for slot, chunk in enumerate(chunks):
                    actions = validate_actions(chunk.actions)
                    chunk_steps = min(len(actions), task.horizon - steps[slot])
                    for offset, action in enumerate(actions):
                        if dones[slot]:
                            break
                        render_images = (
                            args.render_mode == "all" or offset >= chunk_steps - OBSERVATION_STEPS
                            or (writers[slot] is not None and (steps[slot] + 1) % 2 == 1)
                        )
                        native_env = environments[slot].unwrapped.env
                        native_env._roboscope_render_images = render_images
                        conversion_started = time.perf_counter()
                        converted = convert_action(action)
                        conversion_s += time.perf_counter() - conversion_started
                        step_started = time.perf_counter()
                        try:
                            with environment_range("environment.step"):
                                native, _, terminated, truncated, info = environments[slot].step(converted)
                        finally:
                            native_env._roboscope_render_images = True
                        stepping_s += time.perf_counter() - step_started
                        fresh_observation_steps += int(render_images)
                        histories[slot].append(native)
                        steps[slot] += 1
                        success = bool(info["success"])
                        recent_success[slot].append(success)
                        successes[slot] |= success
                        dones[slot] = bool(terminated or truncated or steps[slot] >= task.horizon)
                        if args.audit_trajectories:
                            from .audit import integration_state_bytes

                            audit_started = time.perf_counter()
                            digest = hashes[slot]["physical_state"]
                            state = np.asarray(native_env.sim.get_state().flatten(), dtype=np.float64)
                            digest.update(state.tobytes())
                            digest.update(np.asarray(native_env.sim.data.ctrl, dtype=np.float64).tobytes())
                            digest.update(bytes((success, dones[slot])))
                            integration = hashes[slot]["integration_state"]
                            integration.update(integration_state_bytes(native_env))
                            integration.update(bytes((success, dones[slot])))
                            audit_s += time.perf_counter() - audit_started
                        # Native VideoRecordingWrapper starts its counter at 1
                        # after reset, so a 20 Hz / 10 fps run records steps 1, 3, ...
                        if writers[slot] is not None and steps[slot] % 2 == 1:
                            video_started = time.perf_counter()
                            writers[slot].append_data(native["video.robot0_agentview_right"])
                            video_s += time.perf_counter() - video_started
                # Native info['success'][0] is FIRST of its latest two step values.
                if all(values and values[0] for values in recent_success):
                    termination = "native_group_success"
                    break
            if termination != "native_group_success" and any(step < task.horizon for step in steps):
                termination = "simulator_done"
        finally:
            try:
                for identifier in reset_ids:
                    lifecycle_started = time.perf_counter()
                    if profiling.ENABLED:
                        client._request("end_episode", {"episode_id": identifier})
                    else:
                        client.end_episode(identifier)
                    lifecycle_s += time.perf_counter() - lifecycle_started
            finally:
                if profiling.ENABLED:
                    profiling.episode_end()
    elapsed_s = time.perf_counter() - started
    end_signatures = [{} for _ in environments]
    if args.audit_trajectories:
        from .audit_state import native_control_signature, rng_signature, sensor_schedule_signature

        for slot, environment in enumerate(environments):
            native_env = environment.unwrapped.env
            end_signatures[slot] = {
                "final_rng_sha256": rng_signature(native_env)["sha256"],
                "final_controller_sha256": native_control_signature(native_env)["sha256"],
                "final_sensor_schedule_sha256": sensor_schedule_signature(native_env)["sha256"],
            }
        evidence_started = time.perf_counter()
        path = (
            args.output_dir / "evidence" / f"worker-{args.worker_id:03d}"
            / f"{task.name}-group-{stream_episode:02d}-audit.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 2, "task": task.name, "batches": audit_batches,
        }, separators=(",", ":")))
        evidence_s += time.perf_counter() - evidence_started
    records = [
        EvaluationRecord(
            identifiers[slot], successes[slot], steps[slot], elapsed_s,
            {
                "task": task.name, "split": task.split, "scene_split": SCENE_SPLIT,
                "episode_index": indices[slot], "rollout_label": ROLLOUT_LABEL_BASE + indices[slot],
                "slot": slot, "stream_episode": stream_episode,
                "constructor_seed": CONSTRUCTOR_SEED + slot,
                "horizon": task.horizon, "termination": termination,
                "group_elapsed_s": elapsed_s, "group_reset_s": reset_s,
                "render_mode": args.render_mode,
                "runtime_profile": args.runtime_profile,
                **({"trajectory_sha256": {key: value.hexdigest() for key, value in hashes[slot].items()}}
                   if args.audit_trajectories else {}),
                **fingerprints[slot],
                **end_signatures[slot],
            },
        )
        for slot in range(count)
    ]
    cameras_after = camera_counters(environments)
    return records, {
        "reset_s": reset_s, "inference_wait_s": inference_s,
        "environment_step_s": stepping_s, "video_write_s": video_s,
        "observation_build_s": observation_s, "action_conversion_s": conversion_s,
        "policy_lifecycle_s": lifecycle_s, "evidence_write_s": evidence_s, "audit_s": audit_s,
        "policy_batch_calls": inference_calls, "physical_steps": sum(steps),
        "fresh_observation_steps": fresh_observation_steps,
        **{key: cameras_after[key] - value for key, value in cameras_before.items()},
    }


def run_task(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    task = TASKS_BY_NAME[str(item.payload["task"])]
    episodes = int(item.payload["episodes"])
    count = int(item.payload["native_envs"])
    if count not in (1, NATIVE_ENVS) or not 1 <= episodes <= EPISODES_PER_TASK or episodes % count:
        raise ValueError("invalid RoboCasa365 task workload")
    environments = []
    records = []
    timings = {
        "reset_s": 0.0, "inference_wait_s": 0.0, "environment_step_s": 0.0, "video_write_s": 0.0,
        "observation_build_s": 0.0, "action_conversion_s": 0.0,
        "policy_lifecycle_s": 0.0, "evidence_write_s": 0.0, "audit_s": 0.0,
        "camera_render_s": 0.0, "camera_render_calls": 0.0, "camera_reuse_calls": 0.0,
        "policy_batch_calls": 0, "physical_steps": 0, "fresh_observation_steps": 0,
    }
    started = time.perf_counter()
    try:
        with environment_range("environment.create"):
            for slot in range(count):
                environments.append(create_environment(task, slot))
                configure_rendering(environments[-1], args)
        timings["environment_create_s"] = time.perf_counter() - started
        with PolicyClient(args.policy_socket, timeout_s=300) as client:
            for group in range(episodes // count):
                group_records, group_timing = run_group(args, task, environments, group, client)
                records.extend(group_records)
                for key, value in group_timing.items():
                    timings[key] += value
                WorkQueue(args.queue_dir).publish_partial(item, WorkResult(
                    item.item_id, item.group_id, args.worker_id, tuple(records),
                    {"task": task.name, "timings": dict(timings)},
                ))
    finally:
        close_started = time.perf_counter()
        for environment in environments:
            environment.close()
        timings["environment_close_s"] = time.perf_counter() - close_started
    timings["task_elapsed_s"] = time.perf_counter() - started
    return WorkResult(
        item.item_id, item.group_id, args.worker_id, tuple(records),
        {"task": task.name, "native_envs": count, "render_mode": args.render_mode,
         "audit_trajectories": args.audit_trajectories,
         "diagnostic_timings": args.diagnostic_timings, "timings": timings},
    )


def main() -> None:
    args = parse_args()
    validate_upstream()
    WorkerLoop(
        worker_id=args.worker_id, queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir, poll_interval_s=args.poll_interval,
    ).run(lambda item: run_task(args, item))


if __name__ == "__main__":
    main()
