"""Read-only, bounded Python-state signatures for paired runtime diagnostics.

These signatures cover the fields listed below, not a complete Python object
clone. They never sample an RNG, run a controller, refresh a sensor, or include
object addresses. Pass the native Kitchen instance, not its Gym wrapper.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

_CONTROLLER_FIELDS = (
    "goal_pos", "goal_ori", "goal_qpos", "goal_qvel", "goal_vel", "init_pos", "init_ori",
    "current_vel", "relative_ori", "ori_ref",
    "origin_pos", "origin_ori", "_goal_update_mode", "ref_pos", "ref_ori_mat", "ref_pos_vel",
    "ref_ori_vel", "joint_pos", "joint_vel", "initial_joint", "initial_ref_pos", "initial_ref_ori_mat",
    "mass_matrix", "J_pos", "J_ori", "J_full", "torques", "new_update", "last_err", "summed_err",
    "saturated", "last_joint_vel", "kp", "kd", "ki", "action_scale", "action_input_transform",
    "action_output_transform", "input_type", "input_ref_frame", "impedance_mode", "control_freq",
    "model_timestep", "lite_physics", "control_dim", "input_min", "input_max", "output_min", "output_max",
    "position_limits", "velocity_limits", "actuator_min", "actuator_max", "_action_split_indexes",
)
_NESTED_CONTROLLER_FIELDS = ("interpolator", "interpolator_pos", "interpolator_ori", "derr_buf")
_BUFFER_FIELDS = (
    "start", "goal", "step", "total_steps", "dim", "order", "ori_interpolate", "controller_freq",
    "policy_freq", "ramp_ratio", "buf", "ptr", "_size", "length", "last", "current",
)
_ROBOT_FIELDS = (
    "base_pos", "base_ori", "init_qpos", "init_torso_qpos", "recent_qpos", "recent_actions",
    "recent_torques", "recent_ee_forcetorques", "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer",
    "recent_ee_acc",
)
_FIXTURE_FIELDS = (
    "pos", "_lid", "_switch", "_lid_button", "_lid_speed", "_turned_on", "_num_steps_on",
    "_cooldown_time", "_last_lid_update", "_target_lid_angle",
)
_SENSOR_FIELDS = (
    "_time_since_last_sample", "_current_delay", "_sampled", "_enabled", "_active", "_sampling_timestep",
)


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _primitive(value: Any) -> Any:
    """Encode only known value containers; reject references and object arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, np.generic):
        return _primitive(np.asarray(value))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are outside signature coverage")
        return {
            "dtype": value.dtype.str, "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("only string mapping keys are covered")
        return {key: _primitive(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, deque)):
        return {"container": type(value).__name__, "values": [_primitive(item) for item in value]}
    raise TypeError(f"unsupported value type {_type_name(value)}")


def _summary(states: Any, coverage: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps({"states": states, "coverage": coverage}, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "coverage": coverage}


def rng_signature(native: Any) -> dict[str, Any]:
    """Hash located NumPy RNG states and deterministic sharing relationships."""
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    unsupported: list[dict[str, str]] = []
    aliases: dict[int, str] = {}
    sampler_aliases: dict[int, str] = {}

    def add_rng(owner: Any, path: str) -> None:
        rng = vars(owner).get("rng")
        if rng is None:
            missing.append(f"{path}.rng")
            return
        location = f"{path}.rng"
        first = aliases.setdefault(id(rng), location)
        if isinstance(rng, np.random.Generator):
            state = _primitive(rng.bit_generator.state)
        elif isinstance(rng, np.random.RandomState):
            state = _primitive(rng.get_state())
        else:
            unsupported.append({"path": location, "type": _type_name(rng)})
            return
        entries.append({"path": location, "alias_of": first, "type": _type_name(rng), "state": state})

    def add_sampler(sampler: Any, path: str) -> None:
        first = sampler_aliases.setdefault(id(sampler), path)
        if first != path:
            return
        add_rng(sampler, path)
        children = vars(sampler).get("samplers", ())
        if isinstance(children, Mapping):
            for name in sorted(children):
                add_sampler(children[name], f"{path}.samplers[{name}]")
        elif isinstance(children, (list, tuple)):
            for index, child in enumerate(children):
                add_sampler(child, f"{path}.samplers[{index}]")

    add_rng(native, "environment")
    for index, robot in enumerate(native.robots):
        add_rng(robot, f"robots[{index}]")
    for name, fixture in sorted(native.fixtures.items()):
        add_rng(fixture, f"fixtures[{name}]")
    initializer = vars(native).get("placement_initializer")
    if initializer is not None:
        add_sampler(initializer, "placement_initializer")
    return _summary(entries, {
        "scope": "located NumPy RNG states; environment, robots, fixtures and nested placement samplers",
        "rng_paths": [entry["path"] for entry in entries], "missing_rng_paths": missing,
        "located_rng_states": [{
            "path": entry["path"], "alias_of": entry["alias_of"], "type": entry["type"],
            "state_sha256": hashlib.sha256(json.dumps(
                entry["state"], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        } for entry in entries],
        "unsupported": unsupported, "sampler_count": len(sampler_aliases),
        "unique_rng_count": len(aliases), "complete_python_clone": False,
    })


def native_control_signature(native: Any) -> dict[str, Any]:
    """Hash a whitelist of existing control, buffer and fixture values."""
    states: dict[str, Any] = {}
    included: list[str] = []
    unsupported: list[dict[str, str]] = []

    def record(value: Any, path: str) -> None:
        try:
            states[path] = _primitive(value)
            included.append(path)
        except TypeError:
            unsupported.append({"path": path, "type": _type_name(value)})

    def fields(owner: Any, path: str, names: tuple[str, ...]) -> None:
        values = vars(owner)
        for name in names:
            if name in values:
                record(values[name], f"{path}.{name}")

    def buffer(value: Any, path: str) -> None:
        if value is None:
            record(None, path)
        elif isinstance(value, Mapping):
            for name in sorted(value):
                buffer(value[name], f"{path}[{name}]")
        else:
            fields(value, path, _BUFFER_FIELDS)

    def controller(value: Any, path: str) -> None:
        fields(value, path, _CONTROLLER_FIELDS)
        for name in _NESTED_CONTROLLER_FIELDS:
            if name in vars(value):
                buffer(vars(value)[name], f"{path}.{name}")

    fields(native, "environment", (
        "cur_time", "timestep", "done", "init_robot_base_pos", "init_robot_base_ori",
        "init_robot_base_pos_anchor", "init_robot_base_ori_anchor",
    ))
    for index, robot in enumerate(native.robots):
        path = f"robots[{index}]"
        for name in _ROBOT_FIELDS:
            if name in vars(robot):
                value = vars(robot)[name]
                if name.startswith("recent_"):
                    buffer(value, f"{path}.{name}")
                else:
                    record(value, f"{path}.{name}")
        composite = vars(robot).get("composite_controller")
        if composite is not None:
            controller(composite, f"{path}.composite_controller")
            for name, part in sorted(vars(composite).get("part_controllers", {}).items()):
                controller(part, f"{path}.part_controllers[{name}]")
        for name, gripper in sorted(vars(robot).get("gripper", {}).items()):
            fields(gripper, f"{path}.gripper[{name}]", ("current_action",))
    for name, fixture in sorted(native.fixtures.items()):
        fields(fixture, f"fixtures[{name}]", _FIXTURE_FIELDS)
    return _summary(states, {
        "scope": "whitelisted existing environment/controller/interpolator/buffer/fixture fields",
        "included_fields": included, "unsupported": unsupported, "complete_python_clone": False,
    })


def sensor_schedule_signature(native: Any) -> dict[str, Any]:
    """Hash sensor clocks and non-image values, excluding discarded RGB caches."""
    states: dict[str, Any] = {}
    image_names: list[str] = []
    unsupported: list[dict[str, str]] = []
    names: list[str] = []

    def record(value: Any, path: str) -> None:
        try:
            states[path] = _primitive(value)
        except TypeError:
            unsupported.append({"path": path, "type": _type_name(value)})

    for name, observable in sorted(native._observables.items()):
        names.append(name)
        values = vars(observable)
        sensor = values.get("_sensor")
        image = getattr(sensor, "__modality__", None) == "image"
        if image:
            image_names.append(name)
        for field in _SENSOR_FIELDS:
            if field in values:
                record(values[field], f"observables[{name}].{field}")
        if not image and "_current_observed_value" in values:
            record(values["_current_observed_value"], f"observables[{name}].value")
    excluded_cache: list[str] = []
    for name, value in sorted(native._obs_cache.items()):
        if name in image_names or name.endswith(("_image", "_depth", "_segmentation")):
            excluded_cache.append(name)
        else:
            record(value, f"cache[{name}]")
    return _summary(states, {
        "scope": "all observable schedules and existing non-image values/cache; no sensor refresh",
        "observable_names": names, "excluded_image_values": image_names,
        "excluded_image_cache": excluded_cache, "unsupported": unsupported, "complete_python_clone": False,
    })
