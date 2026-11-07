# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Read-only signatures for matched RoboCasa365 simulator comparisons.

``MjSim.get_state`` contains only time, qpos and qvel. Integration signatures
instead use MuJoCo's full integration-state specification, including warmstart,
control, applied forces, equality activation, mocap, userdata and plugin state.

The model signature covers every public primitive array/scalar/byte/text field
exposed by the pinned Python MjModel binding, every mjOption field and mjStatistic.
It is not a hash of raw mjModel memory, its visual-option structs, Python
controllers, observables or RNGs. Those require separate signatures. Only RGBA
rows of collision-group-zero geoms hidden by the current renderer are omitted;
collision shapes, masses, materials, textures and all other array fields remain.
No getters that update controllers, fixtures or simulation state are invoked.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Number
from typing import Any

import numpy as np

MUJOCO_VERSION = "3.3.1"
SIGNATURE_SCHEMA = 1

# All fields of mjOption in the pinned MuJoCo 3.3.1 mjmodel.h. Enumerations
# become integers explicitly; no binding object representation enters a digest.
_OPTION_INTEGER_FIELDS = (
    "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations",
    "noslip_iterations", "ccd_iterations", "disableflags", "enableflags",
    "disableactuator", "sdf_initpoints", "sdf_iterations",
)
_OPTION_NUMERIC_FIELDS = (
    "timestep", "apirate", "impratio", "tolerance", "ls_tolerance",
    "noslip_tolerance", "ccd_tolerance", "gravity", "wind", "magnetic",
    "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
)
_STATISTIC_FIELDS = ("meaninertia", "meanmass", "meansize", "extent", "center")


def _native_parts(native):
    import mujoco

    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"RoboCasa365 audit requires MuJoCo {MUJOCO_VERSION}")
    if len(native.robots) != 1 or type(native.robots[0].robot_model).__name__ != "PandaOmron":
        raise RuntimeError("RoboCasa365 audit requires the pinned single PandaOmron embodiment")
    model = native.sim.model._model
    data = native.sim.data._data
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("RoboCasa365 audit requires the pinned native MuJoCo model/data bindings")
    return mujoco, model, data


def integration_state_bytes(native) -> bytes:
    """Read all mjSTATE_INTEGRATION fields without forward/step or state mutation."""
    mujoco, model, data = _native_parts(native)
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    values = np.empty(mujoco.mj_stateSize(model, specification), dtype=np.float64)
    mujoco.mj_getState(model, data, values, specification)
    if not np.isfinite(values).all():
        raise RuntimeError("RoboCasa365 integration state contains non-finite values")
    return values.tobytes(order="C")


def _feed(digest, name: str, value: Any) -> None:
    if isinstance(value, str):
        header, payload = {"name": name, "kind": "utf8"}, value.encode("utf-8")
    elif isinstance(value, bytes):
        header, payload = {"name": name, "kind": "bytes"}, value
    else:
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in "biufcSU":
            raise TypeError(f"Unsupported primitive model field {name}: {array.dtype}")
        array = np.ascontiguousarray(array)
        header = {"name": name, "kind": "array", "dtype": array.dtype.str, "shape": list(array.shape)}
        payload = memoryview(array).cast("B") if array.size else b""
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def initial_signatures(environment) -> dict[str, Any]:
    """Describe initial integration state and model data; never alter the environment."""
    native = environment.unwrapped.env
    _, model, _ = _native_parts(native)
    state = integration_state_bytes(native)
    context = native.sim._render_context_offscreen
    if context is None:
        raise RuntimeError("RoboCasa365 model audit requires its initialized offscreen renderer")
    enabled_groups = np.asarray(context.vopt.geomgroup, dtype=np.uint8)
    groups = np.asarray(model.geom_group)
    if enabled_groups.shape != (6,) or groups.shape != (model.ngeom,):
        raise RuntimeError("Unexpected pinned renderer geometry-group layout")
    if np.any(groups < 0) or np.any(groups >= len(enabled_groups)):
        raise RuntimeError("Compiled geometry groups exceed the pinned renderer group layout")
    # RoboSuite assigns collision geometry to group 0. Do not omit RGBA for
    # hidden groups other than 0, or any currently visible collision geometry.
    omitted_rgba = (groups == 0) & (enabled_groups[groups] == 0)
    digest = hashlib.sha256()
    _feed(digest, "signature_schema", SIGNATURE_SCHEMA)
    _feed(digest, "mujoco_version", MUJOCO_VERSION)
    _feed(digest, "renderer.geomgroup", enabled_groups)
    _feed(digest, "geom_rgba.omitted_rows", np.flatnonzero(omitted_rgba))
    primitive_fields = []
    unrepresented_fields = []
    for name in sorted(name for name in dir(model) if not name.startswith("_")):
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            if name == "geom_rgba":
                if value.shape != (model.ngeom, 4):
                    raise RuntimeError("Unexpected pinned geometry RGBA layout")
                _feed(digest, "geom_rgba.full_shape", np.asarray(value.shape, dtype=np.int64))
                value = value[~omitted_rgba]
            _feed(digest, f"model.{name}", value)
            primitive_fields.append(name)
        elif isinstance(value, (str, bytes, Number, np.generic)):
            _feed(digest, f"model.{name}", value)
            primitive_fields.append(name)
        elif not callable(value):
            unrepresented_fields.append(name)
    for name in _OPTION_INTEGER_FIELDS:
        _feed(digest, f"model.opt.{name}", int(getattr(model.opt, name)))
    for name in _OPTION_NUMERIC_FIELDS:
        _feed(digest, f"model.opt.{name}", getattr(model.opt, name))
    for name in _STATISTIC_FIELDS:
        _feed(digest, f"model.stat.{name}", getattr(model.stat, name))
    return {
        "initial_integration_state_sha256": hashlib.sha256(state).hexdigest(),
        "compiled_model_signature_sha256": digest.hexdigest(),
        "compiled_model_signature_coverage": {
            "schema_version": SIGNATURE_SCHEMA,
            "mujoco_version": MUJOCO_VERSION,
            "integration_state_specification": "mjSTATE_INTEGRATION",
            "integration_state_values": len(state) // np.dtype(np.float64).itemsize,
            "primitive_model_fields": primitive_fields,
            "option_fields": list(_OPTION_INTEGER_FIELDS + _OPTION_NUMERIC_FIELDS),
            "statistic_fields": list(_STATISTIC_FIELDS),
            "unrepresented_public_model_structures": unrepresented_fields,
            "omitted_invisible_collision_geom_rgba_rows": np.flatnonzero(omitted_rgba).tolist(),
            "renderer_geomgroup": enabled_groups.tolist(),
            "scope": "primitive model fields and opt/stat; not raw model memory, vis or Python/RNG state",
        },
    }
