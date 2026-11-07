# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Read-only input and RNG auditing around the unchanged official Policy.

Set ROBOCASA365_POLICY_AUDIT to an absolute JSON output path. This wrapper adds
no model calls, RNG draws, precision changes or CUDA synchronization. Diagnostic
runs should not be used as a production throughput comparison.
"""

from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from policy import RoboCasa365DiffusionPolicy

_EFFECTIVE_SOURCES = {
    "crop": "85787b6becd337352c2a1795f1e77be977c073134f91c3d0924b9d0e36c92978",
    "encoder": "dd9e422a19499613cb073d74ee65419359e806adfdb62cd40fa2c261ea87874a",
    "normalizer": "92322852fc04d956d0dd43a2e41fb461dee246bbf97ddc7940186a9307525ed3",
}


def effective_input_contract(model) -> dict:
    """Verify the actual loaded eval path before hashing CPU crop-equivalent inputs."""
    encoder = model.obs_encoder
    normalizer = model.normalizer
    if model.training or encoder.training or normalizer.training:
        raise RuntimeError("Effective input auditing requires the loaded eval model")
    if (type(encoder).__module__, type(encoder).__name__) != (
        "robomimic.models.obs_nets", "ObservationEncoder"
    ) or (type(normalizer).__module__, type(normalizer).__name__) != (
        "diffusion_policy.model.common.normalizer", "LinearNormalizer"
    ):
        raise RuntimeError("Effective input auditing encountered an unknown encoder/normalizer")
    verified_sources = {}
    for key, value in (("encoder", encoder), ("normalizer", normalizer)):
        digest = hashlib.sha256(Path(inspect.getfile(type(value))).read_bytes()).hexdigest()
        if digest != _EFFECTIVE_SOURCES[key]:
            raise RuntimeError(f"Effective input auditing encountered unknown {key} source")
        verified_sources[key] = digest
    images = {}
    for key, shape in encoder.obs_shapes.items():
        randomizers = [value for value in encoder.obs_randomizers[key] if value is not None]
        if key.endswith("_image"):
            if tuple(shape) != (3, 256, 256) or len(randomizers) != 1:
                raise RuntimeError(f"Effective input auditing encountered unknown image path: {key}")
            crop = randomizers[0]
            if (
                (type(crop).__module__, type(crop).__name__) != (
                    "diffusion_policy.model.vision.crop_randomizer", "CropRandomizer"
                ) or crop.training or tuple(crop.input_shape) != (3, 256, 256)
                or (crop.crop_height, crop.crop_width, crop.num_crops, crop.pos_enc) != (224, 224, 1, False)
            ):
                raise RuntimeError(f"Effective input auditing requires a fixed224 eval crop: {key}")
            digest = hashlib.sha256(Path(inspect.getfile(type(crop))).read_bytes()).hexdigest()
            if digest != _EFFECTIVE_SOURCES["crop"]:
                raise RuntimeError("Effective input auditing encountered unknown crop source")
            verified_sources["crop"] = digest
            images[key] = {"input_shape": list(shape), "crop_shape": [224, 224],
                           "rows": [16, 240], "columns": [16, 240], "training": False,
                           "num_crops": 1, "pos_enc": False}
        elif randomizers:
            raise RuntimeError(f"Effective input auditing encountered non-image randomizers: {key}")
    if set(images) != {
        "robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"
    }:
        raise RuntimeError("Effective input auditing requires exactly the three official cameras")
    return {
        "schema_version": 1, "images": images, "verified_source_sha256": verified_sources,
        "stage": "CPU prepared inputs restricted to the actual fixed eval crop, before saved normalization",
        "normalization": "verified pinned elementwise affine; equality commutes with fixed spatial selection",
        "non_image_inputs": "all retained unchanged", "extra_model_calls": 0,
        "extra_rng_draws": 0, "extra_gpu_copies": 0, "extra_cuda_synchronizations": 0,
        "does_not_establish_full_image_equality": True,
    }


def effective_inputs(values: dict, contract: dict) -> dict:
    result = dict(values)
    for key in contract["images"]:
        value = np.asarray(values[key])
        if value.shape != (2, 3, 256, 256) or value.dtype != np.float32:
            raise RuntimeError(f"Unexpected actual prepared image for effective audit: {key}")
        result[key] = value[..., 16:240, 16:240]
    return result


def arrays_hash(values: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.asarray(value)
        header = json.dumps([key, array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rng_hash(state) -> dict:
    return {
        "cpu": hashlib.sha256(state.cpu.numpy().tobytes()).hexdigest(),
        "cuda": hashlib.sha256(state.cuda.numpy().tobytes()).hexdigest()
        if state.cuda is not None else None,
    }


class AuditedRoboCasa365Policy(RoboCasa365DiffusionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.output = Path(os.environ["ROBOCASA365_POLICY_AUDIT"])
        if not self.output.is_absolute() or self.output.exists():
            raise ValueError("ROBOCASA365_POLICY_AUDIT must be a fresh absolute JSON path")
        self.rows: list[dict] = []
        self.resets: list[dict] = []
        self.metadata: dict = {}
        self.prepared_rows: list[dict] | None = None
        self.flush_interval = int(os.environ.get("ROBOCASA365_POLICY_AUDIT_INTERVAL", "10"))
        if self.flush_interval < 1:
            raise ValueError("Policy audit interval must be positive")
        atexit.register(self.flush)

    def load(self, context):
        capabilities = super().load(context)
        import torch

        self.metadata = {
            "context": context.to_dict(), "sdk_version": version("roboscope-kit"),
            "policy_source_sha256": hashlib.sha256(
                Path(__file__).with_name("policy.py").read_bytes()
            ).hexdigest(),
            "audit_wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": torch.__version__, "dtype": str(self.model.dtype),
            "model_training": self.model.training,
            "num_inference_steps": self.model.num_inference_steps,
            "n_action_steps": self.model.n_action_steps, "n_obs_steps": self.model.n_obs_steps,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "initial_rng": rng_hash(self.initial_rng),
            "effective_input_contract": effective_input_contract(self.model),
        }
        return capabilities

    def reset(self, context) -> None:
        super().reset(context)
        self.resets.append({"context": context.to_dict(), "rng": rng_hash(self.streams[context.task_id])})

    def prepare_observation(self, observation):
        result = super().prepare_observation(observation)
        if self.prepared_rows is not None:
            self.prepared_rows.append({
                "episode_id": observation.episode_id, "step": observation.step,
                "prepared_sha256": arrays_hash(result),
                "effective_input_sha256": arrays_hash(effective_inputs(
                    result, self.metadata["effective_input_contract"]
                )),
            })
        return result

    def act_batch(self, observations):
        task = self.episodes[observations[0].episode_id].task
        before = rng_hash(self.streams[task])
        self.prepared_rows = []
        try:
            result = super().act_batch(observations)
            self.rows.append({
                "index": len(self.rows), "task": task, "before": before,
                "after": rng_hash(self.streams[task]), "inputs": self.prepared_rows,
                "actions": [{
                    "episode_id": observation.episode_id, "step": observation.step,
                    "sha256": hashlib.sha256(chunk.actions.tobytes()).hexdigest(),
                } for observation, chunk in zip(observations, result, strict=True)],
            })
            if len(self.rows) % self.flush_interval == 0:
                self.flush()
            return result
        finally:
            self.prepared_rows = None

    def end_episode(self, episode_id: str) -> None:
        super().end_episode(episode_id)
        if not self.episodes:
            self.flush()

    def unload(self) -> None:
        self.flush()
        super().unload()

    def flush(self) -> None:
        if not self.metadata:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "diagnostic_only": True,
            "metadata": self.metadata, "resets": self.resets, "batches": self.rows,
            "extra_model_calls": 0, "extra_rng_draws": 0, "extra_cuda_synchronizations": 0,
        }
        with tempfile.NamedTemporaryFile("w", dir=self.output.parent, delete=False) as stream:
            json.dump(payload, stream, separators=(",", ":"))
            temporary = Path(stream.name)
        os.replace(temporary, self.output)


if __name__ == "__main__":
    from roboscope_kit.server import run

    run(AuditedRoboCasa365Policy())
