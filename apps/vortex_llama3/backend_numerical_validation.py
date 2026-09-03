# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Shared C1/C3 package and GPU-reference validation utilities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from tvm.relax.backend.vortex import (
    BackendParameterArchive,
    LogicalParameterArchive,
    get_vortex_backend_policy,
)

from vortex_llama3.compile_backend_matrix import (
    CASES,
    NUM_LAYERS,
    _import_dependencies,
    _resolve_package_path,
    load_compile_package,
)


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_FORMAT = "vortex-llama3-backend-gpu-reference"
REFERENCE_SEED = 20260831
DECODE_STEPS = 3
SUPPORTED_ALIASES = ("C1", "C3")
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class LoadedBackendPackage:
    """A fail-closed compile package and its checked parameter archives."""

    path: Path
    package: dict
    logical: LogicalParameterArchive
    materialized: BackendParameterArchive
    linear_compute: str
    attention_compute: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def package_identity(package: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable package fields carried by every reference."""

    profile = package["profile"]
    return {
        "alias": package["alias"],
        "shape_case": package["shape_case"],
        "shape": package["shape"],
        "backend_policy": package["backend_policy"],
        "workload_variant": package["workload_variant"],
        "layout_policy": package["layout_policy"],
        "model": package["model"],
        "profile_fingerprint": profile["profile_fingerprint"],
        "xclbin": profile["xclbin"],
        "xclbin_sha256": profile["xclbin_sha256"],
        "logical_archive_manifest_sha256": package["logical_archive_manifest_sha256"],
        "logical_content_sha256": package["logical_content_sha256"],
        "materialization_manifest_sha256": package["materialization_manifest_sha256"],
        "revisions": package["revisions"],
    }


def load_backend_package(
    package_path: str | Path,
    alias_map: str | Path,
    vortex_home: str | Path,
    *,
    expected_alias: str | None = None,
    expected_case: str | None = None,
) -> LoadedBackendPackage:
    """Validate exact alias/profile/artifact/archive identity before execution."""

    package_path = Path(package_path).resolve()
    alias_map = Path(alias_map).resolve()
    vortex_home = Path(vortex_home).resolve()
    dependencies = _import_dependencies(vortex_home)
    package = load_compile_package(package_path, alias_map, dependencies)
    alias = package["alias"]
    shape_case = package["shape_case"]
    if alias not in SUPPORTED_ALIASES:
        raise ValueError(f"numerical validation does not support alias {alias!r}")
    if expected_alias is not None and alias != expected_alias:
        raise ValueError(
            f"backend package alias mismatch: expected {expected_alias}, got {alias}"
        )
    if expected_case is not None and shape_case != expected_case:
        raise ValueError(
            f"backend package case mismatch: expected {expected_case}, got {shape_case}"
        )
    root = package_path.parent
    logical_manifest = _resolve_package_path(package["logical_archive_manifest"], root)
    logical = LogicalParameterArchive(
        logical_manifest,
        expected_num_layers=NUM_LAYERS,
        expected_model_metadata=package["model"],
    )
    materialization_manifest = _resolve_package_path(
        package["materialization_manifest"], root
    )
    materialized = BackendParameterArchive(
        materialization_manifest,
        expected_policy=package["backend_policy"],
        expected_profile_fingerprint=package["profile"]["profile_fingerprint"],
        expected_logical_manifest_sha256=logical.manifest_sha256,
        expected_logical_content_sha256=logical.content_sha256,
    )
    policy = get_vortex_backend_policy(package["backend_policy"])
    linear_compute = "fp16" if policy.linear_compute == "fp16_tcu" else "w4"
    attention_compute = "fp16" if policy.attention_compute == "fp16_tcu" else "w4"
    return LoadedBackendPackage(
        path=package_path,
        package=package,
        logical=logical,
        materialized=materialized,
        linear_compute=linear_compute,
        attention_compute=attention_compute,
    )


def deterministic_prompt(shape_case: str) -> list[list[int]]:
    """Return the frozen, easily audited S1-S4 token matrix."""

    if shape_case not in CASES:
        raise ValueError(f"unknown validation shape case: {shape_case!r}")
    batch, prompt_length, _ = CASES[shape_case]
    return [
        [1 + row * prompt_length + column for column in range(prompt_length)]
        for row in range(batch)
    ]


def _torch_parameters(archive: BackendParameterArchive, names, device):
    result = {}
    for name in names:
        value = archive.tensor(name)
        # Archive tensors are read-only memmaps.  Copy once before constructing
        # a Torch tensor so eager custom operations never receive a writable
        # view over immutable package bytes.
        result[name] = torch.from_numpy(np.array(value, copy=True)).to(device)
    return result


def _local_layer_parameters(parameters: Mapping[str, torch.Tensor]):
    return {
        name.split(".", 2)[2]: value
        for name, value in parameters.items()
        if name.startswith("layers.0.")
    }


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    if value.is_cuda:
        torch.cuda.synchronize(value.device)
    return value.detach().cpu().numpy().copy()


def _gpu_metadata(device: torch.device) -> dict[str, object]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            f"--id={index}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    driver_version, total_mib, free_mib = [part.strip() for part in driver.split(",")]
    return {
        "reference_device": "gpu",
        "torch_device": str(device),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "driver_version": driver_version,
        "memory_total_mib": int(total_mib),
        "memory_free_mib_before_parameters": int(free_mib),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def configure_reference_device(device_label: str) -> tuple[torch.device, dict]:
    """Configure the explicit GPU oracle or labeled CPU diagnostic path."""

    if device_label == "gpu":
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ):
            raise ValueError(
                "GPU reference requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("GPU reference requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        metadata = _gpu_metadata(device)
    elif device_label == "diagnostic_cpu":
        device = torch.device("cpu")
        metadata = {
            "reference_device": "diagnostic_cpu",
            "torch_device": "cpu",
            "torch_version": torch.__version__,
            "diagnostic_only": True,
        }
    else:
        raise ValueError(
            "reference device must be 'gpu' or the explicit 'diagnostic_cpu' fallback"
        )
    return device, metadata


def _record_tensor(arrays: dict[str, np.ndarray], name: str, value) -> None:
    if name in arrays:
        raise ValueError(f"duplicate reference tensor name: {name}")
    arrays[name] = value if isinstance(value, np.ndarray) else _to_numpy(value)


def generate_reference_arrays(
    loaded: LoadedBackendPackage,
    rows: Sequence[Sequence[int]],
    device: torch.device,
    *,
    decode_steps: int = DECODE_STEPS,
) -> dict[str, np.ndarray]:
    """Execute backend-matched eager inference and record every canonical boundary."""

    dependencies = _import_dependencies(
        Path(
            os.environ.get("VORTEX_HOME", "/home/jaeyongjang/project.local/vortex_base")
        )
    )
    from spinquant_inference.llama3_c4_export import (  # pylint: disable=import-outside-toplevel
        Llama3LayerDecodeCheckpoints,
        Llama3LayerPrefillCheckpoints,
        _rms_norm,
        layer_checkpoint_names,
    )

    package = loaded.package
    shape = package["shape"]
    if [len(rows), len(rows[0])] != [shape["batch_size"], shape["prompt_length"]]:
        raise ValueError("reference token matrix does not match package shape")
    config = dependencies["config"](
        shape["batch_size"], shape["prompt_length"], shape["cache_capacity"]
    )
    decode_config = dependencies["config"](
        shape["batch_size"], 1, shape["cache_capacity"]
    )
    orders = package["parameter_orders"]
    layer_parameters = _torch_parameters(loaded.materialized, orders["layer"], device)
    embedding_parameters = _torch_parameters(
        loaded.materialized, orders["embedding"], device
    )
    head_parameters = _torch_parameters(loaded.materialized, orders["head"], device)
    local_parameters = _local_layer_parameters(layer_parameters)
    embedding = dependencies["embedding"]().to(device)
    prefill = dependencies["prefill"](
        config,
        1,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    ).to(device)
    decode = dependencies["decode"](
        decode_config,
        1,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    ).to(device)
    head_prefill = dependencies["head"](
        config, linear_compute=loaded.linear_compute
    ).to(device)
    head_decode = dependencies["head"](
        decode_config, linear_compute=loaded.linear_compute
    ).to(device)
    checkpoint_prefill = Llama3LayerPrefillCheckpoints(
        config,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    ).to(device)
    checkpoint_decode = Llama3LayerDecodeCheckpoints(
        decode_config,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    ).to(device)
    checkpoint_names = layer_checkpoint_names(config)

    arrays: dict[str, np.ndarray] = {}
    states = None
    token_ids = np.asarray(rows, dtype="int64")
    positions = np.broadcast_to(
        np.arange(token_ids.shape[1], dtype="int64"), token_ids.shape
    ).copy()
    with torch.inference_mode():
        for phase_index in range(decode_steps + 1):
            token_torch = torch.from_numpy(token_ids).to(device)
            position_torch = torch.from_numpy(positions).to(device)
            (hidden,) = embedding(token_torch, embedding_parameters)
            _record_tensor(arrays, f"p{phase_index}_token_ids", token_ids.copy())
            _record_tensor(arrays, f"p{phase_index}_positions", positions.copy())
            _record_tensor(arrays, f"p{phase_index}_embedding", hidden)
            next_states = []
            for layer_index in range(NUM_LAYERS):
                _record_tensor(arrays, f"p{phase_index}_l{layer_index}_input", hidden)
                if phase_index == 0:
                    state = prefill(hidden, position_torch, layer_parameters)
                else:
                    state = decode(
                        hidden,
                        position_torch,
                        layer_parameters,
                        *states[layer_index][1:],
                    )
                hidden = state[0]
                next_states.append(state)
                for output_index, value in enumerate(state):
                    _record_tensor(
                        arrays,
                        f"p{phase_index}_l{layer_index}_o{output_index}",
                        value,
                    )
            states = next_states
            logits, normalized = (head_prefill if phase_index == 0 else head_decode)(
                hidden, head_parameters
            )
            _record_tensor(arrays, f"p{phase_index}_head_input", hidden)
            _record_tensor(arrays, f"p{phase_index}_normalized", normalized)
            _record_tensor(arrays, f"p{phase_index}_logits", logits)
            last_logits = _to_numpy(logits)[:, -1, :].astype("float32")
            # Stable ordering gives argmax-compatible token IDs when the
            # synthetic fixture intentionally creates tied logits.
            topk_indices = np.argsort(-last_logits, axis=-1, kind="stable")[:, :5]
            topk_values = np.take_along_axis(last_logits, topk_indices, axis=-1)
            selected = topk_indices[:, 0].astype("int64")
            _record_tensor(arrays, f"p{phase_index}_selected", selected)
            _record_tensor(arrays, f"p{phase_index}_topk_indices", topk_indices)
            _record_tensor(arrays, f"p{phase_index}_topk_values", topk_values)
            _record_tensor(
                arrays,
                f"p{phase_index}_top1_margin",
                (topk_values[:, 0] - topk_values[:, 1]).astype("float32"),
            )
            token_ids = selected[:, None]
            positions = np.full(
                (shape["batch_size"], 1),
                shape["prompt_length"] + phase_index,
                dtype="int64",
            )

        prefill_input = torch.from_numpy(arrays["p0_l0_input"]).to(device)
        prefill_positions = torch.from_numpy(arrays["p0_positions"]).to(device)
        checkpoint_values = checkpoint_prefill(
            prefill_input, prefill_positions, local_parameters
        )
        prefill_checkpoint_values = checkpoint_values
        for index, value in enumerate(checkpoint_values):
            suffix = (
                checkpoint_names[index]
                if index < len(checkpoint_names)
                else f"state_{index-len(checkpoint_names)}"
            )
            _record_tensor(arrays, f"probe_prefill_l0_{suffix}", value)

        decode_input = torch.from_numpy(arrays["p1_l0_input"]).to(device)
        decode_positions = torch.from_numpy(arrays["p1_positions"]).to(device)
        prefill_state = tuple(
            torch.from_numpy(arrays[f"p0_l0_o{index}"]).to(device)
            for index in range(1, 8)
        )
        checkpoint_values = checkpoint_decode(
            decode_input,
            decode_positions,
            local_parameters,
            *(
                value[0] if value.ndim > 0 and value.shape[0] == 1 else value
                for value in prefill_state
            ),
        )
        for index, value in enumerate(checkpoint_values):
            suffix = (
                checkpoint_names[index]
                if index < len(checkpoint_names)
                else f"state_{index-len(checkpoint_names)}"
            )
            _record_tensor(arrays, f"probe_decode_l0_{suffix}", value)

        attention_normalized = _rms_norm(
            prefill_input,
            local_parameters["input_norm.weight"],
            config.rms_norm_eps,
        )
        linear_inputs = {
            "q_proj": attention_normalized,
            "k_proj": attention_normalized,
            "v_proj": attention_normalized,
            "o_proj": prefill_checkpoint_values[5]
            .transpose(1, 2)
            .reshape(config.batch_size, config.query_length, config.hidden_size),
            "gate_proj": prefill_checkpoint_values[8],
            "up_proj": prefill_checkpoint_values[8],
            "down_proj": prefill_checkpoint_values[12],
        }
        linear_expected = {
            "q_proj": prefill_checkpoint_values[0],
            "k_proj": checkpoint_prefill._linear(
                "k_proj", attention_normalized, local_parameters
            ),
            "v_proj": checkpoint_prefill._linear(
                "v_proj", attention_normalized, local_parameters
            ),
            "o_proj": prefill_checkpoint_values[6],
            "gate_proj": prefill_checkpoint_values[9],
            "up_proj": prefill_checkpoint_values[10],
            "down_proj": prefill_checkpoint_values[13],
        }
        for projection_name in linear_inputs:
            output_size = linear_expected[projection_name].shape[-1]
            _record_tensor(
                arrays,
                f"focused_linear_{projection_name}_input",
                linear_inputs[projection_name].reshape(
                    -1, linear_inputs[projection_name].shape[-1]
                ),
            )
            _record_tensor(
                arrays,
                f"focused_linear_{projection_name}_expected",
                linear_expected[projection_name].reshape(-1, output_size),
            )
        # Preserve the original q-projection names for existing focused packages.
        _record_tensor(
            arrays, "focused_linear_input", arrays["focused_linear_q_proj_input"]
        )
        _record_tensor(
            arrays,
            "focused_linear_expected",
            arrays["focused_linear_q_proj_expected"],
        )
        query_grouped = prefill_checkpoint_values[1].reshape(
            config.batch_size,
            config.num_key_value_heads,
            config.query_heads_per_kv_head,
            config.query_length,
            config.head_dim,
        )
        probabilities = prefill_checkpoint_values[4]
        context_grouped = prefill_checkpoint_values[5].reshape_as(query_grouped)
        _record_tensor(arrays, "focused_qk_lhs", query_grouped)
        _record_tensor(arrays, "focused_qk_expected", prefill_checkpoint_values[2])
        _record_tensor(arrays, "focused_pv_lhs", probabilities)
        _record_tensor(arrays, "focused_pv_expected", context_grouped)
        key_payload, key_scale, key_zero = prefill_checkpoint_values[15:18]
        value_payload, value_scale, value_zero = prefill_checkpoint_values[18:21]
        if loaded.attention_compute == "fp16":
            logical_shape = [
                config.batch_size,
                config.num_key_value_heads,
                1,
                config.cache_capacity,
                config.head_dim,
            ]
            key = torch.ops.vortex.dequantize_int4(
                key_payload.unsqueeze(2),
                key_scale.unsqueeze(2),
                key_zero.unsqueeze(2),
                logical_shape,
                4,
                config.kv_group_size,
                4,
                "signed_asymmetric_int4",
            )
            value = torch.ops.vortex.dequantize_int4(
                value_payload.unsqueeze(2),
                value_scale.unsqueeze(2),
                value_zero.unsqueeze(2),
                logical_shape,
                4,
                config.kv_group_size,
                4,
                "signed_asymmetric_int4",
            )
            _record_tensor(arrays, "focused_key_dequant", key)
            _record_tensor(
                arrays,
                "focused_key_payload_matrix",
                key_payload.unsqueeze(2).reshape(-1, key_payload.shape[-1]),
            )
            _record_tensor(
                arrays,
                "focused_key_scale_matrix",
                key_scale.unsqueeze(2).reshape(-1, key_scale.shape[-1]),
            )
            _record_tensor(
                arrays,
                "focused_key_zero_matrix",
                key_zero.unsqueeze(2).reshape(-1, key_zero.shape[-1]),
            )
            _record_tensor(
                arrays,
                "focused_key_dequant_matrix",
                key.reshape(-1, key.shape[-1]),
            )
            for row_count in (8, 16, 32):
                for suffix in ("payload", "scale", "zero", "dequant"):
                    name = f"focused_key_{suffix}_matrix"
                    _record_tensor(
                        arrays,
                        f"{name}_r{row_count}",
                        arrays[name][:row_count].copy(),
                    )
            _record_tensor(arrays, "focused_qk_rhs", key.transpose(-2, -1))
            _record_tensor(arrays, "focused_pv_rhs", value)
        else:
            for prefix, values in (
                ("focused_qk_rhs", (key_payload, key_scale, key_zero)),
                ("focused_pv_rhs", (value_payload, value_scale, value_zero)),
            ):
                _record_tensor(arrays, f"{prefix}_payload", values[0].unsqueeze(2))
                _record_tensor(arrays, f"{prefix}_scale", values[1].unsqueeze(2))
                _record_tensor(arrays, f"{prefix}_zero", values[2].unsqueeze(2))
    return arrays


def tensor_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": value.nbytes,
            "sha256": sha256_array(value),
        }
        for name, value in sorted(arrays.items())
    }


def validate_reference_semantics(
    arrays: Mapping[str, np.ndarray], loaded: LoadedBackendPackage
) -> dict[str, object]:
    """Reject non-finite model state and malformed cache/top-k records."""

    shape = loaded.package["shape"]
    prompt_length = shape["prompt_length"]
    float_tensor_count = 0
    maximum_magnitude = 0.0
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.floating):
            continue
        float_tensor_count += 1
        if name.endswith("attention_masked_scores"):
            if np.any(np.isnan(value)) or np.any(np.isposinf(value)):
                raise ValueError(f"invalid masked-score sentinel in {name}")
            finite = value[np.isfinite(value)]
        else:
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite backend reference tensor: {name}")
            finite = value.reshape(-1)
        if finite.size:
            maximum_magnitude = max(
                maximum_magnitude,
                float(np.max(np.abs(finite.astype("float32")), initial=0)),
            )

    for phase_index in range(DECODE_STEPS + 1):
        expected_length = prompt_length + phase_index
        logits = arrays[f"p{phase_index}_logits"].astype("float32")[:, -1, :]
        selected = np.argmax(logits, axis=-1).astype("int64")
        np.testing.assert_array_equal(selected, arrays[f"p{phase_index}_selected"])
        np.testing.assert_array_equal(
            selected, arrays[f"p{phase_index}_topk_indices"][:, 0]
        )
        margin = (
            arrays[f"p{phase_index}_topk_values"][:, 0]
            - arrays[f"p{phase_index}_topk_values"][:, 1]
        )
        np.testing.assert_array_equal(
            margin.astype("float32"), arrays[f"p{phase_index}_top1_margin"]
        )
        for layer_index in range(NUM_LAYERS):
            prefix = f"p{phase_index}_l{layer_index}_o"
            length = np.atleast_1d(arrays[f"{prefix}7"])
            if np.any(length != expected_length):
                raise ValueError(
                    f"reference cache length mismatch at phase {phase_index}, "
                    f"layer {layer_index}: {length.tolist()}"
                )
            for state_index in range(1, 7):
                state = arrays[f"{prefix}{state_index}"]
                suffix = state[..., expected_length:, :]
                if np.count_nonzero(suffix):
                    raise ValueError(
                        f"reference cache suffix was modified at phase {phase_index}, "
                        f"layer {layer_index}, state {state_index}"
                    )
                if phase_index:
                    previous = arrays[f"p{phase_index-1}_l{layer_index}_o{state_index}"]
                    previous_length = prompt_length + phase_index - 1
                    np.testing.assert_array_equal(
                        state[..., :previous_length, :],
                        previous[..., :previous_length, :],
                    )
    return {
        "float_tensor_count": float_tensor_count,
        "maximum_finite_magnitude": maximum_magnitude,
        "cache_lengths": [prompt_length + phase for phase in range(DECODE_STEPS + 1)],
        "top1_margins": [
            arrays[f"p{phase}_top1_margin"].astype("float32").tolist()
            for phase in range(DECODE_STEPS + 1)
        ],
    }


def write_reference_artifact(
    output_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    loaded: LoadedBackendPackage,
    rows: Sequence[Sequence[int]],
    reference_device: Mapping[str, object],
    *,
    seed: int = REFERENCE_SEED,
    replay_count: int = 1,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_summary = validate_reference_semantics(arrays, loaded)
    np.savez(output_path, **arrays)
    metadata = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "format": REFERENCE_FORMAT,
        "package_path": str(loaded.path),
        "package_sha256": sha256_file(loaded.path),
        "package_identity": package_identity(loaded.package),
        "reference_seed": seed,
        "decode_steps": DECODE_STEPS,
        "prompt_token_ids": [list(row) for row in rows],
        "reference_device": dict(reference_device),
        "determinism_replay_count": replay_count,
        "tensor_count": len(arrays),
        "tensor_inventory": tensor_inventory(arrays),
        "semantic_summary": semantic_summary,
        "npz_file": output_path.name,
        "npz_sha256": sha256_file(output_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def validate_reference_artifact(
    reference_path: str | Path,
    loaded: LoadedBackendPackage,
    *,
    expected_device: str = "gpu",
    expected_seed: int = REFERENCE_SEED,
) -> tuple[dict, np.lib.npyio.NpzFile]:
    reference_path = Path(reference_path).resolve()
    metadata_path = reference_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported backend reference schema")
    if metadata.get("format") != REFERENCE_FORMAT:
        raise ValueError("invalid backend reference format")
    if metadata.get("package_sha256") != sha256_file(loaded.path):
        raise ValueError("backend reference package hash mismatch")
    if metadata.get("package_identity") != package_identity(loaded.package):
        raise ValueError("backend reference package identity mismatch")
    if metadata.get("reference_seed") != expected_seed:
        raise ValueError("backend reference seed mismatch")
    if metadata.get("reference_device", {}).get("reference_device") != expected_device:
        raise ValueError("backend reference device label mismatch")
    if metadata.get("npz_sha256") != sha256_file(reference_path):
        raise ValueError("backend reference NPZ hash mismatch")
    arrays = np.load(reference_path, allow_pickle=False)
    inventory = metadata.get("tensor_inventory", {})
    if set(arrays.files) != set(inventory):
        raise ValueError("backend reference tensor inventory mismatch")
    for name, record in inventory.items():
        value = arrays[name]
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"backend reference tensor descriptor mismatch: {name}")
        if value.nbytes != record["nbytes"] or sha256_array(value) != record["sha256"]:
            raise ValueError(f"backend reference tensor hash mismatch: {name}")
    validate_reference_semantics(arrays, loaded)
    return metadata, arrays


def compare_replays(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> None:
    if set(first) != set(second):
        raise AssertionError("GPU reference replay tensor inventory changed")
    changed = [
        name
        for name in first
        if sha256_array(first[name]) != sha256_array(second[name])
    ]
    if changed:
        raise AssertionError(
            f"GPU reference replay is nondeterministic for {len(changed)} tensors: {changed[:8]}"
        )
