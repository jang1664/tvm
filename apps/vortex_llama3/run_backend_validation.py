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
"""Run canonical and free-running C1/C3 validation on a physical U55C."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import tvm
from tvm import relax

from vortex_llama3.backend_numerical_validation import (
    DECODE_STEPS,
    NUM_LAYERS,
    REFERENCE_SEED,
    deterministic_prompt,
    load_backend_package,
    package_identity,
    sha256_array,
    validate_reference_artifact,
)


DEFAULT_VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
DEFAULT_ALIAS_MAP = DEFAULT_VORTEX_HOME / "ci/fpga_bin_alias_map.yaml"
ARTIFACT_NAMES = (
    "embedding_prefill",
    "embedding_decode",
    "prefill_layer",
    "decode_layer",
    "final_head_prefill",
    "final_head_decode",
)
LOCAL_THRESHOLDS = {
    "split": 0.25,
    "atol": 2e-3,
    "rtol": 2e-3,
    "max_violation_fraction": 0.02,
    "max_relative_l2": 1e-2,
    "min_cosine": 0.999,
}
FINAL_THRESHOLDS = {
    "split": 0.25,
    "atol": 2e-3,
    "rtol": 2e-3,
    "max_violation_fraction": 0.08,
    "max_relative_l2": 5e-2,
    "min_cosine": 0.995,
}


def hybrid_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    name: str,
    enforce: bool = True,
) -> dict[str, float | int]:
    """Use absolute error for small values and relative error otherwise."""

    actual = actual.astype("float32")
    expected = expected.astype("float32")
    actual_nonfinite = int(np.count_nonzero(~np.isfinite(actual)))
    expected_nonfinite = int(np.count_nonzero(~np.isfinite(expected)))
    if actual_nonfinite or expected_nonfinite:
        raise AssertionError(
            f"{name} contains NaN or infinity: actual={actual_nonfinite}, "
            f"expected={expected_nonfinite}"
        )
    absolute_error = np.abs(actual - expected)
    small = np.abs(expected) < thresholds["split"]
    relative_error = np.zeros_like(absolute_error)
    np.divide(
        absolute_error,
        np.abs(expected),
        out=relative_error,
        where=~small,
    )
    violations = (small & (absolute_error > thresholds["atol"])) | (
        ~small & (relative_error > thresholds["rtol"])
    )
    actual_flat = actual.reshape(-1).astype("float64")
    expected_flat = expected.reshape(-1).astype("float64")
    difference = actual_flat - expected_flat
    expected_norm = np.linalg.norm(expected_flat)
    actual_norm = np.linalg.norm(actual_flat)
    relative_l2 = np.linalg.norm(difference) / max(expected_norm, 1e-12)
    cosine = (
        1.0
        if actual_norm == 0 and expected_norm == 0
        else np.dot(actual_flat, expected_flat)
        / max(actual_norm * expected_norm, 1e-12)
    )
    metrics = {
        "max_small_absolute_error": float(np.max(absolute_error[small], initial=0)),
        "max_large_relative_error": float(np.max(relative_error[~small], initial=0)),
        "violation_fraction": float(np.count_nonzero(violations) / violations.size),
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "actual_max_magnitude": float(np.max(np.abs(actual), initial=0)),
        "reference_max_magnitude": float(np.max(np.abs(expected), initial=0)),
        "nonfinite_count": actual_nonfinite,
    }
    passed = (
        metrics["violation_fraction"] <= thresholds["max_violation_fraction"]
        and metrics["relative_l2"] <= thresholds["max_relative_l2"]
        and metrics["cosine"] >= thresholds["min_cosine"]
    )
    metrics["pass"] = int(passed)
    if enforce and not passed:
        raise AssertionError(f"{name} numerical limits exceeded: {metrics}")
    return metrics


def batch_hybrid_metrics(actual, reference, thresholds, *, name):
    """Apply the same fail-closed comparison independently to every batch row."""

    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim == 0:
        raise AssertionError(
            f"{name}: batch comparison shape mismatch: "
            f"actual={actual.shape}, reference={reference.shape}"
        )
    return [
        hybrid_metrics(
            actual[batch_index],
            reference[batch_index],
            thresholds,
            name=f"{name}_batch{batch_index}",
        )
        for batch_index in range(actual.shape[0])
    ]


def _unpack_signed_nibbles(payload: np.ndarray) -> np.ndarray:
    low = payload & np.uint8(15)
    high = payload >> np.uint8(4)
    values = np.stack((low, high), axis=-1).reshape(*payload.shape[:-1], -1)
    return np.where(values >= 8, values.astype("int16") - 16, values).astype("int8")


def _dequantize_cache(payload, scale, zero, valid_length):
    codes = _unpack_signed_nibbles(payload[..., :valid_length, :]).astype("float32")
    group_size = codes.shape[-1] // scale.shape[-1]
    return (
        codes
        - np.repeat(zero[..., :valid_length, :].astype("float32"), group_size, axis=-1)
    ) * np.repeat(scale[..., :valid_length, :].astype("float32"), group_size, axis=-1)


def compare_layer_state(
    actual, expected, valid_length: int, input_cache=None
) -> dict[str, object]:
    """Compare hidden plus signed-asymmetric K4/V4 cache semantics."""

    actual = [
        np.atleast_1d(value.numpy() if hasattr(value, "numpy") else value)
        for value in actual
    ]
    expected = [np.atleast_1d(value) for value in expected]
    summary = {
        "hidden": hybrid_metrics(
            actual[0], expected[0], LOCAL_THRESHOLDS, name="layer_hidden"
        )
    }
    np.testing.assert_array_equal(actual[7], expected[7])
    if input_cache is not None:
        input_cache = [
            np.atleast_1d(value.numpy() if hasattr(value, "numpy") else value)
            for value in input_cache
        ]
        previous_valid_length = valid_length - 1
        for output_index, input_index in zip(range(1, 7), range(6), strict=True):
            np.testing.assert_array_equal(
                actual[output_index][..., :previous_valid_length, :],
                input_cache[input_index][..., :previous_valid_length, :],
                err_msg=f"cache output {output_index} changed its valid prefix",
            )
            np.testing.assert_array_equal(
                actual[output_index][..., valid_length:, :],
                input_cache[input_index][..., valid_length:, :],
                err_msg=f"cache output {output_index} changed its untouched suffix",
            )
    cache_summary = {}
    for prefix, payload_index, scale_index, zero_index in (
        ("key", 1, 2, 3),
        ("value", 4, 5, 6),
    ):
        np.testing.assert_array_equal(
            actual[payload_index][..., valid_length:, :],
            expected[payload_index][..., valid_length:, :],
        )
        np.testing.assert_array_equal(
            actual[scale_index][..., valid_length:, :],
            expected[scale_index][..., valid_length:, :],
        )
        np.testing.assert_array_equal(
            actual[zero_index][..., valid_length:, :],
            expected[zero_index][..., valid_length:, :],
        )
        actual_codes = _unpack_signed_nibbles(
            actual[payload_index][..., :valid_length, :]
        )
        expected_codes = _unpack_signed_nibbles(
            expected[payload_index][..., :valid_length, :]
        )
        code_difference = np.abs(
            actual_codes.astype("int16") - expected_codes.astype("int16")
        )
        code_mismatch_rate = float(
            np.count_nonzero(code_difference) / code_difference.size
        )
        if np.max(code_difference, initial=0) > 2 or code_mismatch_rate > 0.20:
            raise AssertionError(
                f"{prefix} cache INT4 mismatch: max={np.max(code_difference)}, "
                f"rate={code_mismatch_rate}"
            )
        valid_zero_difference = np.abs(
            actual[zero_index][..., :valid_length, :].astype("int32")
            - expected[zero_index][..., :valid_length, :].astype("int32")
        )
        zero_mismatch_rate = float(
            np.count_nonzero(valid_zero_difference) / valid_zero_difference.size
        )
        if np.max(valid_zero_difference, initial=0) > 1 or zero_mismatch_rate > 0.20:
            raise AssertionError(
                f"{prefix} cache zero mismatch: "
                f"max={np.max(valid_zero_difference)}, rate={zero_mismatch_rate}"
            )
        cache_summary[prefix] = {
            "code_max_difference": int(np.max(code_difference, initial=0)),
            "code_mismatch_rate": code_mismatch_rate,
            "zero_max_difference": int(np.max(valid_zero_difference, initial=0)),
            "zero_mismatch_rate": zero_mismatch_rate,
            "scale": hybrid_metrics(
                actual[scale_index][..., :valid_length, :],
                expected[scale_index][..., :valid_length, :],
                LOCAL_THRESHOLDS,
                name=f"{prefix}_cache_scale",
            ),
            "dequantized": hybrid_metrics(
                _dequantize_cache(
                    actual[payload_index],
                    actual[scale_index],
                    actual[zero_index],
                    valid_length,
                ),
                _dequantize_cache(
                    expected[payload_index],
                    expected[scale_index],
                    expected[zero_index],
                    valid_length,
                ),
                LOCAL_THRESHOLDS,
                name=f"{prefix}_cache_dequantized",
            ),
        }
    summary["cache"] = cache_summary
    return summary


def _runtime_tensor(value: np.ndarray, device):
    # np.ascontiguousarray promotes a scalar from shape () to shape (1,),
    # which violates scalar Relax ABI annotations such as cache_length.
    return tvm.runtime.tensor(np.array(value, copy=True, order="C"), device=device)


def _save_mismatch(path: Path, metadata: dict, arrays: Mapping[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


class BackendValidationRuntime:
    """One device, one programming event, one immutable parameter upload."""

    def __init__(
        self,
        loaded,
        exec_mode: str,
        allocator: str,
        canonical_input_staging: str = "fixed",
        verify_canonical_input_readback: bool = True,
    ):
        self.loaded = loaded
        self.package = loaded.package
        self.exec_mode = exec_mode
        self.canonical_input_staging = canonical_input_staging
        self.verify_canonical_input_readback = verify_canonical_input_readback
        root = loaded.path.parent
        self.modules = {}
        for name in ARTIFACT_NAMES:
            key = f"{exec_mode}:{name}"
            if key not in self.package["artifacts"]:
                raise ValueError(f"package has no {key} artifact")
            self.modules[name] = tvm.runtime.load_module(
                str(root / self.package["artifacts"][key]["file"])
            )
        if torch.cuda.is_initialized():
            raise RuntimeError("U55C process initialized CUDA before XRT device open")
        self.device = tvm.vortex(0)
        self.device_address = tvm.get_global_func("runtime.vortex_device_address")
        self.resident = loaded.materialized.upload(self.device)
        self.vms = {}
        for name, module in self.modules.items():
            vm = relax.VirtualMachine(module, device=self.device, memory_cfg=allocator)
            self.vms[name] = vm
        orders = self.package["parameter_orders"]
        self.layer_parameters = [self.resident[name] for name in orders["layer"]]
        self.embedding_parameters = [
            self.resident[name] for name in orders["embedding"]
        ]
        self.head_parameters = [self.resident[name] for name in orders["head"]]

    def _invoke_layer(self, phase_index, hidden, positions, cache=()):
        name = "prefill_layer" if phase_index == 0 else "decode_layer"
        return self.vms[name]["main"](hidden, positions, *self.layer_parameters, *cache)

    def run_canonical(self, reference, mismatch_dir: Path) -> dict[str, object]:
        """Run every observable operation with the exact GPU-recorded input."""

        phases = []
        shape = self.package["shape"]
        for phase_index in range(DECODE_STEPS + 1):
            start = time.perf_counter()
            token_ids = _runtime_tensor(
                reference[f"p{phase_index}_token_ids"], self.device
            )
            positions = _runtime_tensor(
                reference[f"p{phase_index}_positions"], self.device
            )
            embedding_name = (
                "embedding_prefill" if phase_index == 0 else "embedding_decode"
            )
            embedding = self.vms[embedding_name]["main"](
                token_ids, *self.embedding_parameters
            )
            embedding_host = embedding.numpy()
            embedding_reference = reference[f"p{phase_index}_embedding"]
            embedding_metrics = hybrid_metrics(
                embedding_host,
                embedding_reference,
                LOCAL_THRESHOLDS,
                name=f"p{phase_index}_embedding",
            )
            embedding_batch_metrics = batch_hybrid_metrics(
                embedding_host,
                embedding_reference,
                LOCAL_THRESHOLDS,
                name=f"p{phase_index}_embedding",
            )
            layer_records = []
            fixed_hidden = None
            fixed_cache = None
            for layer_index in range(NUM_LAYERS):
                hidden_reference = reference[f"p{phase_index}_l{layer_index}_input"]
                if self.canonical_input_staging == "fixed":
                    if fixed_hidden is None:
                        fixed_hidden = tvm.runtime.empty(
                            hidden_reference.shape,
                            str(hidden_reference.dtype),
                            self.device,
                        )
                    fixed_hidden.copyfrom(hidden_reference)
                    hidden_input = fixed_hidden
                else:
                    hidden_input = _runtime_tensor(hidden_reference, self.device)
                cache = ()
                if phase_index:
                    cache_references = tuple(
                        reference[f"p{phase_index-1}_l{layer_index}_o{index}"]
                        for index in range(1, 8)
                    )
                    if self.canonical_input_staging == "fixed":
                        if fixed_cache is None:
                            fixed_cache = tuple(
                                tvm.runtime.empty(
                                    value.shape, str(value.dtype), self.device
                                )
                                for value in cache_references
                            )
                        for destination, value in zip(
                            fixed_cache, cache_references, strict=True
                        ):
                            destination.copyfrom(value)
                        cache = fixed_cache
                    else:
                        cache = tuple(
                            _runtime_tensor(value, self.device)
                            for value in cache_references
                        )
                output_addresses = None
                try:
                    input_addresses = {
                        "hidden": int(self.device_address(hidden_input)),
                        "positions": int(self.device_address(positions)),
                        "cache": [int(self.device_address(value)) for value in cache],
                    }
                    if self.verify_canonical_input_readback:
                        if not np.array_equal(hidden_input.numpy(), hidden_reference):
                            raise AssertionError(
                                "canonical hidden input readback mismatch"
                            )
                        if phase_index:
                            for index, (actual_input, expected_input) in enumerate(
                                zip(cache, cache_references, strict=True), start=1
                            ):
                                if not np.array_equal(
                                    actual_input.numpy(), expected_input
                                ):
                                    raise AssertionError(
                                        "canonical cache input readback mismatch at "
                                        f"output {index}"
                                    )
                    state = self._invoke_layer(
                        phase_index, hidden_input, positions, cache
                    )
                    output_addresses = [
                        int(self.device_address(value)) for value in state
                    ]
                    expected = tuple(
                        reference[f"p{phase_index}_l{layer_index}_o{index}"]
                        for index in range(8)
                    )
                    metrics = compare_layer_state(
                        state,
                        expected,
                        shape["prompt_length"] + phase_index,
                        cache if phase_index else None,
                    )
                except (AssertionError, ValueError) as error:
                    actual_arrays = {}
                    reread_arrays = {}
                    if "state" in locals():
                        actual_arrays.update(
                            {
                                f"actual_o{index}": value.numpy()
                                for index, value in enumerate(state)
                            }
                        )
                        for reread in range(2):
                            reread_arrays.update(
                                {
                                    f"reread{reread + 1}_o{index}": value.numpy()
                                    for index, value in enumerate(state)
                                }
                            )
                    _save_mismatch(
                        mismatch_dir / f"canonical-p{phase_index}-l{layer_index}.npz",
                        {
                            "alias": self.package["alias"],
                            "phase": phase_index,
                            "layer": layer_index,
                            "error": str(error),
                            "input_addresses": input_addresses,
                            "output_addresses": output_addresses,
                        },
                        {
                            "hidden_input": reference[
                                f"p{phase_index}_l{layer_index}_input"
                            ],
                            **(
                                {
                                    f"cache_input_o{index}": value
                                    for index, value in enumerate(
                                        cache_references, start=1
                                    )
                                }
                                if phase_index
                                else {}
                            ),
                            **{
                                f"expected_o{index}": reference[
                                    f"p{phase_index}_l{layer_index}_o{index}"
                                ]
                                for index in range(8)
                            },
                            **actual_arrays,
                            **reread_arrays,
                        },
                    )
                    raise AssertionError(
                        f"canonical first failure at phase {phase_index}, "
                        f"layer {layer_index}: {error}"
                    ) from error
                hidden_host = state[0].numpy()
                layer_records.append(
                    {
                        "layer": layer_index,
                        "hidden": metrics["hidden"],
                        "hidden_by_batch": batch_hybrid_metrics(
                            hidden_host,
                            expected[0],
                            LOCAL_THRESHOLDS,
                            name=f"p{phase_index}_l{layer_index}_hidden",
                        ),
                        "cache": metrics["cache"],
                        "output_hash": sha256_array(hidden_host),
                        "input_readback_verified": self.verify_canonical_input_readback,
                        "input_addresses": input_addresses,
                        "output_addresses": output_addresses,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "canonical_layer_complete",
                            "phase": phase_index,
                            "layer": layer_index,
                            "hidden_relative_l2": metrics["hidden"]["relative_l2"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del state, hidden_input
            head_input = _runtime_tensor(
                reference[f"p{phase_index}_head_input"], self.device
            )
            head_name = (
                "final_head_prefill" if phase_index == 0 else "final_head_decode"
            )
            logits, normalized = self.vms[head_name]["main"](
                head_input, *self.head_parameters
            )
            logits_host = logits.numpy()
            normalized_host = normalized.numpy()
            logits_metrics = hybrid_metrics(
                logits_host,
                reference[f"p{phase_index}_logits"],
                FINAL_THRESHOLDS,
                name=f"p{phase_index}_logits",
            )
            normalized_metrics = hybrid_metrics(
                normalized_host,
                reference[f"p{phase_index}_normalized"],
                FINAL_THRESHOLDS,
                name=f"p{phase_index}_normalized",
            )
            logits_batch_metrics = batch_hybrid_metrics(
                logits_host,
                reference[f"p{phase_index}_logits"],
                FINAL_THRESHOLDS,
                name=f"p{phase_index}_logits",
            )
            normalized_batch_metrics = batch_hybrid_metrics(
                normalized_host,
                reference[f"p{phase_index}_normalized"],
                FINAL_THRESHOLDS,
                name=f"p{phase_index}_normalized",
            )
            actual_top1 = np.argmax(logits_host[:, -1, :], axis=-1)
            expected_top1 = reference[f"p{phase_index}_selected"]
            if not np.array_equal(actual_top1, expected_top1):
                raise AssertionError(
                    f"canonical top-1 mismatch at phase {phase_index}: "
                    f"actual={actual_top1.tolist()}, expected={expected_top1.tolist()}"
                )
            phases.append(
                {
                    "phase_index": phase_index,
                    "embedding": embedding_metrics,
                    "embedding_by_batch": embedding_batch_metrics,
                    "layers": layer_records,
                    "normalized": normalized_metrics,
                    "normalized_by_batch": normalized_batch_metrics,
                    "logits": logits_metrics,
                    "logits_by_batch": logits_batch_metrics,
                    "top1": actual_top1.tolist(),
                    "gpu_top1_margin": reference[f"p{phase_index}_top1_margin"]
                    .astype("float32")
                    .tolist(),
                    "vm_invocation_count": NUM_LAYERS + 2,
                    "latency_seconds": time.perf_counter() - start,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "canonical_phase_complete",
                        "phase": phase_index,
                        "top1": actual_top1.tolist(),
                        "latency_seconds": phases[-1]["latency_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            gc.collect()
        return {"mode": "canonical", "phases": phases}

    def run_free(self, rows: Sequence[Sequence[int]]) -> dict[str, object]:
        """Run real autoregressive state flow without reference substitution."""

        shape = self.package["shape"]
        token_ids = np.asarray(rows, dtype="int64")
        positions = np.broadcast_to(
            np.arange(token_ids.shape[1], dtype="int64"), token_ids.shape
        ).copy()
        states = None
        generated = [[] for _ in rows]
        phases = []
        for phase_index in range(DECODE_STEPS + 1):
            start = time.perf_counter()
            token_device = _runtime_tensor(token_ids, self.device)
            position_device = _runtime_tensor(positions, self.device)
            embedding_name = (
                "embedding_prefill" if phase_index == 0 else "embedding_decode"
            )
            hidden = self.vms[embedding_name]["main"](
                token_device, *self.embedding_parameters
            )
            hidden = hidden.copyto(self.device)
            next_states = []
            layer_hashes = []
            layer_batch_hashes = []
            for layer_index in range(NUM_LAYERS):
                cache = () if phase_index == 0 else states[layer_index][1:]
                state = self._invoke_layer(phase_index, hidden, position_device, cache)
                persisted = tuple(value.copyto(self.device) for value in state)
                hidden = persisted[0]
                next_states.append(persisted)
                hidden_host = hidden.numpy()
                if not np.all(np.isfinite(hidden_host)):
                    raise AssertionError(
                        f"free-running non-finite hidden at phase {phase_index}, "
                        f"layer {layer_index}"
                    )
                layer_hashes.append(sha256_array(hidden_host))
                layer_batch_hashes.append([sha256_array(row) for row in hidden_host])
                print(
                    json.dumps(
                        {
                            "event": "free_layer_complete",
                            "phase": phase_index,
                            "layer": layer_index,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            states = next_states
            head_name = (
                "final_head_prefill" if phase_index == 0 else "final_head_decode"
            )
            logits, normalized = self.vms[head_name]["main"](
                hidden, *self.head_parameters
            )
            logits_host = logits.numpy()
            normalized_host = normalized.numpy()
            if not np.all(np.isfinite(logits_host)) or not np.all(
                np.isfinite(normalized_host)
            ):
                raise AssertionError(
                    f"free-running final head is non-finite at phase {phase_index}"
                )
            selected = np.argmax(logits_host[:, -1, :], axis=-1).astype("int64")
            lengths = [int(np.atleast_1d(state[7].numpy())[0]) for state in states]
            expected_length = shape["prompt_length"] + phase_index
            if any(length != expected_length for length in lengths):
                raise AssertionError(
                    f"free-running cache length mismatch at phase {phase_index}: {lengths}"
                )
            if phase_index < DECODE_STEPS:
                for batch_index, token in enumerate(selected):
                    generated[batch_index].append(int(token))
            phases.append(
                {
                    "phase_index": phase_index,
                    "selected_token_ids": selected.tolist(),
                    "layer_hidden_hashes": layer_hashes,
                    "layer_hidden_batch_hashes": layer_batch_hashes,
                    "logits_sha256": sha256_array(logits_host),
                    "logits_batch_sha256": [sha256_array(row) for row in logits_host],
                    "normalized_sha256": sha256_array(normalized_host),
                    "normalized_batch_sha256": [
                        sha256_array(row) for row in normalized_host
                    ],
                    "cache_lengths": lengths,
                    "maximum_logit_magnitude": float(
                        np.max(np.abs(logits_host.astype("float32")), initial=0)
                    ),
                    "vm_invocation_count": NUM_LAYERS + 2,
                    "latency_seconds": time.perf_counter() - start,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "free_phase_complete",
                        "phase": phase_index,
                        "selected_token_ids": selected.tolist(),
                        "latency_seconds": phases[-1]["latency_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if phase_index < DECODE_STEPS:
                token_ids = selected[:, None]
                positions = np.full(
                    (shape["batch_size"], 1),
                    shape["prompt_length"] + phase_index,
                    dtype="int64",
                )
            gc.collect()
        return {
            "mode": "free_running",
            "generated_token_ids": generated,
            "phases": phases,
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument(
        "--exec-mode", choices=("bytecode", "compiled"), default="bytecode"
    )
    parser.add_argument(
        "--mode", choices=("canonical", "free-running", "both"), default="both"
    )
    parser.add_argument("--free-repetitions", type=int, default=1)
    parser.add_argument("--allocator", choices=("naive", "pooled"), default="naive")
    parser.add_argument(
        "--canonical-input-staging", choices=("fixed", "fresh"), default="fixed"
    )
    parser.add_argument(
        "--no-verify-canonical-input-readback",
        action="store_false",
        dest="verify_canonical_input_readback",
    )
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--mismatch-dir", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def _configure_xrt_environment(package):
    expected_xclbin = str(Path(package["profile"]["xclbin"]).resolve())
    expected_bin = str(Path(expected_xclbin).parent)
    configured_bin = os.environ.get("FPGA_BIN_DIR")
    if configured_bin and str(Path(configured_bin).resolve()) != expected_bin:
        raise ValueError(
            f"FPGA_BIN_DIR mismatch: expected {expected_bin}, got {configured_bin}"
        )
    os.environ["FPGA_BIN_DIR"] = expected_bin
    configured_xclbin = os.environ.get("XRT_XCLBIN_PATH")
    if configured_xclbin and str(Path(configured_xclbin).resolve()) != expected_xclbin:
        raise ValueError(
            "XRT_XCLBIN_PATH mismatch: "
            f"expected {expected_xclbin}, got {configured_xclbin}"
        )
    os.environ["XRT_XCLBIN_PATH"] = expected_xclbin
    if os.environ.get("XRT_INI_PATH", "/dev/null") != "/dev/null":
        raise ValueError("U55C validation requires XRT_INI_PATH=/dev/null")
    os.environ["XRT_INI_PATH"] = "/dev/null"
    if not os.environ.get("VORTEX_SHM_PATH"):
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        os.environ["VORTEX_SHM_PATH"] = (
            f"/dev/shm/vortex_status_llama3_{job_id}_{os.getpid()}"
        )


def _detect_open_xrt_bdf(
    proc_fd_root: Path = Path("/proc/self/fd"),
    drm_class_root: Path = Path("/sys/class/drm"),
) -> str | None:
    """Resolve the U55C PCI BDF from this process's open render node."""

    bdfs = set()
    try:
        descriptors = proc_fd_root.iterdir()
    except OSError:
        return None
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        render_name = Path(target).name
        if not target.startswith("/dev/dri/renderD"):
            continue
        try:
            bdf = (drm_class_root / render_name / "device").resolve(strict=True).name
        except OSError:
            continue
        if bdf.count(":") == 2 and "." in bdf:
            bdfs.add(bdf)
    return next(iter(bdfs)) if len(bdfs) == 1 else None


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.free_repetitions <= 0:
        raise ValueError("free-running repetition count must be positive")
    if args.case != "S1" and args.exec_mode == "compiled":
        raise ValueError("compiled VM coverage is packaged only for S1")
    loaded = load_backend_package(
        args.package,
        args.alias_map,
        args.vortex_home,
        expected_alias=args.alias,
        expected_case=args.case,
    )
    metadata, reference = validate_reference_artifact(
        args.reference,
        loaded,
        expected_device="gpu",
        expected_seed=REFERENCE_SEED,
    )
    _configure_xrt_environment(loaded.package)
    runtime = BackendValidationRuntime(
        loaded,
        args.exec_mode,
        args.allocator,
        args.canonical_input_staging,
        args.verify_canonical_input_readback,
    )
    records = []
    if args.mode in ("canonical", "both"):
        records.append(runtime.run_canonical(reference, args.mismatch_dir))
    if args.mode in ("free-running", "both"):
        free_records = [
            runtime.run_free(deterministic_prompt(args.case))
            for _ in range(args.free_repetitions)
        ]
        if len(free_records) > 1:
            first_hashes = [
                phase["logits_sha256"] for phase in free_records[0]["phases"]
            ]
            for repetition, record in enumerate(free_records[1:], start=1):
                current = [phase["logits_sha256"] for phase in record["phases"]]
                if current != first_hashes:
                    raise AssertionError(
                        f"persistent free-running hashes changed at repetition {repetition}"
                    )
        records.extend(free_records)
    trace = {
        "format": "vortex-llama3-c1-c3-u55c-validation-trace",
        "package_identity": package_identity(loaded.package),
        "reference_npz": str(args.reference.resolve()),
        "reference_npz_sha256": metadata["npz_sha256"],
        "reference_device": metadata["reference_device"],
        "exec_mode": args.exec_mode,
        "allocator": args.allocator,
        "canonical_input_staging": args.canonical_input_staging,
        "canonical_input_readback_verified": args.verify_canonical_input_readback,
        "device_open_count": 1,
        "parameter_upload_count": 1,
        "cuda_initialized_in_xrt_process": torch.cuda.is_initialized(),
        "runtime_environment": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "xrt_device_index": os.environ.get("XRT_DEVICE_INDEX"),
            "xrt_device_bdf": os.environ.get("XRT_DEVICE_BDF")
            or _detect_open_xrt_bdf(),
            "fpga_bin_dir": os.environ.get("FPGA_BIN_DIR"),
            "xrt_xclbin_path": os.environ.get("XRT_XCLBIN_PATH"),
            "xrt_ini_path": os.environ.get("XRT_INI_PATH"),
            "vortex_shm_path": os.environ.get("VORTEX_SHM_PATH"),
            "vx_ready_timeout_ms": os.environ.get("VX_READY_TIMEOUT_MS", "300000"),
        },
        "thresholds": {"local": LOCAL_THRESHOLDS, "final": FINAL_THRESHOLDS},
        "artifact_kernel_inventories": {
            key: record["kernel_inventory"]
            for key, record in loaded.package["artifacts"].items()
            if key.startswith(f"{args.exec_mode}:")
        },
        "records": records,
        "total_vm_invocation_count": sum(
            phase["vm_invocation_count"]
            for record in records
            for phase in record["phases"]
        ),
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference.close()
    print(
        json.dumps(
            {
                "event": "u55c_backend_validation_complete",
                "alias": args.alias,
                "case": args.case,
                "exec_mode": args.exec_mode,
                "trace": str(args.trace_output.resolve()),
                "record_count": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
