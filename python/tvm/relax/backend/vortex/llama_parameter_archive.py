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
"""Profile-neutral Llama parameters and checked Vortex materializations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tvm.runtime import tensor as runtime_tensor

from .policy import (
    C1_ALL_FP16_TCU,
    C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
    C3_ALL_W4_NAIVE,
    validate_vortex_backend_policy,
)


_LOGICAL_SCHEMA_VERSION = 1
_MATERIALIZATION_SCHEMA_VERSION = 1
_DESCRIPTOR_VERSION = 1
_DATA_ALIGNMENT = 4096


def _align(value: int) -> int:
    return (value + _DATA_ALIGNMENT - 1) // _DATA_ALIGNMENT * _DATA_ALIGNMENT


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_tensor_records(directory: Path, tensors):
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "parameters.bin"
    records = []
    offset = 0
    with data_path.open("wb") as stream:
        for name, value, metadata in tensors:
            value = np.ascontiguousarray(value)
            aligned_offset = _align(offset)
            stream.write(bytes(aligned_offset - offset))
            stream.write(memoryview(value))
            records.append(
                {
                    "name": name,
                    "offset": aligned_offset,
                    "nbytes": value.nbytes,
                    "dtype": str(value.dtype),
                    "physical_shape": list(value.shape),
                    "sha256": _sha256_bytes(value),
                    **metadata,
                }
            )
            offset = aligned_offset + value.nbytes
    return data_path, records


def prepare_logical_parameter_archive(
    directory: str | Path,
    parameters: Mapping[str, np.ndarray],
    *,
    num_layers: int,
    included_layers: Sequence[int],
    model_metadata: Mapping[str, object],
) -> Path:
    """Write canonical W4/FP16 tensors without an accelerator fingerprint."""

    if num_layers <= 0:
        raise ValueError("logical Llama archive layer count must be positive")
    included_layers = tuple(int(value) for value in included_layers)
    if not included_layers or any(not 0 <= value < num_layers for value in included_layers):
        raise ValueError("logical Llama archive included layers are invalid")
    if len(set(included_layers)) != len(included_layers):
        raise ValueError("logical Llama archive included layers contain duplicates")
    directory = Path(directory)
    tensors = []
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        if value.dtype not in (np.dtype("uint8"), np.dtype("float16"), np.dtype("int16")):
            raise ValueError(f"unsupported logical Llama parameter dtype for {name}: {value.dtype}")
        metadata = {"layout": "canonical_row_major"}
        if name.endswith(".qweight"):
            metadata.update(
                quantization_scheme="signed_asymmetric_int4",
                group_size=int(model_metadata.get("weight_group_size", 32)),
                quant_axis=0,
                pack_axis=1,
                logical_shape=[value.shape[0], value.shape[1] * 2],
            )
        tensors.append((name, value, metadata))
    data_path, records = _write_tensor_records(directory, tensors)
    content_identity = {
        "num_layers": num_layers,
        "included_layers": included_layers,
        "model_metadata": dict(model_metadata),
        "records": [
            {
                key: record[key]
                for key in (
                    "name",
                    "dtype",
                    "physical_shape",
                    "sha256",
                    "layout",
                )
            }
            for record in records
        ],
    }
    manifest = {
        "schema_version": _LOGICAL_SCHEMA_VERSION,
        "format": "vortex-llama-logical-parameter-archive",
        "num_layers": num_layers,
        "included_layers": list(included_layers),
        "model_metadata": dict(model_metadata),
        "content_sha256": _json_hash(content_identity),
        "data_file": data_path.name,
        "data_nbytes": data_path.stat().st_size,
        "data_sha256": _sha256_file(data_path),
        "records": records,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


class _CheckedTensorArchive:
    def _initialize_records(self):
        self.data_path = self.manifest_path.parent / self.manifest["data_file"]
        if self.data_path.stat().st_size != self.manifest["data_nbytes"]:
            raise ValueError("Vortex parameter archive data file is truncated")
        if _sha256_file(self.data_path) != self.manifest["data_sha256"]:
            raise ValueError("Vortex parameter archive data hash mismatch")
        self.records = {record["name"]: record for record in self.manifest["records"]}
        if len(self.records) != len(self.manifest["records"]):
            raise ValueError("Vortex parameter archive contains duplicate record names")
        self._resident = {}

    def tensor(self, name: str) -> np.memmap:
        if name not in self.records:
            raise KeyError(name)
        record = self.records[name]
        tensor = np.memmap(
            self.data_path,
            dtype=record["dtype"],
            mode="r",
            offset=record["offset"],
            shape=tuple(record["physical_shape"]),
        )
        if _sha256_bytes(tensor) != record["sha256"]:
            raise ValueError(f"Vortex parameter archive record hash mismatch: {name}")
        return tensor

    def upload(self, device, names: Sequence[str] | None = None):
        device_key = (str(device.type), int(device.index))
        resident = self._resident.setdefault(device_key, {})
        selected = tuple(self.records) if names is None else tuple(names)
        for name in selected:
            if name not in resident:
                resident[name] = runtime_tensor(np.asarray(self.tensor(name)), device=device)
        return resident


class LogicalParameterArchive(_CheckedTensorArchive):
    """Validated profile-neutral canonical Llama parameter archive."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_num_layers: int,
        expected_model_metadata: Mapping[str, object] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != _LOGICAL_SCHEMA_VERSION:
            raise ValueError("unsupported logical Llama parameter archive schema")
        if self.manifest.get("format") != "vortex-llama-logical-parameter-archive":
            raise ValueError("invalid logical Llama parameter archive format")
        if self.manifest.get("num_layers") != expected_num_layers:
            raise ValueError("logical Llama parameter archive layer count mismatch")
        actual_metadata = self.manifest.get("model_metadata", {})
        for name, expected in (expected_model_metadata or {}).items():
            if actual_metadata.get(name) != expected:
                raise ValueError(f"logical Llama model metadata mismatch: {name}")
        self.content_sha256 = self.manifest["content_sha256"]
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        self._initialize_records()


def _dequantize_asymmetric_w4_axis0(
    packed: np.ndarray, scale: np.ndarray, zero: np.ndarray, group_size: int
) -> np.ndarray:
    if packed.ndim != 2 or scale.ndim != 2 or zero.shape != scale.shape:
        raise ValueError("canonical W4 projection tensors must be rank-2")
    logical_k, packed_n = packed.shape
    logical_n = packed_n * 2
    if scale.shape != ((logical_k + group_size - 1) // group_size, logical_n):
        raise ValueError("canonical W4 projection qparam shapes are inconsistent")
    payload = packed.astype("int16")
    low = payload & 15
    high = (payload >> 4) & 15
    values = np.empty((logical_k, logical_n), dtype="int16")
    values[:, 0::2] = low
    values[:, 1::2] = high
    values[values >= 8] -= 16
    group_index = np.arange(logical_k) // group_size
    result = (
        values.astype("float32") - zero[group_index].astype("float32")
    ) * scale[group_index].astype("float32")
    result = result.astype("float16")
    if not np.isfinite(result).all():
        raise ValueError("non-finite value in FP16 parameter materialization")
    return result


def prepare_backend_parameter_archive(
    directory: str | Path,
    logical_archive: LogicalParameterArchive,
    *,
    policy,
    target,
    profile_fingerprint: str,
) -> Path:
    """Materialize one logical archive for an exact checked backend profile."""

    policy = validate_vortex_backend_policy(target, policy)
    if not profile_fingerprint:
        raise ValueError("backend parameter materialization requires a profile fingerprint")
    if policy.name not in (
        C1_ALL_FP16_TCU,
        C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
        C3_ALL_W4_NAIVE,
    ):
        raise ValueError(f"unsupported logical archive materialization policy: {policy.name}")

    tensors = []
    consumed = set()
    group_size = int(logical_archive.manifest["model_metadata"].get("weight_group_size", 32))
    if policy.name == C1_ALL_FP16_TCU:
        for name in sorted(logical_archive.records):
            if name in consumed:
                continue
            if name.endswith(".qweight"):
                projection = name.removesuffix(".qweight")
                source_names = (
                    name,
                    f"{projection}.scales",
                    f"{projection}.zeros",
                )
                if any(value not in logical_archive.records for value in source_names):
                    raise ValueError(f"incomplete logical W4 projection: {projection}")
                weight = _dequantize_asymmetric_w4_axis0(
                    np.asarray(logical_archive.tensor(source_names[0])),
                    np.asarray(logical_archive.tensor(source_names[1])),
                    np.asarray(logical_archive.tensor(source_names[2])),
                    group_size,
                )
                tensors.append(
                    (
                        f"{projection}.weight",
                        weight,
                        {
                            "layout": "row_major_fp16",
                            "descriptor_version": _DESCRIPTOR_VERSION,
                            "source_records": list(source_names),
                            "source_hashes": [
                                logical_archive.records[value]["sha256"]
                                for value in source_names
                            ],
                            "dequantization_rounding": "fp16",
                        },
                    )
                )
                consumed.update(source_names)
            elif not name.endswith((".scales", ".zeros")):
                tensors.append(
                    (
                        name,
                        np.asarray(logical_archive.tensor(name)),
                        {
                            "layout": "canonical_fp16",
                            "descriptor_version": _DESCRIPTOR_VERSION,
                            "source_records": [name],
                            "source_hashes": [logical_archive.records[name]["sha256"]],
                        },
                    )
                )
                consumed.add(name)
    else:
        for name in sorted(logical_archive.records):
            record = logical_archive.records[name]
            tensors.append(
                (
                    name,
                    np.asarray(logical_archive.tensor(name)),
                    {
                        "layout": "canonical_row_major",
                        "descriptor_version": _DESCRIPTOR_VERSION,
                        "source_records": [name],
                        "source_hashes": [record["sha256"]],
                    },
                )
            )

    directory = Path(directory)
    data_path, records = _write_tensor_records(directory, tensors)
    manifest = {
        "schema_version": _MATERIALIZATION_SCHEMA_VERSION,
        "format": "vortex-llama-backend-parameter-archive",
        "descriptor_version": _DESCRIPTOR_VERSION,
        "backend_policy": policy.name,
        "profile_fingerprint": profile_fingerprint,
        "logical_manifest_sha256": logical_archive.manifest_sha256,
        "logical_content_sha256": logical_archive.content_sha256,
        "data_file": data_path.name,
        "data_nbytes": data_path.stat().st_size,
        "data_sha256": _sha256_file(data_path),
        "records": records,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


class BackendParameterArchive(_CheckedTensorArchive):
    """Validated profile-bound materialization of a logical Llama archive."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_policy: str,
        expected_profile_fingerprint: str,
        expected_logical_manifest_sha256: str,
        expected_logical_content_sha256: str,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported backend parameter archive schema")
        if self.manifest.get("format") != "vortex-llama-backend-parameter-archive":
            raise ValueError("invalid backend parameter archive format")
        checks = {
            "backend policy": (self.manifest.get("backend_policy"), expected_policy),
            "profile fingerprint": (
                self.manifest.get("profile_fingerprint"),
                expected_profile_fingerprint,
            ),
            "logical manifest hash": (
                self.manifest.get("logical_manifest_sha256"),
                expected_logical_manifest_sha256,
            ),
            "logical content hash": (
                self.manifest.get("logical_content_sha256"),
                expected_logical_content_sha256,
            ),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(f"backend parameter archive {label} mismatch")
        if self.manifest.get("descriptor_version") != _DESCRIPTOR_VERSION:
            raise ValueError("backend parameter archive descriptor version mismatch")
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        self._initialize_records()
