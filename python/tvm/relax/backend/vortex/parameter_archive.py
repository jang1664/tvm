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
"""Checked external C4 parameter archives for resident W4 model weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tvm.runtime import tensor as runtime_tensor

from .layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)


_SCHEMA_VERSION = 1
_DATA_ALIGNMENT = 4096


@dataclass(frozen=True)
class C4WeightSpec:
    """Logical W4 GEMM metadata needed to create one physical C4 record."""

    name: str
    logical_k: int
    logical_n: int
    qblock: int = 32
    weight_transpose: bool = False
    quant_direction: int = 0


def llama3_c4_weight_specs(
    num_layers: int,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    vocabulary_size: int | None = None,
) -> tuple[C4WeightSpec, ...]:
    """Return the seven Llama3 projection records for every decoder layer."""

    if num_layers <= 0:
        raise ValueError("number of decoder layers must be positive")
    kv_hidden_size = num_key_value_heads * head_dim
    dimensions = {
        "q_proj": (hidden_size, hidden_size),
        "k_proj": (hidden_size, kv_hidden_size),
        "v_proj": (hidden_size, kv_hidden_size),
        "o_proj": (hidden_size, hidden_size),
        "gate_proj": (hidden_size, intermediate_size),
        "up_proj": (hidden_size, intermediate_size),
        "down_proj": (intermediate_size, hidden_size),
    }
    specs = tuple(
        C4WeightSpec(f"layers.{layer_index}.{projection}", logical_k, logical_n)
        for layer_index in range(num_layers)
        for projection, (logical_k, logical_n) in dimensions.items()
    )
    if vocabulary_size is not None:
        if vocabulary_size <= 0 or vocabulary_size % 2:
            raise ValueError("vocabulary size must be a positive even value")
        specs += (C4WeightSpec("lm_head", hidden_size, vocabulary_size),)
    return specs


def _align(value: int, alignment: int = _DATA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_c4_parameter_archive(
    directory: str | Path,
    parameters: Mapping[str, np.ndarray],
    specs: Sequence[C4WeightSpec],
    target,
    profile_fingerprint: str,
    num_layers: int,
) -> Path:
    """Prepack canonical W4 tensors and write a checked external archive."""

    if not profile_fingerprint:
        raise ValueError("C4 parameter archive requires a profile fingerprint")
    if num_layers <= 0:
        raise ValueError("C4 parameter archive layer count must be positive")
    profile = ImproveProfile.from_target(target)
    profile.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "parameters.bin"
    manifest_path = directory / "manifest.json"
    records = []
    offset = 0

    with data_path.open("wb") as stream:
        for spec in specs:
            plan = plan_improve_layout(
                1,
                spec.logical_n,
                spec.logical_k,
                spec.qblock,
                spec.weight_transpose,
                spec.quant_direction,
                profile,
            )
            sources = {
                "qweight": (
                    parameters[f"{spec.name}.qweight"],
                    "uint8",
                    prepack_improve_weight,
                ),
                "scales": (
                    parameters[f"{spec.name}.scales"],
                    "float16",
                    lambda value, layout: prepack_improve_qparam(
                        value, layout, "float16"
                    ),
                ),
                "zeros": (
                    parameters[f"{spec.name}.zeros"],
                    "int16",
                    lambda value, layout: prepack_improve_qparam(
                        value, layout, "int16"
                    ),
                ),
            }
            for role, (source, dtype, prepack) in sources.items():
                physical = np.ascontiguousarray(prepack(source, plan), dtype=dtype)
                aligned_offset = _align(offset)
                stream.write(bytes(aligned_offset - offset))
                stream.write(memoryview(physical))
                records.append(
                    {
                        "name": f"{spec.name}.{role}",
                        "projection": spec.name,
                        "role": role,
                        "offset": aligned_offset,
                        "nbytes": physical.nbytes,
                        "dtype": dtype,
                        "physical_shape": list(physical.shape),
                        "sha256": _sha256_bytes(physical),
                        "logical_k": spec.logical_k,
                        "logical_n": spec.logical_n,
                        "execution_k": plan.execution_k,
                        "execution_n": plan.execution_n,
                        "qblock": spec.qblock,
                        "weight_transpose": spec.weight_transpose,
                        "quant_direction": spec.quant_direction,
                        "layout_abi_version": profile.layout_abi_version,
                        "gemm_abi_version": profile.gemm_abi_version,
                    }
                )
                offset = aligned_offset + physical.nbytes

        for name in sorted(parameters):
            if not (
                name.endswith("_norm.weight")
                or name == "token_embedding.weight"
            ):
                continue
            physical = np.ascontiguousarray(parameters[name], dtype="float16")
            expected_rank = 2 if name == "token_embedding.weight" else 1
            if physical.ndim != expected_rank:
                raise ValueError(
                    f"Llama3 raw FP16 parameter has wrong rank: {name}"
                )
            aligned_offset = _align(offset)
            stream.write(bytes(aligned_offset - offset))
            stream.write(memoryview(physical))
            records.append(
                {
                    "name": name,
                    "projection": None,
                    "role": (
                        "token_embedding"
                        if name == "token_embedding.weight"
                        else "norm_weight"
                    ),
                    "offset": aligned_offset,
                    "nbytes": physical.nbytes,
                    "dtype": "float16",
                    "physical_shape": list(physical.shape),
                    "sha256": _sha256_bytes(physical),
                    "layout": "canonical_fp16",
                    "layout_abi_version": profile.layout_abi_version,
                }
            )
            offset = aligned_offset + physical.nbytes

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "format": "vortex-c4-improve-parameter-archive",
        "profile_fingerprint": profile_fingerprint,
        "num_layers": num_layers,
        "data_file": data_path.name,
        "data_nbytes": data_path.stat().st_size,
        "data_sha256": _sha256_file(data_path),
        "profile": asdict(profile),
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


class C4ParameterArchive:
    """Validated memory-mapped access to a physical C4 parameter archive."""

    def __init__(
        self,
        manifest_path: str | Path,
        expected_profile_fingerprint: str,
        expected_num_layers: int,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported C4 parameter archive schema")
        if self.manifest.get("format") != "vortex-c4-improve-parameter-archive":
            raise ValueError("invalid C4 parameter archive format")
        if self.manifest.get("profile_fingerprint") != expected_profile_fingerprint:
            raise ValueError("C4 parameter archive profile fingerprint mismatch")
        if self.manifest.get("num_layers") != expected_num_layers:
            raise ValueError("C4 parameter archive layer count mismatch")
        self.data_path = self.manifest_path.parent / self.manifest["data_file"]
        actual_size = self.data_path.stat().st_size
        if actual_size != self.manifest["data_nbytes"]:
            raise ValueError("C4 parameter archive data file is truncated")
        if _sha256_file(self.data_path) != self.manifest["data_sha256"]:
            raise ValueError("C4 parameter archive data hash mismatch")
        self.records = {record["name"]: record for record in self.manifest["records"]}
        if len(self.records) != len(self.manifest["records"]):
            raise ValueError("C4 parameter archive contains duplicate record names")
        self._resident = {}

    def tensor(self, name: str) -> np.memmap:
        """Return one read-only physical tensor after validating its record hash."""

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
            raise ValueError(f"C4 parameter archive record hash mismatch: {name}")
        return tensor

    def upload(self, device) -> Mapping[str, object]:
        """Upload every physical tensor once per device and retain its handle."""

        device_key = (str(device.type), int(device.index))
        if device_key not in self._resident:
            self._resident[device_key] = {
                name: runtime_tensor(np.asarray(self.tensor(name)), device=device)
                for name in self.records
            }
        return self._resident[device_key]
