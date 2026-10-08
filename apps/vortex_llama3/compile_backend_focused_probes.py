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
"""Compile exact C1/C3 linear, QK, and PV validation probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    load_backend_package,
    package_identity,
    validate_reference_artifact,
)
from vortex_llama3.compile_backend_matrix import _inventory
from vortex_llama3.run_backend_validation import DEFAULT_ALIAS_MAP, DEFAULT_VORTEX_HOME


PROJECTION_DIMS = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
    "gate_proj": (4096, 14336),
    "up_proj": (4096, 14336),
    "down_proj": (14336, 4096),
}


class LinearProbe(torch.nn.Module):
    def __init__(self, linear_compute: str, projection_name: str) -> None:
        super().__init__()
        self.linear_compute = linear_compute
        self.projection_name = projection_name

    def forward(
        self, lhs: torch.Tensor, parameters: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        parameter_prefix = f"layers.0.{self.projection_name}"
        if self.linear_compute == "fp16":
            return torch.ops.vortex.fp16_matmul(
                lhs,
                parameters[f"{parameter_prefix}.weight"],
                f"linear.{self.projection_name}",
            )
        input_size, output_size = PROJECTION_DIMS[self.projection_name]
        return torch.ops.vortex.mm_w4a16(
            lhs,
            parameters[f"{parameter_prefix}.qweight"],
            parameters[f"{parameter_prefix}.scales"],
            parameters[f"{parameter_prefix}.zeros"],
            [input_size, output_size],
            32,
            0,
            1,
            "signed_asymmetric_int4",
            False,
        )


class QKVProbe(torch.nn.Module):
    """Exercise the three projection jobs in their full-layer order."""

    def __init__(self, linear_compute: str) -> None:
        super().__init__()
        self.q = LinearProbe(linear_compute, "q_proj")
        self.k = LinearProbe(linear_compute, "k_proj")
        self.v = LinearProbe(linear_compute, "v_proj")

    def forward(
        self, lhs: torch.Tensor, parameters: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(lhs, parameters), self.k(lhs, parameters), self.v(lhs, parameters)


class FP16AttentionProbe(torch.nn.Module):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.ops.vortex.fp16_matmul(lhs, rhs, self.role)


class W4AttentionProbe(torch.nn.Module):
    def __init__(self, transpose_rhs: bool, logical_rhs_shape: Sequence[int]) -> None:
        super().__init__()
        self.transpose_rhs = transpose_rhs
        self.logical_rhs_shape = list(logical_rhs_shape)

    def forward(
        self,
        lhs: torch.Tensor,
        payload: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vortex.mm_w4a16(
            lhs,
            payload,
            scale,
            zero,
            self.logical_rhs_shape,
            128,
            len(self.logical_rhs_shape) - 1,
            len(self.logical_rhs_shape) - 1,
            "signed_asymmetric_int4",
            self.transpose_rhs,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_archive_parameters(archive, names):
    return {
        name: torch.from_numpy(np.array(archive.tensor(name), copy=True))
        for name in names
    }


def _build(model, inputs, target, policy, layout_policy=None):
    exported = torch.export.export(model, inputs, strict=True)
    mod = from_exported_program(
        exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True
    )
    lowered = relax.backend.vortex.get_default_pipeline(
        target, backend_policy=policy, layout_policy=layout_policy
    )(
        mod
    )
    executable = relax.build(
        lowered,
        target,
        relax_pipeline=tvm.transform.Sequential([]),
        exec_mode="bytecode",
    )
    return executable, _inventory(lowered)


def _slice_array(value: np.ndarray, extents: Sequence[int | None]) -> np.ndarray:
    """Take a leading-origin diagnostic slice while preserving every rank."""

    if len(extents) != value.ndim:
        raise ValueError(
            f"slice rank {len(extents)} does not match array rank {value.ndim}"
        )
    return np.array(
        value[
            tuple(
                slice(None) if extent is None else slice(0, extent)
                for extent in extents
            )
        ],
        copy=True,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probes",
        nargs="+",
        help="Compile only the named probes (default: compile every probe)",
    )
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    loaded = load_backend_package(
        args.package,
        args.alias_map,
        args.vortex_home,
        expected_alias=args.alias,
        expected_case="S1",
    )
    reference_metadata, reference = validate_reference_artifact(
        args.reference,
        loaded,
        expected_device="gpu",
        expected_seed=REFERENCE_SEED,
    )
    target = tvm.target.Target(loaded.package["profile"]["target"], host="llvm")
    builds = {}
    for projection_name in PROJECTION_DIMS:
        parameter_prefix = f"layers.0.{projection_name}"
        if loaded.linear_compute == "fp16":
            linear_names = (f"{parameter_prefix}.weight",)
        else:
            linear_names = (
                f"{parameter_prefix}.qweight",
                f"{parameter_prefix}.scales",
                f"{parameter_prefix}.zeros",
            )
        probe_name = f"linear_{projection_name.removesuffix('_proj')}"
        input_name = f"focused_linear_{projection_name}_input"
        expected_name = f"focused_linear_{projection_name}_expected"
        builds[probe_name] = (
            LinearProbe(loaded.linear_compute, projection_name),
            (
                torch.from_numpy(np.array(reference[input_name], copy=True)),
                _torch_archive_parameters(loaded.materialized, linear_names),
            ),
            [input_name, *linear_names],
            [expected_name],
        )
    qkv_names = []
    for projection_name in ("q_proj", "k_proj", "v_proj"):
        parameter_prefix = f"layers.0.{projection_name}"
        suffixes = (
            ("weight",)
            if loaded.linear_compute == "fp16"
            else (
                "qweight",
                "scales",
                "zeros",
            )
        )
        qkv_names.extend(f"{parameter_prefix}.{suffix}" for suffix in suffixes)
    builds["linear_qkv"] = (
        QKVProbe(loaded.linear_compute),
        (
            torch.from_numpy(
                np.array(reference["focused_linear_q_proj_input"], copy=True)
            ),
            _torch_archive_parameters(loaded.materialized, qkv_names),
        ),
        ["focused_linear_q_proj_input", *qkv_names],
        [
            "focused_linear_q_proj_expected",
            "focused_linear_k_proj_expected",
            "focused_linear_v_proj_expected",
        ],
    )
    for name, role, transpose in (
        ("attention_qk", "attention.qk", True),
        ("attention_pv", "attention.pv", False),
    ):
        prefix = "focused_qk" if name.endswith("qk") else "focused_pv"
        if loaded.attention_compute == "fp16":
            model = FP16AttentionProbe(role)
            input_names = [f"{prefix}_lhs", f"{prefix}_rhs"]
        else:
            model = W4AttentionProbe(transpose, [1, 8, 1, 8, 128])
            input_names = [
                f"{prefix}_lhs",
                f"{prefix}_rhs_payload",
                f"{prefix}_rhs_scale",
                f"{prefix}_rhs_zero",
            ]
        builds[name] = (
            model,
            tuple(
                torch.from_numpy(np.array(reference[input_name], copy=True))
                for input_name in input_names
            ),
            input_names,
            [f"{prefix}_expected"],
            {},
            {},
        )

    if loaded.attention_compute == "w4":
        qk_input_names = [
            "focused_qk_lhs",
            "focused_qk_rhs_payload",
            "focused_qk_rhs_scale",
            "focused_qk_rhs_zero",
        ]
        for heads, groups in ((1, 1), (1, 2), (1, 4), (2, 4), (4, 4)):
            probe_name = f"attention_qk_h{heads}_g{groups}"
            slice_extents = {
                "focused_qk_lhs": [1, heads, groups, 1, 128],
                "focused_qk_rhs_payload": [1, heads, 1, 8, 64],
                "focused_qk_rhs_scale": [1, heads, 1, 8, 1],
                "focused_qk_rhs_zero": [1, heads, 1, 8, 1],
                "focused_qk_expected": [1, heads, groups, 1, 8],
            }
            builds[probe_name] = (
                W4AttentionProbe(True, [1, heads, 1, 8, 128]),
                tuple(
                    torch.from_numpy(_slice_array(reference[name], slice_extents[name]))
                    for name in qk_input_names
                ),
                qk_input_names,
                ["focused_qk_expected"],
                slice_extents,
                {},
            )
        matrix_slices = {
            "focused_qk_lhs": [1, 1, 1, 1, 128],
            "focused_qk_rhs_payload": [1, 1, 1, 8, 64],
            "focused_qk_rhs_scale": [1, 1, 1, 8, 1],
            "focused_qk_rhs_zero": [1, 1, 1, 8, 1],
            "focused_qk_expected": [1, 1, 1, 1, 8],
        }
        matrix_reshapes = {
            "focused_qk_lhs": [1, 128],
            "focused_qk_rhs_payload": [8, 64],
            "focused_qk_rhs_scale": [8, 1],
            "focused_qk_rhs_zero": [8, 1],
            "focused_qk_expected": [1, 8],
        }
        builds["attention_qk_matrix"] = (
            W4AttentionProbe(True, [8, 128]),
            tuple(
                torch.from_numpy(
                    _slice_array(reference[name], matrix_slices[name]).reshape(
                        matrix_reshapes[name]
                    )
                )
                for name in qk_input_names
            ),
            qk_input_names,
            ["focused_qk_expected"],
            matrix_slices,
            matrix_reshapes,
        )

    if args.probes is not None:
        unknown = sorted(set(args.probes) - set(builds))
        if unknown:
            raise ValueError(f"unknown focused probes: {', '.join(unknown)}")
        builds = {name: builds[name] for name in args.probes}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, build_record in builds.items():
        model, inputs, input_names, expected_names, *optional = build_record
        array_slices = optional[0] if optional else {}
        array_reshapes = optional[1] if len(optional) > 1 else {}
        executable, inventory = _build(
            model, inputs, target, loaded.package["backend_policy"]
        )
        path = args.output_dir / f"{name}.so"
        executable.export_library(str(path))
        tvm.runtime.load_module(str(path))
        artifacts[name] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "nbytes": path.stat().st_size,
            "input_names": input_names,
            "expected_names": expected_names,
            "array_slices": array_slices,
            "array_reshapes": array_reshapes,
            "kernel_inventory": inventory,
        }
    package = {
        "schema_version": 1,
        "format": "vortex-llama3-c1-c3-focused-gemm-probes",
        "source_package": package_identity(loaded.package),
        "source_package_path": str(loaded.path),
        "source_package_sha256": _sha256_file(loaded.path),
        "reference_path": str(Path(args.reference).resolve()),
        "reference_sha256": reference_metadata["npz_sha256"],
        "alias": args.alias,
        "backend_policy": loaded.package["backend_policy"],
        "artifacts": artifacts,
    }
    package_path = args.output_dir / "package.json"
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference.close()
    print(package_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
