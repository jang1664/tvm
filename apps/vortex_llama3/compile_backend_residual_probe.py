#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information regarding copyright
# ownership.  The ASF licenses this file to You under the Apache License, Version 2.0.
"""Compile the exact batch-2 FP16 residual-add shape used by C1 decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import tvm

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    load_backend_package,
    package_identity,
    sha256_array,
    sha256_file,
    validate_reference_artifact,
)
from vortex_llama3.compile_backend_matrix import _inventory
from vortex_llama3.compile_backend_stage_probes import _build
from vortex_llama3.run_backend_validation import DEFAULT_ALIAS_MAP, DEFAULT_VORTEX_HOME


class ResidualAddProbe(torch.nn.Module):
    """Match the exported Llama residual's FP32 add and FP16 round boundary."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left.float() + right.float()).to(torch.float16)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    loaded = load_backend_package(
        args.package,
        args.alias_map,
        args.vortex_home,
        expected_alias="C1",
        expected_case="S3",
    )
    reference_metadata, reference = validate_reference_artifact(
        args.reference,
        loaded,
        expected_device="gpu",
        expected_seed=REFERENCE_SEED,
    )
    left = np.array(reference["p1_l0_input"], copy=True)
    right = np.zeros_like(left)
    target = tvm.target.Target(loaded.package["profile"]["target"], host="llvm")
    executable, inventory = _build(
        ResidualAddProbe(),
        (torch.from_numpy(left), torch.from_numpy(right)),
        target,
        loaded.package["backend_policy"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "residual_add.so"
    executable.export_library(str(artifact))
    tvm.runtime.load_module(str(artifact))
    inputs = args.output_dir / "inputs.npz"
    np.savez_compressed(inputs, left=left, right=right)
    metadata = {
        "schema_version": 1,
        "format": "vortex-llama3-c1-residual-stress-package",
        "source_package": package_identity(loaded.package),
        "source_package_path": str(loaded.path.resolve()),
        "source_package_sha256": sha256_file(loaded.path),
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": reference_metadata["npz_sha256"],
        "artifact": {
            "file": artifact.name,
            "sha256": sha256_file(artifact),
            "nbytes": artifact.stat().st_size,
            "kernel_inventory": inventory,
        },
        "inputs": {
            "file": inputs.name,
            "sha256": sha256_file(inputs),
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "left_sha256": sha256_array(left),
            "right_sha256": sha256_array(right),
        },
    }
    package_path = args.output_dir / "package.json"
    package_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference.close()
    print(package_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
