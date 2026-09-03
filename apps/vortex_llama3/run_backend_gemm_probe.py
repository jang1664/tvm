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
"""Run one exact C1/C3 linear or batched-attention GEMM probe on U55C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import tvm
from tvm import relax

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    load_backend_package,
    package_identity,
    sha256_file,
    validate_reference_artifact,
)
from vortex_llama3.run_backend_validation import (
    DEFAULT_ALIAS_MAP,
    DEFAULT_VORTEX_HOME,
    LOCAL_THRESHOLDS,
    _configure_xrt_environment,
    _runtime_tensor,
    hybrid_metrics,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--focused-package", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument(
        "--probe",
        required=True,
        help="Artifact name from the focused probe package",
    )
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--allocator", choices=("naive", "pooled"), default="naive")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _reference_array(reference, name: str, array_slices, array_reshapes) -> np.ndarray:
    value = reference[name]
    extents = array_slices.get(name)
    if extents is not None:
        if len(extents) != value.ndim:
            raise ValueError(f"slice rank for {name} does not match reference rank")
        value = value[
            tuple(
                slice(None) if extent is None else slice(0, extent)
                for extent in extents
            )
        ]
    reshape = array_reshapes.get(name)
    if reshape is not None:
        value = value.reshape(reshape)
    return np.array(value, copy=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
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
    focused = json.loads(args.focused_package.read_text(encoding="utf-8"))
    if focused.get("format") != "vortex-llama3-c1-c3-focused-gemm-probes":
        raise ValueError("invalid focused GEMM package format")
    if focused.get("source_package") != package_identity(loaded.package):
        raise ValueError("focused GEMM source package identity mismatch")
    if focused.get("reference_sha256") != reference_metadata["npz_sha256"]:
        raise ValueError("focused GEMM reference hash mismatch")
    record = focused["artifacts"].get(args.probe)
    if record is None:
        raise ValueError(f"focused package has no {args.probe} artifact")
    artifact = args.focused_package.parent / record["file"]
    if artifact.stat().st_size != record["nbytes"]:
        raise ValueError("focused GEMM artifact size mismatch")
    if sha256_file(artifact) != record["sha256"]:
        raise ValueError("focused GEMM artifact hash mismatch")
    _configure_xrt_environment(loaded.package)
    print(
        json.dumps({"event": "gemm_probe_device_open", "probe": args.probe}), flush=True
    )
    device = tvm.vortex(0)
    parameter_names = [
        name for name in record["input_names"] if name.startswith("layers.")
    ]
    resident = loaded.materialized.upload(device, parameter_names)
    array_slices = record.get("array_slices", {})
    array_reshapes = record.get("array_reshapes", {})
    inputs = []
    for name in record["input_names"]:
        if name.startswith("layers."):
            inputs.append(resident[name])
        else:
            inputs.append(
                _runtime_tensor(
                    _reference_array(reference, name, array_slices, array_reshapes),
                    device,
                )
            )
    module = tvm.runtime.load_module(str(artifact))
    vm = relax.VirtualMachine(module, device=device, memory_cfg=args.allocator)
    expected_names = record.get("expected_names")
    if expected_names is None:
        expected_names = [record["expected_name"]]
    expected_values = [
        _reference_array(reference, name, array_slices, array_reshapes)
        for name in expected_names
    ]
    repetitions = []
    mismatch_arrays = {}
    for repetition in range(args.repetitions):
        print(
            json.dumps(
                {
                    "event": "gemm_probe_invoke",
                    "probe": args.probe,
                    "repetition": repetition,
                }
            ),
            flush=True,
        )
        raw_actual = vm["main"](*inputs)
        print(
            json.dumps(
                {
                    "event": "gemm_probe_returned",
                    "probe": args.probe,
                    "repetition": repetition,
                }
            ),
            flush=True,
        )
        actual_values = (
            [raw_actual.numpy()]
            if len(expected_names) == 1
            else [value.numpy() for value in raw_actual]
        )
        if len(actual_values) != len(expected_names):
            raise ValueError("focused GEMM output count mismatch")
        metrics = {
            name: hybrid_metrics(
                actual,
                expected,
                LOCAL_THRESHOLDS,
                name=f"{args.probe}:{name}",
                enforce=False,
            )
            for name, actual, expected in zip(
                expected_names, actual_values, expected_values
            )
        }
        repetitions.append(
            {
                "repetition": repetition,
                "actual_sha256": [_array_hash(value) for value in actual_values],
                "metrics": metrics,
            }
        )
        if not all(metric["pass"] for metric in metrics.values()):
            for name, actual, expected in zip(
                expected_names, actual_values, expected_values
            ):
                mismatch_arrays[f"actual_{repetition}__{name}"] = actual
                mismatch_arrays[f"expected__{name}"] = expected
    trace = {
        "format": "vortex-llama3-c1-c3-u55c-focused-gemm-trace",
        "package_identity": package_identity(loaded.package),
        "reference_npz_sha256": reference_metadata["npz_sha256"],
        "focused_package": str(args.focused_package.resolve()),
        "focused_package_sha256": sha256_file(args.focused_package),
        "probe": args.probe,
        "input_names": record["input_names"],
        "expected_names": expected_names,
        "kernel_inventory": record["kernel_inventory"],
        "expected_sha256": [_array_hash(value) for value in expected_values],
        "repetitions": repetitions,
        "runtime_environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "xrt_device_index": os.environ.get("XRT_DEVICE_INDEX"),
            "xrt_device_bdf": os.environ.get("XRT_DEVICE_BDF"),
            "xrt_xclbin_path": os.environ.get("XRT_XCLBIN_PATH"),
        },
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mismatch_arrays:
        mismatch_path = args.trace_output.with_suffix(".mismatch.npz")
        np.savez_compressed(mismatch_path, **mismatch_arrays)
    reference.close()
    print(json.dumps({"event": "gemm_probe_complete", "trace": str(args.trace_output)}))
    if mismatch_arrays:
        raise AssertionError(
            f"focused GEMM numerical mismatch; trace={args.trace_output}; "
            f"mismatch={mismatch_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
