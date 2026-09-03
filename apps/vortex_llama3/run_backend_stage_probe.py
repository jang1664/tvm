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
"""Run one exact C1/C3 layer-stage probe on a physical U55C."""

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
    _detect_open_xrt_bdf,
    _runtime_tensor,
    compare_layer_state,
    hybrid_metrics,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stage-package", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument(
        "--probe",
        required=True,
        help="Artifact name from the focused stage package",
    )
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-prefill-repetitions", type=int, default=0)
    parser.add_argument("--warmup-prefill-chain-repetitions", type=int, default=0)
    parser.add_argument("--full-resident-archive", action="store_true")
    parser.add_argument("--allocator", choices=("naive", "pooled"), default="naive")
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _compare_array(name: str, actual: np.ndarray, expected: np.ndarray):
    if np.issubdtype(expected.dtype, np.integer):
        exact = np.array_equal(actual, expected)
        return {
            "pass": int(exact),
            "exact": int(exact),
            "mismatch_count": int(np.count_nonzero(actual != expected)),
        }
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    if not np.array_equal(actual_finite, expected_finite):
        return {"pass": 0, "nonfinite_mask_match": 0}
    if not np.all(expected_finite):
        if np.any(
            np.signbit(actual[~actual_finite]) != np.signbit(expected[~expected_finite])
        ):
            return {"pass": 0, "nonfinite_mask_match": 1, "nonfinite_sign_match": 0}
        actual = actual[actual_finite]
        expected = expected[expected_finite]
    return hybrid_metrics(actual, expected, LOCAL_THRESHOLDS, name=name, enforce=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.warmup_prefill_repetitions < 0:
        raise ValueError("warmup prefill repetitions cannot be negative")
    if args.warmup_prefill_chain_repetitions < 0:
        raise ValueError("warmup prefill chain repetitions cannot be negative")
    loaded = load_backend_package(
        args.package,
        args.alias_map,
        args.vortex_home,
        expected_alias=args.alias,
        expected_case=args.case,
    )
    reference_metadata, reference = validate_reference_artifact(
        args.reference,
        loaded,
        expected_device="gpu",
        expected_seed=REFERENCE_SEED,
    )
    focused = json.loads(args.stage_package.read_text(encoding="utf-8"))
    if focused.get("format") != "vortex-llama3-c1-c3-focused-stage-probes":
        raise ValueError("invalid focused stage package format")
    if focused.get("source_package") != package_identity(loaded.package):
        raise ValueError("focused stage source package identity mismatch")
    if focused.get("reference_sha256") != reference_metadata["npz_sha256"]:
        raise ValueError("focused stage reference hash mismatch")
    record = focused["artifacts"].get(args.probe)
    if record is None:
        raise ValueError(f"focused stage package has no {args.probe} artifact")
    artifact = args.stage_package.parent / record["file"]
    if (
        artifact.stat().st_size != record["nbytes"]
        or sha256_file(artifact) != record["sha256"]
    ):
        raise ValueError("focused stage artifact identity mismatch")
    _configure_xrt_environment(loaded.package)
    print(
        json.dumps({"event": "stage_probe_device_open", "probe": args.probe}),
        flush=True,
    )
    device = tvm.vortex(0)
    parameter_names = [
        name for name in record["input_names"] if name.startswith("layers.")
    ]
    resident = loaded.materialized.upload(
        device, None if args.full_resident_archive else parameter_names
    )
    vm = relax.VirtualMachine(
        tvm.runtime.load_module(str(artifact)), device=device, memory_cfg=args.allocator
    )
    if args.warmup_prefill_repetitions:
        prefill_record = loaded.package["artifacts"]["bytecode:prefill_layer"]
        prefill_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(loaded.path.parent / prefill_record["file"])),
            device=device,
            memory_cfg=args.allocator,
        )
        prefill_hidden = _runtime_tensor(reference["p0_l0_input"], device)
        prefill_positions = _runtime_tensor(reference["p0_positions"], device)
        prefill_expected = tuple(reference[f"p0_l0_o{index}"] for index in range(8))
        for repetition in range(args.warmup_prefill_repetitions):
            prefill_state = prefill_vm["main"](
                prefill_hidden,
                prefill_positions,
                *(resident[name] for name in parameter_names),
            )
            compare_layer_state(
                prefill_state,
                prefill_expected,
                loaded.package["shape"]["prompt_length"],
            )
            print(
                json.dumps(
                    {
                        "event": "stage_probe_prefill_warmup_complete",
                        "repetition": repetition,
                    }
                ),
                flush=True,
            )
    if args.warmup_prefill_chain_repetitions:
        prefill_record = loaded.package["artifacts"]["bytecode:prefill_layer"]
        prefill_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(loaded.path.parent / prefill_record["file"])),
            device=device,
            memory_cfg=args.allocator,
        )
        prefill_positions = _runtime_tensor(reference["p0_positions"], device)
        first_hidden = reference["p0_l0_input"]
        prefill_hidden = tvm.runtime.empty(
            first_hidden.shape, str(first_hidden.dtype), device
        )
        for chain_repetition in range(args.warmup_prefill_chain_repetitions):
            for layer_index in range(32):
                hidden_reference = reference[f"p0_l{layer_index}_input"]
                prefill_hidden.copyfrom(hidden_reference)
                if not np.array_equal(prefill_hidden.numpy(), hidden_reference):
                    raise AssertionError(
                        "prefill chain canonical hidden input readback mismatch at "
                        f"layer {layer_index}"
                    )
                prefill_state = prefill_vm["main"](
                    prefill_hidden,
                    prefill_positions,
                    *(resident[name] for name in parameter_names),
                )
                prefill_expected = tuple(
                    reference[f"p0_l{layer_index}_o{index}"] for index in range(8)
                )
                compare_layer_state(
                    prefill_state,
                    prefill_expected,
                    loaded.package["shape"]["prompt_length"],
                )
                print(
                    json.dumps(
                        {
                            "event": "stage_probe_prefill_chain_layer_complete",
                            "chain_repetition": chain_repetition,
                            "layer": layer_index,
                        }
                    ),
                    flush=True,
                )
    inputs = [
        (
            resident[name]
            if name.startswith("layers.")
            else _runtime_tensor(
                (
                    np.asarray(reference[name]).reshape(())
                    if name.endswith("_o7")
                    else (
                        np.asarray(reference[name])[0]
                        if name.startswith("p0_l0_o")
                        else reference[name]
                    )
                ),
                device,
            )
        )
        for name in record["input_names"]
    ]
    expected_values = [reference[name] for name in record["expected_names"]]
    device_address = tvm.get_global_func("runtime.vortex_device_address")
    repetition_records = []
    failed_names = []
    failure_repetition = None
    for repetition in range(args.repetitions):
        print(
            json.dumps(
                {
                    "event": "stage_probe_invoke",
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
                    "event": "stage_probe_returned",
                    "probe": args.probe,
                    "repetition": repetition,
                }
            ),
            flush=True,
        )
        actual_tensors = (
            [raw_actual] if len(record["expected_names"]) == 1 else list(raw_actual)
        )
        actual_values = [value.numpy() for value in actual_tensors]
        if len(actual_values) != len(expected_values):
            raise ValueError("focused stage output count mismatch")
        metrics = {
            name: _compare_array(name, actual, expected)
            for name, actual, expected in zip(
                record["expected_names"], actual_values, expected_values
            )
        }
        failed_names = [name for name, result in metrics.items() if not result["pass"]]
        repetition_records.append(
            {
                "repetition": repetition,
                "actual_sha256": [_array_hash(value) for value in actual_values],
                "output_addresses": [
                    int(device_address(value)) for value in actual_tensors
                ],
                "failed_names": failed_names,
            }
        )
        if failed_names:
            failure_repetition = repetition
            break
    trace = {
        "format": "vortex-llama3-c1-c3-u55c-focused-stage-trace",
        "package_identity": package_identity(loaded.package),
        "reference_npz_sha256": reference_metadata["npz_sha256"],
        "stage_package": str(args.stage_package.resolve()),
        "stage_package_sha256": sha256_file(args.stage_package),
        "probe": args.probe,
        "input_names": record["input_names"],
        "expected_names": record["expected_names"],
        "kernel_inventory": record["kernel_inventory"],
        "actual_sha256": [_array_hash(value) for value in actual_values],
        "expected_sha256": [_array_hash(value) for value in expected_values],
        "metrics": metrics,
        "requested_repetitions": args.repetitions,
        "warmup_prefill_repetitions": args.warmup_prefill_repetitions,
        "warmup_prefill_chain_repetitions": args.warmup_prefill_chain_repetitions,
        "full_resident_archive": args.full_resident_archive,
        "completed_repetitions": len(repetition_records),
        "failure_repetition": failure_repetition,
        "repetitions": repetition_records,
        "runtime_environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "xrt_device_index": os.environ.get("XRT_DEVICE_INDEX"),
            "xrt_device_bdf": os.environ.get("XRT_DEVICE_BDF")
            or _detect_open_xrt_bdf(),
            "xrt_xclbin_path": os.environ.get("XRT_XCLBIN_PATH"),
        },
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failed_names:
        mismatch_path = args.trace_output.with_suffix(".mismatch.npz")
        mismatch_arrays = {}
        for name, actual, expected in zip(
            record["expected_names"], actual_values, expected_values
        ):
            mismatch_arrays[f"actual__{name}"] = actual
            mismatch_arrays[f"expected__{name}"] = expected
        for reread in range(2):
            for name, tensor in zip(
                record["expected_names"], actual_tensors, strict=True
            ):
                mismatch_arrays[f"reread{reread + 1}__{name}"] = tensor.numpy()
        np.savez_compressed(mismatch_path, **mismatch_arrays)
        reference.close()
        raise AssertionError(
            "focused stage numerical mismatch for "
            f"{failed_names}; trace={args.trace_output}; mismatch={mismatch_path}"
        )
    reference.close()
    print(
        json.dumps({"event": "stage_probe_complete", "trace": str(args.trace_output)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
