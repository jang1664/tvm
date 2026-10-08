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
import sys
import time
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--matrix-regression" in argv:
        return _matrix_regression(argv)
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


def _matrix_regression(argv):
    """Compile/run small role-aware GEMMs without a full-model reference archive."""
    from vortex_llama3.compile_backend_matrix import (
        DEFAULT_VORTEX_HOME as current_vortex_home,
        _git_revision, _import_dependencies, _profile_identity, resolve_backend,
    )
    from vortex_llama3.compile_backend_focused_probes import _build
    import torch

    parser = argparse.ArgumentParser(description="Config-driven GEMM functionality matrix")
    parser.add_argument("--matrix-regression", action="store_true")
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--aliases", default="C1,C2,C3,C4")
    parser.add_argument("--vortex-home", type=Path, default=current_vortex_home)
    parser.add_argument("--alias-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--width", type=int,
        help="Synthetic power-of-two threads/MXU width; requires --compile-only",
    )
    parser.add_argument("--cases", help="Comma-separated case IDs, for a focused rerun")
    parser.add_argument("--layout-policy", choices=("alone", "fused"))
    parser.add_argument("--baseline-output-dir", type=Path,
                        help="Require bitwise agreement with an earlier alone-layout run")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    candidates = json.loads(args.candidate_map.read_text())
    aliases = [value.strip() for value in args.aliases.split(",") if value.strip()]
    if not aliases or any(value not in candidates for value in aliases):
        raise ValueError("--aliases must select entries in --candidate-map")
    if not args.compile_only and len(aliases) != 1:
        raise ValueError("HW matrix runs require one --aliases entry per process")
    if args.width is not None and not args.compile_only:
        raise ValueError("--width has no matching bitstream and requires --compile-only")
    args.alias_map = args.alias_map or args.vortex_home / "ci/fpga_bin_alias_map.yaml"
    dependencies = _import_dependencies(args.vortex_home)
    dependencies["candidate_map"] = candidates
    # Importing the export module above registers the reference/custom operations.
    from spinquant_inference.vortex_export_ops import _quantize_reference, _unpack_signed_int4

    class MatrixProbe(torch.nn.Module):
        def __init__(self, backend, role, logical_rhs_shape, qblock, qdir, transpose, chain=False):
            super().__init__()
            self.backend, self.role = backend, role
            self.logical_rhs_shape = list(logical_rhs_shape)
            self.qblock, self.qdir, self.transpose = qblock, qdir, transpose
            self.chain = chain

        def forward(self, lhs, rhs, scale, zero, rhs_next=None, scale_next=None, zero_next=None):
            if self.backend == "fp16_tcu":
                return (torch.ops.vortex.fp16_matmul(lhs, rhs, self.role),)
            result = torch.ops.vortex.mm_w4a16(
                lhs, rhs, scale, zero, self.logical_rhs_shape, self.qblock,
                self.qdir, 1, "signed_asymmetric_int4", self.transpose,
            )
            if self.chain:
                # The branch reuses packed A; the chain consumes the first GEMM's
                # physical output descriptor. Distinct weights prevent CSE.
                branch = torch.ops.vortex.mm_w4a16(
                    lhs, rhs_next, scale_next, zero_next, self.logical_rhs_shape,
                    self.qblock, self.qdir, 1, "signed_asymmetric_int4", False,
                )
                first = result
                result = torch.ops.vortex.mm_w4a16(
                    result, rhs_next, scale_next, zero_next, self.logical_rhs_shape,
                    self.qblock, self.qdir, 1, "signed_asymmetric_int4", False,
                )
                return result, branch, first
            return (result,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    selected = set(args.cases.split(",")) if args.cases else None
    for alias in aliases:
        artifacts, profile, target, policy = resolve_backend(alias, args.alias_map, dependencies)
        if args.width is not None:
            attrs = dict(target.attrs)
            for key in (
                "max_num_threads", "max_threads_per_block", "max_block_size_x",
                "max_block_size_y", "max_block_size_z",
            ):
                attrs.pop(key, None)
            attrs.update(kind="vortex", thread_warp_size=args.width)
            # This is a synthetic compile profile, never an image identity claim.
            attrs.update(
                vortex_accelerator_profile_configs="",
                vortex_accelerator_profile_fingerprint="",
            )
            if policy.linear_compute != "fp16_tcu":
                attrs.update(
                    vortex_mxu_row=args.width, vortex_mxu_col=args.width,
                    vortex_mxu_col_tile=args.width,
                    vortex_gemm_acc_mem_depth=8192,
                )
            target = tvm.target.Target(attrs, host="llvm")
        os.environ["TVM_VORTEX_HOME"] = str(args.vortex_home)
        identity = _profile_identity(alias, artifacts, args.alias_map, profile)
        revisions = {
            "tvm": _git_revision(Path(__file__).resolve().parents[2]),
            "vortex": _git_revision(args.vortex_home),
        }
        shape_cases = []
        linear_shapes = (
            [(16, 16, 32), (17, 33, 65), (256, 256, 256)]
            if policy.linear_compute == "fp16_tcu"
            else [(1, 256, 256), (4, 256, 256), (256, 256, 256)]
        )
        for index, shape in enumerate(linear_shapes):
            shape_cases.append((f"linear_{index}", "linear.q_proj", shape, max(32, args.width or 16), 0, False))
        if policy.linear_compute != policy.attention_compute:
            for role, shapes in (
                ("attention.qk", [(1, 256, 128), (4, 256, 128), (256, 256, 128)]),
                ("attention.pv", [(1, 128, 256), (4, 128, 256), (256, 128, 256)]),
            ):
                for index, shape in enumerate(shapes):
                    shape_cases.append((f"{role}_{index}", role, shape, 128, 1, False))
                shape_cases.append((f"{role}_same_shape", role, (4, 256, 256), 32, 0, False))
        if policy.linear_compute != "fp16_tcu":
            for qblock in (16, 32, 128):
                for qdir in (0, 1):
                    for transpose in (False, True):
                        shape_cases.append((
                            f"q{qblock}_d{qdir}_t{int(transpose)}", "linear.q_proj",
                            (4, 256, 256), qblock, qdir, transpose,
                        ))
            shape_cases.append(("tail", "linear.q_proj", (3, 129, 127), 32, 0, False))
        if policy.linear_compute == "w4_improve":
            shape_cases.append(("descriptor_chain", "linear.q_proj", (8, 256, 256), 32, 0, False))
            shape_cases.append(("descriptor_chain_tail", "linear.q_proj", (4, 256, 256), 32, 0, False))
            shape_cases.append(("descriptor_chain_m1", "linear.q_proj", (1, 256, 256), 32, 0, False))
            shape_cases.append(("descriptor_chain_m256", "linear.q_proj", (256, 256, 256), 32, 0, False))
        if selected is not None:
            missing = selected - {case[0] for case in shape_cases}
            if missing:
                raise ValueError(f"unknown cases for {alias}: {sorted(missing)}")
            shape_cases = [case for case in shape_cases if case[0] in selected]
        device = None
        for case_id, role, (m, n, k), qblock, qdir, transpose in shape_cases:
            print(json.dumps({"event": "matrix_case_start", "alias": alias, "case": case_id}), flush=True)
            backend = policy.linear_compute if role.startswith("linear.") else policy.attention_compute
            rng = np.random.default_rng(args.seed)
            lhs = torch.from_numpy(rng.uniform(-0.5, 0.5, (m, k)).astype("float16"))
            logical_shape = (n, k) if transpose else (k, n)
            rhs = torch.from_numpy(rng.uniform(-0.5, 0.5, logical_shape).astype("float16"))
            if backend == "fp16_tcu":
                inputs = (lhs, rhs, torch.ones(1, dtype=torch.float16), torch.zeros(1, dtype=torch.int16))
            else:
                payload, scale, zero = _quantize_reference(
                    rhs, qdir, qblock, 1, "signed_asymmetric_int4"
                )
                inputs = (lhs, payload, scale, zero)
            chain = case_id.startswith("descriptor_chain")
            if chain:
                rhs_next = torch.from_numpy(rng.uniform(-0.5, 0.5, logical_shape).astype("float16"))
                inputs += _quantize_reference(rhs_next, qdir, qblock, 1, "signed_asymmetric_int4")
            model = MatrixProbe(backend, role, logical_shape, qblock, qdir, transpose, chain)
            expected_outputs = [value.detach().numpy() for value in model(*inputs)]
            expected = np.stack(expected_outputs) if chain else expected_outputs[0]
            started = time.perf_counter()
            executable, inventory = _build(model, inputs, target, "auto", args.layout_policy)
            if chain and args.layout_policy != "alone":
                if int(inventory["module_attrs"].get("vortex.improve.reused_a_layouts", 0)) < 1:
                    raise ValueError("descriptor_chain failed to exercise shared A layout reuse")
                reused_c = int(inventory["module_attrs"].get("vortex.improve.reused_c_layouts", 0))
                packed_c = int(target.attrs["vortex_layout_abi_version"]) == 3
                if bool(reused_c) != (packed_c or m % 8 == 0):
                    raise ValueError("descriptor_chain has incorrect C-to-A padding compatibility")
            required = "tcu" if backend == "fp16_tcu" else (
                "improve" if backend == "w4_improve" else "naive"
            )
            for kind in ("tcu", "improve", "naive"):
                count = inventory[f"{kind}_helper_definitions"]
                if bool(count) != (kind == required):
                    raise ValueError(f"role routing mismatch: {alias}/{case_id}: {inventory}")
            directory = args.output_dir / alias
            directory.mkdir(exist_ok=True)
            artifact = directory / f"{case_id}.so"
            executable.export_library(str(artifact))
            record = {
                "alias": alias, "case": case_id, "role": role, "backend": backend,
                "shape_mnk": [m, n, k], "qblock": qblock, "qdir": qdir,
                "weight_transpose": transpose, "seed": args.seed,
                "profile": identity, "kernel_inventory": inventory,
                "revisions": revisions,
                "artifact": str(artifact), "artifact_sha256": sha256_file(artifact),
                "compile_seconds": time.perf_counter() - started,
                "synthetic_width": args.width,
                "layout_policy": args.layout_policy,
                "compiled_target": str(target),
                "status": "COMPILE_PASS",
            }
            if not args.compile_only:
                if device is None:
                    _configure_xrt_environment({"profile": identity})
                    device = tvm.vortex(0)
                module = tvm.runtime.load_module(str(artifact))
                vm = relax.VirtualMachine(module, device=device, memory_cfg="naive")
                output = vm["main"](*[
                    tvm.runtime.tensor(value.detach().numpy(), device=device) for value in inputs
                ])
                actual = np.stack([value.numpy() for value in output]) if chain else output.numpy()
                stage_expected = expected.copy()
                torch_stage_expected = expected.copy()
                if chain:
                    # Apply the unchanged single-GEMM envelope to each operation.
                    # The second GEMM consumes rounded hardware output, so check
                    # it against that actual predecessor and retain the independent
                    # end-to-end reference/error separately.
                    torch_stage_expected[0] = torch.ops.vortex.mm_w4a16(
                        torch.from_numpy(actual[2]), *inputs[4:], list(logical_shape),
                        qblock, qdir, 1, "signed_asymmetric_int4", False,
                    ).detach().numpy()
                    # Match the independent standalone QCOL reference in
                    # fpint_gemm_ffn_hw/test_vectors.h: multiply centered INT4
                    # and scale in FP32, without first rounding W to FP16.
                    weights = []
                    for payload, scale, zero in (inputs[1:4], inputs[4:]):
                        integers = _unpack_signed_int4(payload, list(logical_shape), 1).float()
                        weights.append(
                            (integers - zero.float().repeat_interleave(qblock, 0))
                            * scale.float().repeat_interleave(qblock, 0)
                        )
                    stage_expected = np.stack([
                        (torch.from_numpy(actual[2]).float() @ weights[1]).half().numpy(),
                        (lhs.float() @ weights[1]).half().numpy(),
                        (lhs.float() @ weights[0]).half().numpy(),
                    ])
                # Predeclared FP16 rounding envelope; never widened after a failure.
                difference = np.abs(actual.astype("float32") - stage_expected.astype("float32"))
                allowed = 0.003 + 0.003 * np.abs(stage_expected.astype("float32"))
                mismatches = (~np.isfinite(actual)) | (difference > allowed)
                record.update(
                    status="PASS" if not mismatches.any() else "FAIL",
                    elements=actual.size, mismatches=int(mismatches.sum()),
                    max_abs_error=float(difference.max(initial=0)),
                    max_relative_error=float((difference / np.maximum(np.abs(stage_expected), 1e-6)).max(initial=0)),
                    nonfinite=int((~np.isfinite(actual)).sum()),
                    first_mismatch=np.argwhere(mismatches)[0].tolist() if mismatches.any() else None,
                )
                if chain:
                    end_difference = np.abs(actual.astype("float32") - expected.astype("float32"))
                    record.update(
                        reference_kind="standalone_qcol_fp32_per_operation",
                        torch_fp16_weight_mismatches=int((
                            np.abs(actual.astype("float32") - torch_stage_expected.astype("float32"))
                            > 0.003 + 0.003 * np.abs(torch_stage_expected)
                        ).sum()),
                        end_to_end_mismatches=int((end_difference > 0.003 + 0.003 * np.abs(expected)).sum()),
                        max_end_to_end_abs_error=float(end_difference.max(initial=0)),
                    )
                if args.baseline_output_dir is not None:
                    baseline = np.load(args.baseline_output_dir / alias / f"{case_id}.npz")["actual"]
                    equal = actual.shape == baseline.shape and actual.tobytes() == baseline.tobytes()
                    record["baseline_bitwise_equal"] = equal
                    if not equal:
                        record["status"] = "FAIL"
                np.savez(directory / f"{case_id}.npz", actual=actual, expected=expected,
                         stage_expected=stage_expected, torch_stage_expected=torch_stage_expected)
                del vm, module
            records.append(record)
            (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({"event": "matrix_case_done", "alias": alias, "case": case_id, "status": record["status"]}), flush=True)
        if device is not None:
            del device
    return 1 if any(record["status"] == "FAIL" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
