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
"""Run one isolated C1/C3 package boundary on a physical U55C."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

import tvm
from tvm import relax

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    load_backend_package,
    package_identity,
    validate_reference_artifact,
)
from vortex_llama3.run_backend_validation import (
    DEFAULT_ALIAS_MAP,
    DEFAULT_VORTEX_HOME,
    FINAL_THRESHOLDS,
    LOCAL_THRESHOLDS,
    _configure_xrt_environment,
    _runtime_tensor,
    compare_layer_state,
    hybrid_metrics,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument(
        "--probe", choices=("embedding", "layer", "head"), required=True
    )
    parser.add_argument("--phase", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--layer", type=int, choices=range(32), default=0)
    parser.add_argument(
        "--exec-mode", choices=("bytecode", "compiled"), default="bytecode"
    )
    parser.add_argument("--allocator", choices=("naive", "pooled"), default="naive")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--mini-regression" in argv or "--layer-regression" in argv:
        return _mini_regression(argv)
    args = make_parser().parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.repetitions != 1 and args.probe != "layer":
        raise ValueError(
            "repeated focused execution is supported only for layer probes"
        )
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
    if args.probe == "embedding":
        boundary = "embedding_prefill" if args.phase == 0 else "embedding_decode"
        parameter_order = loaded.package["parameter_orders"]["embedding"]
    elif args.probe == "layer":
        boundary = "prefill_layer" if args.phase == 0 else "decode_layer"
        parameter_order = loaded.package["parameter_orders"]["layer"]
    else:
        boundary = "final_head_prefill" if args.phase == 0 else "final_head_decode"
        parameter_order = loaded.package["parameter_orders"]["head"]
    key = f"{args.exec_mode}:{boundary}"
    record = loaded.package["artifacts"].get(key)
    if record is None:
        raise ValueError(f"package has no {key} artifact")
    print(json.dumps({"event": "probe_device_open", "boundary": boundary}), flush=True)
    device = tvm.vortex(0)
    resident = loaded.materialized.upload(device, parameter_order)
    parameters = [resident[name] for name in parameter_order]
    module = tvm.runtime.load_module(str(loaded.path.parent / record["file"]))
    vm = relax.VirtualMachine(module, device=device, memory_cfg=args.allocator)
    device_address = tvm.get_global_func("runtime.vortex_device_address")
    phase = args.phase
    repetition_records = []
    failure = None
    if args.probe == "embedding":
        token_ids = _runtime_tensor(reference[f"p{phase}_token_ids"], device)
        actual = vm["main"](token_ids, *parameters).numpy()
        expected = reference[f"p{phase}_embedding"]
        metrics = hybrid_metrics(
            actual, expected, LOCAL_THRESHOLDS, name=f"p{phase}_embedding"
        )
        output_hashes = [record_hash(actual)]
    elif args.probe == "layer":
        position_ids = _runtime_tensor(reference[f"p{phase}_positions"], device)
        hidden = _runtime_tensor(reference[f"p{phase}_l{args.layer}_input"], device)
        cache = ()
        if phase:
            cache = tuple(
                _runtime_tensor(reference[f"p{phase-1}_l{args.layer}_o{index}"], device)
                for index in range(1, 8)
            )
        expected_state = tuple(
            reference[f"p{phase}_l{args.layer}_o{index}"] for index in range(8)
        )
        for repetition in range(args.repetitions):
            print(
                json.dumps(
                    {
                        "event": "probe_invoke",
                        "boundary": boundary,
                        "layer": args.layer,
                        "repetition": repetition,
                    }
                ),
                flush=True,
            )
            actual_state = vm["main"](hidden, position_ids, *parameters, *cache)
            print(
                json.dumps(
                    {"event": "probe_invoke_returned", "repetition": repetition}
                ),
                flush=True,
            )
            actual_hidden = actual_state[0].numpy()
            output_hashes = [record_hash(value.numpy()) for value in actual_state]
            try:
                metrics = compare_layer_state(
                    actual_state,
                    expected_state,
                    loaded.package["shape"]["prompt_length"] + phase,
                )
            except (AssertionError, ValueError) as error:
                coordinates = np.argwhere(actual_hidden != expected_state[0])
                metrics = {"error": str(error)}
                failure = {
                    "repetition": repetition,
                    "error": str(error),
                    "hidden_mismatch_count": int(coordinates.shape[0]),
                    "hidden_mismatch_coordinates": coordinates[:32].tolist(),
                }
                break
            repetition_records.append(
                {
                    "repetition": repetition,
                    "output_hashes": output_hashes,
                    "hidden_relative_l2": metrics["hidden"]["relative_l2"],
                    "input_addresses": {
                        "hidden": int(device_address(hidden)),
                        "position_ids": int(device_address(position_ids)),
                        "cache": [int(device_address(value)) for value in cache],
                    },
                    "output_addresses": [
                        int(device_address(value)) for value in actual_state
                    ],
                }
            )
    else:
        hidden = _runtime_tensor(reference[f"p{phase}_head_input"], device)
        logits, normalized = vm["main"](hidden, *parameters)
        logits = logits.numpy()
        normalized = normalized.numpy()
        metrics = {
            "logits": hybrid_metrics(
                logits,
                reference[f"p{phase}_logits"],
                FINAL_THRESHOLDS,
                name=f"p{phase}_logits",
            ),
            "normalized": hybrid_metrics(
                normalized,
                reference[f"p{phase}_normalized"],
                FINAL_THRESHOLDS,
                name=f"p{phase}_normalized",
            ),
        }
        output_hashes = [record_hash(logits), record_hash(normalized)]
    trace = {
        "format": "vortex-llama3-c1-c3-u55c-focused-probe",
        "package_identity": package_identity(loaded.package),
        "reference_npz_sha256": metadata["npz_sha256"],
        "probe": args.probe,
        "boundary": boundary,
        "phase": args.phase,
        "layer": args.layer if args.probe == "layer" else None,
        "exec_mode": args.exec_mode,
        "kernel_inventory": record["kernel_inventory"],
        "parameter_order": list(parameter_order),
        "output_hashes": output_hashes,
        "metrics": metrics,
        "requested_repetitions": args.repetitions,
        "completed_repetitions": len(repetition_records),
        "repetitions": repetition_records,
        "failure": failure,
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
    reference.close()
    if failure is not None:
        raise AssertionError(f"focused layer repetition failed: {failure}")
    print(json.dumps({"event": "probe_complete", "trace": str(args.trace_output)}))
    return 0


def record_hash(value: np.ndarray) -> str:
    return (
        __import__("hashlib")
        .sha256(memoryview(np.ascontiguousarray(value)))
        .hexdigest()
    )


def _mini_regression(argv):
    """Check packaged embedding/prefill/decode/head boundaries against CPU references."""
    from dataclasses import replace
    import torch
    from torch.utils._pytree import tree_flatten
    from tvm.relax.backend.vortex import get_vortex_backend_policy, C4_ALL_W4_IMPROVE, C3_ALL_W4_NAIVE
    from vortex_llama3.compile_backend_matrix import _build_inputs, _import_dependencies, _model_config

    parser = argparse.ArgumentParser(description="Llama graph boundary HW functionality")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--mini-regression", action="store_true")
    scope.add_argument("--layer-regression", action="store_true",
                       help="Check one prefill/decode layer, including full Llama dimensions")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--vortex-home", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exec-mode", choices=("bytecode", "compiled"), default="bytecode")
    args = parser.parse_args(argv)
    alias_map = args.alias_map or args.vortex_home / "ci/fpga_bin_alias_map.yaml"
    loaded = load_backend_package(args.package, alias_map, args.vortex_home)
    if args.mini_regression and loaded.package["model"]["model"] != "llama3-mini":
        raise ValueError("--mini-regression requires a --model-size mini package")
    dependencies = _import_dependencies(args.vortex_home)
    shape = loaded.package["shape"]
    config = _model_config(dependencies, shape["batch_size"], shape["prompt_length"],
                           shape["cache_capacity"], loaded.package["model"])
    policy = get_vortex_backend_policy(loaded.package["backend_policy"])
    reference_policy = replace(policy, name=C3_ALL_W4_NAIVE) if policy.name == C4_ALL_W4_IMPROVE else policy
    reference_archive = loaded.logical if policy.name == C4_ALL_W4_IMPROVE else loaded.materialized
    reference_builds, _, _, _ = _build_inputs(dependencies, reference_policy, config, reference_archive)
    _, layer_order, head_order, embedding_order = _build_inputs(dependencies, policy, config, loaded.materialized)
    _configure_xrt_environment(loaded.package)
    event_prefix = "layer" if args.layer_regression else "mini"
    print(json.dumps({"event": f"{event_prefix}_device_open", "alias": loaded.package["alias"]}), flush=True)
    device = tvm.vortex(0)
    parameter_order = layer_order if args.layer_regression else [*layer_order, *head_order, *embedding_order]
    resident = loaded.materialized.upload(device, parameter_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    actual_cache, expected_cache = None, None
    boundaries = ("prefill_layer", "decode_layer") if args.layer_regression else (
        "embedding_prefill", "embedding_decode", "prefill_layer", "decode_layer",
        "final_head_prefill", "final_head_decode",
    )
    for boundary in boundaries:
        print(json.dumps({"event": f"{event_prefix}_boundary_start", "boundary": boundary}), flush=True)
        record = loaded.package["artifacts"].get(f"{args.exec_mode}:{boundary}")
        if record is None:
            raise ValueError(f"package lacks {args.exec_mode}:{boundary}")
        model, reference_inputs = reference_builds[boundary]
        reference_inputs = list(reference_inputs)
        rng = np.random.default_rng(20261006)
        if boundary.startswith("embedding"):
            reference_inputs[0] = torch.from_numpy(rng.integers(
                0, config.vocabulary_size, tuple(reference_inputs[0].shape), dtype="int64"
            ))
            order = embedding_order
        else:
            reference_inputs[0] = torch.from_numpy(rng.uniform(
                -0.1, 0.1, tuple(reference_inputs[0].shape)
            ).astype("float16"))
            order = layer_order if boundary.endswith("layer") else head_order
        if boundary == "decode_layer":
            reference_inputs[1] = torch.full_like(reference_inputs[1], config.query_length)
            reference_inputs[3:] = expected_cache
        with torch.inference_mode():
            reference_outputs = model(*reference_inputs)
        reference_outputs, _ = tree_flatten(reference_outputs)
        inputs = []
        for value in reference_inputs:
            if isinstance(value, dict):
                inputs.extend(resident[name] for name in order)
            else:
                inputs.append(tvm.runtime.tensor(value.detach().numpy(), device=device))
        if boundary == "decode_layer":
            inputs[-7:] = actual_cache
        module = tvm.runtime.load_module(str(args.package.parent / record["file"]))
        vm = relax.VirtualMachine(module, device=device, memory_cfg="naive")
        actual_outputs = vm["main"](*inputs)
        if hasattr(actual_outputs, "numpy"):
            actual_outputs = [actual_outputs]
        else:
            actual_outputs = list(actual_outputs)
        if len(actual_outputs) != len(reference_outputs):
            raise AssertionError(f"output arity mismatch for {boundary}")
        output_records, arrays = [], {}
        for index, (actual_value, expected_value) in enumerate(zip(actual_outputs, reference_outputs)):
            actual, expected = actual_value.numpy(), expected_value.detach().numpy()
            arrays[f"actual_{index}"], arrays[f"expected_{index}"] = actual, expected
            if actual.dtype.kind in "iu":
                mismatches = int(np.count_nonzero(actual != expected))
                metrics = {"mismatches": mismatches, "elements": actual.size, "pass": int(mismatches == 0)}
            else:
                metrics = hybrid_metrics(actual, expected, LOCAL_THRESHOLDS, name=f"{boundary}:{index}", enforce=False)
            output_records.append(metrics)
        if boundary == "prefill_layer":
            actual_cache = list(actual_outputs[1:])
            expected_cache = list(reference_outputs[1:])
        status = "PASS" if all(value["pass"] for value in output_records) else "FAIL"
        records.append({"boundary": boundary, "status": status, "outputs": output_records,
                        "package_sha256": record_hash(np.frombuffer(args.package.read_bytes(), dtype="uint8")),
                        "profile": loaded.package["profile"],
                        "revisions": loaded.package["revisions"],
                        "exec_mode": args.exec_mode, "shape": shape,
                        "model": loaded.package["model"],
                        "executed_layers": int(boundary.endswith("layer"))})
        np.savez(args.output_dir / f"{boundary}.npz", **arrays)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({"event": f"{event_prefix}_boundary_done", "boundary": boundary, "status": status}), flush=True)
        del vm, module
    return 1 if any(value["status"] == "FAIL" for value in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
