#!/usr/bin/env python3
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
"""Repeat packaged Llama3 decoder layers while recording device addresses."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import tvm
from tvm import relax
from tvm.relax.backend.vortex.parameter_archive import C4ParameterArchive


def _archive_path(root: Path, package: dict) -> Path:
    path = Path(package["archive_manifest"])
    return path if path.is_absolute() else root / path


def _emit(record: dict, output) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.write(line + "\n")
        output.flush()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument(
        "--alternate-layer",
        type=int,
        help="alternate a second fixed layer context with --layer",
    )
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode1", "decode2", "decode3"),
        default="prefill",
    )
    parser.add_argument(
        "--input-artifact",
        type=Path,
        help="replay hidden/position/KV inputs from a runner mismatch artifact",
    )
    parser.add_argument(
        "--sanity-max-abs",
        type=float,
        help="classify mismatch by non-finite or maximum magnitude instead of canonical L2",
    )
    parser.add_argument(
        "--resident-scope", choices=("selected", "full"), default="selected"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--input-mode",
        choices=("fixed", "pingpong", "sweep", "reallocate"),
        default="fixed",
    )
    parser.add_argument("--allocator", choices=("pooled", "naive"), default="pooled")
    parser.add_argument(
        "--copy-state",
        action="store_true",
        help="copy every VM state output to a new device buffer like the full runner",
    )
    parser.add_argument(
        "--copy-state-scope",
        choices=(
            "all",
            "all_except_key_scale",
            "hidden",
            "cache",
            "key",
            "key_payload",
            "key_scale",
            "key_zero",
            "value",
            "length",
        ),
        default="all",
        help="select copied outputs when --copy-state is enabled",
    )
    parser.add_argument(
        "--copy-method",
        choices=("direct", "host"),
        default="direct",
        help="copy selected outputs directly or read to host before destination allocation",
    )
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument(
        "--retry-on-mismatch",
        type=int,
        default=0,
        help="immediately repeat an identical invocation after a numerical mismatch",
    )
    parser.add_argument(
        "--capture-mismatch-dir",
        type=Path,
        help=(
            "save every failed raw state and its retry states after mismatch detection; "
            "capture happens only after the failing invocation completes"
        ),
    )
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument(
        "--call-alias-trace",
        type=Path,
        help=(
            "record VM calls whose output has the requested element count, including "
            "input/output device-address aliasing"
        ),
    )
    parser.add_argument(
        "--call-alias-elements",
        type=int,
        default=14336,
        help="tensor element count selected by --call-alias-trace",
    )
    parser.add_argument("--continue-after-nonfinite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if not 1 <= args.layer < 32:
        raise ValueError("layer must be between 1 and 31 so its canonical input is available")
    if args.alternate_layer is not None and not 1 <= args.alternate_layer < 32:
        raise ValueError("alternate-layer must be between 1 and 31")
    if args.alternate_layer is not None and args.input_mode != "fixed":
        raise ValueError("alternate-layer currently requires --input-mode fixed")
    if args.input_artifact is not None and args.alternate_layer is not None:
        raise ValueError("input-artifact cannot be combined with alternate-layer")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.sleep_ms < 0:
        raise ValueError("sleep-ms must not be negative")
    if args.retry_on_mismatch < 0:
        raise ValueError("retry-on-mismatch must not be negative")
    if args.call_alias_elements <= 0:
        raise ValueError("call-alias-elements must be positive")
    if args.sanity_max_abs is not None and args.sanity_max_abs <= 0:
        raise ValueError("sanity-max-abs must be positive")
    if args.capture_mismatch_dir is not None:
        args.capture_mismatch_dir.mkdir(parents=True, exist_ok=True)

    package_path = args.artifact_dir / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    archive = C4ParameterArchive(
        _archive_path(args.artifact_dir, package),
        package["profile_fingerprint"],
        32,
        expected_model_metadata=package["model"],
    )
    reference = np.load(args.reference_artifact, allow_pickle=False)
    phase_index = {
        "prefill": 0,
        "decode1": 1,
        "decode2": 2,
        "decode3": 3,
    }[args.phase]
    replay = (
        np.load(args.input_artifact, allow_pickle=False)
        if args.input_artifact is not None
        else None
    )
    if replay is not None:
        replay_metadata = json.loads(str(replay["metadata_json"]))
        if int(replay_metadata["layer"]) != args.layer:
            raise ValueError("input-artifact layer does not match --layer")
        if int(replay_metadata["phase_index"]) != phase_index:
            raise ValueError("input-artifact phase does not match --phase")
    reference_hiddens = {
        layer: reference[f"p{phase_index}_l{layer}_o0"].astype("float32")
        for layer in range(32)
    }
    device = tvm.vortex(0)
    device_address = tvm.get_global_func("runtime.vortex_device_address")
    layers = [args.layer]
    if args.alternate_layer is not None:
        if args.alternate_layer == args.layer:
            raise ValueError("alternate-layer must differ from layer")
        layers.append(args.alternate_layer)
    layer_names = {}
    for layer in layers:
        mapping = archive.layer_parameter_names(layer)
        layer_names[layer] = tuple(
            mapping[name] for name in package["layer_parameter_order"]
        )
    if args.resident_scope == "full":
        decoder_names = tuple(
            name for name in archive.records if name.startswith("layers.")
        )
        resident = archive.upload(device, decoder_names)
        archive.upload(
            device,
            tuple(name for name in archive.records if not name.startswith("layers.")),
        )
    else:
        selected_names = tuple(
            name for layer in layers for name in layer_names[layer]
        )
        resident = archive.upload(device, selected_names)
    positions = tvm.runtime.tensor(
        replay["positions"]
        if replay is not None
        else np.full((1, 1), phase_index, dtype="int64"),
        device=device,
    )
    contexts = []
    for layer in layers:
        input_hidden = (
            replay["hidden_input"]
            if replay is not None
            else reference[f"p{phase_index}_l{layer - 1}_o0"]
        )
        cache_inputs = []
        if phase_index:
            cache_inputs = (
                [
                    tvm.runtime.tensor(
                        replay[f"cache_input_{index}"], device=device
                    )
                    for index in range(7)
                ]
                if replay is not None
                else [
                    tvm.runtime.tensor(
                        reference[f"p{phase_index - 1}_l{layer}_o{index}"],
                        device=device,
                    )
                    for index in range(1, 8)
                ]
            )
        contexts.append(
            {
                "layer": layer,
                "input_hidden": input_hidden,
                "expected": reference[f"p{phase_index}_l{layer}_o0"].astype("float32"),
                "names": layer_names[layer],
                "parameters": [resident[name] for name in layer_names[layer]],
                "cache_inputs": cache_inputs,
            }
        )
    module_name = "prefill_layer.so" if phase_index == 0 else "decode_layer.so"
    module = tvm.runtime.load_module(str(args.artifact_dir / module_name))
    vm = relax.VirtualMachine(module, device=device, memory_cfg=args.allocator)
    alias_output = None
    alias_call_index = 0
    if args.call_alias_trace is not None:
        args.call_alias_trace.parent.mkdir(parents=True, exist_ok=True)
        alias_output = args.call_alias_trace.open("w", encoding="utf-8")

        def tensor_values(value):
            if isinstance(value, tvm.runtime.Tensor):
                return [value]
            if isinstance(value, (tuple, list)):
                tensors = []
                for item in value:
                    tensors.extend(tensor_values(item))
                return tensors
            return []

        def record_call_aliases(unused_func, name, before_run, ret_value, *call_args):
            nonlocal alias_call_index
            if before_run or name == "main":
                return relax.VMInstrumentReturnKind.NO_OP
            outputs = tensor_values(ret_value)
            selected_outputs = [
                tensor
                for tensor in outputs
                if int(np.prod(tuple(int(dim) for dim in tensor.shape)))
                == args.call_alias_elements
            ]
            inputs = [
                tensor
                for arg in call_args
                for tensor in tensor_values(arg)
            ]
            selected_inputs = [
                tensor
                for tensor in inputs
                if int(np.prod(tuple(int(dim) for dim in tensor.shape)))
                == args.call_alias_elements
            ]
            if not selected_outputs and not selected_inputs:
                return relax.VMInstrumentReturnKind.NO_OP
            input_records = [
                {
                    "address": device_address(tensor),
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                }
                for tensor in inputs
            ]
            input_addresses = {record["address"] for record in input_records}
            selected_records = [
                {
                    "address": device_address(tensor),
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                }
                for tensor in selected_inputs
            ]
            selected_address_counts = {
                address: sum(
                    record["address"] == address for record in selected_records
                )
                for address in {record["address"] for record in selected_records}
            }
            if not selected_outputs:
                _emit(
                    {
                        "event": "call_alias",
                        "call_index": alias_call_index,
                        "name": name,
                        "output": None,
                        "inputs": input_records,
                        "selected_inputs": selected_records,
                        "duplicate_selected_input_addresses": sorted(
                            address
                            for address, count in selected_address_counts.items()
                            if count > 1
                        ),
                    },
                    alias_output,
                )
                alias_call_index += 1
            for tensor in selected_outputs:
                address = device_address(tensor)
                _emit(
                    {
                        "event": "call_alias",
                        "call_index": alias_call_index,
                        "name": name,
                        "output": {
                            "address": address,
                            "shape": [int(dim) for dim in tensor.shape],
                            "dtype": str(tensor.dtype),
                        },
                        "inputs": input_records,
                        "selected_inputs": selected_records,
                        "output_aliases_input": address in input_addresses,
                    },
                    alias_output,
                )
                alias_call_index += 1
            return relax.VMInstrumentReturnKind.NO_OP

        vm.set_instrument(record_call_aliases)

    input_count = {
        "fixed": 1,
        "pingpong": 2,
        "sweep": args.iterations,
        "reallocate": 0,
    }[args.input_mode]
    for context in contexts:
        context["input_buffers"] = [
            tvm.runtime.tensor(context["input_hidden"], device=device)
            for _ in range(input_count)
        ]
    output = None
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        output = args.trace_output.open("w", encoding="utf-8")

    try:
        _emit(
            {
                "event": "configuration",
                "layer": args.layer,
                "alternate_layer": args.alternate_layer,
                "phase": args.phase,
                "resident_scope": args.resident_scope,
                "iterations": args.iterations,
                "input_mode": args.input_mode,
                "allocator": args.allocator,
                "copy_state": args.copy_state,
                "copy_state_scope": args.copy_state_scope,
                "copy_method": args.copy_method,
                "position_address": device_address(positions),
                "contexts": [
                    {
                        "layer": context["layer"],
                        "input_addresses": [
                            device_address(tensor) for tensor in context["input_buffers"]
                        ],
                        "parameter_addresses": {
                            name: device_address(resident[name]) for name in context["names"]
                        },
                        "cache_input_addresses": [
                            device_address(tensor) for tensor in context["cache_inputs"]
                        ],
                    }
                    for context in contexts
                ],
            },
            output,
        )
        seen_output_addresses = set()
        for iteration in range(args.iterations):
            context = contexts[iteration % len(contexts)]
            if args.input_mode == "reallocate":
                hidden = tvm.runtime.tensor(context["input_hidden"], device=device)
            else:
                buffers = context["input_buffers"]
                buffer_iteration = iteration // len(contexts)
                hidden = buffers[buffer_iteration % len(buffers)]

            raw_state = vm["main"](
                hidden,
                positions,
                *context["parameters"],
                *context["cache_inputs"],
            )
            raw_state_addresses = [device_address(tensor) for tensor in raw_state]
            if args.copy_state:
                copy_indices = {
                    "all": range(len(raw_state)),
                    "all_except_key_scale": (0, 1, 3, 4, 5, 6, 7),
                    "hidden": (0,),
                    "cache": range(1, len(raw_state)),
                    "key": (1, 2, 3),
                    "key_payload": (1,),
                    "key_scale": (2,),
                    "key_zero": (3,),
                    "value": (4, 5, 6),
                    "length": (7,),
                }[args.copy_state_scope]
                copy_indices = set(copy_indices)

                def copy_tensor(tensor):
                    if args.copy_method == "direct":
                        return tensor.copyto(device)
                    return tvm.runtime.tensor(tensor.numpy(), device=device)

                state = tuple(
                    copy_tensor(tensor) if index in copy_indices else tensor
                    for index, tensor in enumerate(raw_state)
                )
            else:
                state = raw_state
            state_addresses = [device_address(tensor) for tensor in state]
            seen_output_addresses.add(state_addresses[0])
            raw_actual = raw_state[0].numpy().astype("float32")
            actual = state[0].numpy().astype("float32")
            raw_nonfinite = int(np.count_nonzero(~np.isfinite(raw_actual)))
            nonfinite = int(np.count_nonzero(~np.isfinite(actual)))
            raw_relative_l2 = None
            raw_max_abs = None
            if not raw_nonfinite:
                raw_relative_l2 = float(
                    np.linalg.norm(
                        (raw_actual - context["expected"]).astype("float64")
                    )
                    / max(
                        np.linalg.norm(context["expected"].astype("float64")), 1e-12
                    )
                )
                raw_max_abs = float(np.max(np.abs(raw_actual), initial=0))
            if nonfinite:
                relative_l2 = None
                max_abs = None
            else:
                relative_l2 = float(
                    np.linalg.norm((actual - context["expected"]).astype("float64"))
                    / max(np.linalg.norm(context["expected"].astype("float64")), 1e-12)
                )
                max_abs = float(np.max(np.abs(actual), initial=0))
            record = {
                    "event": "iteration",
                    "iteration": iteration,
                    "active_layer": context["layer"],
                    "input_address": device_address(hidden),
                    "raw_state_addresses": raw_state_addresses,
                    "state_addresses": state_addresses,
                    "unique_hidden_output_addresses": len(seen_output_addresses),
                    "raw_nonfinite": raw_nonfinite,
                    "raw_max_abs": raw_max_abs,
                    "raw_relative_l2": raw_relative_l2,
                    "nonfinite": nonfinite,
                    "max_abs": max_abs,
                    "relative_l2": relative_l2,
                }
            diagnostic_mismatch = raw_nonfinite or (
                raw_max_abs is not None
                and args.sanity_max_abs is not None
                and raw_max_abs > args.sanity_max_abs
            )
            if args.sanity_max_abs is None:
                diagnostic_mismatch = diagnostic_mismatch or (
                    raw_relative_l2 is not None and raw_relative_l2 > 0.05
                )
            if diagnostic_mismatch:
                mismatch_capture = None
                if args.capture_mismatch_dir is not None:
                    mismatch_capture = {
                        "hidden_input": hidden.numpy(),
                        "positions": positions.numpy(),
                        **{
                            f"failed_output_{index}": tensor.numpy()
                            for index, tensor in enumerate(raw_state)
                        },
                        **{
                            f"canonical_output_{index}": reference[
                                f"p{phase_index}_l{context['layer']}_o{index}"
                            ]
                            for index in range(min(8, len(raw_state)))
                        },
                    }
                    mismatch_capture.update(
                        {
                            f"cache_input_{index}": tensor.numpy()
                            for index, tensor in enumerate(context["cache_inputs"])
                        }
                    )
                candidate_scores = []
                if not raw_nonfinite:
                    for layer, candidate in reference_hiddens.items():
                        candidate_scores.append(
                            (
                                float(
                                    np.linalg.norm(
                                        (raw_actual - candidate).astype("float64")
                                    )
                                    / max(
                                        np.linalg.norm(candidate.astype("float64")),
                                        1e-12,
                                    )
                                ),
                                layer,
                            )
                        )
                    candidate_scores.sort()
                peak_index = int(np.argmax(np.abs(raw_actual)))
                flat_actual = raw_actual.reshape(-1)
                record.update(
                    {
                        "raw_sha256": hashlib.sha256(
                            raw_actual.tobytes()
                        ).hexdigest(),
                        "raw_peak_index": peak_index,
                        "raw_peak_value": float(flat_actual[peak_index]),
                        "raw_prefix": flat_actual[:8].tolist(),
                        "closest_reference_hiddens": [
                            {"layer": layer, "relative_l2": score}
                            for score, layer in candidate_scores[:3]
                        ],
                    }
                )
                mismatch_retries = []
                for attempt in range(args.retry_on_mismatch):
                    retry_state = vm["main"](
                        hidden,
                        positions,
                        *context["parameters"],
                        *context["cache_inputs"],
                    )
                    retry_actual = retry_state[0].numpy().astype("float32")
                    retry_nonfinite = int(
                        np.count_nonzero(~np.isfinite(retry_actual))
                    )
                    retry_relative_l2 = None
                    retry_max_abs = None
                    if not retry_nonfinite:
                        retry_relative_l2 = float(
                            np.linalg.norm(
                                (retry_actual - context["expected"]).astype("float64")
                            )
                            / max(
                                np.linalg.norm(
                                    context["expected"].astype("float64")
                                ),
                                1e-12,
                            )
                        )
                        retry_max_abs = float(
                            np.max(np.abs(retry_actual), initial=0)
                        )
                    mismatch_retries.append(
                        {
                            "attempt": attempt + 1,
                            "state_addresses": [
                                device_address(tensor) for tensor in retry_state
                            ],
                            "nonfinite": retry_nonfinite,
                            "max_abs": retry_max_abs,
                            "relative_l2": retry_relative_l2,
                            "sha256": hashlib.sha256(
                                retry_actual.tobytes()
                            ).hexdigest(),
                        }
                    )
                    if mismatch_capture is not None:
                        mismatch_capture.update(
                            {
                                f"retry_{attempt + 1}_output_{index}": tensor.numpy()
                                for index, tensor in enumerate(retry_state)
                            }
                        )
                    del retry_state
                    gc.collect()
                    retry_matches = not retry_nonfinite and (
                        retry_max_abs is not None
                        and args.sanity_max_abs is not None
                        and retry_max_abs <= args.sanity_max_abs
                    )
                    if args.sanity_max_abs is None:
                        retry_matches = (
                            not retry_nonfinite
                            and retry_relative_l2 is not None
                            and retry_relative_l2 <= 0.05
                        )
                    if retry_matches:
                        break
                record["mismatch_retries"] = mismatch_retries
                if mismatch_capture is not None:
                    capture_path = args.capture_mismatch_dir / (
                        f"{args.phase}-layer-{context['layer']}-iteration-{iteration}.npz"
                    )
                    mismatch_capture["metadata_json"] = np.asarray(
                        json.dumps(
                            {
                                "phase": args.phase,
                                "phase_index": phase_index,
                                "layer": context["layer"],
                                "iteration": iteration,
                                "raw_state_addresses": raw_state_addresses,
                                "mismatch_retries": mismatch_retries,
                            },
                            sort_keys=True,
                        )
                    )
                    np.savez(capture_path, **mismatch_capture)
                    record["mismatch_capture"] = str(capture_path)
            _emit(record, output)
            del state
            del raw_state
            if args.input_mode == "reallocate":
                del hidden
            gc.collect()
            mismatch = nonfinite or raw_nonfinite
            if args.sanity_max_abs is not None:
                mismatch = mismatch or (
                    max_abs is not None and max_abs > args.sanity_max_abs
                ) or (
                    raw_max_abs is not None and raw_max_abs > args.sanity_max_abs
                )
            else:
                mismatch = mismatch or (
                    relative_l2 is not None and relative_l2 > 0.05
                ) or (
                    raw_relative_l2 is not None and raw_relative_l2 > 0.05
                )
            if mismatch and not args.continue_after_nonfinite:
                raise AssertionError(
                    f"layer {context['layer']} mismatched at iteration {iteration}: "
                    f"raw_relative_l2={raw_relative_l2}, copied_relative_l2={relative_l2}"
                )
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        if output is not None:
            output.close()
        if alias_output is not None:
            alias_output.close()


if __name__ == "__main__":
    main()
