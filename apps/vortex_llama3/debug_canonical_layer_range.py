#!/usr/bin/env python3
"""Replay canonical Llama3 phases/layer ranges with durable Vortex event tracing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import tvm
from tvm import relax

try:
    from .run_synthetic_inference import (
        DEFAULT_VORTEX_HOME,
        NUM_LAYERS,
        _chunk_parameter_names,
        _compare_layer_state,
        _git_revision,
        _hybrid_stats,
        _import_model_boundaries,
        _sha256_file,
        load_package,
    )
except ImportError:
    from run_synthetic_inference import (  # type: ignore[no-redef]
        DEFAULT_VORTEX_HOME,
        NUM_LAYERS,
        _chunk_parameter_names,
        _compare_layer_state,
        _git_revision,
        _hybrid_stats,
        _import_model_boundaries,
        _sha256_file,
        load_package,
    )


_BDF_PATTERN = re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")


def parse_phases(value: str, decode_steps: int) -> tuple[int, ...]:
    """Parse ordered phase names into reference-artifact phase indices."""

    result = []
    for raw_name in value.split(","):
        name = raw_name.strip()
        if name == "prefill":
            phase_index = 0
        elif name.startswith("decode_") and name[7:].isdigit():
            phase_index = int(name[7:])
        else:
            raise ValueError(f"invalid diagnostic phase: {name!r}")
        if not 0 <= phase_index <= decode_steps:
            raise ValueError(
                f"diagnostic phase {name!r} is outside decode-steps={decode_steps}"
            )
        if phase_index in result:
            raise ValueError(f"duplicate diagnostic phase: {name!r}")
        result.append(phase_index)
    if not result:
        raise ValueError("at least one diagnostic phase is required")
    return tuple(result)


def parse_layer_range(value: str) -> tuple[int, int]:
    """Parse an inclusive START:END layer range."""

    fields = value.split(":")
    if len(fields) != 2:
        raise ValueError("layer range must use inclusive START:END syntax")
    try:
        start, end = (int(field) for field in fields)
    except ValueError as error:
        raise ValueError("layer range endpoints must be integers") from error
    if not 0 <= start <= end < NUM_LAYERS:
        raise ValueError(f"layer range must satisfy 0 <= START <= END < {NUM_LAYERS}")
    return start, end


def _phase_name(phase_index: int) -> str:
    return "prefill" if phase_index == 0 else f"decode_{phase_index}"


def required_reference_keys(
    phases: Sequence[int], layer_range: tuple[int, int]
) -> tuple[str, ...]:
    """Return every canonical array required before any device is opened."""

    start, end = layer_range
    keys = []
    for phase_index in phases:
        keys.extend((f"p{phase_index}_token_ids", f"p{phase_index}_positions"))
        for layer in range(start, end + 1):
            if layer > 0:
                keys.append(f"p{phase_index}_l{layer - 1}_o0")
            if phase_index > 0:
                keys.extend(
                    f"p{phase_index - 1}_l{layer}_o{output_index}"
                    for output_index in range(1, 8)
                )
            keys.extend(
                f"p{phase_index}_l{layer}_o{output_index}" for output_index in range(8)
            )
    return tuple(dict.fromkeys(keys))


def validate_reference_keys(
    reference: Mapping[str, object], phases: Sequence[int], layer_range: tuple[int, int]
) -> None:
    missing = [
        key
        for key in required_reference_keys(phases, layer_range)
        if key not in reference
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"reference artifact is missing {len(missing)} arrays: {preview}{suffix}"
        )


class EventTrace:
    """A line-buffered JSONL trace that flushes every device-action boundary."""

    def __init__(self, path: Path, common: Mapping[str, object]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8", buffering=1)
        self._common = dict(common)

    def write(self, event: str, **fields: object) -> None:
        record = {
            **self._common,
            "event": event,
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.close()


def query_xrt_bdf(device_index: int) -> dict[str, object]:
    """Resolve the allocated XRT device identity without changing its state."""

    command = ["/opt/xilinx/xrt/bin/xrt-smi", "examine"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result: dict[str, object] = {
        "device_index": device_index,
        "xbutil_exit_code": completed.returncode,
    }
    if completed.returncode == 0:
        table_bdfs = re.findall(
            r"^\|\[([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\]",
            completed.stdout,
            flags=re.MULTILINE,
        )
        bdfs = tuple(match.lower() for match in table_bdfs)
        if not bdfs:
            bdfs = tuple(
                sorted(
                    set(
                        match.lower()
                        for match in _BDF_PATTERN.findall(
                            completed.stdout + completed.stderr
                        )
                    )
                )
            )
        result["available_bdfs"] = list(bdfs)
        result["bdf"] = bdfs[device_index] if device_index < len(bdfs) else None
        result["xrt_smi_sha256"] = hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest()
        if result["bdf"] is None:
            result["xrt_smi_error"] = (
                f"device index {device_index} is outside {len(bdfs)} discovered BDFs"
            )
    else:
        causal_line = next(
            (line for line in completed.stderr.splitlines() if line.strip()), ""
        )
        result["xrt_smi_error"] = causal_line
    if not result.get("bdf"):
        legacy = subprocess.run(
            [
                "/opt/xilinx/xrt/bin/xbutil",
                "examine",
                "-d",
                str(device_index),
                "--format",
                "JSON",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        legacy_bdfs = tuple(
            sorted(
                set(
                    match.lower()
                    for match in _BDF_PATTERN.findall(legacy.stdout + legacy.stderr)
                )
            )
        )
        result["legacy_available_bdfs"] = list(legacy_bdfs)
        if device_index < len(legacy_bdfs):
            result["bdf"] = legacy_bdfs[device_index]
    return result


def _tensor_nbytes(tensor) -> int:
    return (
        int(np.prod(tensor.shape, dtype="int64")) * np.dtype(str(tensor.dtype)).itemsize
    )


def _tensor_record(tensor, device_address) -> dict[str, object]:
    return {
        "address": int(device_address(tensor)),
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "nbytes": _tensor_nbytes(tensor),
    }


def _array_summary(array: np.ndarray) -> dict[str, object]:
    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        return {
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "nonfinite_count": int(np.count_nonzero(~finite)),
            "max_abs": float(np.max(np.abs(array.astype("float32")), initial=0)),
        }
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "min": int(np.min(array, initial=0)),
        "max": int(np.max(array, initial=0)),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--phases", default="prefill,decode_1,decode_2,decode_3")
    parser.add_argument("--layer-range", default="0:31")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--copy-mode", choices=("none", "hidden", "full"), default="full"
    )
    parser.add_argument("--allocator", choices=("pooled", "naive"), default="pooled")
    parser.add_argument("--vm-scope", choices=("shared", "per-call"), default="shared")
    parser.add_argument(
        "--health-probe",
        choices=("none", "start-end", "phase"),
        default="start-end",
    )
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--expected-bdf")
    return parser


def run(args) -> dict[str, object]:
    phase_sequence = parse_phases(args.phases, args.decode_steps)
    phases = phase_sequence * args.repetitions
    layer_range = parse_layer_range(args.layer_range)
    package_path = args.artifact_dir / "package.json"
    package, archive = load_package(package_path, args.xclbin)
    if int(package["compiled_layers"]) != 1:
        raise ValueError("canonical layer-range replay requires one compiled layer")
    reference = np.load(args.reference_artifact, allow_pickle=False)
    validate_reference_keys(reference, phases, layer_range)

    xrt_index = int(os.environ.get("XRT_DEVICE_INDEX", "0"))
    xrt_identity = query_xrt_bdf(xrt_index)
    actual_bdf = xrt_identity.get("bdf")
    if args.expected_bdf and actual_bdf != args.expected_bdf.lower():
        raise ValueError(
            f"allocated XRT BDF mismatch: expected {args.expected_bdf}, got {actual_bdf}"
        )
    run_uuid = str(uuid.uuid4())
    tvm_home = Path(__file__).resolve().parents[2]
    common = {
        "run_uuid": run_uuid,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "xrt": xrt_identity,
    }
    configuration = {
        "package_path": str(package_path.resolve()),
        "package_sha256": _sha256_file(package_path),
        "package_revisions": package["revisions"],
        "current_revisions": {
            "tvm": _git_revision(tvm_home),
            "vortex": _git_revision(args.vortex_home),
        },
        "xclbin": str(args.xclbin.resolve()),
        "xclbin_sha256": _sha256_file(args.xclbin),
        "profile_fingerprint": package["profile_fingerprint"],
        "archive_manifest_sha256": package["archive_manifest_sha256"],
        "reference_artifact": str(args.reference_artifact.resolve()),
        "reference_sha256": _sha256_file(args.reference_artifact),
        "phases": [_phase_name(index) for index in phase_sequence],
        "repetitions": args.repetitions,
        "layer_range": list(layer_range),
        "copy_mode": args.copy_mode,
        "allocator": args.allocator,
        "vm_scope": args.vm_scope,
        "health_probe": args.health_probe,
        "state_transport": "canonical-host-to-device-per-invocation",
        "attempts": 1,
    }

    root = package_path.parent
    modules = {
        name: tvm.runtime.load_module(str(root / package["artifacts"][name]["file"]))
        for name in ("embedding_decode", "prefill_layer", "decode_layer")
    }
    device = tvm.vortex(0)
    device_address = tvm.get_global_func("runtime.vortex_device_address")
    local_names = tuple(package["layer_parameter_order"])
    decoder_names = tuple(
        name for name in archive.records if name.startswith("layers.")
    )
    resident = archive.upload(device, decoder_names + ("token_embedding.weight",))
    layer_inputs = {
        layer: [resident[name] for name in _chunk_parameter_names(local_names, layer)]
        for layer in range(layer_range[0], layer_range[1] + 1)
    }
    embedding_weight = np.asarray(archive.tensor("token_embedding.weight"))
    launch_names: list[str] = []

    def instrument(unused_func, name, before_run, unused_ret_value, *unused_args):
        if not before_run and name != "main":
            launch_names.append(name)
        return relax.VMInstrumentReturnKind.NO_OP

    def make_vm(module_name: str):
        vm = relax.VirtualMachine(
            modules[module_name], device=device, memory_cfg=args.allocator
        )
        vm.set_instrument(instrument)
        return vm

    probe_vm = make_vm("embedding_decode")

    with EventTrace(args.trace_output, common) as trace:
        trace.write("configuration", **configuration)

        def health_probe(boundary: str, phase: str | None = None) -> bool:
            token_ids = np.asarray(reference[f"p{phases[0]}_token_ids"][:, :1])
            expected = embedding_weight[token_ids]
            launch_start = len(launch_names)
            trace.write(
                "health_probe_before_launch",
                boundary=boundary,
                phase=phase,
                cumulative_kernel_launch_count=launch_start,
            )
            try:
                output = probe_vm["main"](
                    tvm.runtime.tensor(token_ids, device=device),
                    resident["token_embedding.weight"],
                )
                trace.write(
                    "health_probe_after_launch",
                    boundary=boundary,
                    phase=phase,
                    kernel_launch_count=len(launch_names) - launch_start,
                    output=_tensor_record(output, device_address),
                )
                trace.write("health_probe_before_d2h", boundary=boundary, phase=phase)
                actual = output.numpy()
                exact = bool(np.array_equal(actual, expected))
                trace.write(
                    "health_probe_after_d2h",
                    boundary=boundary,
                    phase=phase,
                    healthy=exact and bool(np.all(np.isfinite(actual))),
                    output_summary=_array_summary(actual),
                    exact_match=exact,
                )
                return exact and bool(np.all(np.isfinite(actual)))
            except Exception as error:  # pylint: disable=broad-exception-caught
                trace.write(
                    "health_probe_error",
                    boundary=boundary,
                    phase=phase,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                return False

        if args.health_probe != "none" and not health_probe("run_start"):
            raise RuntimeError("Vortex health probe failed before canonical replay")

        completed_layers = 0
        first_failure = None
        try:
            for phase_ordinal, phase_index in enumerate(phases):
                repetition = phase_ordinal // len(phase_sequence)
                phase = _phase_name(phase_index)
                if args.health_probe == "phase" and not health_probe(
                    "phase_start", phase
                ):
                    raise RuntimeError(f"Vortex health probe failed before {phase}")
                module_name = "prefill_layer" if phase_index == 0 else "decode_layer"
                shared_vm = make_vm(module_name) if args.vm_scope == "shared" else None
                positions = tvm.runtime.tensor(
                    np.asarray(reference[f"p{phase_index}_positions"]), device=device
                )
                for layer in range(layer_range[0], layer_range[1] + 1):
                    vm = shared_vm or make_vm(module_name)
                    hidden_host = (
                        embedding_weight[
                            np.asarray(reference[f"p{phase_index}_token_ids"])
                        ]
                        if layer == 0
                        else np.asarray(reference[f"p{phase_index}_l{layer - 1}_o0"])
                    )
                    hidden = tvm.runtime.tensor(hidden_host, device=device)
                    cache_inputs = (
                        ()
                        if phase_index == 0
                        else tuple(
                            tvm.runtime.tensor(
                                np.asarray(
                                    reference[
                                        f"p{phase_index - 1}_l{layer}_o{output_index}"
                                    ]
                                ),
                                device=device,
                            )
                            for output_index in range(1, 8)
                        )
                    )
                    parameters = layer_inputs[layer]
                    launch_start = len(launch_names)
                    invocation = {
                        "repetition": repetition,
                        "phase": phase,
                        "phase_index": phase_index,
                        "layer": layer,
                        "attempt": 0,
                        "vm_identity": id(vm),
                        "logical_operation_group": module_name,
                        "cumulative_kernel_launch_count": launch_start,
                        "inputs": {
                            "hidden": _tensor_record(hidden, device_address),
                            "position": _tensor_record(positions, device_address),
                            "parameters": [
                                _tensor_record(value, device_address)
                                for value in parameters
                            ],
                            "cache": [
                                _tensor_record(value, device_address)
                                for value in cache_inputs
                            ],
                        },
                    }
                    trace.write("layer_before_launch", **invocation)
                    state = vm["main"](hidden, positions, *parameters, *cache_inputs)
                    state = tuple(state[:8]) if len(state) > 8 else tuple(state)
                    trace.write(
                        "layer_after_launch",
                        **{
                            key: value
                            for key, value in invocation.items()
                            if key != "inputs"
                        },
                        kernel_launch_count=len(launch_names) - launch_start,
                        kernel_names=launch_names[launch_start:],
                        outputs=[
                            _tensor_record(value, device_address) for value in state
                        ],
                    )

                    expected = tuple(
                        np.asarray(reference[f"p{phase_index}_l{layer}_o{index}"])
                        for index in range(8)
                    )
                    comparison = None
                    if args.copy_mode != "none":
                        copy_count = 1 if args.copy_mode == "hidden" else 8
                        trace.write(
                            "layer_before_d2h",
                            repetition=repetition,
                            phase=phase,
                            phase_index=phase_index,
                            layer=layer,
                            output_count=copy_count,
                        )
                        actual = tuple(value.numpy() for value in state[:copy_count])
                        trace.write(
                            "layer_after_d2h",
                            repetition=repetition,
                            phase=phase,
                            phase_index=phase_index,
                            layer=layer,
                            output_summaries=[
                                _array_summary(value) for value in actual
                            ],
                        )
                        if args.copy_mode == "hidden":
                            comparison = {
                                "hidden": _hybrid_stats(
                                    actual[0],
                                    expected[0],
                                    split=1.0,
                                    atol=0.25,
                                    rtol=0.15,
                                    max_exceed_fraction=0.08,
                                    max_relative_l2=0.05,
                                    min_cosine=0.995,
                                    name="hidden",
                                )
                            }
                        else:
                            comparison = _compare_layer_state(
                                actual,
                                expected,
                                package["shape"]["prompt_length"] + phase_index,
                                compare_hidden=True,
                            )
                    trace.write(
                        "layer_complete",
                        repetition=repetition,
                        phase=phase,
                        phase_index=phase_index,
                        layer=layer,
                        comparison=comparison,
                    )
                    completed_layers += 1
                    if shared_vm is None:
                        del vm
                    del state, hidden, cache_inputs
                    gc.collect()
                if args.health_probe == "phase" and not health_probe(
                    "phase_end", phase
                ):
                    raise RuntimeError(f"Vortex health probe failed after {phase}")
                if shared_vm is not None:
                    del shared_vm
                del positions
                gc.collect()
        except Exception as error:  # pylint: disable=broad-exception-caught
            first_failure = {
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_layers": completed_layers,
            }
            trace.write("first_failure", **first_failure)
            if args.health_probe != "none":
                health_probe("after_failure")
            raise
        finally:
            if first_failure is None and args.health_probe == "start-end":
                if not health_probe("run_end"):
                    raise RuntimeError(
                        "Vortex health probe failed after canonical replay"
                    )
        trace.write(
            "run_complete",
            completed_layers=completed_layers,
            repetitions=args.repetitions,
            status="pass",
        )

    return {
        "run_uuid": run_uuid,
        "status": "pass",
        "completed_layers": completed_layers,
        "repetitions": args.repetitions,
        "trace_output": str(args.trace_output),
    }


def main() -> None:
    args = make_parser().parse_args()
    if args.decode_steps < 0:
        raise ValueError("decode steps must not be negative")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
