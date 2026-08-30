#!/usr/bin/env python3
"""Build and repeat a Llama3 prefill layer that exposes sub-layer checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.support.vortex import load_vortex_accelerator_profile

try:
    from .run_synthetic_inference import (
        DEFAULT_VORTEX_HOME,
        _build,
        _chunk_parameter_names,
        _import_model_boundaries,
        load_package,
    )
except ImportError:
    from run_synthetic_inference import (  # type: ignore[no-redef]
        DEFAULT_VORTEX_HOME,
        _build,
        _chunk_parameter_names,
        _import_model_boundaries,
        load_package,
    )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_float = actual.astype("float32").reshape(-1).astype("float64")
    expected_float = expected.astype("float32").reshape(-1).astype("float64")
    return float(
        np.linalg.norm(actual_float - expected_float)
        / max(np.linalg.norm(expected_float), 1e-12)
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--alternate-layer", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--retry-on-mismatch", type=int, default=3)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    return parser


def _build_module(args, package: dict, archive) -> None:
    boundaries = _import_model_boundaries(args.vortex_home)
    config = boundaries["config"](
        package["shape"]["batch_size"],
        package["shape"]["prompt_length"],
        package["shape"]["cache_capacity"],
    )
    profile = load_vortex_accelerator_profile(
        args.xclbin.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    local_names = tuple(package["layer_parameter_order"])
    global_names = _chunk_parameter_names(local_names, 0)
    sample_parameters = {
        local: torch.from_numpy(np.array(archive.tensor(global_name), copy=True))
        for local, global_name in zip(local_names, global_names)
    }
    hidden = torch.zeros(
        (
            config.batch_size,
            config.query_length,
            config.hidden_size,
        ),
        dtype=torch.float16,
    )
    positions = torch.arange(config.query_length, dtype=torch.int64).repeat(
        config.batch_size, 1
    )
    executable, build_seconds = _build(
        boundaries["prefill_checkpoints"](
            config, 1, prepacked_weights=True
        ),
        (hidden, positions, sample_parameters),
        target,
        package["layout_policy"],
        package["exec_mode"],
    )
    args.module.parent.mkdir(parents=True, exist_ok=True)
    executable.export_library(str(args.module))
    print(
        json.dumps(
            {
                "checkpoint_module_built": {
                    "path": str(args.module),
                    "build_seconds": build_seconds,
                    "nbytes": args.module.stat().st_size,
                }
            }
        ),
        flush=True,
    )


def main() -> None:
    args = make_parser().parse_args()
    if not 1 <= args.layer < 32 or not 1 <= args.alternate_layer < 32:
        raise ValueError("layer indices must be between 1 and 31")
    if args.layer == args.alternate_layer:
        raise ValueError("layer and alternate-layer must differ")
    if args.iterations <= 0 or args.retry_on_mismatch < 0:
        raise ValueError("iterations must be positive and retries nonnegative")

    package, archive = load_package(args.artifact_dir / "package.json", args.xclbin)
    if package["shape"]["prompt_length"] != 1:
        raise ValueError("checkpoint reproducer currently requires S1 prefill")
    if args.build:
        _build_module(args, package, archive)
    if not args.module.exists():
        raise FileNotFoundError(args.module)

    boundaries = _import_model_boundaries(args.vortex_home)
    config = boundaries["config"](
        package["shape"]["batch_size"],
        package["shape"]["prompt_length"],
        package["shape"]["cache_capacity"],
    )
    checkpoint_names = tuple(boundaries["checkpoint_names"](config))
    reference = np.load(args.reference_artifact, allow_pickle=False)
    device = tvm.vortex(0)
    device_address = tvm.get_global_func("runtime.vortex_device_address")
    decoder_names = tuple(
        name for name in archive.records if name.startswith("layers.")
    )
    resident = archive.upload(device, decoder_names)
    archive.upload(
        device,
        tuple(name for name in archive.records if not name.startswith("layers.")),
    )
    local_names = tuple(package["layer_parameter_order"])
    contexts = []
    for layer in (args.layer, args.alternate_layer):
        global_names = _chunk_parameter_names(local_names, layer)
        contexts.append(
            {
                "layer": layer,
                "hidden": tvm.runtime.tensor(
                    reference[f"p0_l{layer - 1}_o0"], device=device
                ),
                "expected": reference[f"p0_l{layer}_o0"].astype("float32"),
                "parameters": [resident[name] for name in global_names],
            }
        )
    positions = tvm.runtime.tensor(reference["p0_positions"], device=device)
    vm = relax.VirtualMachine(
        tvm.runtime.load_module(str(args.module)), device=device, memory_cfg="pooled"
    )
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)

    with args.trace_output.open("w", encoding="utf-8") as trace:
        configuration = {
            "event": "configuration",
            "checkpoint_names": checkpoint_names,
            "layers": [args.layer, args.alternate_layer],
            "iterations": args.iterations,
            "retry_on_mismatch": args.retry_on_mismatch,
            "hidden_addresses": [device_address(c["hidden"]) for c in contexts],
        }
        trace.write(json.dumps(configuration, sort_keys=True) + "\n")
        trace.flush()
        print(json.dumps(configuration, sort_keys=True), flush=True)

        for iteration in range(args.iterations):
            context = contexts[iteration % len(contexts)]

            def invoke():
                return vm["main"](
                    context["hidden"], positions, *context["parameters"]
                )

            state = invoke()
            final = state[len(checkpoint_names) - 1].numpy().astype("float32")
            relative_l2 = _relative_l2(final, context["expected"])
            record = {
                "event": "iteration",
                "iteration": iteration,
                "layer": context["layer"],
                "final_max_abs": float(np.max(np.abs(final))),
                "final_relative_l2": relative_l2,
                "final_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
            }
            if not np.all(np.isfinite(final)) or relative_l2 > 0.05:
                arrays = {
                    "hidden_input": context["hidden"].numpy(),
                    "positions": positions.numpy(),
                    **{
                        f"failed_checkpoint_{index}": tensor.numpy()
                        for index, tensor in enumerate(state[: len(checkpoint_names)])
                    },
                    **{
                        f"failed_state_{index}": tensor.numpy()
                        for index, tensor in enumerate(state[len(checkpoint_names) :])
                    },
                }
                retries = []
                for attempt in range(1, args.retry_on_mismatch + 1):
                    retry_state = invoke()
                    retry_final = retry_state[len(checkpoint_names) - 1].numpy().astype(
                        "float32"
                    )
                    retry_relative_l2 = _relative_l2(
                        retry_final, context["expected"]
                    )
                    retries.append(
                        {
                            "attempt": attempt,
                            "final_max_abs": float(np.max(np.abs(retry_final))),
                            "final_relative_l2": retry_relative_l2,
                            "final_sha256": hashlib.sha256(
                                retry_final.tobytes()
                            ).hexdigest(),
                        }
                    )
                    arrays.update(
                        {
                            f"retry_{attempt}_checkpoint_{index}": tensor.numpy()
                            for index, tensor in enumerate(
                                retry_state[: len(checkpoint_names)]
                            )
                        }
                    )
                    del retry_state
                    gc.collect()
                    if np.all(np.isfinite(retry_final)) and retry_relative_l2 <= 0.05:
                        break
                capture_path = args.capture_dir / (
                    f"prefill-layer-{context['layer']}-iteration-{iteration}.npz"
                )
                arrays["metadata_json"] = np.asarray(
                    json.dumps(
                        {
                            "layer": context["layer"],
                            "iteration": iteration,
                            "checkpoint_names": checkpoint_names,
                            "retries": retries,
                        },
                        sort_keys=True,
                    )
                )
                np.savez(capture_path, **arrays)
                record["capture"] = str(capture_path)
                record["retries"] = retries
            line = json.dumps(record, sort_keys=True)
            trace.write(line + "\n")
            trace.flush()
            print(line, flush=True)
            del state
            gc.collect()


if __name__ == "__main__":
    main()
