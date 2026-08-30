#!/usr/bin/env python3
"""Compare a captured Vortex layer mismatch against eager execution on the same inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_synthetic_inference import (
    DEFAULT_VORTEX_HOME,
    DEFAULT_XCLBIN,
    _deterministic_parameters,
    _import_model_boundaries,
    load_package,
)


def _stats(actual: np.ndarray, expected: np.ndarray) -> dict:
    if np.issubdtype(actual.dtype, np.floating):
        actual_float = actual.astype("float32")
        expected_float = expected.astype("float32")
        difference = actual_float - expected_float
        actual_flat = actual_float.reshape(-1).astype("float64")
        expected_flat = expected_float.reshape(-1).astype("float64")
        denominator = max(np.linalg.norm(expected_flat), 1e-12)
        cosine_denominator = max(
            np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat), 1e-12
        )
        return {
            "max_absolute_error": float(np.max(np.abs(difference), initial=0)),
            "relative_l2": float(
                np.linalg.norm(difference.reshape(-1).astype("float64"))
                / denominator
            ),
            "cosine": float(
                np.dot(actual_flat, expected_flat) / cosine_denominator
            ),
        }
    difference = actual.astype("int64") - expected.astype("int64")
    return {
        "mismatch_fraction": float(np.count_nonzero(difference) / difference.size),
        "max_integer_difference": int(np.max(np.abs(difference), initial=0)),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mismatch-artifact", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, default=DEFAULT_XCLBIN)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    package, archive = load_package(args.artifact_dir / "package.json", args.xclbin)
    capture = np.load(args.mismatch_artifact, allow_pickle=False)
    metadata = json.loads(str(capture["metadata_json"]))
    layer = int(metadata["layer"])
    phase_index = int(metadata["phase_index"])
    if int(metadata["compiled_layers"]) != 1:
        raise ValueError("mismatch analysis currently requires one compiled layer")

    boundaries = _import_model_boundaries(args.vortex_home)
    hidden = capture["hidden_input"]
    config = boundaries["config"](
        hidden.shape[0], hidden.shape[1], package["shape"]["cache_capacity"]
    )
    module = (
        boundaries["prefill"](config, 1)
        if phase_index == 0
        else boundaries["decode"](config, 1)
    )
    del archive
    canonical_parameters = _deterministic_parameters(
        config,
        lambda current_config, unused_num_layers: boundaries["stack_shapes"](
            current_config, layer + 1
        ),
        int(package["model"]["synthetic_seed"]),
    )
    parameters = {
        local_name: canonical_parameters[
            f"layers.{layer}.{local_name.split('.', 2)[2]}"
        ]
        for local_name in package["layer_parameter_order"]
    }
    call_args = [
        torch.from_numpy(np.array(hidden, copy=True)),
        torch.from_numpy(np.array(capture["positions"], copy=True)),
        parameters,
    ]
    call_args.extend(
        torch.from_numpy(np.array(capture[f"cache_input_{index}"], copy=True))
        for index in range(7)
        if f"cache_input_{index}" in capture
    )
    with torch.no_grad():
        eager_state = module(*call_args)

    comparisons = []
    for index, eager_tensor in enumerate(eager_state):
        eager = eager_tensor.numpy()
        hardware = capture[f"hardware_output_{index}"]
        canonical = capture[f"canonical_output_{index}"]
        comparisons.append(
            {
                "output": index,
                "dtype": str(eager.dtype),
                "shape": list(eager.shape),
                "hardware_vs_eager_same_inputs": _stats(hardware, eager),
                "canonical_vs_eager_same_inputs": _stats(canonical, eager),
                "hardware_vs_canonical": _stats(hardware, canonical),
            }
        )
    print(
        json.dumps(
            {
                "mismatch_artifact": str(args.mismatch_artifact),
                "phase": metadata["phase"],
                "layer": layer,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
