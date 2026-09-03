#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information regarding copyright
# ownership.  The ASF licenses this file to You under the Apache License, Version 2.0.
"""Stress the C1 batch-2 residual kernel with alternating exact inputs on U55C."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import tvm
from tvm import relax

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    load_backend_package,
    package_identity,
    sha256_array,
    sha256_file,
    validate_reference_artifact,
)
from vortex_llama3.run_backend_validation import (
    DEFAULT_ALIAS_MAP,
    DEFAULT_VORTEX_HOME,
    _configure_xrt_environment,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe-package", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=256)
    parser.add_argument("--allocator", choices=("naive", "pooled"), default="pooled")
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
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
    probe = json.loads(args.probe_package.read_text(encoding="utf-8"))
    if probe.get("format") != "vortex-llama3-c1-residual-stress-package":
        raise ValueError("invalid residual probe package format")
    if probe["source_package"] != package_identity(loaded.package):
        raise ValueError("residual probe source package identity mismatch")
    if probe["reference_sha256"] != reference_metadata["npz_sha256"]:
        raise ValueError("residual probe reference mismatch")
    root = args.probe_package.parent
    artifact = root / probe["artifact"]["file"]
    inputs_path = root / probe["inputs"]["file"]
    if sha256_file(artifact) != probe["artifact"]["sha256"]:
        raise ValueError("residual probe artifact hash mismatch")
    if sha256_file(inputs_path) != probe["inputs"]["sha256"]:
        raise ValueError("residual probe input hash mismatch")
    with np.load(inputs_path) as arrays:
        base_left = np.array(arrays["left"], copy=True)
        base_right = np.array(arrays["right"], copy=True)
    if sha256_array(base_left) != probe["inputs"]["left_sha256"]:
        raise ValueError("residual left input hash mismatch")
    if sha256_array(base_right) != probe["inputs"]["right_sha256"]:
        raise ValueError("residual right input hash mismatch")
    _configure_xrt_environment(loaded.package)
    if torch.cuda.is_initialized():
        raise RuntimeError("residual U55C probe initialized CUDA")
    device = tvm.vortex(0)
    module = tvm.runtime.load_module(str(artifact))
    vm = relax.VirtualMachine(module, device=device, memory_cfg=args.allocator)
    left = tvm.runtime.empty(base_left.shape, str(base_left.dtype), device)
    right = tvm.runtime.empty(base_right.shape, str(base_right.dtype), device)
    mismatch = None
    output_hashes = []
    start = time.perf_counter()
    for repetition in range(args.repetitions):
        sign = 1.0 if repetition % 2 == 0 else -1.0
        expected_left = (base_left.astype("float32") * sign).astype("float16")
        left.copyfrom(expected_left)
        right.copyfrom(base_right)
        left_readback = left.numpy()
        right_readback = right.numpy()
        if not np.array_equal(left_readback, expected_left) or not np.array_equal(
            right_readback, base_right
        ):
            raise AssertionError(f"residual input readback mismatch at {repetition}")
        output = vm["main"](left, right)
        actual = output.numpy()
        expected = (
            expected_left.astype("float32") + base_right.astype("float32")
        ).astype("float16")
        if not np.array_equal(actual, expected):
            coordinates = np.argwhere(actual != expected)
            rereads = [output.numpy(), output.numpy()]
            mismatch = {
                "repetition": repetition,
                "count": int(coordinates.shape[0]),
                "coordinates": coordinates[:32].tolist(),
                "flattened_indices": [
                    int(np.ravel_multi_index(tuple(index), expected.shape))
                    for index in coordinates[:32]
                ],
                "actual": [float(actual[tuple(index)]) for index in coordinates[:32]],
                "expected": [
                    float(expected[tuple(index)]) for index in coordinates[:32]
                ],
                "reread_values": [
                    [float(value[tuple(index)]) for index in coordinates[:32]]
                    for value in rereads
                ],
                "rereads_match_expected": [
                    bool(np.array_equal(value, expected)) for value in rereads
                ],
            }
            break
        output_hashes.append(sha256_array(actual))
        if repetition == 0 or (repetition + 1) % 32 == 0:
            print(
                json.dumps(
                    {"event": "residual_probe_progress", "completed": repetition + 1}
                ),
                flush=True,
            )
    trace = {
        "format": "vortex-llama3-c1-residual-stress-trace",
        "package_identity": package_identity(loaded.package),
        "reference_device": reference_metadata["reference_device"],
        "reference_sha256": reference_metadata["npz_sha256"],
        "probe_package_sha256": sha256_file(args.probe_package),
        "allocator": args.allocator,
        "requested_repetitions": args.repetitions,
        "completed_repetitions": len(output_hashes),
        "alternating_input": True,
        "input_readback_verified_each_repetition": True,
        "mismatch": mismatch,
        "latency_seconds": time.perf_counter() - start,
        "output_hashes": output_hashes,
        "runtime_environment": {
            "hostname": socket.gethostname(),
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
    if mismatch is not None:
        raise AssertionError(f"residual kernel mismatch: {mismatch}")
    print(
        json.dumps(
            {"event": "residual_probe_complete", "trace": str(args.trace_output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
