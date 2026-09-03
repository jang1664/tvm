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
"""Generate a deterministic backend-matched C1/C3 PyTorch reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from vortex_llama3.backend_numerical_validation import (
    EXPECTED_CUBLAS_WORKSPACE_CONFIG,
    REFERENCE_SEED,
    compare_replays,
    configure_reference_device,
    deterministic_prompt,
    generate_reference_arrays,
    load_backend_package,
    validate_reference_artifact,
    write_reference_artifact,
)


DEFAULT_VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
DEFAULT_ALIAS_MAP = DEFAULT_VORTEX_HOME / "ci/fpga_bin_alias_map.yaml"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--determinism-replays", type=int, default=2)
    parser.add_argument(
        "--diagnostic-cpu",
        action="store_true",
        help="generate a labeled CPU diagnostic that cannot satisfy GPU acceptance",
    )
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument(
        "--vortex-home",
        type=Path,
        default=Path(os.environ.get("VORTEX_HOME", DEFAULT_VORTEX_HOME)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.seed != REFERENCE_SEED:
        raise ValueError(f"validation seed is frozen to {REFERENCE_SEED}")
    if args.determinism_replays <= 0:
        raise ValueError("determinism replay count must be positive")
    os.environ["VORTEX_HOME"] = str(args.vortex_home.resolve())
    device_label = "diagnostic_cpu" if args.diagnostic_cpu else "gpu"
    if device_label == "gpu" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        raise ValueError(
            "set CUBLAS_WORKSPACE_CONFIG="
            f"{EXPECTED_CUBLAS_WORKSPACE_CONFIG} before launching the GPU process"
        )
    loaded = load_backend_package(
        args.package,
        args.alias_map,
        args.vortex_home,
        expected_alias=args.alias,
        expected_case=args.case,
    )
    rows = deterministic_prompt(args.case)
    device, device_metadata = configure_reference_device(device_label)
    first = generate_reference_arrays(loaded, rows, device)
    for _ in range(1, args.determinism_replays):
        replay = generate_reference_arrays(loaded, rows, device)
        compare_replays(first, replay)
    metadata_path = write_reference_artifact(
        args.output,
        first,
        loaded,
        rows,
        device_metadata,
        seed=args.seed,
        replay_count=args.determinism_replays,
    )
    metadata, arrays = validate_reference_artifact(
        args.output,
        loaded,
        expected_device=device_label,
        expected_seed=args.seed,
    )
    arrays.close()
    print(
        json.dumps(
            {
                "event": "backend_reference_complete",
                "metadata": str(metadata_path),
                "npz": str(args.output.resolve()),
                "alias": args.alias,
                "case": args.case,
                "reference_device": device_label,
                "tensor_count": metadata["tensor_count"],
                "determinism_replay_count": args.determinism_replays,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
