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
"""Compile and package the host-only Llama3-8B C1/C2/C3 matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.relax.backend.vortex import (
    BackendParameterArchive,
    C1_ALL_FP16_TCU,
    C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
    C3_ALL_W4_NAIVE,
    LogicalParameterArchive,
    get_vortex_backend_policy,
    prepare_backend_parameter_archive,
    prepare_logical_parameter_archive,
    validate_vortex_backend_policy,
)
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile


DEFAULT_VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
DEFAULT_ARTIFACT_ROOT = Path("build/llama3_c1_c3_compile_matrix")
MODEL_NAME = "llama3-8b"
NUM_LAYERS = 32
COMPILED_LAYERS = 1
PACKAGE_SCHEMA_VERSION = 1
MATRIX_SCHEMA_VERSION = 1
QUANTIZATION_POLICY = "signed_all_asymmetric_wkv4_v1"
CASES = {
    "S1": (1, 1, 8),
    "S2": (1, 7, 16),
    "S3": (2, 1, 8),
    "S4": (2, 7, 16),
}
ALIAS_POLICIES = {
    "C1": C1_ALL_FP16_TCU,
    "C2": C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
    "C3": C3_ALL_W4_NAIVE,
}
ARTIFACT_NAMES = (
    "embedding_prefill",
    "embedding_decode",
    "prefill_layer",
    "decode_layer",
    "final_head_prefill",
    "final_head_decode",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(directory: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_package_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _import_dependencies(vortex_home: Path):
    spinquant = str(vortex_home / "pytorch/spinquant")
    if spinquant not in sys.path:
        sys.path.insert(0, spinquant)
    vortex_root = str(vortex_home)
    if vortex_root not in sys.path:
        sys.path.insert(0, vortex_root)
    from spinquant_inference.llama3_c4_export import (  # pylint: disable=import-outside-toplevel
        Llama3ExportConfig,
        Llama3FinalHead,
        Llama3StackDecode,
        Llama3StackPrefill,
        Llama3TokenEmbedding,
        embedding_parameter_shapes,
        final_head_parameter_shapes,
        final_head_parameter_shapes_for_compute,
        stack_parameter_shapes,
        stack_parameter_shapes_for_compute,
    )
    from tools.latency_bench.fpga_bins import (  # pylint: disable=import-outside-toplevel
        resolve_fpga_bin_artifacts,
    )

    return {
        "config": Llama3ExportConfig,
        "embedding": Llama3TokenEmbedding,
        "prefill": Llama3StackPrefill,
        "decode": Llama3StackDecode,
        "head": Llama3FinalHead,
        "embedding_shapes": embedding_parameter_shapes,
        "head_shapes_w4": final_head_parameter_shapes,
        "head_shapes_compute": final_head_parameter_shapes_for_compute,
        "stack_shapes_w4": stack_parameter_shapes,
        "stack_shapes_compute": stack_parameter_shapes_for_compute,
        "resolve_alias": resolve_fpga_bin_artifacts,
    }


def _model_metadata(config, seed: int) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "num_layers": NUM_LAYERS,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocabulary_size": config.vocabulary_size,
        "weight_group_size": config.weight_group_size,
        "kv_group_size": config.kv_group_size,
        "quantization_policy": QUANTIZATION_POLICY,
        "synthetic_seed": seed,
        "synthetic_parameter_scheme": "constant_hash_compile_fixture_v1",
    }


def _deterministic_array(name: str, shape, dtype):
    digest = hashlib.sha256(name.encode()).digest()
    dtype = np.dtype(str(dtype).removeprefix("torch."))
    if dtype == np.dtype("uint8"):
        return np.full(shape, digest[0], dtype=dtype)
    if dtype == np.dtype("int16"):
        return np.full(shape, int(digest[0] % 5) - 2, dtype=dtype)
    if name.endswith("norm.weight"):
        return np.ones(shape, dtype=dtype)
    return np.full(shape, (int(digest[0]) - 127) / 32768.0, dtype=dtype)


def prepare_logical_archive(root: Path, dependencies, seed: int) -> Path:
    config = dependencies["config"](1, 1, 8)
    manifest = root / "parameters" / "logical" / "manifest.json"
    if manifest.exists():
        LogicalParameterArchive(
            manifest,
            expected_num_layers=NUM_LAYERS,
            expected_model_metadata=_model_metadata(config, seed),
        )
        return manifest
    shapes = dependencies["stack_shapes_w4"](config, COMPILED_LAYERS)
    shapes.update(dependencies["embedding_shapes"](config))
    shapes.update(dependencies["head_shapes_w4"](config))
    parameters = {
        name: _deterministic_array(name, shape, dtype)
        for name, (shape, dtype) in shapes.items()
    }
    return prepare_logical_parameter_archive(
        manifest.parent,
        parameters,
        num_layers=NUM_LAYERS,
        included_layers=(0,),
        model_metadata=_model_metadata(config, seed),
    )


def _profile_identity(alias, artifacts, alias_map, profile):
    return {
        "alias": alias,
        "alias_map": str(alias_map.resolve()),
        "alias_map_sha256": _sha256_file(alias_map),
        "config": str(artifacts.config),
        "config_sha256": _sha256_file(artifacts.config),
        "manifest": str(artifacts.manifest),
        "manifest_sha256": _sha256_file(artifacts.manifest),
        "xclbin": str(artifacts.xclbin),
        "xclbin_sha256": _sha256_file(artifacts.xclbin),
        "profile_name": profile.name,
        "profile_fingerprint": profile.fingerprint,
        "target": str(profile.target),
    }


def resolve_backend(alias, alias_map: Path, dependencies):
    if alias not in ALIAS_POLICIES:
        raise ValueError(f"unsupported Llama backend alias {alias!r}")
    artifacts = dependencies["resolve_alias"](
        alias, alias_map_path=alias_map, require_alias=True
    )
    profile = load_vortex_accelerator_profile(artifacts.manifest)
    target = tvm.target.Target(profile.target, host="llvm")
    policy = validate_vortex_backend_policy(target, ALIAS_POLICIES[alias])
    return artifacts, profile, target, policy


def prepare_materialization(
    root: Path, alias: str, logical: LogicalParameterArchive, profile, target, policy
):
    manifest = root / "parameters" / alias / "manifest.json"
    if not manifest.exists():
        prepare_backend_parameter_archive(
            manifest.parent,
            logical,
            policy=policy,
            target=target,
            profile_fingerprint=profile.fingerprint,
        )
    return BackendParameterArchive(
        manifest,
        expected_policy=policy.name,
        expected_profile_fingerprint=profile.fingerprint,
        expected_logical_manifest_sha256=logical.manifest_sha256,
        expected_logical_content_sha256=logical.content_sha256,
    )


def _torch_parameters(archive, names: Sequence[str]):
    result = {}
    for name in names:
        value = archive.tensor(name)
        value.flags.writeable = False
        result[name] = torch.from_numpy(value)
    return result


def _sample_cache(config):
    payload = torch.zeros(
        (
            COMPILED_LAYERS,
            config.batch_size,
            config.num_key_value_heads,
            config.cache_capacity,
            config.head_dim // 2,
        ),
        dtype=torch.uint8,
    )
    scale = torch.zeros(
        (
            COMPILED_LAYERS,
            config.batch_size,
            config.num_key_value_heads,
            config.cache_capacity,
            1,
        ),
        dtype=torch.float16,
    )
    zero = torch.zeros_like(scale, dtype=torch.int16)
    length = torch.zeros((COMPILED_LAYERS,), dtype=torch.int64)
    return payload, scale, zero, payload.clone(), scale.clone(), zero.clone(), length


def _build_inputs(dependencies, policy, config, archive):
    decode_config = dependencies["config"](
        config.batch_size, 1, config.cache_capacity
    )
    linear_compute = "fp16" if policy.linear_compute == "fp16_tcu" else "w4"
    attention_compute = (
        "fp16" if policy.attention_compute == "fp16_tcu" else "w4"
    )
    layer_order = tuple(
        dependencies["stack_shapes_compute"](
            config, COMPILED_LAYERS, linear_compute
        )
    )
    head_order = tuple(
        dependencies["head_shapes_compute"](config, linear_compute)
    )
    embedding_order = tuple(dependencies["embedding_shapes"](config))
    layer_parameters = _torch_parameters(archive, layer_order)
    head_parameters = _torch_parameters(archive, head_order)
    embedding_parameters = _torch_parameters(archive, embedding_order)
    token_ids = torch.zeros(
        (config.batch_size, config.query_length), dtype=torch.int64
    )
    positions = torch.arange(config.query_length, dtype=torch.int64).repeat(
        config.batch_size, 1
    )
    hidden = torch.zeros(
        (config.batch_size, config.query_length, config.hidden_size),
        dtype=torch.float16,
    )
    builds = {
        "embedding_prefill": (
            dependencies["embedding"](),
            (token_ids, embedding_parameters),
        ),
        "embedding_decode": (
            dependencies["embedding"](),
            (token_ids[:, :1], embedding_parameters),
        ),
        "prefill_layer": (
            dependencies["prefill"](
                config,
                COMPILED_LAYERS,
                linear_compute=linear_compute,
                attention_compute=attention_compute,
            ),
            (hidden, positions, layer_parameters),
        ),
        "decode_layer": (
            dependencies["decode"](
                decode_config,
                COMPILED_LAYERS,
                linear_compute=linear_compute,
                attention_compute=attention_compute,
            ),
            (
                hidden[:, :1, :],
                positions[:, :1],
                layer_parameters,
                *_sample_cache(decode_config),
            ),
        ),
        "final_head_prefill": (
            dependencies["head"](config, linear_compute=linear_compute),
            (hidden, head_parameters),
        ),
        "final_head_decode": (
            dependencies["head"](decode_config, linear_compute=linear_compute),
            (hidden[:, :1, :], head_parameters),
        ),
    }
    return builds, layer_order, head_order, embedding_order


def _inventory(lowered) -> dict[str, object]:
    script = lowered.script()
    attrs = dict(lowered.attrs or {})
    selected_attrs = {
        str(key): int(value) if str(value).isdigit() else str(value)
        for key, value in attrs.items()
        if str(key).startswith("vortex.")
    }
    return {
        "module_attrs": selected_attrs,
        "unresolved_w4_calls": script.count("relax.vortex.mm_w4a16"),
        "unresolved_fp16_calls": script.count("relax.vortex.fp16_matmul"),
        "naive_helper_definitions": script.count("vx_tvm_gemm_w4a16"),
        "improve_helper_definitions": script.count("vx_tvm_gemm_w4a16_v2"),
        "tcu_helper_definitions": script.count("vx_tvm_tcu_fp16_tile"),
        "hadamard_helper_definitions": script.count("def vortex_hadamard_"),
        "causal_softmax_helper_definitions": script.count(
            "def vortex_causal_softmax"
        ),
    }


def _assert_inventory(name: str, policy, inventory):
    if inventory["unresolved_w4_calls"] or inventory["unresolved_fp16_calls"]:
        raise ValueError(f"unresolved logical GEMM in {name}: {inventory}")
    if inventory["improve_helper_definitions"]:
        raise ValueError(f"C4 IMPROVE kernel leaked into {policy.name} artifact {name}")
    is_compute = name in (
        "prefill_layer",
        "decode_layer",
        "final_head_prefill",
        "final_head_decode",
    )
    if not is_compute:
        return
    module_attrs = inventory["module_attrs"]
    if name in ("prefill_layer", "decode_layer"):
        if inventory["hadamard_helper_definitions"] != 3:
            raise ValueError(f"Hadamard kernel inventory mismatch in {name}: {inventory}")
        if inventory["causal_softmax_helper_definitions"] != 1:
            raise ValueError(f"causal-softmax kernel inventory mismatch in {name}: {inventory}")
    if policy.name == C1_ALL_FP16_TCU:
        if not inventory["tcu_helper_definitions"] or inventory["naive_helper_definitions"]:
            raise ValueError(f"C1 routing mismatch in {name}: {inventory}")
        expected_matmuls = 9 if name in ("prefill_layer", "decode_layer") else 1
        if int(module_attrs.get("vortex.tcu.fp16.lowered_matmuls", 0)) != expected_matmuls:
            raise ValueError(f"C1 logical matmul count mismatch in {name}: {inventory}")
    elif policy.name == C3_ALL_W4_NAIVE:
        if not inventory["naive_helper_definitions"] or inventory["tcu_helper_definitions"]:
            raise ValueError(f"C3 routing mismatch in {name}: {inventory}")
        if module_attrs.get("vortex.w4a16.physical_layout") != "row_major":
            raise ValueError(f"C3 physical layout mismatch in {name}: {inventory}")
        if int(module_attrs.get("vortex.w4a16.lowered", 0)) < (
            9 if name in ("prefill_layer", "decode_layer") else 1
        ):
            raise ValueError(f"C3 logical matmul count mismatch in {name}: {inventory}")
    else:
        if name in ("prefill_layer", "decode_layer") and not (
            inventory["naive_helper_definitions"] and inventory["tcu_helper_definitions"]
        ):
            raise ValueError(f"C2 mixed routing mismatch in {name}: {inventory}")


def _compile_one(model, inputs, target, policy, exec_mode):
    exported = torch.export.export(model, inputs, strict=True)
    mod = from_exported_program(
        exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True
    )
    pipeline = relax.backend.vortex.get_default_pipeline(
        target, backend_policy=policy.name
    )
    start = time.perf_counter()
    lowered = pipeline(mod)
    executable = relax.build(
        lowered,
        target,
        relax_pipeline=tvm.transform.Sequential([]),
        exec_mode=exec_mode,
    )
    return executable, time.perf_counter() - start, _inventory(lowered)


def compile_package(
    root: Path,
    alias: str,
    case: str,
    dependencies,
    alias_map: Path,
    logical: LogicalParameterArchive,
    materialized: BackendParameterArchive,
    artifacts,
    profile,
    target,
    policy,
    seed: int,
    force: bool = False,
):
    batch, prompt, capacity = CASES[case]
    config = dependencies["config"](batch, prompt, capacity)
    builds, layer_order, head_order, embedding_order = _build_inputs(
        dependencies, policy, config, materialized
    )
    package_dir = root / "packages" / alias / case
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "package.json"
    if package_path.exists() and not force:
        load_compile_package(package_path, alias_map, dependencies)
        return package_path
    artifact_records = {}
    modes = ("bytecode", "compiled") if case == "S1" else ("bytecode",)
    for exec_mode in modes:
        for name, (model, inputs) in builds.items():
            executable, seconds, inventory = _compile_one(
                model, inputs, target, policy, exec_mode
            )
            _assert_inventory(name, policy, inventory)
            suffix = "" if exec_mode == "bytecode" else ".compiled"
            artifact_path = package_dir / f"{name}{suffix}.so"
            executable.export_library(str(artifact_path))
            tvm.runtime.load_module(str(artifact_path))
            artifact_records[f"{exec_mode}:{name}"] = {
                "file": artifact_path.name,
                "sha256": _sha256_file(artifact_path),
                "nbytes": artifact_path.stat().st_size,
                "build_seconds": seconds,
                "kernel_inventory": inventory,
            }
            del executable
            gc.collect()
    profile_identity = _profile_identity(alias, artifacts, alias_map, profile)
    package = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "format": "vortex-llama3-backend-compile-package",
        "model": _model_metadata(config, seed),
        "shape_case": case,
        "shape": {
            "batch_size": batch,
            "prompt_length": prompt,
            "cache_capacity": capacity,
        },
        "alias": alias,
        "backend_policy": policy.name,
        "workload_variant": policy.workload_variant,
        "layout_policy": policy.layout_policy,
        "profile": profile_identity,
        "logical_archive_manifest": _relative_or_absolute(
            logical.manifest_path, package_dir
        ),
        "logical_archive_manifest_sha256": logical.manifest_sha256,
        "logical_content_sha256": logical.content_sha256,
        "materialization_manifest": _relative_or_absolute(
            materialized.manifest_path, package_dir
        ),
        "materialization_manifest_sha256": materialized.manifest_sha256,
        "parameter_orders": {
            "embedding": list(embedding_order),
            "layer": list(layer_order),
            "head": list(head_order),
        },
        "artifacts": artifact_records,
        "revisions": {
            "tvm": _git_revision(Path(__file__).resolve().parents[2]),
            "vortex": _git_revision(DEFAULT_VORTEX_HOME),
        },
    }
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_compile_package(package_path, alias_map, dependencies)
    return package_path


def _expected_artifact_keys(case: str) -> set[str]:
    modes = ("bytecode", "compiled") if case == "S1" else ("bytecode",)
    return {f"{mode}:{name}" for mode in modes for name in ARTIFACT_NAMES}


def load_compile_package(package_path: Path, alias_map: Path, dependencies):
    """Fail-closed host reload of package identity, archives, and VM modules."""

    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported Llama backend compile package schema")
    if package.get("format") != "vortex-llama3-backend-compile-package":
        raise ValueError("invalid Llama backend compile package format")
    case = package.get("shape_case")
    if case not in CASES:
        raise ValueError(f"unknown Llama backend compile package shape case: {case!r}")
    batch, prompt, capacity = CASES[case]
    expected_shape = {
        "batch_size": batch,
        "prompt_length": prompt,
        "cache_capacity": capacity,
    }
    if package.get("shape") != expected_shape:
        raise ValueError("Llama backend compile package shape mismatch")
    alias = package["alias"]
    artifacts, profile, target, policy = resolve_backend(
        alias, alias_map, dependencies
    )
    expected_profile = _profile_identity(alias, artifacts, alias_map, profile)
    if package["profile"] != expected_profile:
        raise ValueError("Llama backend compile package profile identity mismatch")
    if package["backend_policy"] != policy.name:
        raise ValueError("Llama backend compile package policy mismatch")
    if package.get("workload_variant") != policy.workload_variant:
        raise ValueError("Llama backend compile package workload variant mismatch")
    if package.get("layout_policy") != policy.layout_policy:
        raise ValueError("Llama backend compile package layout policy mismatch")
    root = package_path.parent
    logical_manifest = _resolve_package_path(package["logical_archive_manifest"], root)
    if _sha256_file(logical_manifest) != package["logical_archive_manifest_sha256"]:
        raise ValueError("Llama backend compile package logical manifest hash mismatch")
    logical = LogicalParameterArchive(
        logical_manifest,
        expected_num_layers=NUM_LAYERS,
        expected_model_metadata=package["model"],
    )
    if package.get("logical_content_sha256") != logical.content_sha256:
        raise ValueError("Llama backend compile package logical content hash mismatch")
    materialization_manifest = _resolve_package_path(
        package["materialization_manifest"], root
    )
    if _sha256_file(materialization_manifest) != package["materialization_manifest_sha256"]:
        raise ValueError("Llama backend materialization manifest hash mismatch")
    BackendParameterArchive(
        materialization_manifest,
        expected_policy=policy.name,
        expected_profile_fingerprint=profile.fingerprint,
        expected_logical_manifest_sha256=logical.manifest_sha256,
        expected_logical_content_sha256=logical.content_sha256,
    )
    linear_compute = "fp16" if policy.linear_compute == "fp16_tcu" else "w4"
    config = dependencies["config"](batch, prompt, capacity)
    expected_orders = {
        "embedding": list(dependencies["embedding_shapes"](config)),
        "layer": list(
            dependencies["stack_shapes_compute"](
                config, COMPILED_LAYERS, linear_compute
            )
        ),
        "head": list(dependencies["head_shapes_compute"](config, linear_compute)),
    }
    if package.get("parameter_orders") != expected_orders:
        raise ValueError("Llama backend compile package parameter order mismatch")
    artifact_records = package.get("artifacts", {})
    expected_artifact_keys = _expected_artifact_keys(case)
    if set(artifact_records) != expected_artifact_keys:
        raise ValueError("Llama backend compile package artifact inventory mismatch")
    for key, record in artifact_records.items():
        mode, name = key.split(":", 1)
        suffix = "" if mode == "bytecode" else ".compiled"
        expected_file = f"{name}{suffix}.so"
        if record.get("file") != expected_file:
            raise ValueError(f"Llama backend artifact filename mismatch: {key}")
        artifact = root / record["file"]
        if artifact.stat().st_size != record["nbytes"]:
            raise ValueError(f"Llama backend artifact size mismatch: {artifact.name}")
        if _sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"Llama backend artifact hash mismatch: {artifact.name}")
        tvm.runtime.load_module(str(artifact))
    del target
    return package


def parse_csv(value: str, valid: Mapping[str, object]) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = tuple(item for item in selected if item not in valid)
    if not selected or invalid:
        raise ValueError(f"invalid selection {invalid or selected}; expected {tuple(valid)}")
    return selected


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    parser.add_argument(
        "--alias-map",
        type=Path,
        default=DEFAULT_VORTEX_HOME / "ci/fpga_bin_alias_map.yaml",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--aliases", default="C1,C3")
    parser.add_argument("--cases", default="S1,S2,S3,S4")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--check-aliases-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    aliases = parse_csv(args.aliases, ALIAS_POLICIES)
    cases = parse_csv(args.cases, CASES)
    dependencies = _import_dependencies(args.vortex_home)
    resolved = {}
    failures = {}
    for alias in aliases:
        try:
            resolved[alias] = resolve_backend(alias, args.alias_map, dependencies)
        except (FileNotFoundError, ValueError) as error:
            failures[alias] = str(error)
    if failures:
        raise SystemExit(json.dumps({"alias_resolution_failures": failures}, indent=2))
    if args.check_aliases_only:
        print(
            json.dumps(
                {
                    alias: _profile_identity(
                        alias, values[0], args.alias_map, values[1]
                    )
                    for alias, values in resolved.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    logical_manifest = prepare_logical_archive(
        args.artifact_root, dependencies, args.seed
    )
    logical = LogicalParameterArchive(
        logical_manifest,
        expected_num_layers=NUM_LAYERS,
        expected_model_metadata=_model_metadata(dependencies["config"](1, 1, 8), args.seed),
    )
    package_paths = []
    backend_records = {}
    for alias in aliases:
        artifacts, profile, target, policy = resolved[alias]
        materialized = prepare_materialization(
            args.artifact_root,
            alias,
            logical,
            profile,
            target,
            policy,
        )
        backend_records[alias] = {
            "policy": policy.name,
            "profile": _profile_identity(
                alias, artifacts, args.alias_map, profile
            ),
            "materialization_manifest": str(materialized.manifest_path),
            "materialization_nbytes": materialized.manifest["data_nbytes"],
        }
        for case in cases:
            package_paths.append(
                compile_package(
                    args.artifact_root,
                    alias,
                    case,
                    dependencies,
                    args.alias_map,
                    logical,
                    materialized,
                    artifacts,
                    profile,
                    target,
                    policy,
                    args.seed,
                    args.force,
                )
            )
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "format": "vortex-llama3-backend-compile-matrix",
        "aliases": list(aliases),
        "cases": list(cases),
        "logical_manifest": str(logical.manifest_path),
        "logical_content_sha256": logical.content_sha256,
        "logical_nbytes": logical.manifest["data_nbytes"],
        "backends": backend_records,
        "packages": [str(path) for path in package_paths],
        "c2_status": (
            "compiled" if "C2" in aliases else "deferred_until_exact_binary_is_available"
        ),
    }
    matrix_path = args.artifact_root / "matrix.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(matrix_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
