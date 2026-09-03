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
"""Compile exact C1/C3 KV-cache and attention stage probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

from vortex_llama3.backend_numerical_validation import (
    REFERENCE_SEED,
    _import_dependencies,
    load_backend_package,
    package_identity,
    sha256_file,
    validate_reference_artifact,
)
from vortex_llama3.compile_backend_matrix import _inventory
from vortex_llama3.run_backend_validation import DEFAULT_ALIAS_MAP, DEFAULT_VORTEX_HOME


class KVCachePrefillProbe(torch.nn.Module):
    """Run K/V projection, quantization, and the S1 prefill cache writes."""

    def __init__(self, layer, config, linear_compute: str) -> None:
        super().__init__()
        self.layer = layer
        self.config = config
        self.linear_compute = linear_compute

    def _local_parameters(
        self, parameters: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        suffixes = (
            ("weight",)
            if self.linear_compute == "fp16"
            else (
                "qweight",
                "scales",
                "zeros",
            )
        )
        return {
            f"{projection}.{suffix}": parameters[f"layers.0.{projection}.{suffix}"]
            for projection in ("k_proj", "v_proj")
            for suffix in suffixes
        }

    def forward(
        self,
        flattened_normalized: torch.Tensor,
        position_ids: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        config = self.config
        normalized = flattened_normalized.reshape(
            config.batch_size, config.query_length, config.hidden_size
        )
        key_update, value_update = self.layer._project_kv(
            normalized, position_ids, self._local_parameters(parameters)
        )
        packed_shape = (
            config.batch_size,
            config.num_key_value_heads,
            config.cache_capacity,
            config.head_dim // 2,
        )
        qparam_shape = (*packed_shape[:-1], 1)
        key_cache = (
            normalized.new_zeros(packed_shape, dtype=torch.uint8),
            normalized.new_zeros(qparam_shape, dtype=torch.float16),
            normalized.new_zeros(qparam_shape, dtype=torch.int16),
        )
        value_cache = tuple(tensor.clone() for tensor in key_cache)
        for position in range(config.query_length):
            key_cache = torch.ops.vortex.kv_cache_update(
                *key_cache,
                *(tensor[..., position : position + 1, :] for tensor in key_update),
                position,
                config.cache_capacity,
            )
            value_cache = torch.ops.vortex.kv_cache_update(
                *value_cache,
                *(tensor[..., position : position + 1, :] for tensor in value_update),
                position,
                config.cache_capacity,
            )
        return (*key_cache, *value_cache)


class LayerCheckpointsProbe(torch.nn.Module):
    """Expose all production layer checkpoints in execution order."""

    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
        *cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        local_parameters = {
            name.removeprefix("layers.0."): value for name, value in parameters.items()
        }
        return self.layer(hidden, position_ids, local_parameters, *cache)


class AttentionProbe(torch.nn.Module):
    """Run cache dequantization, QK, causal softmax, and PV as one stage."""

    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        query: torch.Tensor,
        position_ids: torch.Tensor,
        key_payload: torch.Tensor,
        key_scale: torch.Tensor,
        key_zero: torch.Tensor,
        value_payload: torch.Tensor,
        value_scale: torch.Tensor,
        value_zero: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.layer._attention_checkpoints(
            query,
            (key_payload, key_scale, key_zero),
            (value_payload, value_scale, value_zero),
            valid_length,
            position_ids,
        )


class AttentionFinalProbe(AttentionProbe):
    """Match production liveness by returning only the final attention context."""

    def forward(
        self,
        query: torch.Tensor,
        position_ids: torch.Tensor,
        key_payload: torch.Tensor,
        key_scale: torch.Tensor,
        key_zero: torch.Tensor,
        value_payload: torch.Tensor,
        value_scale: torch.Tensor,
        value_zero: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.layer._attention(
            query,
            (key_payload, key_scale, key_zero),
            (value_payload, value_scale, value_zero),
            valid_length,
            position_ids,
        )


class SoftmaxProbe(torch.nn.Module):
    """Run only the causal mask and row softmax helper."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self,
        scores: torch.Tensor,
        position_ids: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vortex.causal_softmax(
            scores, position_ids, valid_length, self.head_dim
        )


class AttentionSubstageProbe(torch.nn.Module):
    """Isolate dequant-QK, QK-softmax, and softmax-PV producer boundaries."""

    def __init__(self, layer, mode: str) -> None:
        super().__init__()
        self.layer = layer
        self.mode = mode

    def _qk(
        self,
        query: torch.Tensor,
        key_payload: torch.Tensor,
        key_scale: torch.Tensor,
        key_zero: torch.Tensor,
    ) -> torch.Tensor:
        config = self.layer.config
        groups = config.query_heads_per_kv_head
        query_grouped = query.reshape(
            config.batch_size,
            config.num_key_value_heads,
            groups,
            query.shape[-2],
            config.head_dim,
        )
        key_payload = key_payload.unsqueeze(2)
        key_scale = key_scale.unsqueeze(2)
        key_zero = key_zero.unsqueeze(2)
        key_shape = [
            config.batch_size,
            config.num_key_value_heads,
            1,
            config.cache_capacity,
            config.head_dim,
        ]
        if self.layer.attention_compute == "fp16":
            key = torch.ops.vortex.dequantize_int4(
                key_payload,
                key_scale,
                key_zero,
                key_shape,
                4,
                config.kv_group_size,
                4,
                "signed_asymmetric_int4",
            )
            return torch.ops.vortex.fp16_matmul(
                query_grouped, key.transpose(-2, -1), "attention.qk"
            )
        return torch.ops.vortex.mm_w4a16(
            query_grouped,
            key_payload,
            key_scale,
            key_zero,
            key_shape,
            config.kv_group_size,
            4,
            4,
            "signed_asymmetric_int4",
            True,
        )

    def _pv(
        self,
        probabilities: torch.Tensor,
        value_payload: torch.Tensor,
        value_scale: torch.Tensor,
        value_zero: torch.Tensor,
    ) -> torch.Tensor:
        config = self.layer.config
        value_payload = value_payload.unsqueeze(2)
        value_scale = value_scale.unsqueeze(2)
        value_zero = value_zero.unsqueeze(2)
        value_shape = [
            config.batch_size,
            config.num_key_value_heads,
            1,
            config.cache_capacity,
            config.head_dim,
        ]
        if self.layer.attention_compute == "fp16":
            value = torch.ops.vortex.dequantize_int4(
                value_payload,
                value_scale,
                value_zero,
                value_shape,
                4,
                config.kv_group_size,
                4,
                "signed_asymmetric_int4",
            )
            context = torch.ops.vortex.fp16_matmul(probabilities, value, "attention.pv")
        else:
            context = torch.ops.vortex.mm_w4a16(
                probabilities,
                value_payload,
                value_scale,
                value_zero,
                value_shape,
                config.kv_group_size,
                4,
                4,
                "signed_asymmetric_int4",
                False,
            )
        return context.reshape(
            config.batch_size,
            config.num_attention_heads,
            probabilities.shape[-2],
            config.head_dim,
        )

    def forward(
        self,
        query_or_scores: torch.Tensor,
        position_ids: torch.Tensor,
        key_payload: torch.Tensor,
        key_scale: torch.Tensor,
        key_zero: torch.Tensor,
        value_payload: torch.Tensor,
        value_scale: torch.Tensor,
        value_zero: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "dequant_qk":
            return self._qk(query_or_scores, key_payload, key_scale, key_zero)
        if self.mode == "qk_softmax":
            scores = self._qk(query_or_scores, key_payload, key_scale, key_zero)
            _, probabilities = torch.ops.vortex.causal_softmax(
                scores, position_ids, valid_length, self.layer.config.head_dim
            )
            return probabilities
        if self.mode == "softmax_pv":
            _, probabilities = torch.ops.vortex.causal_softmax(
                query_or_scores,
                position_ids,
                valid_length,
                self.layer.config.head_dim,
            )
            return self._pv(probabilities, value_payload, value_scale, value_zero)
        raise ValueError(f"unsupported attention substage: {self.mode}")


class DequantKeyProbe(torch.nn.Module):
    """Materialize the exact C1 key-cache RHS without invoking QK."""

    def __init__(self, config, transpose: bool) -> None:
        super().__init__()
        self.config = config
        self.transpose = transpose

    def forward(
        self,
        key_payload: torch.Tensor,
        key_scale: torch.Tensor,
        key_zero: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        key = torch.ops.vortex.dequantize_int4(
            key_payload.unsqueeze(2),
            key_scale.unsqueeze(2),
            key_zero.unsqueeze(2),
            [
                config.batch_size,
                config.num_key_value_heads,
                1,
                config.cache_capacity,
                config.head_dim,
            ],
            4,
            config.kv_group_size,
            4,
            "signed_asymmetric_int4",
        )
        return key.transpose(-2, -1) if self.transpose else key


class TransposeKeyProbe(torch.nn.Module):
    """Materialize only the key-cache transpose from a GPU-produced tensor."""

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return key.transpose(-2, -1)


class DequantKeyMatrixProbe(torch.nn.Module):
    """Run dequantization directly on its flattened rank-2 physical ABI."""

    def __init__(self, rows: int, columns: int, group_size: int) -> None:
        super().__init__()
        self.logical_shape = [rows, columns]
        self.group_size = group_size

    def forward(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vortex.dequantize_int4(
            packed,
            scale,
            zero,
            self.logical_shape,
            1,
            self.group_size,
            1,
            "signed_asymmetric_int4",
        )


def _torch_parameters(archive, names):
    return {
        name: torch.from_numpy(np.array(archive.tensor(name), copy=True))
        for name in names
    }


def _build(model, inputs, target, policy):
    exported = torch.export.export(model, inputs, strict=True)
    mod = from_exported_program(
        exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True
    )
    lowered = relax.backend.vortex.get_default_pipeline(target, backend_policy=policy)(
        mod
    )
    executable = relax.build(
        lowered,
        target,
        relax_pipeline=tvm.transform.Sequential([]),
        exec_mode="bytecode",
    )
    return executable, _inventory(lowered)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alias", choices=("C1", "C3"), required=True)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probes", nargs="+")
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--vortex-home", type=Path, default=DEFAULT_VORTEX_HOME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
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
    dependencies = _import_dependencies(args.vortex_home)
    from spinquant_inference.llama3_c4_export import (  # pylint: disable=import-outside-toplevel
        Llama3LayerDecodeCheckpoints,
        Llama3LayerPrefillCheckpoints,
        layer_checkpoint_names,
    )

    shape = loaded.package["shape"]
    config = dependencies["config"](
        shape["batch_size"], shape["prompt_length"], shape["cache_capacity"]
    )
    layer = dependencies["prefill"](
        config,
        1,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    ).layers[0]
    checkpoint_layer = Llama3LayerPrefillCheckpoints(
        config,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    )
    decode_config = dependencies["config"](
        shape["batch_size"], 1, shape["cache_capacity"]
    )
    checkpoint_decode_layer = Llama3LayerDecodeCheckpoints(
        decode_config,
        linear_compute=loaded.linear_compute,
        attention_compute=loaded.attention_compute,
    )
    suffixes = (
        ("weight",)
        if loaded.linear_compute == "fp16"
        else (
            "qweight",
            "scales",
            "zeros",
        )
    )
    kv_parameter_names = [
        f"layers.0.{projection}.{suffix}"
        for projection in ("k_proj", "v_proj")
        for suffix in suffixes
    ]
    cache_inputs = [
        "probe_prefill_l0_state_0",
        "probe_prefill_l0_state_1",
        "probe_prefill_l0_state_2",
        "probe_prefill_l0_state_3",
        "probe_prefill_l0_state_4",
        "probe_prefill_l0_state_5",
        "probe_prefill_l0_state_6",
    ]
    attention_inputs = [
        "probe_prefill_l0_query_after_rope",
        "p0_positions",
        *cache_inputs,
    ]
    softmax_pv_inputs = [
        "probe_prefill_l0_attention_scores",
        "p0_positions",
        *cache_inputs,
    ]
    builds = {
        "layer_checkpoints": (
            LayerCheckpointsProbe(checkpoint_layer),
            (
                torch.from_numpy(np.array(reference["p0_l0_input"], copy=True)),
                torch.from_numpy(np.array(reference["p0_positions"], copy=True)),
                _torch_parameters(
                    loaded.materialized, loaded.package["parameter_orders"]["layer"]
                ),
            ),
            [
                "p0_l0_input",
                "p0_positions",
                *loaded.package["parameter_orders"]["layer"],
            ],
            [
                *(
                    f"probe_prefill_l0_{name}"
                    for name in layer_checkpoint_names(config)
                ),
                *(f"probe_prefill_l0_state_{index}" for index in range(7)),
            ],
        ),
        "kv_cache_prefill": (
            KVCachePrefillProbe(layer, config, loaded.linear_compute),
            (
                torch.from_numpy(
                    np.array(reference["focused_linear_q_proj_input"], copy=True)
                ),
                torch.from_numpy(np.array(reference["p0_positions"], copy=True)),
                _torch_parameters(loaded.materialized, kv_parameter_names),
            ),
            [
                "focused_linear_q_proj_input",
                "p0_positions",
                *kv_parameter_names,
            ],
            [f"probe_prefill_l0_state_{index}" for index in range(6)],
        ),
        "attention": (
            AttentionProbe(layer),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in attention_inputs
            ),
            attention_inputs,
            [
                "probe_prefill_l0_attention_scores",
                "probe_prefill_l0_attention_masked_scores",
                "probe_prefill_l0_attention_probabilities",
                "probe_prefill_l0_attention_context",
            ],
        ),
        "attention_final": (
            AttentionFinalProbe(layer),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in attention_inputs
            ),
            attention_inputs,
            ["probe_prefill_l0_attention_context"],
        ),
        "softmax": (
            SoftmaxProbe(config.head_dim),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in [
                    "probe_prefill_l0_attention_scores",
                    "p0_positions",
                    "probe_prefill_l0_state_6",
                ]
            ),
            [
                "probe_prefill_l0_attention_scores",
                "p0_positions",
                "probe_prefill_l0_state_6",
            ],
            [
                "probe_prefill_l0_attention_masked_scores",
                "probe_prefill_l0_attention_probabilities",
            ],
        ),
        "dequant_qk": (
            AttentionSubstageProbe(layer, "dequant_qk"),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in attention_inputs
            ),
            attention_inputs,
            ["probe_prefill_l0_attention_scores"],
        ),
        "qk_softmax": (
            AttentionSubstageProbe(layer, "qk_softmax"),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in attention_inputs
            ),
            attention_inputs,
            ["probe_prefill_l0_attention_probabilities"],
        ),
        "softmax_pv": (
            AttentionSubstageProbe(layer, "softmax_pv"),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in softmax_pv_inputs
            ),
            softmax_pv_inputs,
            ["probe_prefill_l0_attention_context"],
        ),
    }
    decode_cache_inputs = [f"p0_l0_o{index}" for index in range(1, 8)]
    builds["layer_checkpoints_decode"] = (
        LayerCheckpointsProbe(checkpoint_decode_layer),
        (
            torch.from_numpy(np.array(reference["p1_l0_input"], copy=True)),
            torch.from_numpy(np.array(reference["p1_positions"], copy=True)),
            _torch_parameters(
                loaded.materialized, loaded.package["parameter_orders"]["layer"]
            ),
            *(
                torch.from_numpy(
                    np.array(reference[name], copy=True).reshape(())
                    if name.endswith("_o7")
                    else np.array(reference[name], copy=True)[0]
                )
                for name in decode_cache_inputs
            ),
        ),
        [
            "p1_l0_input",
            "p1_positions",
            *loaded.package["parameter_orders"]["layer"],
            *decode_cache_inputs,
        ],
        [
            *(
                f"probe_decode_l0_{name}"
                for name in layer_checkpoint_names(decode_config)
            ),
            *(f"probe_decode_l0_state_{index}" for index in range(7)),
        ],
    )
    if loaded.attention_compute == "fp16":
        dequant_input_names = cache_inputs[:3]
        builds["dequant_key_raw"] = (
            DequantKeyProbe(config, transpose=False),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in dequant_input_names
            ),
            dequant_input_names,
            ["focused_key_dequant"],
        )
        builds["dequant_key"] = (
            DequantKeyProbe(config, transpose=True),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in dequant_input_names
            ),
            dequant_input_names,
            ["focused_qk_rhs"],
        )
        builds["transpose_key"] = (
            TransposeKeyProbe(),
            (torch.from_numpy(np.array(reference["focused_key_dequant"], copy=True)),),
            ["focused_key_dequant"],
            ["focused_qk_rhs"],
        )
        matrix_input_names = [
            "focused_key_payload_matrix",
            "focused_key_scale_matrix",
            "focused_key_zero_matrix",
        ]
        builds["dequant_key_matrix"] = (
            DequantKeyMatrixProbe(
                config.batch_size * config.num_key_value_heads * config.cache_capacity,
                config.head_dim,
                config.kv_group_size,
            ),
            tuple(
                torch.from_numpy(np.array(reference[name], copy=True))
                for name in matrix_input_names
            ),
            matrix_input_names,
            ["focused_key_dequant_matrix"],
        )
        for row_count in (8, 16, 32):
            prefix_input_names = [f"{name}_r{row_count}" for name in matrix_input_names]
            builds[f"dequant_key_matrix_r{row_count}"] = (
                DequantKeyMatrixProbe(
                    row_count,
                    config.head_dim,
                    config.kv_group_size,
                ),
                tuple(
                    torch.from_numpy(np.array(reference[name], copy=True))
                    for name in prefix_input_names
                ),
                prefix_input_names,
                [f"focused_key_dequant_matrix_r{row_count}"],
            )
    if args.probes:
        unknown = sorted(set(args.probes) - set(builds))
        if unknown:
            raise ValueError(f"unknown focused stage probes: {unknown}")
        builds = {name: builds[name] for name in args.probes}
    target = tvm.target.Target(loaded.package["profile"]["target"], host="llvm")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, (model, inputs, input_names, expected_names) in builds.items():
        executable, inventory = _build(
            model, inputs, target, loaded.package["backend_policy"]
        )
        path = args.output_dir / f"{name}.so"
        executable.export_library(str(path))
        tvm.runtime.load_module(str(path))
        artifacts[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
            "input_names": input_names,
            "expected_names": expected_names,
            "kernel_inventory": inventory,
        }
    package = {
        "schema_version": 1,
        "format": "vortex-llama3-c1-c3-focused-stage-probes",
        "source_package": package_identity(loaded.package),
        "source_package_path": str(loaded.path),
        "source_package_sha256": sha256_file(loaded.path),
        "reference_path": str(Path(args.reference).resolve()),
        "reference_sha256": reference_metadata["npz_sha256"],
        "alias": args.alias,
        "shape_case": args.case,
        "backend_policy": loaded.package["backend_policy"],
        "artifacts": artifacts,
    }
    package_path = args.output_dir / "package.json"
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference.close()
    print(package_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
