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

import os
import sys
from pathlib import Path

import pytest
import torch

import tvm
from tvm.relax.backend.vortex import C3_ALL_W4_NAIVE, get_default_pipeline
from tvm.relax.backend.vortex.pipeline import (
    _tcu_tensorize_pass,
    _w4a16_lowering_pass,
)
from tvm.relax.frontend.torch import from_exported_program

VORTEX_HOME = Path(os.environ.get("TVM_VORTEX_HOME", str(Path(__file__).resolve().parents[4] / "vortex_fpint-feat-gemv")))
sys.path.insert(0, str(VORTEX_HOME / "pytorch/spinquant"))

from spinquant_inference.llama3_c4_export import (  # noqa: E402
    Llama3ExportConfig,
    Llama3LayerDecode,
    Llama3LayerPrefill,
    parameter_shapes_for_compute,
)


def _config(query_length, head_dim=32, cache_capacity=8):
    return Llama3ExportConfig(
        batch_size=2,
        query_length=query_length,
        cache_capacity=cache_capacity,
        hidden_size=4 * head_dim,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        weight_group_size=32,
        kv_group_size=head_dim,
        vocabulary_size=128,
    )


def _parameters(config, linear_compute):
    return {
        name: torch.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in parameter_shapes_for_compute(
            config, linear_compute
        ).items()
    }


def _import_layer(phase, linear_compute, attention_compute, head_dim=32, cache_capacity=8):
    query_length = 7 if phase == "prefill" else 1
    config = _config(query_length, head_dim=head_dim, cache_capacity=cache_capacity)
    hidden = torch.zeros((2, query_length, config.hidden_size), dtype=torch.float16)
    positions = torch.arange(query_length, dtype=torch.int64).repeat(2, 1)
    parameters = _parameters(config, linear_compute)
    if phase == "prefill":
        model = Llama3LayerPrefill(
            config,
            linear_compute=linear_compute,
            attention_compute=attention_compute,
        )
        inputs = (hidden, positions, parameters)
    else:
        model = Llama3LayerDecode(
            config,
            linear_compute=linear_compute,
            attention_compute=attention_compute,
        )
        payload = torch.zeros((2, 2, cache_capacity, (head_dim + 1) // 2), dtype=torch.uint8)
        scale = torch.zeros((2, 2, cache_capacity, 1), dtype=torch.float16)
        zero = torch.zeros_like(scale, dtype=torch.int16)
        inputs = (
            hidden,
            positions,
            parameters,
            payload,
            scale,
            zero,
            payload.clone(),
            scale.clone(),
            zero.clone(),
            torch.zeros((), dtype=torch.int64),
        )
    exported = torch.export.export(model, inputs, strict=True)
    return from_exported_program(
        exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True
    )


def _target(*, tcu=False, gemm="none"):
    attrs = {"kind": "vortex", "vortex_gemm_mode": gemm}
    if tcu:
        attrs.update(vortex_tcu_mode="fp", vortex_tcu_fp_formats="fp16")
    return tvm.target.Target(attrs)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    ("name", "linear", "attention", "logical_fp16", "logical_w4"),
    [
        ("C1", "fp16", "fp16", 9, 0),
        ("C2-fixture", "w4", "fp16", 2, 7),
        ("C3", "w4", "w4", 0, 9),
    ],
)
def test_backend_neutral_graph_carries_explicit_compute_roles(
    phase, name, linear, attention, logical_fp16, logical_w4
):
    del name
    script = _import_layer(phase, linear, attention).script()

    assert script.count('R.call_pure_packed("relax.vortex.fp16_matmul"') == logical_fp16
    assert script.count('R.call_pure_packed("relax.vortex.mm_w4a16"') == logical_w4
    if attention == "fp16":
        assert script.count('R.str("attention.qk")') == 1
        assert script.count('R.str("attention.pv")') == 1
        assert script.count('R.call_pure_packed("relax.vortex.dequantize_int4"') == 2


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    ("name", "linear", "attention", "target", "tcu_matmuls", "naive_lowered"),
    [
        ("C1", "fp16", "fp16", _target(tcu=True), 9, 0),
        (
            "C2-synthetic-capability-fixture",
            "w4",
            "fp16",
            _target(tcu=True, gemm="naive"),
            2,
            7,
        ),
        ("C3", "w4", "w4", _target(gemm="naive"), 0, 23),
    ],
)
def test_role_routing_consumes_every_logical_gemm(
    phase, name, linear, attention, target, tcu_matmuls, naive_lowered
):
    del name
    mod = _import_layer(phase, linear, attention)
    mod = _tcu_tensorize_pass(target, require_all=tcu_matmuls > 0)(mod)
    mod = _w4a16_lowering_pass(target)(mod)
    script = mod.script()

    assert script.count('R.call_pure_packed("relax.vortex.fp16_matmul"') == 0
    assert script.count('R.call_pure_packed("relax.vortex.mm_w4a16"') == 0
    assert "vortex_mm_w4a16_improve" not in script
    assert "vortex.c4.layout_policy" not in mod.attrs
    assert int(mod.attrs.get("vortex.tcu.fp16.lowered_matmuls", 0)) == tcu_matmuls
    assert int(mod.attrs.get("vortex.w4a16.lowered", 0)) == naive_lowered
    if tcu_matmuls:
        assert "vx_tvm_tcu_fp16_tile" in script
        assert script.count("def vortex_dequantize_int4_rank5") == 1
        assert script.count("R.call_tir(cls.vortex_dequantize_int4_rank5") == 2
        assert "dequantized_matrix" not in script
    if naive_lowered:
        assert "vx_tvm_gemm_w4a16" in script
        assert str(mod.attrs["vortex.w4a16.physical_layout"]) == "row_major"


def test_c2_fixture_routes_by_role_not_by_matrix_shape():
    mod = _import_layer("prefill", "w4", "fp16")
    target = _target(tcu=True, gemm="naive")
    mod = _tcu_tensorize_pass(target, require_all=True)(mod)
    mod = _w4a16_lowering_pass(target)(mod)

    assert int(mod.attrs["vortex.tcu.fp16.role.attention_qk.matmuls"]) == 1
    assert int(mod.attrs["vortex.tcu.fp16.role.attention_pv.matmuls"]) == 1
    assert int(mod.attrs["vortex.w4a16.lowered"]) == 7


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_c3_default_pipeline_pads_rank5_gqa_tails(phase):
    mod = _import_layer(phase, "w4", "w4")
    target = _target(gemm="naive")

    lowered = get_default_pipeline(target, backend_policy=C3_ALL_W4_NAIVE)(mod)
    script = lowered.script()

    assert 'R.call_pure_packed("relax.vortex.mm_w4a16"' not in script
    assert "R.strided_slice" not in script
    assert "vx_tvm_gemm_w4a16" in script
    assert int(lowered.attrs.get("vortex.w4a16.fused_kv_expands", 0)) == 0
    assert "def vortex_mm_w4a16_naive_batched" not in script
    assert "def vortex_naive_pad_float16" in script
    assert "def vortex_naive_slice_output" in script
    assert "def vortex_gqa_context_reshape" not in script


@pytest.mark.parametrize("width", [16, 32])
def test_c3_production_head_dim_tiles_quant_direction_n_to_mxu_columns(width):
    mod = _import_layer("decode", "w4", "w4", head_dim=128, cache_capacity=128)
    target = tvm.target.Target({
        "kind": "vortex", "vortex_gemm_mode": "naive", "thread_warp_size": width,
        "vortex_mxu_row": width, "vortex_mxu_col": width,
    })

    lowered = get_default_pipeline(target, backend_policy=C3_ALL_W4_NAIVE)(mod)
    script = lowered.script()

    assert int(lowered.attrs["vortex.w4a16.naive_n_tiled_calls"]) == 1
    assert script.count("def vortex_mm_w4a16_naive_batched_n_tiled") == 1
    assert script.count("def vortex_naive_n_tile_packed") == 1
    assert script.count("def vortex_naive_n_tile_scale") == 1
    assert script.count("def vortex_naive_n_tile_zero") == 1
    assert "def vortex_gqa_context_reshape" not in script
