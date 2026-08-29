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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import tvm
from tvm import relax
from tvm.relax.backend.vortex.layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)
from tvm.relax.backend.vortex.pipeline import (
    _make_w4a16_improve,
    _w4a16_lowering_pass,
)
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile

VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
OPS_PATH = VORTEX_HOME / "pytorch/spinquant/spinquant_inference/vortex_export_ops.py"
sys.path.insert(0, str(VORTEX_HOME / "pytorch/spinquant"))

from spinquant_inference.llama3_c4_export import (  # noqa: E402
    Llama3ExportConfig,
    Llama3LayerDecode,
    Llama3LayerPrefill,
    Llama3StackDecode,
    Llama3StackPrefill,
    make_meta_parameters,
    stack_parameter_shapes,
)
NAIVE_XCLBIN = Path(
    "/opt/vortex_fpga_bins/fpint/"
    "xrt_hw_u55c_c_f100_fpint_9600db3a37/bin/vortex_afu.xclbin"
)
IMPROVED_XCLBIN = Path(
    "/opt/vortex_fpga_bins/fpint/"
    "xrt_hw_u55c_c_f100_fpint_64300e5119/bin/vortex_afu.xclbin"
)


def _register_logical_ops():
    if hasattr(torch.ops.vortex, "mm_w4a16"):
        return
    spec = importlib.util.spec_from_file_location("vortex_export_ops_for_tvm", OPS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


class _Attention(torch.nn.Module):
    def forward(self, query, key):
        packed, scale, zero = torch.ops.vortex.quantize_int4(
            key, 1, 4, 1, "signed_asymmetric_int4"
        )
        return torch.ops.vortex.mm_w4a16(
            query,
            packed,
            scale,
            zero,
            [2, 4],
            4,
            1,
            1,
            "signed_asymmetric_int4",
            True,
        )


class _PackedW4A16(torch.nn.Module):
    def __init__(
        self, transpose_rhs=False, quant_axis=0, rhs_shape=None, group_size=32
    ):
        super().__init__()
        self.transpose_rhs = transpose_rhs
        self.quant_axis = quant_axis
        self.group_size = group_size
        self.rhs_shape = rhs_shape or (
            (32, 128) if self.transpose_rhs else (128, 32)
        )

    def forward(self, lhs, packed, scale, zero_point):
        return torch.ops.vortex.mm_w4a16(
            lhs,
            packed,
            scale,
            zero_point,
            list(self.rhs_shape),
            self.group_size,
            self.quant_axis,
            1,
            "signed_symmetric_int4",
            self.transpose_rhs,
        )


class _W4FFN(torch.nn.Module):
    def __init__(self, return_hidden=False):
        super().__init__()
        self.return_hidden = return_hidden

    def forward(
        self,
        lhs,
        packed1,
        scale1,
        zero1,
        packed2,
        scale2,
        zero2,
        bias,
        residual,
    ):
        hidden = torch.ops.vortex.mm_w4a16(
            lhs,
            packed1,
            scale1,
            zero1,
            [33, 31],
            32,
            0,
            1,
            "signed_symmetric_int4",
            False,
        )
        hidden = torch.relu(hidden + bias + residual)
        output = torch.ops.vortex.mm_w4a16(
            hidden,
            packed2,
            scale2,
            zero2,
            [31, 17],
            32,
            0,
            1,
            "signed_symmetric_int4",
            False,
        )
        return (hidden, output) if self.return_hidden else output


class _W4SharedInput(torch.nn.Module):
    def forward(self, lhs, packed1, scale1, zero1, packed2, scale2, zero2):
        first = torch.ops.vortex.mm_w4a16(
            lhs,
            packed1,
            scale1,
            zero1,
            [33, 31],
            32,
            0,
            1,
            "signed_symmetric_int4",
            False,
        )
        second = torch.ops.vortex.mm_w4a16(
            lhs.reshape(1, 7, 33).reshape(7, 33),
            packed2,
            scale2,
            zero2,
            [33, 17],
            32,
            0,
            1,
            "signed_symmetric_int4",
            False,
        )
        return first, second


class _ConstantPackedW4A16(torch.nn.Module):
    def __init__(self, packed, scale, zero, transpose_rhs=False, quant_axis=0):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.transpose_rhs = transpose_rhs
        self.quant_axis = quant_axis

    def forward(self, lhs):
        return torch.ops.vortex.mm_w4a16(
            lhs,
            self.packed,
            self.scale,
            self.zero,
            [31, 33] if self.transpose_rhs else [33, 31],
            32,
            self.quant_axis,
            1,
            "signed_symmetric_int4",
            self.transpose_rhs,
        )


class _QuantizedW4A16(torch.nn.Module):
    def __init__(self, transpose_rhs=False):
        super().__init__()
        self.transpose_rhs = transpose_rhs

    def forward(self, lhs, rhs):
        packed, scale, zero_point = torch.ops.vortex.quantize_int4(
            rhs, 1, 32, 1, "signed_asymmetric_int4"
        )
        return torch.ops.vortex.mm_w4a16(
            lhs,
            packed,
            scale,
            zero_point,
            list(rhs.shape),
            32,
            1,
            1,
            "signed_asymmetric_int4",
            self.transpose_rhs,
        )


class _QuantizeDequantize(torch.nn.Module):
    def __init__(self, scheme):
        super().__init__()
        self.scheme = scheme

    def forward(self, source):
        packed, scale, zero_point = torch.ops.vortex.quantize_int4(
            source, 1, 32, 1, self.scheme
        )
        return torch.ops.vortex.dequantize_int4(
            packed,
            scale,
            zero_point,
            list(source.shape),
            1,
            32,
            1,
            self.scheme,
        )


class _KVCacheUpdate(torch.nn.Module):
    def forward(
        self,
        cache_payload,
        cache_scale,
        cache_zero,
        payload,
        scale,
        zero,
    ):
        return torch.ops.vortex.kv_cache_update(
            cache_payload,
            cache_scale,
            cache_zero,
            payload,
            scale,
            zero,
            2,
            4,
        )


class _W4AttentionFragment(torch.nn.Module):
    def forward(self, query, key, value):
        key_packed, key_scale, key_zero = torch.ops.vortex.quantize_int4(
            key, 1, 32, 1, "signed_asymmetric_int4"
        )
        score = torch.ops.vortex.mm_w4a16(
            query,
            key_packed,
            key_scale,
            key_zero,
            list(key.shape),
            32,
            1,
            1,
            "signed_asymmetric_int4",
            True,
        )
        value_packed, value_scale, value_zero = torch.ops.vortex.quantize_int4(
            value, 1, 32, 1, "signed_asymmetric_int4"
        )
        context = torch.ops.vortex.mm_w4a16(
            score,
            value_packed,
            value_scale,
            value_zero,
            list(value.shape),
            32,
            1,
            1,
            "signed_asymmetric_int4",
            False,
        )
        return context


class _W4CachedDecode(torch.nn.Module):
    def forward(
        self,
        query,
        key_cache_payload,
        key_cache_scale,
        key_cache_zero,
        value_cache_payload,
        value_cache_scale,
        value_cache_zero,
        key_update,
        value_update,
    ):
        key_packed, key_scale, key_zero = torch.ops.vortex.quantize_int4(
            key_update, 1, 32, 1, "signed_asymmetric_int4"
        )
        key_cache = torch.ops.vortex.kv_cache_update(
            key_cache_payload,
            key_cache_scale,
            key_cache_zero,
            key_packed,
            key_scale,
            key_zero,
            7,
            129,
        )
        score = torch.ops.vortex.mm_w4a16(
            query,
            key_cache[0],
            key_cache[1],
            key_cache[2],
            [129, 33],
            32,
            1,
            1,
            "signed_asymmetric_int4",
            True,
        )
        value_packed, value_scale, value_zero = torch.ops.vortex.quantize_int4(
            value_update, 1, 32, 1, "signed_asymmetric_int4"
        )
        value_cache = torch.ops.vortex.kv_cache_update(
            value_cache_payload,
            value_cache_scale,
            value_cache_zero,
            value_packed,
            value_scale,
            value_zero,
            7,
            129,
        )
        context = torch.ops.vortex.mm_w4a16(
            score,
            value_cache[0],
            value_cache[1],
            value_cache[2],
            [129, 17],
            32,
            1,
            1,
            "signed_asymmetric_int4",
            False,
        )
        return context, *key_cache, *value_cache


def test_import_backend_neutral_llama3_prefill_and_decode_graphs():
    config = Llama3ExportConfig(
        batch_size=1,
        query_length=1,
        cache_capacity=32,
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        weight_group_size=32,
        kv_group_size=32,
    )
    parameters = {
        name: torch.empty_like(value, device="cpu")
        for name, value in make_meta_parameters(config).items()
    }
    hidden = torch.empty((1, 1, 128), dtype=torch.float16)
    positions = torch.empty((1, 1), dtype=torch.int64)
    prefill_program = torch.export.export(
        Llama3LayerPrefill(config),
        (hidden, positions, parameters),
        strict=True,
    )
    prefill = from_exported_program(
        prefill_program,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    prefill_script = prefill.script()
    assert prefill_script.count('R.call_pure_packed("relax.vortex.mm_w4a16"') == 9
    assert prefill_script.count('R.call_pure_packed("relax.vortex.quantize_int4"') == 2
    assert prefill_script.count('R.call_pure_packed("relax.vortex.kv_cache_update"') == 2
    assert "tile_input_a" not in prefill_script
    assert "mm_w4a16_gemm_core" not in prefill_script

    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    prefill_alone = _w4a16_lowering_pass(target, layout_policy="alone")(prefill)
    prefill_alone_script = prefill_alone.script()
    assert prefill_alone.attrs["vortex.c4.layout_policy"] == "alone"
    assert prefill_alone.attrs["vortex.w4a16.lowered"] == 15
    assert "vortex_kv_cache_update" in prefill_alone_script
    assert "batched_lhs_slice" in prefill_alone_script
    assert "vortex_batched_output_barrier" in prefill_alone_script
    assert 'R.call_pure_packed("relax.vortex.mm_w4a16"' not in prefill_alone_script
    prefill_fused = _w4a16_lowering_pass(target, layout_policy="fused")(prefill)
    assert prefill_fused.attrs["vortex.improve.reused_a_layouts"] == 3
    assert prefill_alone_script.count("R.call_tir(cls.vortex_gemm_a_tiled") == 15
    assert prefill_fused.script().count("R.call_tir(cls.vortex_gemm_a_tiled") == 12

    packed = torch.empty((1, 2, 32, 16), dtype=torch.uint8)
    scale = torch.empty((1, 2, 32, 1), dtype=torch.float16)
    zero = torch.empty((1, 2, 32, 1), dtype=torch.int16)
    length = torch.empty((), dtype=torch.int64)
    decode_program = torch.export.export(
        Llama3LayerDecode(config),
        (
            hidden,
            positions,
            parameters,
            packed,
            scale,
            zero,
            packed,
            scale,
            zero,
            length,
        ),
        strict=True,
    )
    decode = from_exported_program(
        decode_program,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    decode_script = decode.script()
    assert "@R.function(pure=False)" in decode_script
    assert "R.assert_op" in decode_script
    assert "allocated KV cache capacity" in decode_script
    assert decode_script.index("R.assert_op") < decode_script.index("with R.dataflow()")
    assert decode_script.count('R.call_pure_packed("relax.vortex.mm_w4a16"') == 9
    assert decode_script.count(
        'R.call_pure_packed("relax.vortex.kv_cache_update_dynamic"'
    ) == 2
    decode_alone = _w4a16_lowering_pass(target, layout_policy="alone")(decode)
    decode_alone_script = decode_alone.script()
    assert decode_alone.attrs["vortex.c4.layout_policy"] == "alone"
    assert decode_alone.attrs["vortex.w4a16.lowered"] == 15
    assert "vortex_kv_cache_update_dynamic" in decode_alone_script
    assert (
        'R.call_pure_packed("relax.vortex.kv_cache_update_dynamic"'
        not in decode_alone_script
    )
    decode_fused = _w4a16_lowering_pass(target, layout_policy="fused")(decode)
    assert decode_fused.attrs["vortex.improve.reused_a_layouts"] == 3
    assert decode_alone_script.count("R.call_tir(cls.vortex_gemm_a_tiled") == 15
    assert decode_fused.script().count("R.call_tir(cls.vortex_gemm_a_tiled") == 12
    decode_inplace = _w4a16_lowering_pass(
        target,
        layout_policy="alone",
        inplace_kv_cache=True,
    )(decode)
    assert decode_inplace.attrs["vortex.kv_cache_update_inplace"] == 1
    assert decode_inplace.script().count("R.call_tir_inplace") == 2
    assert "tile_input_a" not in decode_script
    assert "mm_w4a16_gemm_core" not in decode_script


def _import_packed_w4a16(
    transpose_rhs=False, quant_axis=0, m=8, n=32, k=128, group_size=32
):
    _register_logical_ops()
    rhs_shape = (n, k) if transpose_rhs else (k, n)
    packed_shape = (rhs_shape[0], (rhs_shape[1] + 1) // 2)
    qparam_shape = list(rhs_shape)
    qparam_shape[quant_axis] = (
        qparam_shape[quant_axis] + group_size - 1
    ) // group_size
    inputs = (
        torch.ones((m, k), dtype=torch.float16),
        torch.zeros(packed_shape, dtype=torch.uint8),
        torch.ones(qparam_shape, dtype=torch.float16),
        torch.zeros(qparam_shape, dtype=torch.int16),
    )
    return from_exported_program(
        torch.export.export(
            _PackedW4A16(
                transpose_rhs, quant_axis, rhs_shape, group_size=group_size
            ),
            inputs,
        ),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )


def _ffn_inputs(seed=71):
    rng = np.random.default_rng(seed)
    return (
        torch.from_numpy(rng.uniform(-0.1, 0.1, (7, 33)).astype("float16")),
        torch.from_numpy(rng.integers(0, 256, (33, 16), dtype="uint8")),
        torch.from_numpy(rng.uniform(0.01, 0.04, (2, 31)).astype("float16")),
        torch.zeros((2, 31), dtype=torch.int16),
        torch.from_numpy(rng.integers(0, 256, (31, 9), dtype="uint8")),
        torch.from_numpy(rng.uniform(0.01, 0.04, (1, 17)).astype("float16")),
        torch.zeros((1, 17), dtype=torch.int16),
        torch.from_numpy(rng.uniform(-0.05, 0.05, (31,)).astype("float16")),
        torch.from_numpy(rng.uniform(-0.05, 0.05, (7, 31)).astype("float16")),
    )
def _pack_improve_physical(lhs, weight, scale, zero_point, plan):
    """Independently pack logical QCOL/WTRANS=0 tensors for a direct ABI test."""

    assert not plan.weight_transpose and plan.quant_direction == 0
    profile = plan.profile
    tiled_a = np.zeros(plan.a_elements, dtype="float16")
    a_base = 0
    for mt, cur_m in enumerate(plan.m_tiles):
        slot_m = (cur_m + profile.num_dma_channels - 1) // profile.num_dma_channels
        slot_m *= profile.num_dma_channels
        for kt, cur_k in enumerate(plan.k_tiles):
            kt_base = a_base + kt * slot_m * profile.dma_kt
            for micro_k in range(cur_k // profile.mxu_kt):
                for local_m in range(cur_m):
                    for inner_k in range(profile.mxu_kt):
                        logical_k = (
                            kt * profile.dma_kt
                            + micro_k * profile.mxu_kt
                            + inner_k
                        )
                        if logical_k < plan.logical_k:
                            index = kt_base + micro_k * cur_m * profile.mxu_kt
                            index += local_m * profile.mxu_kt + inner_k
                            tiled_a[index] = lhs[
                                mt * profile.dma_mt + local_m, logical_k
                            ]
        a_base += slot_m * plan.execution_k

    tiled_w = np.zeros(plan.weight_bytes, dtype="uint8")
    for logical_k in range(plan.logical_k):
        kt, local_k = divmod(logical_k, profile.dma_kt)
        for logical_n in range(plan.logical_n):
            nt, inner_n = divmod(logical_n, profile.mxu_nt)
            index = kt * profile.dma_kt * plan.execution_n // 2
            cur_k = min(profile.dma_kt, plan.execution_k - kt * profile.dma_kt)
            index += nt * cur_k * profile.mxu_nt // 2
            index += local_k * (profile.mxu_nt // 2) + inner_n // 2
            nibble = np.uint8(int(weight[logical_k, logical_n]) & 0xF)
            if inner_n & 1:
                tiled_w[index] |= np.uint8(nibble << np.uint8(4))
            else:
                tiled_w[index] |= nibble

    tiled_scale = np.zeros(plan.qparam_elements, dtype="float16")
    tiled_zero = np.zeros(plan.qparam_elements, dtype="int16")
    for slot in plan.qparam_slots:
        offset = slot.offset_bytes // 2
        groups = slot.execution_k // plan.qblock
        for group in range(groups):
            global_group = slot.outer_k * (profile.dma_kt // plan.qblock) + group
            for local_n in range(slot.execution_n):
                global_n = slot.outer_n * profile.dma_nt + local_n
                if global_group < scale.shape[0] and global_n < plan.logical_n:
                    micro_n, inner_n = divmod(local_n, profile.mxu_nt)
                    index = offset + micro_n * groups * profile.mxu_nt
                    index += group * profile.mxu_nt + inner_n
                    tiled_scale[index] = scale[global_group, global_n]
                    tiled_zero[index] = zero_point[global_group, global_n]
    return tiled_a, tiled_w, tiled_scale, tiled_zero


def _detile_improve_output(tiled, plan):
    output = np.empty((plan.logical_m, plan.logical_n), dtype="float16")
    c_base = 0
    for mt, cur_m in enumerate(plan.m_tiles):
        slot_m = (cur_m + plan.profile.num_dma_channels - 1)
        slot_m //= plan.profile.num_dma_channels
        slot_m *= plan.profile.num_dma_channels
        for local_m in range(cur_m):
            for logical_n in range(plan.logical_n):
                nt, inner_n = divmod(logical_n, plan.profile.mxu_nt)
                index = c_base + nt * slot_m * plan.profile.mxu_nt
                index += local_m * plan.profile.mxu_nt + inner_n
                output[mt * plan.profile.dma_mt + local_m, logical_n] = tiled[index]
        c_base += slot_m * plan.execution_n
        return output


class _PrepackedW4(torch.nn.Module):
    def forward(self, lhs, weight, scale, zero):
        return (
            torch.ops.vortex.mm_w4a16_prepacked(
                lhs,
                weight,
                scale,
                zero,
                [33, 31],
                32,
                0,
                1,
                "signed_asymmetric_int4",
                False,
            ),
        )


def test_prepacked_w4a16_skips_runtime_weight_layout_kernels():
    _register_logical_ops()
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    plan = plan_improve_layout(7, 31, 33, 32)
    inputs = (
        torch.empty((7, 33), dtype=torch.float16),
        torch.empty((plan.weight_bytes,), dtype=torch.uint8),
        torch.empty((plan.qparam_elements,), dtype=torch.float16),
        torch.empty((plan.qparam_elements,), dtype=torch.int16),
    )
    mod = from_exported_program(
        torch.export.export(_PrepackedW4(), inputs, strict=True),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    assert "relax.vortex.mm_w4a16_prepacked" in mod.script()
    lowered = _w4a16_lowering_pass(target, layout_policy="alone")(mod)
    script = lowered.script()
    assert "vortex_mm_w4a16_improve" in script
    assert "vortex_gemm_a_tiled" in script
    assert "vortex_gemm_w_tiled" not in script
    assert "vortex_gemm_scale_tiled" not in script
    assert "vortex_gemm_zero_point_tiled" not in script


def test_two_layer_llama_stack_imports_external_prepacked_parameters():
    config = Llama3ExportConfig(
        batch_size=1,
        query_length=1,
        cache_capacity=8,
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        weight_group_size=32,
        kv_group_size=32,
    )
    canonical_shapes = stack_parameter_shapes(config, 2)
    parameters = {}
    plans = {}
    for name, (shape, dtype) in canonical_shapes.items():
        if name.endswith("norm.weight"):
            parameters[name] = torch.ones(shape, dtype=dtype)
            continue
        projection = name.rsplit(".", 1)[0]
        if projection not in plans:
            qweight_shape = canonical_shapes[f"{projection}.qweight"][0]
            plans[projection] = plan_improve_layout(
                1, qweight_shape[1] * 2, qweight_shape[0], 32
            )
        plan = plans[projection]
        if name.endswith(".qweight"):
            parameters[name] = torch.empty((plan.weight_bytes,), dtype=dtype)
        else:
            parameters[name] = torch.empty((plan.qparam_elements,), dtype=dtype)

    model = Llama3StackPrefill(config, 2, prepacked_weights=True)
    hidden = torch.zeros((1, 1, 128), dtype=torch.float16)
    positions = torch.zeros((1, 1), dtype=torch.int64)
    exported = torch.export.export(
        model, (hidden, positions, parameters), strict=True
    )
    mod = from_exported_program(
        exported,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    assert mod.script().count("relax.vortex.mm_w4a16_prepacked") == 14
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    lowered = _w4a16_lowering_pass(target, layout_policy="fused")(mod)
    assert lowered.attrs["vortex.improve.external_prepacked_w4a16"] == 14
    assert "relax.vortex.mm_w4a16_prepacked" not in lowered.script()

    packed = torch.zeros((2, 1, 2, 8, 16), dtype=torch.uint8)
    scale = torch.zeros((2, 1, 2, 8, 1), dtype=torch.float16)
    zero = torch.zeros((2, 1, 2, 8, 1), dtype=torch.int16)
    lengths = torch.zeros((2,), dtype=torch.int64)
    decode = Llama3StackDecode(config, 2, prepacked_weights=True)
    decode_exported = torch.export.export(
        decode,
        (
            hidden,
            positions,
            parameters,
            packed,
            scale,
            zero,
            packed,
            scale,
            zero,
            lengths,
        ),
        strict=True,
    )
    decode_mod = from_exported_program(
        decode_exported,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    decode_lowered = _w4a16_lowering_pass(
        target,
        layout_policy="fused",
        inplace_kv_cache=True,
    )(decode_mod)
    assert decode_lowered.attrs["vortex.improve.external_prepacked_w4a16"] == 14
    assert decode_lowered.attrs["vortex.kv_cache_update_inplace"] == 1
    tvm.relax.analysis.well_formed(decode_lowered)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 in the improved GEMM XRT environment",
)
def test_external_prepacked_w4a16_hardware():
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    plan = plan_improve_layout(
        7, 31, 33, 32, profile=ImproveProfile.from_target(target)
    )
    rng = np.random.default_rng(20260830)
    lhs = rng.uniform(-0.1, 0.1, (7, 33)).astype("float16")
    weight = rng.integers(0, 256, (33, 16), dtype="uint8")
    scale = rng.uniform(0.01, 0.04, (2, 31)).astype("float16")
    zero = rng.integers(-2, 3, (2, 31), dtype="int16")
    physical = (
        prepack_improve_weight(weight, plan),
        prepack_improve_qparam(scale, plan, "float16"),
        prepack_improve_qparam(zero, plan, "int16"),
    )
    inputs = (
        torch.from_numpy(lhs),
        *(torch.from_numpy(value) for value in physical),
    )
    mod = from_exported_program(
        torch.export.export(_PrepackedW4(), inputs, strict=True),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(mod, target, exec_mode="bytecode")
    expected = torch.ops.vortex.mm_w4a16(
        torch.from_numpy(lhs),
        torch.from_numpy(weight),
        torch.from_numpy(scale),
        torch.from_numpy(zero),
        [33, 31],
        32,
        0,
        1,
        "signed_asymmetric_int4",
        False,
    ).numpy()
    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *(tvm.runtime.tensor(value.numpy(), device=device) for value in inputs)
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


def _pack_int4_last_axis(source):
    padded_shape = list(source.shape)
    padded_shape[-1] = (padded_shape[-1] + 1) // 2 * 2
    padded = np.zeros(padded_shape, dtype="int8")
    padded[..., : source.shape[-1]] = source
    return np.bitwise_or(
        np.bitwise_and(padded[..., 0::2], 0xF),
        np.left_shift(np.bitwise_and(padded[..., 1::2], 0xF), 4),
    ).astype("uint8")


def test_vortex_logical_int4_ops_import_one_to_one():
    _register_logical_ops()
    query = torch.ones((1, 4), dtype=torch.float16)
    key = torch.ones((2, 4), dtype=torch.float16)
    exported = torch.export.export(_Attention(), (query, key))
    mod = from_exported_program(exported, run_ep_decomposition=False)
    script = mod.script(show_meta=True)

    assert script.count('R.call_pure_packed("relax.vortex.quantize_int4"') == 1
    assert script.count('R.call_pure_packed("relax.vortex.mm_w4a16"') == 1
    assert "mm_w4a16_naive" not in script
    assert "mm_w4a16_improve" not in script
    assert "transpose" not in script

    naive_target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "naive"}, host="llvm"
    )
    lowered = _w4a16_lowering_pass(naive_target)(mod)
    lowered_script = lowered.script()
    assert "vortex_mm_w4a16_naive" in lowered_script
    assert "vx_tvm_gemm_w4a16" in lowered_script
    assert 'R.call_pure_packed("relax.vortex.mm_w4a16"' not in lowered_script
    assert "transpose" not in lowered_script


def test_naive_target_compiles_logical_w4a16_to_gemm_job_source():
    mod = _import_packed_w4a16()
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "naive"}, host="llvm"
    )
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, unused_target):
        del unused_target
        captured.append(source)
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        relax.build(mod, target, exec_mode="bytecode")
    finally:
        tvm.register_global_func(callback_name, previous, override=True)
    assert len(captured) == 1
    assert "#include <vx_tvm_gemm.h>" in captured[0]
    assert "vx_tvm_gemm_w4a16" in captured[0]
    assert "vortex_mm_w4a16_naive" in captured[0]


def test_improved_target_inserts_named_hierarchical_layout_pipeline():
    mod = _import_packed_w4a16()
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    lowered = _w4a16_lowering_pass(target)(mod)
    script = lowered.script()
    for name in (
        "vortex_gemm_a_tiled",
        "vortex_gemm_w_tiled",
        "vortex_gemm_scale_tiled",
        "vortex_gemm_zero_point_tiled",
        "vortex_mm_w4a16_improve",
        "vortex_gemm_c_detile",
    ):
        assert name in script
    assert 'R.call_pure_packed("relax.vortex.mm_w4a16"' not in script


def test_prelegalization_layout_region_fuses_ffn_vector_chain_and_branches():
    _register_logical_ops()
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    inputs = _ffn_inputs()
    mod = from_exported_program(
        torch.export.export(_W4FFN(), inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    fused = relax.transform.DeadCodeElimination()(_w4a16_lowering_pass(target)(mod))
    unfused = relax.transform.DeadCodeElimination()(
        _w4a16_lowering_pass(target, enable_layout_fusion=False)(mod)
    )
    fused_script = fused.script()
    unfused_script = unfused.script()
    assert fused_script.count("R.call_tir(cls.vortex_gemm_a_tiled") == 1
    assert fused_script.count("R.call_tir(cls.vortex_gemm_c_detile") == 1
    assert fused_script.count("R.call_tir(cls.vortex_gemm_tiled_add") == 2
    assert fused_script.count("R.call_tir(cls.vortex_gemm_tiled_relu") == 1
    assert unfused_script.count("R.call_tir(cls.vortex_gemm_a_tiled") == 2
    assert unfused_script.count("R.call_tir(cls.vortex_gemm_c_detile") == 2
    assert "vortex_gemm_tiled_add" not in unfused_script
    assert "vortex_gemm_tiled_relu" not in unfused_script

    branch_mod = from_exported_program(
        torch.export.export(_W4FFN(return_hidden=True), inputs),
        run_ep_decomposition=False,
    )
    branched = relax.transform.DeadCodeElimination()(
        _w4a16_lowering_pass(target)(branch_mod)
    ).script()
    assert branched.count("R.call_tir(cls.vortex_gemm_a_tiled") == 1
    assert branched.count("R.call_tir(cls.vortex_gemm_c_detile") == 2


def test_fused_policy_reuses_shared_gemm_a_layout_across_projection_siblings():
    _register_logical_ops()
    rng = np.random.default_rng(20260828)
    inputs = (
        torch.from_numpy(rng.uniform(-0.1, 0.1, (7, 33)).astype("float16")),
        torch.from_numpy(rng.integers(0, 256, (33, 16), dtype="uint8")),
        torch.ones((2, 31), dtype=torch.float16),
        torch.zeros((2, 31), dtype=torch.int16),
        torch.from_numpy(rng.integers(0, 256, (33, 9), dtype="uint8")),
        torch.ones((2, 17), dtype=torch.float16),
        torch.zeros((2, 17), dtype=torch.int16),
    )
    mod = from_exported_program(
        torch.export.export(_W4SharedInput(), inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    fused = relax.transform.DeadCodeElimination()(
        _w4a16_lowering_pass(target, layout_policy="fused")(mod)
    )
    alone = relax.transform.DeadCodeElimination()(
        _w4a16_lowering_pass(target, layout_policy="alone")(mod)
    )
    assert fused.attrs["vortex.improve.reused_a_layouts"] == 1
    assert fused.script().count("R.call_tir(cls.vortex_gemm_a_tiled") == 1
    assert alone.script().count("R.call_tir(cls.vortex_gemm_a_tiled") == 2


@pytest.mark.parametrize(
    ("transpose_rhs", "quant_axis"),
    [(False, 0), (False, 1), (True, 0), (True, 1)],
)
def test_constant_w4a16_parameters_are_prepacked_before_runtime_lowering(
    transpose_rhs, quant_axis
):
    _register_logical_ops()
    rng = np.random.default_rng(73 + int(transpose_rhs) * 2 + quant_axis)
    rhs_shape = (31, 33) if transpose_rhs else (33, 31)
    packed_shape = (rhs_shape[0], (rhs_shape[1] + 1) // 2)
    qparam_shape = list(rhs_shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    module = _ConstantPackedW4A16(
        torch.from_numpy(rng.integers(0, 256, packed_shape, dtype="uint8")),
        torch.from_numpy(rng.uniform(0.01, 0.04, qparam_shape).astype("float16")),
        torch.zeros(qparam_shape, dtype=torch.int16),
        transpose_rhs,
        quant_axis,
    )
    lhs = torch.ones((7, 33), dtype=torch.float16)
    mod = from_exported_program(
        torch.export.export(module, (lhs,)),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    lowered = _w4a16_lowering_pass(target)(mod)
    script = lowered.script()
    assert "vortex_gemm_w_tiled" not in script
    assert "vortex_gemm_scale_tiled" not in script
    assert "vortex_gemm_zero_point_tiled" not in script
    descriptors = tuple(lowered.attrs["vortex.improve.prepacked_constants"])
    qdir = int(quant_axis != (1 if transpose_rhs else 0))
    assert descriptors == (
        "M=7:N=31:K=33:Nexec=32:Kexec=64:QBLK=32:"
        f"WTRANS={int(transpose_rhs)}:QDIR={qdir}:ABI=2",
    )
    if not transpose_rhs and quant_axis == 0:
        pipelined = relax.backend.vortex.get_default_pipeline(target)(mod)
        assert tuple(
            pipelined.attrs["vortex.improve.prepacked_constants"]
        ) == descriptors


def test_quantize_and_dequantize_lower_to_vortex_tir():
    _register_logical_ops()
    inputs = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.ones((128, 32), dtype=torch.float16),
    )
    mod = from_exported_program(
        torch.export.export(_QuantizedW4A16(False), inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    lowered = _w4a16_lowering_pass(target)(mod)
    script = lowered.script()
    assert "vortex_quantize_int4_row_major" in script
    assert "vortex_mm_w4a16_improve" in script
    assert 'R.call_pure_packed("relax.vortex.quantize_int4"' not in script

    round_trip = from_exported_program(
        torch.export.export(
            _QuantizeDequantize("signed_asymmetric_int4"),
            (torch.ones((3, 65), dtype=torch.float16),),
        ),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    round_trip_script = _w4a16_lowering_pass(target)(round_trip).script()
    assert "vortex_quantize_int4_row_major" in round_trip_script
    assert "vortex_dequantize_int4_row_major" in round_trip_script
    assert 'R.call_pure_packed("relax.vortex.dequantize_int4"' not in round_trip_script


def test_kv_cache_update_and_attention_lower_from_logical_ops():
    _register_logical_ops()
    cache_inputs = (
        torch.zeros((1, 4, 64), dtype=torch.uint8),
        torch.zeros((1, 4, 4), dtype=torch.float16),
        torch.zeros((1, 4, 4), dtype=torch.int16),
        torch.ones((1, 1, 64), dtype=torch.uint8),
        torch.ones((1, 1, 4), dtype=torch.float16),
        torch.ones((1, 1, 4), dtype=torch.int16),
    )
    cache_mod = from_exported_program(
        torch.export.export(_KVCacheUpdate(), cache_inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    cache_script = _w4a16_lowering_pass(target)(cache_mod).script()
    assert "vortex_kv_cache_update" in cache_script
    assert 'R.call_pure_packed("relax.vortex.kv_cache_update"' not in cache_script

    attention_inputs = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.ones((128, 128), dtype=torch.float16),
        torch.ones((128, 32), dtype=torch.float16),
    )
    attention_mod = from_exported_program(
        torch.export.export(_W4AttentionFragment(), attention_inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    script = _w4a16_lowering_pass(target)(attention_mod).script()
    assert script.count("def vortex_mm_w4a16_improve") == 2
    assert script.count("vortex_quantize_int4_row_major") >= 2
    assert "vortex_gemm_w_tiled_transposed" in script
    assert "vortex_gemm_w_tiled" in script
    assert "dequantize" not in script
    assert "transpose(" not in script
    assert script.count("def vortex_gemm_a_tiled") == 1
    unfused_script = _w4a16_lowering_pass(
        target,
        enable_layout_fusion=False,
        lower_auxiliary_ops=False,
    )(attention_mod).script()
    assert unfused_script.count("R.call_tir(cls.vortex_gemm_a_tiled") == 2
    assert unfused_script.count("def vortex_gemm_c_detile") == 2


def test_cached_decode_lowers_quantize_update_qkt_and_pv_without_round_trip():
    _register_logical_ops()
    example = (
        torch.ones((7, 33), dtype=torch.float16),
        torch.zeros((129, 17), dtype=torch.uint8),
        torch.zeros((129, 2), dtype=torch.float16),
        torch.zeros((129, 2), dtype=torch.int16),
        torch.zeros((129, 9), dtype=torch.uint8),
        torch.zeros((129, 1), dtype=torch.float16),
        torch.zeros((129, 1), dtype=torch.int16),
        torch.ones((1, 33), dtype=torch.float16),
        torch.ones((1, 17), dtype=torch.float16),
    )
    mod = from_exported_program(
        torch.export.export(_W4CachedDecode(), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    script = _w4a16_lowering_pass(target)(mod).script()
    assert script.count("def vortex_kv_cache_update") == 2
    assert script.count("def vortex_mm_w4a16_improve") == 2
    assert script.count("def vortex_gemm_a_tiled") == 1
    assert "dequantize" not in script
    assert "transpose(" not in script


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact naive GEMM XRT environment",
)
@pytest.mark.parametrize(
    ("transpose_rhs", "quant_axis"),
    [(False, 0), (False, 1), (True, 1)],
)
def test_naive_w4a16_relax_vm_hardware(transpose_rhs, quant_axis):
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == NAIVE_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        NAIVE_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    executable = relax.build(
        _import_packed_w4a16(transpose_rhs, quant_axis), target, exec_mode="bytecode"
    )

    rng = np.random.default_rng(51)
    lhs_host = rng.uniform(-0.5, 0.5, (8, 128)).astype("float16")
    weight = rng.integers(-3, 4, size=(128, 32), dtype="int8")
    source_rhs = weight.T if transpose_rhs else weight
    packed = np.bitwise_or(
        np.bitwise_and(source_rhs[:, 0::2], 0xF),
        np.left_shift(np.bitwise_and(source_rhs[:, 1::2], 0xF), 4),
    ).astype("uint8")
    qparam_shape = list(source_rhs.shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    scale = np.full(qparam_shape, 0.125, dtype="float16")
    zero_point = np.zeros(qparam_shape, dtype="int16")
    dequantized = weight.astype("float32") * 0.125
    expected = (lhs_host.astype("float32") @ dequantized).astype("float16")

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *[
            tvm.runtime.tensor(array, device=device)
            for array in (lhs_host, packed, scale, zero_point)
        ]
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("qblock", [32, 128])
def test_direct_improved_gemm_abi_v2_hardware(qblock):
    """Bypass Relax transforms and submit known-good physical buffers directly."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    accelerator = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(accelerator.target, host="llvm")
    plan = plan_improve_layout(7, 31, 33, qblock)
    kernel = _make_w4a16_improve(
        plan.a_elements,
        plan.weight_bytes,
        plan.qparam_elements,
        plan.c_elements,
        plan.logical_m,
        plan.execution_n,
        plan.execution_k,
        plan.qblock,
        0,
        0,
        plan.logical_n,
        plan.logical_k,
        plan.profile.layout_abi_version,
    ).with_attr("global_symbol", "direct_improve_gemm")
    executable = tvm.tirx.build(kernel, target=target)

    rng = np.random.default_rng(59)
    lhs = rng.uniform(-0.5, 0.5, (plan.logical_m, plan.logical_k)).astype("float16")
    weight = rng.integers(
        -3, 4, size=(plan.logical_k, plan.logical_n), dtype="int8"
    )
    scale = np.full(
        ((plan.logical_k + plan.qblock - 1) // plan.qblock, plan.logical_n),
        0.125,
        dtype="float16",
    )
    zero_point = np.zeros(scale.shape, dtype="int16")
    physical = _pack_improve_physical(lhs, weight, scale, zero_point, plan)

    device = tvm.vortex(0)
    device_inputs = [tvm.runtime.tensor(value, device=device) for value in physical]
    tiled_output = tvm.runtime.empty((plan.c_elements,), "float16", device=device)
    executable["direct_improve_gemm"](*device_inputs, tiled_output)
    actual = _detile_improve_output(tiled_output.numpy(), plan)

    scale_by_k = np.repeat(scale.astype("float32"), plan.qblock, axis=0)
    dequantized = weight.astype("float32") * scale_by_k[: plan.logical_k]
    expected = (lhs.astype("float32") @ dequantized).astype("float16")
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
def test_improved_qrow_qblock128_relax_vm_hardware():
    """Exercise the Llama KV4 QDIR=1 contract through the full Relax layout path."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    accelerator = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(accelerator.target, host="llvm")
    executable = relax.build(
        _import_packed_w4a16(
            transpose_rhs=False,
            quant_axis=1,
            m=3,
            n=33,
            k=7,
            group_size=128,
        ),
        target,
        exec_mode="bytecode",
    )

    rng = np.random.default_rng(83)
    lhs = rng.uniform(-0.5, 0.5, (3, 7)).astype("float16")
    weight = rng.integers(-3, 4, size=(7, 33), dtype="int8")
    packed = _pack_int4_last_axis(weight)
    scale = rng.uniform(0.05, 0.15, (7, 1)).astype("float16")
    zero_point = rng.integers(-2, 3, size=(7, 1), dtype="int16")
    dequantized = (
        weight.astype("float32") - zero_point.astype("float32")
    ) * scale.astype("float32")
    expected = (lhs.astype("float32") @ dequantized).astype("float16")

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *[
            tvm.runtime.tensor(value, device=device)
            for value in (lhs, packed, scale, zero_point)
        ]
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
def test_improved_w4a16_first_k_tail_group_sentinel_hardware():
    """The first padded K micro-tile must contribute through the full Relax VM."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    accelerator = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(accelerator.target, host="llvm")
    executable = relax.build(
        _import_packed_w4a16(False, 0, m=7, n=31, k=33),
        target,
        exec_mode="bytecode",
    )

    lhs = np.zeros((7, 33), dtype="float16")
    lhs[:, 32] = np.float16(1)
    weight = np.zeros((33, 31), dtype="int8")
    weight[32, :] = np.int8(2)
    packed = _pack_int4_last_axis(weight)
    scale = np.zeros((2, 31), dtype="float16")
    scale[1, :] = np.float16(0.125)
    zero_point = np.zeros((2, 31), dtype="int16")

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *[
            tvm.runtime.tensor(value, device=device)
            for value in (lhs, packed, scale, zero_point)
        ]
    ).numpy()
    np.testing.assert_array_equal(actual, np.full((7, 31), 0.25, dtype="float16"))


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize(
    ("m", "n", "k", "transpose_rhs", "quant_direction"),
    [
        (7, 31, 31, False, 0),
        (7, 31, 32, False, 0),
        (7, 31, 33, False, 0),
        (7, 31, 63, False, 0),
        (7, 31, 64, False, 0),
        (7, 31, 65, False, 0),
        (9, 33, 31, False, 1),
        (9, 33, 31, True, 0),
        (9, 33, 31, True, 1),
        (129, 257, 193, False, 0),
    ],
)
def test_improved_w4a16_arbitrary_shape_hardware(
    m, n, k, transpose_rhs, quant_direction
):
    """Cover K boundaries, both layout directions, transpose, and outer tiles."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    accelerator = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(accelerator.target, host="llvm")
    source_k_axis = 1 if transpose_rhs else 0
    quant_axis = source_k_axis if quant_direction == 0 else 1 - source_k_axis
    executable = relax.build(
        _import_packed_w4a16(
            transpose_rhs, quant_axis, m=m, n=n, k=k
        ),
        target,
        exec_mode="bytecode",
    )

    rng = np.random.default_rng(
        1000 + m * 3 + n * 5 + k * 7 + int(transpose_rhs) * 11 + quant_direction
    )
    lhs = rng.uniform(-0.1, 0.1, (m, k)).astype("float16")
    weight = rng.integers(-2, 3, size=(k, n), dtype="int8")
    source_rhs = weight.T if transpose_rhs else weight
    packed = _pack_int4_last_axis(source_rhs)
    qparam_shape = list(source_rhs.shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    scale = np.full(qparam_shape, 0.125, dtype="float16")
    zero_point = np.zeros(qparam_shape, dtype="int16")
    expected = (lhs.astype("float32") @ (weight.astype("float32") * 0.125)).astype(
        "float16"
    )

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *[
            tvm.runtime.tensor(value, device=device)
            for value in (lhs, packed, scale, zero_point)
        ]
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize(
    ("transpose_rhs", "quant_axis"),
    [(False, 0), (False, 1), (True, 1)],
)
def test_improved_w4a16_layout_pipeline_hardware(transpose_rhs, quant_axis):
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    executable = relax.build(
        _import_packed_w4a16(transpose_rhs, quant_axis), target, exec_mode="bytecode"
    )

    rng = np.random.default_rng(61)
    lhs_host = rng.uniform(-0.5, 0.5, (8, 128)).astype("float16")
    weight = rng.integers(-3, 4, size=(128, 32), dtype="int8")
    source_rhs = weight.T if transpose_rhs else weight
    packed = np.bitwise_or(
        np.bitwise_and(source_rhs[:, 0::2], 0xF),
        np.left_shift(np.bitwise_and(source_rhs[:, 1::2], 0xF), 4),
    ).astype("uint8")
    qparam_shape = list(source_rhs.shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    scale = np.full(qparam_shape, 0.125, dtype="float16")
    zero_point = np.zeros(qparam_shape, dtype="int16")
    expected = (lhs_host.astype("float32") @ (weight.astype("float32") * 0.125)).astype(
        "float16"
    )

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        *[
            tvm.runtime.tensor(array, device=device)
            for array in (lhs_host, packed, scale, zero_point)
        ]
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_quantize_to_improved_w4a16_hardware(transpose_rhs):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    rhs_shape = (32, 128) if transpose_rhs else (128, 32)
    inputs = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.ones(rhs_shape, dtype=torch.float16),
    )
    exported = torch.export.export(_QuantizedW4A16(transpose_rhs), inputs)
    mod = from_exported_program(
        exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True
    )
    executable = relax.build(mod, target, exec_mode="bytecode")

    rng = np.random.default_rng(71)
    lhs_host = rng.uniform(-0.5, 0.5, (8, 128)).astype("float16")
    rhs_host = rng.uniform(-1.0, 1.0, rhs_shape).astype("float16")
    expected = _QuantizedW4A16(transpose_rhs)(
        torch.from_numpy(lhs_host), torch.from_numpy(rhs_host)
    ).numpy()
    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        tvm.runtime.tensor(lhs_host, device=device),
        tvm.runtime.tensor(rhs_host, device=device),
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_layout_fused_ffn_matches_unfused_export_reload_hardware(exec_mode):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    example = _ffn_inputs()
    model = _W4FFN()
    logical = from_exported_program(
        torch.export.export(model, example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    fused = relax.build(logical, target, exec_mode=exec_mode)
    unfused_mod = _w4a16_lowering_pass(
        target, enable_layout_fusion=False
    )(logical)
    unfused = relax.build(unfused_mod, target, exec_mode=exec_mode)

    with tempfile.TemporaryDirectory(prefix="tvm-vortex-w4-ffn-") as directory:
        fused_artifact = Path(directory) / f"ffn-fused-{exec_mode}.so"
        unfused_artifact = Path(directory) / f"ffn-unfused-{exec_mode}.so"
        fused.export_library(str(fused_artifact))
        unfused.export_library(str(unfused_artifact))
        device = tvm.vortex(0)
        fused_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(fused_artifact)),
            device=device,
            memory_cfg="naive",
        )
        unfused_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(unfused_artifact)),
            device=device,
            memory_cfg="naive",
        )
        for seed in (111, 112):
            values = list(_ffn_inputs(seed))
            expected = model(*values).numpy()
            device_values = [
                tvm.runtime.tensor(value.numpy(), device=device) for value in values
            ]
            actual_fused = fused_vm["main"](*device_values).numpy()
            actual_unfused = unfused_vm["main"](*device_values).numpy()
            np.testing.assert_allclose(
                actual_fused, expected, rtol=7e-2, atol=7e-2
            )
            np.testing.assert_allclose(
                actual_unfused, expected, rtol=7e-2, atol=7e-2
            )
            np.testing.assert_allclose(
                actual_fused, actual_unfused, rtol=7e-2, atol=7e-2
            )


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
@pytest.mark.parametrize(
    ("transpose_rhs", "quant_axis"),
    [(False, 0), (False, 1), (True, 0), (True, 1)],
)
def test_constant_prepacked_w4a16_repeated_export_reload_hardware(
    exec_mode, transpose_rhs, quant_axis
):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    rng = np.random.default_rng(120 + int(transpose_rhs) * 2 + quant_axis)
    rhs_shape = (31, 33) if transpose_rhs else (33, 31)
    packed_shape = (rhs_shape[0], (rhs_shape[1] + 1) // 2)
    qparam_shape = list(rhs_shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    packed = torch.from_numpy(rng.integers(0, 256, packed_shape, dtype="uint8"))
    scale = torch.from_numpy(
        rng.uniform(0.01, 0.04, qparam_shape).astype("float16")
    )
    zero = torch.zeros(qparam_shape, dtype=torch.int16)
    model = _ConstantPackedW4A16(
        packed, scale, zero, transpose_rhs, quant_axis
    )
    example_lhs = torch.ones((7, 33), dtype=torch.float16)
    logical = from_exported_program(
        torch.export.export(model, (example_lhs,)),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(logical, target, exec_mode=exec_mode)

    with tempfile.TemporaryDirectory(prefix="tvm-vortex-w4-constant-") as directory:
        artifact = Path(directory) / f"constant-prepacked-{exec_mode}.so"
        executable.export_library(str(artifact))
        device = tvm.vortex(0)
        vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(artifact)), device=device, memory_cfg="naive"
        )
        for seed in (121, 122):
            invocation_rng = np.random.default_rng(seed)
            lhs = invocation_rng.uniform(-0.1, 0.1, (7, 33)).astype("float16")
            expected = model(torch.from_numpy(lhs)).numpy()
            actual = vm["main"](tvm.runtime.tensor(lhs, device=device)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("scheme", ["signed_symmetric_int4", "signed_asymmetric_int4"])
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_quantize_dequantize_export_reload_hardware(scheme, exec_mode):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    example = (torch.ones((3, 65), dtype=torch.float16),)
    mod = from_exported_program(
        torch.export.export(_QuantizeDequantize(scheme), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(mod, target, exec_mode=exec_mode)

    with tempfile.TemporaryDirectory(prefix="tvm-vortex-int4-") as directory:
        artifact = Path(directory) / f"quant-dequant-{exec_mode}.so"
        executable.export_library(str(artifact))
        restored = tvm.runtime.load_module(str(artifact))
        rng = np.random.default_rng(81)
        source = rng.uniform(-2.0, 2.0, (3, 65)).astype("float16")
        expected = _QuantizeDequantize(scheme)(torch.from_numpy(source)).numpy()
        device = tvm.vortex(0)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        actual = vm["main"](tvm.runtime.tensor(source, device=device)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-3)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_kv_cache_update_repeated_export_reload_hardware(exec_mode):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    example = (
        torch.zeros((1, 4, 64), dtype=torch.uint8),
        torch.zeros((1, 4, 4), dtype=torch.float16),
        torch.zeros((1, 4, 4), dtype=torch.int16),
        torch.ones((1, 1, 64), dtype=torch.uint8),
        torch.ones((1, 1, 4), dtype=torch.float16),
        torch.ones((1, 1, 4), dtype=torch.int16),
    )
    mod = from_exported_program(
        torch.export.export(_KVCacheUpdate(), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(mod, target, exec_mode=exec_mode)

    with tempfile.TemporaryDirectory(prefix="tvm-vortex-kv-cache-") as directory:
        artifact = Path(directory) / f"kv-cache-{exec_mode}.so"
        executable.export_library(str(artifact))
        restored = tvm.runtime.load_module(str(artifact))
        device = tvm.vortex(0)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        first = vm["main"](
            *[tvm.runtime.tensor(value.numpy(), device=device) for value in example]
        )
        second_update = (
            np.full((1, 1, 64), 9, dtype="uint8"),
            np.full((1, 1, 4), 0.25, dtype="float16"),
            np.full((1, 1, 4), -2, dtype="int16"),
        )
        second = vm["main"](
            first[0],
            first[1],
            first[2],
            *[tvm.runtime.tensor(value, device=device) for value in second_update],
        )
        actual = tuple(value.numpy() for value in second)

    assert np.count_nonzero(actual[0][:, :2]) == 0
    assert np.count_nonzero(actual[0][:, 3:]) == 0
    np.testing.assert_array_equal(actual[0][:, 2:3], second_update[0])
    np.testing.assert_array_equal(actual[1][:, 2:3], second_update[1])
    np.testing.assert_array_equal(actual[2][:, 2:3], second_update[2])


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_w4_attention_fragment_export_reload_hardware(exec_mode):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    example = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.ones((128, 128), dtype=torch.float16),
        torch.ones((128, 32), dtype=torch.float16),
    )
    mod = from_exported_program(
        torch.export.export(_W4AttentionFragment(), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    fused = relax.build(mod, target, exec_mode=exec_mode)
    unfused_mod = _w4a16_lowering_pass(
        target,
        enable_layout_fusion=False,
        lower_auxiliary_ops=False,
    )(mod)
    unfused = relax.build(unfused_mod, target, exec_mode=exec_mode)

    rng = np.random.default_rng(91)
    inputs = (
        rng.uniform(-0.1, 0.1, (8, 128)).astype("float16"),
        rng.uniform(-0.2, 0.2, (128, 128)).astype("float16"),
        rng.uniform(-0.2, 0.2, (128, 32)).astype("float16"),
    )
    expected = _W4AttentionFragment()(
        *[torch.from_numpy(value) for value in inputs]
    ).numpy()
    with tempfile.TemporaryDirectory(prefix="tvm-vortex-w4-attention-") as directory:
        fused_artifact = Path(directory) / f"w4-attention-fused-{exec_mode}.so"
        unfused_artifact = Path(directory) / f"w4-attention-unfused-{exec_mode}.so"
        fused.export_library(str(fused_artifact))
        unfused.export_library(str(unfused_artifact))
        device = tvm.vortex(0)
        fused_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(fused_artifact)),
            device=device,
            memory_cfg="naive",
        )
        unfused_vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(unfused_artifact)),
            device=device,
            memory_cfg="naive",
        )
        actual_fused = fused_vm["main"](
            *[tvm.runtime.tensor(value, device=device) for value in inputs]
        ).numpy()
        actual_unfused = unfused_vm["main"](
            *[tvm.runtime.tensor(value, device=device) for value in inputs]
        ).numpy()
    # Two quantized GEMMs accumulate FP16 rounding independently; keep the
    # tolerance aligned with the existing per-GEMM 3e-2 budget.
    np.testing.assert_allclose(actual_fused, expected, rtol=7e-2, atol=7e-2)
    np.testing.assert_allclose(actual_unfused, expected, rtol=7e-2, atol=7e-2)
    np.testing.assert_allclose(
        actual_fused, actual_unfused, rtol=7e-2, atol=7e-2
    )


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact improved GEMM XRT environment",
)
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_w4_cached_decode_repeated_export_reload_hardware(exec_mode):
    _register_logical_ops()
    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    example = (
        torch.ones((7, 33), dtype=torch.float16),
        torch.zeros((129, 17), dtype=torch.uint8),
        torch.zeros((129, 2), dtype=torch.float16),
        torch.zeros((129, 2), dtype=torch.int16),
        torch.zeros((129, 9), dtype=torch.uint8),
        torch.zeros((129, 1), dtype=torch.float16),
        torch.zeros((129, 1), dtype=torch.int16),
        torch.ones((1, 33), dtype=torch.float16),
        torch.ones((1, 17), dtype=torch.float16),
    )
    mod = from_exported_program(
        torch.export.export(_W4CachedDecode(), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(mod, target, exec_mode=exec_mode)

    rng = np.random.default_rng(101)
    first_inputs = [value.numpy() for value in example]
    first_inputs[0] = rng.uniform(-0.1, 0.1, (7, 33)).astype("float16")
    first_inputs[7] = rng.uniform(-0.2, 0.2, (1, 33)).astype("float16")
    first_inputs[8] = rng.uniform(-0.2, 0.2, (1, 17)).astype("float16")
    eager_first = _W4CachedDecode()(
        *[torch.from_numpy(value) for value in first_inputs]
    )

    with tempfile.TemporaryDirectory(
        prefix="tvm-vortex-w4-cached-decode-"
    ) as directory:
        artifact = Path(directory) / f"w4-cached-decode-{exec_mode}.so"
        executable.export_library(str(artifact))
        restored = tvm.runtime.load_module(str(artifact))
        device = tvm.vortex(0)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        first = vm["main"](
            *[tvm.runtime.tensor(value, device=device) for value in first_inputs]
        )

        second_inputs = [
            first_inputs[0],
            *[value.numpy() for value in first[1:]],
            rng.uniform(-0.2, 0.2, (1, 33)).astype("float16"),
            rng.uniform(-0.2, 0.2, (1, 17)).astype("float16"),
        ]
        eager_second = _W4CachedDecode()(
            *[torch.from_numpy(value) for value in second_inputs]
        )
        second = vm["main"](
            *[tvm.runtime.tensor(value, device=device) for value in second_inputs]
        )
        actual_first = tuple(value.numpy() for value in first)
        actual_second = tuple(value.numpy() for value in second)

    expected_first = tuple(value.numpy() for value in eager_first)
    expected_second = tuple(value.numpy() for value in eager_second)
    for actual, expected in zip(actual_first[1:], expected_first[1:]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_second[1:], expected_second[1:]):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual_first[0], expected_first[0], rtol=7e-2, atol=7e-2)
    np.testing.assert_allclose(
        actual_second[0], expected_second[0], rtol=7e-2, atol=7e-2
    )


if __name__ == "__main__":
    tvm.testing.main()
