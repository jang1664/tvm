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
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import tvm
from tvm import relax
from tvm.relax.backend.vortex.pipeline import _w4a16_lowering_pass
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile

VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
OPS_PATH = VORTEX_HOME / "pytorch/spinquant/spinquant_inference/vortex_export_ops.py"
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
    def __init__(self, transpose_rhs=False, quant_axis=0):
        super().__init__()
        self.transpose_rhs = transpose_rhs
        self.quant_axis = quant_axis

    def forward(self, lhs, packed, scale, zero_point):
        return torch.ops.vortex.mm_w4a16(
            lhs,
            packed,
            scale,
            zero_point,
            [32, 128] if self.transpose_rhs else [128, 32],
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
            128,
        )
        score = torch.ops.vortex.mm_w4a16(
            query,
            key_cache[0],
            key_cache[1],
            key_cache[2],
            [128, 128],
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
            128,
        )
        context = torch.ops.vortex.mm_w4a16(
            score,
            value_cache[0],
            value_cache[1],
            value_cache[2],
            [128, 32],
            32,
            1,
            1,
            "signed_asymmetric_int4",
            False,
        )
        return context, *key_cache, *value_cache


def _import_packed_w4a16(transpose_rhs=False, quant_axis=0):
    _register_logical_ops()
    rhs_shape = (32, 128) if transpose_rhs else (128, 32)
    packed_shape = (rhs_shape[0], (rhs_shape[1] + 1) // 2)
    qparam_shape = list(rhs_shape)
    qparam_shape[quant_axis] = (qparam_shape[quant_axis] + 31) // 32
    inputs = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.zeros(packed_shape, dtype=torch.uint8),
        torch.ones(qparam_shape, dtype=torch.float16),
        torch.zeros(qparam_shape, dtype=torch.int16),
    )
    return from_exported_program(
        torch.export.export(_PackedW4A16(transpose_rhs, quant_axis), inputs),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )


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


def test_cached_decode_lowers_quantize_update_qkt_and_pv_without_round_trip():
    _register_logical_ops()
    example = (
        torch.ones((8, 128), dtype=torch.float16),
        torch.zeros((128, 64), dtype=torch.uint8),
        torch.zeros((128, 4), dtype=torch.float16),
        torch.zeros((128, 4), dtype=torch.int16),
        torch.zeros((128, 16), dtype=torch.uint8),
        torch.zeros((128, 1), dtype=torch.float16),
        torch.zeros((128, 1), dtype=torch.int16),
        torch.ones((1, 128), dtype=torch.float16),
        torch.ones((1, 32), dtype=torch.float16),
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
    executable = relax.build(mod, target, exec_mode=exec_mode)

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
        artifact = Path(directory) / f"w4-attention-{exec_mode}.so"
        executable.export_library(str(artifact))
        restored = tvm.runtime.load_module(str(artifact))
        device = tvm.vortex(0)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        actual = vm["main"](
            *[tvm.runtime.tensor(value, device=device) for value in inputs]
        ).numpy()
    # Two quantized GEMMs accumulate FP16 rounding independently; keep the
    # tolerance aligned with the existing per-GEMM 3e-2 budget.
    np.testing.assert_allclose(actual, expected, rtol=7e-2, atol=7e-2)


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
        torch.ones((8, 128), dtype=torch.float16),
        torch.zeros((128, 64), dtype=torch.uint8),
        torch.zeros((128, 4), dtype=torch.float16),
        torch.zeros((128, 4), dtype=torch.int16),
        torch.zeros((128, 16), dtype=torch.uint8),
        torch.zeros((128, 1), dtype=torch.float16),
        torch.zeros((128, 1), dtype=torch.int16),
        torch.ones((1, 128), dtype=torch.float16),
        torch.ones((1, 32), dtype=torch.float16),
    )
    mod = from_exported_program(
        torch.export.export(_W4CachedDecode(), example),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    executable = relax.build(mod, target, exec_mode=exec_mode)

    rng = np.random.default_rng(101)
    first_inputs = [value.numpy() for value in example]
    first_inputs[0] = rng.uniform(-0.1, 0.1, (8, 128)).astype("float16")
    first_inputs[7] = rng.uniform(-0.2, 0.2, (1, 128)).astype("float16")
    first_inputs[8] = rng.uniform(-0.2, 0.2, (1, 32)).astype("float16")
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
            rng.uniform(-0.2, 0.2, (1, 128)).astype("float16"),
            rng.uniform(-0.2, 0.2, (1, 32)).astype("float16"),
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
