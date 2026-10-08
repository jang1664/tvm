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
"""Execute fused KV stores and compare with the independent full-layout packer."""

import dataclasses
import json

import numpy as np
import pytest
import torch
from test_vortex_dynamic_kv import _target
from test_vortex_packed_c_layout import _host_kernel
from vortex_llama3.run_packed_kv_probe import AppendAttention, append_inputs

import tvm
from tvm.relax.backend.vortex.layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)
from tvm.relax.backend.vortex.packed_kv import make_quantize_cache_store
from tvm.relax.backend.vortex.pipeline import _make_quantize_int4_row_major, _w4a16_lowering_pass
from tvm.relax.frontend.torch import from_exported_program


@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize(
    "capacity,group,tokens,dim", [(320, 128, 1, 128), (512, 32, 4, 128), (320, 128, 1, 256)]
)
def test_incremental_quantize_layout(trans, capacity, group, tokens, dim):
    profile = dataclasses.replace(ImproveProfile(), mxu_kt=16, mxu_nt=16, layout_abi_version=3)
    plan = plan_improve_layout(
        1,
        capacity if trans else dim,
        dim if trans else capacity,
        group,
        trans,
        0 if trans else 1,
        profile,
    )
    heads = 2
    shapes = [
        (1, heads, capacity, dim // 2),
        (1, heads, capacity, dim // group),
        (1, heads, capacity, dim // group),
    ]
    quant = _make_quantize_int4_row_major((heads * tokens, dim), group, "signed_asymmetric_int4")
    fused = _host_kernel(make_quantize_cache_store(quant, shapes, tokens, plan))
    canonical = _host_kernel(quant)
    cache = [
        tvm.runtime.tensor(np.zeros(s, d)) for s, d in zip(shapes, ("uint8", "float16", "int16"))
    ]
    packed = [
        tvm.runtime.tensor(np.zeros(heads * size, d))
        for size, d in zip(
            (plan.weight_bytes, plan.qparam_elements, plan.qparam_elements),
            ("uint8", "float16", "int16"),
        )
    ]
    rng = np.random.default_rng(20261008)
    for pos in (0, 15, 16, 127, 128, 255, 256, capacity - tokens):
        values = tvm.runtime.tensor(rng.uniform(-4, 4, (heads * tokens, dim)).astype("float16"))
        outputs = [
            tvm.runtime.empty((heads * tokens, cols), d)
            for cols, d in ((dim // 2, "uint8"), (dim // group, "float16"), (dim // group, "int16"))
        ]
        canonical(values, *outputs)
        before = [x.numpy() for x in cache]
        fused(values, *cache, *packed, tvm.runtime.tensor(np.array(pos, "int64")))
        for i, (buf, out) in enumerate(zip(cache, outputs)):
            before[i][:, :, pos : pos + tokens, :] = out.numpy().reshape(1, heads, tokens, -1)
            np.testing.assert_array_equal(buf.numpy(), before[i])
        for h in range(heads):
            refs = [
                prepack_improve_weight(cache[0].numpy()[0, h], plan),
                prepack_improve_qparam(cache[1].numpy()[0, h], plan, "float16"),
                prepack_improve_qparam(cache[2].numpy()[0, h], plan, "int16"),
            ]
            for actual, ref in zip(packed, refs):
                np.testing.assert_array_equal(actual.numpy().reshape(heads, -1)[h], ref)


@pytest.mark.parametrize("prefill,tokens", [(False, 1), (True, 4)])
def test_fused_cache_attention_lowering(tmp_path, prefill, tokens):
    model = AppendAttention(tokens=tokens, prefill=prefill)
    mod = from_exported_program(
        torch.export.export(model, append_inputs(tokens=tokens), strict=True),
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    target = _target(tmp_path, 3)
    with target:
        mod = _w4a16_lowering_pass(target, dynamic_kv_length=True, packed_kv_cache=True)(mod)
    script = mod.script()
    assert "vortex_quantize_packed_kv_append" in script
    assert "vortex_mm_packed_kv_region" in script
    for forbidden in (
        "vortex_gemm_w_tiled",
        "vortex_gemm_scale_tiled",
        "vortex_gemm_zero_point_tiled",
        "vortex_batched_packed_matrix",
        "vortex_quantize_int4_row_major",
        "vortex_kv_cache_update_dynamic",
    ):
        assert forbidden not in script
    metadata = json.loads(str(mod.attrs["vortex.packed_kv_cache"]))
    assert len(metadata) == 2
    assert len(mod["main"].params) == len(append_inputs(tokens=tokens)) + 6


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_llama_layer_packed_kv(tmp_path, phase):
    from test_vortex_llama3_backend_policy import _import_layer

    from tvm.relax.backend.vortex.packed_kv import prepare_packed_kv_cache

    mod = _import_layer(phase, "w4", "w4", head_dim=128, cache_capacity=320)
    original_count = len(mod["main"].params)
    mod, metadata = prepare_packed_kv_cache(mod, _target(tmp_path, 3))
    assert len(metadata) == 2
    assert len(mod["main"].params) == original_count + 6
    script = mod.script()
    assert "vortex_batched_packed_matrix" not in script
    assert "vortex_batched_scale_matrix" not in script
    assert "vortex_quantize_packed_kv_append" in script


@pytest.mark.parametrize("trans", [True, False])
def test_prefill_initialization_preserves_readonly_shared_cache(trans):
    profile = dataclasses.replace(ImproveProfile(), mxu_kt=16, mxu_nt=16, layout_abi_version=3)
    plan = plan_improve_layout(
        1, 320 if trans else 128, 128 if trans else 320, 128, trans, 0 if trans else 1, profile
    )
    shapes = [(1, 1, 320, 64), (1, 1, 320, 1), (1, 1, 320, 1)]
    dtypes = ["uint8", "float16", "int16"]
    initial = [tvm.runtime.tensor(np.zeros(s, d)) for s, d in zip(shapes, dtypes)]
    output = [tvm.runtime.tensor(np.full(s, 7, d)) for s, d in zip(shapes, dtypes)]
    packed = [
        tvm.runtime.tensor(np.zeros(size, d))
        for size, d in zip((plan.weight_bytes, plan.qparam_elements, plan.qparam_elements), dtypes)
    ]
    quant = _make_quantize_int4_row_major((4, 128), 128, "signed_asymmetric_int4")
    kernel = _host_kernel(make_quantize_cache_store(quant, shapes, 4, plan, initialize=True))
    source = tvm.runtime.tensor(np.random.default_rng(8).uniform(1, 5, (4, 128)).astype("float16"))
    kernel(source, *initial, *packed, tvm.runtime.tensor(np.array(0, "int64")), *output)
    for src, dst in zip(initial, output):
        assert np.all(src.numpy() == 0)
        assert np.all(dst.numpy()[:, :, 4:, :] == 0)
        assert np.any(dst.numpy()[:, :, :4, :] != 0)
    refs = [
        prepack_improve_weight(output[0].numpy()[0, 0], plan),
        prepack_improve_qparam(output[1].numpy()[0, 0], plan, "float16"),
        prepack_improve_qparam(output[2].numpy()[0, 0], plan, "int16"),
    ]
    for value, expected in zip(packed, refs):
        np.testing.assert_array_equal(value.numpy(), expected)
