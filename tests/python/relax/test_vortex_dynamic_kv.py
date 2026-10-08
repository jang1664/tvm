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

"""Runtime-prefix lowering and poisoned-tail regression tests, without FPGA."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tvm
from tvm import relax
from tvm.relax.backend.vortex.layout import ImproveProfile, plan_improve_layout
from tvm.relax.backend.vortex.pipeline import (
    _make_prefix_mask, _make_causal_softmax, _make_w4a16_region, _w4a16_lowering_pass,
)
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile
from test_vortex_packed_c_layout import _host_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps"))
from vortex_llama3.run_dynamic_kv_probe import PrefixAttention, make_inputs


@pytest.mark.parametrize("axis,shape", [(0, (320, 1)), (1, (4, 320))])
def test_prefix_mask_neutralizes_nonfinite_tail(axis, shape):
    kernel = _host_kernel(_make_prefix_mask(shape, axis))
    for length in (1, 17, 129, 257, 320):
        source = np.full(shape, np.nan, dtype="float16")
        view = [slice(None)] * 2
        view[axis] = slice(0, length)
        source[tuple(view)] = .25
        expected = np.nan_to_num(source)
        output = tvm.runtime.empty(shape, "float16")
        kernel(tvm.runtime.tensor(source), tvm.runtime.tensor(np.array(length, dtype="int64")), output)
        np.testing.assert_array_equal(output.numpy(), expected)


def test_dynamic_softmax_does_not_read_inactive_scores():
    shape = (1, 1, 2, 1, 320)
    kernel = _host_kernel(_make_causal_softmax(shape, (1, 1), (), 128, True))
    for length in (1, 17, 129, 257, 320):
        source = np.full(shape, np.nan, dtype="float16")
        source[..., :length] = np.arange(length, dtype="float16") / 32
        scores, probability = tvm.runtime.empty(shape, "float32"), tvm.runtime.empty(shape, "float16")
        kernel(tvm.runtime.tensor(source), tvm.runtime.tensor(np.array([[length - 1]], dtype="int64")),
               tvm.runtime.tensor(np.array(length, dtype="int64")), scores, probability)
        expected = torch.softmax(torch.tensor(source[..., :length]).float() / np.sqrt(128), -1).half().numpy()
        np.testing.assert_allclose(probability.numpy()[..., :length], expected, rtol=.001, atol=1e-5)
        assert np.all(probability.numpy()[..., length:] == 0)
        assert np.all(np.isneginf(scores.numpy()[..., length:]))


def _target(tmp_path, abi):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"vortex_layout_abi_version": 3,
        "vortex_gemm_abi_version": abi, "params": {"CONFIGS":
        "-DNUM_THREADS=16 -DMXU_ROW=16 -DMXU_COL=16 -DENABLE_GEMM_ACCEL -DGEMM_IMPROVE"}}))
    return load_vortex_accelerator_profile(path).target


def _attention():
    exported = torch.export.export(PrefixAttention(320), make_inputs(320, 1, 17), strict=True)
    return from_exported_program(exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True)


def test_attention_runtime_extent_lowering(tmp_path):
    target = _target(tmp_path, 3)
    with target:
        mod = _w4a16_lowering_pass(target, dynamic_kv_length=True)(_attention())
    assert int(mod.attrs["vortex.dynamic_kv_length.calls"]) == 2
    script = mod.script()
    assert "vx_tvm_gemm_w4a16_region" in script
    assert "vortex_mm_w4a16_region_n" in script
    assert "vortex_mm_w4a16_region_k" in script
    assert "T.int64(16)" in script
    assert "vortex_prefix_mask" in script


def test_reject_old_fsm_for_dynamic_attention(tmp_path):
    target = _target(tmp_path, 2)
    with target, pytest.raises(ValueError, match="submission ABI 3"):
        _w4a16_lowering_pass(target, dynamic_kv_length=True)(_attention())


def test_region_preserves_original_capacity():
    p = dataclasses.replace(ImproveProfile(), mxu_kt=16, mxu_nt=16, layout_abi_version=3)
    for axis, dims, trans, qdir in [("N", (1, 512, 128), True, 0), ("K", (1, 128, 512), False, 1)]:
        plan = plan_improve_layout(*dims, 128, trans, qdir, p)
        script = _make_w4a16_region(plan, axis).script()
        assert "length[()]" in script
        expected = ", 1, 512, 128, 128," if axis == "N" else ", 1, 128, 512, 128,"
        assert expected in script
