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

"""Execute actual layout TIR on LLVM without requiring an FPGA image."""

import dataclasses

import numpy as np
import pytest

import tvm
from tvm import tirx
from tvm.relax.backend.vortex.layout import ImproveProfile, plan_improve_layout
from tvm.relax.backend.vortex.pipeline import (
    _make_gemm_a_tiled,
    _make_gemm_c_detile,
    _make_gemm_tiled_add,
)


def _host_kernel(func):
    # These pointwise helpers have no barriers or cross-thread dependencies.
    # Keep their actual address expressions; only serialize thread bindings.
    def serialize(node):
        if isinstance(node, tirx.For) and node.thread_binding is not None:
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body)
        return None

    body = tirx.stmt_functor.ir_transform(func.body, None, serialize, ["tirx.For"])
    return tirx.build(func.with_body(body).with_attr("global_symbol", "main"), target="llvm")


def _plan(m, n, layout_abi):
    return plan_improve_layout(m, n, n, 32, profile=dataclasses.replace(
        ImproveProfile(), mxu_kt=16, mxu_nt=16, num_dma_channels=4,
        layout_abi_version=layout_abi,
    ))


def _physical_c(logical, plan):
    """Independent stream oracle: microtiles, then their ABI-specific padding."""
    p = plan.profile
    output = np.full(plan.c_elements, np.nan, dtype="float16")
    cursor = 0
    for m0 in range(0, plan.logical_m, p.dma_mt):
        rows = min(p.dma_mt, plan.logical_m - m0)
        slot_rows = (rows + 7) // 8 * 8
        for n0 in range(0, plan.execution_n, p.dma_nt):
            cols = min(p.dma_nt, plan.execution_n - n0)
            tile_end = cursor + slot_rows * cols
            for micro in range(n0, n0 + cols, p.mxu_nt):
                for row in range(rows):
                    for col in range(micro, micro + p.mxu_nt):
                        output[cursor] = logical[m0 + row, col] if col < plan.logical_n else 0
                        cursor += 1
                if p.layout_abi_version == 2:
                    cursor += (slot_rows - rows) * p.mxu_nt
            cursor = tile_end
    assert cursor == plan.c_elements
    return output


@pytest.mark.parametrize("shape", [(1, 16), (1, 48), (4, 256), (9, 33), (132, 144), (256, 256)])
@pytest.mark.parametrize("layout_abi", [2, 3])
def test_c_detile_matches_stream_oracle(shape, layout_abi):
    m, n = shape
    plan = _plan(m, n, layout_abi)
    logical = ((np.arange(m * n).reshape(m, n) % 251) - 125).astype("float16")
    source = tvm.runtime.tensor(_physical_c(logical, plan))
    output = tvm.runtime.empty((m, n), "float16")
    _host_kernel(_make_gemm_c_detile(plan))(source, output)
    np.testing.assert_array_equal(output.numpy(), logical)
    # For matched execution extents the actual A pack must reproduce all live C bytes.
    if layout_abi == 3 and plan.execution_k == plan.execution_n:
        packed_a = tvm.runtime.empty((plan.a_elements,), "float16")
        _host_kernel(_make_gemm_a_tiled(plan))(tvm.runtime.tensor(logical), packed_a)
        expected = source.numpy()
        live = ~np.isnan(expected)
        np.testing.assert_array_equal(packed_a.numpy()[live], expected[live])


@pytest.mark.parametrize("rhs_kind", ["vector", "row", "column", "matrix", "tiled"])
@pytest.mark.parametrize("layout_abi", [2, 3])
def test_layout_preserving_add_handles_dma_tails(rhs_kind, layout_abi):
    m, n = 132, 144
    plan = _plan(m, n, layout_abi)
    logical = ((np.arange(m * n).reshape(m, n) % 17) - 8).astype("float16")
    shape = {"vector": (n,), "row": (1, n), "column": (m, 1),
             "matrix": (m, n), "tiled": (m, n)}[rhs_kind]
    rhs = ((np.arange(np.prod(shape)).reshape(shape) % 7) - 3).astype("float16")
    rhs_physical = _physical_c(rhs, plan) if rhs_kind == "tiled" else rhs
    add = _host_kernel(_make_gemm_tiled_add(plan, None if rhs_kind == "tiled" else shape))
    result = tvm.runtime.empty((plan.c_elements,), "float16")
    add(tvm.runtime.tensor(_physical_c(logical, plan)), tvm.runtime.tensor(rhs_physical), result)
    output = tvm.runtime.empty((m, n), "float16")
    _host_kernel(_make_gemm_c_detile(plan))(result, output)
    np.testing.assert_array_equal(output.numpy(), logical + rhs)
