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

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T


def _make_vecadd(size, block_size):
    @T.prim_func
    def vecadd(
        a: T.Buffer((size,), "int32"),
        b: T.Buffer((size,), "int32"),
        c: T.Buffer((size,), "int32"),
    ):
        T.func_attr({"global_symbol": "vecadd", "tirx.noalias": True})
        for bx in T.thread_binding(
            (size + block_size - 1) // block_size, thread="blockIdx.x"
        ):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                if bx * block_size + tx < size:
                    c[bx * block_size + tx] = (
                        a[bx * block_size + tx] + b[bx * block_size + tx]
                    )

    return vecadd


def _build_vecadd(size, block_size):
    target = tvm.target.Target("vortex", host="llvm")
    return tvm.tirx.build(_make_vecadd(size, block_size), target=target)


def _make_matmul(m, n, k, block_size):
    @T.prim_func
    def matmul(
        a: T.Buffer((m, k), "float32"),
        b: T.Buffer((k, n), "float32"),
        c: T.Buffer((m, n), "float32"),
    ):
        T.func_attr({"global_symbol": "matmul", "tirx.noalias": True})
        for bx in T.thread_binding(
            (m * n + block_size - 1) // block_size, thread="blockIdx.x"
        ):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                if bx * block_size + tx < m * n:
                    c[(bx * block_size + tx) // n, (bx * block_size + tx) % n] = T.float32(0)
                    for reduction_index in range(k):
                        c[
                            (bx * block_size + tx) // n,
                            (bx * block_size + tx) % n,
                        ] = (
                            c[
                                (bx * block_size + tx) // n,
                                (bx * block_size + tx) % n,
                            ]
                            + a[(bx * block_size + tx) // n, reduction_index]
                            * b[reduction_index, (bx * block_size + tx) % n]
                        )

    return matmul


def _build_matmul(m, n, k, block_size):
    target = tvm.target.Target("vortex", host="llvm")
    return tvm.tirx.build(_make_matmul(m, n, k, block_size), target=target)


def _make_two_kernel_module(size, block_size):
    @T.prim_func
    def z_scale(
        source: T.Buffer((size,), "int32"),
        destination: T.Buffer((size,), "int32"),
        factor: T.int32,
    ):
        T.func_attr({"global_symbol": "z_scale", "tirx.noalias": True})
        for bx in T.thread_binding(
            (size + block_size - 1) // block_size, thread="blockIdx.x"
        ):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                if bx * block_size + tx < size:
                    destination[bx * block_size + tx] = source[bx * block_size + tx] * factor

    @T.prim_func
    def a_increment(
        source: T.Buffer((size,), "int32"), destination: T.Buffer((size,), "int32")
    ):
        T.func_attr({"global_symbol": "a_increment", "tirx.noalias": True})
        for bx in T.thread_binding(
            (size + block_size - 1) // block_size, thread="blockIdx.x"
        ):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                if bx * block_size + tx < size:
                    destination[bx * block_size + tx] = source[bx * block_size + tx] + 1

    # Deliberately insert the lexicographically later symbol first.  Kernel IDs
    # must not depend on IRModule iteration order.
    return tvm.IRModule({"z_scale": z_scale, "a_increment": a_increment})


def _build_two_kernel_module(size, block_size):
    target = tvm.target.Target("vortex", host="llvm")
    return tvm.tirx.build(_make_two_kernel_module(size, block_size), target=target)


def test_vecadd_build_traverses_normal_tirx_pipeline():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, target):
        captured.append((source, target))
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = _build_vecadd(129, 64)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(executable.imports) == 1
    assert executable.imports[0].kind == "vortex"
    assert len(captured) == 1
    assert captured[0][1].kind.name == "vortex"
    assert "vx_spawn_threads" in captured[0][0]


def test_vecadd_rejects_oversized_thread_block(monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match=r"threadIdx\.x.*128|128.*threadIdx\.x"):
        _build_vecadd(129, 129)


def test_matmul_build_traverses_normal_tirx_pipeline():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, target):
        captured.append((source, target))
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = _build_matmul(3, 5, 7, 32)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(executable.imports) == 1
    assert executable.imports[0].kind == "vortex"
    assert len(captured) == 1
    assert captured[0][1].kind.name == "vortex"
    assert "for (int32_t reduction_index = 0; reduction_index < 7" in captured[0][0]
    assert "vx_spawn_threads" in captured[0][0]


def test_matmul_rejects_oversized_thread_block(monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match=r"threadIdx\.x.*128|128.*threadIdx\.x"):
        _build_matmul(3, 5, 7, 129)


def test_multi_kernel_build_emits_deterministic_dispatcher():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, target):
        captured.append(source)
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = _build_two_kernel_module(129, 64)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(executable.imports) == 1
    source = captured[0]
    assert "// Vortex kernel 0: a_increment_kernel\n" in source
    assert "// Vortex kernel 1: z_scale_kernel\n" in source
    assert "__tvm_vortex_kernel_0" in source
    assert "__tvm_vortex_kernel_1" in source
    assert "switch (launch->kernel_id)" in source
    assert "case 0u:" in source
    assert "case 1u:" in source
    assert "launch->num_args != 2u" in source
    assert "launch->num_args != 3u" in source
    assert "default:" in source
    assert "launch->kernel_id != 0u" not in source


def test_multi_kernel_build_rejects_duplicate_global_symbols():
    first = _make_vecadd(64, 32).with_attr("global_symbol", "duplicate")
    second = _make_vecadd(64, 32).with_attr("global_symbol", "duplicate")
    module = tvm.IRModule({"first": first, "second": second})
    build = tvm.get_global_func("target.build.vortex")

    with pytest.raises(ValueError, match="duplicate.*global symbol|global symbol.*duplicate"):
        build(module, tvm.target.Target("vortex"))


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
@pytest.mark.parametrize(
    ("size", "block_size"),
    [
        (1, 32),
        (64, 64),
        (128, 128),
        (129, 128),
        (257, 64),
    ],
)
def test_vecadd_hardware(size, block_size, vortex_hardware_environment):

    executable = _build_vecadd(size, block_size)
    device = tvm.vortex(0)
    lhs_host = np.arange(size, dtype="int32")
    rhs_host = np.arange(size, dtype="int32") * 3 - 7
    lhs = tvm.runtime.tensor(lhs_host, device=device)
    rhs = tvm.runtime.tensor(rhs_host, device=device)
    output = tvm.runtime.empty((size,), "int32", device=device)

    executable["vecadd"](lhs, rhs, output)

    np.testing.assert_array_equal(output.numpy(), lhs_host + rhs_host)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
@pytest.mark.parametrize(
    ("m", "n", "k", "block_size"),
    [
        (1, 1, 1, 32),
        (2, 3, 4, 64),
        (3, 5, 7, 128),
    ],
)
def test_matmul_hardware(m, n, k, block_size, vortex_hardware_environment):

    executable = _build_matmul(m, n, k, block_size)
    device = tvm.vortex(0)
    rng = np.random.default_rng(0)
    lhs_host = rng.uniform(-1.0, 1.0, size=(m, k)).astype("float32")
    rhs_host = rng.uniform(-1.0, 1.0, size=(k, n)).astype("float32")
    lhs = tvm.runtime.tensor(lhs_host, device=device)
    rhs = tvm.runtime.tensor(rhs_host, device=device)
    output = tvm.runtime.empty((m, n), "float32", device=device)

    executable["matmul"](lhs, rhs, output)

    np.testing.assert_allclose(output.numpy(), lhs_host @ rhs_host, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_multi_kernel_hardware_dispatches_both_functions_independently(
    vortex_hardware_environment,
):

    size = 129
    executable = _build_two_kernel_module(size, 64)
    device = tvm.vortex(0)
    source_host = np.arange(size, dtype="int32") - 17
    source = tvm.runtime.tensor(source_host, device=device)
    incremented = tvm.runtime.empty((size,), "int32", device=device)
    scaled = tvm.runtime.empty((size,), "int32", device=device)

    executable["a_increment"](source, incremented)
    executable["z_scale"](source, scaled, 3)

    np.testing.assert_array_equal(incremented.numpy(), source_host + 1)
    np.testing.assert_array_equal(scaled.numpy(), source_host * 3)


if __name__ == "__main__":
    tvm.testing.main()
