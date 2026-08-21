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

import multiprocessing
import os
import traceback

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
                    c[(bx * block_size + tx) // n, (bx * block_size + tx) % n] = (
                        T.float32(0)
                    )
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
                    destination[bx * block_size + tx] = (
                        source[bx * block_size + tx] * factor
                    )

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


def _make_index_mapping_2d():
    @T.prim_func
    def index_mapping_2d(output: T.Buffer((192,), "int32")):
        T.func_attr({"global_symbol": "index_mapping_2d", "tirx.noalias": True})
        for by in T.thread_binding(2, thread="blockIdx.y"):
            for bx in T.thread_binding(3, thread="blockIdx.x"):
                for ty in T.thread_binding(4, thread="threadIdx.y"):
                    for tx in T.thread_binding(8, thread="threadIdx.x"):
                        output[(by * 3 + bx) * 32 + ty * 8 + tx] = (
                            (by * 3 + bx) * 32 + ty * 8 + tx
                        )

    return index_mapping_2d


def _make_index_mapping_3d():
    @T.prim_func
    def index_mapping_3d(output: T.Buffer((192,), "int32")):
        T.func_attr({"global_symbol": "index_mapping_3d", "tirx.noalias": True})
        for bz in T.thread_binding(2, thread="blockIdx.z"):
            for by in T.thread_binding(3, thread="blockIdx.y"):
                for bx in T.thread_binding(2, thread="blockIdx.x"):
                    for tz in T.thread_binding(2, thread="threadIdx.z"):
                        for ty in T.thread_binding(2, thread="threadIdx.y"):
                            for tx in T.thread_binding(4, thread="threadIdx.x"):
                                output[
                                    ((bz * 3 + by) * 2 + bx) * 16
                                    + (tz * 2 + ty) * 4
                                    + tx
                                ] = (
                                    ((bz * 3 + by) * 2 + bx) * 16
                                    + (tz * 2 + ty) * 4
                                    + tx
                                )

    return index_mapping_3d


def _build_index_mapping(func):
    return tvm.tirx.build(func, target=tvm.target.Target("vortex", host="llvm"))


def _make_local_scratch_2d():
    @T.prim_func
    def local_scratch_2d(output: T.Buffer((96,), "int32")):
        T.func_attr({"global_symbol": "local_scratch_2d", "tirx.noalias": True})
        for by in T.thread_binding(3, thread="blockIdx.y"):
            for bx in T.thread_binding(2, thread="blockIdx.x"):
                for ty in T.thread_binding(2, thread="threadIdx.y"):
                    for tx in T.thread_binding(8, thread="threadIdx.x"):
                        scratch = T.alloc_buffer((4,), "int32", scope="local", align=16)
                        linear_thread = ty * 8 + tx
                        linear_block = by * 2 + bx
                        scratch[0] = linear_block * 1000 + linear_thread
                        scratch[1] = scratch[0] + 101
                        scratch[2] = scratch[1] * 3
                        scratch[3] = scratch[2] - linear_thread
                        output[linear_block * 16 + linear_thread] = scratch[3]

    return local_scratch_2d


def _make_shared_arena_no_barrier():
    @T.prim_func
    def shared_arena_no_barrier(output: T.Buffer((192,), "int32")):
        T.func_attr({"global_symbol": "shared_arena_no_barrier", "tirx.noalias": True})
        tags = T.alloc_buffer((32,), "uint8", scope="shared", align=4)
        values = T.alloc_buffer((32,), "int32", scope="shared", align=64)
        for bx in T.thread_binding(6, thread="blockIdx.x"):
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                tags[tx] = T.Cast("uint8", tx)
                values[tx] = bx * 1000 + tx
                output[bx * 32 + tx] = values[tx] + T.Cast("int32", tags[tx])

    return shared_arena_no_barrier


def _make_dynamic_shared():
    @T.prim_func
    def dynamic_shared(output: T.Buffer((32,), "int32")):
        T.func_attr({"global_symbol": "dynamic_shared"})
        scratch = T.alloc_buffer((32,), "int32", scope="shared.dyn")
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            scratch[tx] = tx
            output[tx] = scratch[tx]

    return dynamic_shared


def _make_cross_warp_shared_barrier(block_size):
    num_blocks = 5

    @T.prim_func
    def cross_warp_shared_barrier(
        output: T.Buffer((num_blocks * block_size,), "int32")
    ):
        T.func_attr(
            {"global_symbol": "cross_warp_shared_barrier", "tirx.noalias": True}
        )
        scratch = T.alloc_buffer((block_size,), "int32", scope="shared", align=64)
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                scratch[tx] = bx * 1000 + tx
                T.tvm_storage_sync("shared")
                output[bx * block_size + tx] = scratch[block_size - 1 - tx]

    return cross_warp_shared_barrier


def _make_consecutive_shared_barrier_reuse():
    block_size = 64
    num_blocks = 5

    @T.prim_func
    def consecutive_shared_barrier_reuse(
        output: T.Buffer((num_blocks * block_size,), "int32")
    ):
        T.func_attr(
            {
                "global_symbol": "consecutive_shared_barrier_reuse",
                "tirx.noalias": True,
            }
        )
        scratch = T.alloc_buffer((block_size,), "int32", scope="shared", align=64)
        carried = T.alloc_buffer((1,), "int32", scope="local", align=4)
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                scratch[tx] = bx * 1000 + tx
                T.tvm_storage_sync("shared")
                if tx < 32:
                    carried[0] = scratch[tx + 32]
                else:
                    carried[0] = scratch[tx - 32]
                T.tvm_storage_sync("shared")
                T.tvm_storage_sync("shared")
                scratch[tx] = carried[0] * 2 + 7
                T.tvm_storage_sync("shared")
                output[bx * block_size + tx] = scratch[block_size - 1 - tx]

    return consecutive_shared_barrier_reuse


def _shared_barrier_hardware_worker(case, block_size, connection):
    """Run a possibly hanging barrier kernel in a killable child process."""

    try:
        if case == "cross_warp":
            func = _make_cross_warp_shared_barrier(block_size)
            global_symbol = "cross_warp_shared_barrier"
        elif case == "consecutive_reuse":
            func = _make_consecutive_shared_barrier_reuse()
            global_symbol = "consecutive_shared_barrier_reuse"
        else:
            raise ValueError(f"unknown shared-barrier hardware case: {case}")

        executable = _build_index_mapping(func)
        device = tvm.vortex(0)
        output = tvm.runtime.empty((5 * block_size,), "int32", device=device)
        executable[global_symbol](output)
        connection.send(("ok", output.numpy()))
    except BaseException:  # pylint: disable=broad-exception-caught
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _get_fresh_multiprocessing_context():
    """Return a context that does not inherit initialized TVM or XRT state."""

    return multiprocessing.get_context("spawn")


def _run_shared_barrier_hardware_with_timeout(case, block_size, timeout_seconds=180):
    context = _get_fresh_multiprocessing_context()
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_shared_barrier_hardware_worker,
        args=(case, block_size, child_connection),
    )
    process.start()
    child_connection.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail(
            f"Vortex {case} barrier hardware test exceeded {timeout_seconds} seconds"
        )
    if not parent_connection.poll(5):
        pytest.fail(
            f"Vortex {case} barrier worker exited with code {process.exitcode} without a result"
        )
    status, result = parent_connection.recv()
    parent_connection.close()
    if status == "error":
        pytest.fail(result)
    assert process.exitcode == 0
    return result


def test_shared_barrier_hardware_worker_runs_in_fresh_spawn_process():
    context = _get_fresh_multiprocessing_context()
    assert context.get_start_method() == "spawn"

    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_shared_barrier_hardware_worker,
        args=("invalid-test-case", 1, child_connection),
    )
    process.start()
    child_connection.close()
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail("spawned Vortex barrier worker did not exit within 30 seconds")

    assert parent_connection.poll(5)
    status, result = parent_connection.recv()
    parent_connection.close()
    assert status == "error"
    assert "unknown shared-barrier hardware case: invalid-test-case" in result
    assert process.exitcode == 0


@pytest.mark.parametrize(
    ("make_kernel", "required_tags"),
    [
        (
            _make_index_mapping_2d,
            ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"),
        ),
        (
            _make_index_mapping_3d,
            (
                "blockIdx.x",
                "blockIdx.y",
                "blockIdx.z",
                "threadIdx.x",
                "threadIdx.y",
                "threadIdx.z",
            ),
        ),
    ],
)
def test_multidimensional_index_mapping_traverses_normal_tirx_pipeline(
    make_kernel, required_tags
):
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(range(32)),
        override=True,
    )
    try:
        executable = _build_index_mapping(make_kernel())
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(executable.imports) == 1
    assert executable.imports[0].kind == "vortex"
    assert len(captured) == 1
    for tag in required_tags:
        assert tag in captured[0]
    assert "vx_spawn_threads(3, launch->grid, launch->block" in captured[0]


def test_2d_local_scratch_traverses_normal_tirx_pipeline():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(range(32)),
        override=True,
    )
    try:
        executable = _build_index_mapping(_make_local_scratch_2d())
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(captured) == 1
    source = captured[0]
    assert "alignas(16) int32_t scratch_ptr[4];" in source
    signature = next(
        line
        for line in source.splitlines()
        if line.startswith("static void __tvm_vortex_kernel_0(")
    )
    assert "scratch_ptr" not in signature


def test_shared_arena_without_barrier_traverses_normal_tirx_pipeline():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(range(32)),
        override=True,
    )
    try:
        executable = _build_index_mapping(_make_shared_arena_no_barrier())
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert executable.kind == "llvm"
    assert len(captured) == 1
    assert "__local_mem(192)" in captured[0]
    assert "uint8_t* tags_ptr" in captured[0]
    assert "int32_t* values_ptr" in captured[0]
    metadata = executable.imports[0]["vortex.get_kernel_resource_metadata"]()
    assert list(metadata["shared_arena_no_barrier_kernel"]) == [
        1,
        192,
        4,
        0,
        32,
        1,
        1,
        0,
    ]


def test_dynamic_shared_is_rejected_by_normal_tirx_pipeline():
    with pytest.raises(ValueError, match="dynamic shared memory is not supported"):
        _build_index_mapping(_make_dynamic_shared())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("product", r"129 threads.*max_threads_per_block 128"),
        ("dynamic", r"threadIdx\.x extent must be a compile-time constant"),
        ("zero", r"threadIdx\.x extent must be positive"),
        ("unknown", r"unknown thread binding tag.*threadIdx\.w"),
    ],
)
def test_invalid_multidimensional_bindings_fail_through_normal_tirx_pipeline(
    failure, message
):
    if failure == "product":

        @T.prim_func
        def bad(output: T.Buffer((1,), "int32")):
            T.func_attr({"global_symbol": "bad"})
            for ty in T.thread_binding(3, thread="threadIdx.y"):
                for tx in T.thread_binding(43, thread="threadIdx.x"):
                    if tx + ty == 0:
                        output[0] = 0

    elif failure == "dynamic":

        @T.prim_func
        def bad(n: T.int32, output: T.handle):
            T.func_attr({"global_symbol": "bad"})
            buffer = T.match_buffer(output, (n,), "int32")
            for tx in T.thread_binding(n, thread="threadIdx.x"):
                buffer[tx] = 0

    elif failure == "zero":

        @T.prim_func
        def bad(output: T.Buffer((1,), "int32")):
            T.func_attr({"global_symbol": "bad"})
            for tx in T.thread_binding(0, thread="threadIdx.x"):
                output[0] = tx

    else:

        @T.prim_func
        def bad(output: T.Buffer((1,), "int32")):
            T.func_attr({"global_symbol": "bad"})
            for tw in T.thread_binding(1, thread="threadIdx.w"):
                output[0] = tw

    with pytest.raises(ValueError, match=message):
        _build_index_mapping(bad)


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

    with pytest.raises(
        ValueError, match="duplicate.*global symbol|global symbol.*duplicate"
    ):
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

    np.testing.assert_allclose(
        output.numpy(), lhs_host @ rhs_host, rtol=1e-5, atol=1e-5
    )


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


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
@pytest.mark.parametrize(
    ("make_kernel", "global_symbol"),
    [
        (_make_index_mapping_2d, "index_mapping_2d"),
        (_make_index_mapping_3d, "index_mapping_3d"),
    ],
)
def test_multidimensional_index_mapping_hardware(
    make_kernel, global_symbol, vortex_hardware_environment
):
    executable = _build_index_mapping(make_kernel())
    device = tvm.vortex(0)
    output = tvm.runtime.empty((192,), "int32", device=device)

    executable[global_symbol](output)

    np.testing.assert_array_equal(output.numpy(), np.arange(192, dtype="int32"))


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_2d_local_scratch_is_thread_private_on_hardware(vortex_hardware_environment):
    executable = _build_index_mapping(_make_local_scratch_2d())
    device = tvm.vortex(0)
    output = tvm.runtime.empty((96,), "int32", device=device)

    executable["local_scratch_2d"](output)

    expected = np.empty((96,), dtype="int32")
    for linear_block in range(6):
        for linear_thread in range(16):
            expected[linear_block * 16 + linear_thread] = (
                linear_block * 1000 + linear_thread + 101
            ) * 3 - linear_thread
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_static_shared_arena_is_block_isolated_on_hardware(
    vortex_hardware_environment,
):
    executable = _build_index_mapping(_make_shared_arena_no_barrier())
    device = tvm.vortex(0)
    output = tvm.runtime.empty((192,), "int32", device=device)

    executable["shared_arena_no_barrier"](output)

    expected = np.concatenate(
        [np.arange(32, dtype="int32") * 2 + block * 1000 for block in range(6)]
    )
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
@pytest.mark.parametrize("block_size", [64, 128])
def test_cross_warp_shared_barrier_hardware(block_size, vortex_hardware_environment):
    del vortex_hardware_environment
    actual = _run_shared_barrier_hardware_with_timeout("cross_warp", block_size)
    expected = np.concatenate(
        [
            block * 1000 + np.arange(block_size - 1, -1, -1, dtype="int32")
            for block in range(5)
        ]
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_consecutive_barriers_reuse_shared_memory_on_hardware(
    vortex_hardware_environment,
):
    del vortex_hardware_environment
    block_size = 64
    actual = _run_shared_barrier_hardware_with_timeout("consecutive_reuse", block_size)
    expected = np.empty(5 * block_size, dtype="int32")
    for block in range(5):
        for tx in range(block_size):
            source_thread = (block_size - 1 - tx + 32) % block_size
            expected[block * block_size + tx] = (block * 1000 + source_thread) * 2 + 7
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_target_lmem_larger_than_actual_hardware_is_rejected_before_launch(
    vortex_hardware_environment,
):
    target = tvm.target.Target(
        {"kind": "vortex", "local_mem_size": 2 << 20}, host="llvm"
    )
    executable = tvm.tirx.build(_make_vecadd(32, 32), target=target)
    device = tvm.vortex(0)
    lhs = tvm.runtime.tensor(np.arange(32, dtype="int32"), device=device)
    rhs = tvm.runtime.tensor(np.arange(32, dtype="int32"), device=device)
    output = tvm.runtime.empty((32,), "int32", device=device)

    with pytest.raises(
        ValueError,
        match=r"target local_mem_size 2097152 exceeds actual VX_CAPS_LOCAL_MEM_SIZE 1048576",
    ):
        executable["vecadd"](lhs, rhs, output)


if __name__ == "__main__":
    tvm.testing.main()
