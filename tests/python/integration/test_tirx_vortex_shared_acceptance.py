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
"""Direct TIRx acceptance tests for Vortex shared-memory kernels."""

import os
import subprocess
import sys

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T

_TARGET = tvm.target.Target("vortex", host="llvm")


def _make_index_mapping_2d():
    @T.prim_func
    def index_mapping_2d(output: T.Buffer((192,), "int32")):
        T.func_attr({"global_symbol": "index_mapping_2d", "tirx.noalias": True})
        for by in T.thread_binding(2, thread="blockIdx.y"):
            for bx in T.thread_binding(3, thread="blockIdx.x"):
                for ty in T.thread_binding(4, thread="threadIdx.y"):
                    for tx in T.thread_binding(8, thread="threadIdx.x"):
                        index = (by * 3 + bx) * 32 + ty * 8 + tx
                        output[index] = index

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
                                index = (
                                    ((bz * 3 + by) * 2 + bx) * 16
                                    + (tz * 2 + ty) * 4
                                    + tx
                                )
                                output[index] = index

    return index_mapping_3d


def _make_local_scratch():
    @T.prim_func
    def local_scratch(output: T.Buffer((96,), "int32")):
        T.func_attr({"global_symbol": "local_scratch", "tirx.noalias": True})
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

    return local_scratch


def _make_shared_transpose():
    num_tiles = 5
    tile_size = 8

    @T.prim_func
    def shared_transpose(
        source: T.Buffer((num_tiles, tile_size, tile_size), "int32"),
        destination: T.Buffer((num_tiles, tile_size, tile_size), "int32"),
    ):
        T.func_attr({"global_symbol": "shared_transpose", "tirx.noalias": True})
        tile = T.alloc_buffer((tile_size, tile_size), "int32", scope="shared", align=64)
        for bx in T.thread_binding(num_tiles, thread="blockIdx.x"):
            for ty in T.thread_binding(tile_size, thread="threadIdx.y"):
                for tx in T.thread_binding(tile_size, thread="threadIdx.x"):
                    tile[ty, tx] = source[bx, ty, tx]
                    T.tvm_storage_sync("shared")
                    destination[bx, ty, tx] = tile[tx, ty]

    return shared_transpose


def _make_shared_tiled_matmul():
    m = 13
    n = 11
    k = 10
    tile_size = 8

    @T.prim_func
    def shared_tiled_matmul(
        lhs: T.Buffer((m, k), "int32"),
        rhs: T.Buffer((k, n), "int32"),
        output: T.Buffer((m, n), "int32"),
    ):
        T.func_attr({"global_symbol": "shared_tiled_matmul", "tirx.noalias": True})
        lhs_tile = T.alloc_buffer(
            (tile_size, tile_size), "int32", scope="shared", align=64
        )
        rhs_tile = T.alloc_buffer(
            (tile_size, tile_size), "int32", scope="shared", align=64
        )
        accumulator = T.alloc_buffer((1,), "int32", scope="local", align=4)
        for by in T.thread_binding(2, thread="blockIdx.y"):
            for bx in T.thread_binding(2, thread="blockIdx.x"):
                for ty in T.thread_binding(tile_size, thread="threadIdx.y"):
                    for tx in T.thread_binding(tile_size, thread="threadIdx.x"):
                        accumulator[0] = 0
                        for ko in range(2):
                            if by * tile_size + ty < m and ko * tile_size + tx < k:
                                lhs_tile[ty, tx] = lhs[
                                    by * tile_size + ty, ko * tile_size + tx
                                ]
                            else:
                                lhs_tile[ty, tx] = 0
                            if ko * tile_size + ty < k and bx * tile_size + tx < n:
                                rhs_tile[ty, tx] = rhs[
                                    ko * tile_size + ty, bx * tile_size + tx
                                ]
                            else:
                                rhs_tile[ty, tx] = 0
                            T.tvm_storage_sync("shared")
                            for ki in range(tile_size):
                                accumulator[0] = (
                                    accumulator[0] + lhs_tile[ty, ki] * rhs_tile[ki, tx]
                                )
                            T.tvm_storage_sync("shared")
                        if by * tile_size + ty < m and bx * tile_size + tx < n:
                            output[by * tile_size + ty, bx * tile_size + tx] = (
                                accumulator[0]
                            )

    return shared_tiled_matmul


def _make_two_barrier_slot_reuse():
    # A 64-thread block occupies two warps, so the pinned four-warp U55C can
    # keep two blocks resident.  Seven blocks force several LMEM slot batches.
    num_blocks = 7
    block_size = 64

    @T.prim_func
    def two_barrier_slot_reuse(output: T.Buffer((num_blocks * block_size,), "int32")):
        T.func_attr({"global_symbol": "two_barrier_slot_reuse", "tirx.noalias": True})
        scratch = T.alloc_buffer((block_size,), "int32", scope="shared", align=64)
        carried = T.alloc_buffer((1,), "int32", scope="local", align=4)
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(block_size, thread="threadIdx.x"):
                scratch[tx] = bx * 1000 + tx
                T.tvm_storage_sync("shared")
                carried[0] = scratch[block_size - 1 - tx]
                T.tvm_storage_sync("shared")
                scratch[tx] = carried[0] * 3 + 11
                output[bx * block_size + tx] = scratch[tx]

    return two_barrier_slot_reuse


def _make_non_shared_affine():
    size = 173

    @T.prim_func
    def non_shared_affine(
        source: T.Buffer((size,), "int32"),
        destination: T.Buffer((size,), "int32"),
    ):
        T.func_attr({"global_symbol": "non_shared_affine", "tirx.noalias": True})
        for bx in T.thread_binding(3, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                index = bx * 64 + tx
                if index < size:
                    destination[index] = source[index] * 5 - 9

    return non_shared_affine


def _make_mixed_shared_module():
    return tvm.IRModule(
        {
            "shared_transpose": _make_shared_transpose(),
            "non_shared_affine": _make_non_shared_affine(),
        }
    )


def _build(program):
    return tvm.tirx.build(program, target=_TARGET)


def _find_vortex_modules(module):
    return module._collect_from_import_tree(lambda imported: imported.kind == "vortex")


def test_shared_acceptance_kernels_use_normal_tirx_build_path():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(range(32)),
        override=True,
    )
    try:
        transpose = _build(_make_shared_transpose())
        matmul = _build(_make_shared_tiled_matmul())
        barrier_reuse = _build(_make_two_barrier_slot_reuse())
        mixed = _build(_make_mixed_shared_module())
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert all(
        module.kind == "llvm" for module in (transpose, matmul, barrier_reuse, mixed)
    )
    assert all(
        module.imports[0].kind == "vortex"
        for module in (transpose, matmul, barrier_reuse, mixed)
    )
    assert len(captured) == 4

    assert captured[0].count("__syncthreads()") == 1
    assert "__local_mem(256)" in captured[0]
    assert captured[1].count("__syncthreads()") == 2
    assert "__local_mem(512)" in captured[1]
    assert captured[2].count("__syncthreads()") == 2
    assert "vx_spawn_threads(3, launch->grid, launch->block" in captured[2]

    mixed_source = captured[3]
    assert "// Vortex kernel 0: non_shared_affine_kernel" in mixed_source
    assert "// Vortex kernel 1: shared_transpose_kernel" in mixed_source
    assert mixed_source.count("__local_mem(256)") == 1
    metadata = mixed.imports[0]["vortex.get_kernel_resource_metadata"]()
    assert set(metadata) == {
        "non_shared_affine_kernel",
        "shared_transpose_kernel",
    }
    assert list(metadata["non_shared_affine_kernel"])[1] == 0
    assert list(metadata["shared_transpose_kernel"])[1] == 256


def test_resource_metadata_survives_export_library_and_reload(tmp_path):
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: bytearray(range(32)),
        override=True,
    )
    try:
        executable = _build(_make_mixed_shared_module())
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    artifact = tmp_path / "mixed_shared.so"
    executable.export_library(str(artifact))
    restored = tvm.runtime.load_module(str(artifact))
    [device_module] = _find_vortex_modules(restored)
    metadata = device_module["vortex.get_kernel_resource_metadata"]()

    assert list(metadata["non_shared_affine_kernel"]) == [1, 0, 2, 4, 64, 1, 1, 0]
    assert list(metadata["shared_transpose_kernel"]) == [2, 256, 2, 0, 8, 8, 1, 1]


def _execute_hardware_case(case):
    """Build and execute one acceptance case in a fresh process."""

    device = tvm.vortex(0)
    if case == "mapping_2d":
        executable = _build(_make_index_mapping_2d())
        output = tvm.runtime.empty((192,), "int32", device=device)
        executable["index_mapping_2d"](output)
        actual = output.numpy()
        expected = np.arange(192, dtype="int32")
    elif case == "mapping_3d":
        executable = _build(_make_index_mapping_3d())
        output = tvm.runtime.empty((192,), "int32", device=device)
        executable["index_mapping_3d"](output)
        actual = output.numpy()
        expected = np.arange(192, dtype="int32")
    elif case == "local_scratch":
        executable = _build(_make_local_scratch())
        output = tvm.runtime.empty((96,), "int32", device=device)
        executable["local_scratch"](output)
        actual = output.numpy()
        expected = np.empty((96,), dtype="int32")
        for linear_block in range(6):
            for linear_thread in range(16):
                expected[linear_block * 16 + linear_thread] = (
                    linear_block * 1000 + linear_thread + 101
                ) * 3 - linear_thread
    elif case == "transpose":
        executable = _build(_make_shared_transpose())
        source_host = np.arange(5 * 8 * 8, dtype="int32").reshape(5, 8, 8)
        source = tvm.runtime.tensor(source_host, device=device)
        output = tvm.runtime.empty(source_host.shape, "int32", device=device)
        executable["shared_transpose"](source, output)
        actual = output.numpy()
        expected = source_host.transpose(0, 2, 1)
    elif case == "matmul":
        executable = _build(_make_shared_tiled_matmul())
        lhs_host = (np.arange(13 * 10, dtype="int32").reshape(13, 10) % 9) - 4
        rhs_host = (np.arange(10 * 11, dtype="int32").reshape(10, 11) % 7) - 3
        lhs = tvm.runtime.tensor(lhs_host, device=device)
        rhs = tvm.runtime.tensor(rhs_host, device=device)
        output = tvm.runtime.empty((13, 11), "int32", device=device)
        executable["shared_tiled_matmul"](lhs, rhs, output)
        actual = output.numpy()
        expected = lhs_host @ rhs_host
    elif case == "two_barrier_slot_reuse":
        executable = _build(_make_two_barrier_slot_reuse())
        output = tvm.runtime.empty((7 * 64,), "int32", device=device)
        executable["two_barrier_slot_reuse"](output)
        actual = output.numpy()
        expected = np.concatenate(
            [
                (block * 1000 + np.arange(63, -1, -1, dtype="int32")) * 3 + 11
                for block in range(7)
            ]
        )
    elif case == "mixed_module":
        executable = _build(_make_mixed_shared_module())
        transpose_source_host = np.arange(5 * 8 * 8, dtype="int32").reshape(5, 8, 8)
        transpose_source = tvm.runtime.tensor(transpose_source_host, device=device)
        transpose_output = tvm.runtime.empty(
            transpose_source_host.shape, "int32", device=device
        )
        affine_source_host = np.arange(173, dtype="int32") - 37
        affine_source = tvm.runtime.tensor(affine_source_host, device=device)
        affine_output = tvm.runtime.empty((173,), "int32", device=device)
        executable["shared_transpose"](transpose_source, transpose_output)
        executable["non_shared_affine"](affine_source, affine_output)
        actual = (transpose_output.numpy(), affine_output.numpy())
        expected = (
            transpose_source_host.transpose(0, 2, 1),
            affine_source_host * 5 - 9,
        )
    else:
        raise ValueError(f"unknown Vortex shared acceptance case: {case}")

    if isinstance(actual, tuple):
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value, expected_value)
    else:
        np.testing.assert_array_equal(actual, expected)


def _run_hardware_case_with_timeout(case, timeout_seconds=240):
    # A standalone interpreter avoids inheriting an XRT handle and performs a
    # normal shutdown, which emits Vortex PERF counters for each case.
    environment = os.environ.copy()
    environment["TVM_VORTEX_ACCEPTANCE_WORKER"] = case
    try:
        completed = subprocess.run(
            [sys.executable, __file__],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Vortex {case} hardware test exceeded {timeout_seconds} seconds: {error}"
        )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    assert completed.returncode == 0, f"Vortex {case} worker failed"


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
@pytest.mark.parametrize(
    "case",
    [
        "mapping_2d",
        "mapping_3d",
        "local_scratch",
        "transpose",
        "matmul",
        "two_barrier_slot_reuse",
        "mixed_module",
    ],
)
def test_shared_acceptance_hardware(case, vortex_hardware_environment):
    del vortex_hardware_environment
    _run_hardware_case_with_timeout(case)


if __name__ == "__main__":
    worker_case = os.environ.get("TVM_VORTEX_ACCEPTANCE_WORKER")
    if worker_case:
        _execute_hardware_case(worker_case)
    else:
        tvm.testing.main()
