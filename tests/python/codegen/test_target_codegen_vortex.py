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

import pytest
import tvm_ffi

import tvm
from tvm.script import tirx as T


@T.prim_func
def vecadd(
    a: T.Buffer((256,), "int32"),
    b: T.Buffer((256,), "int32"),
    c: T.Buffer((256,), "float32"),
):
    T.func_attr({"global_symbol": "vecadd", "tirx.noalias": True})
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            if bx * 128 + tx < 256:
                c[bx * 128 + tx] = T.Cast(
                    "float32", a[bx * 128 + tx] + b[bx * 128 + tx]
                )


def _build_source(func=vecadd, target=None):
    target = (
        tvm.target.Target("vortex") if target is None else tvm.target.Target(target)
    )
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    return tvm.get_global_func("target.build.vortex")(mod, target).inspect_source()


def _build_source_with_stub_compiler(func, target=None):
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name, allow_missing=True)
    tvm.register_global_func(
        callback_name, lambda source, unused_target: bytearray(), override=True
    )
    try:
        return _build_source(func, target)
    finally:
        if previous is None:
            tvm_ffi.registry.remove_global_func(callback_name)
        else:
            tvm.register_global_func(callback_name, previous, override=True)


def test_vecadd_source_and_compile_callback():
    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name, allow_missing=True)

    def capture(source, target):
        captured.append((source, target.kind.name))
        return bytearray()

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        source = _build_source()
    finally:
        if previous is None:
            tvm_ffi.registry.remove_global_func(callback_name)
        else:
            tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [(source, "vortex")]
    assert "#include <vx_tvm_abi.h>" in source
    assert "static void __tvm_vortex_kernel_0(" in source
    assert "threadIdx.x" in source
    assert "blockIdx.x" in source
    assert "((float)" in source
    assert "reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args[0]))" in source
    assert "reinterpret_cast<float*>(static_cast<uintptr_t>(args[2]))" in source
    assert "launch->abi_version != VX_TVM_ABI_VERSION" in source
    assert "launch->num_args != 3u" in source
    assert "switch (launch->kernel_id)" in source
    assert "case 0u:" in source
    assert "csr_read(VX_CSR_MSCRATCH)" in source
    assert "vx_spawn_threads(3, launch->grid, launch->block" in source


def test_supports_2d_and_3d_thread_dimensions():
    @T.prim_func
    def multidimensional(a: T.Buffer((24,), "int32")):
        T.func_attr({"global_symbol": "multidimensional"})
        for bz in T.thread_binding(2, thread="blockIdx.z"):
            for by in T.thread_binding(3, thread="blockIdx.y"):
                for bx in T.thread_binding(2, thread="blockIdx.x"):
                    for tz in T.thread_binding(2, thread="threadIdx.z"):
                        for ty in T.thread_binding(2, thread="threadIdx.y"):
                            for tx in T.thread_binding(2, thread="threadIdx.x"):
                                if bz + by + bx + tz + ty + tx == 0:
                                    a[0] = 1

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(multidimensional)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    for tag in [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ]:
        assert tag in source
    assert "vx_spawn_threads(3, launch->grid, launch->block" in source


def test_rejects_unknown_thread_dimension():
    @T.prim_func
    def bad(a: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for tw in T.thread_binding(4, thread="threadIdx.w"):
            a[tw] = 0.0

    with pytest.raises(ValueError, match=r"unknown thread binding tag.*threadIdx\.w"):
        _build_source(bad)


@pytest.mark.parametrize("tag", ["threadIdx.x", "blockIdx.x"])
def test_rejects_duplicate_thread_axis(tag):
    @T.prim_func
    def bad(a: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for tx_outer in T.thread_binding(2, thread=tag):
            for tx_inner in T.thread_binding(2, thread=tag):
                a[tx_outer * 2 + tx_inner] = 0.0

    with pytest.raises(ValueError, match=rf"{tag}.*bound more than once"):
        _build_source(bad)


@pytest.mark.parametrize(("x", "y", "z"), [(43, 3, 1), (3, 43, 1), (3, 1, 43)])
def test_rejects_thread_block_product_over_limit(x, y, z):
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for tz in T.thread_binding(z, thread="threadIdx.z"):
            for ty in T.thread_binding(y, thread="threadIdx.y"):
                for tx in T.thread_binding(x, thread="threadIdx.x"):
                    if tx + ty + tz == 0:
                        a[0] = 0.0

    with pytest.raises(ValueError, match=r"129 threads.*max_threads_per_block 128"):
        _build_source(bad)


@pytest.mark.parametrize(
    ("tag", "limit_name"),
    [
        ("threadIdx.x", "max_block_size_x"),
        ("threadIdx.y", "max_block_size_y"),
        ("threadIdx.z", "max_block_size_z"),
    ],
)
def test_rejects_per_axis_thread_limit(tag, limit_name):
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for thread_index in T.thread_binding(129, thread=tag):
            if thread_index == 0:
                a[0] = 0.0

    with pytest.raises(ValueError, match=rf"{tag} extent 129 exceeds {limit_name} 128"):
        _build_source(bad)


def test_rejects_dynamic_thread_extent():
    @T.prim_func
    def bad(n: T.int32, a: T.handle):
        T.func_attr({"global_symbol": "bad"})
        buffer = T.match_buffer(a, (n,), "float32")
        for tx in T.thread_binding(n, thread="threadIdx.x"):
            buffer[tx] = 0.0

    with pytest.raises(
        ValueError, match=r"threadIdx\.x extent must be a compile-time constant"
    ):
        _build_source(bad)


def test_rejects_zero_thread_extent():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for tx in T.thread_binding(0, thread="threadIdx.x"):
            a[0] = T.Cast("float32", tx)

    with pytest.raises(ValueError, match=r"threadIdx\.x extent must be positive"):
        _build_source(bad)


@pytest.mark.parametrize("extent", [0, 1 << 32])
def test_rejects_static_block_extent_outside_uint32(extent):
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for bx in T.thread_binding(extent, thread="blockIdx.x"):
            if bx == 0:
                a[0] = 0.0

    with pytest.raises(ValueError, match=r"blockIdx\.x extent.*positive uint32"):
        _build_source(bad)


def test_accepts_dynamic_block_extent_for_runtime_validation():
    @T.prim_func
    def dynamic_grid(n: T.int32, a: T.handle):
        T.func_attr({"global_symbol": "dynamic_grid"})
        buffer = T.match_buffer(a, (n,), "float32")
        for bx in T.thread_binding(n, thread="blockIdx.x"):
            buffer[bx] = 0.0

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(dynamic_grid)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert "blockIdx.x" in source


def test_supports_serial_loop():
    @T.prim_func
    def serial(a: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "serial"})
        for i in range(4):
            a[i] = 0.0

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)

    def capture(source, target):
        captured.append(source)
        return bytearray()

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        source = _build_source(serial)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert "for (int32_t i = 0; i < 4; ++i)" in source


def test_supports_max_expression():
    @T.prim_func
    def maximum(a: T.Buffer((1,), "float32"), out: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "maximum"})
        for tx in T.thread_binding(1, thread="threadIdx.x"):
            out[tx] = T.max(a[tx], T.float32(0))

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(maximum)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert "static inline T __tvm_vortex_max(T a, T b)" in source
    assert "__tvm_vortex_max(a[((int32_t)threadIdx.x)], 0.000000e+00f)" in source


def test_rejects_non_serial_cpu_loop():
    @T.prim_func
    def bad(a: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for i in T.vectorized(4):
            a[i] = 0.0

    with pytest.raises(ValueError, match="only serial and thread-bound loops"):
        _build_source(bad)


def test_supports_constant_thread_private_local_allocation():
    @T.prim_func
    def local_scratch(a: T.Buffer((32,), "int32")):
        T.func_attr({"global_symbol": "local_scratch"})
        temporary = T.alloc_buffer(
            (8,),
            "int32",
            scope="local",
            align=32,
            annotations={"tirx.volatile": True},
        )
        for ty in T.thread_binding(4, thread="threadIdx.y"):
            for tx in T.thread_binding(8, thread="threadIdx.x"):
                temporary[tx] = ty * 8 + tx
                a[ty * 8 + tx] = temporary[tx]

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(local_scratch)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert "alignas(32) int32_t temporary_ptr[8];" in source
    assert "((volatile int32_t*)temporary_ptr)" in source
    assert "__local_mem(" not in source


def test_rejects_dynamic_local_allocation():
    @T.prim_func
    def bad(n: T.int32, a: T.handle):
        T.func_attr({"global_symbol": "bad"})
        output = T.match_buffer(a, (n,), "float32")
        temporary = T.alloc_buffer((n,), "float32", scope="local")
        temporary[0] = 1.0
        output[0] = temporary[0]

    with pytest.raises(ValueError, match=r"local buffer.*compile-time constant shape"):
        _build_source(bad)


@pytest.mark.parametrize("extent", [0, -1])
def test_rejects_nonpositive_local_allocation(extent):
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((extent,), "float32", scope="local")
        temporary[0] = a[0]

    with pytest.raises(ValueError, match=r"local buffer.*extent.*must be positive"):
        _build_source(bad)


def test_rejects_local_allocation_shape_overflow():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((1 << 62, 8), "uint8", scope="local")
        temporary[0, 0] = T.uint8(0)

    with pytest.raises(ValueError, match=r"local buffer.*shape product overflows"):
        _build_source(bad)


def test_rejects_local_allocation_byte_overflow():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((1 << 62,), "float64", scope="local")
        temporary[0] = T.float64(0)

    with pytest.raises(ValueError, match=r"local buffer.*byte size overflows"):
        _build_source(bad)


def test_rejects_local_allocation_over_per_thread_limit():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        first = T.alloc_buffer((9,), "int32", scope="local", align=16)
        second = T.alloc_buffer((8,), "int32", scope="local", align=16)
        first[0] = 0
        second[0] = 0

    target = {"kind": "vortex", "max_local_memory_per_thread": 64}
    with pytest.raises(
        ValueError,
        match=r"local allocations require 80 bytes.*max_local_memory_per_thread 64",
    ):
        _build_source(bad, target)


@pytest.mark.parametrize("alignment", [3, 1 << 30])
def test_rejects_unrepresentable_local_alignment(alignment):
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((1,), "float32", scope="local", align=alignment)
        temporary[0] = a[0]

    with pytest.raises(ValueError, match=r"local buffer.*alignment"):
        _build_source(bad)


def test_static_shared_buffers_use_one_aligned_lmem_arena():
    @T.prim_func
    def shared_arena(a: T.Buffer((2,), "int32")):
        T.func_attr({"global_symbol": "shared_arena"})
        first = T.alloc_buffer((3,), "uint8", scope="shared", align=4)
        second = T.alloc_buffer((2,), "int32", scope="shared", align=16)
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            if tx == 0:
                first[0] = T.uint8(7)
                second[0] = 11
                a[0] = T.Cast("int32", first[0])
                a[1] = second[0]

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(shared_arena)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert source.count("__local_mem(32)") == 1
    assert (
        "uint8_t* first_ptr = reinterpret_cast<uint8_t*>(__tvm_vortex_shared_base + 0);"
        in source
    )
    assert (
        "int32_t* second_ptr = reinterpret_cast<int32_t*>(__tvm_vortex_shared_base + 16);"
        in source
    )


def test_shared_barriers_emit_for_consecutive_sites_and_uniform_serial_loop():
    @T.prim_func
    def uniform_barriers(a: T.Buffer((128,), "int32")):
        T.func_attr({"global_symbol": "uniform_barriers"})
        scratch = T.alloc_buffer((64,), "int32", scope="shared")
        for bx in T.thread_binding(2, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                scratch[tx] = bx * 1000 + tx
                T.tvm_storage_sync("shared")
                T.tvm_storage_sync("shared")
                for iteration in range(2):
                    T.tvm_storage_sync("shared")
                    scratch[tx] = scratch[tx] + iteration
                a[bx * 64 + tx] = scratch[63 - tx]

    source = _build_source_with_stub_compiler(uniform_barriers)

    assert source.count("__syncthreads();") == 3
    assert "for (int32_t iteration = 0; iteration < 2" in source


def test_shared_barrier_accepts_block_uniform_condition():
    @T.prim_func
    def block_uniform(a: T.Buffer((128,), "int32")):
        T.func_attr({"global_symbol": "block_uniform"})
        for bx in T.thread_binding(2, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                if bx < 2:
                    T.tvm_storage_sync("shared")
                    a[bx * 64 + tx] = tx

    source = _build_source_with_stub_compiler(block_uniform)

    assert source.count("__syncthreads();") == 1


@pytest.mark.parametrize("block_threads", [32, 64])
def test_shared_barrier_rejects_thread_dependent_condition(block_threads):
    @T.prim_func
    def divergent(a: T.Buffer((block_threads,), "int32")):
        T.func_attr({"global_symbol": "divergent"})
        for tx in T.thread_binding(block_threads, thread="threadIdx.x"):
            take_barrier = tx < block_threads // 2
            if take_barrier:
                T.tvm_storage_sync("shared")
                a[tx] = tx

    with pytest.raises(
        ValueError, match=r"shared barrier.*thread-dependent.*condition"
    ):
        _build_source(divergent)


def test_shared_barrier_rejects_thread_dependent_serial_loop_count():
    @T.prim_func
    def divergent_loop(a: T.Buffer((64,), "int32")):
        T.func_attr({"global_symbol": "divergent_loop"})
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            for iteration in range(tx + 1):
                T.tvm_storage_sync("shared")
                a[tx] = iteration

    with pytest.raises(
        ValueError, match=r"shared barrier.*thread-dependent serial loop"
    ):
        _build_source(divergent_loop)


def test_shared_barrier_rejects_thread_dependent_early_exit():
    @T.prim_func
    def divergent_exit(a: T.Buffer((64,), "int32")):
        T.func_attr({"global_symbol": "divergent_exit"})
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            for iteration in range(2):
                if tx == 0:
                    break
                T.tvm_storage_sync("shared")
                a[tx] = iteration

    with pytest.raises(
        ValueError, match=r"shared barrier.*thread-dependent early exit"
    ):
        _build_source(divergent_exit)


def test_shared_barrier_rejects_early_exit_in_thread_dependent_serial_loop():
    @T.prim_func
    def divergent_loop_exit(a: T.Buffer((64,), "int32")):
        T.func_attr({"global_symbol": "divergent_loop_exit"})
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            for iteration in range(tx + 1):
                break
            T.tvm_storage_sync("shared")
            a[tx] = tx

    with pytest.raises(ValueError, match=r"early exit.*thread-dependent serial loop"):
        _build_source(divergent_loop_exit)


def test_shared_barrier_rejects_assert_in_thread_dependent_condition():
    @T.prim_func
    def divergent_assert(a: T.Buffer((64,), "int32")):
        T.func_attr({"global_symbol": "divergent_assert"})
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            if tx == 0:
                with T.Assert(T.bool(False), "divergent assertion"):
                    T.evaluate(0)
            T.tvm_storage_sync("shared")
            a[tx] = tx

    with pytest.raises(ValueError, match=r"thread-dependent early exit \(assert\)"):
        _build_source(divergent_assert)


def test_rejects_reserved_resource_metadata_global_symbol():
    reserved = vecadd.with_attr("global_symbol", "vortex.get_kernel_resource_metadata")
    with pytest.raises(ValueError, match=r"global symbol.*reserved"):
        _build_source(reserved)


@pytest.mark.parametrize("scope", ["global", "warp", "shared.dyn", "mystery"])
def test_rejects_unsupported_storage_sync_scope(scope):
    @T.prim_func
    def unsupported_sync(a: T.Buffer((64,), "int32")):
        T.func_attr({"global_symbol": "unsupported_sync"})
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            T.tvm_storage_sync(scope)
            a[tx] = tx

    with pytest.raises(ValueError, match=r"storage sync scope"):
        _build_source(unsupported_sync)


@pytest.mark.parametrize(
    ("block_threads", "shared_bytes"),
    [(32, 256 << 10), (64, 512 << 10), (128, 1 << 20)],
)
def test_static_shared_memory_accepts_dimension_dependent_boundary(
    block_threads, shared_bytes
):
    @T.prim_func
    def boundary(a: T.Buffer((1,), "uint8")):
        T.func_attr({"global_symbol": "boundary"})
        scratch = T.alloc_buffer((shared_bytes,), "uint8", scope="shared", align=1)
        for tx in T.thread_binding(block_threads, thread="threadIdx.x"):
            if tx == 0:
                scratch[0] = T.uint8(1)
                a[0] = scratch[0]

    captured = []
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name,
        lambda source, unused_target: captured.append(source) or bytearray(),
        override=True,
    )
    try:
        source = _build_source(boundary)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert captured == [source]
    assert f"__local_mem({shared_bytes})" in source


@pytest.mark.parametrize(
    ("block_threads", "shared_bytes"),
    [(32, (256 << 10) + 1), (64, (512 << 10) + 1), (128, (1 << 20) + 1)],
)
def test_static_shared_memory_rejects_one_byte_over_dimension_dependent_boundary(
    block_threads, shared_bytes
):
    @T.prim_func
    def oversized(a: T.Buffer((1,), "uint8")):
        T.func_attr({"global_symbol": "oversized"})
        scratch = T.alloc_buffer((shared_bytes,), "uint8", scope="shared", align=1)
        for tx in T.thread_binding(block_threads, thread="threadIdx.x"):
            if tx == 0:
                scratch[0] = T.uint8(1)
                a[0] = scratch[0]

    with pytest.raises(
        ValueError, match=r"static shared memory requires.*resident groups"
    ):
        _build_source(oversized)


def test_rejects_dynamic_shared_allocation():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((1,), "float32", scope="shared.dyn")
        temporary[0] = a[0]
        a[0] = temporary[0]

    with pytest.raises(ValueError, match=r"storage scope shared\.dyn is not supported"):
        _build_source(bad)


def test_supports_multiple_kernels():
    mod = tvm.IRModule(
        {"first": vecadd, "second": vecadd.with_attr("global_symbol", "second")}
    )
    build = tvm.get_global_func("target.build.vortex")
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(), override=True
    )
    try:
        source = build(mod, tvm.target.Target("vortex")).inspect_source("vortex")
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert "// Vortex kernel 0: second" in source
    assert "// Vortex kernel 1: vecadd" in source
    assert "case 0u:" in source
    assert "case 1u:" in source


if __name__ == "__main__":
    tvm.testing.main()
