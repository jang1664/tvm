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


def _build_source(func=vecadd):
    target = tvm.target.Target("vortex")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    return tvm.get_global_func("target.build.vortex")(mod, target).inspect_source()


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
    assert "vx_spawn_threads(1, launch->grid, launch->block" in source


def test_rejects_non_x_thread_dimension():
    @T.prim_func
    def bad(a: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        for ty in T.thread_binding(4, thread="threadIdx.y"):
            a[ty] = 0.0

    with pytest.raises(ValueError, match=r"only 1D threadIdx\.x and blockIdx\.x.*threadIdx\.y"):
        _build_source(bad)


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


def test_rejects_unsupported_local_allocation():
    @T.prim_func
    def bad(a: T.Buffer((1,), "float32")):
        T.func_attr({"global_symbol": "bad"})
        temporary = T.alloc_buffer((1,), "float32", scope="local")
        temporary[0] = a[0]
        a[0] = temporary[0]

    with pytest.raises(ValueError, match="local or shared allocation is not supported"):
        _build_source(bad)


def test_supports_multiple_kernels():
    mod = tvm.IRModule({"first": vecadd, "second": vecadd.with_attr("global_symbol", "second")})
    build = tvm.get_global_func("target.build.vortex")
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(callback_name, lambda source, target: bytearray(), override=True)
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
