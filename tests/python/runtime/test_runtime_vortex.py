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


@T.prim_func
def vecadd(
    a: T.Buffer((256,), "int32"),
    b: T.Buffer((256,), "int32"),
    c: T.Buffer((256,), "int32"),
):
    T.func_attr(
        {
            "global_symbol": "vecadd",
            "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x"],
            "tirx.noalias": True,
        }
    )
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            c[bx * 128 + tx] = a[bx * 128 + tx] + b[bx * 128 + tx]


@T.prim_func
def copy_kernel(a: T.Buffer((256,), "int32"), b: T.Buffer((256,), "int32")):
    T.func_attr(
        {
            "global_symbol": "copy_kernel",
            "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x"],
            "tirx.noalias": True,
        }
    )
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            b[bx * 128 + tx] = a[bx * 128 + tx]


@pytest.fixture
def vortex_module():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        yield tvm.get_global_func("target.build.vortex")(
            tvm.IRModule({"vecadd": vecadd}), tvm.target.Target("vortex")
        )
    finally:
        tvm.register_global_func(callback_name, previous, override=True)


@pytest.fixture
def multi_kernel_vortex_module():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        yield tvm.get_global_func("target.build.vortex")(
            tvm.IRModule({"vecadd": vecadd, "copy_kernel": copy_kernel}),
            tvm.target.Target("vortex"),
        )
    finally:
        tvm.register_global_func(callback_name, previous, override=True)


def test_module_serialization_preserves_source_and_function_metadata(vortex_module, tmp_path):
    module_path = tmp_path / "vecadd.vortex"
    vortex_module.write_to_file(str(module_path))

    restored = tvm.runtime.load_module(str(module_path))
    assert restored.kind == "vortex"
    assert restored.inspect_source("vortex") == vortex_module.inspect_source("vortex")

    with pytest.raises(ValueError, match="expected 3 kernel arguments and 2 launch arguments"):
        restored["vecadd"]()


def test_module_serialization_preserves_multi_kernel_mapping(multi_kernel_vortex_module, tmp_path):
    module_path = tmp_path / "multi.vortex"
    multi_kernel_vortex_module.write_to_file(str(module_path))

    restored = tvm.runtime.load_module(str(module_path))
    assert restored.inspect_source("vortex") == multi_kernel_vortex_module.inspect_source("vortex")
    assert "Vortex kernel 0: copy_kernel" in restored.inspect_source("vortex")
    assert "Vortex kernel 1: vecadd" in restored.inspect_source("vortex")

    with pytest.raises(ValueError, match="expected 2 kernel arguments"):
        restored["copy_kernel"]()
    with pytest.raises(ValueError, match="expected 3 kernel arguments"):
        restored["vecadd"]()


def test_launch_rejects_argument_count_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match="expected 3 kernel arguments and 2 launch arguments"):
        vortex_module["vecadd"]()


def test_launch_rejects_target_block_limit_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match="exceeding target limit 128"):
        vortex_module["vecadd"](None, None, None, 2, 129)


def test_launch_rejects_null_pointer_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match="pointer argument is null"):
        vortex_module["vecadd"](None, None, None, 2, 128)


def test_launch_rejects_stale_pointer_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    cpu_array = tvm.runtime.empty((256,), "int32", tvm.cpu())

    with pytest.raises(ValueError, match="not a live allocation owned by the Vortex DeviceAPI"):
        vortex_module["vecadd"](cpu_array, cpu_array, cpu_array, 2, 128)


def test_vortex_device_constructor_requires_an_explicit_driver(monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    device = tvm.vortex(0)
    assert str(device) == "ext_dev:0"
    assert not device.exist


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_hardware_allocation_and_copy_round_trip():
    assert os.environ.get("VORTEX_DRIVER") == "xrt"
    assert os.environ.get("XRT_XCLBIN_PATH")

    host = np.arange(64, dtype="int32")
    device_array = tvm.runtime.tensor(host, device=tvm.vortex(0))
    np.testing.assert_array_equal(device_array.numpy(), host)


if __name__ == "__main__":
    tvm.testing.main()
