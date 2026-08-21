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

import json
import os
import struct

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


@T.prim_func
def shared_copy_kernel(a: T.Buffer((1,), "int32"), b: T.Buffer((1,), "int32")):
    T.func_attr(
        {
            "global_symbol": "shared_copy_kernel",
            "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x"],
        }
    )
    first = T.alloc_buffer((3,), "uint8", scope="shared", align=4)
    second = T.alloc_buffer((2,), "int32", scope="shared", align=16)
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            if tx == 0:
                first[0] = T.uint8(7)
                second[0] = a[0]
                b[0] = second[0] + T.Cast("int32", first[0])


@T.prim_func
def scalar_kernel(
    output: T.Buffer((1,), "int64"),
    i8: T.int8,
    i16: T.int16,
    i32: T.int32,
    i64: T.int64,
    u8: T.uint8,
    u16: T.uint16,
    u32: T.uint32,
    u64: T.uint64,
):
    T.func_attr(
        {
            "global_symbol": "scalar_kernel",
            "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x"],
        }
    )
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for tx in T.thread_binding(1, thread="threadIdx.x"):
            output[bx + tx] = (
                T.Cast("int64", i8)
                + T.Cast("int64", i16)
                + T.Cast("int64", i32)
                + i64
                + T.Cast("int64", u8)
                + T.Cast("int64", u16)
                + T.Cast("int64", u32)
                + T.Cast("int64", u64)
            )


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


@pytest.fixture
def mixed_resource_vortex_module():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        yield tvm.get_global_func("target.build.vortex")(
            tvm.IRModule(
                {"copy_kernel": copy_kernel, "shared_copy_kernel": shared_copy_kernel}
            ),
            tvm.target.Target("vortex"),
        )
    finally:
        tvm.register_global_func(callback_name, previous, override=True)


def test_module_serialization_preserves_source_and_function_metadata(
    vortex_module, tmp_path
):
    module_path = tmp_path / "vecadd.vortex"
    vortex_module.write_to_file(str(module_path))

    restored = tvm.runtime.load_module(str(module_path))
    assert restored.kind == "vortex"
    assert restored.inspect_source("vortex") == vortex_module.inspect_source("vortex")

    with pytest.raises(
        ValueError, match="expected 3 kernel arguments and 2 launch arguments"
    ):
        restored["vecadd"]()


def test_module_serialization_rejects_wrong_version_and_truncation(
    vortex_module, tmp_path
):
    module_path = tmp_path / "vecadd.vortex"
    vortex_module.write_to_file(str(module_path))
    serialized = bytearray(module_path.read_bytes())

    wrong_version = tmp_path / "wrong-version.vortex"
    struct.pack_into("=I", serialized, 0, 0xFFFFFFFF)
    wrong_version.write_bytes(serialized)
    with pytest.raises(
        ValueError, match="Unsupported Vortex module serialization version"
    ):
        tvm.runtime.load_module(str(wrong_version))

    old_version = tmp_path / "old-version.vortex"
    old_serialized = bytearray(module_path.read_bytes())
    struct.pack_into("=I", old_serialized, 0, 2)
    old_version.write_bytes(old_serialized)
    with pytest.raises(
        ValueError, match="Unsupported Vortex module serialization version 2"
    ):
        tvm.runtime.load_module(str(old_version))

    truncated = tmp_path / "truncated.vortex"
    truncated.write_bytes(module_path.read_bytes()[:8])
    with pytest.raises(ValueError, match="Truncated Vortex module serialization"):
        tvm.runtime.load_module(str(truncated))


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("field_count", "must contain exactly 8 fields"),
        ("launch_rank", "launch_rank.*does not match function launch rank 1"),
        ("block_dimension", "thread_block_dim_x.*positive"),
    ],
)
def test_module_serialization_rejects_corrupt_resource_metadata_bytes(
    vortex_module, tmp_path, corruption, message
):
    module_path = tmp_path / "vecadd.vortex"
    vortex_module.write_to_file(str(module_path))
    serialized = bytearray(module_path.read_bytes())

    # Locate the uniquely framed resource entry rather than relying on offsets
    # from unrelated FunctionInfo or source-code serialization details.
    resource_tail = b"vecadd" + struct.pack("=Qqqqqqqqq", 8, 1, 0, 1, 0, 128, 1, 1, 0)
    resource_offset = serialized.find(resource_tail)
    assert resource_offset >= 0
    assert serialized.find(resource_tail, resource_offset + 1) == -1
    field_count_offset = resource_offset + len(b"vecadd")
    if corruption == "field_count":
        struct.pack_into("=Q", serialized, field_count_offset, 3)
    elif corruption == "launch_rank":
        struct.pack_into("=q", serialized, field_count_offset + 8, 2)
    else:
        struct.pack_into("=q", serialized, field_count_offset + 8 + 4 * 8, 0)

    corrupt_path = tmp_path / f"corrupt-resource-{corruption}.vortex"
    corrupt_path.write_bytes(serialized)
    with pytest.raises(ValueError, match=message):
        tvm.runtime.load_module(str(corrupt_path))


def test_all_integer_scalar_widths_are_packable(monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        module = tvm.get_global_func("target.build.vortex")(
            tvm.IRModule({"scalar_kernel": scalar_kernel}), tvm.target.Target("vortex")
        )
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert module["scalar_kernel"] is not None


def test_module_serialization_preserves_multi_kernel_mapping(
    multi_kernel_vortex_module, tmp_path
):
    module_path = tmp_path / "multi.vortex"
    multi_kernel_vortex_module.write_to_file(str(module_path))

    restored = tvm.runtime.load_module(str(module_path))
    assert restored.inspect_source(
        "vortex"
    ) == multi_kernel_vortex_module.inspect_source("vortex")
    assert "Vortex kernel 0: copy_kernel" in restored.inspect_source("vortex")
    assert "Vortex kernel 1: vecadd" in restored.inspect_source("vortex")

    with pytest.raises(ValueError, match="expected 2 kernel arguments"):
        restored["copy_kernel"]()
    with pytest.raises(ValueError, match="expected 3 kernel arguments"):
        restored["vecadd"]()


def test_module_serialization_preserves_per_kernel_resource_metadata(
    mixed_resource_vortex_module, tmp_path
):
    module_path = tmp_path / "mixed-resources.vortex"
    mixed_resource_vortex_module.write_to_file(str(module_path))
    restored = tvm.runtime.load_module(str(module_path))

    metadata = restored["vortex.get_kernel_resource_metadata"]()
    assert list(metadata["copy_kernel"]) == [1, 0, 1, 0, 128, 1, 1, 0]
    assert list(metadata["shared_copy_kernel"]) == [1, 32, 4, 0, 32, 1, 1, 0]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ([1, 0, 1], "must contain exactly 8 fields"),
        (
            [2, 0, 1, 0, 128, 1, 1, 0],
            "launch_rank.*does not match function launch rank 1",
        ),
        ([1, -1, 1, 0, 128, 1, 1, 0], "static_shared_bytes.*non-negative"),
        ([1, 0, 0, 0, 128, 1, 1, 0], "compile_time_resident_groups.*positive"),
        ([1, 0, 1, 0, 0, 1, 1, 0], "thread_block_dim_x.*positive"),
        ([1, 0, 1, 0, 128, 1, 1, 2], "uses_shared_barrier.*0 or 1"),
    ],
)
def test_module_factory_rejects_corrupt_kernel_resource_metadata(replacement, message):
    create_name = "ffi.Module.create.vortex"
    create = tvm.get_global_func(create_name)
    callback_name = "tvm_callback_vortex_compile"
    previous_compile = tvm.get_global_func(callback_name)
    captured = []

    def capture_factory(*args):
        captured.append(args)
        return create(*args)

    tvm.register_global_func(create_name, capture_factory, override=True)
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        tvm.get_global_func("target.build.vortex")(
            tvm.IRModule({"vecadd": vecadd}), tvm.target.Target("vortex")
        )
    finally:
        tvm.register_global_func(create_name, create, override=True)
        tvm.register_global_func(callback_name, previous_compile, override=True)

    assert len(captured) == 1
    args = list(captured[0])
    args[4] = {"vecadd": replacement}
    with pytest.raises(ValueError, match=message):
        create(*args)


def test_module_factory_rejects_missing_kernel_resource_metadata():
    create_name = "ffi.Module.create.vortex"
    create = tvm.get_global_func(create_name)
    callback_name = "tvm_callback_vortex_compile"
    previous_compile = tvm.get_global_func(callback_name)
    captured = []

    tvm.register_global_func(
        create_name,
        lambda *args: captured.append(args) or create(*args),
        override=True,
    )
    tvm.register_global_func(
        callback_name, lambda source, target: bytearray(range(32)), override=True
    )
    try:
        tvm.get_global_func("target.build.vortex")(
            tvm.IRModule({"vecadd": vecadd}), tvm.target.Target("vortex")
        )
    finally:
        tvm.register_global_func(create_name, create, override=True)
        tvm.register_global_func(callback_name, previous_compile, override=True)

    args = list(captured[0])
    args[4] = {}
    with pytest.raises(ValueError, match="exactly one entry per function"):
        create(*args)


def test_launch_rejects_argument_count_before_opening_device(
    vortex_module, monkeypatch
):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(
        ValueError, match="expected 3 kernel arguments and 2 launch arguments"
    ):
        vortex_module["vecadd"]()


def test_launch_rejects_target_block_limit_before_opening_device(
    vortex_module, monkeypatch
):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match="exceeding target limit 128"):
        vortex_module["vecadd"](None, None, None, 2, 129)


def test_launch_rejects_same_residency_wrong_block_shape_before_opening_device(
    vortex_module, monkeypatch
):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(
        ValueError, match=r"block dimension x=64.*compile-time dimension 128"
    ):
        vortex_module["vecadd"](None, None, None, 2, 64)


def test_barrier_configuration_uses_reported_capability():
    validate = tvm.get_global_func("runtime.vortex.validate_barrier_configuration")
    validate(4, 2, "xrt", "")
    with pytest.raises(RuntimeError, match=r"require NUM_BARRIERS.*effective.*=1"):
        validate(4, 1, "xrt", "")


def test_barrier_configuration_fails_closed_without_capability_or_xrt_manifest():
    validate = tvm.get_global_func("runtime.vortex.validate_barrier_configuration")
    with pytest.raises(
        RuntimeError, match=r"capability is unavailable for driver simx"
    ):
        validate(4, 0, "simx", "")


@pytest.mark.parametrize(
    ("configs", "accepted"),
    [("-DNUM_THREADS=32", True), ("-DNUM_BARRIERS=1", False)],
)
def test_legacy_xrt_barrier_configuration_uses_authoritative_manifest(
    tmp_path, configs, accepted
):
    package = tmp_path / "image"
    binary_dir = package / "bin"
    binary_dir.mkdir(parents=True)
    xclbin = binary_dir / "vortex_afu.xclbin"
    xclbin.write_bytes(b"")
    (package / "manifest.json").write_text(
        json.dumps({"params": {"CONFIGS": configs}}), encoding="utf-8"
    )
    validate = tvm.get_global_func("runtime.vortex.validate_barrier_configuration")
    if accepted:
        validate(4, 0, "xrt", str(xclbin))
    else:
        with pytest.raises(RuntimeError, match=r"effective hardware configuration.*=1"):
            validate(4, 0, "xrt", str(xclbin))


def test_launch_rejects_null_pointer_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    with pytest.raises(ValueError, match="pointer argument is null"):
        vortex_module["vecadd"](None, None, None, 2, 128)


def test_launch_rejects_stale_pointer_before_opening_device(vortex_module, monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    cpu_array = tvm.runtime.empty((256,), "int32", tvm.cpu())

    with pytest.raises(
        ValueError, match="not a live allocation owned by the Vortex DeviceAPI"
    ):
        vortex_module["vecadd"](cpu_array, cpu_array, cpu_array, 2, 128)


def test_vortex_device_constructor_requires_an_explicit_driver(monkeypatch):
    monkeypatch.delenv("VORTEX_DRIVER", raising=False)
    device = tvm.vortex(0)
    assert str(device) == "ext_dev:0"
    assert not device.exist


def test_runtime_enabled_reports_vortex_sidecar():
    assert tvm.runtime.enabled("vortex")


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_hardware_allocation_and_copy_round_trip(vortex_hardware_environment):
    host = np.arange((1 << 20) // np.dtype("int32").itemsize + 17, dtype="int32")
    device_array = tvm.runtime.tensor(host, device=tvm.vortex(0))
    np.testing.assert_array_equal(device_array.numpy(), host)
    second_device_array = device_array.copyto(tvm.vortex(0))
    np.testing.assert_array_equal(second_device_array.numpy(), host)
    assert device_array.device.max_shared_memory_per_block == 1 << 20
    assert device_array.device.max_thread_dimensions == [128, 128, 128]


if __name__ == "__main__":
    tvm.testing.main()
