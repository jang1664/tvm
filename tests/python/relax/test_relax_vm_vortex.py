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

import gc
import os
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.script import ir as I, relax as R


VECTOR_SIZE = 129


@I.ir_module
class TwoStageRelaxModule:
    @R.function
    def add_stage(
        x: R.Tensor((VECTOR_SIZE,), "float32"),
        bias: R.Tensor((VECTOR_SIZE,), "float32"),
    ) -> R.Tensor((VECTOR_SIZE,), "float32"):
        return R.add(x, bias)

    @R.function
    def multiply_stage(
        x: R.Tensor((VECTOR_SIZE,), "float32"),
        scale: R.Tensor((VECTOR_SIZE,), "float32"),
    ) -> R.Tensor((VECTOR_SIZE,), "float32"):
        return R.multiply(x, scale)

    @R.function
    def main(
        x: R.Tensor((VECTOR_SIZE,), "float32"),
        bias: R.Tensor((VECTOR_SIZE,), "float32"),
        scale: R.Tensor((VECTOR_SIZE,), "float32"),
    ) -> R.Tensor((VECTOR_SIZE,), "float32"):
        cls = TwoStageRelaxModule
        intermediate = cls.add_stage(x, bias)
        return cls.multiply_stage(intermediate, scale)


def _target():
    return tvm.target.Target("vortex", host="llvm")


def _lower_relax_module():
    target = _target()
    with target:
        return relax.get_default_pipeline(target)(TwoStageRelaxModule)


def _make_inputs(device):
    rng = np.random.default_rng(8)
    x_host = rng.uniform(-2.0, 2.0, size=VECTOR_SIZE).astype("float32")
    bias_host = rng.uniform(-0.5, 0.5, size=VECTOR_SIZE).astype("float32")
    scale_host = rng.uniform(-1.5, 1.5, size=VECTOR_SIZE).astype("float32")
    inputs = [
        tvm.runtime.tensor(array, device=device)
        for array in (x_host, bias_host, scale_host)
    ]
    return inputs, (x_host + bias_host) * scale_host


def _run_and_check(executable, check_launch_order=True):
    device = tvm.vortex(0)
    inputs, expected = _make_inputs(device)
    launch_order = []
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")

    def instrument(unused_func, name, before_run, unused_ret_value, *unused_args):
        if before_run and name in ("add", "multiply"):
            launch_order.append(name)
        return relax.VMInstrumentReturnKind.NO_OP

    vm.set_instrument(instrument)
    output = vm["main"](*inputs)
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-5)
    if check_launch_order:
        assert launch_order == ["add", "multiply"]


def test_relax_pipeline_legalizes_and_schedules_two_vortex_kernels():
    lowered = _lower_relax_module()
    primfuncs = [
        (global_var.name_hint, func)
        for global_var, func in lowered.functions.items()
        if isinstance(func, tvm.tirx.PrimFunc)
    ]

    assert len(primfuncs) == 2
    assert {name for name, _ in primfuncs} == {"add", "multiply"}
    script = lowered.script()
    assert "blockIdx.x" in script
    assert "threadIdx.x" in script


def test_relax_build_imports_vortex_device_module():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, target):
        captured.append((source, target))
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = relax.build(TwoStageRelaxModule, _target(), exec_mode="bytecode")
        generated_device_module = executable.mod.imports[0].imports[0]
        module_with_external_device_code = TwoStageRelaxModule.with_attr(
            "external_mods", [generated_device_module]
        )
        executable_with_external = relax.build(
            module_with_external_device_code, _target(), exec_mode="bytecode"
        )
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    assert len(captured) == 2
    assert captured[0][1].kind.name == "vortex"
    assert "// Vortex kernel 0: add_kernel" in captured[0][0]
    assert "// Vortex kernel 1: multiply_kernel" in captured[0][0]
    assert executable.mod.kind == "relax.VMExecutable"
    assert len(executable.mod.imports) == 1
    host_module = executable.mod.imports[0]
    assert host_module.kind == "llvm"
    assert len(host_module.imports) == 1
    device_module = host_module.imports[0]
    assert device_module.kind == "vortex"
    assert relax.vm_build._is_device_module(device_module)

    # External Vortex modules must join TIR device code under the LLVM host
    # module, rather than being linked as a Relax runtime extension.
    assert len(executable_with_external.mod.imports) == 1
    external_host_module = executable_with_external.mod.imports[0]
    assert external_host_module.kind == "llvm"
    assert [module.kind for module in external_host_module.imports] == ["vortex", "vortex"]


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_relax_vm_bytecode_hardware_and_serialization(tmp_path):
    assert os.environ.get("VORTEX_DRIVER") == "xrt"
    assert os.environ.get("XRT_XCLBIN_PATH")

    executable = relax.build(TwoStageRelaxModule, _target(), exec_mode="bytecode")
    _run_and_check(executable)
    gc.collect()
    available_after_first_run = tvm.vortex(0).available_global_memory

    library_path = Path(tmp_path) / "two_stage_relax_vm.so"
    executable.export_library(str(library_path))
    restored = tvm.runtime.load_module(str(library_path))
    _run_and_check(restored)
    gc.collect()
    assert tvm.vortex(0).available_global_memory == available_after_first_run


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_relax_vm_compiled_hardware_compatibility():
    assert os.environ.get("VORTEX_DRIVER") == "xrt"
    assert os.environ.get("XRT_XCLBIN_PATH")

    executable = relax.build(TwoStageRelaxModule, _target(), exec_mode="compiled")
    # Compiled VM mode executes the generated host function directly, so VM
    # call instrumentation does not observe its internal device-kernel calls.
    _run_and_check(executable, check_launch_order=False)


if __name__ == "__main__":
    tvm.testing.main()
