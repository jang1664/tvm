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
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.script import ir as I, relax as R


VECTOR_SIZE = 129
MATMUL_M = 32
MATMUL_N = 32
MATMUL_K = 32


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


@I.ir_module
class MatmulRelaxModule:
    @R.function
    def main(
        lhs: R.Tensor((MATMUL_M, MATMUL_K), "float32"),
        rhs: R.Tensor((MATMUL_K, MATMUL_N), "float32"),
    ) -> R.Tensor((MATMUL_M, MATMUL_N), "float32"):
        return R.matmul(lhs, rhs)


def _target():
    return tvm.target.Target("vortex", host="llvm")


def _lower_relax_module():
    target = _target()
    with target:
        return relax.get_default_pipeline(target)(TwoStageRelaxModule)


def _fallback_only_pipeline(target):
    from tvm.relax.backend.vortex import (  # pylint: disable=import-outside-toplevel
        pipeline as vortex_pipeline,
    )
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    legalize = vortex_pipeline.legalize_passes(target)
    legalize[-1] = dl.ApplyDefaultSchedule(dl.gpu.Fallback())

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(mod, unused_context):
        del unused_context
        with target:
            return tvm.transform.Sequential(
                vortex_pipeline.library_dispatch_passes(target)
                + legalize
                + vortex_pipeline.dataflow_lower_passes(target)
                + vortex_pipeline.finalize_passes(target)
            )(mod)

    return pipeline


def _matmul_pipeline(schedule):
    target = _target()
    if schedule == "shared":
        return relax.get_default_pipeline(target)
    if schedule == "fallback":
        return _fallback_only_pipeline(target)
    raise ValueError(f"unknown Vortex matmul schedule: {schedule}")


def _find_vortex_modules(module):
    return module._collect_from_import_tree(lambda imported: imported.kind == "vortex")


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
    # Elementwise operators are not matched by the Matmul rule and must retain
    # the one-dimensional fallback path.
    assert "blockIdx.y" not in script
    assert "threadIdx.y" not in script
    assert 'scope="shared"' not in script
    assert 'scope="local"' not in script


def test_matmul_default_shared_schedule_and_1d_fallback_remain_independent():
    target = _target()
    with target:
        shared = _matmul_pipeline("shared")(MatmulRelaxModule)
        fallback = _matmul_pipeline("fallback")(MatmulRelaxModule)

    shared_script = shared.script()
    fallback_script = fallback.script()
    assert "threadIdx.y" in shared_script
    assert 'scope="shared"' in shared_script
    assert "threadIdx.y" not in fallback_script
    assert "blockIdx.y" not in fallback_script
    assert 'scope="shared"' not in fallback_script


def _execute_matmul_performance_worker(schedule):
    target = _target()
    build_start = time.perf_counter()
    executable = relax.build(
        MatmulRelaxModule,
        target,
        relax_pipeline=_matmul_pipeline(schedule),
        exec_mode="bytecode",
    )
    build_ms = (time.perf_counter() - build_start) * 1e3

    with tempfile.TemporaryDirectory(prefix=f"tvm-vortex-{schedule}-") as directory:
        artifact = Path(directory) / f"matmul-{schedule}.so"
        executable.export_library(str(artifact))
        artifact_bytes = artifact.stat().st_size
        restored = tvm.runtime.load_module(str(artifact))

        [device_module] = _find_vortex_modules(restored)
        [resource] = device_module["vortex.get_kernel_resource_metadata"]().values()
        launch_rank, static_shared_bytes, *_ = map(int, resource)
        if schedule == "shared":
            assert launch_rank == 2
            assert static_shared_bytes > 0
        else:
            assert launch_rank == 1
            assert static_shared_bytes == 0

        rng = np.random.default_rng(31)
        lhs_host = rng.uniform(-1.0, 1.0, (MATMUL_M, MATMUL_K)).astype("float32")
        rhs_host = rng.uniform(-1.0, 1.0, (MATMUL_K, MATMUL_N)).astype("float32")
        device = tvm.vortex(0)
        lhs = tvm.runtime.tensor(lhs_host, device=device)
        rhs = tvm.runtime.tensor(rhs_host, device=device)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        call_start = time.perf_counter()
        actual = vm["main"](lhs, rhs)
        host_call_ms = (time.perf_counter() - call_start) * 1e3
        np.testing.assert_allclose(
            actual.numpy(), lhs_host @ rhs_host, rtol=1e-4, atol=1e-4
        )

    print(
        "TVM_VORTEX_PERF_METRICS="
        + json.dumps(
            {
                "schedule": schedule,
                "build_ms": build_ms,
                "artifact_bytes": artifact_bytes,
                "host_call_ms": host_call_ms,
            },
            sort_keys=True,
        )
    )


def _run_matmul_performance_worker(schedule, timeout_seconds=600):
    environment = os.environ.copy()
    environment["TVM_VORTEX_MATMUL_PERF_WORKER"] = schedule
    completed = subprocess.run(
        [sys.executable, __file__],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=timeout_seconds,
    )
    output = completed.stdout + completed.stderr
    print(output, end="")
    assert completed.returncode == 0, f"Vortex {schedule} performance worker failed"
    metric_match = re.search(r"TVM_VORTEX_PERF_METRICS=(\{.*\})", output)
    assert metric_match is not None, f"missing {schedule} performance metrics"
    cycle_matches = re.findall(r"PERF:.*\bcycles=(\d+)", output)
    assert cycle_matches, f"missing {schedule} hardware cycle counter"
    metrics = json.loads(metric_match.group(1))
    metrics["hardware_cycles"] = int(cycle_matches[-1])
    return metrics


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_matmul_1d_fallback_vs_2d_shared_performance(vortex_hardware_environment):
    del vortex_hardware_environment
    metrics = {
        schedule: _run_matmul_performance_worker(schedule)
        for schedule in ("fallback", "shared")
    }
    print("Vortex comparable matmul schedule metrics:", metrics)
    for result in metrics.values():
        assert result["build_ms"] > 0
        assert result["artifact_bytes"] > 0
        assert result["host_call_ms"] > 0
        assert result["hardware_cycles"] > 0


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
    assert [module.kind for module in external_host_module.imports] == [
        "vortex",
        "vortex",
    ]


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_relax_vm_bytecode_hardware_and_serialization(
    tmp_path, vortex_hardware_environment
):

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
def test_relax_vm_compiled_hardware_compatibility(vortex_hardware_environment):

    executable = relax.build(TwoStageRelaxModule, _target(), exec_mode="compiled")
    # Compiled VM mode executes the generated host function directly, so VM
    # call instrumentation does not observe its internal device-kernel calls.
    _run_and_check(executable, check_launch_order=False)


if __name__ == "__main__":
    performance_worker = os.environ.get("TVM_VORTEX_MATMUL_PERF_WORKER")
    if performance_worker:
        _execute_matmul_performance_worker(performance_worker)
    else:
        tvm.testing.main()
