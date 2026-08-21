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
import time
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm import relax


torch = pytest.importorskip("torch")
from tvm.relax.frontend.torch import (
    from_exported_program,
)  # pylint: disable=wrong-import-position


INPUT_FEATURES = 7
HIDDEN_FEATURES = 5
OUTPUT_FEATURES = 3
EXPORT_BATCH = 3
MODEL_SEED = 9
PINNED_XCLBIN = Path(
    "/opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c_f100_fpint_64300e5119/bin/vortex_afu.xclbin"
)


class SmallMLP(torch.nn.Module):
    """An inference-only Linear/ReLU/Linear model with irregular dimensions."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.fc2 = torch.nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SharedMatmulModel(torch.nn.Module):
    """A parameter-free exported model whose core is a tiled matmul."""

    def forward(self, lhs, rhs):
        return torch.relu(lhs @ rhs + 0.25)


def _make_model():
    torch.manual_seed(MODEL_SEED)
    return SmallMLP().eval().requires_grad_(False)


def _export(model, dynamic_batch=False):
    torch.manual_seed(MODEL_SEED + 1)
    example = torch.randn(EXPORT_BATCH, INPUT_FEATURES)
    dynamic_shapes = None
    if dynamic_batch:
        batch = torch.export.Dim("batch", min=1, max=4)
        dynamic_shapes = {"x": {0: batch}}
    return torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)


def _import(exported_program, keep_params_as_input=False):
    return from_exported_program(
        exported_program,
        keep_params_as_input=keep_params_as_input,
        unwrap_unit_return_tuple=True,
    )


def _target():
    return tvm.target.Target("vortex", host="llvm")


def _export_shared_matmul_model():
    torch.manual_seed(MODEL_SEED + 2)
    lhs = torch.randn(17, 19)
    rhs = torch.randn(19, 13)
    return torch.export.export(SharedMatmulModel().eval(), (lhs, rhs))


def _device_source(executable):
    host_module = executable.mod.imports[0]
    assert len(host_module.imports) == 1
    device_module = host_module.imports[0]
    assert device_module.kind == "vortex"
    return device_module.inspect_source()


def _device_resources(executable):
    device_module = executable.mod.imports[0].imports[0]
    return device_module["vortex.get_kernel_resource_metadata"]()


def _compile_without_device_toolchain(mod, exec_mode="bytecode"):
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, unused_target):
        captured.append(source)
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = relax.build(mod, _target(), exec_mode=exec_mode)
    finally:
        tvm.register_global_func(callback_name, previous, override=True)
    assert captured == [_device_source(executable)]
    return executable, captured[0]


def test_parameter_lifecycle_is_bound_at_import():
    exported_program = _export(_make_model())
    bound = _import(exported_program, keep_params_as_input=False)
    explicit = _import(exported_program, keep_params_as_input=True)

    # The acceptance executable snapshots inference parameters as Relax
    # constants.  Its public ABI therefore has only the user tensor input.
    assert len(bound["main"].params) == 1
    assert "params" not in bound["main"].attrs

    # Keep the alternative importer behavior covered so a frontend change
    # cannot silently alter the chosen lifecycle.
    assert len(explicit["main"].params) == 5
    assert int(explicit["main"].attrs["num_input"]) == 1
    assert len(explicit["main"].attrs["params"]) == 4


def test_default_vortex_pipeline_compiles_exported_mlp():
    mod = _import(_export(_make_model()))
    lowered = relax.get_default_pipeline(_target())(mod)
    lowered_script = lowered.script()

    assert "threadIdx.x" in lowered_script
    assert "blockIdx.x" in lowered_script
    assert "threadIdx.y" in lowered_script
    assert "blockIdx.y" in lowered_script
    assert lowered_script.count('scope="shared"') >= 2
    assert 'scope="local"' in lowered_script

    executable, source = _compile_without_device_toolchain(mod)
    assert executable.mod.kind == "relax.VMExecutable"
    # Matmul scheduling fuses each linear epilogue into its producer, reducing
    # the old fallback pipeline's four launches to two tiled kernels.
    assert source.count("// Vortex kernel") == 2
    assert "fused_matmul_add_relu" in source
    assert "fused_matmul1_add1" in source
    assert "__tvm_vortex_max" in source
    assert "__local_mem(" in source
    assert "__syncthreads()" in source


def test_exported_matmul_uses_vortex_shared_schedule_and_resource_limits():
    mod = _import(_export_shared_matmul_model())
    lowered = relax.get_default_pipeline(_target())(mod)
    script = lowered.script()

    # dl.gpu.Matmul's Vortex-compatible default is an 8x8 block.  The two
    # shared tiles and thread-private accumulator must come from scheduling,
    # not handwritten TIR or a manual kernel call.
    assert 'thread="threadIdx.x"' in script
    assert 'thread="threadIdx.y"' in script
    assert "T.int64(8)" in script
    assert script.count('scope="shared"') >= 2
    assert 'scope="local"' in script
    assert "T.call_kernel" not in script

    executable, source = _compile_without_device_toolchain(mod)
    assert executable.mod.kind == "relax.VMExecutable"
    assert source.count("// Vortex kernel") >= 1
    assert "__local_mem(" in source
    assert source.count("__syncthreads()") >= 2
    assert "vx_spawn_threads(3, launch->grid, launch->block" in source

    resources = _device_resources(executable)
    assert len(resources) == 1
    [resource] = resources.values()
    (
        launch_rank,
        static_shared_bytes,
        resident_groups,
        private_local_bytes,
        block_x,
        block_y,
        block_z,
        uses_shared_barrier,
    ) = map(int, resource)
    assert launch_rank == 2
    assert static_shared_bytes == 2048
    assert resident_groups == 2
    assert private_local_bytes == 64
    assert (block_x, block_y, block_z) == (8, 8, 1)
    assert uses_shared_barrier == 1
    assert 64 <= int(_target().attrs["max_threads_per_block"])
    assert resident_groups * static_shared_bytes <= int(
        _target().attrs["local_mem_size"]
    )


def test_dynamic_batch_constraints_compile_for_vortex():
    mod = _import(_export(_make_model(), dynamic_batch=True))
    main_attrs = mod["main"].attrs

    assert list(main_attrs["tir_var_lower_bound"].values()) == [1]
    assert list(main_attrs["tir_var_upper_bound"].values()) == [4]
    executable, source = _compile_without_device_toolchain(mod)
    assert executable.mod.kind == "relax.VMExecutable"
    assert source.count("// Vortex kernel") == 2


def _run_and_compare(executable, model, host_input):
    with torch.inference_mode():
        expected = model(host_input).numpy()
    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    device_input = tvm.runtime.tensor(host_input.numpy(), device=device)
    start = time.perf_counter()
    actual = vm["main"](device_input)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-5)
    return elapsed_ms


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_torch_export_mlp_hardware_bytecode_and_compiled(
    tmp_path, vortex_hardware_environment
):
    assert vortex_hardware_environment == PINNED_XCLBIN
    model = _make_model()
    mod = _import(_export(model, dynamic_batch=True))
    build_ms = {}
    artifacts = {}
    latencies = {}

    for exec_mode in ("bytecode", "compiled"):
        build_start = time.perf_counter()
        executable = relax.build(mod, _target(), exec_mode=exec_mode)
        build_ms[exec_mode] = (time.perf_counter() - build_start) * 1e3

        artifact = Path(tmp_path) / f"torch_export_mlp_{exec_mode}.so"
        executable.export_library(str(artifact))
        artifacts[exec_mode] = artifact.stat().st_size

        # One executable handles both a boundary and the irregular export
        # batch while enforcing torch.export's [1, 4] dynamic constraint.
        for batch_size in (1, EXPORT_BATCH):
            torch.manual_seed(MODEL_SEED + 100 + batch_size)
            host_input = torch.randn(batch_size, INPUT_FEATURES)
            latencies[(exec_mode, batch_size)] = _run_and_compare(
                executable, model, host_input
            )

    print(
        "Vortex torch.export MLP metrics:",
        {"build_ms": build_ms, "run_ms": latencies, "artifact_bytes": artifacts},
    )
    assert all(value > 0 for value in artifacts.values())
    assert all(value > 0 for value in latencies.values())


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside an allocated XRT hardware environment",
)
def test_exported_shared_matmul_hardware_bytecode_compiled_and_reload(
    tmp_path, vortex_hardware_environment
):
    assert vortex_hardware_environment == PINNED_XCLBIN
    model = SharedMatmulModel().eval()
    mod = _import(_export_shared_matmul_model())
    torch.manual_seed(MODEL_SEED + 3)
    lhs = torch.randn(17, 19)
    rhs = torch.randn(19, 13)
    with torch.inference_mode():
        expected = model(lhs, rhs).numpy()

    metrics = {}
    for exec_mode in ("bytecode", "compiled"):
        build_start = time.perf_counter()
        executable = relax.build(mod, _target(), exec_mode=exec_mode)
        build_ms = (time.perf_counter() - build_start) * 1e3

        artifact = Path(tmp_path) / f"torch_export_shared_matmul_{exec_mode}.so"
        executable.export_library(str(artifact))
        restored = tvm.runtime.load_module(str(artifact))

        device = tvm.vortex(0)
        vm = relax.VirtualMachine(restored, device=device, memory_cfg="naive")
        device_lhs = tvm.runtime.tensor(lhs.numpy(), device=device)
        device_rhs = tvm.runtime.tensor(rhs.numpy(), device=device)
        run_start = time.perf_counter()
        actual = vm["main"](device_lhs, device_rhs)
        run_ms = (time.perf_counter() - run_start) * 1e3
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-5)
        metrics[exec_mode] = {
            "build_ms": build_ms,
            "run_ms": run_ms,
            "artifact_bytes": artifact.stat().st_size,
        }

    print("Vortex torch.export shared matmul metrics:", metrics)
    assert all(item["artifact_bytes"] > 0 for item in metrics.values())


if __name__ == "__main__":
    tvm.testing.main()
