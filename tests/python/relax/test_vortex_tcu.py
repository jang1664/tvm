# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.vortex.pipeline import _tcu_tensorize_pass
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.support.vortex import load_vortex_accelerator_profile

TCU_XCLBIN = Path(
    "/opt/vortex_fpga_bins/fpint/"
    "xrt_hw_u55c_c_f100_fpint_tcu_94c5b39919/bin/vortex_afu.xclbin"
)


@I.ir_module
class ExactTCUMatmul:
    @R.function
    def main(
        lhs: R.Tensor((16, 32), "float16"),
        rhs: R.Tensor((32, 16), "float16"),
    ) -> R.Tensor((16, 16), "float16"):
        return R.matmul(lhs, rhs)


@I.ir_module
class TailTCUMatmul:
    @R.function
    def main(
        lhs: R.Tensor((17, 33), "float16"),
        rhs: R.Tensor((33, 19), "float16"),
    ) -> R.Tensor((17, 19), "float16"):
        return R.matmul(lhs, rhs)


@I.ir_module
class BatchedGQATCUMatmul:
    @R.function
    def main(
        lhs: R.Tensor((2, 2, 2, 7, 33), "float16"),
        rhs: R.Tensor((2, 2, 1, 33, 9), "float16"),
    ) -> R.Tensor((2, 2, 2, 7, 9), "float16"):
        return R.matmul(lhs, rhs)


def _target(tcu=True):
    attrs = {"kind": "vortex"}
    if tcu:
        attrs.update(
            {
                "vortex_tcu_mode": "fp",
                "vortex_tcu_fp_formats": "fp16",
            }
        )
    return tvm.target.Target(attrs, host="llvm")


def test_exact_fp16_matmul_uses_versioned_tcu_call():
    lowered = _tcu_tensorize_pass(_target())(ExactTCUMatmul)
    script = lowered.script()
    assert "vx_tvm_tcu_fp16_tile" in script
    assert 'thread="blockIdx.y"' in script
    assert 'thread="blockIdx.x"' in script
    assert 'thread="threadIdx.x"' in script
    assert 'T.thread_binding(32, thread="threadIdx.x")' in script


def test_tcu_tensorization_pads_tail_and_slices_logical_output():
    lowered = _tcu_tensorize_pass(_target())(TailTCUMatmul)
    script = lowered.script()

    assert "vx_tvm_tcu_fp16_tile" in script
    assert "tcu_lhs_padded" in script
    assert "tcu_rhs_padded" in script
    assert "tcu_logical_output" in script
    assert script.count("T.Select(") == 2
    assert script.count("T.float16(0.0)") == 2
    assert 'out_ty=R.Tensor((17, 19), dtype="float16")' in script
    assert "output.data, 32, 32, 64" in script
    assert lowered.attrs["vortex.tcu.fp16.lowered_matmuls"] == 1
    assert lowered.attrs["vortex.tcu.fp16.padded_matmuls"] == 1


def test_tcu_tensorization_enumerates_batched_gqa_without_cross_batch_flattening():
    lowered = _tcu_tensorize_pass(_target(), require_all=True)(BatchedGQATCUMatmul)
    script = lowered.script()

    assert lowered.attrs["vortex.tcu.fp16.lowered_matmuls"] == 1
    assert lowered.attrs["vortex.tcu.fp16.physical_jobs"] == 8
    assert script.count("R.call_tir(cls.vortex_tcu_fp16_matmul_16_16_64") == 8
    assert script.count("= R.strided_slice(lhs") == 8
    assert script.count("= R.strided_slice(rhs") == 8
    assert "R.matmul" not in script


def test_tcu_tensorization_falls_back_for_disabled_target():
    assert (
        "vx_tvm_tcu_fp16_tile"
        not in _tcu_tensorize_pass(_target(False))(ExactTCUMatmul).script()
    )
    with pytest.raises(ValueError, match="target without FP16 TCU"):
        _tcu_tensorize_pass(_target(False), require_all=True)(ExactTCUMatmul)


def test_default_relax_build_emits_tcu_helper_without_scalar_reduction():
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    captured = []

    def capture(source, unused_target):
        del unused_target
        captured.append(source)
        return bytearray(range(32))

    tvm.register_global_func(callback_name, capture, override=True)
    try:
        executable = relax.build(ExactTCUMatmul, _target(), exec_mode="bytecode")
    finally:
        tvm.register_global_func(callback_name, previous, override=True)

    [host_module] = executable.mod.imports
    [device_module] = host_module.imports
    source = device_module.inspect_source()
    assert captured == [source]
    assert "#include <vx_tvm_tcu.h>" in source
    assert "vx_tvm_tcu_fp16_tile" in source
    assert "vx_spawn_threads(3, launch->grid, launch->block" in source
    assert "__tvm_vortex_kernel_0" in source


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_HARDWARE") != "1",
    reason="set TVM_VORTEX_RUN_HARDWARE=1 inside the exact TCU XRT environment",
)
def test_fp16_tcu_relax_vm_hardware():
    configured_xclbin = Path(os.environ["XRT_XCLBIN_PATH"]).resolve()
    assert configured_xclbin == TCU_XCLBIN.resolve()
    profile = load_vortex_accelerator_profile(
        TCU_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    executable = relax.build(ExactTCUMatmul, target, exec_mode="bytecode")

    rng = np.random.default_rng(41)
    lhs_host = rng.uniform(-0.5, 0.5, (16, 32)).astype("float16")
    rhs_host = rng.uniform(-0.5, 0.5, (32, 16)).astype("float16")
    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    actual = vm["main"](
        tvm.runtime.tensor(lhs_host, device=device),
        tvm.runtime.tensor(rhs_host, device=device),
    ).numpy()
    expected = (lhs_host.astype("float32") @ rhs_host.astype("float32")).astype(
        "float16"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tvm.testing.main()
