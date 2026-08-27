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

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.target import Target


def test_vortex_target_defaults():
    target = Target("vortex")

    assert target.kind.name == "vortex"
    assert target.get_target_device_type() == tvm_ffi.DLDeviceType.kDLExtDev
    assert set(target.keys) == {"vortex", "gpu"}
    assert target.attrs["num_warps"] == 4
    assert target.attrs["thread_warp_size"] == 32
    assert target.attrs["max_threads_per_block"] == 128
    assert target.attrs["max_num_threads"] == 128
    assert target.attrs["max_block_size_x"] == 128
    assert target.attrs["max_block_size_y"] == 128
    assert target.attrs["max_block_size_z"] == 128
    assert target.attrs["local_mem_size"] == 1 << 20
    assert target.attrs["max_shared_memory_per_block"] == 1 << 20
    assert target.attrs["max_local_memory_per_thread"] == 4 << 10
    assert target.attrs["xlen"] == 64
    assert target.attrs["mtriple"] == "riscv64-unknown-elf"
    assert target.attrs["vortex_accelerator_profile_version"] == 1
    assert target.attrs["vortex_accelerator_profile_configs"] == ""
    assert target.attrs["vortex_tcu_mode"] == "none"
    assert target.attrs["vortex_tcu_fp_formats"] == ""
    assert target.attrs["vortex_gemm_mode"] == "none"
    assert target.attrs["vortex_mxu_row"] == 32
    assert target.attrs["vortex_mxu_col"] == 32
    assert target.attrs["vortex_mxu_col_tile"] == 1
    assert target.attrs["vortex_tmem_bank_size"] == 64 << 10
    assert target.attrs["vortex_num_dma_channels"] == 8
    assert target.attrs["vortex_gemm_acc_mem_depth"] == 1024
    assert target.attrs["vortex_gemm_dma_mt"] == 128
    assert target.attrs["vortex_gemm_dma_nt"] == 128
    assert target.attrs["vortex_gemm_dma_kt"] == 128
    assert target.attrs["vortex_gemm_qparam_slot_alignment"] == 512
    assert target.attrs["vortex_gemm_tmem_alignment"] == 64
    assert target.attrs["vortex_gemm_dimension_bits"] == 32
    assert target.attrs["vortex_device_address_bits"] == 64
    assert target.attrs["vortex_gemm_tile_counter_bits"] == 32
    assert target.attrs["vortex_gemm_job_entries"] == 4
    assert target.attrs["vortex_num_cores"] == 1
    assert target.attrs["vortex_platform"] == "generic"
    assert target.attrs["vortex_gemm_abi_version"] == 2
    assert target.attrs["vortex_layout_abi_version"] == 2


@pytest.mark.parametrize("mode", ["invalid", "FP", "fp-int"])
def test_vortex_target_rejects_invalid_tcu_mode(mode):
    with pytest.raises(ValueError, match="vortex_tcu_mode"):
        Target({"kind": "vortex", "vortex_tcu_mode": mode})


def test_vortex_target_rejects_tcu_formats_without_fp_path():
    with pytest.raises(ValueError, match="vortex_tcu_fp_formats"):
        Target(
            {
                "kind": "vortex",
                "vortex_tcu_mode": "int",
                "vortex_tcu_fp_formats": "fp16",
            }
        )


@pytest.mark.parametrize("formats", ["float16", "fp16,fp16"])
def test_vortex_target_rejects_invalid_tcu_formats(formats):
    with pytest.raises(ValueError, match="vortex_tcu_fp_formats"):
        Target(
            {
                "kind": "vortex",
                "vortex_tcu_mode": "fp",
                "vortex_tcu_fp_formats": formats,
            }
        )


@pytest.mark.parametrize("mode", ["invalid", "row_major", "IMPROVE"])
def test_vortex_target_rejects_invalid_gemm_mode(mode):
    with pytest.raises(ValueError, match="vortex_gemm_mode"):
        Target({"kind": "vortex", "vortex_gemm_mode": mode})


def test_vortex_accelerator_target_round_trip():
    target = Target(
        {
            "kind": "vortex",
            "vortex_accelerator_profile_fingerprint": "a" * 64,
            "vortex_accelerator_profile_configs": "-DEXT_TCU_ENABLE -DNUM_THREADS=32",
            "vortex_tcu_mode": "fp_int",
            "vortex_tcu_fp_formats": "fp16,bf16",
            "vortex_gemm_mode": "improve",
            "vortex_mxu_col_tile": 32,
            "vortex_platform": "vivado",
        }
    )

    assert dict(Target(str(target)).attrs) == dict(target.attrs)


def test_vortex_accelerator_fingerprint_and_configs_are_coupled():
    with pytest.raises(ValueError, match="fingerprint and CONFIGS"):
        Target(
            {
                "kind": "vortex",
                "vortex_accelerator_profile_fingerprint": "a" * 64,
            }
        )


def test_vortex_target_derives_hardware_limits():
    target = Target(
        {
            "kind": "vortex",
            "num_warps": 8,
            "thread_warp_size": 16,
            "xlen": 32,
            "mcpu": "generic-rv32",
            "mattr": ["+m", "+a"],
        }
    )

    assert target.attrs["max_threads_per_block"] == 128
    assert target.attrs["max_num_threads"] == 128
    assert target.attrs["max_block_size_x"] == 128
    assert target.attrs["max_block_size_y"] == 128
    assert target.attrs["max_block_size_z"] == 128
    assert target.attrs["local_mem_size"] == 1 << 20
    assert target.attrs["max_shared_memory_per_block"] == 1 << 20
    assert target.attrs["max_local_memory_per_thread"] == 4 << 10
    assert target.attrs["mtriple"] == "riscv32-unknown-elf"

    round_tripped = Target(str(target))
    assert round_tripped.kind.name == target.kind.name
    assert set(round_tripped.keys) == set(target.keys)
    assert dict(round_tripped.attrs) == dict(target.attrs)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("num_warps", 0),
        ("thread_warp_size", 0),
        ("local_mem_size", 0),
        ("max_local_memory_per_thread", 0),
        ("max_shared_memory_per_block", -1),
        ("xlen", 128),
    ],
)
def test_vortex_target_rejects_invalid_hardware_attributes(attribute, value):
    with pytest.raises(ValueError):
        Target({"kind": "vortex", attribute: value})


@pytest.mark.parametrize(
    "attribute",
    [
        "max_threads_per_block",
        "max_num_threads",
        "max_block_size_x",
        "max_block_size_y",
        "max_block_size_z",
    ],
)
def test_vortex_target_rejects_inconsistent_thread_limit(attribute):
    with pytest.raises(ValueError, match=attribute):
        Target({"kind": "vortex", attribute: 64})


def test_vortex_target_rejects_inconsistent_local_memory_limit():
    with pytest.raises(ValueError, match="max_shared_memory_per_block"):
        Target(
            {
                "kind": "vortex",
                "local_mem_size": 1024,
                "max_shared_memory_per_block": 512,
            }
        )


def test_vortex_target_derives_shared_limit_from_custom_local_memory():
    target = Target({"kind": "vortex", "local_mem_size": 512 << 10})

    assert target.attrs["local_mem_size"] == 512 << 10
    assert target.attrs["max_shared_memory_per_block"] == 512 << 10


@pytest.mark.parametrize(
    "config",
    [
        {
            "kind": "vortex",
            "local_mem_size": 512 << 10,
            "max_shared_memory_per_block": 0,
        },
        json.dumps(
            {
                "kind": "vortex",
                "local_mem_size": 512 << 10,
                "max_shared_memory_per_block": 0,
            }
        ),
    ],
    ids=["config-map", "json-string"],
)
def test_vortex_target_normalizes_legacy_zero_shared_limit(config):
    target = Target(config)

    assert target.attrs["local_mem_size"] == 512 << 10
    assert target.attrs["max_shared_memory_per_block"] == 512 << 10
    assert Target(str(target)).attrs["max_shared_memory_per_block"] == 512 << 10


def test_vortex_target_preserves_custom_per_thread_local_limit():
    target = Target({"kind": "vortex", "max_local_memory_per_thread": 2 << 10})

    assert target.attrs["max_local_memory_per_thread"] == 2 << 10
    assert Target(str(target)).attrs["max_local_memory_per_thread"] == 2 << 10


def test_vortex_target_rejects_thread_capacity_overflow():
    with pytest.raises(ValueError, match="overflows int64"):
        Target({"kind": "vortex", "num_warps": 1 << 62, "thread_warp_size": 4})


def test_vortex_target_does_not_expose_nondefault_barrier_count():
    with pytest.raises(ValueError, match="num_barriers"):
        Target({"kind": "vortex", "num_barriers": 2})


@pytest.mark.parametrize(
    ("block_threads", "warps_per_group", "resident_groups", "shared_limit"),
    [
        (32, 1, 4, 256 << 10),
        (48, 2, 2, 512 << 10),
        (64, 2, 2, 512 << 10),
        (96, 3, 1, 1 << 20),
        (128, 4, 1, 1 << 20),
    ],
)
def test_vortex_block_resource_usage(
    block_threads, warps_per_group, resident_groups, shared_limit
):
    calculate = tvm.get_global_func("target.vortex.get_block_resource_usage")
    resources = calculate(Target("vortex"), block_threads)

    assert resources["block_threads"] == block_threads
    assert resources["warps_per_group"] == warps_per_group
    assert resources["resident_groups"] == resident_groups
    assert resources["effective_max_shared_memory_per_block"] == shared_limit


@pytest.mark.parametrize("block_threads", [0, 129])
def test_vortex_block_resource_usage_rejects_invalid_block_size(block_threads):
    calculate = tvm.get_global_func("target.vortex.get_block_resource_usage")
    with pytest.raises(ValueError, match="block_threads"):
        calculate(Target("vortex"), block_threads)


def test_vortex_block_resource_usage_rejects_non_vortex_target():
    calculate = tvm.get_global_func("target.vortex.get_block_resource_usage")
    with pytest.raises(ValueError, match="requires a vortex target"):
        calculate(Target("llvm"), 1)


def test_vortex_block_resource_usage_checks_all_resident_lmem_slots():
    validate = tvm.get_global_func("target.vortex.validate_shared_memory_usage")
    target = Target("vortex")

    validate(target, 32, 256 << 10)
    with pytest.raises(ValueError, match="requires 1048580 bytes"):
        validate(target, 32, (256 << 10) + 1)

    with pytest.raises(ValueError, match="overflows int64"):
        validate(target, 32, 1 << 62)


def _oversized_thread_block():
    thread_var = tvm.tirx.Var("threadIdx.x", "int32")
    thread_iter = tvm.tirx.IterVar(
        tvm.ir.Range(0, 129), thread_var, tvm.tirx.IterVar.ThreadIndex, "threadIdx.x"
    )
    body = tvm.tirx.AttrStmt(
        thread_iter, "thread_extent", 129, tvm.tirx.Evaluate(thread_var)
    )
    return tvm.tirx.PrimFunc([], body)


def test_vortex_target_limit_is_consumed_by_gpu_verifier():
    target = Target("vortex")
    constraints = {
        "max_shared_memory_per_block": target.attrs["max_shared_memory_per_block"],
        "max_threads_per_block": target.attrs["max_threads_per_block"],
    }

    assert not tvm.s_tir.analysis.verify_gpu_code(
        _oversized_thread_block(), constraints
    )


if __name__ == "__main__":
    tvm.testing.main()
