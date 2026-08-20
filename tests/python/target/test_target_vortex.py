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
    assert target.attrs["max_shared_memory_per_block"] == 0
    assert target.attrs["xlen"] == 64
    assert target.attrs["mtriple"] == "riscv64-unknown-elf"


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
        ("max_shared_memory_per_block", -1),
        ("xlen", 128),
    ],
)
def test_vortex_target_rejects_invalid_hardware_attributes(attribute, value):
    with pytest.raises(ValueError):
        Target({"kind": "vortex", attribute: value})


@pytest.mark.parametrize("attribute", ["max_threads_per_block", "max_num_threads"])
def test_vortex_target_rejects_inconsistent_thread_limit(attribute):
    with pytest.raises(ValueError, match=attribute):
        Target({"kind": "vortex", attribute: 64})


def test_vortex_target_rejects_shared_memory_until_codegen_supports_it():
    with pytest.raises(ValueError, match="does not support shared memory"):
        Target({"kind": "vortex", "max_shared_memory_per_block": 1})


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

    assert not tvm.s_tir.analysis.verify_gpu_code(_oversized_thread_block(), constraints)


if __name__ == "__main__":
    tvm.testing.main()
