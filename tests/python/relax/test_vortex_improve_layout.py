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

import dataclasses

import pytest

from tvm.relax.backend.vortex.layout import ImproveProfile, plan_improve_layout


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _reference_sizes(m, n, k, qdir):
    n_exec = _align(n, 32)
    k_exec = _align(k, 32)
    a_elements = 0
    c_elements = 0
    for m_base in range(0, m, 128):
        m_slot = _align(min(128, m - m_base), 8)
        a_elements += m_slot * k_exec
        c_elements += m_slot * n_exec
    qparam_bytes = 0
    for k_base in range(0, k_exec, 128):
        cur_k = min(128, k_exec - k_base)
        for n_base in range(0, n_exec, 128):
            cur_n = min(128, n_exec - n_base)
            records = cur_k // 32 * cur_n if qdir == 0 else cur_n // 32 * cur_k
            qparam_bytes += _align(records * 2, 512)
    return n_exec, k_exec, a_elements, k_exec * n_exec // 2, qparam_bytes // 2, c_elements


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (7, 31, 33), (9, 33, 31), (127, 129, 65), (129, 257, 193), (128, 128, 128)],
)
@pytest.mark.parametrize("qdir", [0, 1])
@pytest.mark.parametrize("transpose", [False, True])
def test_improve_plan_matches_independent_boundary_reference(shape, qdir, transpose):
    plan = plan_improve_layout(*shape, 32, transpose, qdir)
    expected = _reference_sizes(*shape, qdir)
    assert (
        plan.execution_n,
        plan.execution_k,
        plan.a_elements,
        plan.weight_bytes,
        plan.qparam_elements,
        plan.c_elements,
    ) == expected
    assert plan.logical_m == shape[0]
    assert plan.logical_n == shape[1]
    assert plan.logical_k == shape[2]
    assert all(slot.offset_bytes % 512 == 0 for slot in plan.qparam_slots)
    assert all(slot.reserved_bytes % 512 == 0 for slot in plan.qparam_slots)


def test_improve_plan_records_every_multi_tile_qparam_slot():
    plan = plan_improve_layout(129, 257, 193, 32)
    assert len(plan.m_tiles) == 2
    assert len(plan.n_tiles) == 3
    assert len(plan.k_tiles) == 2
    assert len(plan.qparam_slots) == 6
    assert [slot.offset_bytes for slot in plan.qparam_slots] == sorted(
        slot.offset_bytes for slot in plan.qparam_slots
    )
    for previous, current in zip(plan.qparam_slots, plan.qparam_slots[1:]):
        assert previous.offset_bytes + previous.reserved_bytes == current.offset_bytes


def test_improve_plan_descriptor_requires_neutral_abi_compatible_padding():
    producer = plan_improve_layout(7, 33, 65, 32)
    consumer = plan_improve_layout(7, 17, 33, 32)
    assert producer.c_descriptor.compatible_gemm_input(consumer.a_descriptor)
    poisoned = dataclasses.replace(producer.c_descriptor, padding="unspecified")
    assert not poisoned.compatible_gemm_input(consumer.a_descriptor)
    old_abi = dataclasses.replace(producer.c_descriptor, layout_abi_version=1)
    assert not old_abi.compatible_gemm_input(consumer.a_descriptor)


@pytest.mark.parametrize("shape", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_improve_plan_rejects_non_positive_logical_extents(shape):
    with pytest.raises(ValueError, match="must be positive"):
        plan_improve_layout(*shape, 32)


def test_improve_plan_rejects_unversioned_qblock_and_exact_limits():
    with pytest.raises(ValueError, match="QBLK=16 is unsupported"):
        plan_improve_layout(8, 32, 32, 16)
    narrow = dataclasses.replace(ImproveProfile(), dimension_bits=8)
    with pytest.raises(ValueError, match="execution N=256.*8-bit limit 255"):
        plan_improve_layout(1, 250, 1, 32, profile=narrow)


def test_improve_plan_rejects_scratch_accumulator_and_aggregate_capacity():
    with pytest.raises(ValueError, match="TMEM scratch requires"):
        plan_improve_layout(
            8, 32, 32, 32, profile=dataclasses.replace(ImproveProfile(), tmem_bank_size=4096)
        )
    with pytest.raises(ValueError, match="GEMM_ACC_MEM_DEPTH"):
        plan_improve_layout(
            8, 32, 32, 32, profile=dataclasses.replace(ImproveProfile(), accumulator_depth=256)
        )
    baseline = plan_improve_layout(9, 33, 31, 32)
    with pytest.raises(ValueError, match="peak live allocation.*DRAM limit"):
        plan_improve_layout(
            9,
            33,
            31,
            32,
            profile=dataclasses.replace(
                ImproveProfile(), dram_capacity_bytes=baseline.peak_live_bytes - 1
            ),
        )


def test_improve_plan_checked_uint64_overflow_names_expression():
    with pytest.raises(ValueError, match="uint64 overflow in N execution extent"):
        plan_improve_layout(1, (1 << 64) - 16, 1, 32)
