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
"""The Relax Vortex backend compilation pipeline."""

import math

import tvm
from tvm import relax
from tvm.relax import expr_functor
from tvm.script import tirx as T

from .. import gpu_generic
from .layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)


def _make_quantize_int4_row_major(shape, group_size, scheme):
    """Create the canonical rank-2 FP16 -> signed INT4 tuple implementation."""

    rows, columns = shape
    groups = (columns + group_size - 1) // group_size
    packed_columns = (columns + 1) // 2

    @T.prim_func(private=True)
    def quantize_int4_row_major(
        source: T.Buffer((rows, columns), "float16"),
        packed: T.Buffer((rows, packed_columns), "uint8"),
        scale: T.Buffer((rows, groups), "float16"),
        zero_point: T.Buffer((rows, groups), "int16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        minimum = T.alloc_buffer((1,), "float32", scope="local")
        maximum = T.alloc_buffer((1,), "float32", scope="local")
        scale_value = T.alloc_buffer((1,), "float32", scope="local")
        zero_value = T.alloc_buffer((1,), "int32", scope="local")
        low_value = T.alloc_buffer((1,), "int32", scope="local")
        high_value = T.alloc_buffer((1,), "int32", scope="local")
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(1, thread="threadIdx.x"):
                for row in T.serial(rows):
                    for group in T.serial(groups):
                        minimum[0] = T.float32(3.4028234663852886e38)
                        maximum[0] = T.float32(-3.4028234663852886e38)
                        for offset in T.serial(group_size):
                            if group * group_size + offset < columns:
                                minimum[0] = T.min(
                                    minimum[0],
                                    T.Cast(
                                        "float32",
                                        source[row, group * group_size + offset],
                                    ),
                                )
                                maximum[0] = T.max(
                                    maximum[0],
                                    T.Cast(
                                        "float32",
                                        source[row, group * group_size + offset],
                                    ),
                                )
                        if scheme == "signed_symmetric_int4":
                            scale_value[0] = T.max(
                                T.Select(
                                    minimum[0] < T.float32(0), -minimum[0], minimum[0]
                                ),
                                T.Select(
                                    maximum[0] < T.float32(0), -maximum[0], maximum[0]
                                ),
                            ) / T.float32(7.0)
                            if scale_value[0] == T.float32(0):
                                scale_value[0] = T.float32(1)
                            zero_value[0] = 0
                        else:
                            minimum[0] = T.min(minimum[0], T.float32(0))
                            maximum[0] = T.max(maximum[0], T.float32(0))
                            scale_value[0] = (maximum[0] - minimum[0]) / T.float32(15)
                            if scale_value[0] == T.float32(0):
                                scale_value[0] = T.float32(1)
                                zero_value[0] = 0
                            else:
                                zero_value[0] = (
                                    T.Cast(
                                        "int32", T.round(-minimum[0] / scale_value[0])
                                    )
                                    - 8
                                )
                                zero_value[0] = T.max(
                                    T.int32(-8), T.min(T.int32(7), zero_value[0])
                                )
                        scale[row, group] = T.Cast("float16", scale_value[0])
                        zero_point[row, group] = T.Cast("int16", zero_value[0])
                    for pair in T.serial(packed_columns):
                        low_value[0] = T.Cast(
                            "int32",
                            T.round(
                                T.Cast("float32", source[row, pair * 2])
                                / T.Cast("float32", scale[row, pair * 2 // group_size])
                            ),
                        ) + T.Cast("int32", zero_point[row, pair * 2 // group_size])
                        low_value[0] = T.max(
                            T.int32(-8), T.min(T.int32(7), low_value[0])
                        )
                        high_value[0] = 0
                        if pair * 2 + 1 < columns:
                            high_value[0] = T.Cast(
                                "int32",
                                T.round(
                                    T.Cast("float32", source[row, pair * 2 + 1])
                                    / T.Cast(
                                        "float32",
                                        scale[row, (pair * 2 + 1) // group_size],
                                    )
                                ),
                            ) + T.Cast(
                                "int32", zero_point[row, (pair * 2 + 1) // group_size]
                            )
                            high_value[0] = T.max(
                                T.int32(-8), T.min(T.int32(7), high_value[0])
                            )
                        packed[row, pair] = T.Cast(
                            "uint8",
                            (low_value[0] & T.int32(15))
                            | ((high_value[0] & T.int32(15)) << T.int32(4)),
                        )

    return quantize_int4_row_major


def _make_dequantize_int4_row_major(shape, group_size):
    """Create the canonical signed INT4 tuple -> rank-2 FP16 implementation."""

    rows, columns = shape
    groups = (columns + group_size - 1) // group_size
    packed_columns = (columns + 1) // 2
    total = rows * columns

    @T.prim_func(private=True)
    def dequantize_int4_row_major(
        packed: T.Buffer((rows, packed_columns), "uint8"),
        scale: T.Buffer((rows, groups), "float16"),
        zero_point: T.Buffer((rows, groups), "int16"),
        output: T.Buffer((rows, columns), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    row = (bx * 128 + tx) // columns
                    column = (bx * 128 + tx) % columns
                    nibble = T.Cast(
                        "int32",
                        (packed[row, column // 2] >> T.Cast("uint8", (column % 2) * 4))
                        & T.uint8(15),
                    )
                    signed_value = T.Select(nibble >= 8, nibble - 16, nibble)
                    output[row, column] = T.Cast(
                        "float16",
                        T.Cast(
                            "float32",
                            signed_value - zero_point[row, column // group_size],
                        )
                        * T.Cast("float32", scale[row, column // group_size]),
                    )

    return dequantize_int4_row_major


def _make_kv_cache_update(cache_shapes, update_shapes, position):
    """Create a functional rank-3 cache update for the coupled INT4 tuple."""

    payload_shape, scale_shape, zero_shape = cache_shapes
    payload_update_shape, scale_update_shape, zero_update_shape = update_shapes
    max_elements = max(
        math.prod(payload_shape),
        math.prod(scale_shape),
        math.prod(zero_shape),
    )

    if len(payload_shape) == 2:

        @T.prim_func(private=True)
        def kv_cache_update_rank2(
            cache_payload: T.Buffer(payload_shape, "uint8"),
            cache_scale: T.Buffer(scale_shape, "float16"),
            cache_zero: T.Buffer(zero_shape, "int16"),
            payload: T.Buffer(payload_update_shape, "uint8"),
            scale: T.Buffer(scale_update_shape, "float16"),
            zero: T.Buffer(zero_update_shape, "int16"),
            output_payload: T.Buffer(payload_shape, "uint8"),
            output_scale: T.Buffer(scale_shape, "float16"),
            output_zero: T.Buffer(zero_shape, "int16"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            for bx in T.thread_binding(
                (max_elements + 127) // 128, thread="blockIdx.x"
            ):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    index = bx * 128 + tx
                    if index < payload_shape[0] * payload_shape[1]:
                        row = index // payload_shape[1]
                        column = index % payload_shape[1]
                        output_payload[row, column] = T.Select(
                            row == position,
                            payload[0, column],
                            cache_payload[row, column],
                        )
                    if index < scale_shape[0] * scale_shape[1]:
                        row = index // scale_shape[1]
                        column = index % scale_shape[1]
                        output_scale[row, column] = T.Select(
                            row == position,
                            scale[0, column],
                            cache_scale[row, column],
                        )
                    if index < zero_shape[0] * zero_shape[1]:
                        row = index // zero_shape[1]
                        column = index % zero_shape[1]
                        output_zero[row, column] = T.Select(
                            row == position,
                            zero[0, column],
                            cache_zero[row, column],
                        )

        return kv_cache_update_rank2

    @T.prim_func(private=True)
    def kv_cache_update(
        cache_payload: T.Buffer(payload_shape, "uint8"),
        cache_scale: T.Buffer(scale_shape, "float16"),
        cache_zero: T.Buffer(zero_shape, "int16"),
        payload: T.Buffer(payload_update_shape, "uint8"),
        scale: T.Buffer(scale_update_shape, "float16"),
        zero: T.Buffer(zero_update_shape, "int16"),
        output_payload: T.Buffer(payload_shape, "uint8"),
        output_scale: T.Buffer(scale_shape, "float16"),
        output_zero: T.Buffer(zero_shape, "int16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((max_elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < payload_shape[0] * payload_shape[1] * payload_shape[2]:
                    batch = index // (payload_shape[1] * payload_shape[2])
                    sequence = index // payload_shape[2] % payload_shape[1]
                    column = index % payload_shape[2]
                    output_payload[batch, sequence, column] = T.Select(
                        sequence == position,
                        payload[batch, 0, column],
                        cache_payload[batch, sequence, column],
                    )
                if index < scale_shape[0] * scale_shape[1] * scale_shape[2]:
                    batch = index // (scale_shape[1] * scale_shape[2])
                    sequence = index // scale_shape[2] % scale_shape[1]
                    column = index % scale_shape[2]
                    output_scale[batch, sequence, column] = T.Select(
                        sequence == position,
                        scale[batch, 0, column],
                        cache_scale[batch, sequence, column],
                    )
                if index < zero_shape[0] * zero_shape[1] * zero_shape[2]:
                    batch = index // (zero_shape[1] * zero_shape[2])
                    sequence = index // zero_shape[2] % zero_shape[1]
                    column = index % zero_shape[2]
                    output_zero[batch, sequence, column] = T.Select(
                        sequence == position,
                        zero[batch, 0, column],
                        cache_zero[batch, sequence, column],
                    )

    return kv_cache_update


def _make_fp16_tcu_matmul(m: int, n: int, k: int):
    """Create the exact-tile FP16 TCU implementation for a rank-2 matmul."""

    @T.prim_func(private=True)
    def tcu_matmul(
        lhs: T.Buffer((m, k), "float16"),
        rhs: T.Buffer((k, n), "float16"),
        output: T.Buffer((m, n), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for by in T.thread_binding(m // 16, thread="blockIdx.y"):
            for bx in T.thread_binding(n // 16, thread="blockIdx.x"):
                for tx in T.thread_binding(32, thread="threadIdx.x"):
                    T.evaluate(
                        T.call_extern(
                            "int32",
                            "vx_tvm_tcu_fp16_tile",
                            lhs.data,
                            rhs.data,
                            output.data,
                            m,
                            n,
                            k,
                        )
                    )

    return tcu_matmul


def _make_w4a16_naive(
    lhs_shape,
    packed_shape,
    qparam_shape,
    output_shape,
    rhs_shape,
    group_size,
    quant_axis,
    transpose_rhs,
):
    """Create a row-major W4A16 GEMM job for the naive hardware node."""

    m, k = lhs_shape
    n = output_shape[1]
    source_k_axis = 1 if transpose_rhs else 0
    quant_direction = 0 if quant_axis == source_k_axis else 1
    weight_transpose = int(transpose_rhs)

    @T.prim_func(private=True)
    def mm_w4a16_naive(
        lhs: T.Buffer(lhs_shape, "float16"),
        packed: T.Buffer(packed_shape, "uint8"),
        scale: T.Buffer(qparam_shape, "float16"),
        zero_point: T.Buffer(qparam_shape, "int16"),
        output: T.Buffer(output_shape, "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(1, thread="threadIdx.x"):
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "vx_tvm_gemm_w4a16",
                        lhs.data,
                        packed.data,
                        scale.data,
                        zero_point.data,
                        output.data,
                        m,
                        n,
                        k,
                        group_size,
                        weight_transpose,
                        quant_direction,
                        1,
                    )
                )

    return mm_w4a16_naive


def _make_gemm_a_tiled(plan):
    m = plan.logical_m
    k = plan.logical_k
    k_exec = plan.execution_k
    dma_mt = plan.profile.dma_mt
    dma_kt = plan.profile.dma_kt
    mxu_kt = plan.profile.mxu_kt
    channels = plan.profile.num_dma_channels
    total = plan.a_elements

    @T.prim_func(private=True)
    def gemm_a_tiled(
        source: T.Buffer((m, k), "float16"),
        tiled: T.Buffer((total,), "float16"),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < total:
                    mt = index // (dma_mt * k_exec)
                    within_mt = index % (dma_mt * k_exec)
                    cur_m = T.min(dma_mt, m - mt * dma_mt)
                    slot_m = (cur_m + channels - 1) // channels * channels
                    kt = within_mt // (slot_m * dma_kt)
                    within_kt = within_mt % (slot_m * dma_kt)
                    cur_k = T.min(dma_kt, k_exec - kt * dma_kt)
                    tiled[index] = T.float16(0)
                    if within_kt < cur_m * cur_k:
                        micro_k = within_kt // (cur_m * mxu_kt)
                        within_micro = within_kt % (cur_m * mxu_kt)
                        local_m = within_micro // mxu_kt
                        inner_k = within_micro % mxu_kt
                        global_m = mt * dma_mt + local_m
                        global_k = kt * dma_kt + micro_k * mxu_kt + inner_k
                        if global_k < k:
                            tiled[index] = source[global_m, global_k]

    return gemm_a_tiled


def _make_gemm_w_tiled(rhs_shape, plan):
    rows, columns = rhs_shape
    transpose_rhs = plan.weight_transpose
    total_bytes = plan.weight_bytes
    k = plan.logical_k
    n = plan.logical_n
    k_exec = plan.execution_k
    n_exec = plan.execution_n
    dma_kt = plan.profile.dma_kt
    mxu_kt = plan.profile.mxu_kt
    mxu_nt = plan.profile.mxu_nt

    @T.prim_func(private=True)
    def gemm_w_tiled(
        source: T.Buffer((rows, (columns + 1) // 2), "uint8"),
        tiled: T.Buffer((total_bytes,), "uint8"),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding((total_bytes + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total_bytes:
                    index = bx * 128 + tx
                    kt = index // (dma_kt * n_exec // 2)
                    within_kt = index % (dma_kt * n_exec // 2)
                    cur_k = T.min(dma_kt, k_exec - kt * dma_kt)
                    bytes_per_nt = cur_k * mxu_nt // 2
                    nt = within_kt // bytes_per_nt
                    within_nt = within_kt % bytes_per_nt
                    tiled[index] = T.uint8(0)
                    if transpose_rhs:
                        kb = within_nt // (mxu_nt * (mxu_kt // 2))
                        within_kb = within_nt % (mxu_nt * (mxu_kt // 2))
                        local_n = within_kb // (mxu_kt // 2)
                        k_pair = within_kb % (mxu_kt // 2)
                        global_n = nt * mxu_nt + local_n
                        global_k = kt * dma_kt + kb * mxu_kt + k_pair * 2
                        if global_n < n and global_k < k:
                            tiled[index] = source[global_n, global_k // 2]
                            if global_k + 1 >= k:
                                tiled[index] = tiled[index] & T.uint8(15)
                    else:
                        local_k = within_nt // (mxu_nt // 2)
                        n_pair = within_nt % (mxu_nt // 2)
                        global_k = kt * dma_kt + local_k
                        global_n = nt * mxu_nt + n_pair * 2
                        if global_k < k and global_n < n:
                            tiled[index] = source[global_k, global_n // 2]
                            if global_n + 1 >= n:
                                tiled[index] = tiled[index] & T.uint8(15)

    return gemm_w_tiled


def _make_gemm_qparam_tiled(source_shape, dtype, plan):
    output_elements = plan.qparam_elements
    quant_direction = plan.quant_direction
    transpose_rhs = plan.weight_transpose
    k = plan.logical_k
    n = plan.logical_n
    k_exec = plan.execution_k
    dma_kt = plan.profile.dma_kt
    dma_nt = plan.profile.dma_nt
    mxu_nt = plan.profile.mxu_nt
    qblock = plan.qblock
    alignment = plan.profile.qparam_slot_alignment
    ng_per_micro = (mxu_nt + qblock - 1) // qblock
    full_k_row_elements = sum(
        slot.reserved_bytes for slot in plan.qparam_slots if slot.outer_k == 0
    ) // 2

    @T.prim_func(private=True)
    def gemm_qparam_tiled(
        source: T.Buffer(source_shape, dtype),
        tiled: T.Buffer((output_elements,), dtype),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding((output_elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < output_elements:
                    index = bx * 128 + tx
                    kt = index // full_k_row_elements
                    within_kt = index % full_k_row_elements
                    cur_k = T.min(dma_kt, k_exec - kt * dma_kt)
                    if quant_direction == 0:
                        full_n_payload_bytes = cur_k // qblock * dma_nt * 2
                    else:
                        full_n_payload_bytes = (
                            dma_nt // mxu_nt * cur_k * ng_per_micro * 2
                        )
                    full_n_slot_elements = (
                        (full_n_payload_bytes + alignment - 1) // alignment * alignment // 2
                    )
                    nt_dma = within_kt // full_n_slot_elements
                    slot_index = within_kt % full_n_slot_elements
                    cur_n = T.min(dma_nt, plan.execution_n - nt_dma * dma_nt)
                    if quant_direction == 0:
                        payload_elements = cur_k // qblock * cur_n
                    else:
                        payload_elements = (
                            cur_n // mxu_nt * cur_k * ng_per_micro
                        )
                    tiled[index] = T.Cast(dtype, 0)
                    if slot_index < payload_elements:
                        if quant_direction == 0:
                            groups = cur_k // qblock
                            nb = slot_index // (groups * mxu_nt)
                            within_nb = slot_index % (groups * mxu_nt)
                            group = within_nb // mxu_nt
                            inner_n = within_nb % mxu_nt
                            global_group = kt * (dma_kt // qblock) + group
                            global_n = nt_dma * dma_nt + nb * mxu_nt + inner_n
                            if global_group < (k + qblock - 1) // qblock and global_n < n:
                                if transpose_rhs:
                                    tiled[index] = source[global_n, global_group]
                                else:
                                    tiled[index] = source[global_group, global_n]
                        else:
                            nb = slot_index // (cur_k * ng_per_micro)
                            within_nb = slot_index % (cur_k * ng_per_micro)
                            local_k = within_nb // ng_per_micro
                            ng = within_nb % ng_per_micro
                            global_k = kt * dma_kt + local_k
                            global_ng = (nt_dma * dma_nt + nb * mxu_nt) // qblock + ng
                            if global_k < k and global_ng < (n + qblock - 1) // qblock:
                                if transpose_rhs:
                                    tiled[index] = source[global_ng, global_k]
                                else:
                                    tiled[index] = source[global_k, global_ng]

    return gemm_qparam_tiled


def _make_w4a16_improve(
    tiled_a_elements,
    tiled_w_bytes,
    tiled_qparam_elements,
    tiled_c_elements,
    m,
    n,
    k,
    group_size,
    weight_transpose,
    quant_direction,
    logical_n,
    logical_k,
    layout_abi_version,
):
    @T.prim_func(private=True)
    def mm_w4a16_improve(
        lhs: T.Buffer((tiled_a_elements,), "float16"),
        packed: T.Buffer((tiled_w_bytes,), "uint8"),
        scale: T.Buffer((tiled_qparam_elements,), "float16"),
        zero_point: T.Buffer((tiled_qparam_elements,), "int16"),
        output: T.Buffer((tiled_c_elements,), "float16"),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(1, thread="threadIdx.x"):
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "vx_tvm_gemm_w4a16_v2",
                        lhs.data,
                        packed.data,
                        scale.data,
                        zero_point.data,
                        output.data,
                        m,
                        n,
                        k,
                        group_size,
                        weight_transpose,
                        quant_direction,
                        2,
                        logical_n,
                        logical_k,
                        layout_abi_version,
                    )
                )

    return mm_w4a16_improve


def _make_gemm_c_detile(plan):
    m = plan.logical_m
    n = plan.logical_n
    n_exec = plan.execution_n
    dma_mt = plan.profile.dma_mt
    mxu_nt = plan.profile.mxu_nt
    channels = plan.profile.num_dma_channels
    total = m * n

    @T.prim_func(private=True)
    def gemm_c_detile(
        tiled: T.Buffer((plan.c_elements,), "float16"),
        output: T.Buffer((m, n), "float16"),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    index = bx * 128 + tx
                    global_m = index // n
                    global_n = index % n
                    mt = global_m // dma_mt
                    local_m = global_m % dma_mt
                    cur_m = T.min(dma_mt, m - mt * dma_mt)
                    slot_m = (cur_m + channels - 1) // channels * channels
                    nt = global_n // mxu_nt
                    inner_n = global_n % mxu_nt
                    output[global_m, global_n] = tiled[
                        mt * dma_mt * n_exec
                        + nt * slot_m * mxu_nt
                        + local_m * mxu_nt
                        + inner_n
                    ]

    return gemm_c_detile


def _make_gemm_tiled_relu(plan):
    """Create an in-layout ReLU that keeps neutral physical padding."""

    total = plan.c_elements

    @T.prim_func(private=True)
    def gemm_tiled_relu(
        source: T.Buffer((total,), "float16"),
        output: T.Buffer((total,), "float16"),
    ):
        T.func_attr(
            {
                "tirx.is_scheduled": True,
                "tirx.noalias": True,
                "op_pattern": 8,
                "vortex.improve.layout_preserving": 1,
            }
        )
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < total:
                    output[index] = T.max(source[index], T.float16(0))

    return gemm_tiled_relu


def _make_gemm_tiled_add(plan, rhs_shape=None):
    """Create a descriptor-preserving tiled add with tiled or row-major RHS."""

    total = plan.c_elements
    m = plan.logical_m
    n = plan.logical_n
    n_exec = plan.execution_n
    dma_mt = plan.profile.dma_mt
    mxu_nt = plan.profile.mxu_nt
    channels = plan.profile.num_dma_channels

    if rhs_shape is None:

        @T.prim_func(private=True)
        def gemm_tiled_add(
            lhs: T.Buffer((total,), "float16"),
            rhs: T.Buffer((total,), "float16"),
            output: T.Buffer((total,), "float16"),
        ):
            T.func_attr(
                {
                    "tirx.is_scheduled": True,
                    "tirx.noalias": True,
                    "op_pattern": 8,
                    "vortex.improve.layout_preserving": 1,
                }
            )
            for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    index = bx * 128 + tx
                    if index < total:
                        output[index] = lhs[index] + rhs[index]

        return gemm_tiled_add

    @T.prim_func(private=True)
    def gemm_tiled_add_row_major(
        lhs: T.Buffer((total,), "float16"),
        rhs: T.Buffer(rhs_shape, "float16"),
        output: T.Buffer((total,), "float16"),
    ):
        T.func_attr(
            {
                "tirx.is_scheduled": True,
                "tirx.noalias": True,
                "op_pattern": 8,
                "vortex.improve.layout_preserving": 1,
            }
        )
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < total:
                    output[index] = T.float16(0)
        for bx in T.thread_binding((m * n + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < m * n:
                    global_m = index // n
                    global_n = index % n
                    mt = global_m // dma_mt
                    local_m = global_m % dma_mt
                    cur_m = T.min(dma_mt, m - mt * dma_mt)
                    slot_m = (cur_m + channels - 1) // channels * channels
                    nt = global_n // mxu_nt
                    inner_n = global_n % mxu_nt
                    physical_index = (
                        mt * dma_mt * n_exec
                        + nt * slot_m * mxu_nt
                        + local_m * mxu_nt
                        + inner_n
                    )
                    if len(rhs_shape) == 1:
                        output[physical_index] = lhs[physical_index] + rhs[global_n]
                    elif rhs_shape[0] == 1 and rhs_shape[1] == n:
                        output[physical_index] = lhs[physical_index] + rhs[0, global_n]
                    elif rhs_shape[0] == m and rhs_shape[1] == 1:
                        output[physical_index] = lhs[physical_index] + rhs[global_m, 0]
                    else:
                        output[physical_index] = (
                            lhs[physical_index] + rhs[global_m, global_n]
                        )

    return gemm_tiled_add_row_major


def _static_tensor_shape(expr, dtype=None):
    tensor_type = expr.ty
    if dtype is not None and str(getattr(tensor_type, "dtype", "")) != dtype:
        return None
    shape = getattr(tensor_type, "shape", None)
    if shape is None:
        return None
    values = list(shape.values)
    if not all(isinstance(value, tvm.tirx.IntImm) for value in values):
        return None
    return tuple(int(value) for value in values)


def _prim_value(expr):
    if isinstance(expr, tvm.tirx.IntImm):
        return expr.value
    if isinstance(expr, relax.StringImm):
        return str(expr.value)
    raise TypeError(f"expected a Vortex logical scalar, got {type(expr).__name__}")


@expr_functor.mutator
class _W4A16Lowerer(relax.PyExprMutator):
    """Lower logical W4A16 calls after target selection."""

    def __init__(
        self,
        mod,
        target,
        enable_layout_fusion=True,
        lower_w4a16=True,
        lower_auxiliary_ops=True,
    ):
        super().__init__(mod)
        self.target = target
        self.mode = str(target.attrs.get("vortex_gemm_mode", "none"))
        self.improve_profile = ImproveProfile.from_target(target)
        self.enable_layout_fusion = enable_layout_fusion
        self.lower_w4a16 = lower_w4a16
        self.lower_auxiliary_ops = lower_auxiliary_ops
        self.implementations = {}
        self.original_bindings = {}
        self.tiled_outputs = {}
        self.prepacked_descriptors = []
        self.lowered_w4a16 = 0
        for _, function in mod.functions_items():
            if not isinstance(function, relax.Function) or not isinstance(
                function.body, relax.SeqExpr
            ):
                continue
            for block in function.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding, relax.VarBinding):
                        self.original_bindings[binding.var] = binding.value

    def _lookup_tiled(self, expr):
        if isinstance(expr, relax.Var):
            expr = self.original_bindings.get(expr)
        return self.tiled_outputs.get(expr)

    def _constant_data(self, expr):
        visited = set()
        while isinstance(expr, relax.Var) and expr not in visited:
            visited.add(expr)
            expr = self.original_bindings.get(expr)
        if isinstance(expr, relax.Constant):
            return expr.data.numpy()
        return None

    @staticmethod
    def _plan_key(plan):
        return (
            plan.logical_m,
            plan.logical_n,
            plan.logical_k,
            plan.execution_n,
            plan.execution_k,
            plan.qblock,
            plan.weight_transpose,
            plan.quant_direction,
            plan.profile.layout_abi_version,
        )

    def _emit_detile(self, tiled, plan, out_ty):
        key = ("improve", "detile", self._plan_key(plan))
        if key not in self.implementations:
            self.implementations[key] = self.builder_.add_func(
                _make_gemm_c_detile(plan), "vortex_gemm_c_detile"
            )
        return relax.call_tir(self.implementations[key], [tiled], out_ty=out_ty)

    def _lower_tiled_vector(self, original_call, call, symbol, tiled_args):
        """Keep supported vector operators inside a compatible IMPROVE region."""

        if symbol == "relax.nn.relu" and tiled_args[0] is not None:
            tiled_input, descriptor, plan = tiled_args[0]
            key = ("improve", "relu", self._plan_key(plan))
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_gemm_tiled_relu(plan), "vortex_gemm_tiled_relu"
                )
            tiled_output = self.builder_.emit(
                relax.call_tir(
                    self.implementations[key],
                    [tiled_input],
                    out_ty=relax.TensorType((plan.c_elements,), "float16"),
                ),
                name_hint="gemm_tiled_relu",
            )
            self.tiled_outputs[original_call] = (tiled_output, descriptor, plan)
            return self._emit_detile(tiled_output, plan, call.ty)

        if symbol != "relax.add" or not any(value is not None for value in tiled_args):
            return call
        primary_index = 0 if tiled_args[0] is not None else 1
        tiled_input, descriptor, plan = tiled_args[primary_index]
        other_index = 1 - primary_index
        other_tiled = tiled_args[other_index]
        if other_tiled is not None:
            if not descriptor.compatible_gemm_input(other_tiled[1]):
                return call
            rhs = other_tiled[0]
            rhs_shape = None
        else:
            rhs = call.args[other_index]
            rhs_shape = _static_tensor_shape(rhs, "float16")
            valid_shapes = {
                (plan.logical_n,),
                (1, plan.logical_n),
                (plan.logical_m, 1),
                (plan.logical_m, plan.logical_n),
            }
            if rhs_shape not in valid_shapes:
                return call
        key = ("improve", "add", self._plan_key(plan), rhs_shape)
        if key not in self.implementations:
            self.implementations[key] = self.builder_.add_func(
                _make_gemm_tiled_add(plan, rhs_shape), "vortex_gemm_tiled_add"
            )
        tiled_output = self.builder_.emit(
            relax.call_tir(
                self.implementations[key],
                [tiled_input, rhs],
                out_ty=relax.TensorType((plan.c_elements,), "float16"),
            ),
            name_hint="gemm_tiled_add",
        )
        self.tiled_outputs[original_call] = (tiled_output, descriptor, plan)
        return self._emit_detile(tiled_output, plan, call.ty)

    def visit_call_(self, call):
        original_call = call
        original_tiled_args = []
        original_symbol = None
        if (
            self.mode == "improve"
            and self.enable_layout_fusion
            and isinstance(call.op, tvm.ir.Op)
        ):
            original_symbol = call.op.name
            if original_symbol in ("relax.nn.relu", "relax.add"):
                original_tiled_args = [self._lookup_tiled(arg) for arg in call.args]
        fused_tiled_lhs = None
        if (
            self.mode == "improve"
            and self.enable_layout_fusion
            and isinstance(call.op, tvm.ir.Op)
            and call.op.name == "relax.call_pure_packed"
            and isinstance(call.args[0], relax.ExternFunc)
            and call.args[0].global_symbol == "relax.vortex.mm_w4a16"
        ):
            fused_tiled_lhs = self._lookup_tiled(call.args[1])
        call = super().visit_call_(call)
        if original_tiled_args:
            return self._lower_tiled_vector(
                original_call, call, original_symbol, original_tiled_args
            )
        if (
            not isinstance(call.op, tvm.ir.Op)
            or call.op.name != "relax.call_pure_packed"
            or not isinstance(call.args[0], relax.ExternFunc)
        ):
            return call
        symbol = call.args[0].global_symbol
        if symbol == "relax.vortex.quantize_int4":
            if not self.lower_auxiliary_ops:
                return call
            source = call.args[1]
            source_shape = _static_tensor_shape(source, "float16")
            quant_axis = int(_prim_value(call.args[2]))
            group_size = int(_prim_value(call.args[3]))
            pack_axis = int(_prim_value(call.args[4]))
            scheme = _prim_value(call.args[5])
            if source_shape is None or len(source_shape) != 2:
                raise ValueError(
                    "Vortex quantize_int4 initially requires static rank-2 FP16"
                )
            if quant_axis < 0:
                quant_axis += 2
            if pack_axis < 0:
                pack_axis += 2
            if quant_axis != 1 or pack_axis != 1:
                raise ValueError(
                    "Vortex quantize_int4 backend initially supports quant_axis=pack_axis=1"
                )
            if group_size <= 0:
                raise ValueError("Vortex quantize_int4 group_size must be positive")
            if scheme not in ("signed_symmetric_int4", "signed_asymmetric_int4"):
                raise ValueError(
                    f"unsupported Vortex INT4 quantization scheme {scheme}"
                )
            rows, columns = source_shape
            output_types = [
                relax.TensorType((rows, (columns + 1) // 2), "uint8"),
                relax.TensorType(
                    (rows, (columns + group_size - 1) // group_size), "float16"
                ),
                relax.TensorType(
                    (rows, (columns + group_size - 1) // group_size), "int16"
                ),
            ]
            key = ("quantize", source_shape, group_size, scheme)
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_quantize_int4_row_major(source_shape, group_size, scheme),
                    "vortex_quantize_int4_row_major",
                )
            return relax.call_tir(
                self.implementations[key], [source], out_ty=output_types
            )

        if symbol == "relax.vortex.dequantize_int4":
            if not self.lower_auxiliary_ops:
                return call
            packed, scale, zero_point = call.args[1:4]
            logical_shape = call.args[4]
            if not isinstance(logical_shape, relax.ShapeExpr):
                raise ValueError(
                    "Vortex dequantize_int4 requires a static logical shape"
                )
            shape = tuple(int(value) for value in logical_shape.values)
            quant_axis = int(_prim_value(call.args[5]))
            group_size = int(_prim_value(call.args[6]))
            pack_axis = int(_prim_value(call.args[7]))
            scheme = _prim_value(call.args[8])
            if len(shape) != 2 or quant_axis not in (1, -1) or pack_axis not in (1, -1):
                raise ValueError(
                    "Vortex dequantize_int4 backend initially supports static rank-2 "
                    "quant_axis=pack_axis=1"
                )
            if group_size <= 0 or scheme not in (
                "signed_symmetric_int4",
                "signed_asymmetric_int4",
            ):
                raise ValueError(
                    "unsupported Vortex dequantize_int4 quantization contract"
                )
            rows, columns = shape
            expected_packed = (rows, (columns + 1) // 2)
            expected_qparams = (rows, (columns + group_size - 1) // group_size)
            if (
                _static_tensor_shape(packed, "uint8") != expected_packed
                or _static_tensor_shape(scale, "float16") != expected_qparams
                or _static_tensor_shape(zero_point, "int16") != expected_qparams
            ):
                raise ValueError(
                    "Vortex dequantize_int4 tuple shapes or dtypes are inconsistent"
                )
            key = ("dequantize", shape, group_size)
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_dequantize_int4_row_major(shape, group_size),
                    "vortex_dequantize_int4_row_major",
                )
            return relax.call_tir(
                self.implementations[key],
                [packed, scale, zero_point],
                out_ty=relax.TensorType(shape, "float16"),
            )

        if symbol == "relax.vortex.kv_cache_update":
            if not self.lower_auxiliary_ops:
                return call
            caches = call.args[1:4]
            updates = call.args[4:7]
            position = int(_prim_value(call.args[7]))
            capacity = int(_prim_value(call.args[8]))
            cache_shapes = (
                _static_tensor_shape(caches[0], "uint8"),
                _static_tensor_shape(caches[1], "float16"),
                _static_tensor_shape(caches[2], "int16"),
            )
            update_shapes = (
                _static_tensor_shape(updates[0], "uint8"),
                _static_tensor_shape(updates[1], "float16"),
                _static_tensor_shape(updates[2], "int16"),
            )
            if any(
                shape is None or len(shape) not in (2, 3)
                for shape in (*cache_shapes, *update_shapes)
            ):
                raise ValueError(
                    "Vortex kv_cache_update initially requires static rank-2 or rank-3 tensors"
                )
            ranks = {len(shape) for shape in (*cache_shapes, *update_shapes)}
            if len(ranks) != 1:
                raise ValueError("Vortex kv_cache_update tuple ranks are inconsistent")
            if position < 0 or position >= capacity:
                raise ValueError("Vortex kv_cache_update position is outside capacity")
            for cache_shape, update_shape in zip(cache_shapes, update_shapes):
                if (
                    cache_shape[-2] != capacity
                    or update_shape[-2] != 1
                    or cache_shape[:-2] != update_shape[:-2]
                    or cache_shape[-1] != update_shape[-1]
                ):
                    raise ValueError(
                        "Vortex kv_cache_update tuple shapes are inconsistent"
                    )
            key = ("kv_cache_update", cache_shapes, update_shapes, position)
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_kv_cache_update(cache_shapes, update_shapes, position),
                    "vortex_kv_cache_update",
                )
            return relax.call_tir(
                self.implementations[key],
                [*caches, *updates],
                out_ty=list(call.ty.fields),
            )

        if symbol != "relax.vortex.mm_w4a16":
            return call
        if not self.lower_w4a16:
            return call
        if self.mode not in ("naive", "improve"):
            return call

        lhs, packed, scale, zero_point = call.args[1:5]
        rhs_shape_expr = call.args[5]
        if not isinstance(rhs_shape_expr, relax.ShapeExpr):
            raise ValueError("Vortex W4A16 requires a static logical RHS shape")
        rhs_shape = tuple(int(value) for value in rhs_shape_expr.values)
        group_size = int(_prim_value(call.args[6]))
        quant_axis = int(_prim_value(call.args[7]))
        pack_axis = int(_prim_value(call.args[8]))
        scheme = _prim_value(call.args[9])
        transpose_rhs = bool(_prim_value(call.args[10]))
        lhs_shape = _static_tensor_shape(lhs, "float16")
        packed_shape = _static_tensor_shape(packed, "uint8")
        scale_shape = _static_tensor_shape(scale, "float16")
        zero_point_shape = _static_tensor_shape(zero_point, "int16")
        output_shape = _static_tensor_shape(call, "float16")
        if None in (
            lhs_shape,
            packed_shape,
            scale_shape,
            zero_point_shape,
            output_shape,
        ):
            raise ValueError("Vortex naive W4A16 requires static rank-2 tensors")
        if any(
            len(shape) != 2
            for shape in (lhs_shape, packed_shape, scale_shape, output_shape)
        ):
            raise ValueError("Vortex naive W4A16 initially supports rank-2 tensors")
        if scale_shape != zero_point_shape:
            raise ValueError(
                "Vortex W4A16 scale and INT16 zero-point shapes must match"
            )
        if scheme not in ("signed_symmetric_int4", "signed_asymmetric_int4"):
            raise ValueError(f"unsupported Vortex W4A16 quantization scheme {scheme}")
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError("Vortex W4A16 group_size must be a positive power of two")
        if quant_axis < 0:
            quant_axis += 2
        if pack_axis < 0:
            pack_axis += 2
        if quant_axis not in (0, 1) or pack_axis not in (0, 1):
            raise ValueError("Vortex W4A16 axes must be valid for the rank-2 RHS")

        source_k_axis = 1 if transpose_rhs else 0
        source_n_axis = 0 if transpose_rhs else 1
        expected_k = rhs_shape[source_k_axis]
        expected_n = rhs_shape[source_n_axis]
        if lhs_shape[1] != expected_k or output_shape != (lhs_shape[0], expected_n):
            raise ValueError("Vortex W4A16 logical matrix shapes are inconsistent")
        expected_packed = list(rhs_shape)
        expected_packed[pack_axis] = (expected_packed[pack_axis] + 1) // 2
        expected_qparam = list(rhs_shape)
        expected_qparam[quant_axis] = (
            expected_qparam[quant_axis] + group_size - 1
        ) // group_size
        if packed_shape != tuple(expected_packed) or scale_shape != tuple(
            expected_qparam
        ):
            raise ValueError(
                "Vortex W4A16 packed payload or qparam shape is inconsistent"
            )

        key = (
            lhs_shape,
            packed_shape,
            scale_shape,
            output_shape,
            rhs_shape,
            group_size,
            quant_axis,
            transpose_rhs,
        )
        self.lowered_w4a16 += 1
        if self.mode == "naive":
            backend_key = ("naive", key)
            if backend_key not in self.implementations:
                self.implementations[backend_key] = self.builder_.add_func(
                    _make_w4a16_naive(*key),
                    "vortex_mm_w4a16_naive",
                )
            return relax.call_tir(
                self.implementations[backend_key],
                [lhs, packed, scale, zero_point],
                out_ty=call.ty,
            )

        m, k = lhs_shape
        n = output_shape[1]
        if pack_axis != 1:
            raise ValueError(
                "Vortex improved W4A16 requires packed INT4 on source RHS axis 1"
            )
        source_k_axis = 1 if transpose_rhs else 0
        quant_direction = 0 if quant_axis == source_k_axis else 1
        plan = plan_improve_layout(
            m,
            n,
            k,
            group_size,
            transpose_rhs,
            quant_direction,
            self.improve_profile,
        )
        tiled_a_elements = plan.a_elements
        tiled_w_bytes = plan.weight_bytes
        tiled_qparam_elements = plan.qparam_elements
        tiled_c_elements = plan.c_elements

        layout_specs = {
            "gemm": (
                _make_w4a16_improve(
                    tiled_a_elements,
                    tiled_w_bytes,
                    tiled_qparam_elements,
                    tiled_c_elements,
                    m,
                    plan.execution_n,
                    plan.execution_k,
                    group_size,
                    int(transpose_rhs),
                    quant_direction,
                    n,
                    k,
                    plan.profile.layout_abi_version,
                ),
                "vortex_mm_w4a16_improve",
            ),
        }
        packed_constant = self._constant_data(original_call.args[2])
        scale_constant = self._constant_data(original_call.args[3])
        zero_constant = self._constant_data(original_call.args[4])
        if packed_constant is None:
            layout_specs["w"] = (
                _make_gemm_w_tiled(rhs_shape, plan),
                (
                    "vortex_gemm_w_tiled_transposed"
                    if transpose_rhs
                    else "vortex_gemm_w_tiled"
                ),
            )
        if scale_constant is None:
            layout_specs["scale"] = (
                _make_gemm_qparam_tiled(scale_shape, "float16", plan),
                "vortex_gemm_scale_tiled",
            )
        if zero_constant is None:
            layout_specs["zero"] = (
                _make_gemm_qparam_tiled(zero_point_shape, "int16", plan),
                "vortex_gemm_zero_point_tiled",
            )
        if fused_tiled_lhs is None:
            layout_specs["a"] = (
                _make_gemm_a_tiled(plan),
                f"vortex_gemm_a_tiled_{m}_{k}",
            )
        elif not fused_tiled_lhs[1].compatible_gemm_input(plan.a_descriptor):
            raise ValueError(
                "Vortex fused GEMM-C to GEMM-A physical layout descriptors are incompatible"
            )
        globals_by_name = {}
        for layout_name, (primfunc, name_hint) in layout_specs.items():
            layout_key = ("improve", layout_name, key)
            if layout_key not in self.implementations:
                self.implementations[layout_key] = self.builder_.add_func(
                    primfunc, name_hint
                )
            globals_by_name[layout_name] = self.implementations[layout_key]

        if fused_tiled_lhs is None:
            tiled_a = self.builder_.emit(
                relax.call_tir(
                    globals_by_name["a"],
                    [lhs],
                    out_ty=relax.TensorType((tiled_a_elements,), "float16"),
                ),
                name_hint="gemm_a_tiled",
            )
        else:
            tiled_a = fused_tiled_lhs[0]
        if packed_constant is None:
            tiled_w = self.builder_.emit(
                relax.call_tir(
                    globals_by_name["w"],
                    [packed],
                    out_ty=relax.TensorType((tiled_w_bytes,), "uint8"),
                ),
                name_hint="gemm_w_tiled",
            )
        else:
            tiled_w = relax.const(prepack_improve_weight(packed_constant, plan))
        if scale_constant is None:
            tiled_scale = self.builder_.emit(
                relax.call_tir(
                    globals_by_name["scale"],
                    [scale],
                    out_ty=relax.TensorType((tiled_qparam_elements,), "float16"),
                ),
                name_hint="gemm_scale_tiled",
            )
        else:
            tiled_scale = relax.const(
                prepack_improve_qparam(scale_constant, plan, "float16")
            )
        if zero_constant is None:
            tiled_zero = self.builder_.emit(
                relax.call_tir(
                    globals_by_name["zero"],
                    [zero_point],
                    out_ty=relax.TensorType((tiled_qparam_elements,), "int16"),
                ),
                name_hint="gemm_zero_point_tiled",
            )
        else:
            tiled_zero = relax.const(
                prepack_improve_qparam(zero_constant, plan, "int16")
            )
        if any(
            value is not None
            for value in (packed_constant, scale_constant, zero_constant)
        ):
            self.prepacked_descriptors.append(
                "M={}:N={}:K={}:Nexec={}:Kexec={}:QBLK={}:WTRANS={}:QDIR={}:ABI={}".format(
                    m,
                    n,
                    k,
                    plan.execution_n,
                    plan.execution_k,
                    group_size,
                    int(transpose_rhs),
                    quant_direction,
                    plan.profile.layout_abi_version,
                )
            )
        tiled_output = self.builder_.emit(
            relax.call_tir(
                globals_by_name["gemm"],
                [tiled_a, tiled_w, tiled_scale, tiled_zero],
                out_ty=relax.TensorType((tiled_c_elements,), "float16"),
            ),
            name_hint="gemm_c_tiled",
        )
        detiled_output = self._emit_detile(tiled_output, plan, call.ty)
        self.tiled_outputs[original_call] = (
            tiled_output,
            plan.c_descriptor,
            plan,
        )
        return detiled_output


def _w4a16_lowering_pass(
    target,
    enable_layout_fusion=True,
    lower_w4a16=True,
    lower_auxiliary_ops=True,
):
    @tvm.transform.module_pass(opt_level=0, name="VortexLowerW4A16")
    def lower(mod, _ctx):
        lowerer = _W4A16Lowerer(
            mod,
            target,
            enable_layout_fusion,
            lower_w4a16,
            lower_auxiliary_ops,
        )
        for global_var, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(global_var, lowerer.visit_expr(func))
        lowered = lowerer.builder_.get()
        if lowerer.lowered_w4a16:
            lowered = lowered.with_attr(
                "vortex.w4a16.lowered", lowerer.lowered_w4a16
            )
        if lowerer.prepacked_descriptors:
            lowered = lowered.with_attr(
                "vortex.improve.prepacked_constants",
                tvm.runtime.convert(tuple(lowerer.prepacked_descriptors)),
            )
        return lowered

    return lower


def _rewrite_dataflow_reshape_before_vortex():
    """Run reshape analysis only before scheduled Vortex PrimFuncs exist."""

    @tvm.transform.module_pass(opt_level=0, name="VortexRewriteDataflowReshape")
    def rewrite(mod, _ctx):
        if mod.attrs and mod.attrs.get("vortex.w4a16.lowered", 0):
            return mod
        return relax.transform.RewriteDataflowReshape()(mod)

    return rewrite


def _static_rank2_fp16_shape(expr):
    tensor_type = expr.ty
    shape = getattr(tensor_type, "shape", None)
    if str(getattr(tensor_type, "dtype", "")) != "float16" or shape is None:
        return None
    values = list(shape.values)
    if len(values) != 2 or not all(
        isinstance(value, tvm.tirx.IntImm) for value in values
    ):
        return None
    return tuple(int(value) for value in values)


def _tcu_tensorize_pass(target: tvm.target.Target):
    """Rewrite eligible logical matmul calls to the versioned Vortex TCU ABI."""

    mode = str(target.attrs.get("vortex_tcu_mode", "none"))
    formats = str(target.attrs.get("vortex_tcu_fp_formats", ""))
    enabled = mode in ("fp", "fp_int") and "fp16" in formats.split(",")

    @tvm.transform.module_pass(opt_level=0, name="VortexTensorizeTCU")
    def tensorize(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        if not enabled:
            return mod

        lhs_pattern = relax.dpl.wildcard()
        rhs_pattern = relax.dpl.wildcard()
        matmul_pattern = relax.dpl.is_op("relax.matmul")(lhs_pattern, rhs_pattern)
        builder = relax.BlockBuilder(mod)
        implementations = {}

        def rewriter(original, matches):
            lhs = matches[lhs_pattern]
            rhs = matches[rhs_pattern]
            lhs_shape = _static_rank2_fp16_shape(lhs)
            rhs_shape = _static_rank2_fp16_shape(rhs)
            output_shape = _static_rank2_fp16_shape(original)
            if lhs_shape is None or rhs_shape is None or output_shape is None:
                return original

            m, k = lhs_shape
            rhs_k, n = rhs_shape
            if (
                rhs_k != k
                or output_shape != (m, n)
                or m % 16 != 0
                or n % 16 != 0
                or k % 32 != 0
            ):
                return original

            shape_key = (m, n, k)
            if shape_key not in implementations:
                implementations[shape_key] = builder.add_func(
                    _make_fp16_tcu_matmul(m, n, k),
                    f"vortex_tcu_fp16_matmul_{m}_{n}_{k}",
                )
            return relax.call_tir(
                implementations[shape_key],
                [lhs, rhs],
                out_ty=original.ty,
            )

        for global_var, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                builder.update_func(
                    global_var,
                    relax.dpl.rewrite_call(matmul_pattern, rewriter, func),
                )
        return builder.finalize()

    return tensorize


def library_dispatch_passes(target: tvm.target.Target):
    """Return library dispatch passes supported by Vortex."""
    return gpu_generic.library_dispatch_passes(target)


def legalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """Legalize Relax and schedule kernels for Vortex."""
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    improve_mode = str(target.attrs.get("vortex_gemm_mode", "none")) == "improve"
    return [
        # Rewrite logical view-only dataflow before introducing pre-scheduled
        # IMPROVE PrimFuncs; the generic analyzer expects unscheduled bodies.
        _rewrite_dataflow_reshape_before_vortex(),
        # Preserve descriptor-compatible IMPROVE regions while logical vector
        # graph structure and immutable constants are still visible.
        _w4a16_lowering_pass(
            target,
            lower_w4a16=improve_mode,
            lower_auxiliary_ops=False,
        ),
        _tcu_tensorize_pass(target),
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        # The generic Matmul rule's conservative 8x8 configuration fits the
        # Vortex target contract (64 threads and a small static shared arena).
        # Rules are tried in order, so unsupported matmul-like shapes and all
        # other operators retain the safe one-dimensional fallback schedule.
        dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback()),
    ]


def dataflow_lower_passes(target: tvm.target.Target):
    """Return Relax dataflow lowering passes for Vortex."""
    passes = gpu_generic.dataflow_lower_passes(target)[1:]
    improve_mode = str(target.attrs.get("vortex_gemm_mode", "none")) == "improve"
    return [
        passes[0],
        _w4a16_lowering_pass(target, lower_w4a16=not improve_mode),
        *passes[1:],
    ]


def finalize_passes(target: tvm.target.Target):
    """Return Relax VM finalization passes for Vortex."""
    return gpu_generic.finalize_passes(target)


def get_default_pipeline(target: tvm.target.Target):
    """Return the default Relax compilation pipeline for Vortex."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            return tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )(mod)

    return _pipeline
