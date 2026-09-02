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

import itertools
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
from .policy import validate_vortex_backend_policy


def _make_hadamard(shape, base_size):
    """Create one parallel mixed-radix FP32 Hadamard device kernel."""

    rank = len(shape)
    rows = math.prod(shape[:-1])
    width = shape[-1]
    factor = width // base_size
    stages = factor.bit_length() - 1
    thread_count = math.gcd(32, width // 2 if stages else width)
    normalization = 1.0 / math.sqrt(width)

    @T.macro
    def load_source(
        source: T.Buffer,
        work: T.Buffer,
        row: T.int64,
        column: T.int64,
    ):
        if rank == 2:
            work[column] = T.Cast("float32", source[row, column])
        elif rank == 3:
            work[column] = T.Cast(
                "float32", source[row // shape[1], row % shape[1], column]
            )
        elif rank == 4:
            work[column] = T.Cast(
                "float32",
                source[
                    row // (shape[1] * shape[2]),
                    row // shape[2] % shape[1],
                    row % shape[2],
                    column,
                ],
            )
        else:
            work[column] = T.Cast(
                "float32",
                source[
                    row // (shape[1] * shape[2] * shape[3]),
                    row // (shape[2] * shape[3]) % shape[1],
                    row // shape[3] % shape[2],
                    row % shape[3],
                    column,
                ],
            )

    @T.macro
    def store_output(
        output: T.Buffer,
        row: T.int64,
        column: T.int64,
        value: T.float32,
    ):
        if rank == 2:
            output[row, column] = T.Cast("float16", value)
        elif rank == 3:
            output[row // shape[1], row % shape[1], column] = T.Cast("float16", value)
        elif rank == 4:
            output[
                row // (shape[1] * shape[2]),
                row // shape[2] % shape[1],
                row % shape[2],
                column,
            ] = T.Cast("float16", value)
        else:
            output[
                row // (shape[1] * shape[2] * shape[3]),
                row // (shape[2] * shape[3]) % shape[1],
                row // shape[3] % shape[2],
                row % shape[3],
                column,
            ] = T.Cast("float16", value)

    @T.prim_func(private=True)
    def hadamard(
        source: T.Buffer(shape, "float16"),
        base: T.Buffer((base_size, base_size), "float32"),
        output: T.Buffer(shape, "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        work = T.alloc_buffer((width,), "float32", scope="shared")
        left = T.alloc_buffer((1,), "float32", scope="local")
        right = T.alloc_buffer((1,), "float32", scope="local")
        accumulator = T.alloc_buffer((1,), "float32", scope="local")
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(thread_count, thread="threadIdx.x"):
                for row in T.serial(rows):
                    for column_chunk in T.serial(width // thread_count):
                        column = column_chunk * thread_count + tx
                        load_source(source, work, row, column)
                    T.tvm_storage_sync("shared")
                    for stage in T.serial(stages):
                        stride = T.shift_left(T.int64(1), stage)
                        for pair_chunk in T.serial(
                            base_size * factor // 2 // thread_count
                        ):
                            linear_pair = pair_chunk * thread_count + tx
                            base_row = linear_pair // (factor // 2)
                            pair = linear_pair % (factor // 2)
                            group = pair // stride
                            offset = pair % stride
                            left_index = base_row * factor + group * stride * 2 + offset
                            right_index = left_index + stride
                            left[0] = work[left_index]
                            right[0] = work[right_index]
                            work[left_index] = left[0] + right[0]
                            work[right_index] = left[0] - right[0]
                        T.tvm_storage_sync("shared")
                    for output_chunk in T.serial(width // thread_count):
                        linear_output = output_chunk * thread_count + tx
                        target_base = linear_output // factor
                        column = linear_output % factor
                        accumulator[0] = T.float32(0)
                        for source_base in T.serial(base_size):
                            accumulator[0] = accumulator[0] + base[
                                target_base, source_base
                            ] * work[source_base * factor + column]
                        store_output(
                            output,
                            row,
                            target_base * factor + column,
                            accumulator[0] * T.float32(normalization),
                        )
                    # Every thread must finish reading the current row from the
                    # shared work buffer before any thread starts loading the
                    # next row into the same storage.
                    T.tvm_storage_sync("shared")

    return hadamard


def _make_causal_softmax(shape, position_shape, valid_length_shape, head_dim):
    """Create one rank-5 scaled causal-mask/softmax device kernel."""

    rows = math.prod(shape[:-1])
    capacity = shape[-1]
    inverse_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func(private=True)
    def causal_softmax(
        scores: T.Buffer(shape, "float16"),
        position_ids: T.Buffer(position_shape, "int64"),
        valid_length: T.Buffer(valid_length_shape, "int64"),
        masked_scores: T.Buffer(shape, "float32"),
        probabilities: T.Buffer(shape, "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        maximum = T.alloc_buffer((1,), "float32", scope="local")
        denominator = T.alloc_buffer((1,), "float32", scope="local")
        for bx in T.thread_binding((rows + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                row = bx * 128 + tx
                if row < rows:
                    batch = row // (shape[1] * shape[2] * shape[3])
                    kv_head = row // (shape[2] * shape[3]) % shape[1]
                    group = row // shape[3] % shape[2]
                    query = row % shape[3]
                    maximum[0] = T.float32(-3.4028234663852886e38)
                    for key in T.serial(capacity):
                        is_valid = T.And(
                            T.Cast("int64", key) < valid_length[()],
                            T.Cast("int64", key) <= position_ids[batch, query],
                        )
                        scaled = T.Cast(
                            "float32", scores[batch, kv_head, group, query, key]
                        ) * T.float32(inverse_scale)
                        masked_scores[batch, kv_head, group, query, key] = T.Select(
                            is_valid, scaled, T.float32(float("-inf"))
                        )
                        if is_valid:
                            maximum[0] = T.max(maximum[0], scaled)
                    denominator[0] = T.float32(0)
                    for key in T.serial(capacity):
                        is_valid = T.And(
                            T.Cast("int64", key) < valid_length[()],
                            T.Cast("int64", key) <= position_ids[batch, query],
                        )
                        if is_valid:
                            denominator[0] = denominator[0] + T.exp(
                                masked_scores[batch, kv_head, group, query, key]
                                - maximum[0]
                            )
                    for key in T.serial(capacity):
                        is_valid = T.And(
                            T.Cast("int64", key) < valid_length[()],
                            T.Cast("int64", key) <= position_ids[batch, query],
                        )
                        probabilities[batch, kv_head, group, query, key] = T.Select(
                            is_valid,
                            T.Cast(
                                "float16",
                                T.exp(
                                    masked_scores[
                                        batch, kv_head, group, query, key
                                    ]
                                    - maximum[0]
                                )
                                / denominator[0],
                            ),
                            T.float16(0),
                        )

    return causal_softmax


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


def _make_batched_output_barrier(shape):
    """Create an opaque copy that keeps a large batched concat out of its consumer."""

    matrices, rows, columns = shape
    elements = matrices * rows * columns

    @T.prim_func(private=True)
    def batched_output_barrier(
        source: T.Buffer(shape, "float16"),
        output: T.Buffer(shape, "float16"),
    ):
        T.func_attr(
            {"tirx.is_scheduled": True, "tirx.noalias": True, "op_pattern": 8}
        )
        for bx in T.thread_binding((elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < elements:
                    matrix = index // (rows * columns)
                    row = index // columns % rows
                    column = index % columns
                    output[matrix, row, column] = source[matrix, row, column]

    return batched_output_barrier


def _make_kv_cache_update(cache_shapes, update_shapes, position):
    """Create a functional rank-2/3/4 cache update for the coupled INT4 tuple."""

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

    if len(payload_shape) == 3:

        @T.prim_func(private=True)
        def kv_cache_update_rank3(
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

        return kv_cache_update_rank3

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
                if index < math.prod(payload_shape):
                    batch = index // (payload_shape[1] * payload_shape[2] * payload_shape[3])
                    head = index // (payload_shape[2] * payload_shape[3]) % payload_shape[1]
                    sequence = index // payload_shape[3] % payload_shape[2]
                    column = index % payload_shape[3]
                    output_payload[batch, head, sequence, column] = T.Select(
                        sequence == position,
                        payload[batch, head, 0, column],
                        cache_payload[batch, head, sequence, column],
                    )
                if index < math.prod(scale_shape):
                    batch = index // (scale_shape[1] * scale_shape[2] * scale_shape[3])
                    head = index // (scale_shape[2] * scale_shape[3]) % scale_shape[1]
                    sequence = index // scale_shape[3] % scale_shape[2]
                    column = index % scale_shape[3]
                    output_scale[batch, head, sequence, column] = T.Select(
                        sequence == position,
                        scale[batch, head, 0, column],
                        cache_scale[batch, head, sequence, column],
                    )
                if index < math.prod(zero_shape):
                    batch = index // (zero_shape[1] * zero_shape[2] * zero_shape[3])
                    head = index // (zero_shape[2] * zero_shape[3]) % zero_shape[1]
                    sequence = index // zero_shape[3] % zero_shape[2]
                    column = index % zero_shape[3]
                    output_zero[batch, head, sequence, column] = T.Select(
                        sequence == position,
                        zero[batch, head, 0, column],
                        cache_zero[batch, head, sequence, column],
                    )

    return kv_cache_update


def _make_kv_cache_update_dynamic(cache_shapes, update_shapes):
    """Create a bounds-safe functional rank-4 cache update at a runtime position."""

    payload_shape, scale_shape, zero_shape = cache_shapes
    payload_update_shape, scale_update_shape, zero_update_shape = update_shapes
    max_elements = max(
        math.prod(payload_shape),
        math.prod(scale_shape),
        math.prod(zero_shape),
    )

    @T.prim_func(private=True)
    def kv_cache_update_dynamic_rank4(
        cache_payload: T.Buffer(payload_shape, "uint8"),
        cache_scale: T.Buffer(scale_shape, "float16"),
        cache_zero: T.Buffer(zero_shape, "int16"),
        payload: T.Buffer(payload_update_shape, "uint8"),
        scale: T.Buffer(scale_update_shape, "float16"),
        zero: T.Buffer(zero_update_shape, "int16"),
        position: T.Buffer((), "int64"),
        output_payload: T.Buffer(payload_shape, "uint8"),
        output_scale: T.Buffer(scale_shape, "float16"),
        output_zero: T.Buffer(zero_shape, "int16"),
    ):
        T.func_attr({"tirx.is_scheduled": True})
        for bx in T.thread_binding((max_elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                if index < math.prod(payload_shape):
                    batch = index // (payload_shape[1] * payload_shape[2] * payload_shape[3])
                    head = index // (payload_shape[2] * payload_shape[3]) % payload_shape[1]
                    sequence = index // payload_shape[3] % payload_shape[2]
                    column = index % payload_shape[3]
                    output_payload[batch, head, sequence, column] = T.Select(
                        T.Cast("int64", sequence) == position[()],
                        payload[batch, head, 0, column],
                        cache_payload[batch, head, sequence, column],
                    )
                if index < math.prod(scale_shape):
                    batch = index // (scale_shape[1] * scale_shape[2] * scale_shape[3])
                    head = index // (scale_shape[2] * scale_shape[3]) % scale_shape[1]
                    sequence = index // scale_shape[3] % scale_shape[2]
                    column = index % scale_shape[3]
                    output_scale[batch, head, sequence, column] = T.Select(
                        T.Cast("int64", sequence) == position[()],
                        scale[batch, head, 0, column],
                        cache_scale[batch, head, sequence, column],
                    )
                if index < math.prod(zero_shape):
                    batch = index // (zero_shape[1] * zero_shape[2] * zero_shape[3])
                    head = index // (zero_shape[2] * zero_shape[3]) % zero_shape[1]
                    sequence = index // zero_shape[3] % zero_shape[2]
                    column = index % zero_shape[3]
                    output_zero[batch, head, sequence, column] = T.Select(
                        T.Cast("int64", sequence) == position[()],
                        zero[batch, head, 0, column],
                        cache_zero[batch, head, sequence, column],
                    )

    return kv_cache_update_dynamic_rank4


def _make_kv_cache_update_dynamic_inplace(cache_shapes, update_shapes):
    """Create a checked rank-4 append that mutates uniquely owned cache buffers."""

    payload_shape, scale_shape, zero_shape = cache_shapes
    payload_update_shape, scale_update_shape, zero_update_shape = update_shapes
    capacity = payload_shape[-2]
    max_elements = max(
        math.prod(payload_update_shape),
        math.prod(scale_update_shape),
        math.prod(zero_update_shape),
    )

    @T.prim_func(private=True)
    def kv_cache_update_dynamic_inplace_rank4(
        cache_payload: T.Buffer(payload_shape, "uint8"),
        cache_scale: T.Buffer(scale_shape, "float16"),
        cache_zero: T.Buffer(zero_shape, "int16"),
        payload: T.Buffer(payload_update_shape, "uint8"),
        scale: T.Buffer(scale_update_shape, "float16"),
        zero: T.Buffer(zero_update_shape, "int16"),
        position: T.Buffer((), "int64"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((max_elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                index = bx * 128 + tx
                valid_position = T.And(
                    T.int64(0) <= position[()], position[()] < T.int64(capacity)
                )
                if index < math.prod(payload_update_shape) and valid_position:
                    batch = index // (
                        payload_update_shape[1]
                        * payload_update_shape[2]
                        * payload_update_shape[3]
                    )
                    head = index // (
                        payload_update_shape[2] * payload_update_shape[3]
                    ) % payload_update_shape[1]
                    column = index % payload_update_shape[3]
                    cache_payload[batch, head, position[()], column] = payload[
                        batch, head, 0, column
                    ]
                if index < math.prod(scale_update_shape) and valid_position:
                    batch = index // (
                        scale_update_shape[1]
                        * scale_update_shape[2]
                        * scale_update_shape[3]
                    )
                    head = index // (
                        scale_update_shape[2] * scale_update_shape[3]
                    ) % scale_update_shape[1]
                    column = index % scale_update_shape[3]
                    cache_scale[batch, head, position[()], column] = scale[
                        batch, head, 0, column
                    ]
                if index < math.prod(zero_update_shape) and valid_position:
                    batch = index // (
                        zero_update_shape[1]
                        * zero_update_shape[2]
                        * zero_update_shape[3]
                    )
                    head = index // (
                        zero_update_shape[2] * zero_update_shape[3]
                    ) % zero_update_shape[1]
                    column = index % zero_update_shape[3]
                    cache_zero[batch, head, position[()], column] = zero[
                        batch, head, 0, column
                    ]

    return kv_cache_update_dynamic_inplace_rank4


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


def _make_pad_fp16_matrix(rows: int, columns: int, padded_rows: int, padded_columns: int):
    """Zero-pad one FP16 matrix without exposing a fusible generic pad."""

    total = padded_rows * padded_columns

    @T.prim_func(private=True)
    def pad_fp16_matrix(
        source: T.Buffer((rows, columns), "float16"),
        output: T.Buffer((padded_rows, padded_columns), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    row = (bx * 128 + tx) // padded_columns
                    column = (bx * 128 + tx) % padded_columns
                    output[row, column] = T.Select(
                        T.And(row < rows, column < columns),
                        source[row, column],
                        T.float16(0),
                    )

    return pad_fp16_matrix


def _make_slice_fp16_matrix(rows: int, columns: int, logical_rows: int, logical_columns: int):
    """Slice the logical top-left matrix region after a padded TCU job."""

    total = logical_rows * logical_columns

    @T.prim_func(private=True)
    def slice_fp16_matrix(
        source: T.Buffer((rows, columns), "float16"),
        output: T.Buffer((logical_rows, logical_columns), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    row = (bx * 128 + tx) // logical_columns
                    column = (bx * 128 + tx) % logical_columns
                    output[row, column] = source[row, column]

    return slice_fp16_matrix


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
        inplace_kv_cache=False,
    ):
        super().__init__(mod)
        self.target = target
        self.mode = str(target.attrs.get("vortex_gemm_mode", "none"))
        self.improve_profile = ImproveProfile.from_target(target)
        self.enable_layout_fusion = enable_layout_fusion
        self.lower_w4a16 = lower_w4a16
        self.lower_auxiliary_ops = lower_auxiliary_ops
        self.inplace_kv_cache = inplace_kv_cache
        self.implementations = {}
        self.original_bindings = {}
        self.tiled_outputs = {}
        self.tiled_inputs = {}
        self.prepacked_descriptors = []
        self.lowered_w4a16 = 0
        self.lowered_hadamard = 0
        self.lowered_causal_softmax = 0
        self.external_prepacked_w4a16 = 0
        self.reused_a_layouts = 0
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

    def _layout_source(self, expr):
        """Look through row-major reshapes when identifying shared GEMM inputs."""

        visited = set()
        while expr not in visited:
            visited.add(expr)
            if isinstance(expr, relax.Var):
                bound = self.original_bindings.get(expr)
                if bound is None:
                    break
                expr = bound
                continue
            if (
                isinstance(expr, relax.Call)
                and isinstance(expr.op, tvm.ir.Op)
                and expr.op.name == "relax.reshape"
            ):
                expr = expr.args[0]
                continue
            break
        return expr

    def _lower_batched_w4a16(
        self,
        call,
        lhs_shape,
        packed_shape,
        scale_shape,
        zero_point_shape,
        output_shape,
        rhs_shape,
        quant_axis,
        pack_axis,
    ):
        """Expand a static batched/GQA call into isolated rank-2 ABI submissions."""

        rank = len(lhs_shape)
        if rank != 5 or any(
            len(shape) != rank
            for shape in (
                packed_shape,
                scale_shape,
                zero_point_shape,
                output_shape,
                rhs_shape,
            )
        ):
            raise ValueError(
                "Vortex batched W4A16 currently requires static rank-5 GQA tensors"
            )
        batch_shape = output_shape[:-2]
        for operand_name, operand_shape in (
            ("lhs", lhs_shape),
            ("packed", packed_shape),
            ("scale", scale_shape),
            ("zero_point", zero_point_shape),
            ("rhs", rhs_shape),
        ):
            for axis, (operand_extent, output_extent) in enumerate(
                zip(operand_shape[:-2], batch_shape)
            ):
                if operand_extent not in (1, output_extent):
                    raise ValueError(
                        "Vortex batched W4A16 broadcast mismatch for "
                        f"{operand_name} axis {axis}: {operand_extent} versus {output_extent}"
                    )
        if quant_axis < 0:
            quant_axis += rank
        if pack_axis < 0:
            pack_axis += rank
        matrix_quant_axis = quant_axis - (rank - 2)
        matrix_pack_axis = pack_axis - (rank - 2)
        if matrix_quant_axis not in (0, 1) or matrix_pack_axis not in (0, 1):
            raise ValueError(
                "Vortex batched W4A16 quantization and packing axes must be matrix axes"
            )

        def emit_matrix_slice(expr, shape, batch_index, name_hint):
            begin = [0 if shape[axis] == 1 else batch_index[axis] for axis in range(3)]
            sliced = self.builder_.emit(
                relax.op.strided_slice(
                    expr,
                    axes=[0, 1, 2],
                    begin=begin,
                    end=[value + 1 for value in begin],
                    assume_inbound=True,
                ),
                name_hint=f"{name_hint}_slice",
            )
            return self.builder_.emit(
                relax.op.reshape(sliced, shape[-2:]),
                name_hint=f"{name_hint}_matrix",
            )

        outputs = []
        for batch_index in itertools.product(*(range(extent) for extent in batch_shape)):
            lhs = emit_matrix_slice(call.args[1], lhs_shape, batch_index, "batched_lhs")
            packed = emit_matrix_slice(
                call.args[2], packed_shape, batch_index, "batched_packed"
            )
            scale = emit_matrix_slice(
                call.args[3], scale_shape, batch_index, "batched_scale"
            )
            zero_point = emit_matrix_slice(
                call.args[4], zero_point_shape, batch_index, "batched_zero_point"
            )
            matrix_call = relax.op.call_pure_packed(
                "relax.vortex.mm_w4a16",
                lhs,
                packed,
                scale,
                zero_point,
                relax.ShapeExpr(rhs_shape[-2:]),
                call.args[6],
                relax.prim_value(matrix_quant_axis),
                relax.prim_value(matrix_pack_axis),
                call.args[9],
                call.args[10],
                ty_args=relax.TensorType(output_shape[-2:], "float16"),
            )
            lowered = self.visit_call_(matrix_call)
            outputs.append(
                self.builder_.emit(
                    relax.op.reshape(lowered, (1, *output_shape[-2:])),
                    name_hint="batched_output_matrix",
                )
            )
        concatenated = self.builder_.emit(
            relax.op.concat(outputs, axis=0), name_hint="batched_output"
        )
        barrier_shape = (math.prod(batch_shape), *output_shape[-2:])
        barrier_key = ("batched_output_barrier", barrier_shape)
        if barrier_key not in self.implementations:
            self.implementations[barrier_key] = self.builder_.add_func(
                _make_batched_output_barrier(barrier_shape),
                "vortex_batched_output_barrier",
            )
        concatenated = self.builder_.emit(
            relax.call_tir(
                self.implementations[barrier_key],
                [concatenated],
                out_ty=relax.TensorType(barrier_shape, "float16"),
            ),
            name_hint="batched_output_barrier",
        )
        return relax.op.reshape(concatenated, output_shape)

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
        if symbol == "relax.vortex.hadamard":
            source, base = call.args[1:3]
            base_size = int(_prim_value(call.args[3]))
            source_shape = _static_tensor_shape(source, "float16")
            base_shape = _static_tensor_shape(base, "float32")
            output_shape = _static_tensor_shape(call, "float16")
            if source_shape is None or len(source_shape) not in (2, 3, 4, 5):
                raise ValueError(
                    "Vortex hadamard requires static rank-2 through rank-5 FP16 input"
                )
            width = source_shape[-1]
            if (
                base_size <= 0
                or width % base_size
                or base_shape != (base_size, base_size)
                or output_shape != source_shape
            ):
                raise ValueError("Vortex hadamard shapes are inconsistent")
            factor = width // base_size
            if factor & (factor - 1):
                raise ValueError("Vortex hadamard factor must be a power of two")
            key = ("hadamard", source_shape, base_size)
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_hadamard(source_shape, base_size),
                    f"vortex_hadamard_{width}",
                )
            self.lowered_hadamard += 1
            return relax.call_tir(
                self.implementations[key], [source, base], out_ty=call.ty
            )

        if symbol == "relax.vortex.causal_softmax":
            scores, position_ids, valid_length = call.args[1:4]
            head_dim = int(_prim_value(call.args[4]))
            scores_shape = _static_tensor_shape(scores, "float16")
            position_shape = _static_tensor_shape(position_ids, "int64")
            valid_length_shape = _static_tensor_shape(valid_length, "int64")
            if scores_shape is None or len(scores_shape) != 5:
                raise ValueError(
                    "Vortex causal_softmax requires static rank-5 FP16 scores"
                )
            if position_shape != (scores_shape[0], scores_shape[-2]):
                raise ValueError(
                    "Vortex causal_softmax position shape must match batch/query"
                )
            if valid_length_shape != ():
                raise ValueError(
                    "Vortex causal_softmax valid_length must be scalar"
                )
            if head_dim <= 0:
                raise ValueError("Vortex causal_softmax head_dim must be positive")
            key = (
                "causal_softmax",
                scores_shape,
                position_shape,
                valid_length_shape,
                head_dim,
            )
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_causal_softmax(
                        scores_shape, position_shape, valid_length_shape, head_dim
                    ),
                    "vortex_causal_softmax",
                )
            self.lowered_causal_softmax += 1
            return relax.call_tir(
                self.implementations[key],
                [scores, position_ids, valid_length],
                out_ty=list(call.ty.fields),
            )

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
            rank = len(shape)
            if quant_axis < 0:
                quant_axis += rank
            if pack_axis < 0:
                pack_axis += rank
            if rank < 2 or quant_axis != rank - 1 or pack_axis != rank - 1:
                raise ValueError(
                    "Vortex dequantize_int4 backend requires a static rank-2-or-higher "
                    "tensor with quant_axis=pack_axis=-1"
                )
            if group_size <= 0 or scheme not in (
                "signed_symmetric_int4",
                "signed_asymmetric_int4",
            ):
                raise ValueError(
                    "unsupported Vortex dequantize_int4 quantization contract"
                )
            columns = shape[-1]
            rows = math.prod(shape[:-1])
            expected_packed = (*shape[:-1], (columns + 1) // 2)
            expected_qparams = (
                *shape[:-1],
                (columns + group_size - 1) // group_size,
            )
            if (
                _static_tensor_shape(packed, "uint8") != expected_packed
                or _static_tensor_shape(scale, "float16") != expected_qparams
                or _static_tensor_shape(zero_point, "int16") != expected_qparams
            ):
                raise ValueError(
                    "Vortex dequantize_int4 tuple shapes or dtypes are inconsistent"
                )
            matrix_shape = (rows, columns)
            matrix_packed_shape = (rows, (columns + 1) // 2)
            matrix_qparam_shape = (
                rows,
                (columns + group_size - 1) // group_size,
            )
            packed = self.builder_.emit(
                relax.op.reshape(packed, matrix_packed_shape),
                name_hint="dequantize_packed_matrix",
            )
            scale = self.builder_.emit(
                relax.op.reshape(scale, matrix_qparam_shape),
                name_hint="dequantize_scale_matrix",
            )
            zero_point = self.builder_.emit(
                relax.op.reshape(zero_point, matrix_qparam_shape),
                name_hint="dequantize_zero_matrix",
            )
            key = ("dequantize", matrix_shape, group_size)
            if key not in self.implementations:
                self.implementations[key] = self.builder_.add_func(
                    _make_dequantize_int4_row_major(matrix_shape, group_size),
                    "vortex_dequantize_int4_row_major",
                )
            output = self.builder_.emit(
                relax.call_tir(
                    self.implementations[key],
                    [packed, scale, zero_point],
                    out_ty=relax.TensorType(matrix_shape, "float16"),
                ),
                name_hint="dequantized_matrix",
            )
            return relax.op.reshape(output, shape)

        if symbol == "relax.vortex.kv_cache_update_dynamic":
            if not self.lower_auxiliary_ops:
                return call
            caches = call.args[1:4]
            updates = call.args[4:7]
            position = call.args[7]
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
            if any(shape is None or len(shape) != 4 for shape in (*cache_shapes, *update_shapes)):
                raise ValueError(
                    "Vortex dynamic kv_cache_update requires static rank-4 cache tensors"
                )
            if _static_tensor_shape(position, "int64") != ():
                raise ValueError(
                    "Vortex dynamic kv_cache_update position must be a scalar INT64 tensor"
                )
            for cache_shape, update_shape in zip(cache_shapes, update_shapes):
                if (
                    cache_shape[-2] != capacity
                    or update_shape[-2] != 1
                    or cache_shape[:-2] != update_shape[:-2]
                    or cache_shape[-1] != update_shape[-1]
                ):
                    raise ValueError(
                        "Vortex dynamic kv_cache_update tuple shapes are inconsistent"
                    )
            key = (
                "kv_cache_update_dynamic",
                cache_shapes,
                update_shapes,
                self.inplace_kv_cache,
            )
            if key not in self.implementations:
                make_update = (
                    _make_kv_cache_update_dynamic_inplace
                    if self.inplace_kv_cache
                    else _make_kv_cache_update_dynamic
                )
                self.implementations[key] = self.builder_.add_func(
                    make_update(cache_shapes, update_shapes),
                    "vortex_kv_cache_update_dynamic",
                )
            cache_args = [*caches, *updates, position]
            if self.inplace_kv_cache:
                return relax.call_tir_inplace(
                    self.implementations[key],
                    cache_args,
                    inplace_indices=[0, 1, 2],
                    out_ty=list(call.ty.fields),
                )
            return relax.call_tir(
                self.implementations[key], cache_args, out_ty=list(call.ty.fields)
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
                shape is None or len(shape) not in (2, 3, 4)
                for shape in (*cache_shapes, *update_shapes)
            ):
                raise ValueError(
                    "Vortex kv_cache_update requires static rank-2, rank-3, or rank-4 tensors"
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

        if symbol not in (
            "relax.vortex.mm_w4a16",
            "relax.vortex.mm_w4a16_prepacked",
        ):
            return call
        parameters_prepacked = symbol == "relax.vortex.mm_w4a16_prepacked"
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
        if scale_shape != zero_point_shape:
            raise ValueError(
                "Vortex W4A16 scale and INT16 zero-point shapes must match"
            )
        if scheme not in ("signed_symmetric_int4", "signed_asymmetric_int4"):
            raise ValueError(f"unsupported Vortex W4A16 quantization scheme {scheme}")
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError("Vortex W4A16 group_size must be a positive power of two")
        if len(lhs_shape) > 2 and parameters_prepacked:
            raise ValueError("Vortex prepacked W4A16 currently requires rank-2 lhs")
        if len(lhs_shape) > 2:
            return self._lower_batched_w4a16(
                call,
                lhs_shape,
                packed_shape,
                scale_shape,
                zero_point_shape,
                output_shape,
                rhs_shape,
                quant_axis,
                pack_axis,
            )
        logical_rank_two = (lhs_shape, output_shape, rhs_shape)
        if any(len(shape) != 2 for shape in logical_rank_two):
            raise ValueError("Vortex W4A16 currently supports rank-2 or rank-5 tensors")
        if parameters_prepacked:
            if any(
                len(shape) != 1
                for shape in (packed_shape, scale_shape, zero_point_shape)
            ):
                raise ValueError("Vortex prepacked W4A16 parameters must be flat")
        elif any(
            len(shape) != 2
            for shape in (packed_shape, scale_shape, zero_point_shape)
        ):
            raise ValueError("Vortex W4A16 currently supports rank-2 or rank-5 tensors")
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
        if not parameters_prepacked:
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
        if parameters_prepacked and (
            packed_shape != (plan.weight_bytes,)
            or scale_shape != (plan.qparam_elements,)
        ):
            raise ValueError(
                "Vortex prepacked W4A16 buffer sizes do not match the physical layout plan"
            )
        if parameters_prepacked:
            self.external_prepacked_w4a16 += 1
        shared_a_key = None
        if self.enable_layout_fusion and fused_tiled_lhs is None:
            shared_a_key = (self._layout_source(original_call.args[1]), m, k)
            shared_tiled_a = self.tiled_inputs.get(shared_a_key)
            if shared_tiled_a is not None:
                if not shared_tiled_a[1].compatible_gemm_input(plan.a_descriptor):
                    raise ValueError(
                        "Vortex shared GEMM-A physical layout descriptors are incompatible"
                    )
                fused_tiled_lhs = shared_tiled_a
                self.reused_a_layouts += 1
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
        packed_constant = (
            None
            if parameters_prepacked
            else self._constant_data(original_call.args[2])
        )
        scale_constant = (
            None
            if parameters_prepacked
            else self._constant_data(original_call.args[3])
        )
        zero_constant = (
            None
            if parameters_prepacked
            else self._constant_data(original_call.args[4])
        )
        if not parameters_prepacked and packed_constant is None:
            layout_specs["w"] = (
                _make_gemm_w_tiled(rhs_shape, plan),
                (
                    "vortex_gemm_w_tiled_transposed"
                    if transpose_rhs
                    else "vortex_gemm_w_tiled"
                ),
            )
        if not parameters_prepacked and scale_constant is None:
            layout_specs["scale"] = (
                _make_gemm_qparam_tiled(scale_shape, "float16", plan),
                "vortex_gemm_scale_tiled",
            )
        if not parameters_prepacked and zero_constant is None:
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
            if shared_a_key is not None:
                self.tiled_inputs[shared_a_key] = (
                    tiled_a,
                    plan.a_descriptor,
                    plan,
                )
        else:
            tiled_a = fused_tiled_lhs[0]
        if parameters_prepacked:
            tiled_w = packed
        elif packed_constant is None:
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
        if parameters_prepacked:
            tiled_scale = scale
        elif scale_constant is None:
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
        if parameters_prepacked:
            tiled_zero = zero_point
        elif zero_constant is None:
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
    layout_policy=None,
    inplace_kv_cache=False,
):
    gemm_mode = str(target.attrs.get("vortex_gemm_mode", "none"))
    if layout_policy is not None:
        if layout_policy not in ("alone", "fused"):
            raise ValueError(
                "Vortex C4 layout policy must be 'alone' or 'fused', "
                f"but got {layout_policy!r}"
            )
        enable_layout_fusion = layout_policy == "fused"
    effective_layout_policy = "fused" if enable_layout_fusion else "alone"

    @tvm.transform.module_pass(opt_level=0, name="VortexLowerW4A16")
    def lower(mod, _ctx):
        lowerer = _W4A16Lowerer(
            mod,
            target,
            enable_layout_fusion,
            lower_w4a16,
            lower_auxiliary_ops,
            inplace_kv_cache,
        )
        for global_var, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(global_var, lowerer.visit_expr(func))
        lowered = lowerer.builder_.get()
        if (
            layout_policy is not None
            and gemm_mode != "improve"
            and lowerer.lowered_w4a16
        ):
            raise ValueError("Vortex C4 layout policy requires GEMM_IMPROVE target mode")
        if lowerer.lowered_w4a16:
            lowered = lowered.with_attr(
                "vortex.w4a16.lowered", lowerer.lowered_w4a16
            )
        if lowerer.lowered_hadamard:
            lowered = lowered.with_attr(
                "vortex.hadamard.lowered", lowerer.lowered_hadamard
            )
        if lowerer.lowered_causal_softmax:
            lowered = lowered.with_attr(
                "vortex.causal_softmax.lowered", lowerer.lowered_causal_softmax
            )
        if lowerer.prepacked_descriptors:
            lowered = lowered.with_attr(
                "vortex.improve.prepacked_constants",
                tvm.runtime.convert(tuple(lowerer.prepacked_descriptors)),
            )
        if lowerer.reused_a_layouts:
            lowered = lowered.with_attr(
                "vortex.improve.reused_a_layouts", lowerer.reused_a_layouts
            )
        if lowerer.external_prepacked_w4a16:
            lowered = lowered.with_attr(
                "vortex.improve.external_prepacked_w4a16",
                lowerer.external_prepacked_w4a16,
            )
        if gemm_mode == "improve":
            lowered = lowered.with_attr(
                "vortex.c4.layout_policy", effective_layout_policy
            )
        elif gemm_mode == "naive" and lowerer.lowered_w4a16:
            lowered = lowered.with_attr("vortex.w4a16.physical_layout", "row_major")
        if inplace_kv_cache:
            lowered = lowered.with_attr("vortex.kv_cache_update_inplace", 1)
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


def _static_fp16_shape(expr):
    tensor_type = expr.ty
    shape = getattr(tensor_type, "shape", None)
    if str(getattr(tensor_type, "dtype", "")) != "float16" or shape is None:
        return None
    values = list(shape.values)
    if len(values) < 2 or not all(isinstance(value, tvm.tirx.IntImm) for value in values):
        return None
    return tuple(int(value) for value in values)


def _static_rank2_fp16_shape(expr):
    shape = _static_fp16_shape(expr)
    return shape if shape is not None and len(shape) == 2 else None


@expr_functor.mutator
class _PaddedFP16TCULowerer(relax.PyExprMutator):
    """Lower static FP16 matmul to isolated, padded rank-2 TCU jobs."""

    def __init__(self, mod, require_all=False):
        super().__init__(mod)
        self.require_all = require_all
        self.implementations = {}
        self.lowered_matmuls = 0
        self.physical_jobs = 0
        self.padded_matmuls = 0
        self.role_matmuls = {}
        self.role_jobs = {}

    @staticmethod
    def _round_up(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    def _matrix_slice(self, expr, shape, batch_index, name_hint):
        leading_rank = len(shape) - 2
        if leading_rank == 0:
            return expr
        begin = [
            0 if shape[axis] == 1 else batch_index[axis]
            for axis in range(leading_rank)
        ]
        sliced = self.builder_.emit(
            relax.op.strided_slice(
                expr,
                axes=list(range(leading_rank)),
                begin=begin,
                end=[value + 1 for value in begin],
                assume_inbound=True,
            ),
            name_hint=f"{name_hint}_slice",
        )
        return self.builder_.emit(
            relax.op.reshape(sliced, shape[-2:]),
            name_hint=f"{name_hint}_matrix",
        )

    def _lower_matrix(self, lhs, rhs, m, n, k, role):
        physical_m = self._round_up(m, 16)
        physical_n = self._round_up(n, 16)
        physical_k = self._round_up(k, 32)
        if (physical_m, physical_n, physical_k) != (m, n, k):
            self.padded_matmuls += 1
            lhs_pad_key = ("tcu_pad", m, k, physical_m, physical_k)
            if lhs_pad_key not in self.implementations:
                self.implementations[lhs_pad_key] = self.builder_.add_func(
                    _make_pad_fp16_matrix(m, k, physical_m, physical_k),
                    f"vortex_tcu_pad_fp16_{m}_{k}_{physical_m}_{physical_k}",
                )
            lhs = self.builder_.emit(
                relax.call_tir(
                    self.implementations[lhs_pad_key],
                    [lhs],
                    out_ty=relax.TensorType((physical_m, physical_k), "float16"),
                ),
                name_hint="tcu_lhs_padded",
            )
            rhs_pad_key = ("tcu_pad", k, n, physical_k, physical_n)
            if rhs_pad_key not in self.implementations:
                self.implementations[rhs_pad_key] = self.builder_.add_func(
                    _make_pad_fp16_matrix(k, n, physical_k, physical_n),
                    f"vortex_tcu_pad_fp16_{k}_{n}_{physical_k}_{physical_n}",
                )
            rhs = self.builder_.emit(
                relax.call_tir(
                    self.implementations[rhs_pad_key],
                    [rhs],
                    out_ty=relax.TensorType((physical_k, physical_n), "float16"),
                ),
                name_hint="tcu_rhs_padded",
            )
        shape_key = (physical_m, physical_n, physical_k)
        if shape_key not in self.implementations:
            self.implementations[shape_key] = self.builder_.add_func(
                _make_fp16_tcu_matmul(physical_m, physical_n, physical_k),
                f"vortex_tcu_fp16_matmul_{physical_m}_{physical_n}_{physical_k}",
            )
        output = self.builder_.emit(
            relax.call_tir(
                self.implementations[shape_key],
                [lhs, rhs],
                out_ty=relax.TensorType((physical_m, physical_n), "float16"),
            ),
            name_hint="tcu_physical_output",
        )
        self.physical_jobs += 1
        self.role_jobs[role] = self.role_jobs.get(role, 0) + 1
        if (physical_m, physical_n) != (m, n):
            slice_key = ("tcu_slice", physical_m, physical_n, m, n)
            if slice_key not in self.implementations:
                self.implementations[slice_key] = self.builder_.add_func(
                    _make_slice_fp16_matrix(physical_m, physical_n, m, n),
                    f"vortex_tcu_slice_fp16_{physical_m}_{physical_n}_{m}_{n}",
                )
            output = self.builder_.emit(
                relax.call_tir(
                    self.implementations[slice_key],
                    [output],
                    out_ty=relax.TensorType((m, n), "float16"),
                ),
                name_hint="tcu_logical_output",
            )
        return output

    def visit_call_(self, call):
        call = super().visit_call_(call)
        role = "unattributed"
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "relax.matmul":
            lhs, rhs = call.args[:2]
        elif (
            isinstance(call.op, tvm.ir.Op)
            and call.op.name == "relax.call_pure_packed"
            and isinstance(call.args[0], relax.ExternFunc)
            and call.args[0].global_symbol == "relax.vortex.fp16_matmul"
        ):
            lhs, rhs = call.args[1:3]
            role = _prim_value(call.args[3])
            if not (role.startswith("linear.") or role in ("attention.qk", "attention.pv")):
                raise ValueError(f"unsupported Vortex FP16 TCU operation role: {role!r}")
        else:
            return call
        lhs_shape = _static_fp16_shape(lhs)
        rhs_shape = _static_fp16_shape(rhs)
        output_shape = _static_fp16_shape(call)
        if lhs_shape is None or rhs_shape is None or output_shape is None:
            if self.require_all:
                raise ValueError(
                    "Vortex FP16 TCU policy requires every matmul to have static FP16 shapes"
                )
            return call
        if not (len(lhs_shape) == len(rhs_shape) == len(output_shape)):
            if self.require_all:
                raise ValueError(
                    "Vortex FP16 TCU policy requires equal-rank explicit batched operands"
                )
            return call

        m, k = lhs_shape[-2:]
        rhs_k, n = rhs_shape[-2:]
        batch_shape = output_shape[:-2]
        if rhs_k != k or output_shape[-2:] != (m, n):
            raise ValueError("Vortex FP16 TCU matmul matrix shapes are inconsistent")
        for operand_name, operand_shape in (("lhs", lhs_shape), ("rhs", rhs_shape)):
            for axis, (operand_extent, output_extent) in enumerate(
                zip(operand_shape[:-2], batch_shape)
            ):
                if operand_extent not in (1, output_extent):
                    raise ValueError(
                        "Vortex FP16 TCU broadcast mismatch for "
                        f"{operand_name} axis {axis}: {operand_extent} versus {output_extent}"
                    )

        batch_indices = tuple(itertools.product(*(range(value) for value in batch_shape)))
        if not batch_indices:
            batch_indices = ((),)
        outputs = []
        for batch_index in batch_indices:
            matrix_lhs = self._matrix_slice(lhs, lhs_shape, batch_index, "tcu_lhs")
            matrix_rhs = self._matrix_slice(rhs, rhs_shape, batch_index, "tcu_rhs")
            matrix_output = self._lower_matrix(
                matrix_lhs, matrix_rhs, m, n, k, role
            )
            if batch_shape:
                matrix_output = self.builder_.emit(
                    relax.op.reshape(matrix_output, (1, m, n)),
                    name_hint="tcu_output_matrix",
                )
            outputs.append(matrix_output)
        self.lowered_matmuls += 1
        self.role_matmuls[role] = self.role_matmuls.get(role, 0) + 1
        if not batch_shape:
            return outputs[0]
        concatenated = (
            outputs[0]
            if len(outputs) == 1
            else self.builder_.emit(
                relax.op.concat(outputs, axis=0), name_hint="tcu_batched_output"
            )
        )
        return relax.op.reshape(concatenated, output_shape)


def _tcu_tensorize_pass(target: tvm.target.Target, require_all=False):
    """Rewrite static FP16 matmul to the versioned, padded Vortex TCU ABI."""

    mode = str(target.attrs.get("vortex_tcu_mode", "none"))
    formats = str(target.attrs.get("vortex_tcu_fp_formats", ""))
    enabled = mode in ("fp", "fp_int") and "fp16" in formats.split(",")

    @tvm.transform.module_pass(opt_level=0, name="VortexTensorizeTCU")
    def tensorize(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        if not enabled:
            if require_all:
                raise ValueError("Vortex FP16 TCU policy selected for a target without FP16 TCU")
            return mod
        lowerer = _PaddedFP16TCULowerer(mod, require_all=require_all)
        for global_var, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(global_var, lowerer.visit_expr(func))
        lowered = lowerer.builder_.get()
        if lowerer.lowered_matmuls:
            lowered = lowered.with_attr(
                "vortex.tcu.fp16.lowered_matmuls", lowerer.lowered_matmuls
            )
            lowered = lowered.with_attr(
                "vortex.tcu.fp16.physical_jobs", lowerer.physical_jobs
            )
            lowered = lowered.with_attr(
                "vortex.tcu.fp16.padded_matmuls", lowerer.padded_matmuls
            )
            for role, count in sorted(lowerer.role_matmuls.items()):
                attr_role = role.replace(".", "_")
                lowered = lowered.with_attr(
                    f"vortex.tcu.fp16.role.{attr_role}.matmuls", count
                )
                lowered = lowered.with_attr(
                    f"vortex.tcu.fp16.role.{attr_role}.jobs",
                    lowerer.role_jobs[role],
                )
        return lowered

    return tensorize


def library_dispatch_passes(target: tvm.target.Target):
    """Return library dispatch passes supported by Vortex."""
    return gpu_generic.library_dispatch_passes(target)


def legalize_passes(
    target: tvm.target.Target,
    layout_policy=None,
    inplace_kv_cache=False,
    backend_policy=None,
):  # pylint: disable=unused-argument
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
            layout_policy=layout_policy,
            inplace_kv_cache=inplace_kv_cache,
        ),
        _tcu_tensorize_pass(
            target,
            require_all=(
                backend_policy is not None
                and "fp16_tcu"
                in (backend_policy.linear_compute, backend_policy.attention_compute)
            ),
        ),
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


def dataflow_lower_passes(
    target: tvm.target.Target, layout_policy=None, inplace_kv_cache=False
):
    """Return Relax dataflow lowering passes for Vortex."""
    passes = gpu_generic.dataflow_lower_passes(target)[1:]
    improve_mode = str(target.attrs.get("vortex_gemm_mode", "none")) == "improve"
    return [
        passes[0],
        _w4a16_lowering_pass(
            target,
            lower_w4a16=not improve_mode,
            layout_policy=layout_policy,
            inplace_kv_cache=inplace_kv_cache,
        ),
        # Naive W4A16 lowering happens after the pipeline's main LegalizeOps
        # pass because the logical packed GEMM must survive until target
        # selection.  Batched/GQA expansion introduces static slices and
        # concats at this point, so legalize those newly-created Relax ops
        # before VM code generation.
        relax.transform.LegalizeOps(),
        *passes[1:],
    ]


def finalize_passes(target: tvm.target.Target):
    """Return Relax VM finalization passes for Vortex."""
    return gpu_generic.finalize_passes(target)


def get_default_pipeline(
    target: tvm.target.Target,
    layout_policy=None,
    inplace_kv_cache=False,
    backend_policy=None,
):
    """Return the default Relax compilation pipeline for Vortex."""

    policy = (
        None
        if backend_policy is None
        else validate_vortex_backend_policy(target, backend_policy)
    )
    if policy is not None and policy.layout_policy != "alone_or_fused" and layout_policy:
        raise ValueError(
            f"Vortex policy {policy.name!r} does not accept C4 layout policy "
            f"{layout_policy!r}"
        )

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            lowered = tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(
                    target,
                    layout_policy=layout_policy,
                    inplace_kv_cache=inplace_kv_cache,
                    backend_policy=policy,
                )
                + dataflow_lower_passes(
                    target,
                    layout_policy=layout_policy,
                    inplace_kv_cache=inplace_kv_cache,
                )
                + finalize_passes(target)
            )(mod)
            if policy is not None:
                lowered = lowered.with_attr("vortex.backend_policy", policy.name)
                lowered = lowered.with_attr(
                    "vortex.backend_workload_variant", policy.workload_variant
                )
            return lowered

    return _pipeline
