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


def _make_gemm_a_tiled(m, k):
    m_pad = (m + 7) // 8 * 8
    total = m * k

    @T.prim_func(private=True)
    def gemm_a_tiled(
        source: T.Buffer((m, k), "float16"),
        tiled: T.Buffer((m_pad * k,), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    tiled[
                        (bx * 128 + tx) % k // 32 * m_pad * 32
                        + (bx * 128 + tx) // k * 32
                        + (bx * 128 + tx) % 32
                    ] = source[(bx * 128 + tx) // k, (bx * 128 + tx) % k]

    return gemm_a_tiled


def _make_gemm_w_tiled(rhs_shape, transpose_rhs):
    rows, columns = rhs_shape
    total_bytes = rows * ((columns + 1) // 2)

    @T.prim_func(private=True)
    def gemm_w_tiled(
        source: T.Buffer((rows, (columns + 1) // 2), "uint8"),
        tiled: T.Buffer((total_bytes,), "uint8"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total_bytes + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total_bytes:
                    if transpose_rhs:
                        tiled[
                            (bx * 128 + tx)
                            // ((columns + 1) // 2)
                            // 32
                            * (columns * 16)
                            + (bx * 128 + tx) % ((columns + 1) // 2) // 16 * (32 * 16)
                            + (bx * 128 + tx) // ((columns + 1) // 2) % 32 * 16
                            + (bx * 128 + tx) % 16
                        ] = source[
                            (bx * 128 + tx) // ((columns + 1) // 2),
                            (bx * 128 + tx) % ((columns + 1) // 2),
                        ]
                    else:
                        tiled[
                            (bx * 128 + tx) // ((columns + 1) // 2) // 32 * (32 * 16)
                            + (bx * 128 + tx) // ((columns + 1) // 2) % 32 * 16
                            + (bx * 128 + tx) % ((columns + 1) // 2)
                        ] = source[
                            (bx * 128 + tx) // ((columns + 1) // 2),
                            (bx * 128 + tx) % ((columns + 1) // 2),
                        ]

    return gemm_w_tiled


def _make_gemm_qparam_tiled(
    source_shape, output_elements, dtype, quant_direction, transpose_rhs
):
    total = source_shape[0] * source_shape[1]

    @T.prim_func(private=True)
    def gemm_qparam_tiled(
        source: T.Buffer(source_shape, dtype),
        tiled: T.Buffer((output_elements,), dtype),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((output_elements + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < output_elements:
                    tiled[bx * 128 + tx] = T.Cast(dtype, 0)
                if bx * 128 + tx < total:
                    if quant_direction == 0 and transpose_rhs:
                        tiled[
                            (bx * 128 + tx) % source_shape[1] * source_shape[0]
                            + (bx * 128 + tx) // source_shape[1]
                        ] = source[
                            (bx * 128 + tx) // source_shape[1],
                            (bx * 128 + tx) % source_shape[1],
                        ]
                    else:
                        tiled[bx * 128 + tx] = source[
                            (bx * 128 + tx) // source_shape[1],
                            (bx * 128 + tx) % source_shape[1],
                        ]

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
):
    @T.prim_func(private=True)
    def mm_w4a16_improve(
        lhs: T.Buffer((tiled_a_elements,), "float16"),
        packed: T.Buffer((tiled_w_bytes,), "uint8"),
        scale: T.Buffer((tiled_qparam_elements,), "float16"),
        zero_point: T.Buffer((tiled_qparam_elements,), "int16"),
        output: T.Buffer((tiled_c_elements,), "float16"),
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
                        2,
                    )
                )

    return mm_w4a16_improve


def _make_gemm_c_detile(m, n):
    m_pad = (m + 7) // 8 * 8
    total = m * n

    @T.prim_func(private=True)
    def gemm_c_detile(
        tiled: T.Buffer((m_pad * n,), "float16"),
        output: T.Buffer((m, n), "float16"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for bx in T.thread_binding((total + 127) // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                if bx * 128 + tx < total:
                    output[(bx * 128 + tx) // n, (bx * 128 + tx) % n] = tiled[
                        (bx * 128 + tx) % n // 32 * m_pad * 32
                        + (bx * 128 + tx) // n * 32
                        + (bx * 128 + tx) % 32
                    ]

    return gemm_c_detile


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

    def __init__(self, mod, target):
        super().__init__(mod)
        self.target = target
        self.mode = str(target.attrs.get("vortex_gemm_mode", "none"))
        self.implementations = {}
        self.original_bindings = {}
        self.tiled_w4a16_outputs = {}
        for _, function in mod.functions_items():
            if not isinstance(function, relax.Function) or not isinstance(
                function.body, relax.SeqExpr
            ):
                continue
            for block in function.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding, relax.VarBinding):
                        self.original_bindings[binding.var] = binding.value

    def visit_call_(self, call):
        original_call = call
        fused_tiled_lhs = None
        if (
            self.mode == "improve"
            and isinstance(call.op, tvm.ir.Op)
            and call.op.name == "relax.call_pure_packed"
            and isinstance(call.args[0], relax.ExternFunc)
            and call.args[0].global_symbol == "relax.vortex.mm_w4a16"
            and isinstance(call.args[1], relax.Var)
        ):
            producer = self.original_bindings.get(call.args[1])
            fused_tiled_lhs = self.tiled_w4a16_outputs.get(producer)
        call = super().visit_call_(call)
        if (
            not isinstance(call.op, tvm.ir.Op)
            or call.op.name != "relax.call_pure_packed"
            or not isinstance(call.args[0], relax.ExternFunc)
        ):
            return call
        symbol = call.args[0].global_symbol
        if symbol == "relax.vortex.quantize_int4":
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
        if (
            m > 128
            or n > 128
            or k != 128
            or n % 32 != 0
            or m % 8 != 0
            or pack_axis != 1
        ):
            raise ValueError(
                "Vortex improved W4A16 initial lowering requires M<=128 divisible by 8, "
                "N<=128 divisible by 32, K=128, and pack_axis=1"
            )
        source_k_axis = 1 if transpose_rhs else 0
        quant_direction = 0 if quant_axis == source_k_axis else 1
        m_pad = (m + 7) // 8 * 8
        tiled_a_elements = m_pad * k
        tiled_w_bytes = packed_shape[0] * packed_shape[1]
        tiled_qparam_elements = (
            (scale_shape[0] * scale_shape[1] * 2 + 511) // 512
        ) * 256
        tiled_c_elements = m_pad * n

        layout_specs = {
            "w": (
                _make_gemm_w_tiled(rhs_shape, transpose_rhs),
                (
                    "vortex_gemm_w_tiled_transposed"
                    if transpose_rhs
                    else "vortex_gemm_w_tiled"
                ),
            ),
            "scale": (
                _make_gemm_qparam_tiled(
                    scale_shape,
                    tiled_qparam_elements,
                    "float16",
                    quant_direction,
                    transpose_rhs,
                ),
                "vortex_gemm_scale_tiled",
            ),
            "zero": (
                _make_gemm_qparam_tiled(
                    zero_point_shape,
                    tiled_qparam_elements,
                    "int16",
                    quant_direction,
                    transpose_rhs,
                ),
                "vortex_gemm_zero_point_tiled",
            ),
            "gemm": (
                _make_w4a16_improve(
                    tiled_a_elements,
                    tiled_w_bytes,
                    tiled_qparam_elements,
                    tiled_c_elements,
                    m,
                    n,
                    k,
                    group_size,
                    int(transpose_rhs),
                    quant_direction,
                ),
                "vortex_mm_w4a16_improve",
            ),
            "detile": (_make_gemm_c_detile(m, n), "vortex_gemm_c_detile"),
        }
        if fused_tiled_lhs is None:
            layout_specs["a"] = (
                _make_gemm_a_tiled(m, k),
                f"vortex_gemm_a_tiled_{m}_{k}",
            )
        elif fused_tiled_lhs[1] != lhs_shape:
            raise ValueError(
                "Vortex fused GEMM-C to GEMM-A logical shapes are inconsistent"
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
        tiled_w = self.builder_.emit(
            relax.call_tir(
                globals_by_name["w"],
                [packed],
                out_ty=relax.TensorType((tiled_w_bytes,), "uint8"),
            ),
            name_hint="gemm_w_tiled",
        )
        tiled_scale = self.builder_.emit(
            relax.call_tir(
                globals_by_name["scale"],
                [scale],
                out_ty=relax.TensorType((tiled_qparam_elements,), "float16"),
            ),
            name_hint="gemm_scale_tiled",
        )
        tiled_zero = self.builder_.emit(
            relax.call_tir(
                globals_by_name["zero"],
                [zero_point],
                out_ty=relax.TensorType((tiled_qparam_elements,), "int16"),
            ),
            name_hint="gemm_zero_point_tiled",
        )
        tiled_output = self.builder_.emit(
            relax.call_tir(
                globals_by_name["gemm"],
                [tiled_a, tiled_w, tiled_scale, tiled_zero],
                out_ty=relax.TensorType((tiled_c_elements,), "float16"),
            ),
            name_hint="gemm_c_tiled",
        )
        detiled_output = relax.call_tir(
            globals_by_name["detile"],
            [tiled_output],
            out_ty=call.ty,
        )
        self.tiled_w4a16_outputs[original_call] = (tiled_output, output_shape)
        return detiled_output


def _w4a16_lowering_pass(target):
    @tvm.transform.module_pass(opt_level=0, name="VortexLowerW4A16")
    def lower(mod, _ctx):
        lowerer = _W4A16Lowerer(mod, target)
        for global_var, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(global_var, lowerer.visit_expr(func))
        return lowerer.builder_.get()

    return lower


def library_dispatch_passes(target: tvm.target.Target):
    """Return library dispatch passes supported by Vortex."""
    return gpu_generic.library_dispatch_passes(target)


def legalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """Legalize Relax and schedule kernels for Vortex."""
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    return [
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
    passes = gpu_generic.dataflow_lower_passes(target)
    # Preserve logical W4A16 calls through the generic reshape/dataflow
    # rewrites, then materialize their pre-scheduled launch chain before
    # RemovePurityChecking changes call_pure_packed into call_packed and before
    # CallTIRRewrite lowers calls into the VM/runtime calling convention.
    return [*passes[:2], _w4a16_lowering_pass(target), *passes[2:]]


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
