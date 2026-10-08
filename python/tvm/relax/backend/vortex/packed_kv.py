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
"""Persistent C4 KV storage and quantization-store fusion.

Packed buffers are explicit, caller-owned state. They must start zeroed and be
appended consecutively; reset all buffers before starting a new sequence.
"""

import json
import math

import tvm
from tvm import relax, tirx

from .layout import plan_improve_layout


def make_quantize_cache_store(quantize, cache_shapes, tokens, plan, initialize=False):
    """Retarget the existing quantizer's output loads/stores, preserving arithmetic."""
    source = quantize.params[0]
    old = list(quantize.params[1:])
    caches = [
        tirx.decl_buffer(s, b.dtype, name="cache_" + b.name) for s, b in zip(cache_shapes, old)
    ]
    heads = math.prod(cache_shapes[0][:2])
    sizes = [plan.weight_bytes, plan.qparam_elements, plan.qparam_elements]
    packed = [
        tirx.decl_buffer((heads * size,), b.dtype, name="tiled_" + b.name)
        for size, b in zip(sizes, old)
    ]
    position = tirx.decl_buffer((), "int64", name="position")
    p = plan.profile
    capacity = cache_shapes[0][-2]
    initial = [tirx.decl_buffer(b.shape, b.dtype, name="initial_" + b.name) for b in caches]
    initialization = []
    if initialize:
        for src, dst in zip(initial, caches):
            index = tirx.Var("init_index", "int32")
            # Each buffer has its own final column count.
            coords = [
                index // math.prod([int(v) for v in dst.shape[axis + 1 :]]) % dst.shape[axis]
                for axis in range(4)
            ]
            initialization.append(
                tirx.For(
                    index,
                    0,
                    math.prod([int(v) for v in dst.shape]),
                    tirx.ForKind.SERIAL,
                    tirx.BufferStore(dst, tirx.BufferLoad(src, coords), coords),
                )
            )

    def coordinates(indices):
        row, col = indices
        head = row // tokens
        token = tirx.Cast("int32", position[()]) + row % tokens
        return head, token, col

    def canonical(buf, indices):
        head, token, col = coordinates(indices)
        return [head // cache_shapes[0][1], head % cache_shapes[0][1], token, col]

    def weight_offset(token, pair):
        k, n = (pair * 2, token) if plan.weight_transpose else (token, pair * 2)
        kt, local_k = k // p.dma_kt, k % p.dma_kt
        cur_k = tirx.min(p.dma_kt, plan.execution_k - kt * p.dma_kt)
        offset = kt * p.dma_kt * plan.execution_n // 2 + n // p.mxu_nt * cur_k * p.mxu_nt // 2
        if plan.weight_transpose:
            return (
                offset
                + local_k // p.mxu_kt * p.mxu_nt * (p.mxu_kt // 2)
                + n % p.mxu_nt * (p.mxu_kt // 2)
                + local_k % p.mxu_kt // 2
            )
        return offset + local_k * (p.mxu_nt // 2) + n % p.mxu_nt // 2

    def packed_stores(kind, value, indices):
        head, token, col = coordinates(indices)
        base = head * sizes[kind]
        if kind == 0:
            return tirx.BufferStore(packed[kind], value, [base + weight_offset(token, col)])
        stores = []
        full_k_row = (
            sum(slot.reserved_bytes for slot in plan.qparam_slots if slot.outer_k == 0) // 2
        )
        repeats = max(1, plan.qblock // p.mxu_nt) if not plan.weight_transpose else 1
        for repeat in range(repeats):
            if plan.weight_transpose:
                kt, nt = col // (p.dma_kt // plan.qblock), token // p.dma_nt
                cur_k = tirx.min(p.dma_kt, plan.execution_k - kt * p.dma_kt)
                groups = cur_k // plan.qblock
                payload_bytes = groups * p.dma_nt * 2
                off = (
                    token % p.dma_nt // p.mxu_nt * groups * p.mxu_nt
                    + col % (p.dma_kt // plan.qblock) * p.mxu_nt
                    + token % p.mxu_nt
                )
            else:
                n = col * plan.qblock + repeat * p.mxu_nt
                kt, nt = token // p.dma_kt, n // p.dma_nt
                cur_k = tirx.min(p.dma_kt, plan.execution_k - kt * p.dma_kt)
                ng = (p.mxu_nt + plan.qblock - 1) // plan.qblock
                payload_bytes = p.dma_nt // p.mxu_nt * cur_k * ng * 2
                off = n % p.dma_nt // p.mxu_nt * cur_k * ng + token % p.dma_kt * ng + col % ng
            slot_size = (
                (payload_bytes + p.qparam_slot_alignment - 1)
                // p.qparam_slot_alignment
                * p.qparam_slot_alignment
                // 2
            )
            store = tirx.BufferStore(
                packed[kind], value, [base + kt * full_k_row + nt * slot_size + off]
            )
            if not plan.weight_transpose:
                store = tirx.IfThenElse(n < plan.logical_n, store, None)
            stores.append(store)
        return stores[0] if len(stores) == 1 else tirx.SeqStmt(stores)

    def rewrite(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            for kind, buf in enumerate(old):
                if node.buffer.same_as(buf):
                    indices = canonical(buf, node.indices)
                    if isinstance(node, tirx.BufferLoad):
                        return tirx.BufferLoad(caches[kind], indices)
                    return tirx.SeqStmt(
                        [
                            tirx.BufferStore(caches[kind], node.value, indices),
                            packed_stores(kind, node.value, node.indices),
                        ]
                    )
        return None

    body = tirx.stmt_functor.ir_transform(
        quantize.body, None, rewrite, ["tirx.BufferLoad", "tirx.BufferStore"]
    )

    def guard_thread(node):
        if (
            isinstance(node, tirx.For)
            and node.thread_binding is not None
            and node.thread_binding.thread_tag == "threadIdx.x"
        ):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                tirx.IfThenElse(
                    tirx.all(position[()] >= 0, position[()] + tokens <= capacity),
                    tirx.SeqStmt([*initialization, node.body]) if initialize else node.body,
                    None,
                ),
                node.thread_binding,
                node.annotations,
            )
        return None

    body = tirx.stmt_functor.ir_transform(body, None, guard_thread, ["tirx.For"])
    buffers = (
        [source, *initial, *packed, position, *caches]
        if initialize
        else [source, *caches, *packed, position]
    )
    return tirx.PrimFunc(buffers, body, attrs=quantize.attrs)


class PackedKVLowering:
    """Optional lowering support mixed into the existing logical GEMM lowerer."""

    def _cache_source(self, expr, slices=False):
        seen = set()
        while expr not in seen:
            seen.add(expr)
            if isinstance(expr, relax.Var) and expr in self.original_bindings:
                expr = self.original_bindings[expr]
            elif (
                isinstance(expr, relax.Call)
                and isinstance(expr.op, tvm.ir.Op)
                and expr.op.name
                in (
                    "relax.reshape",
                    "relax.expand_dims",
                    *(["relax.strided_slice"] if slices else []),
                )
            ):
                expr = expr.args[0]
            else:
                break
        return expr

    def _init_packed_kv(self, mod):
        from .pipeline import _prim_value, _static_tensor_shape

        self.packed_kv_updates = {}
        self.packed_kv_gemms = {}
        self.packed_kv_states = []
        self.packed_kv_results = {}
        self.packed_kv_params = []
        if not self.dynamic_kv_length or not self.enable_layout_fusion:
            raise ValueError("packed KV requires dynamic_kv_length=True and fused layout")
        owners = {}
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function) and isinstance(func.body, relax.SeqExpr):
                for block in func.body.blocks:
                    for binding in block.bindings:
                        if isinstance(binding, relax.VarBinding):
                            owners[binding.value] = gv.name_hint
        by_quant = {}
        for mm, (_, axis) in self.dynamic_prefixes.items():
            fields = [self._cache_source(x) for x in mm.args[2:5]]
            if not all(
                isinstance(x, relax.TupleGetItem) and x.index == i for i, x in enumerate(fields)
            ):
                continue
            update = self._cache_source(fields[0].tuple_value)
            if self._packed_symbol(update) not in (
                "relax.vortex.kv_cache_update_dynamic",
                "relax.vortex.kv_cache_update",
            ):
                continue
            if not all(self._cache_source(x.tuple_value).same_as(update) for x in fields):
                raise ValueError("packed KV payload and qparams must share a cache update")
            chain = []
            cursor = update
            while self._packed_symbol(cursor) in (
                "relax.vortex.kv_cache_update_dynamic",
                "relax.vortex.kv_cache_update",
            ):
                chain.append(cursor)
                parent = self._cache_source(cursor.args[1])
                if not isinstance(parent, relax.TupleGetItem):
                    break
                cursor = self._cache_source(parent.tuple_value)
            first = chain[-1]
            quant_fields = [self._cache_source(x, slices=True) for x in first.args[4:7]]
            if not all(
                isinstance(x, relax.TupleGetItem) and x.index == i
                for i, x in enumerate(quant_fields)
            ):
                raise ValueError("packed KV cache update must come directly from quantization")
            quant = self._cache_source(quant_fields[0].tuple_value)
            if self._packed_symbol(quant) != "relax.vortex.quantize_int4":
                raise ValueError("packed KV cache update requires an FP16 quantization producer")
            if not all(self._cache_source(x.tuple_value).same_as(quant) for x in quant_fields):
                raise ValueError("packed KV quantization tuple mismatch")
            shapes = [
                _static_tensor_shape(x, d)
                for x, d in zip(first.args[1:4], ("uint8", "float16", "int16"))
            ]
            if any(s is None or len(s) != 4 for s in shapes):
                raise ValueError("packed KV requires rank-4 canonical cache tensors")
            source_shape = _static_tensor_shape(quant.args[1], "float16")
            batch, heads, capacity, half_dim = shapes[0]
            if (
                source_shape is None
                or source_shape[1] != half_dim * 2
                or source_shape[0] % (batch * heads)
            ):
                raise ValueError("packed KV quantizer shape mismatch")
            tokens = source_shape[0] // (batch * heads)
            dynamic = self._packed_symbol(first).endswith("_dynamic")
            if dynamic:
                if len(chain) != 1 or tokens != 1:
                    raise ValueError("packed KV decode must append one token")
            else:
                if len(chain) != tokens or [
                    int(_prim_value(x.args[7])) for x in reversed(chain)
                ] != list(range(tokens)):
                    raise ValueError(
                        "packed KV prefill must append a contiguous prefix starting at zero"
                    )
            for chain_index, item in enumerate(chain):
                for field_index, expr in enumerate(item.args[4:7]):
                    field = self._cache_source(expr, slices=True)
                    if not (
                        isinstance(field, relax.TupleGetItem)
                        and field.index == field_index
                        and self._cache_source(field.tuple_value).same_as(quant)
                    ):
                        raise ValueError("packed KV prefix must share one quantization producer")
                    if not dynamic and tokens > 1:
                        sliced = self._cache_source(expr)
                        expected_position = int(_prim_value(item.args[7]))
                        if not (
                            isinstance(sliced, relax.Call)
                            and isinstance(sliced.op, tvm.ir.Op)
                            and sliced.op.name == "relax.strided_slice"
                            and [
                                tuple(int(_prim_value(v)) for v in arg.fields)
                                for arg in sliced.args[1:5]
                            ]
                            == [(2,), (expected_position,), (expected_position + 1,), (1,)]
                        ):
                            raise ValueError("packed KV prefill slices must match append positions")
                if chain_index + 1 < len(chain):
                    for field_index, expr in enumerate(item.args[1:4]):
                        field = self._cache_source(expr)
                        if not (
                            isinstance(field, relax.TupleGetItem)
                            and field.index == field_index
                            and self._cache_source(field.tuple_value).same_as(
                                chain[chain_index + 1]
                            )
                        ):
                            raise ValueError("packed KV prefix must have a consistent cache tuple")
            group = int(_prim_value(quant.args[3]))
            if (
                group != int(_prim_value(mm.args[6]))
                or int(_prim_value(quant.args[2])) != 1
                or int(_prim_value(quant.args[4])) != 1
            ):
                raise ValueError("packed KV quantization contract mismatch")
            dim = source_shape[1]
            qshape = (batch, heads, capacity, (dim + group - 1) // group)
            if shapes[1] != qshape or shapes[2] != qshape:
                raise ValueError("packed KV scale/zero cache shapes do not match quantization")
            trans = axis == "N"
            plan = plan_improve_layout(
                1,
                capacity if trans else dim,
                dim if trans else capacity,
                group,
                trans,
                0 if trans else 1,
                self.improve_profile,
            )
            if quant in by_quant:
                info = by_quant[quant]
                if not info["first"].same_as(first):
                    raise ValueError("a KV quantizer must update one owned cache")
                if info["plan"] != plan:
                    raise ValueError("a KV quantizer cannot have incompatible physical layouts")
            else:
                index = len(self.packed_kv_states)
                sizes = [plan.weight_bytes, plan.qparam_elements, plan.qparam_elements]
                params = [
                    relax.Var(
                        f"packed_kv_{index}_{name}",
                        relax.TensorType((batch * heads * size,), dtype),
                    )
                    for name, dtype, size in zip(
                        ("payload", "scale", "zero"), ("uint8", "float16", "int16"), sizes
                    )
                ]
                info = dict(
                    index=index,
                    first=first,
                    quant=quant,
                    shapes=shapes,
                    tokens=tokens,
                    plan=plan,
                    params=params,
                    dynamic=dynamic,
                    owner=owners[mm],
                )
                by_quant[quant] = info
                self.packed_kv_params.extend(params)
                self.packed_kv_states.append(info)
            for item in chain:
                self.packed_kv_updates[item] = info
            self.packed_kv_gemms[mm] = info
        if not self.packed_kv_states:
            raise ValueError("packed KV requested but no quantize-cache-attention chain was found")
        if len({x["owner"] for x in self.packed_kv_states}) != 1:
            raise ValueError("packed KV preparation expects one exported entry function")

    def packed_kv_metadata(self):
        return json.dumps(
            [
                dict(
                    index=s["index"],
                    transpose=s["plan"].weight_transpose,
                    capacity=s["shapes"][0][-2],
                    head_dim=s["shapes"][0][-1] * 2,
                    buffers=[
                        dict(
                            name=str(p),
                            shape=[int(x) for x in p.ty.shape.values],
                            dtype=str(p.ty.dtype),
                        )
                        for p in s["params"]
                    ],
                )
                for s in self.packed_kv_states
            ]
        )

    def _emit_packed_kv_update(self, original_call, call):
        from .pipeline import _make_quantize_int4_row_major, _prim_value, _static_tensor_shape

        info = self.packed_kv_updates[original_call]
        if info["index"] in self.packed_kv_results:
            return relax.Tuple(self.packed_kv_results[info["index"]][:3])
        quant = info["quant"]
        source = self.visit_expr(quant.args[1])
        kernel = make_quantize_cache_store(
            _make_quantize_int4_row_major(
                _static_tensor_shape(source, "float16"),
                info["plan"].qblock,
                _prim_value(quant.args[5]),
            ),
            info["shapes"],
            info["tokens"],
            info["plan"],
            initialize=not info["dynamic"],
        )
        gv = self.builder_.add_func(kernel, "vortex_quantize_packed_kv_append")
        position = call.args[7] if info["dynamic"] else relax.const(0, "int64")
        buffers = [*call.args[1:4], *info["params"]]
        result = self.builder_.emit(
            relax.call_tir_inplace(
                gv,
                [source, *buffers, position],
                inplace_indices=list(range(1, 7)) if info["dynamic"] else [-1, -1, -1, 4, 5, 6],
                out_ty=[x.ty for x in buffers],
            )
        )
        fields = [self.builder_.emit(relax.TupleGetItem(result, i)) for i in range(6)]
        self.packed_kv_results[info["index"]] = fields
        return relax.Tuple(fields[:3])


def prepare_packed_kv_cache(mod, target):
    """Fuse quantization/cache stores and append caller-owned packed state arguments.

    Returns the rewritten module and JSON-compatible buffer descriptors in argument
    order. Allocate these buffers zeroed once per layer/sequence, pass them after
    the original arguments to both prefill and decode, and retain them across calls.
    Decode canonical cache inputs are updated in place and require unique ownership.
    Prefill creates independent canonical outputs even if its zero inputs alias.
    """
    from .pipeline import _rewrite_dataflow_reshape_before_vortex, _w4a16_lowering_pass

    with target:
        mod = _rewrite_dataflow_reshape_before_vortex()(mod)
        mod = _w4a16_lowering_pass(target, dynamic_kv_length=True, packed_kv_cache=True)(mod)
    return mod, json.loads(str(mod.attrs["vortex.packed_kv_cache"]))
