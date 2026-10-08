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
"""Compile and validate persistent quantization-fused KV attention on C4."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.relax.backend.vortex import get_default_pipeline
from tvm.relax.backend.vortex.layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)
from tvm.relax.backend.vortex.packed_kv import prepare_packed_kv_cache
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile

sys.path.insert(0, str(Path(os.environ["TVM_VORTEX_HOME"]) / "pytorch/spinquant"))
from spinquant_inference import vortex_export_ops  # noqa: F401

from vortex_llama3.run_backend_validation import LOCAL_THRESHOLDS, _runtime_tensor, hybrid_metrics


class AppendAttention(torch.nn.Module):
    def __init__(self, capacity=320, tokens=1, prefill=False, batch=1, heads=1, query_groups=2):
        super().__init__()
        self.capacity, self.tokens, self.prefill = capacity, tokens, prefill
        self.batch, self.heads = batch, heads
        self.query_groups = query_groups

    def forward(self, query, key_new, value_new, kp, ks, kz, vp, vs, vz, pos):
        # Export may alias clone() of the initial zero cache in real Llama prefill.
        # Exercise that case explicitly; fused prefill must allocate separate outputs.
        if self.prefill:
            vp, vs, vz = kp, ks, kz
        caches = []
        for source, cache in ((key_new, (kp, ks, kz)), (value_new, (vp, vs, vz))):
            q = torch.ops.vortex.quantize_int4(
                source.reshape(-1, 128), 1, 128, 1, "signed_asymmetric_int4"
            )
            q = tuple(x.reshape(self.batch, self.heads, self.tokens, -1) for x in q)
            if self.prefill:
                for index in range(self.tokens):
                    cache = torch.ops.vortex.kv_cache_update(
                        *cache, *(x[:, :, index : index + 1] for x in q), index, self.capacity
                    )
            else:
                cache = torch.ops.vortex.kv_cache_update_dynamic(*cache, *q, pos, self.capacity)
            caches.append(cache)
        length = pos + self.tokens
        positions = (pos.reshape(1, 1) + torch.arange(self.tokens).reshape(1, -1)).expand(
            self.batch, -1
        )
        shape = [self.batch, self.heads, 1, self.capacity, 128]
        operands = [tuple(x.unsqueeze(2) for x in cache) for cache in caches]
        scores = torch.ops.vortex.mm_w4a16(
            query, *operands[0], shape, 128, 4, 4, "signed_asymmetric_int4", True
        )
        _, probability = torch.ops.vortex.causal_softmax(scores, positions, length, 128)
        context = torch.ops.vortex.mm_w4a16(
            probability, *operands[1], shape, 128, 4, 4, "signed_asymmetric_int4", False
        )
        return (probability, context, *caches[0], *caches[1])


def append_inputs(capacity=320, tokens=1, batch=1, heads=1, query_groups=2):
    torch.manual_seed(42)
    # Broad V range keeps quantization scales out of the FP16 underflow regime.
    return (
        torch.full((batch, heads, query_groups, tokens, 128), 0.125, dtype=torch.float16),
        torch.rand(batch, heads, tokens, 128, dtype=torch.float16) * 4 + 1,
        torch.rand(batch, heads, tokens, 128, dtype=torch.float16) * 4 + 1,
        *(
            torch.zeros(batch, heads, capacity, cols, dtype=dtype)
            for _ in range(2)
            for cols, dtype in ((64, torch.uint8), (1, torch.float16), (1, torch.int16))
        ),
        torch.tensor(0, dtype=torch.int64),
    )


def build_probe(
    capacity, tokens, prefill, target, output_dir, batch=1, heads=1, query_groups=2, packed=True
):
    model = AppendAttention(capacity, tokens, prefill, batch, heads, query_groups)
    sample = append_inputs(capacity, tokens, batch, heads, query_groups)
    exported = torch.export.export(model, sample, strict=True)
    mod = from_exported_program(exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True)
    descriptors = []
    if packed:
        mod, descriptors = prepare_packed_kv_cache(mod, target)
    label = "prefill" if prefill else "decode"
    (output_dir / (label + ".lowered.py")).write_text(mod.script())
    exe = relax.build(
        mod,
        target,
        relax_pipeline=get_default_pipeline(
            target, layout_policy="fused", dynamic_kv_length=not packed
        ),
    )
    exe.export_library(str(output_dir / (label + ".so")))

    def source(module):
        if module.kind == "vortex":
            return [module.inspect_source()]
        return [text for child in module.imports for text in source(child)]

    (output_dir / (label + ".device.cc")).write_text("\n".join(source(exe.mod)))
    return model, exe, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=320)
    parser.add_argument("--prefill", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--query-heads-per-kv", type=int, default=2)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_vortex_accelerator_profile(args.manifest)
    target = profile.target
    compiled = {}
    all_descriptors = []
    for prefill, tokens in ((True, args.prefill), (False, 1)):
        label = "prefill" if prefill else "decode"
        if args.run_only:
            model = AppendAttention(
                args.capacity,
                tokens,
                prefill,
                args.batch_size,
                args.kv_heads,
                args.query_heads_per_kv,
            )
            exe = tvm.runtime.load_module(str(args.output_dir / (label + ".so")))
            desc = json.loads((args.output_dir / (label + ".json")).read_text())
        else:
            model, exe, desc = build_probe(
                args.capacity,
                tokens,
                prefill,
                target,
                args.output_dir,
                args.batch_size,
                args.kv_heads,
                args.query_heads_per_kv,
            )
            (args.output_dir / (label + ".json")).write_text(json.dumps(desc, indent=2))
        compiled[prefill] = (model, exe)
        all_descriptors.append(desc)
    assert all_descriptors[0] == all_descriptors[1]
    if args.compile_only:
        return
    device = tvm.vortex(0)
    vms = {
        key: relax.VirtualMachine(exe, device=device, memory_cfg="naive")
        for key, (_, exe) in compiled.items()
    }
    packed = [
        _runtime_tensor(np.zeros(b["shape"], b["dtype"]), device)
        for desc in all_descriptors[0]
        for b in desc["buffers"]
    ]
    model, _ = compiled[True]
    initial = append_inputs(
        args.capacity, args.prefill, args.batch_size, args.kv_heads, args.query_heads_per_kv
    )
    cache_device = [_runtime_tensor(x.numpy(), device) for x in initial[3:9]]
    cache_cpu = initial[3:9]
    rng = np.random.default_rng(20261008)
    checkpoints = {args.prefill, 15, 16, 17, 127, 128, 129, 255, 256, 257, 288, 320, 511, 512}
    maximum = args.max_length or args.capacity
    if not args.prefill <= maximum <= args.capacity:
        raise ValueError("require prefill <= max length <= capacity")
    checkpoints.add(maximum)
    records = []
    positions = [0, *range(args.prefill, maximum)]
    for pos in positions:
        prefill = pos == 0
        tokens = args.prefill if prefill else 1
        inputs = list(
            append_inputs(
                args.capacity, tokens, args.batch_size, args.kv_heads, args.query_heads_per_kv
            )
        )
        for i in (1, 2):
            inputs[i] = torch.from_numpy(rng.uniform(1, 5, inputs[i].shape).astype("float16"))
        inputs[3:9] = cache_cpu
        inputs[-1] = torch.tensor(pos, dtype=torch.int64)
        model, _ = compiled[prefill]
        expected = model(*inputs)
        cache_cpu = tuple(x.clone() for x in expected[2:])
        device_inputs = (
            [_runtime_tensor(x.numpy(), device) for x in inputs[:3]]
            + cache_device
            + [_runtime_tensor(inputs[-1].numpy(), device)]
        )
        start = time.perf_counter()
        result = vms[prefill]["main"](*device_inputs, *packed)
        cache_device = list(result[2:])
        elapsed = time.perf_counter() - start
        length = pos + tokens
        if length not in checkpoints:
            continue
        np.savez(
            args.output_dir / f"length_{length}.npz",
            **{f"actual_{i}": x.numpy() for i, x in enumerate(result)},
            **{f"expected_{i}": x.numpy() for i, x in enumerate(expected)},
        )
        record = {"length": length, "seconds": elapsed, "metrics": []}
        for i, name in enumerate(("probability", "context")):
            actual = result[i].numpy()
            record["metrics"].append(
                hybrid_metrics(
                    actual, expected[i].numpy(), LOCAL_THRESHOLDS, name=name, enforce=False
                )
            )
        # Compare packed state against the existing independent CPU packing implementation.
        for index, trans in enumerate((True, False)):
            plan = plan_improve_layout(
                1,
                args.capacity if trans else 128,
                128 if trans else args.capacity,
                128,
                trans,
                0 if trans else 1,
                ImproveProfile.from_target(target),
            )
            for batch in range(args.batch_size):
                for head in range(args.kv_heads):
                    actual_cache = [
                        result[2 + index * 3 + i].numpy()[batch, head] for i in range(3)
                    ]
                    refs = [
                        prepack_improve_weight(actual_cache[0], plan),
                        prepack_improve_qparam(actual_cache[1], plan, "float16"),
                        prepack_improve_qparam(actual_cache[2], plan, "int16"),
                    ]
                    for i in range(3):
                        flat_head = batch * args.kv_heads + head
                        actual = (
                            packed[index * 3 + i]
                            .numpy()
                            .reshape(args.batch_size * args.kv_heads, -1)[flat_head]
                        )
                        np.testing.assert_array_equal(actual, refs[i])
        # Confirm the chosen numerical fixture avoids the known scaler FTZ regime.
        probability = result[0].numpy()[..., :length]
        scale = result[6].numpy()[:, :, None, None, :length, 0]
        products = probability.astype("float32") * scale
        count = int(np.count_nonzero((np.abs(products) > 0) & (np.abs(products) < 2**-14)))
        record["pv_subnormal_products"] = count
        if count:
            raise AssertionError(f"test fixture produced {count} subnormal PV products")
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        if not all(x["pass"] for x in record["metrics"]):
            raise AssertionError("CPU reference limits exceeded")
    print("PASS persistent packed KV", flush=True)


if __name__ == "__main__":
    main()
