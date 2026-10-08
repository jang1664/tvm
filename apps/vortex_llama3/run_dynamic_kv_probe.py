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
"""Validate one compiled attention graph across runtime KV prefix lengths.

Run under an FPGA Slurm allocation with the normal TVM/Vortex environment.
Inactive K/V scales deliberately contain NaN/Inf on the dynamic path.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.backend.vortex import get_default_pipeline
from tvm.support.vortex import load_vortex_accelerator_profile

sys.path.insert(0, str(Path(os.environ["TVM_VORTEX_HOME"]) / "pytorch/spinquant"))
from spinquant_inference import vortex_export_ops  # noqa: F401,E402
from vortex_llama3.run_backend_validation import _runtime_tensor, hybrid_metrics, LOCAL_THRESHOLDS


class PrefixAttention(torch.nn.Module):
    def __init__(self, capacity, head_dim=128):
        super().__init__()
        self.capacity, self.head_dim = capacity, head_dim

    def forward(self, query, key, key_scale, key_zero, value, value_scale, value_zero,
                positions, length):
        shape = [1, 1, 1, self.capacity, self.head_dim]
        scores = torch.ops.vortex.mm_w4a16(query, key, key_scale, key_zero, shape,
            self.head_dim, 4, 4, "signed_asymmetric_int4", True)
        masked, probability = torch.ops.vortex.causal_softmax(scores, positions, length, self.head_dim)
        context = torch.ops.vortex.mm_w4a16(probability, value, value_scale, value_zero, shape,
            self.head_dim, 4, 4, "signed_asymmetric_int4", False)
        return probability, context


def make_inputs(capacity, queries, length, *, poison=False):
    if not 1 <= length <= capacity:
        raise ValueError("valid length must be in [1, capacity]")
    rng = np.random.default_rng(20261007)
    query = torch.from_numpy(rng.normal(0, .25, (1, 1, 2, queries, 128)).astype("float16"))
    operands = []
    for _ in range(2):
        payload = rng.integers(0, 256, (1, 1, 1, capacity, 64), dtype="uint8")
        scale = rng.uniform(.02, .06, (1, 1, 1, capacity, 1)).astype("float16")
        zero = rng.integers(-2, 3, scale.shape, dtype="int16")
        scale[..., length:, :] = np.nan if poison else 0
        if poison:
            scale[..., length + 1::2, :] = np.inf
        operands.extend(torch.from_numpy(x) for x in (payload, scale, zero))
    positions = torch.tensor([list(range(max(0, length - queries), length))], dtype=torch.int64)
    if positions.shape[1] < queries:
        positions = torch.full((1, queries), length - 1, dtype=torch.int64)
    return (query, *operands, positions, torch.tensor(length, dtype=torch.int64))


def build_attention(capacity, queries, target, dynamic):
    model = PrefixAttention(capacity)
    sample = make_inputs(capacity, queries, 1)
    exported = torch.export.export(model, sample, strict=True)
    mod = from_exported_program(exported, run_ep_decomposition=False, unwrap_unit_return_tuple=True)
    exe = relax.build(mod, target, relax_pipeline=get_default_pipeline(
        target, layout_policy="fused", dynamic_kv_length=dynamic))
    return model, mod, exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--lengths", default="1,15,16,17,127,128,129,255,256,257,288,320,511,512")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = load_vortex_accelerator_profile(args.manifest).target
    alignment = int(target.attrs["vortex_mxu_col"])
    compiled = {}
    for dynamic in (False, True):
        print("BUILD", dynamic, flush=True)
        model, mod, exe = build_attention(args.capacity, args.queries, target, dynamic)
        label = "dynamic" if dynamic else "static"
        (args.output_dir / f"{label}.input.py").write_text(mod.script())
        # Preserve the generated device source as evidence of runtime extents.
        def sources(module):
            if module.kind == "vortex":
                return [module.inspect_source()]
            return [text for child in module.imports for text in sources(child)]
        try:
            (args.output_dir / f"{label}.device.cc").write_text("\n".join(sources(exe.mod)))
        except AttributeError:
            pass
        exe.export_library(str(args.output_dir / f"{label}.so"))
        compiled[dynamic] = exe
    if args.compile_only:
        return
    device = tvm.vortex(0)
    vms = {key: relax.VirtualMachine(exe, device=device, memory_cfg="naive") for key, exe in compiled.items()}
    results = []
    for length in map(int, args.lengths.split(",")):
        clean = make_inputs(args.capacity, args.queries, length)
        expected = [x.numpy() for x in model(*clean)]
        outputs = {}
        timings = {}
        for dynamic in (False, True):
            inputs = make_inputs(args.capacity, args.queries, length, poison=dynamic)
            tensors = [_runtime_tensor(x.numpy(), device) for x in inputs]
            start = time.perf_counter()
            outputs[dynamic] = [x.numpy() for x in vms[dynamic]["main"](*tensors)]
            timings[str(dynamic)] = time.perf_counter() - start
        record = {"length": length, "target": (length + alignment - 1) // alignment * alignment, "seconds": timings,
                  "metrics": [], "baseline_delta": []}
        for i, name in enumerate(("probability", "context")):
            record["metrics"].append(hybrid_metrics(outputs[True][i], expected[i], LOCAL_THRESHOLDS, name=name, enforce=False))
            record["baseline_delta"].append(float(np.max(np.abs(outputs[True][i].astype("float32") - outputs[False][i]))))
        record["baseline_metrics"] = [hybrid_metrics(actual, reference, LOCAL_THRESHOLDS,
            name="static", enforce=False) for actual, reference in zip(outputs[False], expected)]
        np.savez(args.output_dir / f"length_{length}.npz",
            dynamic_probability=outputs[True][0], dynamic_context=outputs[True][1],
            static_probability=outputs[False][0], static_context=outputs[False][1],
            reference_probability=expected[0], reference_context=expected[1])
        assert np.all(outputs[True][0][..., length:] == 0)
        record["bitwise_equal_to_static"] = all(np.array_equal(a, b) for a, b in zip(outputs[True], outputs[False]))
        results.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("RESULT", length, record, flush=True)
        assert record["bitwise_equal_to_static"], "dynamic prefix changed a live result"
    if not all(all(m["pass"] for m in row["metrics"]) for row in results):
        raise AssertionError("CPU reference limits exceeded; see saved static/dynamic diagnostics")


if __name__ == "__main__":
    main()
