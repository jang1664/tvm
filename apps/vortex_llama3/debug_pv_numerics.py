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
"""Isolate PV arithmetic with uniform, one-hot and replayed attention inputs.

Run inside a hardware Slurm allocation. Produces diagnostics, not a permissive
functionality pass: IEEE and input-scaler FTZ predictions are recorded separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.backend.vortex import get_default_pipeline
from tvm.support.vortex import load_vortex_accelerator_profile

sys.path.insert(0, str(Path(os.environ["TVM_VORTEX_HOME"]) / "pytorch/spinquant"))
from spinquant_inference import vortex_export_ops  # noqa: E402,F401
from vortex_llama3.run_dynamic_kv_probe import make_inputs
from vortex_llama3.run_backend_validation import _runtime_tensor


class PV(torch.nn.Module):
    def __init__(self, k, n=128):
        super().__init__()
        self.k, self.n = k, n

    def forward(self, p, value, scale, zero):
        return torch.ops.vortex.mm_w4a16(p, value, scale, zero, [self.k, self.n],
            128, 1, 1, "signed_asymmetric_int4", False),


def pack(code):
    bits = code.astype("int16") & 15
    return (bits[:, 0::2] | (bits[:, 1::2] << 4)).astype("uint8")


def metrics(actual, expected):
    a, b = actual.astype("float64"), expected.astype("float64")
    return {"relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.max(np.abs(a - b))),
            "different_elements": int(np.count_nonzero(actual != expected))}


def pattern_cases(k):
    for exp in (-12, -13, -14, -15, -16, -20):
        # A is normal, scale is normal, and the desired intermediate is 2**exp.
        p = np.full((1, k), 2.**-9, dtype="float16")
        scale = np.full((k, 1), 2.**(exp + 9), dtype="float16")
        for zero in (0, 1):
            yield f"uniform_exp{exp}_z{zero}", p, np.full((k,128), 2, "int16"), scale, np.full((k,1),zero,"int16")
    if k == 512:
        for length in (128, 256, 288, 304, 320, 336, 384, 511, 512):
            p=np.zeros((1,k),"float16");p[:,:length]=1./length
            scale=np.full((k,1),.02,"float16")
            yield f"length_{length}",p,np.ones((k,128),"int16"),scale,np.zeros((k,1),"int16")
        for index in (0,15,16,127,128,255,256,511):
            for exp in (-14,-15):
                p=np.zeros((1,k),"float16");p[0,index]=2.**-9
                yield f"onehot_{index}_exp{exp}",p,np.ones((k,128),"int16"),np.full((k,1),2.**(exp+9),"float16"),np.zeros((k,1),"int16")
        # Change scale across K tiles while keeping every intermediate normal.
        p=np.full((1,k),2.**-5,"float16")
        scale=np.repeat(np.array([.125,.25,.5,1],"float16"),128).reshape(k,1)
        code=((np.arange(k)[:,None]+np.arange(128)[None,:])%16-8).astype("int16")
        yield "tagged_tiles",p,code,scale,np.zeros((k,1),"int16")


def qrow_scaler_model(p, scale):
    """Measured Xilinx half multiplier: normal-precision RNE followed by FTZ.

    At the smallest-normal boundary, 11-bit significand rounding has a
    half-ULP of 2**-26 before exponent underflow is flushed. This is distinct
    from IEEE gradual-underflow rounding (half-ULP 2**-25 there).
    """
    a, b = p.astype("float64").copy(), scale[:, 0].astype("float64").copy()
    a[np.abs(a) < 2**-14] = 0
    b[np.abs(b) < 2**-14] = 0
    raw = a * b
    result = raw.astype("float16")
    result[np.abs(raw) < 2**-14 - 2**-26] = 0
    return result


def output_converter_model(value):
    """Current VX_f32_to_f16 drops the hidden bit in its subnormal branch."""
    value = value.astype("float32")
    result = value.astype("float16")
    small = (np.abs(value) < 2**-14) & (value != 0)
    fraction, exponent = np.frexp(np.abs(value[small]))
    missing_hidden = np.ldexp(np.ones_like(fraction), exponent - 1)
    result[small] = ((np.abs(value[small]) - missing_hidden) * np.sign(value[small])).astype("float16")
    return result


def boundary_cases(k):
    # Sweep the exact scaler boundary pairs observed in the saved K=511 replay.
    for a_bits, s_bits in ((0x17bc, 0x2823), (0x1815, 0x27d6)):
        for delta in range(-4, 5):
            a = np.array(a_bits, dtype="uint16").view("float16").item()
            b = np.array(s_bits + delta, dtype="uint16").view("float16").item()
            yield f"boundary_a{a_bits:04x}_s{s_bits+delta:04x}",np.full((1,k),a,"float16"),np.full((k,128),2,"int16"),np.full((k,1),b,"float16"),np.zeros((k,1),"int16")
    # Keep all nonzero P*scale intermediates normal; cancel to a subnormal sum.
    for sign in (1,-1):
        for exponent in (-14,-15,-16,-17):
            p=np.zeros((1,k),"float16")
            p[0,0]=sign*2**-8
            p[0,1]=-sign*(2**-8-16*(2.**exponent+2**-22))
            yield f"output_subnormal_sign{sign}_exp{exponent}",p,np.ones((k,128),"int16"),np.full((k,1),2**-4,"float16"),np.zeros((k,1),"int16")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--replay-dir",type=Path)
    ap.add_argument("--phase",choices=("patterns","boundary"),default="patterns")
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    target=load_vortex_accelerator_profile(args.manifest).target
    device=tvm.vortex(0);rows=[]
    for k in ((16,) if args.phase=="boundary" else (16,128,512)):
        model=PV(k)
        sample=(torch.ones((1,k),dtype=torch.float16),torch.zeros((k,64),dtype=torch.uint8),
                torch.ones((k,1),dtype=torch.float16),torch.zeros((k,1),dtype=torch.int16))
        mod=from_exported_program(torch.export.export(model,sample,strict=True),run_ep_decomposition=False,unwrap_unit_return_tuple=True)
        exe=relax.build(mod,target,relax_pipeline=get_default_pipeline(target,layout_policy="alone"))
        exe.export_library(str(args.output_dir/f"pv_k{k}.so"))
        vm=relax.VirtualMachine(exe,device=device,memory_cfg="naive")
        cases=list(boundary_cases(k) if args.phase=="boundary" else pattern_cases(k))
        if k==512 and args.replay_dir:
            for length in (288,320,511,512):
                data=np.load(args.replay_dir/f"length_{length}.npz")
                inputs=make_inputs(512,1,length)
                packed,scale,zero=[x.numpy().reshape(512,-1) for x in inputs[4:7]]
                code=np.stack([packed&15,packed>>4],axis=-1).reshape(512,128).astype("int16")
                code=np.where(code>=8,code-16,code).astype("int16")
                # Separate groups to reuse the same M=1 binary.
                for group,p in enumerate(data["dynamic_probability"].reshape(2,512)):
                    for factor in (1,2,4,16):
                        cases.append((f"replay_{length}_g{group}_factor{factor}",
                            (p[None,:]*factor).astype("float16"),code,scale,zero))
        for name,p,code,scale,zero in cases:
            arrays=[p,pack(code),scale,zero]
            output=vm["main"](*[_runtime_tensor(x,device) for x in arrays]).numpy()
            expected=model(*[torch.from_numpy(x) for x in arrays])[0].numpy()
            raw=p.astype("float64")*scale[:,0].astype("float64")
            rounded=raw.astype("float16")
            ftz=rounded.copy();ftz[np.abs(ftz)<2**-14]=0
            before=rounded.copy();before[np.abs(raw)<2**-14]=0
            weight=(code-zero).astype("float64")
            round_output=(rounded.astype("float64")@weight).astype("float16")
            ftz_output=(ftz.astype("float64")@weight).astype("float16")
            before_output=(before.astype("float64")@weight).astype("float16")
            rtl_prediction=output_converter_model(qrow_scaler_model(p,scale).astype("float64")@weight)
            record={"rtl_model":metrics(output,rtl_prediction),"case":name,"k":k,"actual_0":float(output.flat[0]),"expected_0":float(expected.flat[0]),
                "ieee":metrics(output,expected),"fp16_intermediate":metrics(output,round_output),
                "ftz_after_round":metrics(output,ftz_output),"ftz_before_round":metrics(output,before_output),
                "subnormal_products":int(np.count_nonzero((np.abs(raw)>0)&(np.abs(raw)<2**-14)))}
            rows.append(record)
            np.savez(args.output_dir/f"k{k}_{name}.npz",p=p,code=code,scale=scale,zero=zero,output=output,expected=expected,
                ftz_prediction=ftz_output,raw_ftz_prediction=before_output)
            (args.output_dir/"results.json").write_text(json.dumps(rows,indent=2)+"\n")
            print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
