<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Packed GEMM-C layout support in TVM

Implemented on 2026-10-07 for the Improve RTL change that packs real M rows inside each DMA tile and reserves padding only at the DMA tile end.

## Selecting the new layout

The existing `vortex_layout_abi_version` target attribute now supports two contracts:

| Version | C layout | M-tail GEMM-C to GEMM-A reuse |
|---|---|---|
| 2 (default) | Row padding after every microtile | Only when physical strides already agree |
| 3 (explicit) | Real rows packed within each DMA tile; padding at the tile end | Enabled when execution extents and tile geometry match |

Keep existing FPGA manifests unchanged. Once the new PnR image has completed and its RTL provenance is confirmed, add this top-level field to **that image's** `manifest.json`:

```json
"vortex_layout_abi_version": 3
```

This is image metadata, not an RTL CONFIGS define, and does not require another PnR. Do not add it to an old image. The default for a manifest without this field remains version 2. Use the exact manifest next to the packaged image, not a copy in an unrelated directory; the XRT runtime reads `<image-package>/manifest.json` for `<image-package>/bin/vortex_afu.xclbin`.

```python
from tvm.support.vortex import load_vortex_accelerator_profile
from tvm.target import Target

profile = load_vortex_accelerator_profile("/path/to/new-image/manifest.json")
target = Target(profile.target, host="llvm")
assert int(target.attrs["vortex_layout_abi_version"]) == 3
```

For compilation before an image exists, an explicit target dictionary may set `vortex_layout_abi_version=3` with the correct TH16/MXU16 and other geometry fields. A synthetic target does not authorize device execution: accelerated launches still require an exact manifest profile. Recompile execution packages after selecting the new image; existing packages retain their old layout contract.

## Implementation

- `layout.py`: versioned C descriptor and both DMA tile dimensions in the compatibility check. An execution-padding mismatch still rejects reuse (for example producer N=33 -> 48 versus consumer K=33 -> 64 at MXU16/QBLK32).
- `pipeline.py`: versioned C detile and row-major/broadcast add indexing. Shared tiled add/ReLU continue to operate in physical layout. First-input packing and final-output detiling remain when the graph interface requires conventional tensors.
- `support/vortex.py`: read the manifest version and incorporate version 3 into profile identity, including when CONFIGS is unchanged.
- `vortex_module.cc`: validate module versus authoritative manifest layout version before upload/launch, in addition to exact CONFIGS matching. A v3 module rejects a legacy image, and a v2 module rejects a v3 image.
- `run_backend_gemm_probe.py`: accept direct C reuse for tail-M descriptor chains on v3 while retaining the legacy expectation on v2.

The job submission ABI stays v2: dimensions, pointers, quantization fields, and the GEMM job descriptor did not change. The existing `vx_tvm_gemm_w4a16_v2` device helper validates/submits those fields but never interprets C addresses. TVM therefore supplies its existing accepted value 2 to that helper while recording physical **layout ABI 3** in the module/target/manifest contract. No external Vortex kernel header or RTL change is required for this TVM migration. Unsupported layout versions are rejected by TVM.

## Verification before FPGA availability

All tests use the existing local TVM build. FPGA-dependent tests remain skipped.

- Host layout/planner tests: 106 PASS, including 22 tests executing actual layout TIR on LLVM with only thread-binding loops serialized. Independent physical stream oracles cover M=1,4,9,132,256; N=16,33,48,144,256; both layout versions; DMA boundary tails; NaN-poisoned unused padding; and vector, row, column, matrix, and tiled addition.
- Profile, target, and runtime tests: 100 PASS, 4 skipped. Checks include both directions of image-layout mismatch and legacy default behavior.
- Export/lowering tests: 35 PASS, 43 skipped. Total across the three final suites: 241 PASS, 47 skipped. New tests verify direct producer-buffer identity at M=1,4,256 with K=N=256, and FFN vector/branch lowering for M=1,4,8,9,132,256 under both ABIs.
- Cross-compilation: an M=1 TH16/MXU16 GEMM -> add -> ReLU -> GEMM graph with layout ABI 3 compiled and exported to `build/vortex_packed_c_validation/ffn_m1_layout3.so` using the existing Vortex device header. This synthetic-profile artifact is compile evidence only, not a launch package for an FPGA.
- The C++ compiler/runtime rebuild and `git diff --check` passed.

Artifacts and logs are in `build/vortex_packed_c_validation/`. No FPGA was programmed and no new device numerical execution was performed. Once PnR finishes, the remaining validation is a numerical fused-versus-unfused chain test on the new image, particularly M=1 and M=4.
