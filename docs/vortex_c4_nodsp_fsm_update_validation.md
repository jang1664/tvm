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

# C4 nodsp FSM update: TVM hardware validation

Completed: 2026-10-07T22:18:06. **PASS**.

Alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`. Config: `configs/improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp.sh`. TH16/MXU16, DMA tiles128, layout ABI3, L2 enabled.
All successful hardware runs used Slurm job5666 and U55C `0000:2a:00.1`.

## Results

| Check | Result |
|---|---|
| TVM unfused GEMM / tail / QBLK16 / descriptor chains | 10/10 PASS |
| TVM fused descriptor chains, M=1/4/8/256 | 4/4 PASS; bitwise equal to unfused |
| Same 14 GEMM output arrays vs previous fix_pad image | 14/14 bitwise equal |
| Full random-weight Llama3-8B | 128/128 layer calls PASS |
| Prefill/decode logits top-1 vs CPU reference | 4/4 phases agree |

GEMM checks retained the existing `0.003 + 0.003 * abs(reference)` envelope. Maximum absolute error across tested GEMMs was0.0078125; out-of-envelope and nonfinite counts were zero. No tolerance changes or exception handling were introduced.

## Full model scope

32 distinct parameter sets; hidden4096, FFN14336, query/KV heads32/8, head dimension128, vocabulary128256. Batch1, prompt token `[1]`, prefill1 token, three consecutive one-token decode steps, cache capacity8. Fused layout and bytecode VM; embedding/head execute on FPGA. This uses random INT4 weights, not a pretrained checkpoint.
The one-layer compiled module runs all32 decoder layers sequentially, preserving each layer's live KV state. The 128 calls are32 layers times(prefill +3 decode steps).
All741 newly generated parameter payload hashes match the previous reference fixture. The CPU reference was reused without modification. Canonical reference checks applied to all phases; no retries, CPU embedding/head substitution, or reference state replacement was enabled.

| Phase | Cache valid length | Logits relative L2 | Top-1 agreement |
|---|---:|---:|---:|
| prefill | 1 | 0.000260891014 | 100% |
| decode_1 | 2 | 0.00021054392 | 100% |
| decode_2 | 3 | 0.000255196646 | 100% |
| decode_3 | 4 | 0.00027142113 | 100% |

Maximum layer relative L2: 0.00473526278. All layer nonfinite counts were zero.
Generated token IDs: `[89754, 29229, 89754]`, identical to the previous fix_pad run. Random-weight token IDs do not establish language quality.

## Environment corrections

- The new image manifest omitted `vortex_layout_abi_version`. Archived synthesized FSM source and PnR provenance confirmed packed-C/original-target addressing. Added ABI3 to the installed manifest; original and updated copies are saved as `manifest.before.json` and `manifest.json`. The xclbin was not modified.
- Initial Slurm job5665 failed before device launch because shared `build/runtime/libvortex-xrt.so` still linked to VCS. Built a dedicated hardware driver with `TARGET=hw DESTDIR=<artifact-root>/runtime`, linked to `libxrt_coreutil.so.2`, and prepended that directory to `LD_LIBRARY_PATH`. Shared simulation libraries were left unchanged. Initial logs are preserved under `preflight_*`.
- No TVM production source, RTL, numerical thresholds, or candidate defaults were changed for these passes.

## Identity and reproduction

Xclbin SHA256: `b9311154b3b56a5266e7e9d0f59f300c6f09c2e7ec6af9cdda01aa6b3770ac53`.
Manifest SHA256 after ABI annotation: `31ee8c2845fbd800086c32996848e956031d2bc56cd58975eeb591fd75a98c41`.
Artifact root: `/home/jaeyongjang/project.local/tvm/build/c4_nodsp_fsm_update_20261007_220215`.

```bash
bash /home/jaeyongjang/project.local/tvm/build/c4_nodsp_fsm_update_20261007_220215/package.sh
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=16G --time=00:45:00 \
  --job-name=tvm-c4-nodsp-fsm bash /home/jaeyongjang/project.local/tvm/build/c4_nodsp_fsm_update_20261007_220215/hardware_session.sh
```

`environment.sh`, `hardware_session.sh`, `verify_gemm.py`, and `verify.py` record the exact setup and verification. `alone/results.json`, `gemm_fused/results.json`, NPZ outputs, `completion.json`, `trace.json`, `previous_image_comparison.json`, and logs preserve evidence.

This validates existing TVM functionality on the new image. TVM still uses its existing capacity-based attention shapes; dynamically shortening GEMM target extents without repacking KV is not implemented or validated by this run. Long-context/large-prefill coverage and inference performance/power benchmarking are outside this result.
