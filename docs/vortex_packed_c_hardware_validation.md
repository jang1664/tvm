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


# New C4 packed-output FPGA validation

Date: 2026-10-07. Result: PASS.

Alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_fix_pad`. Image: `xrt_hw_u55c_c1_f100_fpint_L2cache_ab79baf965`. Source config: `configs/improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4.sh`. TH16/MXU16, L2 enabled.

All runs used Slurm job 5640 and the same U55C board, BDF `0000:2a:00.1`. The job completed and released the allocation.

## Results

- Standalone `build/ci/run_black.sh hw --fpga-bin ... --app fpint_gemm_ffn_hw`: M=1,4,256 with K=N=256, QBLK=32, QDIR=0, WTRANS=0; all PASS.
- TVM unfused (`--layout-policy alone`): 10 cases PASS.
- TVM fused: four GEMM chain cases PASS; intermediate C-to-A reuse recorded in the compiler inventory, and each entire output array is bitwise identical to its unfused baseline.
- No tolerance was changed. Numerical checks use the existing envelope `abs(error) <= 0.003 + 0.003 * abs(reference)` and reject nonfinite output. All cases had zero out-of-envelope elements and zero nonfinite values. This does not mean every output exactly equals the floating-point reference.

| TVM case | M, N, K | QBLK / QDIR / WTRANS | Unfused max absolute error | Fused bitwise match |
|---|---|---|---:|---|
| linear_0 | 1, 256, 256 | 32 / 0 / 0 | 0.001953125 | Not run |
| linear_1 | 4, 256, 256 | 32 / 0 / 0 | 0.001953125 | Not run |
| linear_2 | 256, 256, 256 | 32 / 0 / 0 | 0.00390625 | Not run |
| q16_d0_t0 | 4, 256, 256 | 16 / 0 / 0 | 0.00390625 | Not run |
| q16_d1_t1 | 4, 256, 256 | 16 / 1 / 1 | 0.001953125 | Not run |
| tail | 3, 129, 127 | 32 / 0 / 0 | 0.001953125 | Not run |
| descriptor_chain | 8, 256, 256 | 32 / 0 / 0 | 0.001953125 | Yes |
| descriptor_chain_tail | 4, 256, 256 | 32 / 0 / 0 | 0.001953125 | Yes |
| descriptor_chain_m1 | 1, 256, 256 | 32 / 0 / 0 | 0.00390625 | Yes |
| descriptor_chain_m256 | 256, 256, 256 | 32 / 0 / 0 | 0.0078125 | Yes |

The descriptor-chain graph contains a first GEMM, a sibling GEMM sharing A, and a second GEMM consuming the first result. It returns all three results, so bitwise equality checks the intermediate output as well as the final chained output. For the chain, the unchanged reference validates each operation against FP32 dequantized weights and feeds the actual rounded first output to the second reference operation; independent end-to-end Torch-reference differences are also retained in `results.json`.

## Image metadata

The new manifest initially lacked a layout version. After confirming the image origin identifies the new `v4_fix_pad` build, its original manifest was backed up to `build/hw_c4_fix_pad_20261007/manifest.before.json`, and top-level `vortex_layout_abi_version: 3` was added to the installed manifest. The xclbin and CONFIGS were not changed. The TVM loader and pre-launch runtime validation both consumed this versioned manifest.

Xclbin SHA256: `2e80ec9d2810473fe4f222192615e299959e4c995ff3bcd853d21741f33f4f03`.
Versioned manifest SHA256: `ed1ce9a5b2be95af3327d120e95338e7f715122a12305b5616c5d2ddf2921f4f`.
TVM profile fingerprint: `fd820f259a94c664923c0bfcdfac61407f73a2b81221660e1c6a43a290124e47`.

## Reproduction and artifacts

`build/hw_c4_fix_pad_20261007/run.sh` records the exact environment, config, board detection, commands, and timeouts. The allocation command was:

```bash
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=16G --time=00:35:00 \
  --job-name=tvm-c4-fix-pad \
  bash /home/jaeyongjang/project.local/tvm/build/hw_c4_fix_pad_20261007/run.sh
```

- `build/hw_c4_fix_pad_20261007/candidates.json`: dedicated C4 alias map; historical candidate maps were retained.
- `blackbox_m{1,4,256}.log`: standalone logs.
- `alone/results.json`, `fused/results.json`: compiler inventory, exact image identity, numerical results.
- `alone/C4/*.npz`, `fused/C4/*.npz`: actual/reference arrays; corresponding `.so` files contain the tested TVM programs.
- `session.log`, `device.txt`, `manifest.before.json`, `manifest.json`: session and metadata evidence.

The reusable TVM probe now includes `descriptor_chain_m1` and `descriptor_chain_m256` in addition to the existing M8 and M4 cases. No RTL or numerical tolerance changes were made during this validation. This is a functionality/layout-reuse result, not an end-to-end Llama latency or fused-overhead benchmark.

## Mini-Llama validation on the current C4 image

Completed 2026-10-07, 14:02 KST. **All 36 boundary checks PASS.** Only the
`v4_fix_pad` image above was used, with freshly compiled layout ABI 3 packages.
Slurm job 5657 pinned all six runs to U55C `0000:2a:00.1` and released the
allocation on completion. No compiler, RTL, kernel, or numerical-threshold changes
were needed to obtain these passes.

| Layout | Shape | VM mode | Boundary result | Max small-value absolute error | Max relative L2 |
|---|---|---|---|---:|---:|
| alone | S1 | bytecode | 6/6 PASS | 0.00000190735 | 0.00000239130 |
| alone | S2 | bytecode | 6/6 PASS | 0.000366211 | 0.00434291 |
| alone | S1 | compiled | 6/6 PASS | 0.00000190735 | 0.00000239130 |
| fused | S1 | bytecode | 6/6 PASS | 0.00000190735 | 0.00000239130 |
| fused | S2 | bytecode | 6/6 PASS | 0.000366211 | 0.00434291 |
| fused | S1 | compiled | 6/6 PASS | 0.00000190735 | 0.00000239130 |

- Model: hidden/intermediate/vocabulary=256, query heads=2, KV heads=1,
  head dimension=128. S1 uses batch=1, prompt=1, cache capacity=8; S2 uses
  batch=1, prompt=7, cache capacity=16.
- Packages retain the 32-layer archive metadata, but these probes execute one
  decoder layer. They check six boundaries: prefill/decode embedding,
  prefill/decode decoder layer, and prefill/decode final head. Each boundary
  receives its own deterministic input; this is not a continuous
  embedding-to-logits graph. The decode layer does consume the actual prefill
  cache, while its reference consumes the independently calculated reference cache.
- Parameters use the existing deterministic `constant_hash_compile_fixture_v1`
  synthetic archive (seed 20260902). Probe input seed is 20261006. These results
  establish correctness for these fixtures, not trained-model accuracy.
- Unchanged `LOCAL_THRESHOLDS`: split=0.25, absolute/relative thresholds=0.002,
  violation fraction <=0.02, relative L2 <=0.01, cosine >=0.999.
  Observed violation fraction and nonfinite counts were **zero** for every
  floating output. The minimum observed cosine was 0.999990574. Integer
  payload/zero/valid-length outputs had zero mismatches.
- Entire saved output arrays were **bitwise identical** between alone and fused
  for S1 bytecode, S2 bytecode, and S1 compiled. S1 bytecode/compiled outputs were
  also bitwise identical for both layouts.
- Every result's package SHA and image/profile identity were checked against its
  compiled package. All four packages resolve the same current image SHA above
  and layout ABI 3. No previous-C4 result is counted toward these 36 checks.

Artifacts: `build/mini_c4_fix_pad_20261007/`.

- `environment.sh`, `compile.sh`, `run_hw.sh`, `candidates.json`: exact configuration
  and reproducible commands; `compile.sh alone` and `compile.sh fused` each produce
  S1 bytecode+compiled and S2 bytecode modules before allocating hardware.
- `compile_{alone,fused}/packages/C4/{S1,S2}/package.json`: package identities,
  code-generation inventories, geometry, and module hashes.
- `hw_{alone,fused}_{S1_bytecode,S2_bytecode,S1_compiled}/results.json` and NPZs:
  per-output metrics and complete actual/reference arrays.
- `hardware.session.log`, `device.txt`, `hw_*.log`, `hw_*.exit`: execution evidence.
- `verify.py`, `completion.json`: complete result validation and bitwise comparisons.

Hardware command after compilation:

```bash
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=16G --time=00:45:00 \
  --job-name=tvm-mini-c4-fix-pad \
  bash /home/jaeyongjang/project.local/tvm/build/mini_c4_fix_pad_20261007/run_hw.sh
```

Full-size 32-layer inference, long-context coverage, and TVM end-to-end
latency/power measurements remain outside this mini validation.


## Full-size Llama3-8B: one decoder layer on C4

Completed 2026-10-07, 14:16 KST. **12/12 boundary checks PASS** on the current
`v4_fix_pad` image, Slurm job 5658, BDF `0000:2a:00.1`. The allocation has ended.
Only C4 was executed. Fresh packages use layout ABI 3 and the image SHA above.

The geometry is the actual Llama3-8B geometry: hidden size 4096, intermediate
size 14336, 32 query heads, 8 KV heads, head dimension 128. Exactly **one** decoder
layer executes, although model metadata describes the 32-layer architecture.
Weight/KV quantization groups are 32/128. This uses deterministic synthetic
parameters (`constant_hash_compile_fixture_v1`) and random FP16 hidden inputs
(seed 20261006), rather than pretrained Llama weights.

| Layout | Case | Batch / prefill tokens / cache capacity | VM | Prefill layer | Decode layer |
|---|---|---|---|---|---|
| alone | S1 | 1 / 1 / 8 | bytecode | PASS | PASS |
| alone | S2 | 1 / 7 / 16 | bytecode | PASS | PASS |
| alone | S1 | 1 / 1 / 8 | compiled | PASS | PASS |
| fused | S1 | 1 / 1 / 8 | bytecode | PASS | PASS |
| fused | S2 | 1 / 7 / 16 | bytecode | PASS | PASS |
| fused | S1 | 1 / 1 / 8 | compiled | PASS | PASS |

Each decode processes one token at the next position and consumes the actual
prefill KV cache. The CPU reference independently generates its own prefill
cache. Hidden inputs are supplied directly at each layer boundary; embedding
and final head do not execute in this test. CPU checks use the logical W4
weights, while C4 consumes the corresponding packed materialization.

- Existing `LOCAL_THRESHOLDS` were unchanged. All outputs have zero nonfinite
  values and zero elements outside the absolute/relative error envelope.
  Integer cache payloads, zero points, and valid lengths match exactly.
- S1 outputs match the CPU reference numerically. For S2, maximum small-value
  absolute error is 0.000244140625; maximum large-value relative error is
  0.001949317753314972; maximum relative L2 error is 0.0005771174016372198;
  minimum cosine similarity is 0.9999999328648872.
- Alone/fused outputs are bitwise identical for all three corresponding runs;
  S1 bytecode/compiled outputs are also bitwise identical for both layouts.
- Compiler inventories confirm C4 improve kernels, no naive/TCU helpers,
  no unresolved W4/FP16 calls, and three reused input layouts in fused prefill.
- Existing focused parser/metric checks: 2 passed. `git diff --check` passed.

Artifacts: `build/full_layer_c4_fix_pad_20261007/`. `completion.json` records
verified package hashes, image identity, metrics, and bitwise comparisons;
`hw_*/results.json` and NPZ files preserve every actual/reference output.
`environment.sh`, `compile.sh`, `run_hw.sh`, `verify.py`, logs, and exit files
record reproduction details. Compile both layouts before allocating hardware:

```bash
bash build/full_layer_c4_fix_pad_20261007/compile.sh alone
bash build/full_layer_c4_fix_pad_20261007/compile.sh fused
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=32G --time=00:45:00 \
  --job-name=tvm-layer-c4-fix-pad \
  bash /home/jaeyongjang/project.local/tvm/build/full_layer_c4_fix_pad_20261007/run_hw.sh
```

`run_backend_probe.py --layer-regression` reuses the checked package loader and
CPU comparison path, uploads only layer parameters, and executes only
`prefill_layer` and `decode_layer`. The existing `--mini-regression` still checks
all six mini boundaries. Compilation retains the complete six-boundary package
format, but embedding/head modules are not executed by the layer-only runner.

This validates full hidden/FFN/head dimensions with short sequences and one
layer. It does not establish pretrained-model accuracy, long-context coverage,
32-layer inference, or performance/energy results.


## Full 32-layer Llama3-8B chain with random weights

Completed 2026-10-07. **PASS: 128/128 decoder layer invocations**, comprising
one prefill and three consecutive decode phases. Only current C4 `v4_fix_pad`
was used (Slurm job 5659, BDF `0000:2a:00.1`); the allocation has ended.
Image SHA256 matches the current image documented above, and archive layout
ABI is 3 with MXU 16x16.

This is the full architecture: 32 decoder layers, hidden 4096, intermediate
14336, Q/KV heads 32/8, head dimension 128, vocabulary 128256. Batch is 1,
prompt token IDs are `[1]`, cache capacity is 8. The fused bytecode VM executes
embedding, all 32 layers, and final normalization/head on the FPGA. The compiler
reuses a single-layer executable with 32 distinct parameter slices; it does not
reuse one layer's weights. Hashes confirm all 32 Q-projection weight tensors
are distinct.

Per user instruction, weights are random rather than pretrained. The existing
`depth_scaled_residual_v1` initializer uses seed 20260831, random packed INT4
codes/zero points and random embeddings, unit norm weights, and fixed projection
scales; O/down residual scales are divided by sqrt(2*32). This replaces the
constant-weight fixture used by the preceding one-layer check. The archive
contains 741 tensors, totaling 5,741,617,152 data bytes.

| Phase | Layers checked | Logits relative-L2 error | Top-1 agreement | KV length |
|---|---:|---:|---:|---:|
| Prefill | 32 | 0.02609% | 100% | 1 |
| Decode 1 | 32 | 0.02105% | 100% | 2 |
| Decode 2 | 32 | 0.02552% | 100% | 3 |
| Decode 3 | 32 | 0.02714% | 100% | 4 |

All four phases enforce independent eager CPU comparisons, including stored KV
state and final logits. The FPGA's selected token is fed to its next embedding,
and each layer's FPGA KV state persists by device-to-device copies. Generated
token IDs are `[89754, 29229, 89754]`, matching CPU; the final phase's selected
next token also matches. Maximum per-layer hidden relative-L2 is 0.473526%.
There are no nonfinite hidden outputs, no retries, no reference substitution,
and no phase-limit exemption. Final logits and normalized outputs have zero
elements outside the existing full-stack error envelopes. KV checks use the
existing semantic/dequantized INT4 criteria, rather than demanding bitwise cache
identity. Full-stack thresholds were not changed; they differ from the stricter
local-boundary thresholds used by the earlier mini/single-layer runner.

Artifacts: `build/llama3_32layer_c4_fix_pad_random_20261007/`.

- `environment.sh`, `package.sh`, `reference.py`, `run_hw.sh`: exact setup and commands.
- `fused/package.json`, `fused/parameters/manifest.json`: model, packing and module identity.
- `reference.npz`: all CPU states/logits, generated before FPGA allocation.
- `hardware.log`: 128 per-layer diagnostics and successful process completion.
- `trace.json`: complete inference trace, logits hashes, tokens, state checks and metrics.
- `verify.py`, `completion.json`, `verification.log`: independent completeness/identity checks.

Reproduction from TVM root:

```bash
bash build/llama3_32layer_c4_fix_pad_random_20261007/package.sh
source build/llama3_32layer_c4_fix_pad_random_20261007/environment.sh
"$py" -u "$out/reference.py"
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=32G --time=00:35:00 \
  --job-name=tvm-llama32-c4-random bash "$out/run_hw.sh"
```

The hardware run uses `--reference --diagnostic-layer-checks` and the separately
prepared reference artifact, without retries or diagnostic substitutions. The
reference/package geometry and seed were checked before execution. This run
validates the complete random-weight chain for this short sequence; it does not
measure pretrained-model language quality, longer contexts, or unbiased latency
(the run includes layer diagnostics and host comparisons). No production source
or tolerance changes were needed for this full-model test.
