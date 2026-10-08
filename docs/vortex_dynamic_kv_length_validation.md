# Dynamic KV valid-length execution on updated C4

## Implementation

TVM now has opt-in `get_default_pipeline(target, dynamic_kv_length=True)` and
`run_synthetic_inference.py --mode package --dynamic-kv-length` support.
The decoder executable is reused across scalar valid lengths. Static tensor shapes,
capacity, DMA parent strides, and canonical cache representation are unchanged.
Query count, batch, head geometry, and cache capacity are still compile-time shapes.

The compiler recognizes W4 QK -> causal softmax -> W4 PV. Runtime length is
propagated through the existing rank-5/GQA expansion to rank-2 GEMM submissions.
The existing device submission helper now accepts independent target extents.
No RTL was changed in this task and no LLM-specific fields were added to RTL.

For TH16/MXU16 and a fixed capacity of 512, head dimension 128, M=1:

| Valid length | QK original M/N/K | QK target M/N/K | PV original M/N/K | PV target M/N/K |
|---:|---|---|---|---|
| 1 | 1/512/128 | 1/16/128 | 1/128/512 | 1/128/16 |
| 129 | 1/512/128 | 1/144/128 | 1/128/512 | 1/128/144 |
| 288 | 1/512/128 | 1/288/128 | 1/128/512 | 1/128/288 |
| 512 | 1/512/128 | 1/512/128 | 1/128/512 | 1/128/512 |

At length 288, GEMM executes 128 + 128 + 32 on the variable axis. Parent tile
addressing still uses the capacity-derived 128-wide third tile. With capacity 320,
that physical final tile is 64 wide; the same target extent still computes 32.

Unused source RHS token rows have their scales neutralized before packing, PV A
is zero outside the valid prefix, and inactive QK output is masked before use.
NaN/Inf in inactive storage cannot become a live operand. Softmax's reduction and
exponential loops stop at min(valid_length, query_position + 1); its output tail is
still explicitly zero. The valid-length contract is 1 <= length <= capacity.

## Hardware and compatibility

- Alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`.
- Image: `/opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c1_f100_fpint_L2cache_96c0f69b12/bin/vortex_afu.xclbin`.
- Image SHA256: `b9311154b3b56a5266e7e9d0f59f300c6f09c2e7ec6af9cdda01aa6b3770ac53`.
- Authoritative manifest: layout ABI 3, GEMM submission ABI 3.
- GEMM ABI 3 metadata was added after checking the image's updated-FSM provenance;
  the xclbin was not modified. Original manifest is saved in the output directory.
- Compiler rejects dynamic mode on older targets; runtime rejects a region module
  on a manifest without the new submission ABI. Static ABI 2 submissions remain supported.
- The new ABI participates in the target fingerprint, so existing TVM packages
  with the old fingerprint must be rebuilt against the annotated manifest.

## Focused FPGA results

One static and one dynamic executable were built per shape and reused for all lengths.
Random query and INT4 K/V data, asymmetric zero points, FP16 scales, head dimension
128, two query groups and one KV head were used. Dynamic inputs intentionally had
NaN/Inf in inactive K/V scale rows; static/reference inputs zeroed those rows.

| Capacity | Query rows | Valid lengths | Static/dynamic bitwise agreement | CPU numerical limits |
|---:|---:|---|---|---|
| 512 | 1 | 1,15,16,17,127,128,129,255,256,257,288,320,511,512 | 14/14 | 11/14 |
| 320 | 4 | 1,15,16,17,127,128,129,255,256,257,288,320 | 12/12 | 11/12 |

Thus all 26 dynamic-prefix cases match static hardware execution exactly, and 22/26
meet the existing CPU tolerance. The four CPU failures remain failures in the probe
(exit status 1); they have not been relabeled as full functionality passes.

### Separate long-PV numerical limitation

| Capacity / M | Valid length | Context relative L2 vs CPU, static **and** dynamic |
|---|---:|---:|
| 512 / 1 | 320 | 2.9602% |
| 512 / 1 | 511 | 23.7477% |
| 512 / 1 | 512 | 24.2069% |
| 320 / 4 | 320 | 1.4241% |

QK/softmax probabilities remain close to CPU (relative L2 below 0.01% in the
512-capacity cases). The large context difference is also present without runtime
prefix execution. This establishes that the new prefix path did not introduce it;
the root cause was subsequently identified as QROW input-scaler FTZ plus a GEMM
output-converter subnormal bug. See [PV diagnosis](vortex_pv_numerics_debug.md).
Do not treat long-context CPU
functionality as cleared by the bitwise baseline comparison.

## Performance scope

Update (2026-10-08): an opt-in persistent packed KV path now fuses the K/V layout
stores into quantization. See [implementation and validation](vortex_packed_kv_cache.md).
The measurements and limitations below describe the original dynamic-length path.

Capacity-sized K/V packing, matrix slicing, and temporary allocations are still
performed, and explicit prefix masking introduces extra vector launches. These
small probes were about 14–21 ms slower per graph invocation despite reduced GEMM
work; this is host-inclusive timing from single calls, not a GEMM-cycle benchmark.
The current implementation validates variable compute extents. Persistent packed
KV storage, incremental cache packing, and folding masks into pack/detile kernels
remain performance work. Default execution stays static unless explicitly enabled.

## Artifacts and reproduction

Output: `/home/jaeyongjang/project.local/tvm/build/c4_dynamic_kv_20261007/`.

- `probe512_m1/`, `probe320_m4/`: compiled modules, device source, per-length outputs,
  static/dynamic/CPU metrics. `summary.json` aggregates their results.
- `hardware.sh`, `hardware320.sh`: FPGA probe commands.
- `llama32/`: full Llama3-8B random-weight dynamic package, capacity 320.
- `llama_hardware.sh`: 32-layer prefill + decode hardware validation command.
- `pytest.log`, `compat_tests.log`, `final_host_tests.log`: host/compiler checks.

Host validation: 200 distinct tests passed, four hardware/tool-dependent tests
skipped across the scoped suites; the final dynamic/runtime subset passed 35 tests
with two skips after adding the negative runtime ABI check.

## Full Llama3-8B validation: PASS

Random deterministic weights, actual 32 layers, hidden 4096, FFN 14336, 32 Q heads,
8 KV heads, head dimension 128, vocabulary 128256. Batch 1, prefill 1 token,
three decode steps, fixed capacity 320. One compiled decoder layer executable is
reused with each layer's own parameters and KV state, as in the existing app.

- Slurm job 5670: COMPLETED, exit 0, BDF `0000:2a:00.1`; allocation released.
- All 128 layer invocations passed canonical CPU reference checks.
- Valid lengths across all 32 layer caches: 1 -> 2 -> 3 -> 4.
- No retries, no CPU embedding/head replacement, no reference-state injection.
- Maximum layer relative L2: 0.00473526 (0.474%).
- All four final logits agree with CPU top-1.
- Generated token IDs: `[89754, 29229, 89754]`.
- The four final logits SHA256 hashes match the earlier static capacity-8 run
  exactly. This comparison does not replace the larger-prefix focused tests.
- Details: `llama_trace.json`, `llama_completion.json`, `llama_hardware.log`.

This clears dynamic KV execution for the tested short full-model generation and
its equivalence to static hardware through the tested larger prefix boundaries.
The separate long-PV CPU discrepancies above are now diagnosed; RTL remediation
is pending. See [PV diagnosis](vortex_pv_numerics_debug.md).
