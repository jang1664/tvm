# Persistent packed KV validation — 2026-10-08

## Result

All nine stages were executed in order. The new packing path matches the existing
hardware path in the tested comparisons. Full random-weight Llama3-8B completed
prefill 32 + decode 32, and all 33 final-logit comparisons and generated token IDs
agree with the CPU reference within the existing application criteria.

**This is not an unconditional CPU-accuracy PASS:**14 of 1056 full-model layer
states exceed the independent CPU-chain gates. All 14 pass when their eager CPU
layer receives exactly the saved FPGA hidden input and previous FPGA KV state.
The separate random-hidden four-layer fixture also exhibits large accumulated CPU
drift. These failures remain recorded; thresholds were not loosened.

Actual checkpoint validation was excluded following the user's explicit choice.
The long-context stage validates real-size attention with KV lengths512/1024/2048;
it is not a full32-layer model prefill at those lengths.

## Configuration and provenance

- C4 alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`.
- TH16/MXU16; IMPROVE layout ABI3, GEMM submission ABI3.
- Image: `/opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c1_f100_fpint_L2cache_96c0f69b12/bin/vortex_afu.xclbin`.
- Image SHA256: `b9311154b3b56a5266e7e9d0f59f300c6f09c2e7ec6af9cdda01aa6b3770ac53`.
- Each baseline/fused comparison used the same allocated FPGA. The complete
  32-layer run stayed on BDF `0000:2a:00.1` (Slurm5693). The supplementary
  same-input replay used a separate allocation/BDF `0000:3d:00.1` (Slurm5694).
- Raw results, scripts, binaries, source hashes:
  `build/packed_kv_validation_20261008/`.
- CPU fixture parameters were independently repacked and checked against the
  FPGA archive: all 92 tensor hashes for the first4 layers match.
- Existing model parameter generation and RTL arithmetic were preserved. Probe
  support was extended for configurable query heads per KV head and baseline
  compilation. Twelve packed-KV CPU/compiler tests passed after that extension.

## Stages

| # | Test | Outcome |
|---|---|---|
|1| Quantization/packing oracles; D128/256, tile boundaries | Prior64-test suite PASS; relevant12 tests rerun PASS |
|2| Sequential attention, capacities320/512, B1/2, KV heads1/2 | Prior826 calls and26 checked boundaries PASS |
|3| Baseline vs fused, prefill 4 + decode 32, capacity 512 | Canonical KV bitwise equal; CPU probability/context PASS |
|4| Same VM reset/reuse; two interleaved sequences; three repeats | Outputs reproduce bitwise; no cross-sequence contamination observed |
|5| Q32/KV8/D128 attention, prefill 32 + decode 32 | Canonical KV bitwise equal; CPU probability/context PASS |
|6| One real-dimension decoder, prefill 32 + decode 32 |33 calls: hardware baseline/fused bitwise equal; strict CPU-chain failures recorded |
|7| Two, then four decoders |66 +132 calls: hardware baseline/fused bitwise equal; accumulated CPU drift recorded |
|8| Full random Llama3-8B,32 layers, prefill 32 + decode 32 | Execution and final logits/tokens PASS;14 intermediate CPU-chain gate failures,14/14 same-input rechecks PASS |
|9| Long-context attention, valid lengths512/1024/2048, capacity 2048 | CPU PASS, all outputs bitwise equal between baseline and fused |

Stages1–2 are backed by the prior [implementation report](vortex_packed_kv_cache.md)
and `build/packed_kv_20261008/` logs. They were not needlessly repeated on FPGA.
Stage 4 resets a sequence by allocating fresh zero-filled KV buffers on the same
compiled VMs; it is not an in-place buffer rewind test. Host peak RSS was stable
across its three repeats; this is not a direct FPGA allocator leak measurement.

## Attention graph time and device launches

Times below are individual diagnostic runs, including graph execution and
synchronization but excluding CPU reference calculations and output snapshots.
They are not statistically controlled kernel-cycle benchmarks. Device launch
counts exclude VM allocation/shape helpers and were verified against the XRT
completion logs (3198 total for stage 3;31538 for stage 5).

| Shape | Phase | Baseline launches | Fused launches |
|---|---|---:|---:|
|Q2/KV1/D128|Prefill 4|97|29|
|Q2/KV1/D128|Decode 1|67|29|
|Q32/KV8/D128|Prefill 32|903|299|
|Q32/KV8/D128|Decode 1|649|299|

| Sequence | Baseline graph time | Fused graph time | Reduction |
|---|---:|---:|---:|
|Q2/KV1, prefill 4 + decode 32|7.784s|2.531s|67.5%|
|Q32/KV8, prefill 32 + decode 32|89.677s|27.243s|69.6%|

## Decoder comparison and accumulated numerical differences

Hidden4096, FFN14336, Q32/KV8/D128, capacity 512. Each sequence uses32 prefill
tokens and32 decode appends. Distinct layer parameter slices are used with the
same compiled one-layer VM. Every one of the231 layer calls across1/2/4 layers
has bitwise-equal hidden output and canonical KV payload/scale/zero/length between
the two hardware paths.

| Layers | Baseline graph time | Fused graph time | Maximum hidden CPU-chain relative L2 |
|---|---:|---:|---:|
|1|131.761s|68.998s|0.2196%|
|2|263.245s|137.907s|4.2512%|
|4|531.822s|278.185s|70.6569%|

The large four-layer discrepancy was isolated by replaying all 120 calls through
valid length 60, giving each CPU layer the exact FPGA hidden input and previous
FPGA KV state. All 120 passed the existing full-stack hidden/KV criteria. Maximum
local hidden relative L2 was 0.1730%.

| Layer index at valid length 60 | Independent CPU chain | Same FPGA inputs |
|---|---:|---:|
|0|0.1326%|0.0429%|
|1|4.2512%|0.0604%|
|2|70.6569%|0.0473%|
|3|55.2035%|0.0253%|

This supports amplification of small arithmetic/quantization differences through
the stateful random-weight chain. It does not isolate one particular RTL rounding
or subnormal operation as the initial cause. Shared CPU-chain failures are not
relabeled as arithmetic passes merely because baseline and fused agree.

## Full random-weight Llama3-8B

All 32 distinct layer parameter slices, hidden4096, FFN14336, Q32/KV8/D128,
vocabulary128256. Prefill 32, decode 32, capacity 512. Thirty-two output tokens were
recorded; the application also evaluates the final decode phase's next logits.
All 1056 layer invocations completed, cache lengths advanced through 64, and no
NaN/Inf or hang occurred. Decode reused one compiled executable across all valid
lengths. Every layer state and phase logits are saved for offline comparison.

- Final-logit comparisons:33/33 PASS; all selected token IDs match CPU.
- Maximum final-logit relative L2: 0.1807%.
- Independent CPU-chain layer state gates:1042/1056 PASS;14 failures retained.
- Same-input rechecks of those14 failures:14/14 PASS; maximum hidden relative L2 0.1074%.
- Prefill time: 560.33s; decode mean: 52.97s; total: 2255.48s (37.59min). These include validation snapshots/readbacks.
- Host peak RSS: 17.90GiB; physical parameter archive: 5.35GiB. Host RSS is not FPGA SRAM usage.

| Independent-chain failures | Reason | Same-input outcome |
|---|---|---|
|Decode 7, layer 1|Hidden relative L2 5.0262% exceeds5%|PASS, relative L2 0.0535%|
|Decode 15, layer 3|8.3008% of elements exceed hybrid bounds; limit8%|PASS, relative L2 0.0630%|
|Decode 26, layers 1–12|Intermediate hidden drift; layer 1 reaches40.1098% relative L2|All 12 PASS; largest local relative L2 0.1074%|

Layer indices are zero-based. Full-stack gates are the existing application's
criteria, not the stricter LOCAL_THRESHOLDS used by isolated attention probes.
The independent CPU and FPGA token histories match throughout, so these are
same-token-history comparisons. Their hidden/KV numerical states still evolve
independently, which is what the supplementary same-input checks distinguish.

Authoritative files: `full32/trace.json`, `comparison.json`,
`comparison_summary.json`, `local_failures.json`, `local_failures_summary.json`.

## Long-context attention

Q32/KV8/D128, one query token, physical capacity 2048. The prior cache is seeded
using the independent CPU quantizer and layout packers, then a new token is
quantized/appended by each compiled graph. Nonzero data is intentionally retained
in future cache positions: masked probabilities must remain exactly zero beyond
the valid length. Baseline and fused probability, context, and canonical cache
outputs are bitwise equal in all three cases.

K data is drawn from[1,5), V from[1,17), with a fixed seed; this controlled fixture
avoids the known FP16 PV underflow regime. Measured subnormal P×scale product count
is zero. It validates long-prefix addressing and computation, not a full32-layer
512/1024/2048-token prefill or a long autoregressive generation trajectory.

| Valid KV length | Baseline | Fused | Probability CPU relative L2 | Context CPU relative L2 |
|---|---:|---:|---:|---:|
|512|6.123s|1.230s|0.0744%|0.0163%|
|1024|6.431s|1.550s|0.0742%|0.0136%|
|2048|7.071s|2.187s|0.0799%|0.0098%|

Raw data: `long_attention/results.json`; log: `hardware_long.log`.

## Reproduction and limits

The task directory contains `attention_validation.py` (stages3–5),
`layer_validation.py` (stages6–7), `capture_full.py` / `compare_full.py`
(stage 8), and `long_attention.py` (stage 9), plus their compile/run wrappers.
Hardware wrappers use the existing private real-XRT environment and select the
single board visible to their Slurm allocation. Run them from the TVM root with
one allocated U55C; CPU compilation stays outside the hardware allocation.

The opt-in packing implementation and state ownership contract are described in
[vortex_packed_kv_cache.md](vortex_packed_kv_cache.md). This test work added
`--query-heads-per-kv` to the focused probe. No actual checkpoint was used, and the
known FP16 arithmetic limitations have not been repaired by this validation work.
