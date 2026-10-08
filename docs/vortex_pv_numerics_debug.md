# PV numerical error diagnosis (2026-10-08)

## Conclusion

The four long-PV CPU discrepancies are fully explained by two arithmetic effects:

1. The QROW input scaler multiplies P by the per-token scale in FP16. The loaded
   Xilinx half multiplier flushes subnormal products to zero. This causes the large
   1.4–24.2% relative-L2 errors as probabilities shrink with sequence length.
2. The GEMM output FP32-to-FP16 converter omits the implicit leading 1 when shifting
   into an FP16 subnormal. This accounts for the small remaining output differences
   (one output at M=1/L=511, three at M=4/L=320).

Using the observed FPGA probabilities as PV input, the combined arithmetic model
reproduces **every output bit in all four original failing cases**. It also matches
94 isolated PV experiments and 26 boundary/cancellation experiments exactly.
Matching the buggy arithmetic model is diagnostic evidence, not a functionality PASS.
RTL remediation has not been applied in this task.

## Exact hardware scope

- Alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`.
- BDF: `0000:2a:00.1`; TH16, MXU16, L1 DMA tile 128.
- Image SHA256: `b9311154b3b56a5266e7e9d0f59f300c6f09c2e7ec6af9cdda01aa6b3770ac53`.
- Isolated PV: M=1, N=128, K=16/128/512, QDIR=1, WTRANS=0, qblock=128.
- Jobs 5672 and 5674 completed successfully; allocations released.
- Existing isolated HW runtime was used; no shared VCS runtime was overwritten.

## Why the error grows with length

PV computes `C[m,n] = sum_k P[m,k] * scale[k] * (Q[k,n] - zero[k])`.
The QROW hardware first creates `FP16(P * scale)` and then performs the integer
weighted accumulation. A CPU reference normally dequantizes V first, so a small
P*scale product is not prematurely flushed in the same way.

For near-uniform softmax, P is roughly 1/L. With scale about 0.02, the product crosses
FP16's minimum normal magnitude, 2^-14 = 0.00006103515625, around L=328.
The actual random inputs have varying P and scales, so some terms disappear already
at L=320 and many more disappear at L=511/512. This is an amplitude threshold,
not a fixed DMA-tile-count threshold.

### Uniform length sweep, physical K=512

All valid rows have Q=1, zero=0, scale=FP16(0.02), and P=FP16(1/L).
Inactive P entries are zero. Expected values below use the FP16 input values.

| Valid L | Expected output | FPGA output |
|---:|---:|---:|
| 128 | 0.02000427246 | 0.02000427246 |
| 256 | 0.02000427246 | 0.02000427246 |
| 288 | 0.02000427246 | 0.02000427246 |
| 304 | 0.02000427246 | 0.02000427246 |
| 320 | 0.02000427246 | 0.02000427246 |
| 336 | 0.02000427246 | 0 |
| 384 | 0.02000427246 | 0 |
| 511 | 0.02000427246 | 0 |
| 512 | 0.02000427246 | 0 |

### Minimal failure: only one MXU K microtile

For K=16, all A=P=2^-9, Q=2, zero=0:

| Scale | P*scale | Correct sum over 16 terms | FPGA |
|---|---|---|---|
| 2^-5 | 2^-14, normal | 2^-9 = 0.001953125 | 0.001953125 |
| 2^-6 | 2^-15, subnormal | 2^-10 = 0.0009765625 | 0 |

Both inputs and the final correct sum are normal FP16 values. The loss occurs in
an intermediate. K=128 and K=512 show the same product threshold. Setting zero=1
instead of zero=0 retains the failure, excluding asymmetric zero-point handling
as the source of the large discrepancy.

One-hot positions 0,15,16,127,128,255,256,511 all behave identically: normal products
survive and subnormal products disappear. Per-DMA-tile scale tags with normal
products match exactly. These checks and the exact arithmetic reconstruction do
not require an address-reordering or RAW-hazard explanation for these failures.

## RTL path and loaded-image evidence

- `hw/rtl/core/gemm/VX_gemm_compute_core.sv:1312`: QROW input scalers; `a_data` is
  the input activation, `b_data` the scale register, with `USE_LATENCY1_IP(1)`.
- `hw/rtl/core/gemm/VX_fp16_mul.sv:347`: the VIVADO hardware branch selects
  `xil_f16mul_latency1`.
- `hw/scripts/xilinx_ip_gen.tcl:200`: half-precision inputs/result, exponent width 5,
  fraction width 11, multiply operation, latency 1.
- `hw/rtl/core/gemm/VX_gemm_acc_internal.sv:290`: final `VX_f32_to_f16` conversion.
- `hw/rtl/core/gemm/VX_f32_to_f16.sv:74,151`: the subnormal branch shifts
  `fp32_mant_with_pad`, which only contains the fraction bits and padding; it never
  restores the implicit leading 1 of a normal FP32 input.

The archived preprocessed RTL beside this exact xclbin has the same multiplier
selection and missing-hidden-bit expression. This analysis is grounded in the
loaded image as well as the current source.

### Rounding at the FTZ boundary

A half-precision gradual-underflow model alone does not match the Xilinx boundary.
The measured multiplier rounds with an 11-bit normalized significand, then flushes
underflow. Near the smallest normal, the observed positive cutoff is
`2^-14 - 2^-26`; IEEE subnormal rounding would instead use a half-ULP of `2^-25`.
Eighteen bit-pattern cases around the saved L=511 A/scale pairs confirmed this.
The model implements that boundary; it is not merely an approximate magnitude test.

## Independent output-converter failure

For K=16, use two nonzero P values with opposite signs, scale=2^-4, Q=1, zero=0.
Both nonzero scaled inputs are normal. Their sum is chosen to be subnormal.
The same tests were repeated with signs reversed.

| Correct accumulated value | FPGA output | Missing contribution |
|---:|---:|---:|
| 6.12735748e-5 (normal control) | 6.12735748e-5 | 0 |
| 3.07559967e-5 | 2.38418579e-7 | 2^-15 |
| 1.54972076e-5 | 2.38418579e-7 | 2^-16 |
| 7.86781311e-6 | 2.38418579e-7 | 2^-17 |

The missing contribution is exactly the FP32 hidden bit. This is the GEMM-specific
output converter, separate from the SIMT F2F converter discussed in earlier work.

## Original four cases: complete reconstruction

| Capacity / M | Valid L | Nonzero scaled terms flushed | Final subnormal outputs | Original CPU relative L2 | Model vs FPGA |
|---|---:|---:|---:|---:|---|
| 512 / 1 | 320 | 6 | 0 | 2.9602% | Bitwise equal, 256 outputs |
| 512 / 1 | 511 | 301 | 1 | 23.7477% | Bitwise equal, 256 outputs |
| 512 / 1 | 512 | 303 | 0 | 24.2069% | Bitwise equal, 256 outputs |
| 320 / 4 | 320 | 5 | 3 | 1.4241% | Bitwise equal, 1024 outputs |

Term counts cover both query groups. The model uses FPGA softmax probabilities,
so this comparison isolates PV from the small upstream QK/softmax differences.

## Amplitude intervention

Replaying the saved probabilities into standalone PV, multiplying P by 2 before
GEMM and dividing the final output by 2 removes the large error without changing
addresses, capacity, tile count, V, scales, or zero points:

| Valid L | Original isolated PV relative L2 | P*2, output/2 relative L2 |
|---:|---:|---:|
| 320 | 2.9604% | 0.0298% |
| 511 | 23.7478% | 0.0201% |
| 512 | 24.2077% | 0.0283% |

Factors 4 and 16 also remove the large error in these fixtures. This is diagnostic
intervention, not a universal fix: real distributions may contain much smaller
probabilities/scales, and a fixed gain introduces overflow/dynamic-range concerns.

## Remediation scope

1. QROW scaling must preserve the small product before integer accumulation.
   Supporting gradual underflow in the half multiplier would fix these particular
   fixtures; for longer contexts, preserving a wider exponent range or tracking
   an exponent correction avoids FP16 intermediate range/precision loss more generally.
   Changing only the final output converter cannot recover terms already flushed.
2. Restore the FP32 hidden bit before shifting in the output converter's subnormal
   branch, and test positive/negative values, zero, rounding boundaries, and the
   subnormal-to-normal transition independently.

No production RTL or TVM lowering behavior was changed during this diagnosis.
The diagnostic software and evidence were added; the reported functionality
failures remain valid until fixes are implemented and verified.

## Reproduction and artifacts

- Script: `apps/vortex_llama3/debug_pv_numerics.py`.
- Output: `build/pv_debug_20261008/`.
- `hardware.sh`, `hardware_boundary.sh`: sourced-config/Slurm launch scripts.
- `hardware/`: 94 isolated cases, per-case NPZ inputs/outputs and `results.json`.
- `boundary/`: 26 final boundary/cancellation cases and arithmetic-model checks.
- `original_four_cases_explained.json`: flushed-term counts and exact reconstruction.
- `scaling_experiment.json`: amplitude interventions.
- `analyze.py`: reruns the original-four-case reconstruction without an FPGA.

The first export attempt failed before device execution because this Torch export
version requires a tuple return. It was corrected. `boundary_initial/` records an
initial cancellation control whose own intermediate underflowed; the final
`boundary/` uses normal intermediates for every output-converter test.
