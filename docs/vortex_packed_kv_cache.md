# Quantization-fused persistent C4 KV cache

## Result

Implemented an opt-in packed KV path for the C4 IMPROVE layout ABI 3 and GEMM
submission ABI 3. The existing FP16-to-INT4 quantizer now writes each new token's
payload, scale, and zero point directly into the final GEMM layout. Packed K/V
buffers remain allocated across prefill and decode. QK/PV use head offsets into
those buffers and runtime valid-length extents; they no longer slice and repack
the complete K/V capacity before each GEMM.

The original quantization arithmetic is reused by transforming its output buffer
loads/stores. This is not a second quantization algorithm. The existing canonical
cache output ABI is retained: the fused kernel also writes the new token into the
canonical cache. Decode does not scan the old canonical tokens. Prefill creates
independent canonical outputs with a one-time initialization copy, because exported
`clone()` operations may alias the initial K/V zero tensors. Other graph operations
that assemble/copy canonical cache outputs, or pack the probability matrix as
GEMM-A, remain outside this change.

No RTL or floating-point arithmetic fixes were applied. Subnormal behavior is still
an acknowledged limitation of the image.

## Implementation

- `python/tvm/relax/backend/vortex/packed_kv.py`: quantizer store fusion, cache-chain
  recognition, explicit persistent state descriptors, and preparation API.
- `python/tvm/relax/backend/vortex/pipeline.py`: route QK/PV to persistent packed
  operands, retaining the existing GEMM-A and result helpers.
- `apps/vortex_llama3/run_synthetic_inference.py`: `--packed-kv-cache` package option,
  shared prefill/decode state descriptors, and per-layer persistent allocations.
- `apps/vortex_llama3/run_packed_kv_probe.py`: sequential FPGA append/attention probe.
- `tests/python/relax/test_vortex_packed_kv.py`: independent layout-oracle and
  compiler integration tests.

Supported graph pattern: canonical signed INT4 quantization followed by a contiguous
prefill cache-update chain starting at token zero, or a one-token dynamic append,
then W4 QK / causal softmax / W4 PV. Batch, KV head, capacity, and head dimension
are compile-time shapes. Decode valid length is runtime state. The initial path
requires the existing canonical batch/head ordering and singleton GQA storage axis.

The physical layout always uses capacity-derived tile strides. QK stores use the
transposed/QCOL layout; PV stores use QROW. QROW scales/zero points are replicated
into the N microtiles covered by the quantization group. Address calculations use
the configured DMA/MXU dimensions. The decode append kernel has no capacity-sized loop;
its work depends on the newly quantized token count and head dimension. The prefill
initialization copy is performed once and is not repeated during decode.

The position bounds guard is inside the device thread. Placing it outside the
thread region allowed host/device splitting to move the condition to the host and
caused the initial FPGA probe to leave cache buffers unchanged. Moving that guard
inside the device kernel resolved the issue; the final tests below use that fix.

## State contract and usage

The packed buffers are explicit caller-owned arguments appended after the original
function arguments. Allocate them as zero-filled buffers once per layer and sequence,
then pass the same buffers to prefill and every decode invocation. Compare prefill
and decode descriptors before reusing state. The inference application does this
automatically when packaging with `--packed-kv-cache` (which implies dynamic KV
length and requires `--layout-policy fused`). Existing packages must be rebuilt.

Decode canonical cache inputs are updated in place and must be uniquely owned.
Prefill creates independent outputs even when the initial K/V inputs alias.
Reset all packed buffers before starting a new sequence or rewinding the cache;
substituting a different canonical cache alone does not reconstruct packed state.
The application rejects its reference-cache substitution diagnostic in this mode.
An imported nonempty cache needs an explicit initialization conversion; the fused
path currently starts from an empty cache and fills it through quantization.

For direct compiler use:

```python
from tvm.relax.backend.vortex import prepare_packed_kv_cache, get_default_pipeline

mod, descriptors = prepare_packed_kv_cache(mod, target)
exe = relax.build(
    mod, target,
    relax_pipeline=get_default_pipeline(target, layout_policy="fused"),
)
# Allocate descriptors[*]["buffers"] once and append them to the original VM args.
# Preparation already lowers dynamic QK/PV/softmax; do not lower that pattern twice.
```

For the existing synthetic inference package command, add:

```text
--packed-kv-cache --layout-policy fused
```

Its normal parameter generation is unchanged. The targeted numerical fixture below
is separate from model-quality validation; this change does not clamp user inputs,
change model weights, or promise subnormal-free execution of arbitrary checkpoints.

## FPGA verification

Image alias: `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`.
TH16/MXU16; existing image, no new synthesis.

| Capacity | Batch | KV heads | Query heads per KV head | Prefill tokens | Decode appends | Checked valid lengths | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 320 | 1 | 1 | 2 | 4 | 316 | 4,15,16,17,127,128,129,255,256,257,288,320 | PASS |
| 512 | 2 | 2 | 2 | 4 | 508 | Above plus 511,512 | PASS |

All 826 graph invocations completed. At all 26 checkpoints, the persistent INT4,
scale, and zero-point buffers match the independent existing CPU full-layout packer
bit for bit when given the actual canonical hardware cache. This verifies old-token
preservation, inactive zero tails, multi-head offsets, and microtile/DMA-tile boundaries.
Both probabilities and context pass the existing CPU numerical thresholds.

| Capacity / batch / KV heads | Maximum probability relative L2 | Maximum PV relative L2 | Subnormal PV scaler products at checkpoints |
| --- | ---: | ---: | ---: |
| 320 / 1 / 1 | 0.102475% | 0.026180% | 0 |
| 512 / 2 / 2 | 0.110607% | 0.039050% | 0 |

The test uses FP16 queries of 0.125 and random FP16 K/V source values in [1,5),
producing quantization scales around 0.33. The probe explicitly checks nonzero
`probability * value_scale` products against the FP16 minimum normal value. This
avoids the known FTZ issue without suppressing numerical failures or changing
production arithmetic. Small ordinary quantization/rounding differences from the
PyTorch reference remain within the existing thresholds.

Host timing printed by the probe is not an isolated packing benchmark and is not
an end-to-end Llama speedup measurement.

## Additional checks and artifacts

CPU tests execute the transformed TIR and compare it with the original quantizer
and independent packers, including head dimensions 128/256, quantization groups
32/128, capacities 320/512, and multi-token prefill. Compiler tests cover both
prefill and decode Llama layer graphs.

Actual Llama3-8B dimensions (hidden 4096, intermediate 14336, head dimension 128,
32 query heads, 8 KV heads), one layer, prefill 4, capacity 512: both prefill and
decode compiled successfully through the inference application's build helper,
and their persistent state descriptors agree. This was a compile/ABI check, not a
new full-model numerical run on FPGA.

Artifacts: `build/packed_kv_20261008/`.

- `probe320/` and `probe512_b2h2/`: compiled modules, generated device source,
  lowered IR, state descriptors, numerical checkpoints, and `results.json`.
- `hardware_final320.log` / `hardware_final512.log`: final FPGA runs, including shared
  initial K/V cache inputs to exercise the prefill alias case.
- `llama_layer/` and `compile_llama_final.log`: actual-dimension layer compile evidence.
- `final_tests3.log`: 64 focused regression tests passed.
- Initial `hardware.log` / `hardware2.log` document the corrected guard-placement issue.

Reproduction (compile outside Slurm, run inside a single FPGA allocation with the
real-XRT environment):

```bash
python apps/vortex_llama3/run_packed_kv_probe.py \
  --manifest /opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c1_f100_fpint_L2cache_96c0f69b12/manifest.json \
  --output-dir build/packed_kv_20261008/probe512_b2h2 \
  --capacity 512 --batch-size 2 --kv-heads 2 --compile-only
# Use the same arguments with --run-only under the configured FPGA allocation.
```
