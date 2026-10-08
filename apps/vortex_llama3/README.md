# Vortex Llama3-8B synthetic inference

This utility packages and reloads a partitioned, real-geometry Llama3-8B path with deterministic
synthetic asymmetric W4/K4/V4 parameters. It compiles phase-specialized token embedding and final
head modules plus one reusable prefill layer and one reusable decode layer. Each decoder module is
invoked 32 times with a distinct resident archive slice; this avoids embedding the 5.7 GB parameter
set in compiler IR.

## Config-driven TH16 C1–C4 functionality checks

The current candidate fixture is [candidates_th16_axi_fix.json](candidates_th16_axi_fix.json).
It maps C1–C4 to the exact U55C images in
`vortex_fpint-feat-gemv/ci/fpga_bin_alias_map.yaml`. Capability resolution uses each image's
manifest rather than its alias spelling. C2 uses naive W4A16 MXU for all linear projections
and FP16 TCU for QKᵀ and PV, in both prefill and decode.

C4 now exclusively selects `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_fix_pad`
(packed output, layout ABI 3). Recompile packages for this image; packages built
for the previous C4 image are historical artifacts. The new image passed 10
unfused GEMM cases and four fused chains with bitwise agreement against their
unfused baselines. The same image also passed all 36 mini-Llama boundary checks:
alone/fused S1/S2 bytecode and S1 compiled VM, with bitwise-identical alone/fused
outputs and S1 bytecode/compiled outputs. These check embedding, one decoder
layer, and the head separately, carrying the actual prefill cache into decode;
they do not establish 32-layer end-to-end inference coverage. See
[the new C4 validation record](../../docs/vortex_packed_c_hardware_validation.md).

These commands use the existing configured builds and the installed LP64F profile. They
do not use the historical `vortex_base` image shown in the later sections of this README.

```bash
export TVM_HOME=/home/jaeyongjang/project.local/tvm
export TVM_VORTEX_HOME=/home/jaeyongjang/project.local/vortex_fpint-feat-gemv
export TVM_VORTEX_PROFILE_ROOT=/opt/vortex_profiles/rv64imaf_zfh_lp64f
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/.local/python310-runtime:$TVM_HOME/apps"
export TVM_LIBRARY_PATH="$TVM_HOME/build/lib"
export LD_LIBRARY_PATH="$TVM_HOME/build/lib:$TVM_VORTEX_HOME/build/runtime:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TVM_PYTHON=/home/jaeyongjang/.conda/envs/vortex/bin/python
cd "$TVM_HOME"
```

TVM's CMake `USE_VORTEX` must point to
`$TVM_VORTEX_HOME/build/runtime/libvortex.so`. Rebuild after compiler/runtime changes:

```bash
cmake -S . -B build -DUSE_VORTEX="$TVM_VORTEX_HOME/build/runtime/libvortex.so"
cmake --build build --parallel 4
```

Check source config versus image manifest, then compile a short device-code smoke test:

```bash
"$TVM_PYTHON" apps/vortex_llama3/compile_backend_matrix.py \
  --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C1,C2,C3,C4 --check-aliases-only
"$TVM_PYTHON" apps/vortex_llama3/run_backend_gemm_probe.py \
  --matrix-regression --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C1,C2,C3,C4 --cases linear_0 --compile-only \
  --output-dir build/config_gemm_th16/compile_smoke
```

Hardware matrix runs accept one candidate per process. Run candidates sequentially on an
allocated FPGA. For example, this C2 allocation rebuilds the Vortex kernel library using
the matching source config, selects the allocated device, and tests linear plus attention:

```bash
srun --gres=fpga:u55c:1 --cpus-per-task=4 --mem=16G --time=00:30:00 bash -s <<'BASH'
set -euo pipefail
source "$TVM_VORTEX_HOME/configs/naive_th16_tcol16_m16_L16_bigmem_all_bram_acc_tcu_base_pnr_v3.sh"
make -C "$TVM_VORTEX_HOME/build/kernel" CONFIGS="$CONFIGS"
source "$TVM_VORTEX_HOME/ci/xrt_device_detect.sh"
xrt_smi_bin="$(resolve_xrt_smi)"
export XRT_DEVICE_INDEX="$(detect_single_accessible_xrt_index "$xrt_smi_bin")"
export XRT_DEVICE_BDF="$(resolve_xrt_user_bdf "$XRT_DEVICE_INDEX")"
export XRT_INI_PATH=/dev/null VORTEX_DRIVER=xrt TARGET=hw
unset FPGA_BIN_DIR XRT_XCLBIN_PATH
cd "$TVM_HOME"
"$TVM_PYTHON" apps/vortex_llama3/run_backend_gemm_probe.py \
  --matrix-regression --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C2 --output-dir build/config_gemm_th16/hw_C2
BASH
```

For C1/C3/C4, change both the sourced config and `--aliases` to the corresponding fixture
entry. `--cases` selects a focused rerun; use a separate output directory to preserve earlier
results. Each run writes `results.json`, exported modules, and actual/reference NPZ arrays.
The GEMM rounding envelope is fixed before execution at `0.003 + 0.003 * abs(reference)`;
nonfinite output fails. C4 additionally has `descriptor_chain`, which checks a branched and
chained GEMM graph. Run it with `--layout-policy alone` and then `fused`; the fused run
requires a recorded shared-A layout reuse. `descriptor_chain` uses M8 and directly reuses
GEMM-C as GEMM-A; `descriptor_chain_tail` uses M4 and checks direct C-to-A reuse
on the current layout ABI 3 image. In the matching C4 hardware environment, run:

```bash
"$TVM_PYTHON" apps/vortex_llama3/run_backend_gemm_probe.py \
  --matrix-regression --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C4 --cases descriptor_chain,descriptor_chain_tail --layout-policy alone \
  --output-dir build/config_gemm_th16/descriptor_alone
"$TVM_PYTHON" apps/vortex_llama3/run_backend_gemm_probe.py \
  --matrix-regression --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C4 --cases descriptor_chain,descriptor_chain_tail --layout-policy fused \
  --baseline-output-dir build/config_gemm_th16/descriptor_alone \
  --output-dir build/config_gemm_th16/descriptor_fused
```

The chain checks each GEMM with the standalone QCOL FP32 dequantization contract and
keeps the Torch FP16-weight and end-to-end reference errors separately. The rounding
envelope remains unchanged. The fused run also requires bitwise equality to the earlier
alone-layout run.

Compile small Llama embedding/layer/head packages before allocating the FPGA:

```bash
"$TVM_PYTHON" apps/vortex_llama3/compile_backend_matrix.py \
  --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C1,C2,C3,C4 --cases S1,S2 --model-size mini \
  --artifact-root build/config_gemm_th16/mini_matrix
"$TVM_PYTHON" apps/vortex_llama3/compile_backend_matrix.py \
  --candidate-map apps/vortex_llama3/candidates_th16_axi_fix.json \
  --aliases C4 --cases S1,S2 --model-size mini --c4-layout-policy fused \
  --artifact-root build/config_gemm_th16/mini_fused
```

Inside the same hardware environment as above, run the checked package loader and all six
boundaries, including decode using the preceding prefill cache:

```bash
"$TVM_PYTHON" apps/vortex_llama3/run_backend_probe.py --mini-regression \
  --package build/config_gemm_th16/mini_matrix/packages/C2/S1/package.json \
  --vortex-home "$TVM_VORTEX_HOME" --exec-mode bytecode \
  --output-dir build/config_gemm_th16/mini_hw_C2
```

S1 is batch 1, prompt length 1, cache capacity 8; S2 is batch 1, prompt length 7, capacity 16.
Both use hidden/intermediate/vocabulary size 256, head dimension 128, two query heads and
one KV head. S1 also exports compiled-VM modules: repeat with `--exec-mode compiled` and
a different output directory. For C4 fused, select the package in `mini_fused`.

Synthetic `--width 8`, `32`, or `64` is restricted to `--compile-only`; it does not establish
hardware functionality without a matching bitstream. Host packing tests cover width 16 too.
There is no 16/32 whitelist. Packed INT4 requires A≥2, square power-of-two A and threads=A
for MXU paths; DMA/SRAM/accumulator constraints can reject larger A.

Use a fresh artifact root after changing image/config, profile ABI, layout/packing, model
geometry, quantization, or device helper contracts. Profile version 1 and mismatched image
packages are rejected. `--force` recompiles graph modules when archive contracts remain
unchanged; it is not a packing migration. The package loader verifies image/profile,
archive and module hashes. See the [validation report](../../docs/vortex_config_driven_gemm_validation.md)
for measured coverage and the standalone TCU tolerance caveat.

## Environment

Use the configured TVM build and source the configuration matching the pinned U55C image:

```bash
source /home/jaeyongjang/project.local/vortex_base/configs/improve_th32_tcol32_hwexp_dcache_sxbar_f16_bigmem.sh
export TVM_HOME=/home/jaeyongjang/project.local/tvm
export VORTEX_HOME=/home/jaeyongjang/project.local/vortex_base
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/.local/python310-runtime:$TVM_HOME/apps"
export TVM_LIBRARY_PATH="$TVM_HOME/build"
export LD_LIBRARY_PATH="$TVM_HOME/build:$VORTEX_HOME/build/runtime:$LD_LIBRARY_PATH"
export VORTEX_DRIVER=xrt
export XRT_XCLBIN_PATH=/opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c_f100_fpint_64300e5119/bin/vortex_afu.xclbin
```

## Package and run

### Host-only C1/C3 compile matrix

`compile_backend_matrix.py` exports the same backend-neutral Llama3-8B graph for the exact C1 and
C3 aliases, materializes profile-bound parameters, compiles S1-S4, and reloads every generated VM
module. S1 includes both bytecode and compiled VM modes; S2-S4 use bytecode mode. No FPGA is opened
by this command.

```bash
source "$VORTEX_HOME/configs/tcu_th32_c1_rev2.sh"
export TVM_VORTEX_HOME="$VORTEX_HOME"
export TVM_VORTEX_BUILD_DIR="$VORTEX_HOME/build"
export TVM_VORTEX_LLVM_ROOT=/opt/vortex/llvm-vortex
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

/home/jaeyongjang/.conda/envs/vortex/bin/python \
  apps/vortex_llama3/compile_backend_matrix.py \
  --artifact-root "$TVM_HOME/build/llama3_c1_c3_compile_matrix" \
  --aliases C1,C3 \
  --cases S1,S2,S3,S4
```

The runner resolves each alias through `ci/fpga_bin_alias_map.yaml` and records the exact config,
manifest, xclbin hashes, normalized target, profile fingerprint, logical archive hash, physical
materialization hash, artifact hash, kernel inventory, compile time, and size. It fails closed when
any of those identities change. C2 policy and lowering have synthetic fixture coverage, but its
profile-bound compile remains deferred until the mapped C2 image directory contains the intended
binary and sibling manifest; another config or xclbin must not be substituted.

### C1/C3 GPU-versus-U55C numerical validation

Acceptance uses a CUDA reference only. CPU execution is available solely through the explicit
`--diagnostic-cpu` reference-generator option, is labelled `diagnostic_cpu`, and is rejected by the
physical U55C runner. Generate one reference for each package after compilation, for example:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jaeyongjang/.conda/envs/hw_autogen/bin/python \
  apps/vortex_llama3/generate_backend_reference.py \
  --package build/llama3_c1_c3_compile_matrix/packages/C3/S4/package.json \
  --output build/llama3_c1_c3_numerical_validation/references/C3/S4.npz \
  --alias C3 --case S4 --determinism-replays 2
```

For a physical run, source the alias-matched Vortex config, select its exact mapped xclbin, and keep
one process for canonical prefill/decode plus free-running inference:

```bash
source "$VORTEX_HOME/configs/naive_gemm_th32_tcol32_hwexp_dcache.sh"
export VORTEX_DRIVER=xrt XRT_INI_PATH=/dev/null
export XRT_XCLBIN_PATH=/opt/vortex_fpga_bins/fpint/xrt_hw_u55c_c_f100_fpint_9600db3a37/bin/vortex_afu.xclbin
/home/jaeyongjang/.conda/envs/hw_autogen/bin/python \
  apps/vortex_llama3/run_backend_validation.py \
  --package build/llama3_c1_c3_compile_matrix/packages/C3/S4/package.json \
  --reference build/llama3_c1_c3_numerical_validation/references/C3/S4.npz \
  --alias C3 --case S4 --exec-mode bytecode --mode both \
  --trace-output build/llama3_c1_c3_numerical_validation/runs/C3/S4-bytecode-both.json \
  --mismatch-dir build/llama3_c1_c3_numerical_validation/mismatches/C3/S4
```

`run_backend_validation.py` validates package, profile, xclbin, reference, prompt, and tensor
inventory before opening XRT. Canonical mode compares all 32 layer states, asymmetric K4/V4 cache
state, normalized hidden, logits, and top-1 against GPU-recorded inputs. Free-running mode enforces
finite outputs and exact cache lengths. For S1, use `--mode free-running --free-repetitions 2` to
prove stable hashes in one process and one device open.

Canonical validation defaults to fixed device buffers for hidden/KV inputs and performs an exact
device readback before every layer invocation. Use `--canonical-input-staging fresh` or
`--no-verify-canonical-input-readback` only for diagnosis; acceptance traces retain the defaults.
The runner also assigns a process-unique `VORTEX_SHM_PATH` when one is not supplied and records the
actual opened U55C BDF, resolving it from the DRM render node when Slurm did not export a BDF.

Generate a compact, fail-closed coverage artifact from the finished traces with:

```bash
/home/jaeyongjang/.conda/envs/hw_autogen/bin/python \
  apps/vortex_llama3/collect_backend_validation_evidence.py \
  --package-root build/llama3_c1_c3_compile_matrix/packages \
  --reference-root build/llama3_c1_c3_numerical_validation/references \
  --run-root build/llama3_c1_c3_numerical_validation/runs \
  --failure-manifest /absolute/path/to/backend_failure_evidence.json \
  --output build/llama3_c1_c3_numerical_validation/evidence.json \
  --require-complete
```

The collector rejects CPU-labelled or stale references, cross-profile traces, non-finite metrics,
invalid cache lengths, changed persistent hashes, CUDA initialization in the XRT process, and any
missing C1/C3 S1-S4 coverage. A failure manifest can retain hash-verified failed attempts and
diagnostic counterexamples even when a successful trace cannot be produced. Such evidence makes
the affected backend verdict `FAIL`; it does not fill the missing successful-coverage slot, so
`--require-complete` still exits unsuccessfully.

For production failures that disappear in a fresh full-layer replay, compile and run the exact
stage probes. The runner can first execute repeated production prefill layers, or replay the full
32-layer canonical prefill chain, before repeating a decode checkpoint graph in the same process:

```bash
/home/jaeyongjang/.conda/envs/hw_autogen/bin/python \
  apps/vortex_llama3/run_backend_stage_probe.py \
  --package build/llama3_c1_c3_compile_matrix/packages/C1/S3/package.json \
  --reference build/llama3_c1_c3_numerical_validation/references/C1/S3.npz \
  --stage-package \
    build/llama3_c1_c3_numerical_validation/stage_packages/C1/S3-decode-checkpoints/package.json \
  --alias C1 --case S3 --probe layer_checkpoints_decode \
  --warmup-prefill-chain-repetitions 1 --full-resident-archive \
  --repetitions 32 \
  --trace-output build/llama3_c1_c3_numerical_validation/probes/C1/S3-stage.json
```

These options are diagnostic only: a passing reduced or stage-localized graph cannot replace a
failed end-to-end acceptance trace.

The initial S1/alone compile, package, eager-reference generation, and run is:

```bash
/home/jaeyongjang/.conda/envs/py310/bin/python \
  apps/vortex_llama3/run_synthetic_inference.py \
  --mode package-and-run \
  --layout-policy alone \
  --prompt-token-ids 1 \
  --decode-steps 3 \
  --cache-capacity 8 \
  --sampling argmax \
  --reference \
  --artifact-dir "$TVM_HOME/build/llama3-s1-alone" \
  --trace-output "$TVM_HOME/build/llama3-s1-alone/trace.json"
```

To prove fresh-process reload without PyTorch export or TVM compilation, rerun the same shape and
policy with `--mode run`. Batch rows use semicolons, for example
`--prompt-token-ids '1,2,3;4,5,6'`. Use `--archive-manifest .../parameters/manifest.json` when
packaging another shape or policy to share one validated archive instead of writing another copy.
The run path also accepts `--allocator pooled|naive`; use `pooled` for controlled address reuse and
`naive` for the original allocation behavior. `--state-persistence` and `--fixed-hidden-input` are
diagnostic controls for separating VM-output lifetime, state-copy, and hidden-address effects;
they are not accepted production workarounds.

Use `--inference-repetitions N` to keep one Python process, one device open/xclbin programming
event, and one resident parameter archive while repeating complete inference. Add
`--continue-after-inference-failure` to collect later repetitions after a numerical mismatch. The
aggregate trace is written to `--trace-output`, with each successful repetition written beside it
as `*.repetition-N.json`.

`--reference` generates canonical eager tensors in an isolated process, saves them as `.npz`, exits
that process before XRT opens, and applies hybrid FP16 plus semantic KV-cache checks. Small FP16
reference values use absolute error; larger values use relative error, with relative-L2, cosine,
and violation-fraction guards. The JSON trace records token IDs, cache lengths, hashes, top-k,
launch/transfer counts, latency, revisions, fingerprints, and comparison summaries.

For the accepted S1/alone hardware path, use the default `--state-transport device-copy` and keep
`--diagnostic-layer-retries 0`. `--diagnostic-layer-checks
--diagnostic-canonical-phase-limit 2` enforces canonical comparisons through prefill, decode 1,
and decode 2; decode 3 and later retain finite and hidden-magnitude sanity checks while recording
canonical drift. This boundary is based on exact-input eager replay of the quantized decode state.
The retry option remains available for fault capture, but it is not required by the accepted dense
Hadamard package.

## Current hardware status

Packaging, reload, archive validation, real Llama3-8B shapes, and the host reference path are
implemented. Physical S1/alone embedding, 32 decoder layers, LM head, prefill, and three decode
steps pass on the pinned U55C with retries disabled. Two complete chains pass in one persistent
process with one XRT initialization, unchanged resident parameters, and device-to-device state
transport; both generate `[89754, 29229, 89754]`, preserve exact cache lengths 1/2/3/4, and produce
identical per-step hashes.

The old `inf`/large-hidden symptom was not gradual model divergence. Checkpointed runs isolated the
first bad result to the final pairwise stage of the MLP R4 Hadamard transform. The production graph
now uses an equivalent dense mixed-radix Hadamard, avoiding the intermittent butterfly-kernel
boundary while preserving the normal eight-output decoder ABI. It passed an alternating 100-call
stress test and two persistent device-copy inference repetitions with zero retries and zero
non-finite values. No RTL or xclbin change was required.

Independently, W4/K4/V4 cache requantization makes the live chain drift from the canonical PyTorch
chain by decode 3. Replaying a captured live input in eager matches hardware at hidden relative-L2
0.0005878 and cosine 0.999999827, proving the local layer calculation remains accurate. Fused and
S2-S4 remain later milestones; S1/alone is accepted. See the Vortex-side execution report for the
full evidence.

To distinguish invocation-count failures from device-address failures, use
`debug_repeated_layer_addresses.py`. It records the physical input, parameter, and VM state
addresses and supports fixed, ping-pong, preallocated address-sweep, and reallocated inputs with
pooled or naive VM allocation. `--alternate-layer` switches two complete fixed decoder contexts;
`--copy-state`, `--copy-state-scope`, and `--copy-method` isolate individual state-transfer paths.
Start with fixed input and pooled output reuse:

```bash
/home/jaeyongjang/.conda/envs/py310/bin/python \
  apps/vortex_llama3/debug_repeated_layer_addresses.py \
  --artifact-dir "$TVM_HOME/build/llama3_synthetic_s1_alone_stable" \
  --reference-artifact "$TVM_HOME/build/llama3_synthetic_s1_alone_stable/reference-7c9fa136d441-steps0.npz" \
  --layer 28 --iterations 20 --input-mode fixed --allocator pooled \
  --trace-output "$TVM_HOME/build/llama3-address-fixed.jsonl"
```

For long canonical checkpoint stability tests, use `debug_canonical_layer_range.py`. It validates
that every requested reference array exists before opening XRT, accepts inclusive layer ranges,
and flushes a JSONL event before and after each layer launch and D2H boundary. The trace includes
the Slurm allocation, BDF, current/package revisions, xclbin/package/reference hashes, tensor
addresses and sizes, internal launch names/counts, output hashes, finite/magnitude summaries, and
the first runtime error. `--copy-mode none|hidden|full`, `--allocator pooled|naive`, and
`--repetitions N` isolate copy volume, allocation policy, and persistent-process call count without
using retries. The embedding probe runs only at explicit diagnostic boundaries and never resets or
reprograms a failed device.

```bash
/home/jaeyongjang/.conda/envs/py310/bin/python \
  apps/vortex_llama3/debug_canonical_layer_range.py \
  --artifact-dir "$TVM_HOME/build/llama3-s2-fused" \
  --reference-artifact "$TVM_HOME/build/llama3-s2-fused/reference.npz" \
  --xclbin "$XRT_XCLBIN_PATH" \
  --phases prefill,decode_1,decode_2,decode_3 \
  --layer-range 0:31 --repetitions 3 \
  --copy-mode full --allocator pooled --vm-scope shared \
  --health-probe phase --expected-bdf 0000:3d:00.1 \
  --trace-output "$TVM_HOME/build/llama3-s2-fused/device-events.jsonl"
```

The generated tokens are deterministic interface evidence only. They have no language meaning
until a real checkpoint is converted and loaded.


### Full-size single-layer check on the current C4

The `v4_fix_pad` C4 image also passes the actual Llama3-8B hidden/FFN/head sizes
with one decoder layer: prefill lengths 1 and 7, batch 1, followed by one-token
decode using the prefill cache. All 12 checks pass across alone/fused and S1
bytecode/compiled execution, with bitwise-equal alone/fused outputs. Parameters
are synthetic; this is not a pretrained 32-layer accuracy test.

Use `run_backend_probe.py --layer-regression --package <package.json>
--vortex-home <vortex-root> --exec-mode bytecode --output-dir <results>` with a
full-size compile package under a configured FPGA allocation. The existing
`--mini-regression` behavior is retained. See
[`vortex_packed_c_hardware_validation.md`](../../docs/vortex_packed_c_hardware_validation.md)
for dimensions, metrics, and the exact allocation/compilation commands.


### Full random-weight Llama3-8B on current C4 (2026-10-07)

The current `v4_fix_pad` C4 also passes the complete 32-layer chain with distinct
random weights: batch 1, one-token prefill followed by three stateful decode
phases, fused layout, bytecode VM. All 128 layer invocations and all four final
logits/cache checks pass, with CPU top-1 agreement in every phase. Maximum
logits relative-L2 error is 0.02714%. Unlike the historical result above, this
run enforces the canonical reference through decode 3 with no phase exemption,
retries, or reference-input replacement. These are random-weight functionality
results, not language-quality results. Exact commands and evidence are in
[`vortex_packed_c_hardware_validation.md`](../../docs/vortex_packed_c_hardware_validation.md),
under the full 32-layer section.

### Dynamic KV valid length on the updated C4 FSM

`run_synthetic_inference.py --mode package --dynamic-kv-length ...` enables
runtime QK/PV prefix submissions. Use the
`improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_nodsp_fsm_update`
image, whose authoritative manifest declares `vortex_layout_abi_version: 3`
and `vortex_gemm_abi_version: 3`. An older FSM image is rejected.

The compiled decode executable accepts changing scalar valid lengths without
recompilation. Physical KV capacity and packed DMA strides stay fixed. QK uses
`target_n = ceil(valid_length / MXU_COL) * MXU_COL`; PV uses the corresponding
`target_k` with `MXU_ROW`. Tail scales and PV activations are neutralized before
packing, and softmax only reduces over the causal valid prefix. The valid length
must be between 1 and capacity. Batch, query count, head geometry, and capacity
remain compile-time shapes; changing those still requires another package.

This option currently repacks capacity-sized canonical K/V buffers and adds
masking kernels. It reduces GEMM work but does not promise lower end-to-end
latency. The ordinary static-capacity path remains the default. Runtime execution
uses the setting recorded in `package.json`; `--dynamic-kv-length` is a packaging
option and does not change a previously built package.

`run_dynamic_kv_probe.py` compares static and dynamic attention on the FPGA using
one executable per mode across many valid lengths. It poisons unused K/V scales
with NaN/Inf, checks bitwise agreement with static execution, and independently
checks CPU numerical limits. It records failures and exits nonzero when CPU
limits fail, even if static and dynamic FPGA outputs match.

See [dynamic KV validation](../../docs/vortex_dynamic_kv_length_validation.md)
for the current hardware evidence and the separate long-PV numerical limitation.
