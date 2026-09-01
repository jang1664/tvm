# Vortex Llama3-8B synthetic inference

This utility packages and reloads a partitioned, real-geometry Llama3-8B path with deterministic
synthetic asymmetric W4/K4/V4 parameters. It compiles phase-specialized token embedding and final
head modules plus one reusable prefill layer and one reusable decode layer. Each decoder module is
invoked 32 times with a distinct resident archive slice; this avoids embedding the 5.7 GB parameter
set in compiler IR.

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
