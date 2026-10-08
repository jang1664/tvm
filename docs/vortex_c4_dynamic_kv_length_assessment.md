> Update (2026-10-07): the updated `nodsp_fsm_update` C4 FSM now supports
> independent physical-parent and target extents. TVM's opt-in implementation
> and current validation are in [dynamic KV validation](vortex_dynamic_kv_length_validation.md).
> The assessment below describes the earlier RTL and remains historical context.

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

# C4 dynamic KV length: RTL feasibility assessment

Date: 2026-10-07. Scope: TH16/MXU16 C4 `v4_fix_pad`, layout ABI 3,
DMA tiles 128x128x128, head dimension 128. Static RTL/data-layout analysis only;
no RTL changes, VCS run, or new FPGA run were performed for this assessment.

## Conclusion

Fixed-capacity allocation with runtime attention length is feasible on the
current C4 arithmetic hardware. A software-only implementation can compact-pack
the active KV prefix and submit a runtime-sized GEMM. Reusing an already packed
capacity-sized KV buffer requires more care: QK can shorten N at MXU granularity,
but PV cannot shorten K inside a 128-row storage tile by changing target_K alone.

This distinction concerns the physical packed view consumed by C4, not whether
the canonical KV allocation must change. Current TVM keeps canonical fixed-shape
KV tensors and packs attention operands through the static layout plan; a
persistent packed KV representation is a separate optimization.

## Existing RTL and software contract

`hw/rtl/core/gemm/VX_gemm_fsm.sv` has separate orig_M/N/K and target_M/N/K
(job_t, lines 393-395). Target dimensions drive tile counts and tail sizes
(lines 1229-1253). Original dimensions drive several storage strides
(lines 1262-1276), so the basic storage/compute distinction already exists.

However, the distinction is incomplete: W_NT_STRIDE_LAST and SCALE_PK_FN use
k_last computed from target_K. The TMEM weight N-microtile stride and QROW
scale/ZP stride also use effective compute K (lines 1532-1534). Weight/scale DMA
loads transfer one contiguous compact tile sized by target K/N (lines 1725-1765).
Therefore target_K is also a layout assumption for a partial K tile.

The TVM helper currently writes identical M/N/K values to orig and target
registers (`kernel/include/vx_tvm_gemm.h:208-214`). Its ABI-v2 logical_n/logical_k
arguments only validate logical/padded extents and are not separately programmed
into the RTL. Existing software does not expose an independent storage stride.

## QK versus PV

For each KV head, GQA groups four query heads. GEMM M is 4 * query_length;
head dimension is 128. Quantization group size is 128.

| Operation | Variable dimension | C4 quantization / transpose | Existing packed-buffer prefix behavior |
|---|---|---|---|
| QK | N = active tokens; K = 128 | QDIR=0, WTRANS=1 | N may be rounded to 16; active N microtiles form a valid prefix |
| PV | K = active tokens; N = 128 | QDIR=1, WTRANS=0 | A partial K tile changes spacing between N microtiles |

For QK, keep storage orig_N at capacity if output/storage strides require it,
and use target_N=align_up(length,16). K stays 128 and M is unchanged. The host
still must handle per-head base addresses, packed output stride, active softmax,
and neutral padding. This is not a promise that the current static TVM graph
can use the smaller dimension without software changes.

For PV, take capacity=4096 and length=33. Align compute K to 48. The physical
first K tile still contains 128 rows and eight N microtiles. Each N microtile's
weight block occupies 128*16/2=1024 bytes. Existing shortened-K handling instead
uses 48*16/2=384 bytes, so N-microtile 1 reads from the wrong location. Its QROW
scale/ZP block similarly changes from 128*2=256 bytes to 48*2=96 bytes. The weight
DMA would read a contiguous 48*128/2=3072-byte prefix rather than gathering 48 rows
from every N microtile. Thus changing orig_K to capacity alone cannot fix PV.

## Viable implementations

1. **Active-prefix packing, no RTL change.** Keep canonical KV allocations at
   capacity, read only valid rows with capacity-based head strides, and pack a
   compact active operand with extent aligned to 16. Pass that runtime extent to
   the existing job interface. Allocation sizes may stay fixed; only contents
   and job dimensions change. Runtime packing, softmax, detiling, and head offsets
   must stop relying on compile-time execution extents. This retains per-step
   packing work proportional to length, but avoids processing all capacity.
2. **Fixed packed layout with 128-token PV buckets, no RTL change.** With a
   capacity that is a multiple of 128, use PV target_K=align_up(length,128),
   retaining the original storage dimensions. Every fetched K tile remains full,
   so the existing layout matches. QK can still use 16-token alignment. Initialize
   unread cache padding to finite neutral values and explicitly zero probability
   entries from length to PV's execution extent. For a non-multiple capacity,
   account separately for its actual physical tail; the simple formula is only
   asserted here for full 128-row storage tiles.
3. **Fixed packed layout with 16-token PV execution, focused RTL changes.**
   Separate physical K tile width from compute K tile width. Fetch the physical
   W/scale/ZP tile, keep physical N-microtile strides, and schedule only active
   K microtiles. Derive physical tails from orig_K; keep loop termination and
   accumulator is_last based on target_K. Fetching a whole final storage tile
   bounds residual DMA overhead to one tile and avoids adding a gather DMA path.
   This is a design proposal requiring VCS validation, not a verified patch.

For capacity 4096 and length 33, option 1 or 3 computes 48 sequence positions
rather than 4096; option 2 computes 128 for PV. Those ratios describe sequence-axis
work, not end-to-end speedup. Projection GEMMs and host/kernel launch overhead do
not shrink by the same ratios. Prefill causal triangles remain a separate issue:
limiting to prompt length does not itself eliminate all future-position QK work.

Recommendation: implement the runtime-length software contract first using
active-prefix packing, keeping current fix_pad hardware. If persistent packed
KV is desired immediately, start with 128-token PV buckets on this image. Optimize
the last physical tile in RTL only after separating these contracts and measuring
whether packing or rounded-tail work warrants it. No MXU datapath change is
indicated. RTL area/clock costs cannot be quantified from this static analysis.

## Checks performed and next validation

The existing TVM packing functions were exercised for lengths
1,15,16,17,33,127,128,129,255,256,257,4095,4096 with capacity 4096, head 128,
TH16/MXU16 and random payloads/nonconstant scales. All 13 QK aligned-16 prefixes
match the corresponding active packed data. PV aligned-16 prefixes fail exactly
when the final physical K tile is shortened; all 13 aligned-128 PV prefixes match.
Unused 512-byte scale-slot padding was excluded from payload comparisons.

Evidence: `build/c4_dynamic_kv_rtl_assessment_20261007/check_packed_prefix.py` and
`results.json`. These are host-side address/layout checks, not RTL simulation or
functional hardware proof.

Before deployment, run xrt-vcs-sim with the actual C4 config and fixed-capacity
buffers at the lengths above. Cover GQA heads with distinct sentinels, N microtiles
beyond the first, multiple K tiles, and M=4/128/132 (decode and prefill tile tails).
Verify QK/PV outputs, ignored-tail neutrality, unchanged allocated head strides,
DMA addresses/lengths, and fewer scheduled K/N microtiles. Separately test runtime
updates within a capacity and reject capacity overflow before submission.
