# Config-driven GEMM 검증 기록

2026-10-07 기준 현재 C4는 `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_fix_pad`
만 사용한다. 아래 2026-10-06의 C4 결과는 이전 이미지의 이력이며 현재 C4의 검증 범위로
계산하지 않는다. 현재 C4는 unfused 10개와 fused GEMM chain 4개를 통과했고, 새 이미지의
mini-Llama도 alone/fused S1/S2 bytecode 및 S1 compiled VM에서 36개 boundary를 모두
통과했다. [현재 C4 검증 기록](vortex_packed_c_hardware_validation.md)을 참조한다.

2026-10-06. 계획의 필수 구현과 검증을 완료했다. TH16 실제 FPGA 기능 검증과 다른 A의 host/device compile 검증을 구분하여 기록한다.

## 확인한 구현

- Config/manifest capability 기반 `auto` policy: C1=TCU, C2 linear=naive MXU 및 QKᵀ/PV=TCU, C3=naive MXU, C4=improve MXU.
- 명시적 LMEM bytes, 물리 TMEM bank 수, naive accumulator mode를 target/compiler/runtime metadata까지 전달한다. Profile version 2를 사용하고 이전 profile을 거부한다.
- MXU는 square power-of-two A≥2 및 threads=A 조건을 검사한다. A≥2는 packed INT4 계약이며 16/32 whitelist는 사용하지 않는다. TCU-only에는 MXU geometry 비교를 적용하지 않는다.
- TCU tile은 기존 C++ WMMA geometry와 일치하며, pointwise helper의 launch 크기는 target 한도에 맞춘다.
- Naive transpose qparam은 하드웨어의 K/N 순서로 변환한다. Rank-2 tail은 padding/slicing으로 처리한다.
- C4 DRAM stripe는 DMA channel 수와 별개로 ABI v2의 8-row 단위를 사용한다. RTL `VX_gemm_fsm.sv:700` 및 standalone `fpint_gemm_ffn_hw/layout.h`의 정의를 따른다.
- C4 A는 DMA tile 끝을 padding하고 C는 각 microtile의 행을 padding한다. Descriptor가 이 차이를 기록하며, 정렬된 M에서만 C→A를 직접 재사용한다. 비정렬 M에서는 detile/repack하고, sibling projection의 A→A 공유는 유지한다.
- C4 logical archive materialization은 기존 improve packing primitive를 사용한다.
- 기존 matrix/probe 도구에 candidate map, compile-only geometry 검증, 작은 graph 실행 모드를 추가했다.

## TH16 실제 FPGA kernel 기능 검증

고정 seed 20261006의 random input/weight를 사용했다. 실행 전에 정한 허용치는 `0.003 + 0.003 * abs(reference)`이며 실패 후 변경하지 않았다. NaN/Inf는 실패한다. 모두 실제 TVM export/load/VM/XRT 경로로 실행했다.

| Candidate | Case 수 | PASS | 최대 절대 오차 | 증거 |
| --- | ---: | ---: | ---: | --- |
| C1 | 3 | 3 | 0.00195312 | `build/config_gemm_th16/hw_C1/results.json` |
| C2 | 22 | 22 | 0.00390625 | `build/config_gemm_th16/hw_C2/results.json` |
| C3 | 16 | 16 | 0.00390625 | `build/config_gemm_th16/hw_C3/results.json` |
| C4 | 16 | 16 | 0.00390625 | `build/config_gemm_th16/hw_C4/results.json` |

- C1: exact, tail, 256³ TCU case 3개.
- C2: linear 3개, QKᵀ 3개, PV 3개, QBLK16/32/128 × QDIR0/1 × transpose 0/1의 12개, tail 1개.
- C3/C4: linear 3개, quantization/transpose 12개, tail 1개.
- Transpose/tail 및 작은 M의 큰 layout 오류는 수정 후 재실행해 통과했다. 초기 실패를 FP16 conversion 예외로 처리하지 않았다.
- 추가 C2 동일 shape (M4/N256/K256)의 linear/QKᵀ/PV 3개도 PASS했다. Inventory는 각각 naive/TCU/TCU이며 `hw_C2_role_shape/results.json`에 기록했다. Linear는 기본 matrix와 중복된 확인이다.
- C4 M8/M4 descriptor chain을 alone/fused 각각 실행한 4개도 PASS했다. Fused 두 shape 모두 alone output과 bitwise equality를 통과했다. 기본 57개와 추가 attention 2개 및 descriptor 4개를 합하면 63개의 서로 다른 검증 조합이다.

## Host 및 device compile

- 최종 관련 host tests: **271 PASS / 48 SKIP**. Skip은 기존 opt-in HW/compile 환경 검사이며, 위 TH16 기능 검증을 대체하는 증거로 세지 않았다. 로그는 `build/config_gemm_th16/logs/tvm-config-gemm-tests-final-verified.log`에 보존한다.
- A=8/16/32/64 weight packing의 독립적인 index 계약 검사와 C++ WMMA geometry oracle 비교가 통과했다.
- TH16 네 candidate 및 합성 A=8/32/64 네 backend 경로의 실제 device compile이 통과했다. `build/config_gemm_th16/synthetic_<A>/results.json`을 참조한다.
- 다른 A의 bitstream을 실행하지 않았다. 합성 profile은 `--compile-only`에서만 허용한다.
- Source config와 image manifest의 normalized compile fields를 비교하여 충돌하면 실행 전에 실패한다. Build가 추가한 macro 차이는 `--check-aliases-only`로 표시한다.
- Target 검증에서 A=0/24, 비정사각 MXU, threads 불일치, DMA tile 비정렬 및 이전 profile version을 거부한다. TCU-only target은 사용하지 않는 MXU의 shape/정렬 비교를 적용하지 않는다.
- 실제 C2 package의 다른 image 선택, 변조된 profile, 변조된 module hash가 모두 loader에서 거부됐다. `build/config_gemm_th16/package_rejection_results.json`.

## Standalone 기준 검증

Configured Vortex `build/`에서 `ci/run_black.sh hw --fpga-bin ...`를 사용했다.

- C2/C3 naive, M4/N256/K256/QBLK16: PASS (`/tmp/tvm-th16-baselines.log`).
- C4 improve, M4/N256/K256/QBLK32: PASS (`/tmp/tvm-th16-c4-baseline-q32.log`). 기존 standalone host는 QBLK16을 거부한다. TVM helper의 QBLK16 HW 기능 검증은 위 표에 포함돼 있다.
- C1 standalone TCU M16/N16/K32: strict absolute threshold 0.001에서 256개 중 1개가 실패했다(오차 0.00390625). TVM의 random TCU 3개는 사전에 정한 허용치로 PASS했다. 로그 `/tmp/tvm-th16-c1-baseline.log`.

## Llama graph 기능 검증

C1~C4의 S1 mini graph는 embedding/prefill/decode/head 6개 boundary를 모두 통과했다. C1~C3는 S2 및 S1 compiled VM도 각각 6/6 PASS다. C4 alone S1 compiled VM도 6/6 PASS다.

S1은 batch 1/prompt 1/cache 8, S2는 batch 1/prompt 7/cache 16이다. Model은 hidden/intermediate/vocab 256, head_dim 128, query head 2/KV head 1이며 한 decoder layer를 실행한다. Decode는 실제 prefill cache를 이어 사용하고, reference는 독립적인 기대 cache로 계산한다. Float 출력은 기존 `LOCAL_THRESHOLDS`(absolute/relative 0.002, violation fraction≤0.02, relative L2≤0.01, cosine≥0.999)를 사용하며 INT4 cache payload/zero는 정확히 비교한다.

| 경로 | S1 bytecode | S2 bytecode | S1 compiled VM |
| --- | --- | --- | --- |
| C1 TCU | 6/6 PASS | 6/6 PASS | 6/6 PASS |
| C2 hybrid | 6/6 PASS | 6/6 PASS | 6/6 PASS |
| C3 naive | 6/6 PASS | 6/6 PASS | 6/6 PASS |
| C4 alone | 6/6 PASS | 미실행 | 6/6 PASS |
| C4 fused | 6/6 PASS | 6/6 PASS | 6/6 PASS |

총 84개 boundary 실행이 PASS했다. `mini_hw_C1`, `mini_hw_C2`, `mini_hw_C3`, `mini_hw_C4`, `mini_hw_C1_S2` 등 각 결과 directory의 `results.json`/NPZ가 증거다. C4 fused는 최신 package를 재생성한 뒤 `mini_fused_final_hw_C4_S1`, `mini_fused_final_hw_C4_S2`, `mini_fused_final_hw_C4_compiled`에서 다시 통과했다. 각 결과의 package SHA256이 현재 package와 일치하는 것도 검사했다. 큰 32-layer E2E speedup/power workflow는 이번 검증 범위에 포함하지 않는다.

## Descriptor 검증 중 발견하고 수정한 문제

- M4/N256/K256의 branched + chained GEMM에서 sibling A 공유는 맞았지만, C→A 직접 재사용은 잘못됐다. 최초 fused 결과에서 1,024개 chain output이 모두 unfused와 달랐다(최대 차이 25.4141). 단순한 FP16 작은 값 예외로 처리하지 않았다.
- 원인은 `cur_m`으로 A microtile 행 stride를 계산하는 input layout과 `align8(cur_m)`으로 C microtile 행 stride를 계산하는 output layout을 descriptor가 같다고 판단한 것이다. `row_padding`을 구분하고 직접 재사용에 행 정렬 조건을 추가했다.
- Aligned M8과 tail M4 모두 유지하여 검사했다. Fused는 sibling A 재사용을 반드시 검증하고, M8은 C→A 재사용을 검증하며 M4는 잘못된 직접 재사용을 금지한다. 두 shape의 최종 HW 결과 모두 PASS이며 unfused output과 bitwise equality도 통과했다. `descriptor_hw_alone_final`, `descriptor_hw_fused_final`의 결과를 참조한다.
- Compound graph를 단일 GEMM용 Torch golden으로 비교하면 중간 반올림이 누적된다. 추가로 Torch는 W를 dequantize한 뒤 FP16으로 먼저 반올림한다. Standalone QCOL reference(`fpint_gemm_ffn_hw/test_vectors.h:67`)는 FP32에서 `(INT4-zero)*scale`을 계산한다. Descriptor probe의 각 GEMM은 후자의 계약으로 비교하고, 소비 GEMM에는 실제 predecessor output을 사용한다. 허용치는 기존 `0.003 + 0.003*abs(reference)` 그대로다.
- Torch FP16-weight reference 및 독립적인 end-to-end 오차도 NPZ/JSON에 남긴다. Alone M8에서 Torch per-operation reference와 2개가 허용치를 넘었지만 standalone QCOL reference에서는 0개였다. M4의 end-to-end Torch mismatch 7개도 삭제하거나 허용치를 넓혀 통과 처리하지 않는다.

## 실행 방법

[실행 README](../apps/vortex_llama3/README.md#config-driven-th16-c1c4-functionality-checks)에 repository/build/profile 지정, compile-only, Slurm HW, mini package 생성/실행, descriptor 비교, 재생성 조건을 기록했다.

## 요구사항별 완료 감사

| 계획 요구사항 | 현재 증거 |
| --- | --- |
| Config/manifest 기반 policy, C2 역할별 routing | `policy.py` resolver, candidate-independent routing/config drift tests, 실제 C2 동일 shape 3개 및 hybrid graph PASS |
| 실제 LMEM/TMEM 및 accumulator mode 전달 | Support/target/runtime metadata tests, image profile hash, C4 banks8×32768=262144 bytes; DMA channels4와 분리 |
| Square power-of-two A, threads=A, TCU-only 예외 | Target negative/positive tests; A=8/16/32/64 host 검증과 다른 A 12개 compile PASS; 16/32 whitelist 없음 |
| TCU geometry 및 launch | C++ WMMA oracle 8/16/32/64 비교, TH16 TCU 실제 HW exact/tail/256³ PASS |
| Packing/QBLK/transpose/tail/capacity 및 오류 거부 | 독립 packing index/size oracle, A±1 및 127/128/129 경계, SRAM/ACC/overflow 거부 tests, 각 MXU 12개 quantization 조합과 tail HW PASS |
| Candidate fixture, archive 및 checked package | 정확한 alias JSON, C4 archive materialization tests, 실제 다른 image/profile/module hash 거부 |
| TH16 kernel 기능 검증 | 기본 57개 PASS, 역할별 동일 shape 및 descriptor 추가 검증 PASS; shape/오차/nonfinite/첫 mismatch/NPZ 보존 |
| 작은 Llama graph 연결 | C1~C4 실제 prefill/decode/cache/head, bytecode 및 compiled VM, 84개 boundary PASS |
| C4 alone 이후 fused/descriptor reuse | M8 direct C→A 재사용 및 M4 detile/repack, 두 shape의 sibling A 재사용, alone/fused bitwise equality; fused graph 재생성/재검증 |
| 산출물 및 범위 보존 | [계획](vortex_config_driven_gemm_plan.md), 이 검증 기록, 실행 README; RTL 변경 및 새 합성 없음 |

Machine-readable 감사는 [completion_audit.json](../build/config_gemm_th16/completion_audit.json)의 50개 검사에 보존한다. [결과 root](../build/config_gemm_th16)에는 profile/image/config/module hash, 수치 결과와 NPZ, compile packages 및 로그가 있다. 결과의 원본 identity와 감사 시점의 TVM/Vortex revision을 보존했다. 소스 변경은 commit하지 않은 working tree 상태다.

현재 확인 범위는 TH16/MXU16 이미지 네 개와 작은 한-layer Llama graph다. 다른 A의 실제 FPGA 기능, 대형 32-layer inference, 성능 개선 및 power 측정은 검증했다고 주장하지 않는다. Standalone TCU의 strict threshold 실패와 Torch/standalone QCOL reference 차이는 위에 별도로 남겼다.
