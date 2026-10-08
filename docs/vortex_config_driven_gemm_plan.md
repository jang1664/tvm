# Config 기반 GEMM 선택과 power-of-two MXU 지원 계획

작성일: 2026-10-06
상태: **필수 구현 및 검증 완료** — 실제 측정 범위와 제한은 [검증 기록](vortex_config_driven_gemm_validation.md)을 참조한다.

2026-10-07 변경: 앞으로 C4는 `v4_fix_pad`(layout ABI 3)만 사용한다. 이전 C4의
검증 이력과 구분하며, 현재 이미지는 [GEMM/chain 검증](vortex_packed_c_hardware_validation.md)을
통과했다. 현재 C4의 mini-Llama도 alone/fused S1/S2 bytecode 및 S1 compiled VM의
36개 boundary 검증을 통과했다. 이는 1-layer 단계별 검증이며 32-layer E2E 검증은 아니다.

## 1. 목표와 범위

Vortex config에서 실제 accelerator capability와 geometry를 읽고, TVM의 GEMM backend, parameter packing, layout, launch geometry를 일관되게 결정한다. 먼저 지정된 TH16 FPGA 이미지 네 개를 검증하고, 같은 구현이 하드웨어 제약을 만족하는 다른 power-of-two A×A MXU에도 적용되도록 한다.

- GEMM의 역할과 config capability를 함께 사용한다. Naive MXU와 FP16 TCU가 모두 있으면 linear는 naive MXU, QKᵀ·PV는 FP16 TCU로 선택한다(C2).
- MXU만 있으면 linear와 QKᵀ·PV 모두 해당 MXU를 사용한다. `GEMM_NAIVE`/`GEMM_IMPROVE`로 구현을 구분한다(C3/C4).
- MXU가 없고 FP16 TCU가 있으면 linear와 QKᵀ·PV 모두 TCU를 사용한다(C1).
- MXU 크기는 `MXU_ROW == MXU_COL == A`, `A > 0`, `A & (A - 1) == 0`을 만족해야 한다. 16/32 whitelist는 만들지 않는다.
- 앞서 합의한 `NUM_THREADS == A` 조건은 MXU가 있는 경로에 적용한다. C1에는 MXU가 없으므로 이 비교를 적용하지 않는다.
- TCU tile geometry는 MXU geometry와 별개이며, 실제 thread 수와 기존 `wmma_config_t`/`wmma_context`에 맞춘다.
- 이번 구현에서 RTL 기능 변경이나 새로운 FPGA 합성은 하지 않는다. 기존 이미지로 검증한다.
- 먼저 kernel 기능 검증을 완료한 다음 Llama graph 통합 검증으로 확장한다. 전체 latency/power workflow 재실행은 이 계획의 필수 단계가 아니다.

여기서 “다른 A 지원”은 모든 양의 power-of-two를 무조건 실행한다는 뜻이 아니다. DMA tile, packed INT4, accumulator, SRAM, thread geometry 등 실제 구현 조건을 만족하는 A를 허용하고, 불가능한 조합은 컴파일 전에 구체적인 이유와 함께 거부한다.

## 2. 대상 저장소와 FPGA 이미지

| 구분 | 경로 |
| --- | --- |
| TVM compiler/runtime/application | `/home/jaeyongjang/project.local/tvm` |
| Vortex device helper/config/runtime | `/home/jaeyongjang/project.local/vortex_fpint-feat-gemv` |
| Alias source | Vortex의 `ci/fpga_bin_alias_map.yaml` |
| HW compile profile | 선택한 xclbin의 sibling `manifest.json`에 저장된 `params.CONFIGS` |

| 테스트 ID | 정확한 FPGA alias | Capability | 자동 선택 결과 |
| --- | --- | --- | --- |
| C1 | `tcu_th16_c1_v3_axi_fix` | TH16, FP16 TCU, MXU 없음 | FP16 TCU |
| C2 | `naive_th16_tcol16_m16_L16_bigmem_all_bram_acc_tcu_base_pnr_v3_axi_fix` | TH16, FP16 TCU + naive MXU16 | Linear: naive MXU / QKᵀ·PV: FP16 TCU |
| C3 | `naive_th16_tcol16_m16_L16_bigmem_all_bram_acc_base_pnr_v3_axi_fix` | TH16, naive MXU16 | FP–INT naive MXU |
| C4 | `improve_th16_tcol16_m16_t8_bigmem_all_bram_spread_v4_fix_pad` | TH16, improve MXU16, layout ABI 3 | FP–INT improve MXU |

**C2의 자동 선택은 기존 `c2_linear_w4_naive_attention_fp16_tcu` 정책을 따른다.** Linear는 naive MXU, attention의 QKᵀ와 PV는 FP16 TCU로 실행한다. C1~C4는 테스트 행의 이름이며, 실제 policy는 config의 accelerator 조합에서 결정한다. 개별 GEMM의 역할은 기존 graph의 linear/QKᵀ/PV 의미 정보를 사용하고, shape만으로 추정하지 않는다.

C2의 연산별 경계는 다음과 같다. Attention에 속한 Q/K/V/O projection도 linear이므로 naive MXU를 사용한다.

C2는 config에서 두 accelerator의 사용 가능 여부를 확인한 뒤, 각 GEMM의 operation role에 따라 backend를 선택하는 hybrid 구성이다. MXU가 있다는 이유로 C2의 모든 GEMM을 naive MXU에 배치하지 않는다. Prefill과 decode 모두 아래 분류를 동일하게 적용한다.

| C2 연산 | 선택 backend | Operand 표현 |
| --- | --- | --- |
| Q/K/V/O projection, FFN linear, linear로 export된 LM head | FP–INT naive MXU | 기존 W4A16 quantization contract |
| QKᵀ attention GEMM | FP16 TCU | FP16 Q와 K operand |
| PV attention GEMM | FP16 TCU | FP16 P와 V operand |

HW에서는 현재 `.sh`만 읽어 이미지를 추정하지 않는다. Alias로 config와 xclbin을 찾은 다음, 실제 이미지 manifest를 기준으로 compile profile을 생성한다. 현재 config와 build-time manifest의 차이는 표시하고, 테스트에 필요한 capability/geometry가 충돌하면 실행 전에 실패시킨다. Simulation에서는 source한 config와 configured build의 설정으로 profile을 구성한다.

## 3. 구현 시작 시점에 확인한 사실

| 항목 | 현재 상태 | 수정 필요성 |
| --- | --- | --- |
| MXU geometry 전달 | Manifest → target → device compiler의 `MXU_ROW/COL` 전달은 존재 | 기존 경로 확장, 별도 profile parser를 만들지 않음 |
| TCU lowering | `thread_binding(32)`, M/N=16, K=32 기준 padding | TH16 geometry로 일반화 필요 |
| TCU device helper | 특정 16×16×32 tile이 아니면 `-1` 반환 | 기존 WMMA geometry에 맞춰 검사/실행 필요 |
| C4 TMEM 용량 | `TMEM_BANK_SIZE × NUM_DMA_CHANNELS`로 계산 | 물리 bank 수와 DMA channel 수 분리 필요 |
| C1~C3 LMEM 용량 | 명시적인 `LMEM_SIZE`를 무시하고 `1 << LMEM_LOG_SIZE` 사용 | 실제 용량을 target과 compiler에 전달해야 함 |
| QBLK16 | Improve planner와 device ABI v2 helper 모두 거부 | MXU/quantization 조건에 따른 검증으로 변경 필요 |
| Naive attention | quant-direction-N 경로에 `tile_n=32` | 과거 workaround의 목적을 보존하며 geometry 기반 분할 검증 필요 |
| Llama compile matrix | `vortex_base` 기본 경로, C1/C2/C3 전용 alias-policy 연결 | 명시적 candidate map과 C4 통합 필요 |
| Parameter archive | Llama의 backend materializer에 C1/C2/C3 정책별 분기 | config 기반 policy와 C4 packing 경로 연결 필요 |

실제 manifest로 재현한 값:

- C1: 실제 LMEM 1,572,864 bytes, TVM target 2,097,152 bytes.
- C2/C3: 실제 LMEM 1,310,720 bytes, TVM target 2,097,152 bytes.
- C4: 실제 TMEM `8 × 32768 = 262144` bytes, TVM planner는 `4 × 32768 = 131072` bytes로 해석.
- C4의 `M=N=K=256`, QBLK32: `TMEM scratch requires 151552 bytes, limit is 131072`로 host layout 생성 실패.
- C1의 TH16 target으로 TCU lowering을 수행해도 32-thread launch가 생성됨.
- 기존 관련 테스트: 84 PASS, HW 실행 테스트 1 SKIP. 이 결과는 새로운 TH16 이미지의 기능 검증 통과를 의미하지 않음.

주요 코드:

- [Profile normalization 및 compile flags](../python/tvm/support/vortex.py)
- [Target attribute 검증](../src/backend/vortex/codegen/target_kind.cc)
- [Backend policy](../python/tvm/relax/backend/vortex/policy.py)
- [Relax lowering](../python/tvm/relax/backend/vortex/pipeline.py)
- [Improve layout planner](../python/tvm/relax/backend/vortex/layout.py)
- [Llama parameter archive](../python/tvm/relax/backend/vortex/llama_parameter_archive.py)
- [기존 C4 parameter archive](../python/tvm/relax/backend/vortex/parameter_archive.py)
- [Vortex TCU helper](../../vortex_fpint-feat-gemv/kernel/include/vx_tvm_tcu.h)
- [Vortex GEMM helper](../../vortex_fpint-feat-gemv/kernel/include/vx_tvm_gemm.h)

## 4. Backend 선택 규칙

### 4.1 Capability 판정

기존 `_normalize_accelerator_profile`과 `validate_vortex_backend_policy`를 확장한다.

```text
먼저 ENABLE_GEMM_ACCEL 및 GEMM_NAIVE/GEMM_IMPROVE의 일관성을 검증한다.

MXU 없음 + FP16 TCU 있음:
    linear=fp16_tcu, QKᵀ=fp16_tcu, PV=fp16_tcu                 # C1
naive MXU + FP16 TCU 있음:
    linear=fpint_naive, QKᵀ=fp16_tcu, PV=fp16_tcu             # C2
naive MXU + FP16 TCU 없음:
    linear=fpint_naive, QKᵀ=fpint_naive, PV=fpint_naive        # C3
improve MXU + FP16 TCU 없음:
    linear=fpint_improve, QKᵀ=fpint_improve, PV=fpint_improve  # C4
그 외 조합 또는 식별 불가능한 MXU backend:
    -> auto policy를 결정할 수 없다는 compile-time error
```

TCU 판정에는 `EXT_TCU_ENABLE`뿐 아니라 `DISABLE_TCU_FP`, `DISABLE_FP16`도 반영한다. 이름이 `tcu`인 alias인지, `MXU_ROW` 매크로가 기본값으로 존재하는지는 capability 판정 근거로 사용하지 않는다.

현재 대상에 없는 improve MXU + FP16 TCU 조합의 자동 정책은 임의로 정의하지 않는다. 지원할 때 역할별 정책과 검증을 함께 추가한다. C2에서는 graph의 operation role이 누락되거나 모호하면 shape로 backend를 추측하지 않고 진단한다.

이 실패 규칙은 새로운 `auto` accelerated GEMM 경로에 적용한다. 기존 일반 SIMT TVM target의 모든 matmul을 일괄 금지하는 변경은 하지 않는다.

### 4.2 Policy와 모델 표현

- `backend_policy="auto"`를 기존 pipeline 진입점에 추가하고, candidate matrix에서는 이를 기본으로 사용한다.
- Alias 해석 직후, 모델 export와 parameter materialization **이전**에 policy를 확정한다. 동일한 resolved policy를 graph 생성, packing, lowering, package metadata에서 사용한다.
- Candidate ID와 독립적인 capability resolver가 기존 C1/C2/C3/C4 policy를 선택하도록 한다. C2는 기존 hybrid policy를 재사용하고, 기존 C3의 “TCU 없음” 검증도 유지한다. 동일한 정책을 다른 이름으로 중복 구현하지 않는다.
- 기존 명시적 C1/C2/C3/C4 정책은 유지한다. 기존 package의 의미가 자동으로 바뀌지 않게 한다.
- 동일한 logical quantized parameter archive를 기준으로 역할별 표현을 생성한다. TCU는 FP16 dequantized 표현, naive는 row-major W4, improve는 profile별 prepacked W4를 사용한다. C2의 linear weight는 row-major W4를 유지한다.
- Attention의 K/V quantization 또는 dequantization도 선택한 policy와 기존 quantization contract에 맞춰 생성한다. C2는 quantized KV를 필요한 FP16 operand로 변환해 QKᵀ·PV TCU에 공급하며, attention까지 naive W4 경로로 바꾸지 않는다.
- 일반적인 FP16×FP16 GEMM을 MXU 존재만으로 임의 INT4 양자화하지 않는다. 자동 선택 대상인 Llama/W4 graph는 기존 quantization scheme, scale, zero-point가 정의된 상태여야 한다. 표현이 불충분하면 필요한 정보를 명시하고 실패한다.
- Package에 requested policy(`auto`)와 resolved policy 및 역할별 backend를 함께 기록한다. C2에서는 linear의 naive 호출과 QKᵀ·PV의 TCU 호출이 모두 존재하고, 각 역할이 뒤바뀌지 않았는지 생성 kernel inventory로 검사한다.

## 5. Profile과 geometry 정리

### 5.1 정확한 memory profile

기존 target/profile/compile-config에 필요한 필드를 끝까지 전달한다.

| 필드 | 의미와 계산 |
| --- | --- |
| LMEM bytes | 명시적 `LMEM_SIZE` 우선, 없을 때만 `1 << LMEM_LOG_SIZE` |
| TMEM bank count | `NUM_TMEM_BANKS`; DMA channel 수로 대체하지 않음 |
| TMEM bank bytes | `TMEM_BANK_SIZE` |
| TMEM total bytes | bank count × bank bytes |
| DMA channels | DMA 병렬 경로 수; 물리 memory bank 수와 구분 |
| Improve row alignment | Layout ABI v2의 8-row DRAM stripe; DMA channel 수와 구분 |
| MXU row/col/col tile | 실제 manifest/config 기반 |
| Accumulator depth | `GEMM_ACC_MEM_DEPTH` |
| Threads/warps | 실제 config 기반 |

`support/vortex.py`, `target_kind.cc`, `build_vortex.cc`, runtime module의 metadata 저장/복원 및 device compile flags를 함께 갱신한다. 숫자 표현은 기존 macro parser를 재사용하고, 필요한 표현을 해석할 수 없으면 기본값으로 조용히 바꾸지 않는다.

Kernel compile에 필요한 LMEM/TMEM geometry와 naive ACC mode 등 누락된 매크로를 확인하고 명시적으로 전달한다. Build header의 이전 config 기본값에 의존하지 않는다. 단, raw CONFIGS 문자열 전체를 무검증으로 복사하는 별도 compile 경로는 만들지 않는다.

Profile fingerprint와 geometry로 parameter archive/package의 재사용을 검사한다. 새 필드 때문에 metadata format이 달라지면 version을 올리고, 이전 package는 명시적으로 재생성을 요구한다. Device job ABI가 실제로 바뀌지 않는 경우 GEMM ABI 번호까지 불필요하게 변경하지 않는다.

### 5.2 Power-of-two A 검증

MXU가 활성화된 경우 다음을 TVM target/planner와 device C++ compile-time 검사에서 일치시킨다.

1. `MXU_ROW == MXU_COL == A`이며 A가 양의 power-of-two.
2. `NUM_THREADS == A`. TCU-only target에는 적용하지 않음.
3. `MXU_COL_TILE` 및 DMA K/N tile과 A 사이의 실제 나눗셈 조건.
4. Packed INT4 저장에 필요한 최소 geometry와 byte 정렬 조건.
5. Double-buffer scratch와 accumulator 요구량이 실제 용량 이내인지.

DMA tile은 현재 helper의 128×128×128 계약을 우선 유지한다. 그러므로 이 계약과 맞지 않는 A는 명확히 거부한다. 더 큰 A를 위해 DMA tile까지 확장하는 작업은 별도의 ABI/layout 변경으로 구분하며, 이번 결과를 무제한 A 지원으로 표시하지 않는다.

### 5.3 TCU geometry

- `_PaddedFP16TCULowerer`와 `_make_fp16_tcu_matmul`에 resolved TCU geometry를 전달한다.
- M/N/K padding, block grid, thread extent를 같은 geometry로 계산한다.
- Vortex의 `wmma_config_t`/`wmma_context`를 기준으로 host-side 계산을 검증한다. Python에 TH16/TH32 숫자 테이블만 추가하는 방식은 피한다.
- `vx_tvm_tcu_fp16_tile`의 특정 tile 형상 제한을 제거하고, 실제 context와 launch의 일치 여부를 검증한다.
- 기존 `T.evaluate(call_extern(...))`가 helper 오류 반환을 버리는 경로를 점검한다. 정적으로 알 수 있는 오류는 compile-time에 거부하고, runtime 오류가 필요하다면 기존 launch 상태 전달 경로를 사용해 host에서 실패로 확인되게 한다.

## 6. Layout과 device helper 수정

### 6.1 Improve MXU

- `ImproveProfile`에 physical TMEM bank count를 추가하고 scratch 용량 검사를 수정한다.
- 기존 A/W/QPARAM/C layout 계산을 재사용한다. 전체 packing 코드를 새로 작성하지 않는다.
- QBLK16을 단순 whitelist 추가로 처리하지 않는다. Q direction, MXU microtile, DMA tile, qparam 저장 형식과의 조건을 planner/device helper에서 동일하게 검증한다.
- 최소 검증 대상 QBLK는 16/32/128이다. QBLK가 A보다 작을 때 가능한 Q direction 조합은 기존 RTL 및 통과한 regression contract를 확인한 후 허용한다.
- QPARAM slot alignment와 DMA row alignment를 physical bank count 변경과 혼동하지 않는다. 용량 계산 수정 때문에 기존 layout 순서가 불필요하게 바뀌지 않게 한다.
- Tail padding, neutral qparam, output unpadding 및 fused producer-consumer descriptor 호환성을 검사한다.

### 6.2 Naive MXU

- quant-direction-N의 `tile_n=32`가 도입된 descriptor/workaround 목적을 먼저 확인한다.
- Microtile 기반 제약이면 profile에서 가져오고, algorithmic chunk라면 geometry에서 안전하게 유도한다. 숫자 32를 일괄 A로 치환하지 않는다.
- TVM helper의 LMEM scratch 배치, ACC mode, MMIO register 사용을 현재 `fpint_gemm_ffn_hw_naive`의 검증된 경로와 대조한다.
- Multi-tile, quant-direction, transpose에서 weight/qparam offset과 결과 범위를 검증한다.
- Naive 전용 변경은 필요한 경우 `GEMM_NAIVE` guard 안에 둔다. Improve 경로의 RTL·동작을 바꾸지 않는다.

## 7. 테스트 도구와 실행 구조

기존 `apps/vortex_llama3/compile_backend_matrix.py`와 backend probe 도구를 확장한다. Candidate마다 별도의 새 runner를 만들지 않는다.

- `--candidate-map`으로 테스트 ID → 정확한 FPGA alias를 전달한다. 위 네 alias를 담은 작은 fixture를 추가한다.
- `--vortex-home`은 현재 Vortex 저장소로 지정하고, 기본 alias-map 경로도 선택한 repository 기준으로 해석한다.
- `ALIAS_POLICIES`의 C1/C2/C3 제한을 candidate-independent policy resolution으로 대체한다.
- 기존 `parameter_archive.py`의 improve packing을 Llama archive에 연결하여 C4를 지원한다.
- `_assert_inventory`의 improve 무조건 거부를 resolved backend별 검사로 바꾼다.
- Package loader, reference 생성, numerical validation도 같은 resolved policy와 profile identity를 사용한다.
- 실행 결과에는 alias, config/manifest/xclbin hash, TVM/Vortex revision, GEMM 역할별 backend, A, threads, tile, shape, quantization 정보, PASS/FAIL, 오차 통계를 기록한다.

HW 후보는 우선 한 FPGA에서 순차 실행한다. Candidate별 kernel runtime/build directory를 분리하거나 동일 directory의 설정과 runtime library를 확실하게 재생성하여 이전 config artifact를 재사용하지 않는다. 병렬 simulation이 필요하면 반드시 build directory를 분리한다.

Vortex standalone 기준 검증은 configured build에서 `ci/run_black.sh hw --fpga-bin <alias>`를 사용한다. TVM 생성 module은 기존 TVM XRT runner로 실제 실행하며, standalone kernel PASS를 TVM codegen PASS로 대신하지 않는다. Simulation이 필요하면 `xrt-vcs-sim`을 사용한다.

## 8. 검증 순서와 통과 기준

### 단계 1 — Host capability/profile/layout

- 실제 네 manifest의 capability, LMEM, TMEM, A, DMA channels 검사.
- MXU-only, TCU-only, naive MXU+FP16 TCU, 둘 다 없음, 모순된 backend macro의 routing 검사. C2는 linear→naive, QKᵀ/PV→TCU를 각각 확인한다.
- 합성 profile A=8/16/32/64의 geometry 및 layout 검사. 각 profile의 memory/ACC 크기는 테스트 의도에 맞게 명시한다.
- A=0/24, 비정사각 MXU, thread 불일치, tile 비정렬, SRAM/ACC 부족은 명확한 실패를 확인한다.
- C1에는 의미 없는 MXU size 검사로 오류를 발생시키지 않는다.
- QBLK16/32/128, QDIR0/1, transpose, tail의 독립적인 packing/index 기준과 비교한다.
- Old profile/package의 거부 또는 명시적 migration, 다른 image의 package 재사용 거부를 검증한다.
- 기존 관련 unit tests를 재실행하고, 숫자 상수를 그대로 따라 하는 테스트 대신 크기·offset·복원 값·routing을 검사한다.

### 단계 2 — Device compile만 수행

- C1~C4 각각의 config로 TVM device 코드를 compile한다.
- TCU geometry를 C++ 기존 helper와 비교하고, generated TIR/source의 thread 및 grid를 확인한다.
- Compiler flags가 실제 이미지의 LMEM/TMEM/MXU/ACC mode와 일치하는지 확인한다.
- C2 `auto`에서는 linear에 FP–INT naive 호출, QKᵀ와 PV 각각에 FP16 TCU 호출이 생성되는지 확인한다. 같은 shape라도 operation role에 따라 올바르게 분기하는 case를 포함한다.
- C2의 Q/K/V/O projection이 attention 연산이라는 이유로 TCU로 분류되지 않는지 확인한다. Projection은 linear, QKᵀ·PV만 attention GEMM으로 구분한다.
- 가능한 합성 A profile은 compile까지 확인하되, 대응 bitstream이 없으면 HW 검증됐다고 표시하지 않는다.

### 단계 3 — TH16 실 FPGA kernel 기능 검증

| 경로 | 기본 대표 shape (M,N,K) | 추가 관점 |
| --- | --- | --- |
| C1 TCU | (16,16,32), (17,33,65), (256,256,256) | Exact/tail, geometry별 padding과 output slicing |
| C2 linear / C3 naive MXU | (1,256,256), (4,256,256), (256,256,256) | 작은 M, multi-DMA-tile, QDIR/transpose |
| C2 QKᵀ·PV TCU | QKᵀ: (1,256,128), (4,256,128), (256,256,128); PV: (1,128,256), (4,128,256), (256,128,256) | 두 operation role, FP16 KV operand 준비, decode/prefill |
| C4 improve MXU | (1,256,256), (4,256,256), (256,256,256) | QBLK16/32/128, prepacked layout 및 multi-tile |

- Boundary shape는 A−1/A/A+1, DMA tile 127/128/129를 host 테스트에서 넓게 검증하고, HW에서는 각 경로의 대표 tail 하나를 추가한다. 지원하지 않는 tail은 조용히 잘못 실행하지 않고 사전 거부를 확인한다.
- 동일한 logical input/weight/scale/zero-point와 고정 random seed를 사용한다. Candidate별로 정당한 packing/dequantization 경로를 거친 reference와 비교한다.
- Mismatch 수/전체 수, 최대 절대·상대 오차, NaN/Inf, 처음 틀린 위치를 기록한다. FP16 reduction/rounding의 허용 오차는 실행 전에 정하고, 실패를 보고 허용치를 넓히지 않는다.
- 기존에 논의된 작은 FP16 값 예외가 필요하면 별도 항목으로 기록한다. Layout/offset 오류를 conversion 예외로 숨기지 않는다.
- 모든 candidate의 기본 functionality가 끝나기 전 성능 최적화나 전체 pipeline 실행으로 넘어가지 않는다.

### 단계 4 — Llama graph 연결

- C1~C4의 작은 prefill/decode graph를 compile/package/load한다.
- Linear, QKᵀ, PV가 resolved backend로 가는지, GQA broadcast 및 KV quantization 표현이 맞는지 확인한다.
- C2는 linear MXU와 QKᵀ·PV TCU가 같은 graph 및 FPGA image에서 함께 동작하는 것을 확인한다. 개별 backend 테스트만 통과한 상태를 hybrid graph 통과로 취급하지 않는다.
- C4는 우선 unfused 경로로 correctness를 확인한 뒤 descriptor 재사용/fused 경로를 확인한다.
- 전체 32-layer 대형 workload보다 작은 graph/probe를 먼저 실행한다. 별도 reference 생성 환경이 필요하면 기존 도구와 artifact를 활용한다.
- 기능 통과 후 cycle을 수집할 수 있으나 이번 완료 기준에 E2E speedup 목표나 전체 power 재측정은 두지 않는다.

## 9. 작업 단위와 의존성

| 순서 | 작업 | 주요 파일/영역 | 완료 조건 |
| --- | --- | --- | --- |
| 1 | Profile 정확성 및 metadata 전달 | `support/vortex.py`, `target_kind.cc`, `build_vortex.cc`, runtime module | 실제 네 manifest의 용량/geometry 일치 |
| 2 | Config 기반 backend resolution | `policy.py`, pipeline 진입점 | capability routing 및 기존 explicit policy 테스트 통과 |
| 3 | TCU geometry 일반화 | `pipeline.py`, Vortex `vx_tvm_tcu.h` | TH16/TH32 lowering·compile 일치 |
| 4 | MXU layout/helper 일반화 | `layout.py`, naive lowering, `vx_tvm_gemm.h` | A/QBLK/packing/capacity tests 통과 |
| 5 | Candidate matrix와 archive 연결 | Llama apps, 두 parameter archive 모듈 | C1~C4 package 생성 및 inventory 일치 |
| 6 | TH16 HW 검증 | 기존 probe/runner 및 작은 추가 case | 네 후보 functionality PASS, 재현 가능한 결과 요약 |

Config parsing, layout planning, policy validation, parameter packing은 기존 abstraction에 넣는다. 불필요한 RTL 수정, broad refactor, 전체 repository lint/test는 수행하지 않는다. 각 단계 후 변경 diff와 해당 단계의 검사 결과를 확인한다.

## 10. 최종 산출물

1. Config/manifest 기반 GEMM 자동 선택과 geometry/memory 검증 구현.
2. TH16 TCU 및 MXU16 대응, 가능한 power-of-two A를 표현하는 profile/layout 구현.
3. 네 candidate alias를 지정하는 테스트 fixture와 기존 runner 확장.
4. Host/device compile/HW 검증 결과 Markdown: candidate별 사용 backend, shape, 오차, PASS/FAIL, 측정 범위.
5. 실행 README: repository/build/profile 지정, 작은 기능 검증 명령, package 재생성 조건.

검증 완료 전에는 “모든 A에서 동작”이라고 결론 내리지 않는다. TH16 실제 HW 결과와 다른 A의 host/compile 결과를 분리해 보고한다.

## 11. Review에서 확인할 정책

- **Auto의 C2는 linear=naive MXU, QKᵀ·PV=FP16 TCU**이며 기존 hybrid policy를 그대로 선택한다.
- MXU가 활성화된 경우 `NUM_THREADS == A`; TCU-only에는 이 조건을 적용하지 않는다.
- 최초 구현은 현재 DMA tile 128 계약 및 기존 RTL이 허용하는 power-of-two A를 대상으로 한다.
- 이번 단계의 필수 완료점은 네 후보의 기능 검증과 작은 Llama graph 연결이며, 전체 latency/power pipeline 재실행은 별도다.

구현과 검증의 현재 증거는 [검증 기록](vortex_config_driven_gemm_validation.md)에 정리한다. Kernel 기능 검증과 graph 통합 검증을 별도로 완료해야 한다.
