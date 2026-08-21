/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_vortex.h
 * \brief Generate Vortex native-kernel C++ from TIRx.
 */
#ifndef TVM_BACKEND_VORTEX_CODEGEN_CODEGEN_VORTEX_H_
#define TVM_BACKEND_VORTEX_CODEGEN_CODEGEN_VORTEX_H_

#include <tvm/target/target.h>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../target/source/codegen_c.h"

namespace tvm {
namespace codegen {

struct VortexKernelResourceMetadata {
  uint32_t launch_rank;
  uint64_t static_shared_bytes;
  uint64_t compile_time_resident_groups;
  uint64_t private_local_bytes_per_thread;
  std::array<uint64_t, 3> thread_block_dimensions;
  bool uses_shared_barrier;
};

class CodeGenVortex final : public CodeGenC {
 public:
  explicit CodeGenVortex(Target target);

  void AddKernel(const GlobalVar& gvar, const PrimFunc& func, uint32_t kernel_id,
                 const std::string& global_symbol);
  void FinishDispatcher();
  void InitFuncState(const PrimFunc& func) final;
  void PreFunctionBody(const PrimFunc& func) final;
  void PrintFuncPrefix(std::ostream& os) final;  // NOLINT(*)
  using CodeGenC::PrintType;
  void PrintType(const Type& type, std::ostream& os) final;                  // NOLINT(*)
  void PrintStorageSync(const CallNode* op) final;                           // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;
  void VisitStmt_(const AttrStmtNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AllocBufferNode* op) final;
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;  // NOLINT(*)

  const std::vector<VortexKernelResourceMetadata>& kernel_resources() const {
    return kernel_resources_;
  }

 private:
  void PlanSharedMemory(const PrimFunc& func);
  void ValidateBindingExtent(const IterVar& iv, const PrimExpr& extent);
  std::string PrintTypeString(const Type& type);
  void EmitLaunchWrapper(const PrimFunc& func, uint32_t kernel_id);

  struct KernelDispatchInfo {
    size_t num_args;
  };

  Target target_;
  std::array<bool, 3> thread_axis_bound_{};
  std::array<bool, 3> block_axis_bound_{};
  std::array<uint64_t, 3> thread_extents_{{1, 1, 1}};
  uint32_t launch_rank_{1};
  uint64_t local_allocation_bytes_{0};
  uint64_t static_shared_bytes_{0};
  std::string shared_arena_base_var_;
  std::unordered_map<const VarNode*, uint64_t> shared_allocation_offsets_;
  std::vector<KernelDispatchInfo> kernels_;
  std::vector<VortexKernelResourceMetadata> kernel_resources_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_BACKEND_VORTEX_CODEGEN_CODEGEN_VORTEX_H_
