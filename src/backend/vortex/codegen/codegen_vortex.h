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

#include <string>
#include <vector>

#include "../../../target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenVortex final : public CodeGenC {
 public:
  explicit CodeGenVortex(Target target);

  void AddKernel(const GlobalVar& gvar, const PrimFunc& func, uint32_t kernel_id,
                 const std::string& global_symbol);
  void FinishDispatcher();
  void InitFuncState(const PrimFunc& func) final;
  void PrintFuncPrefix(std::ostream& os) final;  // NOLINT(*)
  using CodeGenC::PrintType;
  void PrintType(const Type& type, std::ostream& os) final;  // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;
  void VisitStmt_(const AttrStmtNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AllocBufferNode* op) final;
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;  // NOLINT(*)

 private:
  void ValidateThreadExtent(const IterVar& iv, const PrimExpr& extent);
  std::string PrintTypeString(const Type& type);
  void EmitLaunchWrapper(const PrimFunc& func, uint32_t kernel_id);

  struct KernelDispatchInfo {
    size_t num_args;
  };

  Target target_;
  std::vector<KernelDispatchInfo> kernels_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_BACKEND_VORTEX_CODEGEN_CODEGEN_VORTEX_H_
