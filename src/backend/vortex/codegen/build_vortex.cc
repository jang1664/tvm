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
 * \file build_vortex.cc
 * \brief Vortex target build entry point.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>

#include <string>

#include "../../../target/source/codegen_source_base.h"
#include "codegen_vortex.h"

namespace tvm {
namespace codegen {

ffi::Module BuildVortex(IRModule mod, Target target) {
  TVM_FFI_CHECK_EQ(mod->functions.size(), 1, ValueError)
      << "CodeGenVortex: the single-kernel MVP requires exactly one PrimFunc, but got "
      << mod->functions.size();

  auto [gvar, base_func] = *mod->functions.begin();
  TVM_FFI_CHECK(base_func->IsInstance<PrimFuncNode>(), TypeError)
      << "CodeGenVortex: only PrimFunc is supported";
  PrimFunc func = base_func.as_or_throw<PrimFunc>();

  CodeGenVortex cg(target);
  cg.Init(/*output_ssa=*/false);
  cg.AddKernel(gvar, func);
  std::string code = cg.Finish();

  // Phase 2 exposes generated source through a CSourceModule.  The optional
  // callback is intentionally invoked now so tests and future compiler support
  // share the stable tvm_callback_vortex_compile contract.  Phase 3 will retain
  // its returned binary in a Vortex runtime module.
  if (auto compile = ffi::Function::GetGlobal("tvm_callback_vortex_compile")) {
    (*compile)(ffi::String(code), target);
  }

  // Do not advertise a runnable TVM function from this inspection-only seam.
  // The emitted translation unit exports the Vortex-native main entry point;
  // Phase 3 will replace this with a Vortex module that owns TVM function metadata.
  return CSourceModuleCreate(ffi::String(code), ffi::String("cpp"), {});
}

void RegisterVortexCodegen() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.vortex", BuildVortex);
}

}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() { tvm::codegen::RegisterVortexCodegen(); }
