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

#include <cstdint>
#include <limits>
#include <string>

#include "../../../target/build_common.h"
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

  auto compile = ffi::Function::GetGlobal("tvm_callback_vortex_compile");
  TVM_FFI_CHECK(compile.has_value(), RuntimeError)
      << "target.build.vortex requires tvm_callback_vortex_compile; import tvm.support.vortex";
  ffi::Bytes binary = (*compile)(ffi::String(code), target).cast<ffi::Bytes>();

  auto get_u32_attr = [&target](const char* name) {
    int64_t value = target->GetAttr<int64_t>(name).value();
    TVM_FFI_CHECK_GE(value, 0, ValueError)
        << "Vortex target attribute " << name << " must be non-negative";
    TVM_FFI_CHECK_LE(static_cast<uint64_t>(value), std::numeric_limits<uint32_t>::max(), ValueError)
        << "Vortex target attribute " << name << " does not fit uint32";
    return static_cast<uint32_t>(value);
  };

  auto create = ffi::Function::GetGlobal("ffi.Module.create.vortex");
  TVM_FFI_CHECK(create.has_value(), RuntimeError)
      << "Vortex runtime module is not loaded. Rebuild with USE_VORTEX set to the explicit "
         "Vortex repository path.";
  return (*create)(binary, ffi::String(code), ExtractFuncInfo(mod), uint32_t{1},
                   get_u32_attr("num_warps"), get_u32_attr("thread_warp_size"),
                   get_u32_attr("max_threads_per_block"), get_u32_attr("xlen"))
      .cast<ffi::Module>();
}

void RegisterVortexCodegen() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.vortex", BuildVortex);
}

}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() { tvm::codegen::RegisterVortexCodegen(); }
