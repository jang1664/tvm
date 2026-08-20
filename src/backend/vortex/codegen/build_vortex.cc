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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "../../../target/build_common.h"
#include "codegen_vortex.h"

namespace tvm {
namespace codegen {

ffi::Module BuildVortex(IRModule mod, Target target) {
  struct KernelDefinition {
    std::string global_symbol;
    GlobalVar gvar;
    PrimFunc func;
  };
  std::vector<KernelDefinition> kernels;
  kernels.reserve(mod->functions.size());
  std::set<std::string> global_symbols;
  for (const auto& [gvar, base_func] : mod->functions) {
    TVM_FFI_CHECK(base_func->IsInstance<PrimFuncNode>(), TypeError)
        << "CodeGenVortex: only PrimFunc is supported";
    PrimFunc func = base_func.as_or_throw<PrimFunc>();
    auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value() && !symbol.value().empty(), ValueError)
        << "CodeGenVortex: every PrimFunc must have a non-empty global symbol";
    std::string global_symbol = symbol.value();
    TVM_FFI_CHECK(global_symbols.insert(global_symbol).second, ValueError)
        << "CodeGenVortex: duplicate global symbol " << global_symbol;
    kernels.push_back({std::move(global_symbol), gvar, std::move(func)});
  }
  TVM_FFI_CHECK(!kernels.empty(), ValueError) << "CodeGenVortex: at least one PrimFunc is required";
  std::sort(kernels.begin(), kernels.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.global_symbol < rhs.global_symbol; });

  CodeGenVortex cg(target);
  cg.Init(/*output_ssa=*/false);
  ffi::Map<ffi::String, int64_t> kernel_ids;
  for (size_t index = 0; index < kernels.size(); ++index) {
    TVM_FFI_CHECK_LE(index, std::numeric_limits<uint32_t>::max(), ValueError)
        << "CodeGenVortex: too many kernels for the launch ABI";
    uint32_t kernel_id = static_cast<uint32_t>(index);
    const KernelDefinition& kernel = kernels[index];
    cg.AddKernel(kernel.gvar, kernel.func, kernel_id, kernel.global_symbol);
    kernel_ids.Set(ffi::String(kernel.global_symbol), static_cast<int64_t>(kernel_id));
  }
  cg.FinishDispatcher();
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
  return (*create)(binary, ffi::String(code), ExtractFuncInfo(mod), kernel_ids, uint32_t{1},
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
