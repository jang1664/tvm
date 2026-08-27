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

#include "../../../runtime/metadata.h"
#include "../../../target/build_common.h"
#include "../vortex_common.h"
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
    for (const Var& param : func->params) {
      if (const auto* buffer_type = param->ty.as<BufferTypeNode>()) {
        TVM_FFI_CHECK_NE(buffer_type->storage_scope, "shared.dyn", ValueError)
            << "CodeGenVortex: dynamic shared memory is not supported";
      } else if (const auto* pointer_type = param->ty.as<PointerTypeNode>()) {
        TVM_FFI_CHECK_NE(pointer_type->storage_scope, "shared.dyn", ValueError)
            << "CodeGenVortex: dynamic shared memory is not supported";
      }
    }
    if (auto launch_params =
            func->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams)) {
      for (const ffi::String& tag : launch_params.value()) {
        TVM_FFI_CHECK_NE(tag, runtime::launch_param::kUseDynamicSharedMemoryTag, ValueError)
            << "CodeGenVortex: dynamic shared memory is not supported";
      }
    }
    auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value() && !symbol.value().empty(), ValueError)
        << "CodeGenVortex: every PrimFunc must have a non-empty global symbol";
    std::string global_symbol = symbol.value();
    TVM_FFI_CHECK_NE(global_symbol, runtime::vortex::kKernelResourceMetadataFunction, ValueError)
        << "CodeGenVortex: global symbol " << global_symbol
        << " is reserved for Vortex runtime metadata inspection";
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

  const std::vector<VortexKernelResourceMetadata>& resources = cg.kernel_resources();
  TVM_FFI_ICHECK_EQ(resources.size(), kernels.size());
  ffi::Map<ffi::String, ffi::Array<int64_t>> kernel_resources;
  for (size_t index = 0; index < kernels.size(); ++index) {
    const VortexKernelResourceMetadata& resource = resources[index];
    auto checked_i64 = [&kernels, index](uint64_t value, const char* field) {
      TVM_FFI_CHECK_LE(value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                       ValueError)
          << "CodeGenVortex: " << field << " for kernel " << kernels[index].global_symbol
          << " does not fit int64 module metadata";
      return static_cast<int64_t>(value);
    };
    kernel_resources.Set(
        ffi::String(kernels[index].global_symbol),
        ffi::Array<int64_t>{
            static_cast<int64_t>(resource.launch_rank),
            checked_i64(resource.static_shared_bytes, "static_shared_bytes"),
            checked_i64(resource.compile_time_resident_groups, "compile_time_resident_groups"),
            checked_i64(resource.private_local_bytes_per_thread, "private_local_bytes_per_thread"),
            checked_i64(resource.thread_block_dimensions[0], "thread_block_dim_x"),
            checked_i64(resource.thread_block_dimensions[1], "thread_block_dim_y"),
            checked_i64(resource.thread_block_dimensions[2], "thread_block_dim_z"),
            static_cast<int64_t>(resource.uses_shared_barrier)});
  }

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
  auto get_string_attr = [&target](const char* name) {
    return target->GetAttr<ffi::String>(name).value();
  };

  auto create = ffi::Function::GetGlobal("ffi.Module.create.vortex");
  TVM_FFI_CHECK(create.has_value(), RuntimeError)
      << "Vortex runtime module is not loaded. Rebuild with USE_VORTEX set to the explicit "
         "Vortex repository path.";
  return (*create)(binary, ffi::String(code), ExtractFuncInfo(mod), kernel_ids, kernel_resources,
                   get_u32_attr("num_warps"), get_u32_attr("thread_warp_size"),
                   get_u32_attr("max_threads_per_block"), get_u32_attr("local_mem_size"),
                   get_u32_attr("xlen"), get_u32_attr("vortex_accelerator_profile_version"),
                   get_string_attr("vortex_accelerator_profile_fingerprint"),
                   get_string_attr("vortex_accelerator_profile_configs"),
                   get_string_attr("vortex_tcu_mode"), get_string_attr("vortex_tcu_fp_formats"),
                   get_string_attr("vortex_gemm_mode"), get_string_attr("vortex_platform"),
                   get_u32_attr("vortex_gemm_abi_version"),
                   get_u32_attr("vortex_layout_abi_version"), get_u32_attr("vortex_mxu_row"),
                   get_u32_attr("vortex_mxu_col"), get_u32_attr("vortex_mxu_col_tile"),
                   get_u32_attr("vortex_tmem_bank_size"), get_u32_attr("vortex_num_dma_channels"),
                   get_u32_attr("vortex_gemm_acc_mem_depth"))
      .cast<ffi::Module>();
}

void RegisterVortexCodegen() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.vortex", BuildVortex);
}

}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() { tvm::codegen::RegisterVortexCodegen(); }
