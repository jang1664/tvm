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
 * \file target_kind.cc
 * \brief Vortex compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <cstdint>
#include <limits>
#include <string>

#include "vortex_resource.h"

namespace tvm {
namespace backend {
namespace vortex {

ffi::Map<ffi::String, ffi::Any> CanonicalizeVortexTarget(ffi::Map<ffi::String, ffi::Any> target) {
  int64_t num_warps = target.at("num_warps").cast<int64_t>();
  int64_t thread_warp_size = target.at("thread_warp_size").cast<int64_t>();
  int64_t local_mem_size = target.at("local_mem_size").cast<int64_t>();
  int64_t max_local_memory_per_thread = target.at("max_local_memory_per_thread").cast<int64_t>();
  int64_t xlen = target.at("xlen").cast<int64_t>();

  TVM_FFI_CHECK_GT(num_warps, 0, ValueError) << "Vortex num_warps must be positive";
  TVM_FFI_CHECK_GT(thread_warp_size, 0, ValueError) << "Vortex thread_warp_size must be positive";
  TVM_FFI_CHECK_GT(local_mem_size, 0, ValueError) << "Vortex local_mem_size must be positive";
  TVM_FFI_CHECK_GT(max_local_memory_per_thread, 0, ValueError)
      << "Vortex max_local_memory_per_thread must be positive";
  TVM_FFI_CHECK(xlen == 32 || xlen == 64, ValueError)
      << "Vortex xlen must be either 32 or 64, but got " << xlen;
  TVM_FFI_CHECK_LE(num_warps, std::numeric_limits<int64_t>::max() / thread_warp_size, ValueError)
      << "Vortex thread capacity overflows int64: num_warps=" << num_warps
      << ", thread_warp_size=" << thread_warp_size;

  int64_t thread_capacity = num_warps * thread_warp_size;
  for (const char* attr_name : {"max_threads_per_block", "max_num_threads", "max_block_size_x",
                                "max_block_size_y", "max_block_size_z"}) {
    if (target.count(attr_name)) {
      int64_t configured_limit = target.at(attr_name).cast<int64_t>();
      TVM_FFI_CHECK_EQ(configured_limit, thread_capacity, ValueError)
          << "Vortex " << attr_name << " must equal num_warps * thread_warp_size ("
          << thread_capacity << "), but got " << configured_limit;
    } else {
      target.Set(attr_name, thread_capacity);
    }
  }

  if (target.count("max_shared_memory_per_block")) {
    int64_t configured_limit = target.at("max_shared_memory_per_block").cast<int64_t>();
    TVM_FFI_CHECK(configured_limit == 0 || configured_limit == local_mem_size, ValueError)
        << "Vortex max_shared_memory_per_block must be the legacy zero sentinel or equal "
           "physical local_mem_size ("
        << local_mem_size << "), but got " << configured_limit;
  }
  target.Set("max_shared_memory_per_block", local_mem_size);

  std::string triple_prefix = xlen == 64 ? "riscv64" : "riscv32";
  if (target.count("mtriple")) {
    std::string mtriple = target.at("mtriple").cast<std::string>();
    TVM_FFI_CHECK(mtriple.rfind(triple_prefix, 0) == 0, ValueError)
        << "Vortex mtriple must match xlen=" << xlen << ", but got " << mtriple;
  } else {
    target.Set("mtriple", ffi::String(triple_prefix + "-unknown-elf"));
  }

  return target;
}

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("vortex", kDLExtDev)
      .add_attr_option<int64_t>("num_warps", refl::DefaultValue(4))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
      .add_attr_option<int64_t>("max_threads_per_block")
      .add_attr_option<int64_t>("max_num_threads")
      .add_attr_option<int64_t>("max_block_size_x")
      .add_attr_option<int64_t>("max_block_size_y")
      .add_attr_option<int64_t>("max_block_size_z")
      .add_attr_option<int64_t>("local_mem_size", refl::DefaultValue(int64_t{1} << 20))
      .add_attr_option<int64_t>("max_shared_memory_per_block")
      .add_attr_option<int64_t>("max_local_memory_per_thread", refl::DefaultValue(int64_t{4} << 10))
      .add_attr_option<int64_t>("xlen", refl::DefaultValue(64))
      .add_attr_option<ffi::String>("mtriple")
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      .set_default_keys({"vortex", "gpu"})
      .set_target_canonicalizer(CanonicalizeVortexTarget);
}

void RegisterResourceFunctions() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.vortex.get_block_resource_usage",
           [](Target target, int64_t block_threads) {
             VortexBlockResourceUsage usage = GetVortexBlockResourceUsage(target, block_threads);
             ffi::Map<ffi::String, int64_t> result;
             result.Set("block_threads", usage.block_threads);
             result.Set("warps_per_group", usage.warps_per_group);
             result.Set("resident_groups", usage.resident_groups);
             result.Set("effective_max_shared_memory_per_block",
                        usage.effective_max_shared_memory_per_block);
             return result;
           })
      .def("target.vortex.validate_shared_memory_usage", ValidateVortexSharedMemoryUsage);
}

}  // namespace vortex
}  // namespace backend
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::vortex::RegisterTargetKind();
  tvm::backend::vortex::RegisterResourceFunctions();
}
