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
 * \file vortex_resource.cc
 * \brief Static Vortex thread-block residency and LMEM resource accounting.
 */
#include "vortex_resource.h"

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <limits>

namespace tvm {
namespace backend {
namespace vortex {

VortexBlockResourceUsage GetVortexBlockResourceUsage(const Target& target, int64_t block_threads) {
  TVM_FFI_CHECK_EQ(target->kind->name, "vortex", ValueError)
      << "Vortex resource accounting requires a vortex target, but got " << target->kind->name;
  int64_t threads_per_warp = target->GetAttr<int64_t>("thread_warp_size").value();
  int64_t num_warps = target->GetAttr<int64_t>("num_warps").value();
  int64_t local_mem_size = target->GetAttr<int64_t>("local_mem_size").value();
  int64_t max_threads_per_block = target->GetAttr<int64_t>("max_threads_per_block").value();

  TVM_FFI_CHECK_GT(block_threads, 0, ValueError)
      << "Vortex block_threads must be positive, but got " << block_threads;
  TVM_FFI_CHECK_LE(block_threads, max_threads_per_block, ValueError)
      << "Vortex block_threads " << block_threads << " exceeds max_threads_per_block "
      << max_threads_per_block;

  // This form of ceil(block_threads / threads_per_warp) cannot overflow.
  int64_t warps_per_group = 1 + (block_threads - 1) / threads_per_warp;
  int64_t resident_groups = num_warps / warps_per_group;
  TVM_FFI_ICHECK_GT(resident_groups, 0)
      << "Canonicalized Vortex target cannot schedule a valid thread block";

  return {block_threads, warps_per_group, resident_groups, local_mem_size / resident_groups};
}

void ValidateVortexSharedMemoryUsage(const Target& target, int64_t block_threads,
                                     int64_t static_shared_bytes) {
  TVM_FFI_CHECK_GE(static_shared_bytes, 0, ValueError)
      << "Vortex static_shared_bytes must be non-negative, but got " << static_shared_bytes;
  VortexBlockResourceUsage usage = GetVortexBlockResourceUsage(target, block_threads);
  TVM_FFI_CHECK_LE(static_shared_bytes, std::numeric_limits<int64_t>::max() / usage.resident_groups,
                   ValueError)
      << "Vortex total resident LMEM requirement overflows int64: resident_groups="
      << usage.resident_groups << ", static_shared_bytes=" << static_shared_bytes;

  int64_t required_lmem = usage.resident_groups * static_shared_bytes;
  int64_t local_mem_size = target->GetAttr<int64_t>("local_mem_size").value();
  TVM_FFI_CHECK_LE(required_lmem, local_mem_size, ValueError)
      << "Vortex block with " << block_threads << " threads and " << static_shared_bytes
      << " bytes of static shared memory requires " << required_lmem << " bytes across "
      << usage.resident_groups << " resident groups, exceeding target local_mem_size "
      << local_mem_size;
}

}  // namespace vortex
}  // namespace backend
}  // namespace tvm
