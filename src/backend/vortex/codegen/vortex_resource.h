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
 * \file vortex_resource.h
 * \brief Static Vortex thread-block residency and LMEM resource accounting.
 */
#ifndef TVM_BACKEND_VORTEX_CODEGEN_VORTEX_RESOURCE_H_
#define TVM_BACKEND_VORTEX_CODEGEN_VORTEX_RESOURCE_H_

#include <tvm/target/target.h>

#include <cstdint>

namespace tvm {
namespace backend {
namespace vortex {

struct VortexBlockResourceUsage {
  int64_t block_threads;
  int64_t warps_per_group;
  int64_t resident_groups;
  int64_t effective_max_shared_memory_per_block;
};

VortexBlockResourceUsage GetVortexBlockResourceUsage(const Target& target, int64_t block_threads);

void ValidateVortexSharedMemoryUsage(const Target& target, int64_t block_threads,
                                     int64_t static_shared_bytes);

}  // namespace vortex
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_VORTEX_CODEGEN_VORTEX_RESOURCE_H_
