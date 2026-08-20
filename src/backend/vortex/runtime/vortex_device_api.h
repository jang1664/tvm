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
 * \file vortex_device_api.h
 * \brief Shared host-runtime services for the Vortex external device.
 */
#ifndef TVM_BACKEND_VORTEX_RUNTIME_VORTEX_DEVICE_API_H_
#define TVM_BACKEND_VORTEX_RUNTIME_VORTEX_DEVICE_API_H_

#include <tvm/ffi/any.h>
#include <tvm/runtime/device_api.h>
#include <vortex.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vortex {

struct VortexAllocation {
  vx_buffer_h buffer{nullptr};
  uint64_t address{0};
  uint64_t size{0};
  int device_id{0};
};

class VortexDeviceAPI final : public DeviceAPI {
 public:
  ~VortexDeviceAPI() final;

  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;

  uint64_t ResolveAddress(void* ptr, uint64_t required_size = 1) const;
  uint64_t ActualThreadCapacity();
  vx_buffer_h UploadKernel(const void* data, size_t size);
  vx_buffer_h UploadPacket(const void* data, size_t size);
  void ReleaseRuntimeBuffer(vx_buffer_h buffer);
  void Launch(vx_buffer_h kernel, vx_buffer_h packet);

  static VortexDeviceAPI* Global();

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                      size_t num_bytes, Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

 private:
  static void ValidateDevice(Device dev);
  void EnsureOpenLocked();
  VortexAllocation* LookupAllocationLocked(const void* ptr) const;

  mutable std::mutex mutex_;
  vx_device_h device_{nullptr};
  std::unordered_map<const void*, VortexAllocation*> allocations_;
  std::vector<vx_buffer_h> runtime_buffers_;
};

}  // namespace vortex
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_BACKEND_VORTEX_RUNTIME_VORTEX_DEVICE_API_H_
