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
 * \file vortex_device_api.cc
 * \brief Vortex implementation of TVM's kDLExtDev device API.
 */
#include "vortex_device_api.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace tvm {
namespace runtime {
namespace vortex {

#define TVM_VORTEX_CALL(call)                                                           \
  do {                                                                                  \
    int _vortex_error = (call);                                                         \
    TVM_FFI_CHECK_EQ(_vortex_error, 0, RuntimeError)                                    \
        << "Vortex runtime call failed (" << #call << ") with error " << _vortex_error; \
  } while (false)

void VortexDeviceAPI::ValidateDevice(Device dev) {
  TVM_FFI_CHECK_EQ(dev.device_type, kDLExtDev, ValueError)
      << "Vortex requires kDLExtDev, but got device type " << dev.device_type;
  TVM_FFI_CHECK_EQ(dev.device_id, 0, ValueError)
      << "Vortex currently supports only device 0, but got device " << dev.device_id;
}

void VortexDeviceAPI::EnsureOpenLocked() {
  if (device_ != nullptr) return;
  const char* driver = std::getenv("VORTEX_DRIVER");
  TVM_FFI_CHECK(driver != nullptr && driver[0] != '\0', RuntimeError)
      << "VORTEX_DRIVER must be set explicitly before opening Vortex. "
         "Use the XRT hardware environment from ci/run_black.sh; simx is only for debugging.";
  TVM_VORTEX_CALL(vx_dev_open(&device_));
}

VortexDeviceAPI::~VortexDeviceAPI() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& item : allocations_) {
    vx_mem_free(item.second->buffer);
    delete item.second;
  }
  allocations_.clear();
  for (vx_buffer_h buffer : runtime_buffers_) {
    vx_mem_free(buffer);
  }
  runtime_buffers_.clear();
  if (kernel_buffer_ != nullptr) {
    vx_mem_free(kernel_buffer_);
    kernel_buffer_ = nullptr;
  }
  kernel_binary_.clear();
  if (device_ != nullptr) {
    vx_dev_close(device_);
    device_ = nullptr;
  }
}

VortexDeviceAPI* VortexDeviceAPI::Global() {
  static VortexDeviceAPI instance;
  return &instance;
}

void VortexDeviceAPI::SetDevice(Device dev) { ValidateDevice(dev); }

void VortexDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) {
  ValidateDevice(dev);
  if (kind == kExist) {
    const char* driver = std::getenv("VORTEX_DRIVER");
    if (driver == nullptr || driver[0] == '\0') {
      *rv = 0;
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == nullptr && vx_dev_open(&device_) != 0) {
      device_ = nullptr;
      *rv = 0;
      return;
    }
    *rv = 1;
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  uint64_t value = 0;
  switch (kind) {
    case kMaxThreadsPerBlock: {
      uint64_t warps = 0;
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_THREADS, &value));
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_WARPS, &warps));
      *rv = static_cast<int64_t>(value * warps);
      return;
    }
    case kWarpSize:
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_THREADS, &value));
      *rv = static_cast<int64_t>(value);
      return;
    case kMaxSharedMemoryPerBlock:
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_LOCAL_MEM_SIZE, &value));
      *rv = static_cast<int64_t>(value);
      return;
    case kDeviceName:
      *rv = ffi::String("Vortex");
      return;
    case kMaxThreadDimensions:
      *rv = ffi::String("[4294967295, 4294967295, 4294967295]");
      return;
    case kTotalGlobalMemory:
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_GLOBAL_MEM_SIZE, &value));
      *rv = static_cast<int64_t>(value);
      return;
    case kAvailableGlobalMemory: {
      uint64_t used = 0;
      TVM_VORTEX_CALL(vx_mem_info(device_, &value, &used));
      *rv = static_cast<int64_t>(value);
      return;
    }
    case kApiVersion:
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_VERSION, &value));
      *rv = static_cast<int64_t>(value);
      return;
    default:
      *rv = 0;
      return;
  }
}

void* VortexDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                      DLDataType type_hint) {
  ValidateDevice(dev);
  TVM_FFI_CHECK_GT(nbytes, 0, ValueError) << "Vortex allocation size must be positive";
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();

  auto* allocation = new VortexAllocation();
  allocation->size = nbytes;
  allocation->device_id = dev.device_id;
  int error = alignment > 1 ? vx_mem_alloc_aligned(device_, nbytes, alignment, VX_MEM_READ_WRITE,
                                                   &allocation->buffer)
                            : vx_mem_alloc(device_, nbytes, VX_MEM_READ_WRITE, &allocation->buffer);
  if (error != 0) {
    delete allocation;
    TVM_FFI_THROW(RuntimeError) << "Vortex allocation failed with error " << error;
  }
  error = vx_mem_address(allocation->buffer, &allocation->address);
  if (error != 0) {
    vx_mem_free(allocation->buffer);
    delete allocation;
    TVM_FFI_THROW(RuntimeError) << "Vortex address query failed with error " << error;
  }
  allocations_.emplace(allocation, allocation);
  return allocation;
}

VortexAllocation* VortexDeviceAPI::LookupAllocationLocked(const void* ptr) const {
  auto it = allocations_.find(ptr);
  TVM_FFI_CHECK(it != allocations_.end(), ValueError)
      << "Pointer is not a live allocation owned by the Vortex DeviceAPI";
  return it->second;
}

uint64_t VortexDeviceAPI::ResolveAddress(void* ptr, uint64_t required_size) const {
  std::lock_guard<std::mutex> lock(mutex_);
  VortexAllocation* allocation = LookupAllocationLocked(ptr);
  TVM_FFI_CHECK_LE(required_size, allocation->size, ValueError)
      << "Vortex pointer access exceeds its allocation";
  return allocation->address;
}

void VortexDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  ValidateDevice(dev);
  std::lock_guard<std::mutex> lock(mutex_);
  VortexAllocation* allocation = LookupAllocationLocked(ptr);
  TVM_VORTEX_CALL(vx_mem_free(allocation->buffer));
  allocations_.erase(ptr);
  delete allocation;
}

void VortexDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, size_t num_bytes, Device dev_from,
                                     Device dev_to, DLDataType type_hint, TVMStreamHandle stream) {
  TVM_FFI_CHECK(stream == nullptr, ValueError) << "Vortex does not support non-default streams";
  bool from_vortex = dev_from.device_type == kDLExtDev;
  bool to_vortex = dev_to.device_type == kDLExtDev;
  TVM_FFI_CHECK(from_vortex || to_vortex, ValueError)
      << "Vortex copies must have a kDLExtDev endpoint";
  if (from_vortex) ValidateDevice(dev_from);
  if (to_vortex) ValidateDevice(dev_to);

  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  if (from_vortex && to_vortex) {
    VortexAllocation* src = LookupAllocationLocked(from);
    VortexAllocation* dst = LookupAllocationLocked(to);
    TVM_FFI_CHECK_LE(from_offset + num_bytes, src->size, ValueError)
        << "Vortex source copy exceeds its allocation";
    TVM_FFI_CHECK_LE(to_offset + num_bytes, dst->size, ValueError)
        << "Vortex destination copy exceeds its allocation";
    std::vector<uint8_t> staging(num_bytes);
    TVM_VORTEX_CALL(vx_copy_from_dev(staging.data(), src->buffer, from_offset, num_bytes));
    TVM_VORTEX_CALL(vx_copy_to_dev(dst->buffer, staging.data(), to_offset, num_bytes));
  } else if (from_vortex) {
    VortexAllocation* src = LookupAllocationLocked(from);
    TVM_FFI_CHECK_LE(from_offset + num_bytes, src->size, ValueError)
        << "Vortex source copy exceeds its allocation";
    TVM_VORTEX_CALL(vx_copy_from_dev(static_cast<uint8_t*>(to) + to_offset, src->buffer,
                                     from_offset, num_bytes));
  } else {
    VortexAllocation* dst = LookupAllocationLocked(to);
    TVM_FFI_CHECK_LE(to_offset + num_bytes, dst->size, ValueError)
        << "Vortex destination copy exceeds its allocation";
    TVM_VORTEX_CALL(vx_copy_to_dev(dst->buffer, static_cast<const uint8_t*>(from) + from_offset,
                                   to_offset, num_bytes));
  }
}

void VortexDeviceAPI::StreamSync(Device dev, TVMStreamHandle stream) {
  ValidateDevice(dev);
  TVM_FFI_CHECK(stream == nullptr, ValueError) << "Vortex does not support non-default streams";
  // Vortex copies are synchronous and the launch path waits immediately after
  // vx_start.  Calling vx_ready_wait without a preceding launch can wait for a
  // completion that will never be produced, so there is no additional work to
  // perform here.
}

uint64_t VortexDeviceAPI::ActualThreadCapacity() {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  uint64_t threads = 0;
  uint64_t warps = 0;
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_THREADS, &threads));
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_WARPS, &warps));
  return threads * warps;
}

vx_buffer_h VortexDeviceAPI::UploadPacket(const void* data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  vx_buffer_h buffer = nullptr;
  TVM_VORTEX_CALL(vx_upload_bytes(device_, data, size, &buffer));
  runtime_buffers_.push_back(buffer);
  return buffer;
}

void VortexDeviceAPI::ReleaseRuntimeBuffer(vx_buffer_h buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = std::find(runtime_buffers_.begin(), runtime_buffers_.end(), buffer);
  TVM_FFI_CHECK(it != runtime_buffers_.end(), ValueError) << "Vortex runtime buffer is not live";
  TVM_VORTEX_CALL(vx_mem_free(buffer));
  runtime_buffers_.erase(it);
}

void VortexDeviceAPI::Launch(const void* kernel_data, size_t kernel_size, vx_buffer_h packet) {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  const auto* bytes = static_cast<const uint8_t*>(kernel_data);
  bool kernel_matches =
      kernel_buffer_ != nullptr && kernel_binary_.size() == kernel_size &&
      std::equal(kernel_binary_.begin(), kernel_binary_.end(), bytes);
  if (!kernel_matches) {
    if (kernel_buffer_ != nullptr) {
      TVM_VORTEX_CALL(vx_mem_free(kernel_buffer_));
      kernel_buffer_ = nullptr;
      kernel_binary_.clear();
    }
    vx_buffer_h new_kernel = nullptr;
    TVM_VORTEX_CALL(vx_upload_kernel_bytes(device_, kernel_data, kernel_size, &new_kernel));
    kernel_buffer_ = new_kernel;
    kernel_binary_.assign(bytes, bytes + kernel_size);
  }
  TVM_VORTEX_CALL(vx_start(device_, kernel_buffer_, packet));
  TVM_VORTEX_CALL(vx_ready_wait(device_, VX_MAX_TIMEOUT));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.ext_dev", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = static_cast<void*>(VortexDeviceAPI::Global());
  });
}

}  // namespace vortex
}  // namespace runtime
}  // namespace tvm
