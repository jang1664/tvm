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
#include <vx_tvm_abi.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace vortex {

namespace {

constexpr size_t kCopyStagingBytes = 1 << 20;
constexpr uint64_t kDefaultKernelTimeoutMs = 5 * 60 * 1000;

uint64_t KernelTimeoutMs() {
  const char* value = std::getenv("TVM_VORTEX_KERNEL_TIMEOUT_MS");
  if (value == nullptr || value[0] == '\0') return kDefaultKernelTimeoutMs;
  errno = 0;
  char* end = nullptr;
  unsigned long long timeout = std::strtoull(value, &end, 10);
  TVM_FFI_CHECK(value[0] != '-' && end != value && *end == '\0' && errno != ERANGE && timeout > 0,
                ValueError)
      << "TVM_VORTEX_KERNEL_TIMEOUT_MS must be a positive integer";
  return static_cast<uint64_t>(timeout);
}

void CheckCopyBounds(size_t offset, size_t num_bytes, size_t allocation_size,
                     const char* description) {
  TVM_FFI_CHECK_LE(offset, allocation_size, ValueError)
      << description << " copy offset exceeds its allocation";
  TVM_FFI_CHECK_LE(num_bytes, allocation_size - offset, ValueError)
      << description << " copy exceeds its allocation";
}

}  // namespace

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

void VortexDeviceAPI::ValidateEnvironmentLocked() const {
  const char* driver = std::getenv("VORTEX_DRIVER");
  TVM_FFI_CHECK(driver != nullptr && driver[0] != '\0', RuntimeError)
      << "VORTEX_DRIVER must be set explicitly before opening Vortex. "
         "Use the XRT hardware environment from ci/run_black.sh; simx is only for debugging.";
  TVM_FFI_CHECK(!poisoned_, RuntimeError)
      << "The Vortex device is unavailable after a failed kernel wait; restart the process";
  if (device_ == nullptr) return;
  TVM_FFI_CHECK_EQ(driver_name_, driver, RuntimeError)
      << "VORTEX_DRIVER changed after the Vortex device was opened; restart the process";
  if (driver_name_ == "xrt") {
    const char* xclbin = std::getenv("XRT_XCLBIN_PATH");
    TVM_FFI_CHECK(xclbin != nullptr && xclbin[0] != '\0' && xclbin_path_ == xclbin, RuntimeError)
        << "XRT_XCLBIN_PATH changed after the Vortex device was opened; restart the process";
  }
}

void VortexDeviceAPI::EnsureOpenLocked() {
  ValidateEnvironmentLocked();
  if (device_ != nullptr) return;
  const char* driver = std::getenv("VORTEX_DRIVER");
  const char* xclbin = std::getenv("XRT_XCLBIN_PATH");
  TVM_VORTEX_CALL(vx_dev_open(&device_));
  driver_name_ = driver;
  xclbin_path_ = driver_name_ == "xrt" && xclbin != nullptr ? xclbin : "";
}

VortexDeviceAPI::~VortexDeviceAPI() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (VortexAllocation* allocation : allocations_) {
    vx_mem_free(allocation->buffer);
    delete allocation;
  }
  allocations_.clear();
  if (packet_buffer_ != nullptr) {
    vx_mem_free(packet_buffer_);
    packet_buffer_ = nullptr;
  }
  packet_capacity_ = 0;
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
    if (device_ == nullptr) {
      if (poisoned_ || vx_dev_open(&device_) != 0) {
        device_ = nullptr;
        *rv = 0;
        return;
      }
      driver_name_ = driver;
      const char* xclbin = std::getenv("XRT_XCLBIN_PATH");
      xclbin_path_ = driver_name_ == "xrt" && xclbin != nullptr ? xclbin : "";
    } else {
      ValidateEnvironmentLocked();
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
    case kMaxThreadDimensions: {
      uint64_t warps = 0;
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_THREADS, &value));
      TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_WARPS, &warps));
      std::string capacity = std::to_string(value * warps);
      *rv = ffi::String("[" + capacity + ", " + capacity + ", " + capacity + "]");
    }
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
  allocations_.insert(allocation);
  return allocation;
}

VortexAllocation* VortexDeviceAPI::LookupAllocationLocked(const void* ptr) const {
  auto* allocation = static_cast<VortexAllocation*>(const_cast<void*>(ptr));
  auto it = allocations_.find(allocation);
  TVM_FFI_CHECK(it != allocations_.end(), ValueError)
      << "Pointer is not a live allocation owned by the Vortex DeviceAPI";
  return *it;
}

uint64_t VortexDeviceAPI::ResolveAddress(void* ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  VortexAllocation* allocation = LookupAllocationLocked(ptr);
  return allocation->address;
}

void VortexDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  ValidateDevice(dev);
  std::lock_guard<std::mutex> lock(mutex_);
  VortexAllocation* allocation = LookupAllocationLocked(ptr);
  TVM_VORTEX_CALL(vx_mem_free(allocation->buffer));
  allocations_.erase(allocation);
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
    CheckCopyBounds(from_offset, num_bytes, src->size, "Vortex source");
    CheckCopyBounds(to_offset, num_bytes, dst->size, "Vortex destination");
    std::vector<uint8_t> staging(std::min(num_bytes, kCopyStagingBytes));
    for (size_t copied = 0; copied < num_bytes; copied += staging.size()) {
      size_t chunk = std::min(staging.size(), num_bytes - copied);
      TVM_VORTEX_CALL(vx_copy_from_dev(staging.data(), src->buffer, from_offset + copied, chunk));
      TVM_VORTEX_CALL(vx_copy_to_dev(dst->buffer, staging.data(), to_offset + copied, chunk));
    }
  } else if (from_vortex) {
    VortexAllocation* src = LookupAllocationLocked(from);
    CheckCopyBounds(from_offset, num_bytes, src->size, "Vortex source");
    TVM_VORTEX_CALL(vx_copy_from_dev(static_cast<uint8_t*>(to) + to_offset, src->buffer,
                                     from_offset, num_bytes));
  } else {
    VortexAllocation* dst = LookupAllocationLocked(to);
    CheckCopyBounds(to_offset, num_bytes, dst->size, "Vortex destination");
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

VortexActualResourceProfile VortexDeviceAPI::ActualResourceProfile() {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  uint64_t threads = 0;
  uint64_t warps = 0;
  uint64_t local_mem_size = 0;
  uint64_t num_barriers = 0;
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_THREADS, &threads));
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_WARPS, &warps));
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size));
  TVM_VORTEX_CALL(vx_dev_caps(device_, VX_CAPS_NUM_BARRIERS, &num_barriers));
  return {warps, threads, local_mem_size, num_barriers, driver_name_, xclbin_path_};
}

void VortexDeviceAPI::Launch(const void* kernel_data, size_t kernel_size, const void* packet_data,
                             size_t packet_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureOpenLocked();
  if (packet_buffer_ == nullptr || packet_size > packet_capacity_) {
    vx_buffer_h new_packet = nullptr;
    TVM_VORTEX_CALL(vx_upload_bytes(device_, packet_data, packet_size, &new_packet));
    if (packet_buffer_ != nullptr) TVM_VORTEX_CALL(vx_mem_free(packet_buffer_));
    packet_buffer_ = new_packet;
    packet_capacity_ = packet_size;
  } else {
    TVM_VORTEX_CALL(vx_copy_to_dev(packet_buffer_, packet_data, 0, packet_size));
  }
  const auto* bytes = static_cast<const uint8_t*>(kernel_data);
  bool kernel_matches = kernel_buffer_ != nullptr && kernel_binary_.size() == kernel_size &&
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
  uint64_t timeout_ms = KernelTimeoutMs();
  TVM_VORTEX_CALL(vx_start(device_, kernel_buffer_, packet_buffer_));
  int error = vx_ready_wait(device_, timeout_ms);
  if (error != 0) {
    poisoned_ = true;
    TVM_FFI_THROW(RuntimeError) << "Vortex kernel wait failed with error " << error
                                << "; the device is poisoned and the process must be restarted";
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.ext_dev",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = static_cast<void*>(VortexDeviceAPI::Global());
                  })
      .def_packed("device_api.vortex",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = static_cast<void*>(VortexDeviceAPI::Global());
                  })
      .def("runtime.vortex_abi_version", []() { return int64_t{VX_TVM_ABI_VERSION}; });
}

}  // namespace vortex
}  // namespace runtime
}  // namespace tvm
