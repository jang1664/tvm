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
 * \file vortex_module.cc
 * \brief Serializable Vortex binary module and packed launch wrapper.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <vx_tvm_abi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../../../runtime/file_utils.h"
#include "../../../runtime/metadata.h"
#include "../../../runtime/pack_args.h"
#include "../../../runtime/thread_storage_scope.h"
#include "../../../support/bytes_io.h"
#include "vortex_device_api.h"

namespace tvm {
namespace runtime {
namespace vortex {

static constexpr uint32_t kVortexModuleSerializationVersion = 1;

class VortexModuleNode final : public ffi::ModuleObj {
 public:
  VortexModuleNode(ffi::Bytes binary, ffi::String source, ffi::Map<ffi::String, FunctionInfo> fmap,
                   uint32_t abi_version, uint32_t num_warps, uint32_t thread_warp_size,
                   uint32_t max_threads_per_block, uint32_t xlen)
      : binary_(std::move(binary)),
        source_(std::move(source)),
        fmap_(std::move(fmap)),
        abi_version_(abi_version),
        num_warps_(num_warps),
        thread_warp_size_(thread_warp_size),
        max_threads_per_block_(max_threads_per_block),
        xlen_(xlen) {
    TVM_FFI_CHECK_EQ(abi_version_, VX_TVM_ABI_VERSION, ValueError)
        << "Unsupported Vortex TVM ABI version " << abi_version_;
    TVM_FFI_CHECK_EQ(fmap_.size(), 1, ValueError)
        << "The Vortex single-kernel runtime requires exactly one function";
    TVM_FFI_CHECK_EQ(static_cast<uint64_t>(num_warps_) * thread_warp_size_, max_threads_per_block_,
                     ValueError)
        << "Vortex target capacity is inconsistent";
    TVM_FFI_CHECK(xlen_ == 32 || xlen_ == 64, ValueError)
        << "Vortex pointer width must be 32 or 64 bits";
  }

  const char* kind() const final { return "vortex"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(kVortexModuleSerializationVersion);
    stream.Write(binary_);
    stream.Write(source_);
    stream.Write(fmap_);
    stream.Write(abi_version_);
    stream.Write(num_warps_);
    stream.Write(thread_warp_size_);
    stream.Write(max_threads_per_block_);
    stream.Write(xlen_);
    return ffi::Bytes(std::move(result));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format.empty() || format == "vortex" || format == "cpp") return source_;
    if (format == "vxbin") return ffi::String(binary_.data(), binary_.size());
    return ffi::String();
  }

  ffi::Array<ffi::String> GetWriteFormats() const final { return {"vortex"}; }

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
    TVM_FFI_CHECK(format.empty() || format == "vortex", ValueError)
        << "Vortex modules can only be written with the .vortex format";
    SaveBinaryToFile(file_name, SaveToBytes());
  }

  void Launch(const FunctionInfo& info, ffi::PackedArgs args, void** void_args) {
    const size_t num_kernel_args = info->arg_types.size();
    TVM_FFI_CHECK_EQ(args.size(), num_kernel_args + info->launch_param_tags.size(), ValueError)
        << "Vortex function " << info->name << " expected " << num_kernel_args
        << " kernel arguments and " << info->launch_param_tags.size()
        << " launch arguments, but got " << args.size();
    TVM_FFI_CHECK_LE(num_kernel_args, std::numeric_limits<uint32_t>::max(), ValueError)
        << "Too many Vortex kernel arguments";

    for (size_t i = 0; i < info->launch_param_tags.size(); ++i) {
      const ffi::String& tag = info->launch_param_tags[i];
      TVM_FFI_CHECK(tag != launch_param::kUseDynamicSharedMemoryTag &&
                        tag != launch_param::kUseProgramaticDependentLaunch &&
                        tag != launch_param::kUseCooperativeLaunch,
                    ValueError)
          << "Vortex does not support launch tag " << tag;
      int64_t dimension = args[num_kernel_args + i].cast<int64_t>();
      TVM_FFI_CHECK_GT(dimension, 0, ValueError) << "Vortex launch dimensions must be positive";
      TVM_FFI_CHECK_LE(static_cast<uint64_t>(dimension), std::numeric_limits<uint32_t>::max(),
                       ValueError)
          << "Vortex launch dimension does not fit the ABI";
    }

    LaunchParamConfig launch_config;
    launch_config.Init(num_kernel_args, info->launch_param_tags);
    ThreadWorkLoad workload = launch_config.Extract(args);
    uint64_t block_size = 1;
    for (size_t i = 0; i < 3; ++i) {
      TVM_FFI_CHECK_LE(workload.grid_dim(i), std::numeric_limits<uint32_t>::max(), ValueError);
      TVM_FFI_CHECK_LE(workload.block_dim(i), std::numeric_limits<uint32_t>::max(), ValueError);
      TVM_FFI_CHECK_LE(block_size, std::numeric_limits<uint64_t>::max() / workload.block_dim(i),
                       ValueError)
          << "Vortex block size overflows";
      block_size *= workload.block_dim(i);
    }
    TVM_FFI_CHECK_LE(block_size, max_threads_per_block_, ValueError)
        << "Vortex block contains " << block_size << " threads, exceeding target limit "
        << max_threads_per_block_;

    VortexDeviceAPI* device_api = VortexDeviceAPI::Global();
    std::vector<uint64_t> slots(num_kernel_args, 0);
    for (size_t i = 0; i < num_kernel_args; ++i) {
      DLDataType dtype = info->arg_types[i];
      TVM_FFI_CHECK_EQ(dtype.lanes, 1, ValueError) << "Vortex kernel arguments must be scalar";
      if (dtype.code == kDLOpaqueHandle) {
        void* pointer = *static_cast<void**>(void_args[i]);
        TVM_FFI_CHECK(pointer != nullptr, ValueError) << "Vortex pointer argument is null";
        slots[i] = device_api->ResolveAddress(pointer);
        if (xlen_ == 32) {
          TVM_FFI_CHECK_LE(slots[i], std::numeric_limits<uint32_t>::max(), ValueError)
              << "Vortex device address does not fit xlen=32";
        }
      } else if (dtype.code == kDLInt || dtype.code == kDLUInt) {
        TVM_FFI_CHECK(dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64,
                      ValueError)
            << "Unsupported Vortex integer argument width " << static_cast<int>(dtype.bits);
        std::memcpy(&slots[i], void_args[i], dtype.bits / 8);
      } else if (dtype.code == kDLFloat && dtype.bits == 32) {
        std::memcpy(&slots[i], void_args[i], sizeof(float));
      } else if (dtype.code == kDLFloat && dtype.bits == 64) {
        std::memcpy(&slots[i], void_args[i], sizeof(double));
      } else {
        TVM_FFI_THROW(ValueError) << "Unsupported Vortex argument dtype " << dtype;
      }
    }

    uint64_t actual_capacity = device_api->ActualThreadCapacity();
    TVM_FFI_CHECK_LE(block_size, std::min<uint64_t>(max_threads_per_block_, actual_capacity),
                     ValueError)
        << "Vortex block contains " << block_size
        << " threads, exceeding the actual hardware capacity " << actual_capacity;

    std::vector<uint8_t> packet(sizeof(vx_tvm_launch_header_t) + slots.size() * sizeof(uint64_t));
    auto* header = reinterpret_cast<vx_tvm_launch_header_t*>(packet.data());
    header->abi_version = abi_version_;
    header->num_args = static_cast<uint32_t>(slots.size());
    header->kernel_id = 0;
    header->reserved = 0;
    for (size_t i = 0; i < 3; ++i) {
      header->grid[i] = static_cast<uint32_t>(workload.grid_dim(i));
      header->block[i] = static_cast<uint32_t>(workload.block_dim(i));
    }
    std::memcpy(packet.data() + sizeof(vx_tvm_launch_header_t), slots.data(),
                slots.size() * sizeof(uint64_t));

    TVM_FFI_CHECK_GT(binary_.size(), 16, ValueError)
        << "Vortex vxbin is too small to contain its address header";
    vx_buffer_h packet_buffer = device_api->UploadPacket(packet.data(), packet.size());
    try {
      device_api->Launch(binary_.data(), binary_.size(), packet_buffer);
    } catch (...) {
      device_api->ReleaseRuntimeBuffer(packet_buffer);
      throw;
    }
    device_api->ReleaseRuntimeBuffer(packet_buffer);
  }

 private:
  ffi::Bytes binary_;
  ffi::String source_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  uint32_t abi_version_;
  uint32_t num_warps_;
  uint32_t thread_warp_size_;
  uint32_t max_threads_per_block_;
  uint32_t xlen_;
};

ffi::Optional<ffi::Function> VortexModuleNode::GetFunction(const ffi::String& name) {
  auto info = fmap_.Get(name);
  if (!info.has_value()) return std::nullopt;
  ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
  FunctionInfo function_info = info.value();
  auto launch = [self, this, function_info](ffi::PackedArgs args, ffi::Any* rv,
                                            void** void_args) {
    this->Launch(function_info, args, void_args);
  };
  return PackFuncVoidAddr(launch, function_info->arg_types, function_info->arg_extra_tags);
}

static ffi::Module VortexModuleCreate(ffi::Bytes binary, ffi::String source,
                                      ffi::Map<ffi::String, FunctionInfo> fmap,
                                      uint32_t abi_version, uint32_t num_warps,
                                      uint32_t thread_warp_size, uint32_t max_threads_per_block,
                                      uint32_t xlen) {
  auto node = ffi::make_object<VortexModuleNode>(std::move(binary), std::move(source),
                                                 std::move(fmap), abi_version, num_warps,
                                                 thread_warp_size, max_threads_per_block, xlen);
  return ffi::Module(node);
}

static ffi::Module VortexModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  uint32_t serialization_version = 0;
  ffi::Bytes binary;
  ffi::String source;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  uint32_t abi_version = 0;
  uint32_t num_warps = 0;
  uint32_t thread_warp_size = 0;
  uint32_t max_threads_per_block = 0;
  uint32_t xlen = 0;
  TVM_FFI_CHECK(stream.Read(&serialization_version), ValueError)
      << "Invalid Vortex module serialization";
  TVM_FFI_CHECK_EQ(serialization_version, kVortexModuleSerializationVersion, ValueError)
      << "Unsupported Vortex module serialization version " << serialization_version;
  TVM_FFI_CHECK(stream.Read(&binary) && stream.Read(&source) && stream.Read(&fmap) &&
                    stream.Read(&abi_version) && stream.Read(&num_warps) &&
                    stream.Read(&thread_warp_size) && stream.Read(&max_threads_per_block) &&
                    stream.Read(&xlen),
                ValueError)
      << "Truncated Vortex module serialization";
  return VortexModuleCreate(std::move(binary), std::move(source), std::move(fmap), abi_version,
                            num_warps, thread_warp_size, max_threads_per_block, xlen);
}

static ffi::Module VortexModuleLoadFromFile(const ffi::String& file_name,
                                            const ffi::String& format) {
  TVM_FFI_CHECK(format.empty() || format == "vortex", ValueError)
      << "Vortex modules can only be loaded with the .vortex format";
  std::string bytes;
  LoadBinaryFromFile(file_name, &bytes);
  return VortexModuleLoadFromBytes(ffi::Bytes(std::move(bytes)));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.create.vortex", VortexModuleCreate)
      .def("ffi.Module.load_from_file.vortex", VortexModuleLoadFromFile)
      .def("ffi.Module.load_from_bytes.vortex", VortexModuleLoadFromBytes);
}

}  // namespace vortex
}  // namespace runtime
}  // namespace tvm
