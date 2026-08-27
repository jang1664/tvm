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
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <vx_tvm_abi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../../runtime/file_utils.h"
#include "../../../runtime/metadata.h"
#include "../../../runtime/pack_args.h"
#include "../../../runtime/thread_storage_scope.h"
#include "../../../support/bytes_io.h"
#include "../vortex_common.h"
#include "vortex_device_api.h"

namespace tvm {
namespace runtime {
namespace vortex {

static constexpr uint32_t kVortexModuleSerializationVersion = 6;
static constexpr size_t kVortexKernelResourceFieldCount = 8;
static constexpr const char* kAcceleratorProfileMetadataFunction =
    "vortex.get_accelerator_profile_metadata";

using SerializedKernelResources = ffi::Map<ffi::String, ffi::Array<int64_t>>;
using SerializedAcceleratorProfile = ffi::Map<ffi::String, ffi::String>;

struct KernelResourceMetadata {
  uint32_t launch_rank;
  uint64_t static_shared_bytes;
  uint64_t compile_time_resident_groups;
  uint64_t private_local_bytes_per_thread;
  uint32_t thread_block_dimensions[3];
  bool uses_shared_barrier;
};

std::pair<std::filesystem::path, std::string> ReadAuthoritativeXrtManifestConfigs(
    const std::string& xclbin_path) {
  TVM_FFI_CHECK(!xclbin_path.empty(), RuntimeError)
      << "Vortex XRT manifest validation requires a non-empty XRT_XCLBIN_PATH";
  std::filesystem::path manifest_path =
      std::filesystem::path(xclbin_path).parent_path().parent_path() / "manifest.json";
  std::ifstream stream(manifest_path);
  TVM_FFI_CHECK(stream.good(), RuntimeError)
      << "The authoritative Vortex XRT manifest cannot be opened: " << manifest_path.string();
  std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  namespace json = ::tvm::ffi::json;
  json::Object root = json::Parse(content).cast<json::Object>();
  json::Object params = root.at("params").cast<json::Object>();
  std::string configs = params.at("CONFIGS").cast<ffi::String>();
  return {manifest_path, configs};
}

uint64_t ReadLegacyXrtBarrierCount(const std::string& xclbin_path, uint64_t default_count) {
  auto [manifest_path, configs] = ReadAuthoritativeXrtManifestConfigs(xclbin_path);
  std::regex override_pattern(R"((^|\s)-DNUM_BARRIERS(?:=([0-9]+))?(?=\s|$))");
  std::smatch match;
  if (!std::regex_search(configs, match, override_pattern)) return default_count;
  if (!match[2].matched) return 1;
  try {
    return std::stoull(match[2].str());
  } catch (const std::exception&) {
    TVM_FFI_THROW(RuntimeError) << "Invalid NUM_BARRIERS override in authoritative XRT manifest "
                                << manifest_path.string();
  }
}

void ValidateAcceleratorProfile(const SerializedAcceleratorProfile& expected,
                                const std::string& driver_name, const std::string& xclbin_path) {
  auto required = [&expected](const char* name) -> std::string {
    auto value = expected.Get(ffi::String(name));
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "Vortex accelerator profile metadata is missing " << name;
    return value.value();
  };
  std::string tcu_mode = required("tcu_mode");
  std::string gemm_mode = required("gemm_mode");
  bool accelerated = tcu_mode != "none" || gemm_mode != "none";
  if (!accelerated) return;

  TVM_FFI_CHECK_EQ(driver_name, "xrt", RuntimeError)
      << "Vortex accelerator profile validation requires the XRT driver, but got " << driver_name;
  std::string fingerprint = required("fingerprint");
  std::string expected_configs = required("configs");
  TVM_FFI_CHECK_EQ(fingerprint.size(), 64, RuntimeError)
      << "Accelerated Vortex modules require a 64-digit manifest fingerprint";
  TVM_FFI_CHECK(!expected_configs.empty(), RuntimeError)
      << "Accelerated Vortex modules require exact manifest CONFIGS metadata; rebuild the target "
         "with load_vortex_accelerator_profile";
  auto [manifest_path, actual_configs] = ReadAuthoritativeXrtManifestConfigs(xclbin_path);
  TVM_FFI_CHECK_EQ(actual_configs, expected_configs, RuntimeError)
      << "Loaded Vortex xclbin accelerator profile does not match the module compiled profile; "
         "module fingerprint="
      << fingerprint << ", manifest=" << manifest_path.string();
}

void ValidateBarrierConfiguration(uint64_t num_warps, uint64_t reported_num_barriers,
                                  const std::string& driver_name, const std::string& xclbin_path) {
  TVM_FFI_CHECK_GT(num_warps, 0, RuntimeError)
      << "Vortex barrier validation requires a positive hardware warp count";
  uint64_t expected = (num_warps + 1) / 2;
  uint64_t effective = reported_num_barriers;
  if (effective == 0) {
    TVM_FFI_CHECK_EQ(driver_name, "xrt", RuntimeError)
        << "Vortex barrier capability is unavailable for driver " << driver_name
        << "; refusing to launch a barrier-using kernel";
    effective = ReadLegacyXrtBarrierCount(xclbin_path, expected);
  }
  TVM_FFI_CHECK_EQ(effective, expected, RuntimeError)
      << "Vortex barrier-using kernels require NUM_BARRIERS=ceil(NUM_WARPS/2)=" << expected
      << ", but the effective hardware configuration has NUM_BARRIERS=" << effective;
}

class VortexModuleNode final : public ffi::ModuleObj {
 public:
  VortexModuleNode(ffi::Bytes binary, ffi::String source, ffi::Map<ffi::String, FunctionInfo> fmap,
                   ffi::Map<ffi::String, int64_t> kernel_ids,
                   SerializedKernelResources serialized_kernel_resources, uint32_t abi_version,
                   uint32_t num_warps, uint32_t thread_warp_size, uint32_t max_threads_per_block,
                   uint32_t local_mem_size, uint32_t xlen,
                   SerializedAcceleratorProfile accelerator_profile)
      : binary_(std::move(binary)),
        source_(std::move(source)),
        fmap_(std::move(fmap)),
        kernel_ids_(std::move(kernel_ids)),
        serialized_kernel_resources_(std::move(serialized_kernel_resources)),
        abi_version_(abi_version),
        num_warps_(num_warps),
        thread_warp_size_(thread_warp_size),
        max_threads_per_block_(max_threads_per_block),
        local_mem_size_(local_mem_size),
        xlen_(xlen),
        accelerator_profile_(std::move(accelerator_profile)) {
    TVM_FFI_CHECK(!fmap_.empty(), ValueError)
        << "The Vortex runtime requires at least one function";
    TVM_FFI_CHECK_EQ(kernel_ids_.size(), fmap_.size(), ValueError)
        << "Vortex kernel-ID mapping must contain exactly one entry per function";
    TVM_FFI_CHECK_EQ(serialized_kernel_resources_.size(), fmap_.size(), ValueError)
        << "Vortex kernel resource metadata must contain exactly one entry per function";
    kernel_resources_.resize(fmap_.size());
    std::set<int64_t> observed_ids;
    for (const auto& entry : fmap_) {
      const ffi::String& name = entry.first;
      auto kernel_id = kernel_ids_.Get(name);
      TVM_FFI_CHECK(kernel_id.has_value(), ValueError)
          << "Vortex kernel-ID mapping is missing function " << name;
      TVM_FFI_CHECK_GE(kernel_id.value(), 0, ValueError)
          << "Vortex kernel ID for " << name << " must be non-negative";
      TVM_FFI_CHECK_LT(static_cast<uint64_t>(kernel_id.value()), fmap_.size(), ValueError)
          << "Vortex kernel ID for " << name << " is outside the dispatcher range";
      TVM_FFI_CHECK(observed_ids.insert(kernel_id.value()).second, ValueError)
          << "Vortex kernel-ID mapping contains duplicate ID " << kernel_id.value();
      auto serialized_resource = serialized_kernel_resources_.Get(name);
      TVM_FFI_CHECK(serialized_resource.has_value(), ValueError)
          << "Vortex kernel resource metadata is missing function " << name;
      const ffi::Array<int64_t>& fields = serialized_resource.value();
      TVM_FFI_CHECK_EQ(fields.size(), kVortexKernelResourceFieldCount, ValueError)
          << "Vortex kernel resource metadata for " << name << " must contain exactly "
          << kVortexKernelResourceFieldCount << " fields";
      TVM_FFI_CHECK(fields[0] >= 1 && fields[0] <= 3, ValueError)
          << "Vortex launch_rank for " << name << " must be in [1, 3]";
      if (!entry.second->launch_param_tags.empty()) {
        LaunchParamConfig launch_config;
        launch_config.Init(0, entry.second->launch_param_tags);
        TVM_FFI_CHECK_EQ(fields[0], launch_config.work_dim(), ValueError)
            << "Vortex launch_rank for " << name << " does not match function launch rank "
            << launch_config.work_dim();
      }
      TVM_FFI_CHECK_GE(fields[1], 0, ValueError)
          << "Vortex static_shared_bytes for " << name << " must be non-negative";
      TVM_FFI_CHECK_GT(fields[2], 0, ValueError)
          << "Vortex compile_time_resident_groups for " << name << " must be positive";
      TVM_FFI_CHECK_GE(fields[3], 0, ValueError)
          << "Vortex private_local_bytes_per_thread for " << name << " must be non-negative";
      uint64_t block_threads = 1;
      for (size_t axis = 0; axis < 3; ++axis) {
        TVM_FFI_CHECK_GT(fields[4 + axis], 0, ValueError)
            << "Vortex thread_block_dim_" << static_cast<char>('x' + axis) << " for " << name
            << " must be positive";
        TVM_FFI_CHECK_LE(static_cast<uint64_t>(fields[4 + axis]),
                         std::numeric_limits<uint32_t>::max(), ValueError)
            << "Vortex thread_block_dim_" << static_cast<char>('x' + axis) << " for " << name
            << " does not fit uint32";
        TVM_FFI_CHECK_LE(
            block_threads,
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(fields[4 + axis]),
            ValueError)
            << "Vortex thread block dimensions for " << name << " overflow uint64";
        block_threads *= static_cast<uint64_t>(fields[4 + axis]);
      }
      TVM_FFI_CHECK(fields[7] == 0 || fields[7] == 1, ValueError)
          << "Vortex uses_shared_barrier for " << name << " must be 0 or 1";
      kernel_resources_[kernel_id.value()] = {
          static_cast<uint32_t>(fields[0]),
          static_cast<uint64_t>(fields[1]),
          static_cast<uint64_t>(fields[2]),
          static_cast<uint64_t>(fields[3]),
          {static_cast<uint32_t>(fields[4]), static_cast<uint32_t>(fields[5]),
           static_cast<uint32_t>(fields[6])},
          fields[7] != 0};
    }
    TVM_FFI_CHECK_EQ(abi_version_, VX_TVM_ABI_VERSION, ValueError)
        << "Unsupported Vortex TVM ABI version " << abi_version_;
    TVM_FFI_CHECK_EQ(static_cast<uint64_t>(num_warps_) * thread_warp_size_, max_threads_per_block_,
                     ValueError)
        << "Vortex target capacity is inconsistent";
    TVM_FFI_CHECK_GT(local_mem_size_, 0, ValueError)
        << "Vortex target local_mem_size must be positive";
    for (size_t kernel_id = 0; kernel_id < kernel_resources_.size(); ++kernel_id) {
      const KernelResourceMetadata& resource = kernel_resources_[kernel_id];
      TVM_FFI_CHECK_LE(resource.compile_time_resident_groups, num_warps_, ValueError)
          << "Vortex compile_time_resident_groups for kernel ID " << kernel_id
          << " exceeds target num_warps";
      uint64_t block_threads = 1;
      for (uint32_t dimension : resource.thread_block_dimensions) {
        TVM_FFI_CHECK_LE(block_threads, max_threads_per_block_ / dimension, ValueError)
            << "Vortex compile-time thread block dimensions for kernel ID " << kernel_id
            << " exceed max_threads_per_block " << max_threads_per_block_;
        block_threads *= dimension;
      }
      uint64_t warps_per_group = 1 + (block_threads - 1) / thread_warp_size_;
      TVM_FFI_CHECK_EQ(num_warps_ / warps_per_group, resource.compile_time_resident_groups,
                       ValueError)
          << "Vortex compile-time thread block dimensions for kernel ID " << kernel_id
          << " do not match compile_time_resident_groups";
      TVM_FFI_CHECK_LE(resource.static_shared_bytes,
                       std::numeric_limits<uint64_t>::max() / resource.compile_time_resident_groups,
                       ValueError)
          << "Vortex resident static shared-memory requirement overflows for kernel ID "
          << kernel_id;
      TVM_FFI_CHECK_LE(resource.static_shared_bytes * resource.compile_time_resident_groups,
                       local_mem_size_, ValueError)
          << "Vortex resident static shared-memory requirement exceeds target local_mem_size for "
             "kernel ID "
          << kernel_id;
    }
    TVM_FFI_CHECK(xlen_ == 32 || xlen_ == 64, ValueError)
        << "Vortex pointer width must be 32 or 64 bits";
    for (const char* field :
         {"profile_version", "fingerprint", "configs", "tcu_mode", "tcu_fp_formats", "gemm_mode",
          "platform", "gemm_abi_version", "layout_abi_version", "mxu_row", "mxu_col",
          "mxu_col_tile", "tmem_bank_size", "num_dma_channels", "gemm_acc_mem_depth",
          "dma_mt", "dma_nt", "dma_kt", "qparam_slot_alignment", "tmem_alignment",
          "dimension_bits", "device_address_bits", "tile_counter_bits", "job_entries",
          "num_cores"}) {
      TVM_FFI_CHECK(accelerator_profile_.count(ffi::String(field)), ValueError)
          << "Vortex accelerator profile metadata is missing " << field;
    }
    auto profile_value = [this](const char* field) {
      return std::string(accelerator_profile_.at(ffi::String(field)));
    };
    TVM_FFI_CHECK_EQ(profile_value("profile_version"), "1", ValueError)
        << "Unsupported Vortex accelerator profile metadata version";
    std::string fingerprint = profile_value("fingerprint");
    std::string configs = profile_value("configs");
    TVM_FFI_CHECK(fingerprint.empty() == configs.empty(), ValueError)
        << "Vortex accelerator profile fingerprint and CONFIGS must be coupled";
    TVM_FFI_CHECK(fingerprint.empty() ||
                      (fingerprint.size() == 64 &&
                       fingerprint.find_first_not_of("0123456789abcdef") == std::string::npos),
                  ValueError)
        << "Vortex accelerator profile fingerprint must be empty or 64 lowercase hex digits";
    std::string tcu_mode = profile_value("tcu_mode");
    std::string gemm_mode = profile_value("gemm_mode");
    std::string platform = profile_value("platform");
    TVM_FFI_CHECK(
        tcu_mode == "none" || tcu_mode == "fp" || tcu_mode == "int" || tcu_mode == "fp_int",
        ValueError)
        << "Invalid serialized Vortex TCU mode " << tcu_mode;
    TVM_FFI_CHECK(gemm_mode == "none" || gemm_mode == "naive" || gemm_mode == "non_naive" ||
                      gemm_mode == "improve",
                  ValueError)
        << "Invalid serialized Vortex GEMM mode " << gemm_mode;
    TVM_FFI_CHECK(platform == "generic" || platform == "vivado", ValueError)
        << "Invalid serialized Vortex platform " << platform;
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
    stream.Write(kernel_ids_);
    stream.Write(accelerator_profile_);
    stream.Write(serialized_kernel_resources_);
    stream.Write(abi_version_);
    stream.Write(num_warps_);
    stream.Write(thread_warp_size_);
    stream.Write(max_threads_per_block_);
    stream.Write(local_mem_size_);
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

  void Launch(const FunctionInfo& info, uint32_t kernel_id, ffi::PackedArgs args,
              void** void_args) {
    TVM_FFI_CHECK_LT(kernel_id, kernel_ids_.size(), ValueError)
        << "Vortex kernel ID " << kernel_id << " is outside the dispatcher range";
    const KernelResourceMetadata& resource = kernel_resources_.at(kernel_id);
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
    for (size_t axis = 0; axis < 3; ++axis) {
      TVM_FFI_CHECK_EQ(workload.block_dim(axis), resource.thread_block_dimensions[axis], ValueError)
          << "Vortex runtime block dimension " << static_cast<char>('x' + axis) << "="
          << workload.block_dim(axis) << " does not match the compile-time dimension "
          << resource.thread_block_dimensions[axis];
    }
    uint64_t warps_per_group = 1 + (block_size - 1) / thread_warp_size_;
    uint64_t resident_groups = num_warps_ / warps_per_group;
    TVM_FFI_CHECK_EQ(resident_groups, resource.compile_time_resident_groups, ValueError)
        << "Vortex launch block dimensions imply " << resident_groups
        << " resident groups, but kernel metadata was compiled for "
        << resource.compile_time_resident_groups;

    VortexDeviceAPI* device_api = VortexDeviceAPI::Global();
    bool accelerated = accelerator_profile_.at("tcu_mode") != "none" ||
                       accelerator_profile_.at("gemm_mode") != "none";
    auto validate_actual_profile = [&]() {
      VortexActualResourceProfile actual = device_api->ActualResourceProfile();
      ValidateAcceleratorProfile(accelerator_profile_, actual.driver_name, actual.xclbin_path);
      TVM_FFI_CHECK_EQ(actual.thread_warp_size, thread_warp_size_, ValueError)
          << "Vortex target thread_warp_size " << thread_warp_size_
          << " does not match actual hardware " << actual.thread_warp_size;
      TVM_FFI_CHECK_EQ(actual.num_warps, num_warps_, ValueError)
          << "Vortex target num_warps " << num_warps_ << " does not match actual hardware "
          << actual.num_warps;
      TVM_FFI_CHECK_GE(actual.local_mem_size, local_mem_size_, ValueError)
          << "Vortex target local_mem_size " << local_mem_size_
          << " exceeds actual VX_CAPS_LOCAL_MEM_SIZE " << actual.local_mem_size;
      TVM_FFI_CHECK_LE(resource.static_shared_bytes,
                       actual.local_mem_size / resource.compile_time_resident_groups, ValueError)
          << "Vortex kernel requires "
          << resource.static_shared_bytes * resource.compile_time_resident_groups
          << " resident shared-memory bytes, exceeding actual VX_CAPS_LOCAL_MEM_SIZE "
          << actual.local_mem_size;
      if (resource.uses_shared_barrier) {
        ValidateBarrierConfiguration(actual.num_warps, actual.num_barriers, actual.driver_name,
                                     actual.xclbin_path);
      }
    };
    if (accelerated) validate_actual_profile();

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
    if (!accelerated) validate_actual_profile();

    std::vector<uint8_t> packet(sizeof(vx_tvm_launch_header_t) + slots.size() * sizeof(uint64_t));
    auto* header = reinterpret_cast<vx_tvm_launch_header_t*>(packet.data());
    header->abi_version = VX_TVM_ABI_VERSION;
    header->num_args = static_cast<uint32_t>(slots.size());
    header->kernel_id = kernel_id;
    header->reserved = 0;
    for (size_t i = 0; i < 3; ++i) {
      header->grid[i] = static_cast<uint32_t>(workload.grid_dim(i));
      header->block[i] = static_cast<uint32_t>(workload.block_dim(i));
    }
    std::memcpy(packet.data() + sizeof(vx_tvm_launch_header_t), slots.data(),
                slots.size() * sizeof(uint64_t));

    TVM_FFI_CHECK_GT(binary_.size(), 16, ValueError)
        << "Vortex vxbin is too small to contain its address header";
    device_api->Launch(binary_.data(), binary_.size(), packet.data(), packet.size());
  }

 private:
  ffi::Bytes binary_;
  ffi::String source_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  ffi::Map<ffi::String, int64_t> kernel_ids_;
  SerializedKernelResources serialized_kernel_resources_;
  std::vector<KernelResourceMetadata> kernel_resources_;
  uint32_t abi_version_;
  uint32_t num_warps_;
  uint32_t thread_warp_size_;
  uint32_t max_threads_per_block_;
  uint32_t local_mem_size_;
  uint32_t xlen_;
  SerializedAcceleratorProfile accelerator_profile_;
};

ffi::Optional<ffi::Function> VortexModuleNode::GetFunction(const ffi::String& name) {
  if (name == kKernelResourceMetadataFunction) {
    ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
    return ffi::Function([self, this](ffi::PackedArgs args, ffi::Any* rv) {
      TVM_FFI_CHECK_EQ(args.size(), 0, ValueError)
          << kKernelResourceMetadataFunction << " expects no arguments";
      *rv = serialized_kernel_resources_;
    });
  }
  if (name == kAcceleratorProfileMetadataFunction) {
    ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
    return ffi::Function([self, this](ffi::PackedArgs args, ffi::Any* rv) {
      TVM_FFI_CHECK_EQ(args.size(), 0, ValueError)
          << kAcceleratorProfileMetadataFunction << " expects no arguments";
      *rv = accelerator_profile_;
    });
  }
  auto info = fmap_.Get(name);
  if (!info.has_value()) return std::nullopt;
  auto mapped_id = kernel_ids_.Get(name);
  TVM_FFI_CHECK(mapped_id.has_value(), ValueError)
      << "Vortex kernel-ID mapping is missing function " << name;
  TVM_FFI_CHECK_GE(mapped_id.value(), 0, ValueError)
      << "Vortex kernel ID for " << name << " must be non-negative";
  uint32_t kernel_id = static_cast<uint32_t>(mapped_id.value());
  ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
  FunctionInfo function_info = info.value();
  auto launch = [self, this, function_info, kernel_id](ffi::PackedArgs args, ffi::Any* rv,
                                                       void** void_args) {
    this->Launch(function_info, kernel_id, args, void_args);
  };
  return PackFuncVoidAddr(launch, function_info->arg_types, function_info->arg_extra_tags);
}

static ffi::Module VortexModuleCreate(
    ffi::Bytes binary, ffi::String source, ffi::Map<ffi::String, FunctionInfo> fmap,
    ffi::Map<ffi::String, int64_t> kernel_ids, SerializedKernelResources kernel_resources,
    uint32_t num_warps, uint32_t thread_warp_size, uint32_t max_threads_per_block,
    uint32_t local_mem_size, uint32_t xlen, uint32_t accelerator_profile_version,
    ffi::String accelerator_profile_fingerprint, ffi::String accelerator_profile_configs,
    ffi::String tcu_mode, ffi::String tcu_fp_formats, ffi::String gemm_mode, ffi::String platform,
    uint32_t gemm_abi_version, uint32_t layout_abi_version, uint32_t mxu_row, uint32_t mxu_col,
    uint32_t mxu_col_tile, uint32_t tmem_bank_size, uint32_t num_dma_channels,
    uint32_t gemm_acc_mem_depth, uint32_t dma_mt, uint32_t dma_nt, uint32_t dma_kt,
    uint32_t qparam_slot_alignment, uint32_t tmem_alignment, uint32_t dimension_bits,
    uint32_t device_address_bits, uint32_t tile_counter_bits, uint32_t job_entries,
    uint32_t num_cores) {
  SerializedAcceleratorProfile accelerator_profile{
      {"profile_version", ffi::String(std::to_string(accelerator_profile_version))},
      {"fingerprint", std::move(accelerator_profile_fingerprint)},
      {"configs", std::move(accelerator_profile_configs)},
      {"tcu_mode", std::move(tcu_mode)},
      {"tcu_fp_formats", std::move(tcu_fp_formats)},
      {"gemm_mode", std::move(gemm_mode)},
      {"platform", std::move(platform)},
      {"gemm_abi_version", ffi::String(std::to_string(gemm_abi_version))},
      {"layout_abi_version", ffi::String(std::to_string(layout_abi_version))},
      {"mxu_row", ffi::String(std::to_string(mxu_row))},
      {"mxu_col", ffi::String(std::to_string(mxu_col))},
      {"mxu_col_tile", ffi::String(std::to_string(mxu_col_tile))},
      {"tmem_bank_size", ffi::String(std::to_string(tmem_bank_size))},
      {"num_dma_channels", ffi::String(std::to_string(num_dma_channels))},
      {"gemm_acc_mem_depth", ffi::String(std::to_string(gemm_acc_mem_depth))},
      {"dma_mt", ffi::String(std::to_string(dma_mt))},
      {"dma_nt", ffi::String(std::to_string(dma_nt))},
      {"dma_kt", ffi::String(std::to_string(dma_kt))},
      {"qparam_slot_alignment", ffi::String(std::to_string(qparam_slot_alignment))},
      {"tmem_alignment", ffi::String(std::to_string(tmem_alignment))},
      {"dimension_bits", ffi::String(std::to_string(dimension_bits))},
      {"device_address_bits", ffi::String(std::to_string(device_address_bits))},
      {"tile_counter_bits", ffi::String(std::to_string(tile_counter_bits))},
      {"job_entries", ffi::String(std::to_string(job_entries))},
      {"num_cores", ffi::String(std::to_string(num_cores))}};
  auto node = ffi::make_object<VortexModuleNode>(
      std::move(binary), std::move(source), std::move(fmap), std::move(kernel_ids),
      std::move(kernel_resources), VX_TVM_ABI_VERSION, num_warps, thread_warp_size,
      max_threads_per_block, local_mem_size, xlen, std::move(accelerator_profile));
  return ffi::Module(node);
}

static ffi::Module VortexModuleCreateFromSerialized(
    ffi::Bytes binary, ffi::String source, ffi::Map<ffi::String, FunctionInfo> fmap,
    ffi::Map<ffi::String, int64_t> kernel_ids, SerializedKernelResources kernel_resources,
    uint32_t abi_version, uint32_t num_warps, uint32_t thread_warp_size,
    uint32_t max_threads_per_block, uint32_t local_mem_size, uint32_t xlen,
    SerializedAcceleratorProfile accelerator_profile) {
  auto node = ffi::make_object<VortexModuleNode>(
      std::move(binary), std::move(source), std::move(fmap), std::move(kernel_ids),
      std::move(kernel_resources), abi_version, num_warps, thread_warp_size, max_threads_per_block,
      local_mem_size, xlen, std::move(accelerator_profile));
  return ffi::Module(node);
}

static ffi::Module VortexModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  uint32_t serialization_version = 0;
  ffi::Bytes binary;
  ffi::String source;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, int64_t> kernel_ids;
  SerializedKernelResources kernel_resources;
  uint32_t abi_version = 0;
  uint32_t num_warps = 0;
  uint32_t thread_warp_size = 0;
  uint32_t max_threads_per_block = 0;
  uint32_t local_mem_size = 0;
  uint32_t xlen = 0;
  SerializedAcceleratorProfile accelerator_profile;
  TVM_FFI_CHECK(stream.Read(&serialization_version), ValueError)
      << "Invalid Vortex module serialization";
  TVM_FFI_CHECK_EQ(serialization_version, kVortexModuleSerializationVersion, ValueError)
      << "Unsupported Vortex module serialization version " << serialization_version;
  TVM_FFI_CHECK(stream.Read(&binary) && stream.Read(&source) && stream.Read(&fmap) &&
                    stream.Read(&kernel_ids) && stream.Read(&accelerator_profile) &&
                    stream.Read(&kernel_resources) && stream.Read(&abi_version) &&
                    stream.Read(&num_warps) && stream.Read(&thread_warp_size) &&
                    stream.Read(&max_threads_per_block) && stream.Read(&local_mem_size) &&
                    stream.Read(&xlen),
                ValueError)
      << "Truncated Vortex module serialization";
  return VortexModuleCreateFromSerialized(
      std::move(binary), std::move(source), std::move(fmap), std::move(kernel_ids),
      std::move(kernel_resources), abi_version, num_warps, thread_warp_size, max_threads_per_block,
      local_mem_size, xlen, std::move(accelerator_profile));
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
      .def("ffi.Module.load_from_bytes.vortex", VortexModuleLoadFromBytes)
      .def("runtime.vortex.validate_barrier_configuration",
           [](int64_t num_warps, int64_t reported_num_barriers, ffi::String driver_name,
              ffi::String xclbin_path) {
             TVM_FFI_CHECK_GT(num_warps, 0, ValueError) << "num_warps must be positive";
             TVM_FFI_CHECK_GE(reported_num_barriers, 0, ValueError)
                 << "reported_num_barriers must be non-negative";
             ValidateBarrierConfiguration(static_cast<uint64_t>(num_warps),
                                          static_cast<uint64_t>(reported_num_barriers), driver_name,
                                          xclbin_path);
           })
      .def("runtime.vortex.validate_accelerator_profile",
           [](SerializedAcceleratorProfile expected, ffi::String driver_name,
              ffi::String xclbin_path) {
             ValidateAcceleratorProfile(expected, driver_name, xclbin_path);
           });
}

}  // namespace vortex
}  // namespace runtime
}  // namespace tvm
