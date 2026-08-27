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
#include <initializer_list>
#include <limits>
#include <set>
#include <string>
#include <utility>

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
  int64_t profile_version = target.at("vortex_accelerator_profile_version").cast<int64_t>();
  std::string tcu_mode = target.at("vortex_tcu_mode").cast<std::string>();
  std::string tcu_fp_formats = target.at("vortex_tcu_fp_formats").cast<std::string>();
  std::string gemm_mode = target.at("vortex_gemm_mode").cast<std::string>();
  std::string platform = target.at("vortex_platform").cast<std::string>();

  TVM_FFI_CHECK_GT(num_warps, 0, ValueError) << "Vortex num_warps must be positive";
  TVM_FFI_CHECK_GT(thread_warp_size, 0, ValueError) << "Vortex thread_warp_size must be positive";
  TVM_FFI_CHECK_GT(local_mem_size, 0, ValueError) << "Vortex local_mem_size must be positive";
  TVM_FFI_CHECK_GT(max_local_memory_per_thread, 0, ValueError)
      << "Vortex max_local_memory_per_thread must be positive";
  TVM_FFI_CHECK(xlen == 32 || xlen == 64, ValueError)
      << "Vortex xlen must be either 32 or 64, but got " << xlen;
  TVM_FFI_CHECK_EQ(profile_version, 1, ValueError)
      << "Vortex vortex_accelerator_profile_version must be 1";
  TVM_FFI_CHECK(tcu_mode == "none" || tcu_mode == "fp" || tcu_mode == "int" || tcu_mode == "fp_int",
                ValueError)
      << "Vortex vortex_tcu_mode must be none, fp, int, or fp_int, but got " << tcu_mode;
  TVM_FFI_CHECK(gemm_mode == "none" || gemm_mode == "naive" || gemm_mode == "non_naive" ||
                    gemm_mode == "improve",
                ValueError)
      << "Vortex vortex_gemm_mode must be none, naive, non_naive, or improve, but got "
      << gemm_mode;
  TVM_FFI_CHECK(platform == "generic" || platform == "vivado", ValueError)
      << "Vortex vortex_platform must be generic or vivado, but got " << platform;
  if (target.count("vortex_mabi")) {
    std::string mabi = target.at("vortex_mabi").cast<std::string>();
    TVM_FFI_CHECK(
        (xlen == 64 && (mabi == "lp64f" || mabi == "lp64d")) || (xlen == 32 && mabi == "ilp32f"),
        ValueError)
        << "Vortex vortex_mabi does not match xlen=" << xlen << ": " << mabi;
  }
  if (target.count("vortex_march")) {
    std::string march = target.at("vortex_march").cast<std::string>();
    std::string march_prefix = xlen == 64 ? "rv64" : "rv32";
    TVM_FFI_CHECK(march.rfind(march_prefix, 0) == 0, ValueError)
        << "Vortex vortex_march does not match xlen=" << xlen << ": " << march;
  }

  std::set<std::string> unique_fp_formats;
  size_t format_begin = 0;
  while (format_begin < tcu_fp_formats.size()) {
    size_t format_end = tcu_fp_formats.find(',', format_begin);
    if (format_end == std::string::npos) format_end = tcu_fp_formats.size();
    std::string value = tcu_fp_formats.substr(format_begin, format_end - format_begin);
    TVM_FFI_CHECK(value == "fp16" || value == "bf16", ValueError)
        << "Vortex vortex_tcu_fp_formats supports only fp16 and bf16, but got " << value;
    TVM_FFI_CHECK(unique_fp_formats.insert(value).second, ValueError)
        << "Vortex vortex_tcu_fp_formats contains duplicate " << value;
    format_begin = format_end + 1;
  }
  bool has_fp_tcu = tcu_mode == "fp" || tcu_mode == "fp_int";
  TVM_FFI_CHECK_EQ(has_fp_tcu, !unique_fp_formats.empty(), ValueError)
      << "Vortex vortex_tcu_fp_formats must be non-empty exactly when vortex_tcu_mode "
         "contains the FP path";
  std::string canonical_fp_formats;
  for (const char* format : {"fp16", "bf16"}) {
    if (unique_fp_formats.count(format)) {
      if (!canonical_fp_formats.empty()) canonical_fp_formats += ',';
      canonical_fp_formats += format;
    }
  }
  target.Set("vortex_tcu_fp_formats", ffi::String(canonical_fp_formats));

  auto require_positive = [&target](const char* name) {
    int64_t value = target.at(name).cast<int64_t>();
    TVM_FFI_CHECK_GT(value, 0, ValueError) << "Vortex " << name << " must be positive";
    return value;
  };
  int64_t mxu_row = require_positive("vortex_mxu_row");
  int64_t mxu_col = require_positive("vortex_mxu_col");
  int64_t mxu_col_tile = require_positive("vortex_mxu_col_tile");
  require_positive("vortex_tmem_bank_size");
  require_positive("vortex_num_dma_channels");
  require_positive("vortex_gemm_acc_mem_depth");
  int64_t dma_mt = require_positive("vortex_gemm_dma_mt");
  int64_t dma_nt = require_positive("vortex_gemm_dma_nt");
  int64_t dma_kt = require_positive("vortex_gemm_dma_kt");
  int64_t qparam_alignment = require_positive("vortex_gemm_qparam_slot_alignment");
  int64_t tmem_alignment = require_positive("vortex_gemm_tmem_alignment");
  require_positive("vortex_gemm_dimension_bits");
  require_positive("vortex_device_address_bits");
  require_positive("vortex_gemm_tile_counter_bits");
  int64_t job_entries = require_positive("vortex_gemm_job_entries");
  int64_t num_cores = require_positive("vortex_num_cores");
  require_positive("vortex_gemm_abi_version");
  require_positive("vortex_layout_abi_version");
  TVM_FFI_CHECK_EQ(mxu_col % mxu_col_tile, 0, ValueError)
      << "Vortex vortex_mxu_col must be divisible by vortex_mxu_col_tile";
  TVM_FFI_CHECK_GT(mxu_row, 0, ValueError);
  auto is_power_of_two = [](int64_t value) { return (value & (value - 1)) == 0; };
  for (const auto& [name, value] :
       std::initializer_list<std::pair<const char*, int64_t>>{{"vortex_gemm_dma_mt", dma_mt},
                                                              {"vortex_gemm_dma_nt", dma_nt},
                                                              {"vortex_gemm_dma_kt", dma_kt},
                                                              {"vortex_gemm_qparam_slot_alignment", qparam_alignment},
                                                              {"vortex_gemm_tmem_alignment", tmem_alignment}}) {
    TVM_FFI_CHECK(is_power_of_two(value), ValueError)
        << "Vortex " << name << " must be a power of two";
  }
  TVM_FFI_CHECK_EQ(dma_kt % mxu_row, 0, ValueError)
      << "Vortex vortex_gemm_dma_kt must be divisible by vortex_mxu_row";
  TVM_FFI_CHECK_EQ(dma_nt % mxu_col, 0, ValueError)
      << "Vortex vortex_gemm_dma_nt must be divisible by vortex_mxu_col";
  TVM_FFI_CHECK_LE(num_cores, job_entries, ValueError)
      << "Vortex vortex_num_cores cannot exceed vortex_gemm_job_entries";

  std::string fingerprint = target.at("vortex_accelerator_profile_fingerprint").cast<std::string>();
  std::string profile_configs = target.at("vortex_accelerator_profile_configs").cast<std::string>();
  TVM_FFI_CHECK(fingerprint.empty() ||
                    (fingerprint.size() == 64 &&
                     fingerprint.find_first_not_of("0123456789abcdef") == std::string::npos),
                ValueError)
      << "Vortex vortex_accelerator_profile_fingerprint must be empty or 64 lowercase hex digits";
  TVM_FFI_CHECK(fingerprint.empty() == profile_configs.empty(), ValueError)
      << "Vortex accelerator profile fingerprint and CONFIGS must either both be empty or both "
         "be present";
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
      .add_attr_option<ffi::String>("vortex_march")
      .add_attr_option<ffi::String>("vortex_mabi")
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      .add_attr_option<int64_t>("vortex_accelerator_profile_version", refl::DefaultValue(1))
      .add_attr_option<ffi::String>("vortex_accelerator_profile_fingerprint",
                                    refl::DefaultValue(ffi::String("")))
      .add_attr_option<ffi::String>("vortex_accelerator_profile_configs",
                                    refl::DefaultValue(ffi::String("")))
      .add_attr_option<ffi::String>("vortex_tcu_mode", refl::DefaultValue(ffi::String("none")))
      .add_attr_option<ffi::String>("vortex_tcu_fp_formats", refl::DefaultValue(ffi::String("")))
      .add_attr_option<ffi::String>("vortex_gemm_mode", refl::DefaultValue(ffi::String("none")))
      .add_attr_option<int64_t>("vortex_mxu_row", refl::DefaultValue(32))
      .add_attr_option<int64_t>("vortex_mxu_col", refl::DefaultValue(32))
      .add_attr_option<int64_t>("vortex_mxu_col_tile", refl::DefaultValue(1))
      .add_attr_option<int64_t>("vortex_tmem_bank_size", refl::DefaultValue(int64_t{64} << 10))
      .add_attr_option<int64_t>("vortex_num_dma_channels", refl::DefaultValue(8))
      .add_attr_option<int64_t>("vortex_gemm_acc_mem_depth", refl::DefaultValue(1024))
      .add_attr_option<ffi::String>("vortex_platform", refl::DefaultValue(ffi::String("generic")))
      .add_attr_option<int64_t>("vortex_gemm_dma_mt", refl::DefaultValue(128))
      .add_attr_option<int64_t>("vortex_gemm_dma_nt", refl::DefaultValue(128))
      .add_attr_option<int64_t>("vortex_gemm_dma_kt", refl::DefaultValue(128))
      .add_attr_option<int64_t>("vortex_gemm_qparam_slot_alignment", refl::DefaultValue(512))
      .add_attr_option<int64_t>("vortex_gemm_tmem_alignment", refl::DefaultValue(64))
      .add_attr_option<int64_t>("vortex_gemm_dimension_bits", refl::DefaultValue(32))
      .add_attr_option<int64_t>("vortex_device_address_bits", refl::DefaultValue(64))
      .add_attr_option<int64_t>("vortex_gemm_tile_counter_bits", refl::DefaultValue(32))
      .add_attr_option<int64_t>("vortex_gemm_job_entries", refl::DefaultValue(4))
      .add_attr_option<int64_t>("vortex_num_cores", refl::DefaultValue(1))
      .add_attr_option<int64_t>("vortex_gemm_abi_version", refl::DefaultValue(2))
      .add_attr_option<int64_t>("vortex_layout_abi_version", refl::DefaultValue(2))
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
