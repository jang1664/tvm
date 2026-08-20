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
 * \file codegen_vortex.cc
 */
#include "codegen_vortex.h"

#include <tvm/runtime/logging.h>

#include <sstream>
#include <string>

namespace tvm {
namespace codegen {

CodeGenVortex::CodeGenVortex(Target target) : target_(std::move(target)) {
  restrict_keyword_ = "__restrict__";
  decl_stream
      << "#include <stdint.h>\n"
         "#include <vx_intrinsics.h>\n"
         "#include <vx_spawn.h>\n"
         "#include <vx_tvm_abi.h>\n\n"
         "template <typename T>\n"
         "static inline T __tvm_vortex_decode_scalar(uint64_t bits) {\n"
         "  static_assert(sizeof(T) <= sizeof(bits), \"Vortex ABI scalar exceeds one slot\");\n"
         "  T value = {};\n"
         "  __builtin_memcpy(&value, &bits, sizeof(T));\n"
         "  return value;\n"
         "}\n\n";
}

void CodeGenVortex::InitFuncState(const PrimFunc& func) {
  CodeGenC::InitFuncState(func);
  for (const Var& param : func->params) {
    if (param->ty.as<PointerTypeNode>()) {
      alloc_storage_scope_[param.get()] = "global";
    } else if (const auto* buffer_type = param->ty.as<BufferTypeNode>()) {
      alloc_storage_scope_[param.get()] = "global";
      RegisterHandleType(param.get(), buffer_type->dtype);
    }
  }
}

void CodeGenVortex::PrintFuncPrefix(std::ostream& os) { os << "static "; }

void CodeGenVortex::PrintType(const Type& type, std::ostream& os) {
  if (const auto* buffer_type = type.as<BufferTypeNode>()) {
    CodeGenC::PrintType(buffer_type->dtype, os);
    os << '*';
    return;
  }
  CodeGenC::PrintType(type, os);
}

void CodeGenVortex::BindThreadIndex(const IterVar& iv) {
  TVM_FFI_CHECK(!var_idmap_.count(iv->var.get()), ValueError)
      << "CodeGenVortex: thread variable is bound more than once: " << iv->thread_tag;
  TVM_FFI_CHECK(iv->thread_tag == "threadIdx.x" || iv->thread_tag == "blockIdx.x", ValueError)
      << "CodeGenVortex: only 1D threadIdx.x and blockIdx.x are supported, but got "
      << iv->thread_tag;
  var_idmap_[iv->var.get()] =
      CastFromTo(std::string(iv->thread_tag), PrimType::UInt(32), iv->var.ty());
}

void CodeGenVortex::VisitStmt_(const ForNode* op) {
  TVM_FFI_CHECK(op->thread_binding.has_value(), ValueError)
      << "CodeGenVortex: serial loops are not supported by the vector-add MVP";
  TVM_FFI_CHECK(is_zero(op->min), ValueError)
      << "CodeGenVortex: thread-bound loops must have a zero minimum";
  const IterVar& binding = op->thread_binding.value();
  BindThreadIndex(binding);
  if (!binding->var.same_as(op->loop_var)) {
    var_idmap_[op->loop_var.get()] =
        CastFromTo(std::string(binding->thread_tag), PrimType::UInt(32), op->loop_var.ty());
  }
  PrintStmt(op->body);
}

void CodeGenVortex::VisitStmt_(const AllocBufferNode* op) {
  TVM_FFI_THROW(ValueError)
      << "CodeGenVortex: local or shared allocation is not supported by the vector-add MVP (buffer "
      << op->buffer.name() << ")";
}

std::string CodeGenVortex::PrintTypeString(const Type& type) {
  std::ostringstream os;
  PrintType(type, os);
  return os.str();
}

void CodeGenVortex::EmitLaunchWrapper(const PrimFunc& func) {
  stream << "static void __tvm_vortex_kernel_entry(void* opaque) {\n"
            "  const vx_tvm_launch_header_t* launch =\n"
            "      static_cast<const vx_tvm_launch_header_t*>(opaque);\n"
            "  const uint64_t* args = vx_tvm_launch_args(launch);\n"
            "  __tvm_vortex_kernel(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    if (i != 0) stream << ", ";
    const Type& type = func->params[i]->ty;
    std::string type_name = PrintTypeString(type);
    if (type.as<PointerTypeNode>() || type.as<BufferTypeNode>()) {
      stream << "reinterpret_cast<" << type_name << ">(static_cast<uintptr_t>(args[" << i << "]))";
    } else {
      TVM_FFI_CHECK(type.as<PrimTypeNode>(), ValueError)
          << "CodeGenVortex: unsupported launch argument type " << type;
      stream << "__tvm_vortex_decode_scalar<" << type_name << ">(args[" << i << "])";
    }
  }
  stream << ");\n}\n\n"
            "int main() {\n"
            "  const vx_tvm_launch_header_t* launch =\n"
            "      reinterpret_cast<const vx_tvm_launch_header_t*>(csr_read(VX_CSR_MSCRATCH));\n"
            "  if (launch == nullptr || launch->abi_version != VX_TVM_ABI_VERSION ||\n"
         << "      launch->num_args != " << func->params.size()
         << "u || launch->kernel_id != 0u) {\n"
            "    return -1;\n"
            "  }\n"
            "  return vx_spawn_threads(1, launch->grid, launch->block,\n"
            "                          (vx_kernel_func_cb)__tvm_vortex_kernel_entry, launch);\n"
            "}\n";
}

void CodeGenVortex::AddKernel(const GlobalVar& gvar, const PrimFunc& func) {
  TVM_FFI_CHECK(!has_kernel_, ValueError)
      << "CodeGenVortex: the single-kernel MVP accepts exactly one PrimFunc";
  has_kernel_ = true;

  PrimFunc kernel = WithAttr(func, tvm::attr::kGlobalSymbol, ffi::String("__tvm_vortex_kernel"));
  CodeGenC::AddFunction(gvar, kernel);
  EmitLaunchWrapper(kernel);
}

}  // namespace codegen
}  // namespace tvm
