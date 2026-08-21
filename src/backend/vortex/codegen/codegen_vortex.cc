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
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "vortex_resource.h"

namespace tvm {
namespace codegen {

namespace {

struct VortexThreadTag {
  bool is_thread;
  size_t axis;
};

VortexThreadTag ParseVortexThreadTag(const ffi::String& thread_tag) {
  static constexpr std::array<const char*, 3> kThreadTags = {"threadIdx.x", "threadIdx.y",
                                                             "threadIdx.z"};
  static constexpr std::array<const char*, 3> kBlockTags = {"blockIdx.x", "blockIdx.y",
                                                            "blockIdx.z"};
  for (size_t axis = 0; axis < kThreadTags.size(); ++axis) {
    if (thread_tag == kThreadTags[axis]) return {true, axis};
    if (thread_tag == kBlockTags[axis]) return {false, axis};
  }
  TVM_FFI_THROW(ValueError) << "CodeGenVortex: unknown thread binding tag " << thread_tag
                            << "; expected threadIdx.{x,y,z} or blockIdx.{x,y,z}";
}

const char* ThreadAxisLimitName(size_t axis) {
  static constexpr std::array<const char*, 3> kLimitNames = {"max_block_size_x", "max_block_size_y",
                                                             "max_block_size_z"};
  return kLimitNames.at(axis);
}

constexpr uint64_t kMaxCompilerAlignment = uint64_t{1} << 28;

bool IsPowerOfTwo(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

uint64_t AlignUpChecked(uint64_t value, uint64_t alignment, const ffi::String& buffer_name) {
  uint64_t padding = alignment - 1;
  TVM_FFI_CHECK_LE(value, std::numeric_limits<uint64_t>::max() - padding, ValueError)
      << "CodeGenVortex: local buffer " << buffer_name
      << " aligned allocation size overflows uint64";
  return (value + padding) & ~padding;
}

uint64_t GetConstantAllocationBytes(const AllocBufferNode* op, const char* scope_name) {
  const ffi::String& buffer_name = op->buffer.name();
  uint64_t constant_size = 1;
  for (const PrimExpr& dim : op->buffer->shape) {
    const auto* dim_imm = dim.as<IntImmNode>();
    TVM_FFI_CHECK(dim_imm != nullptr, ValueError)
        << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
        << " must have a compile-time constant shape";
    TVM_FFI_CHECK_GT(dim_imm->value, 0, ValueError)
        << "CodeGenVortex: " << scope_name << " buffer " << buffer_name << " extent "
        << dim_imm->value << " must be positive";
    uint64_t extent = static_cast<uint64_t>(dim_imm->value);
    TVM_FFI_CHECK_LE(extent, std::numeric_limits<uint64_t>::max() / constant_size, ValueError)
        << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
        << " shape product overflows uint64";
    constant_size *= extent;
  }

  PrimType dtype = op->buffer->dtype;
  TVM_FFI_CHECK(!dtype.IsScalableVector(), ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
      << " cannot use a scalable-vector dtype";
  uint64_t dtype_bits = static_cast<uint64_t>(dtype.bits());
  uint64_t dtype_lanes = static_cast<uint64_t>(dtype.lanes());
  TVM_FFI_CHECK_GT(dtype_bits, 0, ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
      << " dtype must have a positive bit width";
  TVM_FFI_CHECK_EQ(dtype_bits % 8, 0, ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
      << " requires a byte-addressable dtype";
  TVM_FFI_CHECK_LE(dtype_lanes, std::numeric_limits<uint64_t>::max() / dtype_bits, ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
      << " dtype size overflows uint64";
  uint64_t element_bytes = (dtype_bits * dtype_lanes) / 8;
  TVM_FFI_CHECK_LE(constant_size, std::numeric_limits<uint64_t>::max() / element_bytes, ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
      << " byte size overflows uint64";
  return constant_size * element_bytes;
}

uint64_t GetAllocationAlignment(const AllocBufferNode* op, const char* scope_name) {
  const ffi::String& buffer_name = op->buffer.name();
  int64_t alignment = op->buffer->data_alignment;
  auto alignment_annotation = op->annotations.find(tirx::attr::buffer_data_alignment);
  if (alignment_annotation != op->annotations.end()) {
    const auto* alignment_imm = (*alignment_annotation).second.as<IntImmNode>();
    TVM_FFI_CHECK(alignment_imm != nullptr, ValueError)
        << "CodeGenVortex: " << scope_name << " buffer " << buffer_name
        << " alignment annotation must be a compile-time integer";
    alignment = alignment_imm->value;
  }
  TVM_FFI_CHECK_GT(alignment, 0, ValueError) << "CodeGenVortex: " << scope_name << " buffer "
                                             << buffer_name << " alignment must be positive";
  uint64_t alignment_u64 = static_cast<uint64_t>(alignment);
  TVM_FFI_CHECK(IsPowerOfTwo(alignment_u64) && alignment_u64 <= kMaxCompilerAlignment, ValueError)
      << "CodeGenVortex: " << scope_name << " buffer " << buffer_name << " alignment " << alignment
      << " must be a power of two no greater than " << kMaxCompilerAlignment;
  return alignment_u64;
}

class SharedAllocationCollector final : public tirx::StmtVisitor {
 public:
  std::vector<const AllocBufferNode*> allocations;

  void VisitStmt_(const AllocBufferNode* op) final {
    std::string scope = op->buffer.scope();
    TVM_FFI_CHECK_NE(scope, "shared.dyn", ValueError)
        << "CodeGenVortex: storage scope shared.dyn is not supported for buffer "
        << op->buffer.name() << "; dynamic shared memory is not supported by the Vortex ABI";
    if (scope == "shared") allocations.push_back(op);
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

class SharedBarrierFinder final : public tirx::StmtVisitor {
 public:
  bool found{false};

  void VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync()) && !call->args.empty()) {
        if (const auto* scope = call->args[0].as<StringImmNode>()) {
          if (scope->value == "shared") found = true;
        }
      }
    }
    if (!found) tirx::StmtVisitor::VisitStmt_(op);
  }
};

class BarrierUniformityValidator final : public tirx::StmtVisitor {
 public:
  explicit BarrierUniformityValidator(const PrimFunc& func) {
    SharedBarrierFinder finder;
    finder(func->body);
    has_shared_barrier_ = finder.found;
  }

  void Validate(const PrimFunc& func) {
    if (has_shared_barrier_) VisitStmt(func->body);
  }

  bool has_shared_barrier() const { return has_shared_barrier_; }

 private:
  bool DependsOnThread(const PrimExpr& expr) const {
    if (tirx::UsesVar(expr, [this](const VarNode* var) { return thread_dependent_.count(var); })) {
      return true;
    }
    bool reads_thread_private_storage = false;
    tirx::PostOrderVisit(expr, [&reads_thread_private_storage](const ffi::ObjectRef& node) {
      if (const auto* load = node.as<BufferLoadNode>()) {
        if (load->buffer.scope() == "local") reads_thread_private_storage = true;
      }
    });
    return reads_thread_private_storage;
  }

  void CheckEarlyExit(const char* kind) const {
    TVM_FFI_CHECK_EQ(thread_dependent_condition_depth_, 0, ValueError)
        << "CodeGenVortex: shared barrier cannot be combined with a thread-dependent early exit ("
        << kind << ")";
    TVM_FFI_CHECK_EQ(thread_dependent_serial_loop_depth_, 0, ValueError)
        << "CodeGenVortex: shared barrier cannot be combined with an early exit inside a "
           "thread-dependent serial loop ("
        << kind << ")";
    TVM_FFI_CHECK_EQ(unproven_loop_depth_, 0, ValueError)
        << "CodeGenVortex: shared barrier cannot be combined with an early exit inside a loop "
           "whose uniform iteration count cannot be proven ("
        << kind << ")";
  }

  void VisitStmt_(const BindNode* op) final {
    if (auto value = op->value.as<PrimExpr>(); value && DependsOnThread(value.value())) {
      thread_dependent_.insert(op->var.get());
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key != tirx::attr::thread_extent) {
      tirx::StmtVisitor::VisitStmt_(op);
      return;
    }
    const IterVar& binding = op->node.as_or_throw<IterVar>();
    VortexThreadTag tag = ParseVortexThreadTag(binding->thread_tag);
    bool inserted = false;
    if (tag.is_thread) {
      inserted = thread_dependent_.insert(binding->var.get()).second;
    }
    VisitExpr(op->value);
    VisitStmt(op->body);
    if (inserted) thread_dependent_.erase(binding->var.get());
  }

  void VisitStmt_(const ForNode* op) final {
    if (op->thread_binding.has_value()) {
      const IterVar& binding = op->thread_binding.value();
      VortexThreadTag tag = ParseVortexThreadTag(binding->thread_tag);
      bool binding_inserted = false;
      bool loop_var_inserted = false;
      if (tag.is_thread) {
        binding_inserted = thread_dependent_.insert(binding->var.get()).second;
        loop_var_inserted = thread_dependent_.insert(op->loop_var.get()).second;
      }
      VisitExpr(op->min);
      VisitExpr(op->extent);
      if (op->step.has_value()) VisitExpr(op->step.value());
      VisitStmt(op->body);
      if (loop_var_inserted) thread_dependent_.erase(op->loop_var.get());
      if (binding_inserted) thread_dependent_.erase(binding->var.get());
      return;
    }

    bool thread_dependent_count = DependsOnThread(op->min) || DependsOnThread(op->extent) ||
                                  (op->step.has_value() && DependsOnThread(op->step.value()));
    if (thread_dependent_count) ++thread_dependent_serial_loop_depth_;
    tirx::StmtVisitor::VisitStmt_(op);
    if (thread_dependent_count) --thread_dependent_serial_loop_depth_;
  }

  void VisitStmt_(const WhileNode* op) final {
    ++unproven_loop_depth_;
    tirx::StmtVisitor::VisitStmt_(op);
    --unproven_loop_depth_;
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    bool thread_dependent = DependsOnThread(op->condition);
    if (thread_dependent) ++thread_dependent_condition_depth_;
    tirx::StmtVisitor::VisitStmt_(op);
    if (thread_dependent) --thread_dependent_condition_depth_;
  }

  void VisitStmt_(const ReturnNode* op) final {
    CheckEarlyExit("return");
    tirx::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BreakNode* op) final { CheckEarlyExit("break"); }

  void VisitStmt_(const ContinueNode* op) final { CheckEarlyExit("continue"); }

  void VisitStmt_(const AssertStmtNode* op) final {
    if (DependsOnThread(op->condition)) {
      TVM_FFI_THROW(ValueError)
          << "CodeGenVortex: shared barrier cannot be combined with a thread-dependent early "
             "exit (assert)";
    }
    CheckEarlyExit("assert");
    tirx::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    const auto* call = op->value.as<CallNode>();
    if (call != nullptr &&
        (call->op.same_as(builtin::thread_return()) || call->op.same_as(builtin::break_loop()) ||
         call->op.same_as(builtin::continue_loop()))) {
      const char* kind = call->op.same_as(builtin::thread_return())
                             ? "thread return"
                             : (call->op.same_as(builtin::break_loop()) ? "break" : "continue");
      CheckEarlyExit(kind);
    } else if (call != nullptr && call->op.same_as(builtin::tvm_storage_sync()) &&
               !call->args.empty()) {
      const auto* scope = call->args[0].as<StringImmNode>();
      if (scope != nullptr && scope->value == "shared") {
        TVM_FFI_CHECK_EQ(thread_dependent_condition_depth_, 0, ValueError)
            << "CodeGenVortex: shared barrier is under a thread-dependent condition";
        TVM_FFI_CHECK_EQ(thread_dependent_serial_loop_depth_, 0, ValueError)
            << "CodeGenVortex: shared barrier is inside a thread-dependent serial loop";
        TVM_FFI_CHECK_EQ(unproven_loop_depth_, 0, ValueError)
            << "CodeGenVortex: shared barrier is inside a loop whose uniform iteration count "
               "cannot be proven";
      }
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }

  bool has_shared_barrier_{false};
  std::unordered_set<const VarNode*> thread_dependent_;
  int thread_dependent_condition_depth_{0};
  int thread_dependent_serial_loop_depth_{0};
  int unproven_loop_depth_{0};
};

}  // namespace

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
         "}\n\n"
         "template <typename T>\n"
         "static inline T __tvm_vortex_max(T a, T b) {\n"
         "  return a > b ? a : b;\n"
         "}\n\n";
}

void CodeGenVortex::InitFuncState(const PrimFunc& func) {
  CodeGenC::InitFuncState(func);
  thread_axis_bound_.fill(false);
  block_axis_bound_.fill(false);
  thread_extents_.fill(1);
  launch_rank_ = 1;
  local_allocation_bytes_ = 0;
  shared_arena_base_var_.clear();
  for (const Var& param : func->params) {
    if (param->ty.as<PointerTypeNode>()) {
      alloc_storage_scope_[param.get()] = "global";
    } else if (const auto* buffer_type = param->ty.as<BufferTypeNode>()) {
      alloc_storage_scope_[param.get()] = "global";
      RegisterHandleType(param.get(), buffer_type->dtype);
    }
  }
}

void CodeGenVortex::PlanSharedMemory(const PrimFunc& func) {
  shared_allocation_offsets_.clear();
  static_shared_bytes_ = 0;
  SharedAllocationCollector collector;
  collector(func->body);

  uint64_t arena_alignment = 1;
  for (const AllocBufferNode* op : collector.allocations) {
    const ffi::String& buffer_name = op->buffer.name();
    uint64_t size = GetConstantAllocationBytes(op, "shared");
    uint64_t alignment = GetAllocationAlignment(op, "shared");
    uint64_t offset = AlignUpChecked(static_shared_bytes_, alignment, buffer_name);
    TVM_FFI_CHECK_LE(size, std::numeric_limits<uint64_t>::max() - offset, ValueError)
        << "CodeGenVortex: shared buffer " << buffer_name
        << " cumulative arena byte size overflows uint64";
    TVM_FFI_CHECK(shared_allocation_offsets_.emplace(op->buffer.get(), offset).second, ValueError)
        << "CodeGenVortex: shared buffer " << buffer_name << " is allocated more than once";
    static_shared_bytes_ = offset + size;
    arena_alignment = std::max(arena_alignment, alignment);
  }
  static_shared_bytes_ = AlignUpChecked(static_shared_bytes_, arena_alignment, "shared arena");
  TVM_FFI_CHECK_LE(static_shared_bytes_, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   ValueError)
      << "CodeGenVortex: static shared-memory arena does not fit int64 metadata";
}

void CodeGenVortex::PreFunctionBody(const PrimFunc& func) {
  CodeGenC::PreFunctionBody(func);
  if (static_shared_bytes_ == 0) return;
  shared_arena_base_var_ = name_supply_->FreshName("__tvm_vortex_shared_base");
  PrintIndent();
  stream << "uint8_t* " << shared_arena_base_var_ << " = static_cast<uint8_t*>(__local_mem("
         << static_shared_bytes_ << "));\n";
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

void CodeGenVortex::PrintStorageScope(const std::string& scope, std::ostream&) {  // NOLINT(*)
  TVM_FFI_CHECK(scope == "global" || scope == "local" || scope == "shared", ValueError)
      << "CodeGenVortex: storage scope " << scope << " is not supported";
}

void CodeGenVortex::PrintStorageSync(const CallNode* op) {
  TVM_FFI_CHECK_GE(op->args.size(), 1, ValueError)
      << "CodeGenVortex: tvm_storage_sync requires a storage scope argument";
  const auto* scope_imm = op->args[0].as<StringImmNode>();
  TVM_FFI_CHECK(scope_imm != nullptr, ValueError)
      << "CodeGenVortex: tvm_storage_sync storage scope must be a compile-time string";
  const std::string& scope = scope_imm->value;
  TVM_FFI_CHECK_EQ(scope, "shared", ValueError)
      << "CodeGenVortex: storage sync scope " << scope
      << " is not supported; only block-wide shared synchronization is available";
  PrintIndent();
  stream << "__syncthreads();\n";
}

void CodeGenVortex::BindThreadIndex(const IterVar& iv) {
  VortexThreadTag tag = ParseVortexThreadTag(iv->thread_tag);
  TVM_FFI_CHECK(!var_idmap_.count(iv->var.get()), ValueError)
      << "CodeGenVortex: thread variable is bound more than once: " << iv->thread_tag;
  std::array<bool, 3>& bound_axes = tag.is_thread ? thread_axis_bound_ : block_axis_bound_;
  TVM_FFI_CHECK(!bound_axes[tag.axis], ValueError)
      << "CodeGenVortex: " << iv->thread_tag << " axis is bound more than once";
  bound_axes[tag.axis] = true;
  var_idmap_[iv->var.get()] =
      CastFromTo(std::string(iv->thread_tag), PrimType::UInt(32), iv->var.ty());
}

void CodeGenVortex::ValidateBindingExtent(const IterVar& iv, const PrimExpr& extent_expr) {
  VortexThreadTag tag = ParseVortexThreadTag(iv->thread_tag);
  launch_rank_ = std::max(launch_rank_, static_cast<uint32_t>(tag.axis + 1));
  if (!tag.is_thread) {
    if (const auto* extent = extent_expr.as<IntImmNode>()) {
      TVM_FFI_CHECK(extent->value > 0 && static_cast<uint64_t>(extent->value) <=
                                             std::numeric_limits<uint32_t>::max(),
                    ValueError)
          << "CodeGenVortex: " << iv->thread_tag << " extent must be a positive uint32, but got "
          << extent->value;
    }
    return;
  }
  const auto* extent = extent_expr.as<IntImmNode>();
  TVM_FFI_CHECK(extent != nullptr, ValueError)
      << "CodeGenVortex: " << iv->thread_tag << " extent must be a compile-time constant";
  TVM_FFI_CHECK_GT(extent->value, 0, ValueError)
      << "CodeGenVortex: " << iv->thread_tag << " extent must be positive";

  const char* axis_limit_name = ThreadAxisLimitName(tag.axis);
  int64_t axis_limit = target_->GetAttr<int64_t>(axis_limit_name).value();
  TVM_FFI_CHECK_LE(extent->value, axis_limit, ValueError)
      << "CodeGenVortex: " << iv->thread_tag << " extent " << extent->value << " exceeds "
      << axis_limit_name << " " << axis_limit;

  int64_t max_threads = target_->GetAttr<int64_t>("max_threads_per_block").value();
  uint64_t max_threads_u64 = static_cast<uint64_t>(max_threads);
  thread_extents_[tag.axis] = static_cast<uint64_t>(extent->value);
  uint64_t product = 1;
  for (uint64_t axis_extent : thread_extents_) {
    if (axis_extent > max_threads_u64 / product) {
      if (axis_extent <= std::numeric_limits<uint64_t>::max() / product) {
        product *= axis_extent;
        TVM_FFI_THROW(ValueError) << "CodeGenVortex: thread block contains " << product
                                  << " threads, exceeding max_threads_per_block " << max_threads;
      }
      TVM_FFI_THROW(ValueError)
          << "CodeGenVortex: thread block product overflows uint64 and exceeds "
             "max_threads_per_block "
          << max_threads;
    }
    product *= axis_extent;
  }
}

void CodeGenVortex::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    const IterVar& binding = op->node.as_or_throw<IterVar>();
    ValidateBindingExtent(binding, op->value);
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenVortex::VisitStmt_(const ForNode* op) {
  if (!op->thread_binding.has_value()) {
    TVM_FFI_CHECK(op->kind == ForKind::kSerial, ValueError)
        << "CodeGenVortex: only serial and thread-bound loops are supported";
    CodeGenC::VisitStmt_(op);
    return;
  }
  TVM_FFI_CHECK(is_zero(op->min), ValueError)
      << "CodeGenVortex: thread-bound loops must have a zero minimum";
  const IterVar& binding = op->thread_binding.value();
  ValidateBindingExtent(binding, op->extent);
  BindThreadIndex(binding);
  if (!binding->var.same_as(op->loop_var)) {
    var_idmap_[op->loop_var.get()] =
        CastFromTo(std::string(binding->thread_tag), PrimType::UInt(32), op->loop_var.ty());
  }
  PrintStmt(op->body);
}

void CodeGenVortex::VisitStmt_(const AllocBufferNode* op) {
  TVM_FFI_CHECK(op->buffer.defined(), ValueError)
      << "CodeGenVortex: allocation must contain a defined buffer";
  const ffi::String& buffer_name = op->buffer.name();
  std::string scope = op->buffer.scope();
  if (scope == "shared") {
    auto allocation = shared_allocation_offsets_.find(op->buffer.get());
    TVM_FFI_ICHECK(allocation != shared_allocation_offsets_.end())
        << "Shared-memory allocation was not present in the precomputed arena plan";
    TVM_FFI_ICHECK(!shared_arena_base_var_.empty());
    alloc_storage_scope_[op->buffer.get()] = scope;
    std::string vid = AllocVarID(op->buffer.get(), buffer_name + "_ptr");
    PrintIndent();
    PrintType(op->buffer->dtype, stream);
    stream << "* " << vid << " = reinterpret_cast<";
    PrintType(op->buffer->dtype, stream);
    stream << "*>(" << shared_arena_base_var_ << " + " << allocation->second << ");\n";
    RegisterHandleType(op->buffer.get(), op->buffer->dtype);
    if (op->annotations.count(tirx::attr::kVolatile)) MarkVolatile(op->buffer.get());
    return;
  }
  TVM_FFI_CHECK_EQ(scope, "local", ValueError)
      << "CodeGenVortex: storage scope " << scope << " is not supported for buffer " << buffer_name;

  uint64_t constant_size = 1;
  for (const PrimExpr& dim : op->buffer->shape) {
    const auto* dim_imm = dim.as<IntImmNode>();
    TVM_FFI_CHECK(dim_imm != nullptr, ValueError) << "CodeGenVortex: local buffer " << buffer_name
                                                  << " must have a compile-time constant shape";
    TVM_FFI_CHECK_GT(dim_imm->value, 0, ValueError)
        << "CodeGenVortex: local buffer " << buffer_name << " extent " << dim_imm->value
        << " must be positive";
    uint64_t extent = static_cast<uint64_t>(dim_imm->value);
    TVM_FFI_CHECK_LE(extent, std::numeric_limits<uint64_t>::max() / constant_size, ValueError)
        << "CodeGenVortex: local buffer " << buffer_name << " shape product overflows uint64";
    constant_size *= extent;
  }

  PrimType dtype = op->buffer->dtype;
  TVM_FFI_CHECK(!dtype.IsScalableVector(), ValueError)
      << "CodeGenVortex: local buffer " << buffer_name << " cannot use a scalable-vector dtype";
  uint64_t dtype_bits = static_cast<uint64_t>(dtype.bits());
  uint64_t dtype_lanes = static_cast<uint64_t>(dtype.lanes());
  TVM_FFI_CHECK_GT(dtype_bits, 0, ValueError)
      << "CodeGenVortex: local buffer " << buffer_name << " dtype must have a positive bit width";
  TVM_FFI_CHECK_LE(dtype_lanes, std::numeric_limits<uint64_t>::max() / dtype_bits, ValueError)
      << "CodeGenVortex: local buffer " << buffer_name << " dtype size overflows uint64";
  uint64_t element_bits = dtype_bits * dtype_lanes;
  uint64_t element_bytes = (element_bits + 7) / 8;
  TVM_FFI_CHECK_LE(constant_size, std::numeric_limits<uint64_t>::max() / element_bytes, ValueError)
      << "CodeGenVortex: local buffer " << buffer_name << " byte size overflows uint64";
  uint64_t allocation_bytes = constant_size * element_bytes;

  uint64_t alignment = GetAllocationAlignment(op, "local");

  uint64_t aligned_total = AlignUpChecked(local_allocation_bytes_, alignment, buffer_name);
  TVM_FFI_CHECK_LE(allocation_bytes, std::numeric_limits<uint64_t>::max() - aligned_total,
                   ValueError)
      << "CodeGenVortex: local buffer " << buffer_name
      << " cumulative allocation byte size overflows uint64";
  uint64_t new_total = aligned_total + allocation_bytes;
  int64_t configured_limit = target_->GetAttr<int64_t>("max_local_memory_per_thread").value();
  TVM_FFI_CHECK_LE(new_total, static_cast<uint64_t>(configured_limit), ValueError)
      << "CodeGenVortex: local allocations require " << new_total
      << " bytes per thread, exceeding max_local_memory_per_thread " << configured_limit
      << " while allocating buffer " << buffer_name;
  local_allocation_bytes_ = new_total;

  alloc_storage_scope_[op->buffer.get()] = scope;
  std::string vid = AllocVarID(op->buffer.get(), buffer_name + "_ptr");
  PrintIndent();
  stream << "alignas(" << alignment << ") ";
  PrintType(dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  RegisterHandleType(op->buffer.get(), dtype);
  if (op->annotations.count(tirx::attr::kVolatile)) {
    MarkVolatile(op->buffer.get());
  }
}

void CodeGenVortex::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  os << "__tvm_vortex_max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

std::string CodeGenVortex::PrintTypeString(const Type& type) {
  std::ostringstream os;
  PrintType(type, os);
  return os.str();
}

void CodeGenVortex::EmitLaunchWrapper(const PrimFunc& func, uint32_t kernel_id) {
  stream << "static void __tvm_vortex_kernel_entry_" << kernel_id
         << "(void* opaque) {\n"
            "  const vx_tvm_launch_header_t* launch =\n"
            "      static_cast<const vx_tvm_launch_header_t*>(opaque);\n"
            "  const uint64_t* args = vx_tvm_launch_args(launch);\n"
         << "  __tvm_vortex_kernel_" << kernel_id << "(";
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
  stream << ");\n}\n\n";
}

void CodeGenVortex::FinishDispatcher() {
  TVM_FFI_CHECK(!kernels_.empty(), ValueError)
      << "CodeGenVortex: at least one launchable PrimFunc is required";
  stream << "int main() {\n"
            "  const vx_tvm_launch_header_t* launch =\n"
            "      reinterpret_cast<const vx_tvm_launch_header_t*>(csr_read(VX_CSR_MSCRATCH));\n"
            "  if (launch == nullptr || launch->abi_version != VX_TVM_ABI_VERSION) {\n"
            "    return -1;\n"
            "  }\n"
            "  switch (launch->kernel_id) {\n";
  for (size_t kernel_id = 0; kernel_id < kernels_.size(); ++kernel_id) {
    const KernelDispatchInfo& kernel = kernels_[kernel_id];
    stream << "    case " << kernel_id << "u:\n"
           << "      if (launch->num_args != " << kernel.num_args << "u) return -1;\n"
           << "      return vx_spawn_threads(3, launch->grid, launch->block,\n"
           << "                              (vx_kernel_func_cb)__tvm_vortex_kernel_entry_"
           << kernel_id << ", launch);\n";
  }
  stream << "    default:\n"
            "      return -1;\n"
            "  }\n"
            "}\n";
}

void CodeGenVortex::AddKernel(const GlobalVar& gvar, const PrimFunc& func, uint32_t kernel_id,
                              const std::string& global_symbol) {
  TVM_FFI_CHECK_EQ(kernel_id, kernels_.size(), ValueError)
      << "CodeGenVortex: kernel IDs must be dense and emitted in order";
  stream << "// Vortex kernel " << kernel_id << ": " << global_symbol << "\n";
  PrimFunc kernel = WithAttr(func, tvm::attr::kGlobalSymbol,
                             ffi::String("__tvm_vortex_kernel_" + std::to_string(kernel_id)));
  BarrierUniformityValidator barrier_validator(kernel);
  barrier_validator.Validate(kernel);
  PlanSharedMemory(kernel);
  CodeGenC::AddFunction(gvar, kernel);
  uint64_t block_threads = 1;
  for (uint64_t extent : thread_extents_) {
    TVM_FFI_ICHECK_LE(block_threads, std::numeric_limits<uint64_t>::max() / extent);
    block_threads *= extent;
  }
  TVM_FFI_CHECK_LE(block_threads, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   ValueError)
      << "CodeGenVortex: thread block size does not fit resource metadata";
  TVM_FFI_CHECK_LE(local_allocation_bytes_,
                   static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), ValueError)
      << "CodeGenVortex: private local byte count does not fit resource metadata";
  backend::vortex::ValidateVortexSharedMemoryUsage(target_, static_cast<int64_t>(block_threads),
                                                   static_cast<int64_t>(static_shared_bytes_));
  backend::vortex::VortexBlockResourceUsage usage =
      backend::vortex::GetVortexBlockResourceUsage(target_, static_cast<int64_t>(block_threads));
  EmitLaunchWrapper(kernel, kernel_id);
  kernels_.push_back({func->params.size()});
  kernel_resources_.push_back(
      {launch_rank_, static_shared_bytes_, static_cast<uint64_t>(usage.resident_groups),
       local_allocation_bytes_, thread_extents_, barrier_validator.has_shared_barrier()});
}

}  // namespace codegen
}  // namespace tvm
