# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_VORTEX)
  if(${USE_VORTEX} MATCHES ${IS_TRUE_PATTERN})
    message(FATAL_ERROR
      "USE_VORTEX must be an explicit Vortex repository path, not ON")
  endif()

  get_filename_component(VORTEX_ROOT "${USE_VORTEX}" ABSOLUTE)
  set(VORTEX_INCLUDE_DIR "${VORTEX_ROOT}/runtime/include")
  set(VORTEX_ABI_INCLUDE_DIR "${VORTEX_ROOT}/kernel/include")
  set(VORTEX_RUNTIME_LIBRARY "${VORTEX_ROOT}/runtime/libvortex.so")

  if(NOT EXISTS "${VORTEX_INCLUDE_DIR}/vortex.h")
    message(FATAL_ERROR "Vortex runtime header not found: ${VORTEX_INCLUDE_DIR}/vortex.h")
  endif()
  if(NOT EXISTS "${VORTEX_ABI_INCLUDE_DIR}/vx_tvm_abi.h")
    message(FATAL_ERROR "Vortex TVM ABI header not found: ${VORTEX_ABI_INCLUDE_DIR}/vx_tvm_abi.h")
  endif()
  if(NOT EXISTS "${VORTEX_RUNTIME_LIBRARY}")
    message(FATAL_ERROR "Vortex runtime library not found: ${VORTEX_RUNTIME_LIBRARY}")
  endif()

  message(STATUS "Build with Vortex runtime: ${VORTEX_ROOT}")
  tvm_file_glob(GLOB RUNTIME_VORTEX_SRCS src/backend/vortex/runtime/*.cc)

  add_library(tvm_runtime_vortex_objs OBJECT ${RUNTIME_VORTEX_SRCS})
  target_include_directories(tvm_runtime_vortex_objs PRIVATE
    "${VORTEX_INCLUDE_DIR}"
    "${VORTEX_ABI_INCLUDE_DIR}"
    "${VORTEX_ROOT}/hw"
  )
  target_link_libraries(tvm_runtime_vortex_objs PUBLIC tvm_ffi_header)
  set_target_properties(tvm_runtime_vortex_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TVM_VISIBILITY_FLAG)
    target_compile_options(tvm_runtime_vortex_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
  endif()

  add_library(tvm_runtime_vortex SHARED $<TARGET_OBJECTS:tvm_runtime_vortex_objs>)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_vortex)
  target_link_libraries(tvm_runtime_vortex PUBLIC tvm_runtime "${VORTEX_RUNTIME_LIBRARY}")
  set_target_properties(tvm_runtime_vortex PROPERTIES
    BUILD_RPATH "${VORTEX_ROOT}/runtime"
    INSTALL_RPATH "${VORTEX_ROOT}/runtime"
  )
  tvm_configure_target_library(tvm_runtime_vortex RUNTIME_MODULE)
endif()
