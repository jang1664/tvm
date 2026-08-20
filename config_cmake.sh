#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

if [[ "${PWD}" != "${build_dir}" ]]; then
  echo "Error: run this script from ${build_dir}" >&2
  exit 1
fi

# Never let the Vortex-specific LLVM leak into a TVM build.  Prefer an
# explicitly selected llvm-config, then the active Conda environment, then a
# versioned system installation.
llvm_config="${TVM_LLVM_CONFIG:-}"
if [[ -z "${llvm_config}" && -n "${CONDA_PREFIX:-}" && \
      -x "${CONDA_PREFIX}/bin/llvm-config" ]]; then
  llvm_config="${CONDA_PREFIX}/bin/llvm-config"
fi

if [[ -z "${llvm_config}" ]]; then
  for candidate in \
    /usr/bin/llvm-config-21 \
    /usr/bin/llvm-config-20 \
    /usr/bin/llvm-config-19 \
    /usr/bin/llvm-config-18 \
    /usr/bin/llvm-config-17 \
    /usr/bin/llvm-config-16 \
    /usr/bin/llvm-config-15 \
    /usr/bin/llvm-config; do
    if [[ -x "${candidate}" ]]; then
      llvm_config="${candidate}"
      break
    fi
  done
fi

if [[ -z "${llvm_config}" || ! -x "${llvm_config}" ]]; then
  echo "Error: a non-Vortex LLVM 15+ installation is required." >&2
  echo "Set TVM_LLVM_CONFIG to its llvm-config executable." >&2
  exit 1
fi

llvm_prefix="$("${llvm_config}" --prefix)"
llvm_version="$("${llvm_config}" --version)"
llvm_major="${llvm_version%%.*}"

if [[ "${llvm_config}" == /opt/vortex/llvm-vortex/* || \
      "${llvm_prefix}" == /opt/vortex/llvm-vortex* ]]; then
  echo "Error: refusing Vortex LLVM for the TVM host build: ${llvm_config}" >&2
  exit 1
fi

if [[ ! "${llvm_major}" =~ ^[0-9]+$ || "${llvm_major}" -lt 15 ]]; then
  echo "Error: TVM requires LLVM 15+, found ${llvm_version}." >&2
  exit 1
fi

# Regenerate only config.cmake.  Preserve every other build artifact so this
# script is safe to rerun for an incremental build.
cp "${script_dir}/cmake/config.cmake" "${build_dir}/config.cmake"

{
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    printf 'include_directories("%s/include")\n' "${CONDA_PREFIX}"
    printf 'list(APPEND CMAKE_LIBRARY_PATH "%s/lib")\n' "${CONDA_PREFIX}"
  fi

  # Controls default compilation flags: Release, Debug, or RelWithDebInfo.
  echo 'set(CMAKE_BUILD_TYPE Debug)'
  echo 'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)'

  # Use an absolute llvm-config path; USE_LLVM=ON would auto-detect the
  # Vortex-specific LLVM from PATH on this server.
  printf 'set(USE_LLVM "%s")\n' "${llvm_config}"
  echo 'set(HIDE_PRIVATE_SYMBOLS ON)'

  # GPU SDKs.
  echo 'set(USE_CUDA OFF)'
  echo 'set(USE_METAL OFF)'
  echo 'set(USE_VULKAN OFF)'
  echo 'set(USE_OPENCL OFF)'
  echo 'set(USE_CUBLAS OFF)'
  echo 'set(USE_CUDNN OFF)'
  echo 'set(USE_CUTLASS OFF)'

  echo 'set(USE_GTEST OFF)'
  echo 'set(USE_RELAY_DEBUG ON)'
  echo 'set(USE_VTA_FSIM ON)'
  echo 'set(USE_DNNL C_SRC)'
  echo 'set(USE_IMCFLOW ON)'
  echo 'set(USE_MICRO ON)'
} >> "${build_dir}/config.cmake"

echo "Configured TVM with LLVM ${llvm_version}: ${llvm_config}"
