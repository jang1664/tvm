#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly build_dir="${TVM_BUILD_DIR:-${script_dir}/build}"
readonly default_llvm_config="/opt/conda-pkgs/llvmdev-18.1.8-default_h99862b1_12/bin/llvm-config"
readonly host_library_dir="${TVM_HOST_LIBRARY_DIR:-/opt/anaconda3/lib}"

llvm_config="${TVM_LLVM_CONFIG:-${default_llvm_config}}"
if [[ "${llvm_config}" != */* ]]; then
  llvm_config="$(command -v -- "${llvm_config}" || true)"
fi

if [[ -z "${llvm_config}" || ! -x "${llvm_config}" ]]; then
  echo "Error: TVM_LLVM_CONFIG must name an executable llvm-config." >&2
  echo "Default: ${default_llvm_config}" >&2
  exit 1
fi

readonly llvm_bin="$(cd -- "$(dirname -- "${llvm_config}")" && pwd)"
llvm_config="${llvm_bin}/${llvm_config##*/}"

# llvm-config may invoke companion LLVM tools, so sanitize PATH here as well
# as in build.sh.  This also makes standalone config generation deterministic.
export PATH="${llvm_bin}:/usr/bin:/bin"

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

if [[ ! -e "${host_library_dir}/libxml2.so" ]]; then
  echo "Error: LLVM's libxml2 dependency was not found in ${host_library_dir}." >&2
  echo "Set TVM_HOST_LIBRARY_DIR to a directory containing libxml2.so." >&2
  exit 1
fi

/usr/bin/mkdir -p "${build_dir}"

# Regenerate only config.cmake.  Preserve every other build artifact so this
# script remains safe for incremental builds.
/usr/bin/cp "${script_dir}/cmake/config.cmake" "${build_dir}/config.cmake"

{
  printf 'link_directories("%s")\n' "${host_library_dir}"
  echo 'set(CMAKE_BUILD_TYPE Debug)'
  echo 'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)'

  # Use an absolute llvm-config path.  USE_LLVM=ON would rediscover whichever
  # LLVM happens to come first in the caller's PATH.
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
  echo 'set(USE_DNNL OFF)'
  echo 'set(USE_IMCFLOW ON)'
  echo 'set(USE_MICRO ON)'
} >> "${build_dir}/config.cmake"

echo "Configured TVM with LLVM ${llvm_version}: ${llvm_config}"
echo "Sanitized host PATH: ${PATH}"
echo "Host link library directory: ${host_library_dir}"
