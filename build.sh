#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly source_dir="${script_dir}"
readonly build_dir="${TVM_BUILD_DIR:-${source_dir}/build}"
readonly default_llvm_config="/opt/conda-pkgs/llvmdev-18.1.8-default_h99862b1_12/bin/llvm-config"
readonly python_bin="${TVM_PYTHON:-/usr/bin/python3}"
readonly python_build_root="${source_dir}/.local/python-build"
readonly python_runtime_root="${source_dir}/.local/python"
readonly python_ffi_build_dir="${build_dir}/python-tvm-ffi"

llvm_config="${TVM_LLVM_CONFIG:-${default_llvm_config}}"
if [[ "${llvm_config}" != */* ]]; then
  llvm_config="$(command -v -- "${llvm_config}" || true)"
fi

if [[ -z "${llvm_config}" || ! -x "${llvm_config}" ]]; then
  echo "Error: TVM_LLVM_CONFIG must name an executable llvm-config." >&2
  exit 1
fi

readonly llvm_bin="$(cd -- "$(dirname -- "${llvm_config}")" && pwd)"
llvm_config="${llvm_bin}/${llvm_config##*/}"
export TVM_LLVM_CONFIG="${llvm_config}"

# Make the host-tool selection independent of the interactive shell.  In
# particular, do not allow /opt/vortex/llvm-vortex from ~/.zshrc to leak into
# the TVM host build.
export PATH="${llvm_bin}:/usr/bin:/bin"

if [[ ! -x "${python_bin}" ]]; then
  echo "Error: TVM_PYTHON must name an executable Python interpreter." >&2
  exit 1
fi

if ! PYTHONPATH="${python_build_root}" "${python_bin}" -c \
  'import Cython; assert tuple(map(int, Cython.__version__.split(".")[:3])) >= (3, 2, 8)' \
  >/dev/null 2>&1; then
  echo "Error: Cython 3.2.8+ is required to build tvm_ffi.core." >&2
  echo "Install it locally with:" >&2
  echo "  ${python_bin} -m pip install --target ${python_build_root} 'cython>=3.2.8'" >&2
  exit 1
fi

export PYTHONPATH="${python_build_root}${PYTHONPATH:+:${PYTHONPATH}}"

/usr/bin/mkdir -p "${build_dir}"
/usr/bin/bash "${source_dir}/config_cmake.sh"

/usr/bin/cmake -S "${source_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DUSE_LLVM="${TVM_LLVM_CONFIG}"

/usr/bin/cmake --build "${build_dir}" --parallel "${TVM_BUILD_JOBS:-32}"

# tvm-ffi intentionally skips its Python module when included as a CMake
# subproject.  Build that module in its own directory, then install it beside a
# copied pure-Python package.  This avoids editable installs, which can make one
# TVM worktree silently import another worktree's extension.
/usr/bin/cmake -S "${source_dir}/3rdparty/tvm-ffi" -B "${python_ffi_build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DPython_EXECUTABLE="${python_bin}" \
  -DTVM_FFI_BUILD_PYTHON_MODULE=ON \
  -DTVM_FFI_BUILD_TESTS=OFF
/usr/bin/cmake --build "${python_ffi_build_dir}" \
  --parallel "${TVM_BUILD_JOBS:-32}" --target tvm_ffi_cython

/usr/bin/mkdir -p "${python_runtime_root}"
/usr/bin/cp -a "${source_dir}/3rdparty/tvm-ffi/python/tvm_ffi" "${python_runtime_root}/"
/usr/bin/cmake --install "${python_ffi_build_dir}" --prefix "${python_runtime_root}/tvm_ffi"

printf '\nTVM build complete. Source these optional runtime settings in your current shell:\n'
printf 'export TVM_HOME=%q\n' "${source_dir}"
printf 'export PYTHONPATH="${TVM_HOME}/python:${TVM_HOME}/.local/python${PYTHONPATH:+:${PYTHONPATH}}"\n'
printf 'export TVM_LIBRARY_PATH=%q\n' "${build_dir}/lib"
printf 'export LD_LIBRARY_PATH="${TVM_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\n'
