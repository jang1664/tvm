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
"""Compile generated Vortex C++ kernels into uploadable ``vxbin`` images.

The command line intentionally mirrors ``tests/regression/common.mk`` in the
Vortex repository.  Compiler and sysroot paths are resolved explicitly; the
host shell's compiler selection and startup files are never consulted.
"""

import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tvm_ffi


_DEFAULT_LLVM_ROOT = Path("/opt/vortex/llvm-vortex")
_DEFAULT_LP64F_PROFILE = Path("/opt/vortex_profiles/rv64imaf_zfh_lp64f")


class VortexCompileError(RuntimeError):
    """Error raised when a Vortex compilation command fails."""


@dataclass(frozen=True)
class VortexCompileConfig:
    """Fully resolved paths and target settings for Vortex compilation."""

    vortex_home: Path
    llvm_root: Path
    profile_root: Path
    toolchain_root: Path
    sysroot: Path
    libc_root: Path
    libcrt_root: Path
    xlen: int
    mtriple: str
    mcpu: str | None
    mattr: tuple[str, ...]
    march: str
    mabi: str
    startup_addr: int
    num_warps: int
    thread_warp_size: int


def _first_setting(value, environment_names, default=None):
    if value is not None:
        return value
    for name in environment_names:
        if setting := os.environ.get(name):
            return setting
    return default


def _as_path(value, description):
    if value is None:
        raise ValueError(description)
    return Path(value).expanduser().resolve()


def resolve_vortex_compile_config(
    target=None,
    *,
    vortex_home=None,
    llvm_root=None,
    profile_root=None,
    toolchain_root=None,
    sysroot=None,
    libc_root=None,
    libcrt_root=None,
    march=None,
    mabi=None,
    startup_addr=None,
):
    """Resolve the complete Vortex compiler configuration.

    ``vortex_home`` is deliberately required through either the argument,
    ``TVM_VORTEX_HOME``, or ``VORTEX_HOME``.  The installed LLVM-Vortex and
    pinned fpint LP64F profile use their deterministic machine locations by
    default.  They can be overridden with ``TVM_VORTEX_LLVM_ROOT``,
    ``TVM_VORTEX_PROFILE_ROOT``, ``TVM_VORTEX_TOOLCHAIN_ROOT``,
    ``TVM_VORTEX_SYSROOT``, ``TVM_VORTEX_LIBC_ROOT``, and
    ``TVM_VORTEX_LIBCRT_ROOT``.  Legacy Vortex build variables are also
    accepted, but no executable is ever resolved from the ambient ``PATH``.
    """

    xlen = int(getattr(target, "xlen", 64))
    if xlen not in (32, 64):
        raise ValueError(f"Vortex xlen must be 32 or 64, but got {xlen}")

    vortex_home = _as_path(
        _first_setting(vortex_home, ("TVM_VORTEX_HOME", "VORTEX_HOME")),
        "Vortex repository root is required; set TVM_VORTEX_HOME or pass vortex_home",
    )
    llvm_root = _as_path(
        _first_setting(
            llvm_root,
            ("TVM_VORTEX_LLVM_ROOT", "LLVM_VORTEX"),
            _DEFAULT_LLVM_ROOT,
        ),
        "LLVM-Vortex root is required",
    )

    default_profile = _DEFAULT_LP64F_PROFILE if xlen == 64 else None
    profile_root = _as_path(
        _first_setting(
            profile_root,
            ("TVM_VORTEX_PROFILE_ROOT", "VORTEX_LP64F_PROFILE"),
            default_profile,
        ),
        f"A Vortex runtime profile is required for xlen={xlen}; set TVM_VORTEX_PROFILE_ROOT",
    )
    prefix = f"riscv{xlen}-unknown-elf"
    toolchain_root = _as_path(
        _first_setting(
            toolchain_root,
            ("TVM_VORTEX_TOOLCHAIN_ROOT", "RISCV_TOOLCHAIN_PATH"),
            profile_root / f"riscv{xlen}-gnu-toolchain",
        ),
        "Vortex RISC-V toolchain root is required",
    )
    sysroot = _as_path(
        _first_setting(
            sysroot,
            ("TVM_VORTEX_SYSROOT", "RISCV_SYSROOT"),
            toolchain_root / prefix,
        ),
        "Vortex RISC-V sysroot is required",
    )
    libc_root = _as_path(
        _first_setting(
            libc_root,
            ("TVM_VORTEX_LIBC_ROOT", "LIBC_VORTEX"),
            profile_root / f"libc{xlen}",
        ),
        "Vortex libc root is required",
    )
    libcrt_root = _as_path(
        _first_setting(
            libcrt_root,
            ("TVM_VORTEX_LIBCRT_ROOT", "LIBCRT_VORTEX"),
            profile_root / f"libcrt{xlen}",
        ),
        "Vortex compiler-rt root is required",
    )

    mtriple = str(getattr(target, "mtriple", prefix))
    if not mtriple.startswith(f"riscv{xlen}"):
        raise ValueError(f"Vortex mtriple {mtriple!r} does not match xlen={xlen}")

    if march is None:
        march = "rv64imaf_zfh" if xlen == 64 else "rv32imaf"
    if mabi is None:
        mabi = "lp64f" if xlen == 64 else "ilp32f"
    if startup_addr is None:
        startup_addr = 0x180000000 if xlen == 64 else 0x80000000

    return VortexCompileConfig(
        vortex_home=vortex_home,
        llvm_root=llvm_root,
        profile_root=profile_root,
        toolchain_root=toolchain_root,
        sysroot=sysroot,
        libc_root=libc_root,
        libcrt_root=libcrt_root,
        xlen=xlen,
        mtriple=mtriple,
        mcpu=None if getattr(target, "mcpu", None) is None else str(target.mcpu),
        mattr=tuple(str(value) for value in (getattr(target, "mattr", None) or ())),
        march=str(march),
        mabi=str(mabi),
        startup_addr=int(startup_addr),
        num_warps=int(getattr(target, "num_warps", 4)),
        thread_warp_size=int(getattr(target, "thread_warp_size", 32)),
    )


def _require_file(path, description):
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")


def _require_directory(path, description):
    if not path.is_dir():
        raise ValueError(f"{description} does not exist: {path}")


def _validate_config(config):
    _require_file(config.llvm_root / "bin/clang++", "LLVM-Vortex clang++")
    _require_file(config.llvm_root / "bin/llvm-objcopy", "LLVM-Vortex llvm-objcopy")
    _require_directory(config.toolchain_root, "Vortex RISC-V toolchain root")
    _require_directory(config.sysroot, "Vortex RISC-V sysroot")
    _require_file(config.vortex_home / "kernel/libvortex.a", "Vortex kernel runtime library")
    abi_header = config.vortex_home / "kernel/include/vx_tvm_abi.h"
    _require_file(abi_header, "Vortex TVM ABI header")
    match = re.search(
        r"^#define\s+VX_TVM_ABI_VERSION\s+(\d+)[uU]?\s*$",
        abi_header.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Vortex TVM ABI version is missing from {abi_header}")
    runtime_abi = tvm_ffi.get_global_func("runtime.vortex_abi_version", allow_missing=True)
    if runtime_abi is None:
        raise ValueError("Vortex runtime sidecar is not loaded; cannot validate the device ABI")
    if int(match.group(1)) != int(runtime_abi()):
        raise ValueError(
            f"Vortex compiler ABI {match.group(1)} does not match runtime ABI {runtime_abi()}"
        )
    _require_file(
        config.vortex_home / f"kernel/scripts/link{config.xlen}.ld",
        "Vortex kernel linker script",
    )
    _require_file(config.vortex_home / "kernel/scripts/vxbin.py", "Vortex vxbin packager")
    _require_file(config.libc_root / "lib/libc.a", "Vortex profile libc")
    _require_file(config.libc_root / "lib/libm.a", "Vortex profile libm")
    _require_file(
        config.libcrt_root
        / f"lib/baremetal/libclang_rt.builtins-riscv{config.xlen}.a",
        "Vortex profile compiler-rt builtins",
    )
    _require_file(Path("/usr/bin/readelf"), "host ELF reader used by vxbin.py")


def _command_environment(config):
    """Return a minimal deterministic environment for compiler subprocesses."""

    return {
        "LC_ALL": "C",
        "PATH": ":".join(
            [
                str(config.llvm_root / "bin"),
                str(config.toolchain_root / "bin"),
                "/usr/bin",
                "/bin",
            ]
        ),
    }


def _run_command(command, *, stage, environment, cwd):
    timeout_text = os.environ.get("TVM_VORTEX_COMPILE_TIMEOUT_SECONDS", "300")
    try:
        timeout = float(timeout_text)
    except ValueError as error:
        raise ValueError("TVM_VORTEX_COMPILE_TIMEOUT_SECONDS must be positive") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("TVM_VORTEX_COMPILE_TIMEOUT_SECONDS must be positive")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise VortexCompileError(
            f"Vortex {stage} timed out after {timeout:g} seconds\n"
            f"Command: {shlex.join(command)}\n"
            f"stdout:\n{error.stdout or ''}\n"
            f"stderr:\n{error.stderr or ''}"
        ) from error
    except OSError as error:
        raise VortexCompileError(
            f"Vortex {stage} could not start\nCommand: {shlex.join(command)}\nError: {error}"
        ) from error

    if result.returncode != 0:
        raise VortexCompileError(
            f"Vortex {stage} failed with exit code {result.returncode}\n"
            f"Command: {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _compile_command(config, source_path, elf_path):
    linker_script = config.vortex_home / f"kernel/scripts/link{config.xlen}.ld"
    libvortex = config.vortex_home / "kernel/libvortex.a"
    builtins = (
        config.libcrt_root
        / f"lib/baremetal/libclang_rt.builtins-riscv{config.xlen}.a"
    )
    command = [
        str(config.llvm_root / "bin/clang++"),
        f"--target={config.mtriple}",
        f"--sysroot={config.sysroot}",
        f"--gcc-toolchain={config.toolchain_root}",
        "-Xclang",
        "-target-feature",
        "-Xclang",
        "+vortex",
        "-Xclang",
        "-target-feature",
        "-Xclang",
        "+zicond",
        "-mllvm",
        "-disable-loop-idiom-all",
        f"-march={config.march}",
        f"-mabi={config.mabi}",
        "-O3",
        "-mcmodel=medany",
        "-fno-rtti",
        "-fno-exceptions",
        "-nostartfiles",
        "-nostdlib",
        "-fdata-sections",
        "-ffunction-sections",
        f"-I{config.vortex_home / 'kernel/include'}",
        f"-I{config.vortex_home / 'hw'}",
        f"-I{config.vortex_home / 'sim/common'}",
        f"-DXLEN_{config.xlen}",
        f"-DNUM_WARPS={config.num_warps}",
        f"-DNUM_THREADS={config.thread_warp_size}",
        "-DNDEBUG",
    ]
    if config.mcpu:
        command.append(f"-mcpu={config.mcpu}")
    for feature in config.mattr:
        command.extend(["-Xclang", "-target-feature", "-Xclang", feature])
    if config.mabi.endswith("f"):
        command.append("-DEXT_D_DISABLE")
    if "zfh" in config.march:
        command.append("-DEXT_ZFH_ENABLE")
    command.extend(
        [
            str(source_path),
            "-Wl,-Bstatic,--gc-sections,"
            f"-T,{linker_script},--defsym=STARTUP_ADDR={hex(config.startup_addr)}",
            str(libvortex),
            f"-L{config.libc_root / 'lib'}",
            "-lm",
            "-lc",
            str(builtins),
            "-o",
            str(elf_path),
        ]
    )
    return command


def compile_vortex(code, target=None, **config_overrides):
    """Compile Vortex C++ source and return an uploadable ``vxbin`` bytearray.

    Parameters are accepted as keyword overrides by
    :func:`resolve_vortex_compile_config`.  Failures include the complete argv,
    stdout, and stderr for the failing compiler or packaging stage.
    """

    config = resolve_vortex_compile_config(target, **config_overrides)
    _validate_config(config)
    environment = _command_environment(config)

    with tempfile.TemporaryDirectory(prefix="tvm-vortex-") as temp_directory:
        temp_directory = Path(temp_directory)
        source_path = temp_directory / "kernel.cpp"
        elf_path = temp_directory / "kernel.elf"
        vxbin_path = temp_directory / "kernel.vxbin"
        source_path.write_text(str(code), encoding="utf-8")

        _run_command(
            _compile_command(config, source_path, elf_path),
            stage="kernel compilation",
            environment=environment,
            cwd=temp_directory,
        )

        package_environment = dict(environment)
        package_environment["OBJCOPY"] = str(config.llvm_root / "bin/llvm-objcopy")
        _run_command(
            [
                sys.executable,
                str(config.vortex_home / "kernel/scripts/vxbin.py"),
                str(elf_path),
                str(vxbin_path),
            ],
            stage="vxbin packaging",
            environment=package_environment,
            cwd=temp_directory,
        )
        return bytearray(vxbin_path.read_bytes())


@tvm_ffi.register_global_func("tvm_callback_vortex_compile")
def tvm_callback_vortex_compile(code, target=None):
    """TVM global callback used by ``target.build.vortex``."""

    return compile_vortex(code, target)
