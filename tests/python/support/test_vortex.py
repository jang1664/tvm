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

import struct
import subprocess
from pathlib import Path

import pytest
import tvm
from tvm.script import tirx as T
from tvm.support import vortex


VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
VORTEX_BUILD_DIR = VORTEX_HOME / "build"
LLVM_ROOT = Path("/opt/vortex/llvm-vortex")
PROFILE_ROOT = Path("/opt/vortex_profiles/rv64imaf_zfh_lp64f")


@T.prim_func
def vecadd(
    a: T.Buffer((256,), "int32"),
    b: T.Buffer((256,), "int32"),
    c: T.Buffer((256,), "int32"),
):
    T.func_attr({"global_symbol": "vecadd", "tirx.noalias": True})
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            if bx * 128 + tx < 256:
                c[bx * 128 + tx] = a[bx * 128 + tx] + b[bx * 128 + tx]


def _have_pinned_toolchain():
    return all(
        path.exists()
        for path in [
            VORTEX_BUILD_DIR / "kernel/libvortex.a",
            VORTEX_HOME / "kernel/scripts/link64.ld",
            VORTEX_HOME / "kernel/scripts/vxbin.py",
            LLVM_ROOT / "bin/clang++",
            LLVM_ROOT / "bin/llvm-objcopy",
            PROFILE_ROOT / "libc64/lib/libc.a",
            PROFILE_ROOT / "libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a",
        ]
    )


def _pinned_compile_kwargs():
    return {
        "vortex_home": VORTEX_HOME,
        "llvm_root": LLVM_ROOT,
        "profile_root": PROFILE_ROOT,
    }


def _generated_source(target):
    callback_name = "tvm_callback_vortex_compile"
    previous = tvm.get_global_func(callback_name)
    tvm.register_global_func(
        callback_name, lambda source, unused_target: bytearray(), override=True
    )
    try:
        mod = tvm.IRModule({"vecadd": vecadd})
        return tvm.get_global_func("target.build.vortex")(mod, target).inspect_source()
    finally:
        tvm.register_global_func(callback_name, previous, override=True)


def test_callback_is_registered_at_tvm_import():
    assert tvm.get_global_func("tvm_callback_vortex_compile") is not None


def test_requires_explicit_vortex_repository_root(monkeypatch):
    monkeypatch.delenv("TVM_VORTEX_HOME", raising=False)
    monkeypatch.delenv("VORTEX_HOME", raising=False)

    with pytest.raises(ValueError, match="TVM_VORTEX_HOME"):
        vortex.compile_vortex("int main() { return 0; }")


def test_rv32_configuration_is_xlen_aware(tmp_path):
    profile_root = tmp_path / "rv32-profile"
    target = tvm.target.Target({"kind": "vortex", "xlen": 32})
    config = vortex.resolve_vortex_compile_config(
        target,
        vortex_home=VORTEX_HOME,
        profile_root=profile_root,
    )

    assert config.mtriple == "riscv32-unknown-elf"
    assert config.march == "rv32imaf"
    assert config.mabi == "ilp32f"
    assert config.startup_addr == 0x80000000
    assert config.toolchain_root == profile_root / "riscv32-gnu-toolchain"
    assert config.build_dir == VORTEX_HOME / "build"


def test_build_directory_override_is_independent_from_source_root(
    monkeypatch, tmp_path
):
    build_dir = tmp_path / "vortex-out"
    monkeypatch.setenv("TVM_VORTEX_BUILD_DIR", str(build_dir))

    config = vortex.resolve_vortex_compile_config(
        tvm.target.Target("vortex"),
        vortex_home=VORTEX_HOME,
        profile_root=PROFILE_ROOT,
    )

    assert config.vortex_home == VORTEX_HOME
    assert config.build_dir == build_dir.resolve()


def test_target_cpu_and_features_reach_compiler_argv(tmp_path):
    target = tvm.target.Target(
        {"kind": "vortex", "mcpu": "generic-rv64", "mattr": ["+m", "-d"]}
    )
    config = vortex.resolve_vortex_compile_config(
        target,
        vortex_home=VORTEX_HOME,
        profile_root=PROFILE_ROOT,
    )
    command = vortex._compile_command(
        config, tmp_path / "kernel.cpp", tmp_path / "kernel.elf"
    )

    assert "-mcpu=generic-rv64" in command
    assert command.count("-target-feature") == 4
    assert "+m" in command
    assert "-d" in command
    assert str(VORTEX_BUILD_DIR / "kernel/libvortex.a") in command
    assert f"-I{VORTEX_BUILD_DIR / 'hw'}" in command


def test_compile_timeout_preserves_stage_and_output(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], kwargs["timeout"], output="partial-out", stderr="partial-err"
        )

    monkeypatch.setattr(vortex.subprocess, "run", timeout)
    monkeypatch.setenv("TVM_VORTEX_COMPILE_TIMEOUT_SECONDS", "2")

    with pytest.raises(
        vortex.VortexCompileError, match="timed out after 2 seconds"
    ) as error:
        vortex._run_command(
            ["fake-compiler", "kernel.cpp"],
            stage="kernel compilation",
            environment={},
            cwd=tmp_path,
        )

    assert "partial-out" in str(error.value)
    assert "partial-err" in str(error.value)


@pytest.mark.skipif(
    not _have_pinned_toolchain(), reason="pinned Vortex fpint toolchain unavailable"
)
def test_generated_vecadd_compiles_to_vxbin(monkeypatch):
    target = tvm.target.Target("vortex")
    source = _generated_source(target)
    monkeypatch.setenv("TVM_VORTEX_HOME", str(VORTEX_HOME))
    monkeypatch.setenv("TVM_VORTEX_LLVM_ROOT", str(LLVM_ROOT))
    monkeypatch.setenv("TVM_VORTEX_PROFILE_ROOT", str(PROFILE_ROOT))
    monkeypatch.setenv("PATH", "/ambient/path/must/not/be-used")
    binary = tvm.get_global_func("tvm_callback_vortex_compile")(source, target)

    assert isinstance(binary, (bytes, bytearray))
    assert len(binary) > 16
    min_vma, max_vma = struct.unpack_from("<QQ", binary)
    assert min_vma == 0x180000000
    assert max_vma > min_vma
    assert len(binary) <= 16 + max_vma - min_vma


@pytest.mark.skipif(
    not _have_pinned_toolchain(), reason="pinned Vortex fpint toolchain unavailable"
)
def test_compile_failure_preserves_command_and_stderr():
    with pytest.raises(vortex.VortexCompileError) as error:
        vortex.compile_vortex(
            "#error intentional-vortex-compile-error\n",
            tvm.target.Target("vortex"),
            **_pinned_compile_kwargs(),
        )

    message = str(error.value)
    assert str(LLVM_ROOT / "bin/clang++") in message
    assert "intentional-vortex-compile-error" in message
    assert "Command:" in message
    assert "stderr:" in message


if __name__ == "__main__":
    tvm.testing.main()
