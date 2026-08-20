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
"""Vortex backend registration and runtime sidecar loading."""

from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from tvm.base import _LOADED_LIBS


def register_backend():
    """Load the Vortex runtime module for registration side effects."""

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _LOADED_LIBS["tvm_runtime_vortex"] = load_lib_ctypes(
            package="tvm",
            target_name="tvm_runtime_vortex",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass


__all__ = ["register_backend"]
