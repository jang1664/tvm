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

import pytest

import tvm
from tvm.relax.backend.vortex import (
    C1_ALL_FP16_TCU,
    C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
    C3_ALL_W4_NAIVE,
    C4_ALL_W4_IMPROVE,
    get_vortex_backend_policy,
    validate_vortex_backend_policy,
)


def _target(*, tcu=False, gemm="none"):
    values = {"kind": "vortex", "vortex_gemm_mode": gemm}
    if tcu:
        values.update(vortex_tcu_mode="fp", vortex_tcu_fp_formats="fp16")
    return tvm.target.Target(values)


@pytest.mark.parametrize(
    ("name", "target"),
    [
        (C1_ALL_FP16_TCU, _target(tcu=True)),
        (
            C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
            _target(tcu=True, gemm="naive"),
        ),
        (C3_ALL_W4_NAIVE, _target(gemm="naive")),
        (C4_ALL_W4_IMPROVE, _target(gemm="improve")),
    ],
)
def test_backend_policy_accepts_exact_capability_contract(name, target):
    assert validate_vortex_backend_policy(target, name).name == name


@pytest.mark.parametrize(
    ("name", "target", "message"),
    [
        (C1_ALL_FP16_TCU, _target(), "requires an FP16 TCU"),
        (C1_ALL_FP16_TCU, _target(tcu=True, gemm="naive"), "requires no FPINT"),
        (
            C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
            _target(gemm="naive"),
            "requires an FP16 TCU",
        ),
        (
            C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
            _target(tcu=True),
            "requires vortex_gemm_mode='naive'",
        ),
        (C3_ALL_W4_NAIVE, _target(tcu=True, gemm="naive"), "all-naive C3"),
        (C4_ALL_W4_IMPROVE, _target(gemm="naive"), "requires vortex_gemm_mode='improve'"),
    ],
)
def test_backend_policy_rejects_cross_policy_capabilities(name, target, message):
    with pytest.raises(ValueError, match=message):
        validate_vortex_backend_policy(target, name)


def test_backend_policy_has_explicit_role_routing():
    c2 = get_vortex_backend_policy(C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU)

    assert c2.workload_variant == "attn_sgemm_tcu_fpint_gemm_naive"
    assert c2.linear_compute == "w4_naive"
    assert c2.attention_compute == "fp16_tcu"
    assert c2.layout_policy == "row_major"


def test_backend_policy_rejects_unknown_name():
    with pytest.raises(ValueError, match="unsupported Vortex backend policy"):
        get_vortex_backend_policy("automatic_fallback")
