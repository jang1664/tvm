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
"""Explicit Llama GEMM policies checked against Vortex target capabilities."""

from __future__ import annotations

from dataclasses import dataclass


C1_ALL_FP16_TCU = "c1_all_fp16_tcu"
C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU = (
    "c2_linear_w4_naive_attention_fp16_tcu"
)
C3_ALL_W4_NAIVE = "c3_all_w4_naive"
C4_ALL_W4_IMPROVE = "c4_all_w4_improve"

LINEAR_W4_NAIVE = "w4_naive"
LINEAR_FP16_TCU = "fp16_tcu"
ATTENTION_W4_NAIVE = "w4_naive"
ATTENTION_FP16_TCU = "fp16_tcu"


@dataclass(frozen=True)
class VortexBackendPolicy:
    """Backend assignment for semantic Llama GEMM roles."""

    name: str
    workload_variant: str
    linear_compute: str
    attention_compute: str
    physical_parameter_format: str
    layout_policy: str


_POLICIES = {
    C1_ALL_FP16_TCU: VortexBackendPolicy(
        name=C1_ALL_FP16_TCU,
        workload_variant="all_sgemm_tcu",
        linear_compute=LINEAR_FP16_TCU,
        attention_compute=ATTENTION_FP16_TCU,
        physical_parameter_format="fp16_dequantized",
        layout_policy="not_applicable",
    ),
    C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU: VortexBackendPolicy(
        name=C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
        workload_variant="attn_sgemm_tcu_fpint_gemm_naive",
        linear_compute=LINEAR_W4_NAIVE,
        attention_compute=ATTENTION_FP16_TCU,
        physical_parameter_format="row_major_w4",
        layout_policy="row_major",
    ),
    C3_ALL_W4_NAIVE: VortexBackendPolicy(
        name=C3_ALL_W4_NAIVE,
        workload_variant="all_fpint_gemm_naive",
        linear_compute=LINEAR_W4_NAIVE,
        attention_compute=ATTENTION_W4_NAIVE,
        physical_parameter_format="row_major_w4",
        layout_policy="row_major",
    ),
    C4_ALL_W4_IMPROVE: VortexBackendPolicy(
        name=C4_ALL_W4_IMPROVE,
        workload_variant="all_fpint_gemm_improve",
        linear_compute="w4_improve",
        attention_compute="w4_improve",
        physical_parameter_format="improve_prepacked_w4",
        layout_policy="alone_or_fused",
    ),
}


def get_vortex_backend_policy(name: str) -> VortexBackendPolicy:
    """Return a named policy, rejecting implicit fallback."""

    try:
        return _POLICIES[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported Vortex backend policy {name!r}; expected one of "
            f"{tuple(_POLICIES)}"
        ) from error


def _target_attr(target, name: str, default: str) -> str:
    attrs = getattr(target, "attrs", target)
    return str(attrs.get(name, default)) if hasattr(attrs, "get") else default


def validate_vortex_backend_policy(target, policy: str | VortexBackendPolicy):
    """Fail before compilation when ``target`` cannot implement ``policy``."""

    policy = get_vortex_backend_policy(policy) if isinstance(policy, str) else policy
    tcu_mode = _target_attr(target, "vortex_tcu_mode", "none")
    tcu_formats = _target_attr(target, "vortex_tcu_fp_formats", "")
    gemm_mode = _target_attr(target, "vortex_gemm_mode", "none")
    has_fp16_tcu = tcu_mode in ("fp", "fp_int") and "fp16" in tcu_formats.split(",")

    requires_tcu = "fp16_tcu" in (policy.linear_compute, policy.attention_compute)
    requires_naive = "w4_naive" in (policy.linear_compute, policy.attention_compute)
    requires_improve = "w4_improve" in (policy.linear_compute, policy.attention_compute)
    if requires_tcu and not has_fp16_tcu:
        raise ValueError(f"Vortex policy {policy.name!r} requires an FP16 TCU target")
    if requires_naive and gemm_mode != "naive":
        raise ValueError(
            f"Vortex policy {policy.name!r} requires vortex_gemm_mode='naive', "
            f"got {gemm_mode!r}"
        )
    if requires_improve and gemm_mode != "improve":
        raise ValueError(
            f"Vortex policy {policy.name!r} requires vortex_gemm_mode='improve', "
            f"got {gemm_mode!r}"
        )
    if policy.name == C1_ALL_FP16_TCU and gemm_mode != "none":
        raise ValueError(
            f"Vortex policy {policy.name!r} requires no FPINT GEMM accelerator, "
            f"got {gemm_mode!r}"
        )
    if policy.name == C3_ALL_W4_NAIVE and has_fp16_tcu:
        raise ValueError(
            f"Vortex policy {policy.name!r} requires the all-naive C3 capability contract"
        )
    return policy
