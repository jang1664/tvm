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

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import tvm
from tvm.relax.backend.vortex import (
    BackendParameterArchive,
    C1_ALL_FP16_TCU,
    C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
    C3_ALL_W4_NAIVE,
    LogicalParameterArchive,
    prepare_backend_parameter_archive,
    prepare_logical_parameter_archive,
)

VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
sys.path.insert(0, str(VORTEX_HOME / "pytorch/spinquant"))

from spinquant_inference.vortex_export_ops import (  # noqa: E402
    _dequantize_reference,
    _quantize_reference,
)


def _target(*, tcu=False, gemm="none"):
    attrs = {"kind": "vortex", "vortex_gemm_mode": gemm}
    if tcu:
        attrs.update(vortex_tcu_mode="fp", vortex_tcu_fp_formats="fp16")
    return tvm.target.Target(attrs)


def _logical_archive(tmp_path):
    generator = torch.Generator().manual_seed(19)
    weight = (
        torch.randn((32, 16), generator=generator, dtype=torch.float32) * 0.25
    ).to(torch.float16)
    packed, scale, zero = _quantize_reference(
        weight, 0, 32, 1, "signed_asymmetric_int4"
    )
    parameters = {
        "layers.0.q_proj.qweight": packed.numpy(),
        "layers.0.q_proj.scales": scale.numpy(),
        "layers.0.q_proj.zeros": zero.numpy(),
        "layers.0.input_norm.weight": np.ones((32,), dtype="float16"),
    }
    metadata = {
        "model": "llama3-8b",
        "weight_group_size": 32,
        "quantization_policy": "signed_all_asymmetric_wkv4_v1",
    }
    manifest = prepare_logical_parameter_archive(
        tmp_path / "logical",
        parameters,
        num_layers=32,
        included_layers=(0,),
        model_metadata=metadata,
    )
    archive = LogicalParameterArchive(
        manifest,
        expected_num_layers=32,
        expected_model_metadata=metadata,
    )
    expected = _dequantize_reference(
        packed,
        scale,
        zero,
        [32, 16],
        0,
        32,
        1,
        "signed_asymmetric_int4",
    ).numpy()
    return archive, expected


def _assert_fp16_error_policy(actual, expected):
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    actual = actual.astype("float32")
    expected = expected.astype("float32")
    difference = np.abs(actual - expected)
    small = np.abs(expected) < 0.25
    assert np.max(difference[small], initial=0.0) <= 2e-3
    large = ~small
    relative = difference[large] / np.maximum(np.abs(expected[large]), 1e-6)
    assert np.max(relative, initial=0.0) <= 2e-3


def _open_materialization(manifest, policy, profile, logical):
    return BackendParameterArchive(
        manifest,
        expected_policy=policy,
        expected_profile_fingerprint=profile,
        expected_logical_manifest_sha256=logical.manifest_sha256,
        expected_logical_content_sha256=logical.content_sha256,
    )


def test_logical_archive_is_profile_neutral_and_c1_materializes_fp16(tmp_path):
    logical, expected = _logical_archive(tmp_path)
    assert "profile_fingerprint" not in logical.manifest

    manifest = prepare_backend_parameter_archive(
        tmp_path / "c1",
        logical,
        policy=C1_ALL_FP16_TCU,
        target=_target(tcu=True),
        profile_fingerprint="c1-profile",
    )
    archive = _open_materialization(
        manifest, C1_ALL_FP16_TCU, "c1-profile", logical
    )

    assert "layers.0.q_proj.qweight" not in archive.records
    weight = archive.tensor("layers.0.q_proj.weight")
    assert archive.records["layers.0.q_proj.weight"]["layout"] == "row_major_fp16"
    _assert_fp16_error_policy(weight, expected)


@pytest.mark.parametrize(
    ("policy", "target", "profile"),
    [
        (C3_ALL_W4_NAIVE, _target(gemm="naive"), "c3-profile"),
        (
            C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU,
            _target(tcu=True, gemm="naive"),
            "synthetic-c2-fixture-profile",
        ),
    ],
)
def test_row_major_materialization_preserves_canonical_w4(
    tmp_path, policy, target, profile
):
    logical, _ = _logical_archive(tmp_path)
    manifest = prepare_backend_parameter_archive(
        tmp_path / policy,
        logical,
        policy=policy,
        target=target,
        profile_fingerprint=profile,
    )
    archive = _open_materialization(manifest, policy, profile, logical)

    for name in logical.records:
        np.testing.assert_array_equal(archive.tensor(name), logical.tensor(name))
        assert archive.records[name]["layout"] == "canonical_row_major"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("policy", "backend policy mismatch"),
        ("profile", "profile fingerprint mismatch"),
        ("manifest", "logical manifest hash mismatch"),
        ("content", "logical content hash mismatch"),
    ],
)
def test_materialization_reload_fails_closed_on_identity_mismatch(
    tmp_path, field, message
):
    logical, _ = _logical_archive(tmp_path)
    manifest = prepare_backend_parameter_archive(
        tmp_path / "c3",
        logical,
        policy=C3_ALL_W4_NAIVE,
        target=_target(gemm="naive"),
        profile_fingerprint="c3-profile",
    )
    values = {
        "expected_policy": C3_ALL_W4_NAIVE,
        "expected_profile_fingerprint": "c3-profile",
        "expected_logical_manifest_sha256": logical.manifest_sha256,
        "expected_logical_content_sha256": logical.content_sha256,
    }
    keys = {
        "policy": "expected_policy",
        "profile": "expected_profile_fingerprint",
        "manifest": "expected_logical_manifest_sha256",
        "content": "expected_logical_content_sha256",
    }
    values[keys[field]] = "wrong"

    with pytest.raises(ValueError, match=message):
        BackendParameterArchive(manifest, **values)


def test_materialization_record_hash_detects_data_corruption(tmp_path):
    logical, _ = _logical_archive(tmp_path)
    manifest = prepare_backend_parameter_archive(
        tmp_path / "c3",
        logical,
        policy=C3_ALL_W4_NAIVE,
        target=_target(gemm="naive"),
        profile_fingerprint="c3-profile",
    )
    data_path = manifest.parent / "parameters.bin"
    data = bytearray(data_path.read_bytes())
    data[-1] ^= 1
    data_path.write_bytes(data)

    with pytest.raises(ValueError, match="data hash mismatch"):
        _open_materialization(manifest, C3_ALL_W4_NAIVE, "c3-profile", logical)
