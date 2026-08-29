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
"""Tests for checked external Vortex C4 parameter archives."""

import json

import numpy as np
import pytest

import tvm
from tvm.relax.backend.vortex.layout import (
    ImproveProfile,
    plan_improve_layout,
    prepack_improve_qparam,
    prepack_improve_weight,
)
from tvm.relax.backend.vortex.parameter_archive import (
    C4ParameterArchive,
    C4WeightSpec,
    llama3_c4_weight_specs,
    prepare_c4_parameter_archive,
)


def _test_parameters(specs):
    generator = np.random.default_rng(20260828)
    parameters = {}
    for spec in specs:
        groups = (spec.logical_k + spec.qblock - 1) // spec.qblock
        parameters[f"{spec.name}.qweight"] = generator.integers(
            0,
            256,
            size=(spec.logical_k, (spec.logical_n + 1) // 2),
            dtype="uint8",
        )
        parameters[f"{spec.name}.scales"] = generator.uniform(
            0.001, 0.1, size=(groups, spec.logical_n)
        ).astype("float16")
        parameters[f"{spec.name}.zeros"] = generator.integers(
            -4, 5, size=(groups, spec.logical_n), dtype="int16"
        )
    layer_prefixes = {
        ".".join(spec.name.split(".", 2)[:2]) for spec in specs
    }
    for prefix in layer_prefixes:
        parameters[f"{prefix}.input_norm.weight"] = np.ones((33,), dtype="float16")
        parameters[f"{prefix}.post_attention_norm.weight"] = np.ones(
            (33,), dtype="float16"
        )
    return parameters


def test_c4_parameter_archive_prepack_and_validation(tmp_path):
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    specs = (
        C4WeightSpec("layers.0.q_proj", 33, 31),
        C4WeightSpec("layers.1.q_proj", 33, 31),
    )
    parameters = _test_parameters(specs)
    manifest_path = prepare_c4_parameter_archive(
        tmp_path,
        parameters,
        specs,
        target,
        profile_fingerprint="test-profile",
        num_layers=2,
    )
    archive = C4ParameterArchive(manifest_path, "test-profile", 2)
    assert len(archive.records) == 10

    plan = plan_improve_layout(
        1, 31, 33, 32, profile=ImproveProfile.from_target(target)
    )
    np.testing.assert_array_equal(
        archive.tensor("layers.0.q_proj.qweight"),
        prepack_improve_weight(parameters["layers.0.q_proj.qweight"], plan),
    )
    np.testing.assert_array_equal(
        archive.tensor("layers.0.q_proj.scales"),
        prepack_improve_qparam(
            parameters["layers.0.q_proj.scales"], plan, "float16"
        ),
    )
    np.testing.assert_array_equal(
        archive.tensor("layers.0.q_proj.zeros"),
        prepack_improve_qparam(
            parameters["layers.0.q_proj.zeros"], plan, "int16"
        ),
    )
    first_upload = archive.upload(tvm.cpu())
    second_upload = archive.upload(tvm.cpu())
    assert first_upload is second_upload
    assert first_upload["layers.0.q_proj.qweight"] is second_upload[
        "layers.0.q_proj.qweight"
    ]
    np.testing.assert_array_equal(
        archive.tensor("layers.1.post_attention_norm.weight"),
        np.ones((33,), dtype="float16"),
    )

    with pytest.raises(ValueError, match="profile fingerprint mismatch"):
        C4ParameterArchive(manifest_path, "other-profile", 2)
    with pytest.raises(ValueError, match="layer count mismatch"):
        C4ParameterArchive(manifest_path, "test-profile", 4)


def test_c4_parameter_archive_rejects_size_and_hash_corruption(tmp_path):
    target = tvm.target.Target(
        {"kind": "vortex", "vortex_gemm_mode": "improve"}, host="llvm"
    )
    specs = (C4WeightSpec("layers.0.q_proj", 32, 32),)
    manifest_path = prepare_c4_parameter_archive(
        tmp_path,
        _test_parameters(specs),
        specs,
        target,
        profile_fingerprint="test-profile",
        num_layers=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_size = manifest["data_nbytes"]
    manifest["data_nbytes"] = original_size + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data file is truncated"):
        C4ParameterArchive(manifest_path, "test-profile", 1)

    manifest["data_nbytes"] = original_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data_path = tmp_path / manifest["data_file"]
    with data_path.open("r+b") as stream:
        stream.seek(manifest["records"][0]["offset"])
        original = stream.read(1)
        stream.seek(manifest["records"][0]["offset"])
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="data hash mismatch"):
        C4ParameterArchive(manifest_path, "test-profile", 1)


def test_llama3_weight_specs_cover_every_layer_and_projection():
    specs = llama3_c4_weight_specs(32)
    assert len(specs) == 32 * 7
    assert specs[0].name == "layers.0.q_proj"
    assert specs[-1].name == "layers.31.down_proj"
    full_specs = llama3_c4_weight_specs(32, vocabulary_size=128256)
    assert len(full_specs) == 32 * 7 + 1
    assert full_specs[-1].name == "lm_head"
