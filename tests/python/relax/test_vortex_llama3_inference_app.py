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
"""Host-only tests for the directly runnable Vortex Llama3 utility."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tvm


TVM_HOME = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TVM_HOME / "apps"))

from vortex_llama3.run_synthetic_inference import (  # noqa: E402
    ARTIFACT_NAMES,
    BASE_WEIGHT_SCALE,
    COMPILED_LAYERS,
    NUM_LAYERS,
    SYNTHETIC_PARAMETER_SCHEME,
    _compare_layer_state,
    _deterministic_parameters,
    _chunk_parameter_names,
    _repetition_trace_path,
    _persist_layer_state,
    _topk,
    make_parser,
    parse_prompt_token_ids,
)


def test_preallocated_state_persistence_reuses_destination():
    device = tvm.cpu(0)
    first = tvm.runtime.tensor(np.array([1.0, 2.0], dtype="float16"), device=device)
    persisted = _persist_layer_state((first,), device, "preallocated")
    second = tvm.runtime.tensor(np.array([3.0, 4.0], dtype="float16"), device=device)
    reused = _persist_layer_state((second,), device, "preallocated", persisted)

    assert reused[0].same_as(persisted[0])
    np.testing.assert_array_equal(reused[0].numpy(), np.array([3.0, 4.0], dtype="float16"))


def test_persistent_repetition_cli_and_trace_paths():
    args = make_parser().parse_args(
        [
            "--layout-policy",
            "alone",
            "--prompt-token-ids",
            "1",
            "--cache-capacity",
            "8",
            "--artifact-dir",
            "artifact",
            "--inference-repetitions",
            "3",
            "--continue-after-inference-failure",
            "--diagnostic-host-embedding",
            "--diagnostic-layer-retries",
            "3",
            "--diagnostic-canonical-phase-limit",
            "2",
            "--diagnostic-reference-head",
            "--diagnostic-reference-decode-inputs",
            "--decode-state-persistence",
            "retain-all",
            "--state-transport",
            "host-snapshot",
            "--decode-allocator",
            "naive",
            "--layer-vm-scope",
            "per-call",
            "--decode-layer-vm-scope",
            "per-call",
        ]
    )
    assert args.inference_repetitions == 3
    assert args.continue_after_inference_failure
    assert args.diagnostic_host_embedding
    assert args.diagnostic_layer_retries == 3
    assert args.diagnostic_canonical_phase_limit == 2
    assert args.diagnostic_reference_head
    assert args.diagnostic_reference_decode_inputs
    assert args.decode_state_persistence == "retain-all"
    assert args.state_transport == "host-snapshot"
    assert args.decode_allocator == "naive"
    assert args.layer_vm_scope == "per-call"
    assert args.decode_layer_vm_scope == "per-call"
    assert _repetition_trace_path(Path("trace.json"), 2) == Path(
        "trace.repetition-2.json"
    )


def test_parse_prompt_token_ids_supports_single_and_batched_prompts():
    assert parse_prompt_token_ids("1,2,3") == [[1, 2, 3]]
    assert parse_prompt_token_ids("1,2; 3,4") == [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="same length"):
        parse_prompt_token_ids("1;2,3")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_prompt_token_ids("1;")


def test_partitioned_package_has_phase_specific_boundaries():
    assert ARTIFACT_NAMES == (
        "embedding_prefill",
        "embedding_decode",
        "prefill_layer",
        "decode_layer",
        "final_head_prefill",
        "final_head_decode",
    )
    assert COMPILED_LAYERS == 1


def test_chunk_parameter_names_map_local_layers_to_global_archive():
    assert _chunk_parameter_names(
        ("layers.0.q_proj.scales", "layers.3.down_proj.zeros"), 12
    ) == ("layers.12.q_proj.scales", "layers.15.down_proj.zeros")


def test_topk_is_sorted_per_batch():
    logits = np.array(
        [[[0.0, 4.0, 1.0, 3.0, 2.0]], [[7.0, 6.0, 9.0, 8.0, 5.0]]],
        dtype="float16",
    )
    topk = _topk(logits, count=3)
    assert [entry["token_id"] for entry in topk[0]] == [1, 3, 4]
    assert [entry["token_id"] for entry in topk[1]] == [2, 3, 0]


def test_synthetic_parameters_depth_scale_residual_projections():
    def shapes(unused_config, unused_num_layers):
        return {
            "layers.0.q_proj.scales": ((2,), torch.float16),
            "layers.0.o_proj.scales": ((2,), torch.float16),
            "layers.0.down_proj.scales": ((2,), torch.float16),
            "lm_head.scales": ((2,), torch.float16),
        }

    parameters = _deterministic_parameters(object(), shapes, seed=7)
    residual_scale = BASE_WEIGHT_SCALE / math.sqrt(2.0 * NUM_LAYERS)
    np.testing.assert_allclose(
        parameters["layers.0.q_proj.scales"].numpy(), BASE_WEIGHT_SCALE
    )
    np.testing.assert_allclose(
        parameters["layers.0.o_proj.scales"].numpy(), residual_scale
    )
    np.testing.assert_allclose(
        parameters["layers.0.down_proj.scales"].numpy(), residual_scale
    )
    np.testing.assert_allclose(parameters["lm_head.scales"].numpy(), BASE_WEIGHT_SCALE)
    assert SYNTHETIC_PARAMETER_SCHEME == "depth_scaled_residual_v1"


def test_layer_state_comparison_accepts_scalar_compiled_cache_length():
    hidden = np.zeros((1, 1, 4), dtype="float16")
    payload = np.zeros((1, 1, 2, 2), dtype="uint8")
    scale = np.ones((1, 1, 2, 1), dtype="float16")
    zero = np.zeros((1, 1, 2, 1), dtype="int16")
    actual = (hidden, payload, scale, zero, payload, scale, zero, np.array(1))
    expected = (
        hidden.copy(),
        payload.copy(),
        scale.copy(),
        zero.copy(),
        payload.copy(),
        scale.copy(),
        zero.copy(),
        np.array([1]),
    )

    summary = _compare_layer_state(actual, expected, 1, compare_hidden=True)

    assert summary["hidden"]["relative_l2"] == 0.0
    assert summary["cache"]["key"]["code_mismatch_rate"] == 0.0
