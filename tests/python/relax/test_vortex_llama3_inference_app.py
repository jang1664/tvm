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

import json
import math
import sys
from types import SimpleNamespace
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
from vortex_llama3.debug_canonical_layer_range import (  # noqa: E402
    EventTrace,
    make_parser as make_canonical_parser,
    parse_layer_range,
    parse_phases,
    required_reference_keys,
    validate_reference_keys,
)
from vortex_llama3.backend_numerical_validation import (  # noqa: E402
    compare_replays,
    deterministic_prompt,
)
from vortex_llama3.run_backend_validation import (  # noqa: E402
    FINAL_THRESHOLDS,
    LOCAL_THRESHOLDS,
    _configure_xrt_environment,
    _detect_open_xrt_bdf,
    _runtime_tensor,
    compare_layer_state as compare_backend_layer_state,
    hybrid_metrics,
    make_parser as make_backend_validation_parser,
)
from vortex_llama3.run_backend_probe import (  # noqa: E402
    make_parser as make_backend_probe_parser,
)
from vortex_llama3.run_backend_residual_probe import (  # noqa: E402
    make_parser as make_residual_probe_parser,
)
from vortex_llama3.run_backend_stage_probe import (  # noqa: E402
    make_parser as make_stage_probe_parser,
)


def test_candidate_resolution_uses_capabilities_and_rejects_config_drift(tmp_path):
    from vortex_llama3.compile_backend_matrix import resolve_backend
    from tvm.relax.backend.vortex import C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU

    configs = (
        "-DNUM_THREADS=16 -DMXU_ROW=16 -DMXU_COL=16 "
        "-DENABLE_GEMM_ACCEL -DGEMM_NAIVE -DEXT_TCU_ENABLE "
        "-DDISABLE_TCU_INT -DDISABLE_BF16"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"params": {"CONFIGS": configs}}))
    config = tmp_path / "candidate.sh"
    config.write_text(f"CONFIGS='{configs}'\n")
    requested = []

    def resolve(alias, **kwargs):
        requested.append(alias)
        return SimpleNamespace(manifest=manifest, config=config)

    dependencies = {"resolve_alias": resolve, "candidate_map": {"unrelated-name": "exact-image"}}
    _, _, target, policy = resolve_backend("unrelated-name", tmp_path / "aliases.yaml", dependencies)
    assert requested == ["exact-image"]
    assert policy.name == C2_LINEAR_W4_NAIVE_ATTENTION_FP16_TCU
    assert target.attrs["thread_warp_size"] == 16
    config.write_text(f"CONFIGS='{configs} -DLMEM_SIZE=1048576'\n")
    with pytest.raises(ValueError, match="source config conflicts with FPGA manifest"):
        resolve_backend("unrelated-name", tmp_path / "aliases.yaml", dependencies)


def test_canonical_layer_range_parsing_and_required_reference_keys():
    phases = parse_phases("prefill,decode_2", decode_steps=3)
    layers = parse_layer_range("8:9")

    assert phases == (0, 2)
    assert layers == (8, 9)
    keys = required_reference_keys(phases, layers)
    assert "p0_l7_o0" in keys
    assert "p2_l8_o7" in keys
    assert "p1_l9_o6" in keys

    with pytest.raises(ValueError, match="invalid diagnostic phase"):
        parse_phases("decode_x", decode_steps=3)
    with pytest.raises(ValueError, match="outside decode-steps"):
        parse_phases("decode_4", decode_steps=3)
    with pytest.raises(ValueError, match="0 <= START"):
        parse_layer_range("9:8")


def test_canonical_layer_range_rejects_missing_reference_arrays():
    with pytest.raises(ValueError, match="missing"):
        validate_reference_keys({}, (3,), (10, 10))


def test_event_trace_flushes_one_json_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventTrace(path, {"run_uuid": "test-run"}) as trace:
        trace.write("before", layer=2)
        assert path.read_text(encoding="utf-8").count("\n") == 1
        trace.write("after", healthy=True)

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == ["before", "after"]
    assert all(record["run_uuid"] == "test-run" for record in records)


def test_preallocated_state_persistence_reuses_destination():
    device = tvm.cpu(0)
    first = tvm.runtime.tensor(np.array([1.0, 2.0], dtype="float16"), device=device)
    persisted = _persist_layer_state((first,), device, "preallocated")
    second = tvm.runtime.tensor(np.array([3.0, 4.0], dtype="float16"), device=device)
    reused = _persist_layer_state((second,), device, "preallocated", persisted)

    assert reused[0].same_as(persisted[0])
    np.testing.assert_array_equal(
        reused[0].numpy(), np.array([3.0, 4.0], dtype="float16")
    )


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

    canonical_args = make_canonical_parser().parse_args(
        [
            "--artifact-dir",
            "artifact",
            "--reference-artifact",
            "reference.npz",
            "--xclbin",
            "image.xclbin",
            "--trace-output",
            "events.jsonl",
            "--repetitions",
            "3",
        ]
    )
    assert canonical_args.repetitions == 3


def test_parse_prompt_token_ids_supports_single_and_batched_prompts():
    assert parse_prompt_token_ids("1,2,3") == [[1, 2, 3]]
    assert parse_prompt_token_ids("1,2; 3,4") == [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="same length"):
        parse_prompt_token_ids("1;2,3")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_prompt_token_ids("1;")


def test_backend_validation_uses_frozen_s1_s4_prompts():
    assert deterministic_prompt("S1") == [[1]]
    assert deterministic_prompt("S2") == [[1, 2, 3, 4, 5, 6, 7]]
    assert deterministic_prompt("S3") == [[1], [2]]
    assert deterministic_prompt("S4") == [
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
    ]
    with pytest.raises(ValueError, match="unknown validation shape"):
        deterministic_prompt("S5")


def test_backend_reference_replay_requires_exact_tensor_hashes():
    first = {"hidden": np.array([1.0, 2.0], dtype="float16")}
    compare_replays(first, {"hidden": first["hidden"].copy()})
    with pytest.raises(AssertionError, match="nondeterministic"):
        compare_replays(first, {"hidden": np.array([1.0, 2.1], dtype="float16")})
    with pytest.raises(AssertionError, match="inventory"):
        compare_replays(first, {"logits": first["hidden"]})


def test_backend_hybrid_metrics_split_small_absolute_and_large_relative():
    expected = np.array([0.1, 1.0], dtype="float16")
    actual = np.array([0.101, 1.001], dtype="float16")
    metrics = hybrid_metrics(actual, expected, LOCAL_THRESHOLDS, name="test")
    assert metrics["pass"] == 1
    assert metrics["max_small_absolute_error"] > 0
    assert metrics["max_large_relative_error"] > 0
    with pytest.raises(AssertionError, match="numerical limits"):
        hybrid_metrics(
            np.array([0.2, 2.0], dtype="float16"),
            expected,
            FINAL_THRESHOLDS,
            name="bad",
        )


def test_backend_layer_comparison_checks_cache_suffix_and_length():
    hidden = np.zeros((1, 1, 4), dtype="float16")
    payload = np.zeros((1, 1, 1, 2, 2), dtype="uint8")
    scale = np.ones((1, 1, 1, 2, 1), dtype="float16")
    zero = np.zeros((1, 1, 1, 2, 1), dtype="int16")
    state = (hidden, payload, scale, zero, payload, scale, zero, np.array([1]))
    summary = compare_backend_layer_state(state, state, 1)
    assert summary["hidden"]["pass"] == 1
    assert summary["cache"]["key"]["code_mismatch_rate"] == 0.0

    bad = list(state)
    bad[1] = payload.copy()
    bad[1][..., 1:, :] = 1
    with pytest.raises(AssertionError):
        compare_backend_layer_state(tuple(bad), state, 1)


def test_backend_layer_comparison_rejects_rewritten_decode_prefix():
    hidden = np.zeros((1, 1, 4), dtype="float16")
    payload = np.zeros((1, 1, 1, 3, 2), dtype="uint8")
    scale = np.ones((1, 1, 1, 3, 1), dtype="float16")
    zero = np.zeros((1, 1, 1, 3, 1), dtype="int16")
    input_cache = (payload, scale, zero, payload, scale, zero, np.array([1]))
    rewritten_payload = payload.copy()
    rewritten_payload[..., 0, :] = 1
    output_state = (
        hidden,
        rewritten_payload,
        scale,
        zero,
        rewritten_payload,
        scale,
        zero,
        np.array([2]),
    )

    with pytest.raises(AssertionError, match="changed its valid prefix"):
        compare_backend_layer_state(
            output_state, output_state, 2, input_cache=input_cache
        )


def test_backend_runtime_tensor_preserves_scalar_rank():
    scalar = _runtime_tensor(np.array(1, dtype="int64"), tvm.cpu(0))
    vector = _runtime_tensor(np.array([1], dtype="int64"), tvm.cpu(0))

    assert scalar.shape == ()
    assert vector.shape == (1,)


def test_backend_probe_accepts_repeated_layer_execution():
    args = make_backend_probe_parser().parse_args(
        [
            "--package",
            "package.json",
            "--reference",
            "reference.npz",
            "--alias",
            "C1",
            "--case",
            "S3",
            "--probe",
            "layer",
            "--repetitions",
            "16",
            "--trace-output",
            "trace.json",
        ]
    )
    assert args.repetitions == 16


def test_residual_probe_accepts_long_repetition_count():
    args = make_residual_probe_parser().parse_args(
        [
            "--package",
            "package.json",
            "--reference",
            "reference.npz",
            "--probe-package",
            "probe.json",
            "--repetitions",
            "100000",
            "--trace-output",
            "trace.json",
        ]
    )
    assert args.repetitions == 100000


def test_stage_probe_accepts_s3_repetition_count():
    args = make_stage_probe_parser().parse_args(
        [
            "--package",
            "package.json",
            "--reference",
            "reference.npz",
            "--stage-package",
            "stage.json",
            "--alias",
            "C1",
            "--case",
            "S3",
            "--probe",
            "layer_checkpoints_decode",
            "--repetitions",
            "64",
            "--warmup-prefill-repetitions",
            "32",
            "--warmup-prefill-chain-repetitions",
            "2",
            "--full-resident-archive",
            "--trace-output",
            "trace.json",
        ]
    )
    assert args.case == "S3"
    assert args.repetitions == 64
    assert args.warmup_prefill_repetitions == 32
    assert args.warmup_prefill_chain_repetitions == 2
    assert args.full_resident_archive


def test_backend_validation_defaults_to_fixed_verified_canonical_inputs():
    args = make_backend_validation_parser().parse_args(
        [
            "--package",
            "package.json",
            "--reference",
            "reference.npz",
            "--alias",
            "C1",
            "--case",
            "S3",
            "--trace-output",
            "trace.json",
            "--mismatch-dir",
            "mismatches",
        ]
    )
    assert args.canonical_input_staging == "fixed"
    assert args.verify_canonical_input_readback


def test_xrt_configuration_assigns_process_unique_status_path(monkeypatch):
    for name in ("FPGA_BIN_DIR", "XRT_XCLBIN_PATH", "XRT_INI_PATH", "VORTEX_SHM_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    _configure_xrt_environment({"profile": {"xclbin": "/tmp/c1/bin/image.xclbin"}})

    assert Path(__import__("os").environ["VORTEX_SHM_PATH"]).name.startswith(
        "vortex_status_llama3_1234_"
    )


def test_backend_validation_detects_open_xrt_bdf(tmp_path):
    proc_fd = tmp_path / "fd"
    drm_class = tmp_path / "drm"
    pci_device = tmp_path / "pci" / "0000:3d:00.1"
    proc_fd.mkdir()
    pci_device.mkdir(parents=True)
    (proc_fd / "17").symlink_to("/dev/dri/renderD130")
    render_class = drm_class / "renderD130"
    render_class.mkdir(parents=True)
    (render_class / "device").symlink_to(pci_device)

    assert _detect_open_xrt_bdf(proc_fd, drm_class) == "0000:3d:00.1"


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
