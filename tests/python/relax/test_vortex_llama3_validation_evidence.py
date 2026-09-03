import copy
import hashlib
import json

import numpy as np
import pytest

from vortex_llama3.collect_backend_validation_evidence import (
    _failure_manifest_summary,
    _trace_summary,
)
from vortex_llama3.run_backend_validation import LOCAL_THRESHOLDS, batch_hybrid_metrics


def _metric():
    return {
        "pass": 1,
        "nonfinite_count": 0,
        "relative_l2": 0.0,
        "max_small_absolute_error": 0.0,
        "max_large_relative_error": 0.0,
    }


def _canonical_record():
    return {
        "mode": "canonical",
        "phases": [
            {
                "phase_index": phase,
                "layers": [
                    {
                        "layer": layer,
                        "hidden": _metric(),
                        "hidden_by_batch": [_metric(), _metric()],
                    }
                    for layer in range(32)
                ],
                "embedding": _metric(),
                "embedding_by_batch": [_metric(), _metric()],
                "normalized": _metric(),
                "normalized_by_batch": [_metric(), _metric()],
                "logits": _metric(),
                "logits_by_batch": [_metric(), _metric()],
                "top1": [0, 0],
                "latency_seconds": 1.0,
            }
            for phase in range(4)
        ],
    }


def _free_record():
    return {
        "mode": "free_running",
        "generated_token_ids": [[0, 0, 0], [0, 0, 0]],
        "phases": [
            {
                "phase_index": phase,
                "layer_hidden_hashes": [f"hidden-{layer}" for layer in range(32)],
                "layer_hidden_batch_hashes": [
                    [f"hidden-{layer}-batch-0", f"hidden-{layer}-batch-1"]
                    for layer in range(32)
                ],
                "cache_lengths": [7 + phase] * 32,
                "maximum_logit_magnitude": 0.0,
                "logits_sha256": f"logits-{phase}",
                "logits_batch_sha256": [
                    f"logits-{phase}-batch-0",
                    f"logits-{phase}-batch-1",
                ],
                "normalized_batch_sha256": [
                    f"normalized-{phase}-batch-0",
                    f"normalized-{phase}-batch-1",
                ],
                "latency_seconds": 1.0,
            }
            for phase in range(4)
        ],
    }


def _trace(identity):
    return {
        "format": "vortex-llama3-c1-c3-u55c-validation-trace",
        "package_identity": identity,
        "reference_npz_sha256": "reference-sha",
        "reference_device": {"reference_device": "gpu"},
        "exec_mode": "bytecode",
        "allocator": "naive",
        "canonical_input_staging": "fixed",
        "canonical_input_readback_verified": True,
        "device_open_count": 1,
        "parameter_upload_count": 1,
        "cuda_initialized_in_xrt_process": False,
        "runtime_environment": {"xrt_xclbin_path": identity["xclbin"]},
        "records": [_canonical_record(), _free_record(), _free_record()],
        "total_vm_invocation_count": 408,
    }


def test_trace_summary_accepts_gpu_canonical_and_persistent_free_running(tmp_path):
    identity = {
        "alias": "C3",
        "shape_case": "S4",
        "shape": {"batch_size": 2, "prompt_length": 7, "cache_capacity": 16},
        "xclbin": "/opt/fpga/c3/vortex_afu.xclbin",
    }
    trace = _trace(identity)
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    summary = _trace_summary(path, trace, identity, "reference-sha")

    assert len(summary["canonical"]) == 1
    assert summary["canonical"][0]["layer_comparison_count"] == 128
    assert len(summary["free_running"]) == 2
    assert summary["free_running"][0]["generated_token_ids"] == [[0, 0, 0]] * 2
    assert summary["canonical_input_staging"] == "fixed"
    assert summary["canonical_input_readback_verified"] is True


def test_trace_summary_rejects_cpu_acceptance_reference(tmp_path):
    identity = {
        "alias": "C1",
        "shape_case": "S1",
        "shape": {"batch_size": 1, "prompt_length": 1, "cache_capacity": 8},
        "xclbin": "/opt/fpga/c1/vortex_afu.xclbin",
    }
    trace = _trace(identity)
    trace["reference_device"] = {"reference_device": "diagnostic_cpu"}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(ValueError, match="non-GPU acceptance trace"):
        _trace_summary(path, trace, identity, "reference-sha")


def test_trace_summary_rejects_changed_persistent_hashes(tmp_path):
    identity = {
        "alias": "C3",
        "shape_case": "S4",
        "shape": {"batch_size": 2, "prompt_length": 7, "cache_capacity": 16},
        "xclbin": "/opt/fpga/c3/vortex_afu.xclbin",
    }
    trace = _trace(identity)
    trace["records"][2] = copy.deepcopy(trace["records"][2])
    trace["records"][2]["phases"][3]["logits_sha256"] = "changed"
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(ValueError, match="persistent free-running hashes changed"):
        _trace_summary(path, trace, identity, "reference-sha")


def test_batch_hybrid_metrics_enforces_each_row_independently():
    expected = np.ones((2, 4), dtype="float16")
    actual = expected.copy()
    actual[1, 0] = 2

    with pytest.raises(AssertionError, match="batch1 numerical limits exceeded"):
        batch_hybrid_metrics(
            actual,
            expected,
            LOCAL_THRESHOLDS,
            name="hidden",
        )


def test_failure_manifest_requires_gpu_reference_and_artifact_hash(tmp_path):
    artifact = tmp_path / "mismatch.npz"
    artifact.write_bytes(b"failure")
    manifest = tmp_path / "failures.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "vortex-llama3-c1-c3-u55c-failure-evidence",
                "schema_version": 1,
                "failures": [
                    {
                        "id": "c1-s3-one-zero",
                        "alias": "C1",
                        "case": "S3",
                        "reference_device": "gpu",
                        "counts_against_acceptance": True,
                        "artifacts": [
                            {
                                "path": str(artifact),
                                "sha256": hashlib.sha256(b"failure").hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failures = _failure_manifest_summary(manifest)

    assert failures[0]["id"] == "c1-s3-one-zero"
    assert failures[0]["artifacts"][0]["nbytes"] == len(b"failure")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["failures"][0]["reference_device"] = "diagnostic_cpu"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not GPU-versus-Vortex"):
        _failure_manifest_summary(manifest)
