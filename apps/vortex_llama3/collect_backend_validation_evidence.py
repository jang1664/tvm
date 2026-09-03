#!/usr/bin/env python3
"""Collect and strictly validate compact C1/C3 GPU-versus-U55C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from vortex_llama3.backend_numerical_validation import package_identity


ALIASES = ("C1", "C3")
CASES = ("S1", "S2", "S3", "S4")
PHASES = 4
LAYERS = 32
TRACE_FORMAT = "vortex-llama3-c1-c3-u55c-validation-trace"
FAILURE_FORMAT = "vortex-llama3-c1-c3-u55c-failure-evidence"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_label(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        label = value.get("reference_device")
        return label if isinstance(label, str) else None
    return None


def _metric_dicts(value: object):
    if isinstance(value, Mapping):
        if "pass" in value and "nonfinite_count" in value:
            yield value
        for nested in value.values():
            yield from _metric_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _metric_dicts(nested)


def _canonical_record_summary(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    phases = record.get("phases")
    if not isinstance(phases, list) or len(phases) != PHASES:
        raise ValueError("canonical record must contain four phases")
    metrics: list[Mapping[str, Any]] = []
    batch_size = int(identity["shape"]["batch_size"])
    explicit_batch_comparison_count = 0
    for phase_index, phase in enumerate(phases):
        if phase.get("phase_index") != phase_index:
            raise ValueError("canonical phase indices are not contiguous")
        layers = phase.get("layers")
        if not isinstance(layers, list) or len(layers) != LAYERS:
            raise ValueError("canonical phase must contain 32 layer records")
        if [layer.get("layer") for layer in layers] != list(range(LAYERS)):
            raise ValueError("canonical layer indices are not contiguous")
        if batch_size > 1:
            for key in ("embedding_by_batch", "normalized_by_batch", "logits_by_batch"):
                if len(phase.get(key, [])) != batch_size:
                    raise ValueError(f"canonical phase lacks explicit {key} metrics")
            if any(
                len(layer.get("hidden_by_batch", [])) != batch_size for layer in layers
            ):
                raise ValueError(
                    "canonical layer lacks explicit per-batch hidden metrics"
                )
            explicit_batch_comparison_count += batch_size * (LAYERS + 3)
        metrics.extend(_metric_dicts(phase))
    if not metrics:
        raise ValueError("canonical record contains no numerical metrics")
    if any(metric.get("pass") != 1 for metric in metrics):
        raise ValueError("canonical trace contains a failed metric")
    if any(metric.get("nonfinite_count") != 0 for metric in metrics):
        raise ValueError("canonical trace contains a non-finite metric")
    return {
        "phase_count": len(phases),
        "layer_comparison_count": len(phases) * LAYERS,
        "metric_count": len(metrics),
        "explicit_batch_comparison_count": explicit_batch_comparison_count,
        "maximum_relative_l2": max(float(metric["relative_l2"]) for metric in metrics),
        "maximum_small_absolute_error": max(
            float(metric["max_small_absolute_error"]) for metric in metrics
        ),
        "maximum_large_relative_error": max(
            float(metric["max_large_relative_error"]) for metric in metrics
        ),
        "top1": [phase["top1"] for phase in phases],
        "phase_latency_seconds": [float(phase["latency_seconds"]) for phase in phases],
    }


def _free_record_summary(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    phases = record.get("phases")
    if not isinstance(phases, list) or len(phases) != PHASES:
        raise ValueError("free-running record must contain four phases")
    shape = identity["shape"]
    batch_size = int(shape["batch_size"])
    prompt_length = int(shape["prompt_length"])
    for phase_index, phase in enumerate(phases):
        if phase.get("phase_index") != phase_index:
            raise ValueError("free-running phase indices are not contiguous")
        if len(phase.get("layer_hidden_hashes", [])) != LAYERS:
            raise ValueError("free-running phase must contain 32 hidden hashes")
        if batch_size > 1:
            batch_hashes = phase.get("layer_hidden_batch_hashes", [])
            if len(batch_hashes) != LAYERS or any(
                len(layer_hashes) != batch_size for layer_hashes in batch_hashes
            ):
                raise ValueError("free-running phase lacks per-batch hidden hashes")
            if len(phase.get("logits_batch_sha256", [])) != batch_size:
                raise ValueError("free-running phase lacks per-batch logits hashes")
            if len(phase.get("normalized_batch_sha256", [])) != batch_size:
                raise ValueError("free-running phase lacks per-batch normalized hashes")
        # The runtime trace records one scalar valid length for each logical
        # decoder-layer state, rather than one value for each batch row.
        expected_lengths = [prompt_length + phase_index] * LAYERS
        if phase.get("cache_lengths") != expected_lengths:
            raise ValueError("free-running cache lengths are invalid")
        maximum = float(phase["maximum_logit_magnitude"])
        if not math.isfinite(maximum):
            raise ValueError("free-running logits are non-finite")
    generated = record.get("generated_token_ids")
    if not isinstance(generated, list) or len(generated) != batch_size:
        raise ValueError("free-running generated-token batch is invalid")
    if any(len(row) != PHASES - 1 for row in generated):
        raise ValueError(
            "free-running record must contain three generated tokens per batch"
        )
    return {
        "phase_count": len(phases),
        "generated_token_ids": generated,
        "phase_logits_sha256": [phase["logits_sha256"] for phase in phases],
        "maximum_logit_magnitude": max(
            float(phase["maximum_logit_magnitude"]) for phase in phases
        ),
        "phase_latency_seconds": [float(phase["latency_seconds"]) for phase in phases],
    }


def _trace_summary(
    path: Path,
    trace: Mapping[str, Any],
    identity: Mapping[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    if trace.get("format") != TRACE_FORMAT:
        raise ValueError(f"unsupported trace format: {path}")
    if trace.get("package_identity") != identity:
        raise ValueError(f"stale or cross-package trace: {path}")
    if _device_label(trace.get("reference_device")) != "gpu":
        raise ValueError(f"non-GPU acceptance trace: {path}")
    if trace.get("reference_npz_sha256") != reference_sha256:
        raise ValueError(f"trace/reference hash mismatch: {path}")
    if trace.get("device_open_count") != 1 or trace.get("parameter_upload_count") != 1:
        raise ValueError(f"trace did not preserve one-open/one-upload lifetime: {path}")
    if trace.get("cuda_initialized_in_xrt_process") is not False:
        raise ValueError(f"XRT process initialized CUDA: {path}")
    environment = trace.get("runtime_environment", {})
    if environment.get("xrt_xclbin_path") != identity["xclbin"]:
        raise ValueError(f"trace xclbin identity mismatch: {path}")
    canonical = []
    free = []
    for record in trace.get("records", []):
        if record.get("mode") == "canonical":
            canonical.append(_canonical_record_summary(record, identity))
        elif record.get("mode") == "free_running":
            free.append(_free_record_summary(record, identity))
        else:
            raise ValueError(f"unknown trace record mode: {path}")
    if len(free) > 1:
        first = free[0]["phase_logits_sha256"]
        if any(record["phase_logits_sha256"] != first for record in free[1:]):
            raise ValueError(f"persistent free-running hashes changed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "exec_mode": trace["exec_mode"],
        "allocator": trace["allocator"],
        "canonical_input_staging": trace.get("canonical_input_staging"),
        "canonical_input_readback_verified": trace.get(
            "canonical_input_readback_verified"
        ),
        "runtime_environment": environment,
        "total_vm_invocation_count": trace["total_vm_invocation_count"],
        "canonical": canonical,
        "free_running": free,
    }


def _failure_manifest_summary(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("format") != FAILURE_FORMAT or payload.get("schema_version") != 1:
        raise ValueError("invalid failure evidence manifest")
    failures = payload.get("failures")
    if not isinstance(failures, list):
        raise ValueError("failure evidence manifest has no failures list")
    result = []
    seen_ids = set()
    for failure in failures:
        failure_id = failure.get("id")
        if not isinstance(failure_id, str) or not failure_id or failure_id in seen_ids:
            raise ValueError("failure evidence IDs must be unique non-empty strings")
        seen_ids.add(failure_id)
        if failure.get("alias") not in ALIASES or failure.get("case") not in CASES:
            raise ValueError(f"invalid failure backend/case: {failure_id}")
        if failure.get("reference_device") != "gpu":
            raise ValueError(f"failure evidence is not GPU-versus-Vortex: {failure_id}")
        if not isinstance(failure.get("counts_against_acceptance"), bool):
            raise ValueError(f"failure evidence lacks acceptance policy: {failure_id}")
        artifacts = []
        for artifact in failure.get("artifacts", []):
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            artifact_path = artifact_path.resolve()
            if not artifact_path.is_file():
                raise ValueError(f"missing failure artifact: {artifact_path}")
            actual_sha256 = _sha256(artifact_path)
            if artifact.get("sha256") != actual_sha256:
                raise ValueError(f"failure artifact hash mismatch: {artifact_path}")
            artifacts.append(
                {
                    **artifact,
                    "path": str(artifact_path),
                    "sha256": actual_sha256,
                    "nbytes": artifact_path.stat().st_size,
                }
            )
        result.append({**failure, "artifacts": artifacts})
    return result


def collect_evidence(
    package_root: Path,
    reference_root: Path,
    run_root: Path,
    failure_manifest: Path | None = None,
) -> dict[str, Any]:
    traces_by_case: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(run_root.glob("*/*.json")):
        trace = _read_json(path)
        if trace.get("format") != TRACE_FORMAT:
            continue
        identity = trace.get("package_identity", {})
        key = (identity.get("alias"), identity.get("shape_case"))
        traces_by_case.setdefault(key, []).append((path, trace))

    backends: dict[str, Any] = {}
    missing: list[str] = []
    for alias in ALIASES:
        cases: dict[str, Any] = {}
        for case in CASES:
            package_path = package_root / alias / case / "package.json"
            reference_path = reference_root / alias / f"{case}.json"
            package = _read_json(package_path)
            identity = package_identity(package)
            reference = _read_json(reference_path)
            if reference.get("package_identity") != identity:
                raise ValueError(f"reference/package identity mismatch: {alias}/{case}")
            if reference.get("package_sha256") != _sha256(package_path):
                raise ValueError(
                    f"reference/package file hash mismatch: {alias}/{case}"
                )
            if _device_label(reference.get("reference_device")) != "gpu":
                raise ValueError(f"acceptance reference is not GPU: {alias}/{case}")
            if int(reference.get("determinism_replay_count", 0)) < 2:
                raise ValueError(
                    f"reference lacks deterministic replay: {alias}/{case}"
                )
            prompt_rows = reference.get("prompt_token_ids", [])
            batch_size = int(identity["shape"]["batch_size"])
            if len(prompt_rows) != batch_size:
                raise ValueError(f"reference prompt batch mismatch: {alias}/{case}")
            distinct_batch_prompts = (
                len({tuple(row) for row in prompt_rows}) == batch_size
            )
            if batch_size > 1 and not distinct_batch_prompts:
                raise ValueError(
                    f"batch-isolation prompts are not distinct: {alias}/{case}"
                )
            traces = [
                _trace_summary(path, trace, identity, reference["npz_sha256"])
                for path, trace in traces_by_case.get((alias, case), [])
            ]
            bytecode_canonical = any(
                trace["exec_mode"] == "bytecode" and trace["canonical"]
                for trace in traces
            )
            bytecode_free = any(
                trace["exec_mode"] == "bytecode" and trace["free_running"]
                for trace in traces
            )
            persistent_repetitions = max(
                (
                    len(trace["free_running"])
                    for trace in traces
                    if trace["exec_mode"] == "bytecode"
                ),
                default=0,
            )
            compiled_canonical = any(
                trace["exec_mode"] == "compiled" and trace["canonical"]
                for trace in traces
            )
            required = {
                "bytecode_canonical": bytecode_canonical,
                "bytecode_free_running": bytecode_free,
                "persistent_free_running_twice": case != "S1"
                or persistent_repetitions >= 2,
                "compiled_canonical": case != "S1" or compiled_canonical,
            }
            missing.extend(
                f"{alias}/{case}/{name}"
                for name, passed in required.items()
                if not passed
            )
            cases[case] = {
                "package": {
                    "path": str(package_path.resolve()),
                    "sha256": _sha256(package_path),
                    "identity": identity,
                },
                "reference": {
                    "path": str(reference_path.resolve()),
                    "sha256": _sha256(reference_path),
                    "npz_sha256": reference["npz_sha256"],
                    "reference_device": reference["reference_device"],
                    "determinism_replay_count": reference["determinism_replay_count"],
                    "tensor_count": reference["tensor_count"],
                    "prompt_token_ids": prompt_rows,
                },
                "traces": traces,
                "required_coverage": required,
                "batch_isolation": {
                    "batch_size": batch_size,
                    "distinct_prompt_rows": distinct_batch_prompts,
                    "explicit_per_batch_trace_checks_required": batch_size > 1,
                },
            }
        backends[alias] = {"cases": cases}
    failures = (
        _failure_manifest_summary(failure_manifest) if failure_manifest is not None else []
    )
    for failure in failures:
        case = backends[failure["alias"]]["cases"][failure["case"]]
        if failure.get("package_identity") != case["package"]["identity"]:
            raise ValueError(f"failure/package identity mismatch: {failure['id']}")
        if failure.get("reference_npz_sha256") != case["reference"]["npz_sha256"]:
            raise ValueError(f"failure/reference hash mismatch: {failure['id']}")
    backend_verdicts = {}
    for alias in ALIASES:
        has_failure = any(
            failure["alias"] == alias and failure["counts_against_acceptance"]
            for failure in failures
        )
        has_missing = any(item.startswith(f"{alias}/") for item in missing)
        backend_verdicts[alias] = (
            "FAIL" if has_failure else "INCOMPLETE" if has_missing else "PASS"
        )
    return {
        "format": "vortex-llama3-c1-c3-u55c-validation-evidence",
        "schema_version": 1,
        "acceptance_oracle": "gpu_vs_vortex",
        "cpu_policy": "explicit_diagnostic_fallback_only",
        "complete": not missing,
        "acceptance_pass": not missing and not any(
            failure["counts_against_acceptance"] for failure in failures
        ),
        "missing_coverage": missing,
        "backend_verdicts": backend_verdicts,
        "failure_manifest": (
            {
                "path": str(failure_manifest.resolve()),
                "sha256": _sha256(failure_manifest),
            }
            if failure_manifest is not None
            else None
        ),
        "failures": failures,
        "backends": backends,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    evidence = collect_evidence(
        args.package_root,
        args.reference_root,
        args.run_root,
        args.failure_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "backend_validation_evidence_complete",
                "complete": evidence["complete"],
                "missing_coverage": evidence["missing_coverage"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    if args.require_complete and not evidence["complete"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
