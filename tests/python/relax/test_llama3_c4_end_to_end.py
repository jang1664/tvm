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
"""Physical C4 acceptance tests for the backend-neutral Llama3 layer export."""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import tvm
from tvm import relax
from tvm.relax.backend.vortex.parameter_archive import (
    C4ParameterArchive,
    llama3_c4_weight_specs,
    prepare_c4_parameter_archive,
)
from tvm.relax.frontend.torch import from_exported_program
from tvm.support.vortex import load_vortex_accelerator_profile


VORTEX_HOME = Path("/home/jaeyongjang/project.local/vortex_base")
sys.path.insert(0, str(VORTEX_HOME / "pytorch/spinquant"))

from spinquant_inference.llama3_c4_export import (  # noqa: E402
    Llama3ExportConfig,
    Llama3LayerDecode,
    Llama3LayerPrefill,
    Llama3ModelDecode,
    Llama3ModelPrefill,
    Llama3StackDecode,
    Llama3StackPrefill,
    full_model_parameter_shapes,
    parameter_shapes,
    stack_parameter_shapes,
)


IMPROVED_XCLBIN = Path(
    "/opt/vortex_fpga_bins/fpint/"
    "xrt_hw_u55c_c_f100_fpint_64300e5119/bin/vortex_afu.xclbin"
)
CASES = {
    "S1": (1, 1, 8),
    "S2": (1, 7, 16),
    "S3": (2, 1, 8),
    "S4": (2, 7, 16),
}
MAX_CACHE_CODE_MISMATCH_RATE = 0.002


def _deterministic_parameters(config, seed=20260828):
    generator = torch.Generator().manual_seed(seed)
    parameters = {}
    for name, (shape, dtype) in parameter_shapes(config).items():
        if dtype == torch.uint8:
            parameters[name] = torch.randint(
                0, 256, shape, dtype=dtype, generator=generator
            )
        elif dtype == torch.int16:
            values = torch.randint(-2, 3, shape, dtype=dtype, generator=generator)
            parameters[name] = torch.where(values == 0, torch.ones_like(values), values)
        elif name.endswith("norm.weight"):
            parameters[name] = torch.ones(shape, dtype=dtype)
        else:
            parameters[name] = torch.full(shape, 1.0 / 4096.0, dtype=dtype)
    return parameters


def _unpack_signed_nibbles(packed):
    low = packed & np.uint8(15)
    high = packed >> np.uint8(4)
    values = np.stack((low, high), axis=-1).reshape(*packed.shape[:-1], -1)
    return np.where(values >= 8, values.astype("int16") - 16, values).astype("int8")


def _assert_cache_payload(
    actual,
    expected,
    valid_length,
    name,
    max_mismatch_rate=MAX_CACHE_CODE_MISMATCH_RATE,
):
    np.testing.assert_array_equal(
        actual[..., valid_length:, :],
        expected[..., valid_length:, :],
        err_msg=f"{name} untouched suffix",
    )
    actual_codes = _unpack_signed_nibbles(actual[..., :valid_length, :])
    expected_codes = _unpack_signed_nibbles(expected[..., :valid_length, :])
    differences = np.abs(actual_codes.astype("int16") - expected_codes.astype("int16"))
    mismatch_count = np.count_nonzero(differences)
    mismatch_rate = mismatch_count / differences.size
    mismatch_limit = max(4, math.ceil(max_mismatch_rate * differences.size))
    assert np.max(differences, initial=0) <= 1, f"{name} differs by more than one INT4 code"
    assert mismatch_count <= mismatch_limit, (
        f"{name} has {mismatch_count}/{differences.size} one-code mismatches "
        f"({mismatch_rate:.6f}), exceeding limit {mismatch_limit}"
    )


def _assert_cache_zero(actual, expected, valid_length, name, max_mismatch_rate):
    if max_mismatch_rate == 0:
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        return
    np.testing.assert_array_equal(
        actual[..., valid_length:, :],
        expected[..., valid_length:, :],
        err_msg=f"{name} untouched suffix",
    )
    differences = np.abs(
        actual[..., :valid_length, :].astype("int32")
        - expected[..., :valid_length, :].astype("int32")
    )
    mismatch_count = np.count_nonzero(differences)
    mismatch_rate = mismatch_count / differences.size
    mismatch_limit = max(1, math.ceil(max_mismatch_rate * differences.size))
    assert np.max(differences, initial=0) <= 1, f"{name} differs by more than one code"
    assert mismatch_count <= mismatch_limit, (
        f"{name} has {mismatch_count}/{differences.size} one-code mismatches "
        f"({mismatch_rate:.6f}), exceeding limit {mismatch_limit}"
    )


def _assert_hybrid_fp16(
    actual,
    expected,
    name,
    split,
    atol,
    rtol,
    max_exceed_fraction,
    max_relative_l2,
    min_cosine,
):
    actual = actual.astype("float32")
    expected = expected.astype("float32")
    assert np.all(np.isfinite(actual)), f"{name} actual contains non-finite values"
    assert np.all(np.isfinite(expected)), f"{name} expected contains non-finite values"
    absolute_error = np.abs(actual - expected)
    small = np.abs(expected) < split
    large = ~small
    relative_error = np.zeros_like(absolute_error)
    np.divide(
        absolute_error,
        np.abs(expected),
        out=relative_error,
        where=large,
    )
    failures = (small & (absolute_error > atol)) | (large & (relative_error > rtol))
    difference = actual - expected
    reference_norm = np.linalg.norm(expected.reshape(-1))
    relative_l2 = np.linalg.norm(difference.reshape(-1)) / max(reference_norm, 1e-12)
    actual_norm = np.linalg.norm(actual.reshape(-1))
    cosine = (
        1.0
        if actual_norm == 0 and reference_norm == 0
        else np.dot(actual.reshape(-1), expected.reshape(-1))
        / max(actual_norm * reference_norm, 1e-12)
    )
    exceed_fraction = np.count_nonzero(failures) / failures.size
    if (
        exceed_fraction > max_exceed_fraction
        or relative_l2 > max_relative_l2
        or cosine < min_cosine
    ):
        small_max = np.max(absolute_error[small], initial=0)
        large_max = np.max(relative_error[large], initial=0)
        raise AssertionError(
            f"{name} has {np.count_nonzero(failures)}/{failures.size} hybrid FP16 "
            f"violations (|reference| < {split}: max_abs={small_max:.6g}, "
            f"atol={atol}; otherwise: max_rel={large_max:.6g}, rtol={rtol}); "
            f"exceed_fraction={exceed_fraction:.6g}/{max_exceed_fraction}, "
            f"relative_l2={relative_l2:.6g}/{max_relative_l2}, "
            f"cosine={cosine:.6g}/{min_cosine}"
        )


def _dequantize_cache(payload, scale, zero, valid_length):
    codes = _unpack_signed_nibbles(payload[..., :valid_length, :]).astype("float32")
    group_size = codes.shape[-1] // scale.shape[-1]
    expanded_scale = np.repeat(
        scale[..., :valid_length, :].astype("float32"), group_size, axis=-1
    )
    expanded_zero = np.repeat(
        zero[..., :valid_length, :].astype("float32"), group_size, axis=-1
    )
    return (codes - expanded_zero) * expanded_scale


def test_hybrid_fp16_comparison_uses_absolute_then_relative_error():
    expected = np.array([0.0, 10.0], dtype="float16")
    _assert_hybrid_fp16(
        np.array([0.019, 10.4], dtype="float16"),
        expected,
        "hybrid",
        1.0,
        0.02,
        0.05,
        0,
        0.1,
        0.99,
    )
    with pytest.raises(AssertionError, match="max_abs"):
        _assert_hybrid_fp16(
            np.array([0.03, 10.4], dtype="float16"),
            expected,
            "hybrid",
            1.0,
            0.02,
            0.05,
            0,
            0.1,
            0.99,
        )
    with pytest.raises(AssertionError, match="max_rel"):
        _assert_hybrid_fp16(
            np.array([0.019, 10.6], dtype="float16"),
            expected,
            "hybrid",
            1.0,
            0.02,
            0.05,
            0,
            0.1,
            0.99,
        )


def test_dequantize_cache_uses_signed_codes_scale_and_zero():
    payload = np.array([[[0xF8], [0x00]]], dtype="uint8")
    scale = np.array([[[0.5], [0.0]]], dtype="float16")
    zero = np.array([[[1], [0]]], dtype="int16")
    np.testing.assert_array_equal(
        _dequantize_cache(payload, scale, zero, valid_length=1),
        np.array([[[-4.5, -1.0]]], dtype="float32"),
    )


def _assert_layer_outputs(
    actual,
    expected,
    valid_length,
    cache_code_mismatch_rate=MAX_CACHE_CODE_MISMATCH_RATE,
    cache_zero_mismatch_rate=0,
    scale_atol=5e-4,
    scale_rtol=0,
    hybrid_fp16=False,
    hidden_split=1.0,
    hidden_atol=0.08,
    hidden_rtol=0.12,
    hidden_max_exceed_fraction=0,
    hidden_max_relative_l2=0,
    hidden_min_cosine=1,
    scale_split=0.1,
    scale_max_exceed_fraction=0,
    scale_max_relative_l2=0,
    scale_min_cosine=1,
    cache_value_split=0.1,
    cache_value_atol=0.05,
    cache_value_rtol=0.05,
    cache_value_max_exceed_fraction=0.05,
    cache_value_max_relative_l2=0.05,
    cache_value_min_cosine=0.995,
):
    output_names = (
        "hidden",
        "k_payload",
        "k_scale",
        "k_zero",
        "v_payload",
        "v_scale",
        "v_zero",
        "cache_length",
    )
    assert len(actual) == len(expected) == 8
    failures = []
    try:
        if hybrid_fp16:
            _assert_hybrid_fp16(
                actual[0].numpy(),
                expected[0].numpy(),
                output_names[0],
                hidden_split,
                hidden_atol,
                hidden_rtol,
                hidden_max_exceed_fraction,
                hidden_max_relative_l2,
                hidden_min_cosine,
            )
        else:
            np.testing.assert_allclose(
                actual[0].numpy(),
                expected[0].numpy(),
                rtol=hidden_rtol,
                atol=hidden_atol,
                equal_nan=False,
                err_msg=output_names[0],
            )
    except AssertionError as error:
        failures.append(str(error))
    for index in (1, 4):
        try:
            _assert_cache_payload(
                actual[index].numpy(),
                expected[index].numpy(),
                valid_length,
                output_names[index],
                cache_code_mismatch_rate,
            )
        except AssertionError as error:
            failures.append(str(error))
    for index in (3, 6):
        try:
            _assert_cache_zero(
                actual[index].numpy(),
                expected[index].numpy(),
                valid_length,
                output_names[index],
                cache_zero_mismatch_rate,
            )
        except AssertionError as error:
            failures.append(str(error))
    try:
        np.testing.assert_array_equal(
            actual[7].numpy(),
            expected[7].numpy(),
            err_msg=output_names[7],
        )
    except AssertionError as error:
        failures.append(str(error))
    for index in (2, 5):
        try:
            if hybrid_fp16:
                _assert_hybrid_fp16(
                    actual[index].numpy(),
                    expected[index].numpy(),
                    output_names[index],
                    scale_split,
                    scale_atol,
                    scale_rtol,
                    scale_max_exceed_fraction,
                    scale_max_relative_l2,
                    scale_min_cosine,
                )
            else:
                np.testing.assert_allclose(
                    actual[index].numpy(),
                    expected[index].numpy(),
                    rtol=scale_rtol,
                    atol=scale_atol,
                    err_msg=output_names[index],
                )
        except AssertionError as error:
            failures.append(str(error))
    if hybrid_fp16:
        for payload_index, scale_index, zero_index, name in (
            (1, 2, 3, "k_dequantized"),
            (4, 5, 6, "v_dequantized"),
        ):
            try:
                _assert_hybrid_fp16(
                    _dequantize_cache(
                        actual[payload_index].numpy(),
                        actual[scale_index].numpy(),
                        actual[zero_index].numpy(),
                        valid_length,
                    ),
                    _dequantize_cache(
                        expected[payload_index].numpy(),
                        expected[scale_index].numpy(),
                        expected[zero_index].numpy(),
                        valid_length,
                    ),
                    name,
                    cache_value_split,
                    cache_value_atol,
                    cache_value_rtol,
                    cache_value_max_exceed_fraction,
                    cache_value_max_relative_l2,
                    cache_value_min_cosine,
                )
            except AssertionError as error:
                failures.append(str(error))
    if failures:
        raise AssertionError("\n".join(failures))


def _build_bound_model(
    model,
    sample_inputs,
    parameters,
    target,
    layout_policy,
    inplace_kv_cache=False,
):
    exported = torch.export.export(model, sample_inputs, strict=True)
    mod = from_exported_program(
        exported,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    parameter_vars = list(mod["main"].params[2 : 2 + len(parameters)])
    assert len(parameter_vars) == len(parameters)
    bound_parameters = {
        str(var): tvm.runtime.tensor(value.numpy())
        for var, value in zip(parameter_vars, parameters.values())
    }
    mod = relax.transform.BindParams("main", bound_parameters)(mod)
    pipeline = relax.backend.vortex.get_default_pipeline(
        target,
        layout_policy=layout_policy,
        inplace_kv_cache=inplace_kv_cache,
    )
    start = time.perf_counter()
    executable = relax.build(
        mod,
        target,
        relax_pipeline=pipeline,
        exec_mode="bytecode",
    )
    return executable, time.perf_counter() - start


def _build_external_model(
    model, sample_inputs, target, layout_policy, exec_mode="bytecode"
):
    exported = torch.export.export(model, sample_inputs, strict=True)
    mod = from_exported_program(
        exported,
        run_ep_decomposition=False,
        unwrap_unit_return_tuple=True,
    )
    start = time.perf_counter()
    executable = relax.build(
        mod,
        target,
        relax_pipeline=relax.backend.vortex.get_default_pipeline(
            target, layout_policy=layout_policy
        ),
        exec_mode=exec_mode,
    )
    return executable, time.perf_counter() - start


def _deterministic_stack_parameters(config, num_layers):
    generator = torch.Generator().manual_seed(20260831)
    parameters = {}
    for name, (shape, dtype) in stack_parameter_shapes(config, num_layers).items():
        if dtype == torch.uint8:
            parameters[name] = torch.randint(
                0, 256, shape, dtype=dtype, generator=generator
            )
        elif dtype == torch.int16:
            parameters[name] = torch.randint(
                -2, 3, shape, dtype=dtype, generator=generator
            )
        elif name.endswith("norm.weight"):
            parameters[name] = torch.ones(shape, dtype=dtype)
        else:
            parameters[name] = torch.full(shape, 1.0 / 256.0, dtype=dtype)
    return parameters


def _chunk_parameter_names(config, chunk_layers, layer_offset):
    names = []
    for local_name in stack_parameter_shapes(config, chunk_layers):
        _, local_index, suffix = local_name.split(".", 2)
        names.append(f"layers.{layer_offset + int(local_index)}.{suffix}")
    return tuple(names)


def _local_chunk_parameters(config, chunk_layers, layer_offset, parameters):
    local_names = tuple(stack_parameter_shapes(config, chunk_layers))
    global_names = _chunk_parameter_names(config, chunk_layers, layer_offset)
    return {
        local_name: parameters[global_name]
        for local_name, global_name in zip(local_names, global_names)
    }


def _copy_prefill_chunk_state(state, device):
    return tuple(tensor.copyto(device) for tensor in state)


def _copy_decode_chunk_outputs(state, device):
    return (
        state[0].copyto(device),
        *state[1:7],
        state[7].copyto(device),
    )


def test_stack_chunk_parameter_mapping_preserves_local_order():
    config = Llama3ExportConfig(
        batch_size=1,
        query_length=1,
        cache_capacity=2,
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        weight_group_size=32,
        kv_group_size=32,
    )
    parameters = _deterministic_stack_parameters(config, 8)
    chunk = _local_chunk_parameters(config, 4, 4, parameters)
    assert tuple(chunk) == tuple(stack_parameter_shapes(config, 4))
    assert chunk["layers.0.q_proj.qweight"] is parameters[
        "layers.4.q_proj.qweight"
    ]
    assert chunk["layers.3.down_proj.zeros"] is parameters[
        "layers.7.down_proj.zeros"
    ]


def _deterministic_full_model_parameters(config, num_layers):
    parameters = _deterministic_stack_parameters(config, num_layers)
    generator = torch.Generator().manual_seed(20260902)
    for name, (shape, dtype) in full_model_parameter_shapes(
        config, num_layers
    ).items():
        if name in parameters:
            continue
        if dtype == torch.uint8:
            parameters[name] = torch.randint(
                0, 256, shape, dtype=dtype, generator=generator
            )
        elif dtype == torch.int16:
            parameters[name] = torch.randint(
                -2, 3, shape, dtype=dtype, generator=generator
            )
        elif name.endswith("norm.weight"):
            parameters[name] = torch.ones(shape, dtype=dtype)
        else:
            parameters[name] = (
                torch.randn(shape, generator=generator) * (1.0 / 256.0)
            ).to(dtype)
    return parameters


def _assert_full_model_outputs(actual, expected, valid_length):
    np.testing.assert_allclose(
        actual[0].numpy(),
        expected[0].numpy(),
        rtol=0.12,
        atol=0.08,
        equal_nan=False,
        err_msg="logits",
    )
    _assert_layer_outputs(actual[1:], expected[1:], valid_length)


def _enable_launch_trace(vm, scope):
    if os.environ.get("TVM_VORTEX_TRACE_LLAMA3_C4") != "1":
        return
    launch_index = 0

    def trace_launch(unused_func, name, before_run, unused_ret_value, *unused_args):
        nonlocal launch_index
        print(
            json.dumps(
                {
                    "scope": scope,
                    "launch_index": launch_index,
                    "phase": "before" if before_run else "after",
                    "name": name,
                }
            ),
            flush=True,
        )
        if not before_run:
            launch_index += 1
        return relax.VMInstrumentReturnKind.NO_OP

    vm.set_instrument(trace_launch)


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_LLAMA3_C4") != "1",
    reason="set TVM_VORTEX_RUN_LLAMA3_C4=1 in the pinned C4 U55C environment",
)
@pytest.mark.parametrize("layout_policy", ["alone", "fused"])
@pytest.mark.parametrize("case_name", list(CASES))
def test_llama3_one_layer_prefill_c4_hardware(case_name, layout_policy):
    """Run a real-geometry Llama3 layer and compare hidden plus exact KV4 state."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    batch, prompt_length, capacity = CASES[case_name]
    config = Llama3ExportConfig(
        batch_size=batch,
        query_length=prompt_length,
        cache_capacity=capacity,
    )
    parameters = _deterministic_parameters(config)
    generator = torch.Generator().manual_seed(20260829 + batch * 10 + prompt_length)
    hidden_shape = (batch, prompt_length, config.hidden_size)
    hidden = (torch.randn(hidden_shape, generator=generator) * 0.05).to(torch.float16)
    positions = torch.arange(prompt_length, dtype=torch.int64).repeat(batch, 1)
    model = Llama3LayerPrefill(config)

    start = time.perf_counter()
    with torch.no_grad():
        expected = model(hidden, positions, parameters)
    eager_seconds = time.perf_counter() - start

    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    executable, build_seconds = _build_bound_model(
        model,
        (hidden, positions, parameters),
        parameters,
        target,
        layout_policy,
    )

    device = tvm.vortex(0)
    vm = relax.VirtualMachine(executable, device=device, memory_cfg="naive")
    _enable_launch_trace(vm, "prefill")
    host_inputs = [hidden.numpy(), positions.numpy()]
    device_inputs = [
        tvm.runtime.tensor(value, device=device) for value in host_inputs
    ]
    start = time.perf_counter()
    actual = vm["main"](*device_inputs)
    run_seconds = time.perf_counter() - start
    _assert_layer_outputs(actual, expected, prompt_length)

    print(
        json.dumps(
            {
                "case": case_name,
                "layout_policy": layout_policy,
                "eager_seconds": eager_seconds,
                "build_seconds": build_seconds,
                "run_seconds": run_seconds,
                "profile_fingerprint": profile.fingerprint,
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_LLAMA3_C4_DECODE") != "1",
    reason="set TVM_VORTEX_RUN_LLAMA3_C4_DECODE=1 for the stateful U55C chain",
)
@pytest.mark.parametrize("layout_policy", ["alone", "fused"])
@pytest.mark.parametrize("case_name", list(CASES))
def test_llama3_one_layer_prefill_decode_c4_hardware(case_name, layout_policy):
    """Feed resident prefill cache tensors through three one-token decode calls."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    batch, prompt_length, capacity = CASES[case_name]
    prefill_config = Llama3ExportConfig(batch, prompt_length, capacity)
    decode_config = Llama3ExportConfig(batch, 1, capacity)
    parameters = _deterministic_parameters(prefill_config)
    prefill_model = Llama3LayerPrefill(prefill_config)
    decode_model = Llama3LayerDecode(decode_config)
    generator = torch.Generator().manual_seed(20260900 + batch * 10 + prompt_length)
    prompt = (torch.randn((batch, prompt_length, 4096), generator=generator) * 0.05).to(
        torch.float16
    )
    prompt_positions = torch.arange(prompt_length, dtype=torch.int64).repeat(batch, 1)

    with torch.no_grad():
        expected_state = prefill_model(prompt, prompt_positions, parameters)

    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    prefill_executable, prefill_build_seconds = _build_bound_model(
        prefill_model,
        (prompt, prompt_positions, parameters),
        parameters,
        target,
        layout_policy,
    )
    sample_hidden = torch.zeros((batch, 1, 4096), dtype=torch.float16)
    sample_positions = torch.full((batch, 1), prompt_length, dtype=torch.int64)
    decode_executable, decode_build_seconds = _build_bound_model(
        decode_model,
        (sample_hidden, sample_positions, parameters, *expected_state[1:]),
        parameters,
        target,
        layout_policy,
        inplace_kv_cache=True,
    )

    device = tvm.vortex(0)
    prefill_vm = relax.VirtualMachine(prefill_executable, device=device, memory_cfg="naive")
    decode_vm = relax.VirtualMachine(decode_executable, device=device, memory_cfg="naive")
    _enable_launch_trace(prefill_vm, "prefill")
    _enable_launch_trace(decode_vm, "decode")
    prompt_device = tvm.runtime.tensor(prompt.numpy(), device=device)
    prompt_positions_device = tvm.runtime.tensor(prompt_positions.numpy(), device=device)
    start = time.perf_counter()
    actual_state = prefill_vm["main"](prompt_device, prompt_positions_device)
    prefill_seconds = time.perf_counter() - start
    _assert_layer_outputs(actual_state, expected_state, prompt_length)

    decode_seconds = []
    for step in range(3):
        old_length = prompt_length + step
        hidden = (torch.randn((batch, 1, 4096), generator=generator) * 0.05).to(torch.float16)
        positions = torch.full((batch, 1), old_length, dtype=torch.int64)
        with torch.no_grad():
            expected_state = decode_model(
                hidden,
                positions,
                parameters,
                *expected_state[1:],
            )
        previous_cache = [actual_state[index].numpy() for index in range(1, 7)]
        input_cache_tensors = list(actual_state[1:7])
        device_inputs = [
            tvm.runtime.tensor(hidden.numpy(), device=device),
            tvm.runtime.tensor(positions.numpy(), device=device),
            *actual_state[1:],
        ]
        start = time.perf_counter()
        actual_state = decode_vm["main"](*device_inputs)
        decode_seconds.append(time.perf_counter() - start)
        for index, input_cache in enumerate(input_cache_tensors, start=1):
            assert actual_state[index] == input_cache
        _assert_layer_outputs(actual_state, expected_state, old_length + 1)
        for cache_index, previous in enumerate(previous_cache, start=1):
            current = actual_state[cache_index].numpy()
            np.testing.assert_array_equal(
                current[..., :old_length, :],
                previous[..., :old_length, :],
                err_msg=f"cache {cache_index} changed its valid prefix at step {step}",
            )
            np.testing.assert_array_equal(
                current[..., old_length + 1 :, :],
                previous[..., old_length + 1 :, :],
                err_msg=f"cache {cache_index} changed its suffix at step {step}",
            )

    if case_name == "S4" and layout_policy == "fused":
        cache_before_overflow = [actual_state[index].numpy() for index in range(1, 7)]
        invalid_inputs = [
            tvm.runtime.tensor(hidden.numpy(), device=device),
            tvm.runtime.tensor(positions.numpy(), device=device),
            *actual_state[1:7],
            tvm.runtime.tensor(np.array(capacity, dtype="int64"), device=device),
        ]
        with pytest.raises(AssertionError, match="allocated KV cache capacity"):
            decode_vm["main"](*invalid_inputs)
        for cache_index, expected_cache in enumerate(cache_before_overflow, start=1):
            np.testing.assert_array_equal(
                actual_state[cache_index].numpy(),
                expected_cache,
                err_msg=f"cache {cache_index} changed after rejected overflow",
            )

    print(
        json.dumps(
            {
                "case": case_name,
                "layout_policy": layout_policy,
                "prefill_build_seconds": prefill_build_seconds,
                "decode_build_seconds": decode_build_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "profile_fingerprint": profile.fingerprint,
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_LLAMA3_C4_STACK") != "1",
    reason="set TVM_VORTEX_RUN_LLAMA3_C4_STACK=1 for the multi-layer U55C smoke test",
)
@pytest.mark.parametrize("layout_policy", ["alone", "fused"])
@pytest.mark.parametrize("num_layers", [2, 4, 32])
@pytest.mark.parametrize("geometry", ["tiny", "real"])
def test_llama3_multi_layer_external_archive_c4_hardware(
    geometry, num_layers, layout_policy, tmp_path
):
    """Reuse one physical parameter upload through multiple layers and decodes."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    if geometry == "real":
        if os.environ.get("TVM_VORTEX_RUN_LLAMA3_C4_REAL_STACK") != "1":
            pytest.skip("set TVM_VORTEX_RUN_LLAMA3_C4_REAL_STACK=1 for real geometry")
        requested_layers = int(
            os.environ.get("TVM_VORTEX_LLAMA3_C4_REAL_STACK_LAYERS", "2")
        )
        if num_layers != requested_layers:
            pytest.skip(f"real-geometry run selected {requested_layers} layers")
        config = Llama3ExportConfig(1, 1, 8)
    else:
        if num_layers == 32:
            pytest.skip("32-layer acceptance uses real Llama3 geometry")
        config = Llama3ExportConfig(
            batch_size=1,
            query_length=1,
            cache_capacity=8,
            hidden_size=128,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            weight_group_size=32,
            kv_group_size=32,
        )
    canonical_parameters = _deterministic_stack_parameters(config, num_layers)
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    specs = llama3_c4_weight_specs(
        num_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    manifest_path = prepare_c4_parameter_archive(
        tmp_path / f"{geometry}-{layout_policy}",
        {name: value.numpy() for name, value in canonical_parameters.items()},
        specs,
        target,
        profile.fingerprint,
        num_layers,
    )
    archive = C4ParameterArchive(manifest_path, profile.fingerprint, num_layers)
    compiled_layers = (
        int(os.environ.get("TVM_VORTEX_LLAMA3_C4_COMPILED_LAYERS", "1"))
        if num_layers == 32
        else num_layers
    )
    assert num_layers % compiled_layers == 0
    chunk_offsets = tuple(range(0, num_layers, compiled_layers))
    local_parameter_order = tuple(stack_parameter_shapes(config, compiled_layers))
    first_chunk_global_names = _chunk_parameter_names(config, compiled_layers, 0)
    physical_parameters = {
        local_name: torch.from_numpy(np.array(archive.tensor(global_name), copy=True))
        for local_name, global_name in zip(
            local_parameter_order, first_chunk_global_names
        )
    }
    canonical_chunks = [
        _local_chunk_parameters(
            config, compiled_layers, layer_offset, canonical_parameters
        )
        for layer_offset in chunk_offsets
    ]

    generator = torch.Generator().manual_seed(20260901)
    hidden = (
        torch.randn((1, 1, config.hidden_size), generator=generator) * 0.05
    ).to(torch.float16)
    positions = torch.zeros((1, 1), dtype=torch.int64)
    eager_prefill = Llama3StackPrefill(config, compiled_layers)
    expected_chunk_states = []
    expected_hidden = hidden
    for chunk_parameters in canonical_chunks:
        expected_state = eager_prefill(
            expected_hidden, positions, chunk_parameters
        )
        expected_hidden = expected_state[0]
        expected_chunk_states.append(expected_state)
    physical_prefill = Llama3StackPrefill(
        config, compiled_layers, prepacked_weights=True
    )
    prefill_executable, prefill_build_seconds = _build_external_model(
        physical_prefill,
        (hidden, positions, physical_parameters),
        target,
        layout_policy,
    )
    physical_decode = Llama3StackDecode(
        config, compiled_layers, prepacked_weights=True
    )
    decode_executable, decode_build_seconds = _build_external_model(
        physical_decode,
        (
            hidden,
            positions,
            physical_parameters,
            *expected_chunk_states[0][1:],
        ),
        target,
        layout_policy,
    )

    device = tvm.vortex(0)
    resident_parameters = archive.upload(device)
    assert archive.upload(device) is resident_parameters
    chunk_parameter_inputs = [
        [resident_parameters[name] for name in global_names]
        for global_names in (
            _chunk_parameter_names(config, compiled_layers, layer_offset)
            for layer_offset in chunk_offsets
        )
    ]
    prefill_vm = relax.VirtualMachine(prefill_executable, device=device, memory_cfg="naive")
    comparison_kwargs = (
        {
            "cache_code_mismatch_rate": 0.10 if num_layers == 32 else 0.05,
            "cache_zero_mismatch_rate": 0.04,
            "hybrid_fp16": True,
            "hidden_atol": 0.25,
            "hidden_rtol": 0.15,
            "hidden_max_exceed_fraction": 0.02,
            "hidden_max_relative_l2": 0.05,
            "hidden_min_cosine": 0.995,
            "scale_atol": 0.003,
            "scale_rtol": 0.05,
            "scale_max_exceed_fraction": 0.01,
            "scale_max_relative_l2": 0.03,
            "scale_min_cosine": 0.999,
            "cache_value_max_exceed_fraction": 0.055 if num_layers == 32 else 0.05,
            "cache_value_max_relative_l2": 0.055 if num_layers == 32 else 0.05,
        }
        if geometry == "real"
        else {}
    )
    start = time.perf_counter()
    actual_hidden = tvm.runtime.tensor(hidden.numpy(), device=device)
    positions_device = tvm.runtime.tensor(positions.numpy(), device=device)
    actual_chunk_states = []
    for parameter_inputs in chunk_parameter_inputs:
        actual_state = prefill_vm["main"](
            actual_hidden,
            positions_device,
            *parameter_inputs,
        )
        if len(chunk_offsets) > 1:
            actual_state = _copy_prefill_chunk_state(actual_state, device)
        actual_hidden = actual_state[0]
        actual_chunk_states.append(actual_state)
    prefill_seconds = time.perf_counter() - start
    for chunk_index, (actual_state, expected_state) in enumerate(
        zip(actual_chunk_states, expected_chunk_states)
    ):
        try:
            _assert_layer_outputs(actual_state, expected_state, 1, **comparison_kwargs)
        except AssertionError as error:
            first_layer = chunk_offsets[chunk_index]
            raise AssertionError(
                f"chunk {chunk_index} layers {first_layer}-"
                f"{first_layer + compiled_layers - 1}: {error}"
            ) from error

    del prefill_vm
    gc.collect()
    decode_vm = relax.VirtualMachine(decode_executable, device=device, memory_cfg="naive")

    eager_decode = Llama3StackDecode(config, compiled_layers)
    decode_seconds = []
    for step in range(3):
        old_length = 1 + step
        hidden = (
            torch.randn((1, 1, config.hidden_size), generator=generator) * 0.05
        ).to(torch.float16)
        positions = torch.full((1, 1), old_length, dtype=torch.int64)
        expected_hidden = hidden
        next_expected_chunk_states = []
        for chunk_parameters, expected_state in zip(
            canonical_chunks, expected_chunk_states
        ):
            expected_state = eager_decode(
                expected_hidden,
                positions,
                chunk_parameters,
                *expected_state[1:],
            )
            expected_hidden = expected_state[0]
            next_expected_chunk_states.append(expected_state)
        start = time.perf_counter()
        actual_hidden = tvm.runtime.tensor(hidden.numpy(), device=device)
        positions_device = tvm.runtime.tensor(positions.numpy(), device=device)
        next_actual_chunk_states = []
        for parameter_inputs, actual_state in zip(
            chunk_parameter_inputs, actual_chunk_states
        ):
            actual_state = decode_vm["main"](
                actual_hidden,
                positions_device,
                *parameter_inputs,
                *actual_state[1:],
            )
            if len(chunk_offsets) > 1:
                actual_state = _copy_decode_chunk_outputs(actual_state, device)
            actual_hidden = actual_state[0]
            next_actual_chunk_states.append(actual_state)
        decode_seconds.append(time.perf_counter() - start)
        for chunk_index, (actual_state, expected_state) in enumerate(
            zip(next_actual_chunk_states, next_expected_chunk_states)
        ):
            try:
                _assert_layer_outputs(
                    actual_state,
                    expected_state,
                    old_length + 1,
                    **comparison_kwargs,
                )
            except AssertionError as error:
                first_layer = chunk_offsets[chunk_index]
                raise AssertionError(
                    f"decode step {step} chunk {chunk_index} layers {first_layer}-"
                    f"{first_layer + compiled_layers - 1}: {error}"
                ) from error
        actual_chunk_states = next_actual_chunk_states
        expected_chunk_states = next_expected_chunk_states

    print(
        json.dumps(
            {
                "layers": num_layers,
                "compiled_layers": compiled_layers,
                "chunks": len(chunk_offsets),
                "geometry": geometry,
                "layout_policy": layout_policy,
                "prefill_build_seconds": prefill_build_seconds,
                "decode_build_seconds": decode_build_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "resident_parameter_bytes": archive.manifest["data_nbytes"],
                "profile_fingerprint": profile.fingerprint,
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(
    os.environ.get("TVM_VORTEX_RUN_LLAMA3_C4_FULL_MODEL") != "1",
    reason="set TVM_VORTEX_RUN_LLAMA3_C4_FULL_MODEL=1 for token-to-logits U55C",
)
@pytest.mark.parametrize("layout_policy", ["alone", "fused"])
@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_llama3_full_model_external_archive_c4_hardware(
    exec_mode, layout_policy, tmp_path
):
    """Run token embedding, decoder, final norm, and W4 LM head on U55C."""

    assert Path(os.environ["XRT_XCLBIN_PATH"]).resolve() == IMPROVED_XCLBIN.resolve()
    num_layers = 2
    config = Llama3ExportConfig(
        batch_size=1,
        query_length=1,
        cache_capacity=8,
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        weight_group_size=32,
        kv_group_size=32,
        vocabulary_size=64,
    )
    canonical_parameters = _deterministic_full_model_parameters(config, num_layers)
    profile = load_vortex_accelerator_profile(
        IMPROVED_XCLBIN.parent.parent / "manifest.json"
    )
    target = tvm.target.Target(profile.target, host="llvm")
    specs = llama3_c4_weight_specs(
        num_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocabulary_size=config.vocabulary_size,
    )
    manifest_path = prepare_c4_parameter_archive(
        tmp_path / layout_policy,
        {name: value.numpy() for name, value in canonical_parameters.items()},
        specs,
        target,
        profile.fingerprint,
        num_layers,
    )
    archive = C4ParameterArchive(manifest_path, profile.fingerprint, num_layers)
    parameter_order = tuple(full_model_parameter_shapes(config, num_layers))
    physical_parameters = {
        name: torch.from_numpy(np.array(archive.tensor(name), copy=True))
        for name in parameter_order
    }
    token_ids = torch.tensor([[3]], dtype=torch.int64)
    positions = torch.zeros((1, 1), dtype=torch.int64)
    eager_prefill = Llama3ModelPrefill(config, num_layers)
    expected_state = eager_prefill(token_ids, positions, canonical_parameters)
    physical_prefill = Llama3ModelPrefill(
        config, num_layers, prepacked_weights=True
    )
    prefill_executable, prefill_build_seconds = _build_external_model(
        physical_prefill,
        (token_ids, positions, physical_parameters),
        target,
        layout_policy,
        exec_mode,
    )
    physical_decode = Llama3ModelDecode(
        config, num_layers, prepacked_weights=True
    )
    decode_executable, decode_build_seconds = _build_external_model(
        physical_decode,
        (
            token_ids,
            positions,
            physical_parameters,
            *expected_state[2:],
        ),
        target,
        layout_policy,
        exec_mode,
    )

    prefill_artifact = tmp_path / layout_policy / f"prefill-{exec_mode}.so"
    decode_artifact = tmp_path / layout_policy / f"decode-{exec_mode}.so"
    prefill_executable.export_library(str(prefill_artifact))
    decode_executable.export_library(str(decode_artifact))
    prefill_executable = tvm.runtime.load_module(str(prefill_artifact))
    decode_executable = tvm.runtime.load_module(str(decode_artifact))

    device = tvm.vortex(0)
    resident_parameters = archive.upload(device)
    parameter_inputs = [resident_parameters[name] for name in parameter_order]
    prefill_vm = relax.VirtualMachine(prefill_executable, device=device, memory_cfg="naive")
    decode_vm = relax.VirtualMachine(decode_executable, device=device, memory_cfg="naive")
    start = time.perf_counter()
    actual_state = prefill_vm["main"](
        tvm.runtime.tensor(token_ids.numpy(), device=device),
        tvm.runtime.tensor(positions.numpy(), device=device),
        *parameter_inputs,
    )
    prefill_seconds = time.perf_counter() - start
    _assert_full_model_outputs(actual_state, expected_state, 1)

    eager_decode = Llama3ModelDecode(config, num_layers)
    decode_seconds = []
    for step in range(3):
        token_ids = torch.tensor([[4 + step]], dtype=torch.int64)
        positions = torch.tensor([[1 + step]], dtype=torch.int64)
        expected_state = eager_decode(
            token_ids,
            positions,
            canonical_parameters,
            *expected_state[2:],
        )
        start = time.perf_counter()
        actual_state = decode_vm["main"](
            tvm.runtime.tensor(token_ids.numpy(), device=device),
            tvm.runtime.tensor(positions.numpy(), device=device),
            *parameter_inputs,
            *actual_state[2:],
        )
        decode_seconds.append(time.perf_counter() - start)
        _assert_full_model_outputs(actual_state, expected_state, step + 2)

    print(
        json.dumps(
            {
                "boundary": "token_to_logits",
                "layers": num_layers,
                "layout_policy": layout_policy,
                "exec_mode": exec_mode,
                "prefill_build_seconds": prefill_build_seconds,
                "decode_build_seconds": decode_build_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "resident_parameter_bytes": archive.manifest["data_nbytes"],
                "profile_fingerprint": profile.fingerprint,
            },
            sort_keys=True,
        )
    )
