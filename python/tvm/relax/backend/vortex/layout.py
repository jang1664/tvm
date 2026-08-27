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
"""Checked physical-layout planning for Vortex ``GEMM_IMPROVE``."""

from dataclasses import dataclass
from math import gcd


_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1


def _checked_add(name, lhs, rhs):
    if lhs < 0 or rhs < 0 or lhs > _U64_MAX - rhs:
        raise ValueError(f"Vortex GEMM layout uint64 overflow in {name}: {lhs} + {rhs}")
    return lhs + rhs


def _checked_mul(name, lhs, rhs):
    if lhs < 0 or rhs < 0 or (lhs and rhs > _U64_MAX // lhs):
        raise ValueError(f"Vortex GEMM layout uint64 overflow in {name}: {lhs} * {rhs}")
    return lhs * rhs


def _checked_align(name, value, alignment):
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"Vortex GEMM profile {name} alignment must be a power of two")
    return _checked_add(name, value, alignment - 1) // alignment * alignment


def _ceil_div(value, divisor):
    return value // divisor + int(value % divisor != 0)


def _lcm(lhs, rhs):
    return lhs // gcd(lhs, rhs) * rhs


def _target_int(target, name, default):
    attrs = getattr(target, "attrs", target)
    value = attrs.get(name, default) if hasattr(attrs, "get") else default
    return int(value)


@dataclass(frozen=True)
class ImproveProfile:
    """Versioned hardware fields that affect an IMPROVE layout."""

    dma_mt: int = 128
    dma_nt: int = 128
    dma_kt: int = 128
    mxu_kt: int = 32
    mxu_nt: int = 32
    num_dma_channels: int = 8
    tmem_bank_size: int = 64 << 10
    accumulator_depth: int = 1024
    qparam_slot_alignment: int = 512
    tmem_alignment: int = 64
    dimension_bits: int = 32
    address_bits: int = 64
    tile_counter_bits: int = 32
    job_entries: int = 4
    num_cores: int = 1
    dram_capacity_bytes: int = _U64_MAX
    gemm_abi_version: int = 2
    layout_abi_version: int = 2
    supported_qblocks: tuple = (32,)

    @staticmethod
    def from_target(target):
        return ImproveProfile(
            dma_mt=_target_int(target, "vortex_gemm_dma_mt", 128),
            dma_nt=_target_int(target, "vortex_gemm_dma_nt", 128),
            dma_kt=_target_int(target, "vortex_gemm_dma_kt", 128),
            mxu_kt=_target_int(target, "vortex_mxu_row", 32),
            mxu_nt=_target_int(target, "vortex_mxu_col", 32),
            num_dma_channels=_target_int(target, "vortex_num_dma_channels", 8),
            tmem_bank_size=_target_int(target, "vortex_tmem_bank_size", 64 << 10),
            accumulator_depth=_target_int(target, "vortex_gemm_acc_mem_depth", 1024),
            qparam_slot_alignment=_target_int(
                target, "vortex_gemm_qparam_slot_alignment", 512
            ),
            tmem_alignment=_target_int(target, "vortex_gemm_tmem_alignment", 64),
            dimension_bits=_target_int(target, "vortex_gemm_dimension_bits", 32),
            address_bits=_target_int(target, "vortex_device_address_bits", 64),
            tile_counter_bits=_target_int(target, "vortex_gemm_tile_counter_bits", 32),
            job_entries=_target_int(target, "vortex_gemm_job_entries", 4),
            num_cores=_target_int(target, "vortex_num_cores", 1),
            dram_capacity_bytes=_target_int(
                target, "vortex_dram_capacity_bytes", _U64_MAX
            ),
            gemm_abi_version=_target_int(target, "vortex_gemm_abi_version", 2),
            layout_abi_version=_target_int(target, "vortex_layout_abi_version", 2),
        )

    def validate(self):
        power_of_two = {
            "DMA_MT": self.dma_mt,
            "DMA_NT": self.dma_nt,
            "DMA_KT": self.dma_kt,
            "MXU_KT": self.mxu_kt,
            "MXU_NT": self.mxu_nt,
            "NUM_DMA_CHANNELS": self.num_dma_channels,
            "qparam slot": self.qparam_slot_alignment,
            "TMEM": self.tmem_alignment,
        }
        for name, value in power_of_two.items():
            if value <= 0 or value & (value - 1):
                raise ValueError(f"Vortex GEMM profile {name} must be a positive power of two")
        if self.dma_kt % self.mxu_kt:
            raise ValueError("Vortex GEMM profile DMA_KT must be divisible by MXU_KT")
        if self.dma_nt % self.mxu_nt:
            raise ValueError("Vortex GEMM profile DMA_NT must be divisible by MXU_NT")
        for name in (
            "tmem_bank_size",
            "accumulator_depth",
            "dimension_bits",
            "address_bits",
            "tile_counter_bits",
            "job_entries",
            "num_cores",
            "gemm_abi_version",
            "layout_abi_version",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Vortex GEMM profile {name} must be positive")
        if self.dimension_bits > 64 or self.address_bits > 64 or self.tile_counter_bits > 64:
            raise ValueError("Vortex GEMM profile integer field widths cannot exceed 64")
        if self.num_cores > self.job_entries:
            raise ValueError(
                "Vortex GEMM profile num_cores exceeds simultaneous job_entries: "
                f"{self.num_cores} > {self.job_entries}"
            )


@dataclass(frozen=True)
class ImproveTile:
    outer_m: int
    outer_n: int
    outer_k: int
    logical_m: int
    logical_n: int
    logical_k: int
    slot_m: int
    execution_n: int
    execution_k: int


@dataclass(frozen=True)
class QParamSlot:
    outer_k: int
    outer_n: int
    offset_bytes: int
    payload_bytes: int
    reserved_bytes: int
    execution_k: int
    execution_n: int


@dataclass(frozen=True)
class ImproveLayoutDescriptor:
    dtype: str
    logical_shape: tuple
    execution_shape: tuple
    dma_tiles: tuple
    micro_tiles: tuple
    row_alignment: int
    layout_abi_version: int
    padding: str

    def compatible_gemm_input(self, consumer):
        return (
            self.dtype == consumer.dtype == "float16"
            and self.logical_shape == consumer.logical_shape
            and self.execution_shape == consumer.execution_shape
            and self.dma_tiles[0] == consumer.dma_tiles[0]
            and self.micro_tiles[1] == consumer.micro_tiles[1]
            and self.row_alignment == consumer.row_alignment
            and self.layout_abi_version == consumer.layout_abi_version
            and self.padding == "neutral"
        )


@dataclass(frozen=True)
class ImproveLayoutPlan:
    profile: ImproveProfile
    logical_m: int
    logical_n: int
    logical_k: int
    execution_n: int
    execution_k: int
    qblock: int
    weight_transpose: bool
    quant_direction: int
    m_tiles: tuple
    n_tiles: tuple
    k_tiles: tuple
    tiles: tuple
    qparam_slots: tuple
    a_elements: int
    weight_bytes: int
    qparam_elements: int
    c_elements: int
    peak_live_bytes: int
    tmem_scratch_bytes: int

    @property
    def a_descriptor(self):
        return ImproveLayoutDescriptor(
            "float16",
            (self.logical_m, self.logical_k),
            (self.logical_m, self.execution_k),
            (self.profile.dma_mt, self.profile.dma_kt),
            (self.profile.mxu_nt, self.profile.mxu_kt),
            self.profile.num_dma_channels,
            self.profile.layout_abi_version,
            "neutral",
        )

    @property
    def c_descriptor(self):
        return ImproveLayoutDescriptor(
            "float16",
            (self.logical_m, self.logical_n),
            (self.logical_m, self.execution_n),
            (self.profile.dma_mt, self.profile.dma_nt),
            (self.profile.mxu_kt, self.profile.mxu_nt),
            self.profile.num_dma_channels,
            self.profile.layout_abi_version,
            "neutral",
        )


def plan_improve_layout(
    m,
    n,
    k,
    qblock,
    weight_transpose=False,
    quant_direction=0,
    profile=None,
):
    """Return the complete checked physical plan for one static rank-2 GEMM."""

    profile = profile or ImproveProfile()
    profile.validate()
    for name, value in (("M", m), ("N", n), ("K", k)):
        if value <= 0:
            raise ValueError(f"Vortex GEMM logical {name} must be positive, got {value}")
    if qblock not in profile.supported_qblocks:
        raise ValueError(
            f"Vortex GEMM QBLK={qblock} is unsupported; supported values are "
            f"{profile.supported_qblocks}"
        )
    if quant_direction not in (0, 1):
        raise ValueError(f"Vortex GEMM quant_direction must be 0 or 1, got {quant_direction}")

    n_alignment = _lcm(profile.mxu_nt, qblock if quant_direction == 1 else 1)
    k_alignment = _lcm(profile.mxu_kt, qblock if quant_direction == 0 else 1)
    execution_n = _checked_align("N execution extent", n, n_alignment)
    execution_k = _checked_align("K execution extent", k, k_alignment)
    dimension_limit = (1 << profile.dimension_bits) - 1
    for name, value in (
        ("logical M", m),
        ("logical N", n),
        ("logical K", k),
        ("execution N", execution_n),
        ("execution K", execution_k),
    ):
        if value > dimension_limit:
            raise ValueError(
                f"Vortex GEMM {name}={value} exceeds {profile.dimension_bits}-bit limit "
                f"{dimension_limit}"
            )

    m_tiles = tuple(min(profile.dma_mt, m - base) for base in range(0, m, profile.dma_mt))
    n_tiles = tuple(
        min(profile.dma_nt, execution_n - base)
        for base in range(0, execution_n, profile.dma_nt)
    )
    k_tiles = tuple(
        min(profile.dma_kt, execution_k - base)
        for base in range(0, execution_k, profile.dma_kt)
    )
    counter_limit = (1 << profile.tile_counter_bits) - 1
    for name, count in (
        ("M outer-tile count", len(m_tiles)),
        ("N outer-tile count", len(n_tiles)),
        ("K outer-tile count", len(k_tiles)),
    ):
        if count > counter_limit:
            raise ValueError(f"Vortex GEMM {name}={count} exceeds limit {counter_limit}")

    tiles = []
    for mt, cur_m in enumerate(m_tiles):
        slot_m = _checked_align("M tile slot", cur_m, profile.num_dma_channels)
        for kt, cur_k in enumerate(k_tiles):
            for nt, cur_n in enumerate(n_tiles):
                tiles.append(
                    ImproveTile(mt, nt, kt, cur_m, min(cur_n, n), min(cur_k, k), slot_m, cur_n, cur_k)
                )

    a_elements = 0
    c_elements = 0
    for cur_m in m_tiles:
        slot_m = _checked_align("M tile slot", cur_m, profile.num_dma_channels)
        a_elements = _checked_add(
            "A elements", a_elements, _checked_mul("A tile", slot_m, execution_k)
        )
        c_elements = _checked_add(
            "C elements", c_elements, _checked_mul("C tile", slot_m, execution_n)
        )
    weight_bytes = _checked_mul("packed W bytes", execution_k, execution_n) // 2

    slots = []
    qparam_bytes = 0
    ng_per_micro_n = _ceil_div(profile.mxu_nt, qblock)
    for kt, cur_k in enumerate(k_tiles):
        for nt, cur_n in enumerate(n_tiles):
            if quant_direction == 0:
                payload = _checked_mul("QCOL slot records", cur_k // qblock, cur_n)
            else:
                payload = _checked_mul(
                    "QROW slot records",
                    _checked_mul("QROW slot K/N", cur_k, cur_n // profile.mxu_nt),
                    ng_per_micro_n,
                )
            payload = _checked_mul("qparam slot bytes", payload, 2)
            reserved = _checked_align(
                "qparam slot bytes", payload, profile.qparam_slot_alignment
            )
            slots.append(QParamSlot(kt, nt, qparam_bytes, payload, reserved, cur_k, cur_n))
            qparam_bytes = _checked_add("qparam buffer bytes", qparam_bytes, reserved)

    # The native scratch contract uses two buffers for every operand category.
    tile_a = _checked_mul("TMEM input tile bytes", profile.dma_mt, profile.dma_kt * 2)
    tile_w = _checked_mul("TMEM weight tile bytes", profile.dma_kt, profile.dma_nt // 2)
    if quant_direction == 0:
        tile_q = _checked_mul(
            "TMEM QCOL tile bytes", profile.dma_kt // qblock, profile.dma_nt * 2
        )
    else:
        tile_q = _checked_mul(
            "TMEM QROW tile bytes",
            profile.dma_kt * (profile.dma_nt // profile.mxu_nt),
            ng_per_micro_n * 2,
        )
    tile_c = _checked_mul("TMEM output tile bytes", profile.dma_mt, profile.dma_nt * 2)
    scratch = 0
    for name, size in (("input", tile_a), ("weight", tile_w), ("scale", tile_q), ("zero", tile_q), ("output", tile_c)):
        for index in range(2):
            scratch = _checked_align(f"TMEM {name}{index}", scratch, profile.tmem_alignment)
            scratch = _checked_add(f"TMEM {name}{index}", scratch, size)
    tmem_capacity = _checked_mul(
        "TMEM capacity", profile.tmem_bank_size, profile.num_dma_channels
    )
    if scratch > tmem_capacity:
        raise ValueError(
            f"Vortex GEMM TMEM scratch requires {scratch} bytes, limit is {tmem_capacity}"
        )
    accumulator_records = profile.dma_mt * (profile.dma_nt // profile.mxu_nt)
    if accumulator_records > profile.accumulator_depth:
        raise ValueError(
            "Vortex GEMM accumulator double-buffer tile requires "
            f"{accumulator_records} records, GEMM_ACC_MEM_DEPTH limit is "
            f"{profile.accumulator_depth}"
        )

    a_bytes = _checked_mul("A bytes", a_elements, 2)
    c_bytes = _checked_mul("C bytes", c_elements, 2)
    peak_live = 0
    for name, size in (
        ("A", a_bytes),
        ("W", weight_bytes),
        ("scale", qparam_bytes),
        ("zero_point", qparam_bytes),
        ("C", c_bytes),
    ):
        peak_live = _checked_add(f"peak live allocation plus {name}", peak_live, size)
    address_limit = (1 << profile.address_bits) - 1
    for name, size in (
        ("A buffer bytes", a_bytes),
        ("W buffer bytes", weight_bytes),
        ("qparam buffer bytes", qparam_bytes),
        ("C buffer bytes", c_bytes),
        ("peak live allocation", peak_live),
    ):
        if size > address_limit:
            raise ValueError(
                f"Vortex GEMM {name}={size} exceeds {profile.address_bits}-bit address limit "
                f"{address_limit}"
            )
    if peak_live > profile.dram_capacity_bytes:
        raise ValueError(
            f"Vortex GEMM peak live allocation requires {peak_live} bytes, DRAM limit is "
            f"{profile.dram_capacity_bytes}"
        )

    return ImproveLayoutPlan(
        profile,
        m,
        n,
        k,
        execution_n,
        execution_k,
        qblock,
        bool(weight_transpose),
        quant_direction,
        m_tiles,
        n_tiles,
        k_tiles,
        tuple(tiles),
        tuple(slots),
        a_elements,
        weight_bytes,
        qparam_bytes // 2,
        c_elements,
        peak_live,
        scratch,
    )
