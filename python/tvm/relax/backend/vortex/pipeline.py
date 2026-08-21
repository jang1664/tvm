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
"""The Relax Vortex backend compilation pipeline."""

import tvm
from tvm import relax

from .. import gpu_generic


def library_dispatch_passes(target: tvm.target.Target):
    """Return library dispatch passes supported by Vortex."""
    return gpu_generic.library_dispatch_passes(target)


def legalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """Legalize Relax and schedule kernels for Vortex."""
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    return [
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        # The generic Matmul rule's conservative 8x8 configuration fits the
        # Vortex target contract (64 threads and a small static shared arena).
        # Rules are tried in order, so unsupported matmul-like shapes and all
        # other operators retain the safe one-dimensional fallback schedule.
        dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback()),
    ]


def dataflow_lower_passes(target: tvm.target.Target):
    """Return Relax dataflow lowering passes for Vortex."""
    return gpu_generic.dataflow_lower_passes(target)


def finalize_passes(target: tvm.target.Target):
    """Return Relax VM finalization passes for Vortex."""
    return gpu_generic.finalize_passes(target)


def get_default_pipeline(target: tvm.target.Target):
    """Return the default Relax compilation pipeline for Vortex."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            return tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )(mod)

    return _pipeline
