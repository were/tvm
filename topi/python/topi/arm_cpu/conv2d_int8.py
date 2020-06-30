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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on ARM"""
import tvm
from tvm import te
from tvm import autotvm
from .. import tag
from ..util import get_const_tuple
from ..generic import conv2d as conv2d_generic
from .. import nn
from ..nn.conv2d import _get_workload as _get_conv2d_workload
from .tensor_intrin import dot_int8_int8_int32


def is_int8_hw_support(data_dtype, kernel_dtype):
    """
    Checks to ensure that we can use Intel DLBoost instructions
    1) The datatypes are correct.
    2) LLVM version has support for the instructions.
    3) Target is skylake and above.
    """

    # 1) Check datatypes
    is_dtype_support = data_dtype == 'int8' and kernel_dtype == 'int8'

    # 2) Check LLVM support
    llvm_version = tvm.target.codegen.llvm_version_major()
    is_llvm_support = llvm_version >= 8

    # 3) Check target
    is_target_support = False
    for i in tvm.target.Target.current().options:
        if i.startswith('-mattr=') and ('+dotprod' in i):
            is_target_support = True

    return is_dtype_support and is_llvm_support and is_target_support


def _get_default_config(cfg, data, kernel, strides, padding, out_dtype):
    """
    Get default int8 schedule config for the workload
    """
    wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype)
    is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
    if is_kernel_1x1:
        conv2d_generic.fallback_schedule_cpu_1x1_int8(
            cfg, wkl, int32_lanes=2, num_int8_elements=4)
    else:
        conv2d_generic.fallback_schedule_cpu_common_int8(
            cfg, wkl, int32_lanes=2, num_int8_elements=4)

def _pack_data(cfg, data, kernel):
    n_elems = 2
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = te.compute((n, ic_chunk, ih, iw, ic_bn),
                      lambda bs, c, h, w, vc: data[bs, c*ic_bn + vc, h, w],
                      name="data_vec")

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn//n_elems, oc_bn, n_elems),
        lambda occ, icc, k_h, k_w, icbc, ocb, icbb:
        kernel[occ * oc_bn + ocb,
               icc * ic_bn + icbc * ic_bn//n_elems + icbb, k_h, k_w],
        name="kernel_vec")

    return data, kernel

@autotvm.register_topi_compute("conv2d_NCHWc_int8.arm_cpu")
def conv2d_NCHWc_int8(cfg, data, kernel, strides,
                      padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d int8 with NCHWc layout"""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        in_channel = ic_chunk * ic_bn

        oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn, n_elems = get_const_tuple(kernel.shape)
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, ic_chunk, kh, kw = get_const_tuple(kernel.shape)
        cfg.define_split('tile_ic', in_channel, num_outputs=2,
                         filter=lambda y: y.size[-1] % 4 == 0)
        cfg.define_split('tile_oc', num_filter, num_outputs=2,
                         filter=lambda y: y.size[-1] % 4 == 0)
        data, kernel = _pack_data(cfg, data, kernel)


    # If no config was set, we can fallback to NCHW config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            te.placeholder((num_filter, in_channel, kh, kw), dtype=kernel.dtype),
                            strides, padding, out_dtype)

    return nn.conv2d_NCHWc_int8(data,
                                kernel,
                                strides,
                                padding,
                                dilation,
                                layout,
                                out_layout,
                                out_dtype)


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.arm_cpu")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_NCHWc_int8' in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, _ = get_const_tuple(kernel_vec.shape)
            dtype = "uint" if data.dtype == "uint8" else "int"
            if kh == 1 and kw == 1:
                conv2d_generic.schedule_conv_NCHWc_cpu_1x1_int8(
                    *args, int32_lanes=4, intrin=dot_int8_int8_int32(int32_lanes=4, dtype=dtype))
            else:
                conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(
                    *args, int32_lanes=4, intrin=dot_int8_int8_int32(int32_lanes=4, dtype=dtype))

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
