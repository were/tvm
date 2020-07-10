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
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""x86 dense operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas

from .util import get_fp32_len
from .. import generic, tag
from ..util import traverse_inline, get_const_tuple
from ..nn.dense import dense_alter_layout, dense_legalize
from .conv2d_int8 import is_int8_hw_support
from ..util import get_const_tuple

def _schedule_dense_pack_template(cfg, s, C):
    A, packedB = s[C].op.input_tensors

    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis

    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packedB].op.axis
    s[packedB].reorder(z, x, y)
    s[packedB].parallel(z)
    return s


def _schedule_dense_nopack_template(cfg, s, C):
    y, x = s[C].op.axis
    kk, = s[C].op.reduce_axis
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    CC, = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    return s


def _default_dense_pack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, tvm.tir.Var):
        M = 16
    if isinstance(N, tvm.tir.Var):
        N = 16
    if isinstance(K, tvm.tir.Var):
        K = 16

    vec_width = get_fp32_len()
    tilex_ii = 1
    for bn in range(vec_width*2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K, 1])


def _default_dense_nopack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, tvm.tir.Var):
        M = 16
    if isinstance(N, tvm.tir.Var):
        N = 16
    if isinstance(K, tvm.tir.Var):
        K = 16

    vec_width = get_fp32_len()
    tilek_bn = 1
    for bn in range(vec_width*2, 0, -1):
        if K % bn == 0:
            tilek_bn = bn
            break
    cfg["tile_k"] = SplitEntity([K // tilek_bn, tilek_bn])
    cfg["tile_x"] = SplitEntity([N, 1])
    cfg["tile_y"] = SplitEntity([1, M])

@autotvm.register_topi_compute("dense_nopack.x86")
def dense_nopack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense without packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split("tile_y", 32 if isinstance(M, tvm.tir.Var) else M, num_outputs=2)
    cfg.define_split("tile_x", 32 if isinstance(N, tvm.tir.Var) else N, num_outputs=2)
    cfg.define_split("tile_k", 32 if isinstance(K, tvm.tir.Var) else K, num_outputs=2)
    if cfg.is_fallback:
        _default_dense_nopack_config(cfg, M, N, K)

    vec = cfg["tile_k"].size[-1]
    k = te.reduce_axis((0, K // vec), "k")
    CC = te.compute((M, N, vec),
                    lambda z, y, x: te.sum(
                        data[z, k * vec + x].astype(out_dtype) *
                        weight[y, k * vec + x].astype(out_dtype), axis=k))

    kk = te.reduce_axis((0, vec), "kk")
    C = te.compute((M, N),
                   lambda y, x: te.sum(CC[y, x, kk], axis=kk),
                   tag="dense_nopack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                       tag=tag.BROADCAST)
    return C

@autotvm.register_topi_compute("dense_dotprod")
def dense_dotprod(cfg, data, weight, bias=None, out_dtype=None, out_lanes=16, red_lanes=4):
    """Compute dense without packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    n, k = get_const_tuple(data.shape)
    if len(weight.shape) == 2:
        m, kk = get_const_tuple(weight.shape)
        weight = te.compute((n // out_lanes, kk // red_lanes, out_lanes, red_lanes),
                            lambda i, j, k, l: weight[i * out_lanes + k, j * red_lanes + l])
        assert kk == k
    else:
        m, kk, out_lanes, red_lanes = get_const_tuple(weight.shape)


    red = te.reduce_axis((0, k), "k")
    C = te.compute((n, m),
                    lambda x, y: te.sum(
                        data[x, red].astype(out_dtype) *
                        weight[y // out_lanes, red // red_lanes, y % out_lanes, red % red_lanes].astype(out_dtype), axis=red),
                    tag='dense_dotprod', name='dense_dotprod')

    if bias is not None:
        C = te.compute((n, m), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                       tag=tag.BROADCAST)

    return C


@autotvm.register_topi_schedule("dense_nopack.x86")
def schedule_dense_nopack(cfg, outs):
    """Create the schedule for dense_nopack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'dense_nopack' in op.tag:
            _schedule_dense_nopack_template(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute("dense_pack.x86")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape) # batch, in_dim
    N, _ = get_const_tuple(weight.shape) # out_dim
    # create tuning space
    cfg.define_split("tile_y", M, num_outputs=3)
    cfg.define_split("tile_x", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, M, N, K)

    packw_bn = cfg["tile_x"].size[-1]
    packw_shape = (N // packw_bn, K, packw_bn)
    packw = te.compute(packw_shape,
                       lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N),
                   lambda y, x: te.sum(
                       data[y, k].astype(out_dtype) *
                       packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
                       axis=k),
                   tag="dense_pack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                       tag=tag.BROADCAST)
    return C

@autotvm.register_topi_schedule("dense_pack.x86")
def schedule_dense_pack(cfg, outs):
    """Create the schedule for dense_pack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _schedule_dense_pack_template(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute("dense_cblas.x86")
def dense_cblas(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using cblas library"""
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    cfg.add_flop(M * K * N * 2)
    C = cblas.matmul(data, weight, False, True)
    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                       tag=tag.BROADCAST)
    return C

@autotvm.register_topi_schedule("dense_cblas.x86")
def schedule_dense_cblas(_, outs):
    """Create schedule for dense_cblas"""
    return generic.schedule_extern(outs)

@dense_alter_layout.register("cpu")
def _dense_alter_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    _, outs = relay.backend.compile_engine.select_implementation(
        relay.op.get("nn.dense"), attrs, tinfos, out_type, target)
    
    workload = autotvm.task.get_workload(outs)

    print('legalize', workload)

    if workload is None:
        # The best implementation is not an AutoTVM template,
        # we then assume it's not necessary to alter this op.
        return None
    cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]
    new_attrs = {k : attrs[k] for k in attrs.keys()}

    print('alter dense template: ', topi_tmpl)

    if topi_tmpl == 'dense_dotprod':
        data_expr, weight_expr = inputs
        data_tensor, weight_tensor = tinfos
        N, kk = get_const_tuple(data_tensor.shape)
        M, K = get_const_tuple(weight_tensor.shape)
        assert kk == K

        weight_MK = weight_expr
        weight_MKk = relay.reshape(weight_MK, (M, K // 4, 4))
        weight_KkM = relay.transpose(weight_MKk, axes=(1, 2, 0))
        weight_KkMm = relay.reshape(weight_KkM, (K // 4, 4, M // 16, 16))
        weight_MKmk = relay.transpose(weight_KkMm, axes=(2, 0, 3, 1))

        # update new attrs
        new_attrs['out_lanes'] = 16
        new_attrs['red_lanes'] = 4

        # Store altered operator's config.
        new_data = te.placeholder((N, kk), dtype=data_tensor.dtype)
        new_weight = te.placeholder((M // 16, K // 4, 16, 4), dtype=weight_tensor.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_weight, new_attrs['units'], new_attrs['out_type'], 16, 4], topi_tmpl)
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.denseDotProd(data_expr, weight_MKmk, **new_attrs)


    return None


@dense_legalize.register("cpu")
def _dense_legalize(attrs, inputs, arg_types):

    # Collect the input tensors.
    data_tensor, weight_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype
    weight_dtype = weight_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, weight = inputs

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    n, k = get_const_tuple(data_tensor.shape)
    m, kk = get_const_tuple(weight_tensor.shape)

    is_int8_inputs = False
    # If both the inputs are int8, we can add 128 to make the input dtype uint8, and then adjust the
    # output. This will help picking up Intel VNNI instructions.
    # Original --> C = A (conv) B
    # A and B are int8
    #   C = (A + 128 - 128) (conv) B
    #   C = (A' conv B) - 128 (conv) B
    # where A' = A + 128
    # and 128 (conv) B is basically a reduce on CRS axis for weights.
    if data_tensor.dtype == 'int8' and weight_tensor.dtype == 'int8':
        is_int8_inputs = True

        data = relay.cast(data, 'int32')
        data = relay.add(data, relay.const(128, 'int32'))
        data = relay.cast(data, 'uint8')

        # The data type is now shifted to uint8
        data_dtype = 'uint8'

        # Multiply 128 to adjust shift.
        adjust_weight = relay.multiply(weight, relay.const(128, 'int32'))
        adjust_weight = relay.sum(adjust_weight, axis=[1])
        adjust_weight = relay.reshape(adjust_weight, (1, m))

    # Legalize if the datatypes are suitable for fast Int8 instructions.  Int8 instructions require
    # input channel to be a multiple of 4 and output channels to be a multiple of 16. For input
    # channels, we pad both the inputs and weights input channels. For output channels, we pad the
    # weight and stride_slice the output.
    if is_int8_hw_support(data_dtype, weight_dtype):

        assert k == kk

        if k % 4 != 0:
            diff = 4 - k % 4
            data = relay.nn.pad(data, pad_width=((0, 0), (0, diff)))
            weight = relay.nn.pad(weight, pad_width=((0, 0), (0, diff)))

        shape_changed = False
        if m % 16 != 0:
            diff = 16 - m % 16
            weight = relay.nn.pad(weight, pad_width=((0, diff), (0, 0)))
            shape_changed = True
            new_attrs['units'] = m + diff

        out = relay.nn.dense(data, weight, **new_attrs)

        if shape_changed:
            original_out_shape = [x.value for x in output_tensor.shape]
            out = relay.strided_slice(out,
                                      begin=relay.const([0, 0, 0, 0], "int32"),
                                      end=relay.const(original_out_shape, "int32"))

        if is_int8_inputs:
            out = relay.subtract(out, adjust_weight)

        return out

    return None