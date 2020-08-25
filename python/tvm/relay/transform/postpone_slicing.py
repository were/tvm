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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""
A pass for manifesting explicit memory allocations.
"""
import numpy as np
from ..expr_functor import ExprMutator
from ..scope_builder import ScopeBuilder
from . import transform
from .. import op
from ... import DataType, register_func
from .. import ty, expr
from ..backend import compile_engine
from ..op.memory import flatten_tuple_type, from_tuple_type, to_tuple_type
from ...import cpu
from ..op.memory import alloc_storage
from topi.util import get_const_tuple
from ...runtime import ndarray

class PostponeSlicingPass(ExprMutator):
    """A pass for explicitly manifesting all memory allocations in Relay."""

    def __init__(self):
        self.rewrite = False
        super().__init__()

    def visit_call(self, call):
        if isinstance(call.args[0], expr.Call) and call.args[0].op.name == 'strided_slice':
            strided_slice = super().visit_call(call.args[0])
            if str(call.op.name) in ('multiply', 'add'):
                a_shape = get_const_tuple(call.args[0].type_args[0].shape)
                b_shape = get_const_tuple(call.type_args[1].shape)
                cond = len(a_shape) == 5 and len(b_shape) == 5
                cond = cond and (1 == b_shape[0])
                cond = cond and (a_shape[1] == b_shape[1])
                cond = cond and (1 == b_shape[2])
                cond = cond and (1 == b_shape[3])
                cond = cond and (a_shape[4] == b_shape[4])
                #print('broadcasting: %s' % call.op.name, a_shape, b_shape, cond, sep='\n')
                if cond:
                    self.rewrite = True
                    bin_args = [strided_slice.args[0], call.args[1]]
                    binop = expr.Call(call.op, bin_args, call.attrs)
                    strided_args = [binop, strided_slice.args[1], strided_slice.args[2], strided_slice.args[3]]
                    res = expr.Call(strided_slice.op, strided_args, strided_slice.attrs)
                    return res
            elif str(call.op.name) in ('nn.relu', 'cast'):
                self.rewrite = True
                unary_args = [strided_slice.args[0]]
                unary_op = expr.Call(call.op, unary_args, call.attrs)
                strided_args = [unary_op, strided_slice.args[1], strided_slice.args[2], strided_slice.args[3]]
                res = expr.Call(strided_slice.op, strided_args, strided_slice.attrs)
                #print('elementwise ', call.op.name)
                return res

        return super().visit_call(call)


@transform.function_pass(opt_level=0)
class PostponeStridedSlice:
    """The explicit pass wrapper around ManifestAlloc."""
    def __init__(self):
        pass

    def transform_function(self, func, mod, _):
        return func
        transform.InferType()(mod)
        pps = PostponeSlicingPass()
        func = pps.visit(func)
        return func


register_func("relay.transform.PostponeStridedSlice", PostponeStridedSlice)
