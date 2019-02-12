"""Hybrid Programming APIs of TVM Python Package.

This package maps a subset of python to HalideIR so that:
1. Users can write some preliminary versions of the computation patterns
have not been supported yet and verify it across the real execution and
python semantic emulation.
2. So far, it is a text format dedicated to HalideIR Phase 0. Refer tvm.lower
for more details. A larger ambition of this module is to support all levels of
HalideIR.
"""

# TODO(@were): Make this module more complete.
# 1. Support HalideIR dumping to Hybrid Script
# 2. Support multi-level HalideIR

from __future__ import absolute_import as _abs

from .._ffi.base import decorate
from .._ffi.function import _init_api
from .. import _api_internal as _tvm_internal
from ..tensor import Tensor
from ..build_module import form_body

from .parser import parse_python
from .util import _pruned_source
from .module import HybridModule as Module


def script(pyfunc):
    """Decorate a python function function as hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    hybrid_func : function
        A decorated hybrid script function.
    """
    def wrapped_func(func, *args, **kwargs): #pylint: disable=missing-docstring
        from .runtime import _enter_hybrid_runtime, _restore_runtime
        from .util import _is_tvm_arg_types
        if _is_tvm_arg_types(args):
            src = _pruned_source(func)
            parser = parse_python(src, func.__globals__, args)

            input_tensors = []
            for i in args:
                if isinstance(i, Tensor):
                    input_tensors.append(i)
            op = _tvm_internal._HybridOp(parser.func_name, "HybridOp", None, input_tensors,
                                         parser.outputs, parser.parsed_body)
            res = [op.output(i) for i in range(len(parser.outputs))]
            return res[0] if len(res) == 1 else res

        intersect = _enter_hybrid_runtime(func)
        value = func(*args, **kwargs)
        _restore_runtime(func, intersect)
        return value

    return decorate(pyfunc, wrapped_func)


_init_api("tvm.hybrid")


def build(sch, inputs, outputs, name="hybrid_func"):
    """Dump the corrent schedule to hybrid module

    Parameters
    ----------
    sch: Schedule
        The schedule to be dumped

    inputs: An array of Tensors or Vars
        The inputs of the function body

    outputs: An array of Tensors
        The outputs of the function body

    Returns
    -------
    module: HybridModule
        The built results is wrapped in a HybridModule.
        The usage of HybridModule is roughly the same as normal TVM-built modules.
    """

    stmt = form_body(sch)
    src = dump(stmt, inputs, outputs, name)

    return Module(src, name)
