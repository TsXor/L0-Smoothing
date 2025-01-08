import pyopencl as cl
import pyopencl.cltypes as cltypes
import numpy as np

def cltype_map(dtype):
    if dtype.type==cltypes.int:
        return 'int'
    elif dtype.type==cltypes.float:
        return 'float'
    elif dtype.type==cltypes.double:
        return 'double'
    elif dtype.type==np.int32:
        return 'int'
    elif dtype.type==np.int64:
        return 'long'
    elif dtype.type==np.complex64:
        return 'float2'
    elif dtype.type==np.complex128:
        return 'double2'
    else:
        raise NotImplementedError(dtype.type)

class CLFunction:
    def __init__(self, funcname, clcode, pyfunc):
        self.funcname = funcname
        self.clcode = clcode
        self.pyfunc = pyfunc
        self.ctxmap = {}

    def get_func(self, ctx, dtype):
        dtype = cltype_map(dtype)
        try:
            prg = self.ctxmap[(ctx, dtype)]
        except:
            prg = cl.Program(ctx, self.clcode.replace('DTYPE', dtype)).build()
            self.ctxmap[(ctx, dtype)] = prg
        func = getattr(prg, self.funcname)
        return func

    def __call__(self, arr, *args, **kwargs):
        ctx = arr.context
        queue = arr.queue
        func = self.get_func(ctx, arr.dtype)
        return self.pyfunc(func, queue, arr, *args, **kwargs)
