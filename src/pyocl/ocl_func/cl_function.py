import pyopencl as cl
import pyopencl.cltypes as cltypes

def cltype_map(dtype):
    if dtype.type==cltypes.int:
        return 'int'
    if dtype.type==cltypes.float:
        return 'float'
    if dtype.type==cltypes.double:
        return 'double'
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