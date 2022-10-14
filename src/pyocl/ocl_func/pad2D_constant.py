from L0_Smoothing.pyocl.GCArray import GCArray as clArray
import pyopencl.cltypes as cltypes

def pad2D_constant(func, queue, arr, padv, constant):
    pady, padx = padv
    l, r = padx; t, b = pady
    sy, sx = arr.shape
    nsx = sx+l+r; nsy = sy+t+b 
    padded = clArray.empty(queue, (nsy, nsx), arr.dtype)
    func(
        queue, (nsy, nsx), None,
        arr.data,
        padded.data,
        cltypes.int(l),
        cltypes.int(r),
        cltypes.int(t),
        cltypes.int(b),
        arr.dtype.type(constant)
    )
    return padded