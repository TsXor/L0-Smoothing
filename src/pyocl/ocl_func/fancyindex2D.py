from L0_Smoothing.pyocl.GCArray import GCArray as clArray
import pyopencl.cltypes as cltypes

def fancyindex2D(func, queue, arr, mask, value=0):
    mask = mask.astype(cltypes.int)
    masked = clArray.empty_like(arr)
    func(
        queue, arr.shape, None,
        arr.data,
        mask.data,
        masked.data,
        arr.dtype.type(value)
    )
    return masked