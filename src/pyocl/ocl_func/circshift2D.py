from L0_Smoothing.pyocl.GCArray import GCArray as clArray
import pyopencl.cltypes as cltypes

def circshift2D(func, queue, arr, shift: tuple):
    shifty, shiftx = shift
    shifted = clArray.empty_like(arr)
    func(
        queue, arr.shape, None,
        arr.data,
        shifted.data,
        cltypes.int(shiftx),
        cltypes.int(shifty)
    )
    return shifted