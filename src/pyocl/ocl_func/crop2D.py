from L0_Smoothing.pyocl.GCArray import GCArray as clArray
import pyopencl.cltypes as cltypes

def crop2D(func, queue, arr, slicey=None, slicex=None):
    sy, sx = arr.shape
    yt, yb, _ = slicey.indices(sy)
    xl, xr, _ = slicex.indices(sx)
    nsx = xr-xl; nsy = yb-yt
    cropped = clArray.empty(queue, (nsy, nsx), arr.dtype)
    func(
        queue, arr.shape, None,
        arr.data,
        cropped.data,
        cltypes.int(xl),
        cltypes.int(xr),
        cltypes.int(yt),
        cltypes.int(yb),
    )
    return cropped