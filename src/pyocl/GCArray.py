'''A monkey patch to pyopencl.array with __del__ function.
Can prevent program from using too much memory when builtin operators
(+, =, *, /, **, ...) and array functions (conj(), real, imag, ...)
are used frequently.
Note: This GCArray have a bug when interoperating with reikna: when a
reikna.cluda.api.Thread object is GC-ed in python, it will somehow
destroy the Memoryobject of the array created inside it! The reason may
be that reikna have its own GC implemented.'''

# Usage: from L0_Smoothing.pyocl.GCArray import GCArray as clArray

import pyopencl.array as GCArray

def _patch_del(self):
    self.finish()
    try:
        self.base_data.release()
    except:
        pass

GCArray.Array.__del__ = _patch_del