"""
File: matlab_utils.py
Author: TsXor
Email: zhang050525@qq.com
Github: https://github.com/TsXor
Description: Implementation of matlab's psf2otf
             but with PyOpenCL and pyvkfft
             contains not only psf2otf but some other kernel functions
             *FASTER!FASTER!NOT ONLY CUDA!*

Notes: In order to understand psf2otf:

FFT does cyclic convolution. To understand what cyclic convolution is
please refer to the document below (also in the docs)
https://www.docdroid.net/YSKkZ5Y/fft-based-2d-cyclic-convolution-pdf#page=5
"""
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as cltypes
import pyopencl.array as clArray
from .ocl_func import circshift2D, pad2D_constant
from pyvkfft.fft import fftn as vkfftn
from pyvkfft.fft import ifftn as vkifftn


def fftn(arr, *args, **kwargs):
    arr = arr.astype(np.complex64)
    return vkfftn(arr)

def ifftn(arr, *args, **kwargs):
    arr = arr.astype(np.complex64)
    return vkifftn(arr)


def psf2otf(psf, out_size: tuple):
    """Implementation of matlab's psf2otf

    @psf: point spread function
    @out_size: out size
    """
    if not psf.any():
        print('Input psf should not contain zeros')

    psf_size = psf.shape
    py, px = psf.shape
    ny, nx = out_size
    pads = ((0, ny-py), (0, nx-px))
    new_psf = pad2D_constant(psf, pads, 0)

    offset = tuple(-(d // 2) for d in psf_size)
    new_psf = circshift2D(new_psf, offset)

    otf = fftn(new_psf)

    return otf