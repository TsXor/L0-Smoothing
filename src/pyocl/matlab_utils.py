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
from L0_Smoothing.pyocl.GCArray import GCArray as clArray

from .ocl_func import circshift2D, pad2D_constant, crop2D

from pyvkfft.fft import fftn as vkfftn
from pyvkfft.fft import ifftn as vkifftn

import reikna.cluda as cluda
from reikna.fft import FFT as rkfft

from gpyfft.fft import FFT as gpyfft


rk_api = cluda.ocl_api()
rkfft_func_cache = {}


def legal_axes(axes, ndim):
    if not (axes is None):
        axes = tuple((i if i>=0 else i+ndim) for i in axes)
    return axes

def fftn(arr, axes=None, mode='pyvkfft'):
    arr = arr.astype(np.complex64)
    if mode=='pyvkfft':
        out = vkfftn(arr, axes=axes)
    elif mode=='reikna':
        out = clArray.empty_like(arr)
        axes = legal_axes(axes, arr.ndim)
        try:
            rkfft_func = rkfft_func_cache[arr.queue]
        except KeyError:
            thr = rk_api.Thread(arr.queue)
            rkfft_func = rkfft(arr, axes=axes).compile(thr)
            rkfft_func_cache[arr.queue] = rkfft_func
        rkfft_func(out, arr)
    elif mode=='gpyfft':
        out = clArray.empty_like(arr)
        axes = legal_axes(axes, arr.ndim)
        axes = axes if axes is None else sorted(axes, reverse=True)
        transform = gpyfft(arr.context, arr.queue, arr, out, axes=axes)
        event, = transform.enqueue()
        event.wait()
    return out

def ifft_wrapper(arr, *args, **kwargs):
    arr = arr.conj()
    out = fftn(arr, *args, **kwargs)
    out = out.conj()
    npoints = 1
    for d in arr.shape:
        npoints = npoints*d
    out = out/npoints
    return out

def ifftn(arr, axes=None, mode='pyvkfft'):
    arr = arr.astype(np.complex64)
    if mode=='pyvkfft':
        out = vkifftn(arr, axes=axes)
    elif mode=='reikna':
        out = ifft_wrapper(arr, axes=axes, mode=mode)
    elif mode=='gpyfft':
        out = ifft_wrapper(arr, axes=axes, mode=mode)
    return out


def psf2otf(psf, out_size: tuple, mode='pyvkfft'):
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

    otf = fftn(new_psf, mode=mode)

    return otf