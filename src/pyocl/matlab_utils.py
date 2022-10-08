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
from textwrap import dedent
import pyopencl as cl
import pyopencl.cltypes as cltypes
import pyopencl.array as clArray
from pyvkfft.fft import fftn as vkfftn
from pyvkfft.fft import ifftn as vkifftn


def fftn(arr, *args, **kwargs):
    arr = arr.astype(np.complex64)
    return vkfftn(arr)

def ifftn(arr, *args, **kwargs):
    arr = arr.astype(np.complex64)
    return vkifftn(arr)


ctx = None
queue = None
np_ints = (np.int, np.int64, np.int32, np.int16, np.int8, np.uint, np.uint64, np.uint32, np.uint16, np.uint8)
np_floats = (np.float, np.float64, np.float32, np.float16)


def sync_ctx(f_ctx, f_queue):
    global ctx, queue
    global circshift_prg, ld_padz_prg, mask_index_prg
    ctx = f_ctx; queue = f_queue

    circshift_prg = cl.Program(ctx, 
        dedent(
            r"""
            __kernel void circshift_int(
                __global int * arr_in,
                __global int * arr_out,
                int shift_x,
                int shift_y
            )
            {
                int id_x; id_x = get_global_id(1);
                int id_y; id_y = get_global_id(0);
                int w; w = get_global_size(1);
                int h; h = get_global_size(0);
                int nid_x; nid_x = id_x + shift_x;
                int nid_y; nid_y = id_y + shift_y;
                nid_x = nid_x % w; nid_x = (nid_x < 0) ? nid_x + w : nid_x;
                nid_y = nid_y % h; nid_y = (nid_y < 0) ? nid_y + h : nid_y;
                arr_out[nid_y*w+nid_x] = arr_in[id_y*w+id_x];
            }
            
            __kernel void circshift_float(
                __global float * arr_in,
                __global float * arr_out,
                int shift_x,
                int shift_y
            )
            {
                int id_x; id_x = get_global_id(1);
                int id_y; id_y = get_global_id(0);
                int w; w = get_global_size(1);
                int h; h = get_global_size(0);
                int nid_x; nid_x = id_x + shift_x;
                int nid_y; nid_y = id_y + shift_y;
                nid_x = nid_x % w; nid_x = (nid_x < 0) ? nid_x + w : nid_x;
                nid_y = nid_y % h; nid_y = (nid_y < 0) ? nid_y + h : nid_y;
                arr_out[nid_y*w+nid_x] = arr_in[id_y*w+id_x];
            }
            """
        )
    ).build()

    ld_padz_prg = cl.Program(ctx, 
        dedent(
            r"""
            __kernel void ld_padz(
                __global int * arr_in,
                __global int * arr_out,
                int orig_x,
                int orig_y
            )
            {
                int id_x; id_x = get_global_id(1);
                int id_y; id_y = get_global_id(0);
                int w; w = get_global_size(1);
                if (id_x >= orig_x | id_y >= orig_y)
                {
                    arr_out[id_y*w+id_x] = 0;
                    return;
                }
                arr_out[id_y*w+id_x] = arr_in[id_y*orig_x+id_x];
            }
            """
        )
    ).build()

    mask_index_prg = cl.Program(ctx, 
        dedent(
            r"""
            __kernel void mask_index(
                __global int * arr_in,
                __global bool * arr_mask,
                __global int * arr_out,
                int value
            )
            {
                int id_x; id_x = get_global_id(1);
                int id_y; id_y = get_global_id(0);
                int w; w = get_global_size(1);
                if (arr_mask[id_y*w+id_x])
                {
                    arr_out[id_y*w+id_x] = value;
                    return;
                }
                arr_out[id_y*w+id_x] = arr_in[id_y*w+id_x];
            }
            """
        )
    ).build()


#int or float
def circshift(arr, shift: tuple):
    shifty, shiftx = shift
    if arr.dtype in np_ints:
        arr = arr.astype(np.int32)
        shifted = clArray.empty_like(arr)
        circshift_prg.circshift_int(
            queue, arr.shape, None,
            arr.data,
            shifted.data,
            np.int32(shiftx),
            np.int32(shifty)
        )
    elif arr.dtype in np_floats:
        arr = arr.astype(np.float32)
        shifted = clArray.empty_like(arr)
        circshift_prg.circshift_float(
            queue, arr.shape, None,
            arr.data,
            shifted.data,
            np.int32(shiftx),
            np.int32(shifty)
        )
    else:
        raise NotImplementedError
    return shifted


# int only
def ld_padz(arr, size: tuple):
    sy, sx = arr.shape
    padded = clArray.empty(queue, size, arr.dtype)
    ld_padz_prg.ld_padz(
        queue, size, None,
        arr.data,
        padded.data,
        np.int32(sx),
        np.int32(sy)
    )
    return padded


def psf2otf(psf, out_size: tuple):
    """Implementation of matlab's psf2otf

    @psf: point spread function
    @out_size: out size
    """
    if not psf.any():
        print('Input psf should not contain zeros')

    psf_size = psf.shape
    new_psf = ld_padz(psf, out_size)

    offset = tuple(-(d // 2) for d in psf_size)
    new_psf = circshift(new_psf, offset)

    otf = fftn(new_psf)

    return otf


# int only
def mask_index(arr, mask, value=0):
    masked = clArray.empty_like(arr)
    mask_index_prg.mask_index(
        queue, arr.shape, None,
        arr.data,
        mask.data,
        masked.data,
        np.int32(value)
    )
    return masked