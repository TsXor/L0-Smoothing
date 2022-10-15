"""
File: L0_Smoothing.py
Author: TsXor
Email: zhang050525@qq.com
Github: https://github.com/TsXor
Description: Implementation of the following paper
             but with PyOpenCL and pyvkfft
             *FASTER!FASTER!NOT ONLY CUDA!*

Paper details
Title:'Image Smoothing via L0 Gradient Minimization'
Link: http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf
"""
from typing import Optional
import cv2
import numpy as np
import pyopencl as cl
from L0_Smoothing.pyocl.GCArray import GCArray as clArray

from .matlab_utils import psf2otf, circshift2D, fftn, ifftn
from .ocl_func import fancyindex2D, pad2D_constant, crop2D
from .get_size import get_nearest_bigger


ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

def setctx(arg: int):
    global ctx, queue
    platforms = cl.get_platforms()
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[arg])]
    )
    queue = cl.CommandQueue(ctx)

def listpl():
    return cl.get_platforms()

def split(img):
    return [img[...,n] for n in range(img.shape[-1])]

def merge(imgseq):
    return np.transpose(np.array(imgseq, dtype=np.uint8), (1, 2, 0))


def multi_channel(decorated):
    def multi_op(img, *args, asHSV=False,**kwargs):
        isgray = (img.ndim==2)
        asHSV = (not isgray) and asHSV
        img = cv2.cvtColor(img, cv2.RGB2HSV) if asHSV else img
        ret = decorated(img, *args, **kwargs) \
              if isgray else \
              merge([decorated(islice, *args, **kwargs) for islice in split(img)])
        ret = cv2.cvtColor(ret, cv2.HSV2RGB) if asHSV else ret
        return ret
    return multi_op

def gpyfft_pad2D(arr):
    sy, sx = arr.shape
    nsx = get_nearest_bigger(sx)
    nsy = get_nearest_bigger(sy)
    pads = ((0, nsy-sy), (0, nsx-sx))
    return pad2D_constant(arr, pads, 0.5)


@multi_channel
def L0_Smoothing(
    img: np.ndarray,
    lambda_: Optional[float] = 2e-2,
    kappa: Optional[float] = 2.0,
    beta_max: Optional[float] = 1e5,
    mode='pyvkfft',
):
    S = img / 255
    S = clArray.to_device(queue, S)
    S = L0_Smoothing_CL(S, lambda_, kappa, beta_max, mode)
    ret = S.get()
    ret = np.clip((ret*255), 0, 255).astype(np.uint8)
    return ret


def L0_Smoothing_CL(
    S: clArray.Array,
    lambda_: Optional[float] = 2e-2,
    kappa: Optional[float] = 2.0,
    beta_max: Optional[float] = 1e5,
    mode='pyvkfft'
):
    beta = 2 * lambda_
    sizey, sizex = S.shape

    ################
    # Padding for gpyfft
    ################
    if mode=='gpyfft':
        print('Padding array... ', end='')
        S = gpyfft_pad2D(S)
        print('done.')
    ################

    Normin1 = fftn(S, axes = (-2,-1), mode=mode)
    otfx = psf2otf(clArray.to_device(S.queue, np.array([[-1,   1]])), S.shape, mode=mode)
    otfy = psf2otf(clArray.to_device(S.queue, np.array([[-1], [1]])), S.shape, mode=mode)

    Denormin2 = otfx.real**2 + otfx.imag**2 + otfy.real**2 + otfy.imag**2
    
    while beta < beta_max:
        Denormin = 1 + beta*Denormin2
        
        h = circshift2D(S, (0, -1))-S
        v = circshift2D(S, (-1, 0))-S
        grad = h**2 + v**2
        idx = grad < (lambda_ / beta)
        h = fancyindex2D(h, idx, 0); v = fancyindex2D(v, idx, 0)
        
        h_diff = circshift2D(h, (0, 1))-h
        v_diff = circshift2D(v, (1, 0))-v
        Normin2 = h_diff + v_diff
        Normin2 = beta * fftn(Normin2, axes = (-2,-1), mode=mode)

        FS = (Normin1 + Normin2) / Denormin
        S = ifftn(FS, axes = (-2,-1), mode=mode).real

        beta = beta * kappa

    ################
    # Cropping for gpyfft
    ################
    if mode=='gpyfft':
        S = crop2D(S, slice(sizey), slice(sizex))
    ################

    return S