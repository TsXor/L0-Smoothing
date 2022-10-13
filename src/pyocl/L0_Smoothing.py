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
import pyopencl.array as clArray
from .matlab_utils import psf2otf, circshift2D, fftn, ifftn
from .ocl_func import fancyindex2D


ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

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


@multi_channel
def L0_Smoothing(
    img: np.ndarray,
    lambda_: Optional[float] = 2e-2,
    kappa: Optional[float] = 2.0,
    beta_max: Optional[float] = 1e5
):
    beta = 2 * lambda_
    sizey, sizex = img.shape
    
    S = img / 255
    S = clArray.to_device(queue, S)
    
    Normin1 = fftn(S, axes = (-2,-1))
    otfx = psf2otf(clArray.to_device(queue, np.array([[-1,   1]])), img.shape)
    otfy = psf2otf(clArray.to_device(queue, np.array([[-1], [1]])), img.shape)
    
    #Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
    # 开方再平方挺浪费的，所以直接实部和虚部平方相加罢！
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
        Normin2 = beta * fftn(Normin2, axes = (-2,-1))

        FS = (Normin1 + Normin2) / Denormin
        S = ifftn(FS, axes = (-2,-1)).real
        
        # 下面是我见过最铸币的玩意：
        #if False:
        #    ...（若干行代码）
        
        beta = beta * kappa

    ret = S.get()
    ret = np.clip((ret*255), 0, 255).astype(np.uint8)
    return ret