"""
File: L0_Smoothing.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Implementation of the following paper

Paper details
Title:'Image Smoothing via L0 Gradient Minimization'
Link: http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf
"""
from typing import Optional
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2
from .psf2otf import psf2otf


D2_cache = {}


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

def calc_D2(shape):
    global D2_cache
    cached = D2_cache.get(shape, None)
    if cached is None:
        otfx = psf2otf(np.array([[-1,   1]]), shape)
        otfy = psf2otf(np.array([[-1], [1]]), shape)
        
        #Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
        # 开方再平方挺浪费的，所以直接实部和虚部平方相加罢！
        Denormin2 = otfx.real**2 + otfx.imag**2 + otfy.real**2 + otfy.imag**2
        
        D2_cache[shape] = Denormin2
    else:
        Denormin2 = cached
    
    return Denormin2

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
    
    Normin1 = np.fft.fft2(S)
    Denormin2 = calc_D2(img.shape)

    while beta < beta_max:
        Denormin = 1 + beta*Denormin2
        
        h = np.diff(np.pad(S, ((0, 0), (0, 1)), 'wrap'), axis=1)
        v = np.diff(np.pad(S, ((0, 1), (0, 0)), 'wrap'), axis=0)
        grad = h**2 + v**2
        idx = grad < (lambda_ / beta)
        h[idx] = 0; v[idx] = 0
        
        h_diff = -np.diff(np.pad(h, ((0, 0), (1, 0)), 'wrap'), axis=1)
        v_diff = -np.diff(np.pad(v, ((1, 0), (0, 0)), 'wrap'), axis=0)
        Normin2 = h_diff + v_diff
        Normin2 = beta * np.fft.fft2(Normin2)

        FS = (Normin1 + Normin2) / Denormin
        S = np.fft.ifft2(FS).real
        
        # 下面是我见过最铸币的玩意：
        #if False:
        #    ...（若干行代码）
        
        beta = beta * kappa

    ret =  np.clip((S*255), 0, 255).astype(np.uint8)
    return ret