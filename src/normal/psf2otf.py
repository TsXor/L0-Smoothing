"""
File: psf2otf.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Implementation of matlab's psf2otf

Notes: In order to understand psf2otf:

FFT does cyclic convolution. To understand what cyclic convolution is
please refer to the document below (also in the docs)
https://www.docdroid.net/YSKkZ5Y/fft-based-2d-cyclic-convolution-pdf#page=5
"""
import numpy as np


def circshift(psf: np.ndarray, shift: tuple) -> np.ndarray:
    """Circular shifts

    @psf: input psf
    @shift: shifts correspoinding to each dimension
    @returns: TODO

    """
    for ax, offset in enumerate(shift):
        psf = np.roll(psf, offset, axis=ax)

    return psf

def psf2otf(psf: np.ndarray, out_size: tuple) -> np.ndarray:
    """Implementation of matlab's psf2otf

    @psf: point spread function
    @out_size: out size
    """
    if not np.any(psf):
        print('Input psf should not contain zeros')

    psf_size = psf.shape
    new_psf = np.zeros(out_size, dtype=np.float32)
    new_psf[:psf_size[0], :psf_size[1]] = psf[:, :]

    offset = tuple(-(d // 2) for d in psf.shape)
    new_psf = circshift(new_psf, offset)

    otf = np.fft.fftn(new_psf)

    return np.complex64(otf)
