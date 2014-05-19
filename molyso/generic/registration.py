from __future__ import division, unicode_literals, print_function

from .util import *
from .signal import find_phase


def translation_2x1d(im_a=None, im_b=None, ffts_a=(), ffts_b=(), return_a=False, return_b=False):
    if ffts_a != ():
        fft_av, fft_ah = ffts_a
        signal_av, signal_ah = None, None
    else:
        fft_av, fft_ah = None, None
        signal_av, signal_ah = vertical_mean(im_a), horizontal_mean(im_b)

    if ffts_b != ():
        fft_bv, fft_bh = ffts_b
        signal_bv, signal_bh = None, None
    else:
        fft_bv, fft_bh = None, None
        signal_bv, signal_bh = vertical_mean(im_b), horizontal_mean(im_b)

    v, fft_av, fft_bv = find_phase(signal1=signal_av, signal2=signal_bv,
                                   fft1=fft_av, fft2=fft_bv, return1=True, return2=True)

    h, fft_ah, fft_bh = find_phase(signal1=signal_ah, signal2=signal_bh,
                                   fft1=fft_ah, fft2=fft_bh, return1=True, return2=True)

    result = ([float(-v), float(-h)],)

    if return_a:
        result += ((fft_av, fft_ah),)

    if return_b:
        result += ((fft_bv, fft_bh),)

    return result
