# -*- coding: utf-8 -*-
"""
registration.py contains a simple 2D image registration function,
which registers images by checking the individual horizontal and vertical shifts
"""
from __future__ import division, unicode_literals, print_function

from .signal import find_phase, vertical_mean, horizontal_mean


def translation_2x1d(image_a=None, image_b=None, ffts_a=(), ffts_b=(), return_a=False, return_b=False):
    if ffts_a != ():
        fft_av, fft_ah = ffts_a
        signal_av, signal_ah = None, None
    else:
        fft_av, fft_ah = None, None
        signal_av, signal_ah = vertical_mean(image_a), horizontal_mean(image_a)

    if ffts_b != ():
        fft_bv, fft_bh = ffts_b
        signal_bv, signal_bh = None, None
    else:
        fft_bv, fft_bh = None, None
        signal_bv, signal_bh = vertical_mean(image_b), horizontal_mean(image_b)

    v, fft_av, fft_bv = find_phase(signal_1=signal_av, signal_2=signal_bv,
                                   fft_1=fft_av, fft_2=fft_bv, return_1=True, return_2=True)

    h, fft_ah, fft_bh = find_phase(signal_1=signal_ah, signal_2=signal_bh,
                                   fft_1=fft_ah, fft_2=fft_bh, return_1=True, return_2=True)

    result = ([float(-v), float(-h)],)

    if return_a:
        result += ((fft_av, fft_ah),)

    if return_b:
        result += ((fft_bv, fft_bh),)

    return result
