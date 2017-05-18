# -*- coding: utf-8 -*-
"""
registration.py contains a simple 2D image registration function,
which registers images by checking the individual horizontal and vertical shifts
"""
from __future__ import division, unicode_literals, print_function

from .signal import find_phase, vertical_mean, horizontal_mean

import numpy as np


def translation_2x1d(image_a=None, image_b=None, ffts_a=(), ffts_b=(), return_a=False, return_b=False):
    """

    :param image_a:
    :param image_b:
    :param ffts_a:
    :param ffts_b:
    :param return_a:
    :param return_b:
    :return:
    """
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


def shift_image(image, shift, background='input'):
    """

    :param image:
    :param shift:
    :param background:
    :return: :raise ValueError:
    """

    vertical, horizontal = shift
    vertical, horizontal = round(vertical), round(horizontal)
    height, width = image.shape

    if vertical < 0:
        source_vertical_lower = -vertical
        source_vertical_upper = height
        destination_vertical_lower = 0
        destination_vertical_upper = vertical
    else:
        source_vertical_lower = 0
        source_vertical_upper = height - vertical
        destination_vertical_lower = vertical
        destination_vertical_upper = height

    if horizontal < 0:
        source_horizontal_lower = -horizontal
        source_horizontal_upper = width
        destination_horizontal_lower = 0
        destination_horizontal_upper = horizontal
    else:
        source_horizontal_lower = 0
        source_horizontal_upper = width - horizontal
        destination_horizontal_lower = horizontal
        destination_horizontal_upper = width

    if background == 'input':
        new_image = image.copy()
    elif background == 'blank':
        new_image = np.zeros_like(image)
    else:
        raise ValueError("Unsupported background method passed. Use background or blank.")

    new_image[
        destination_vertical_lower:destination_vertical_upper,
        destination_horizontal_lower:destination_horizontal_upper
    ] = image[
        source_vertical_lower:source_vertical_upper,
        source_horizontal_lower:source_horizontal_upper
    ]
    return new_image
