# -*- coding: utf-8 -*-
"""
rotation.py
"""

from __future__ import division, unicode_literals, print_function
import math
import numpy

from .util import each_image_slice, vertical_mean, remove_outliers
from .smoothing import hamming_smooth
from .signal import find_phase


def find_rotation(im, steps=10, smoothing_signal_length=15):
    """
    tries to detect the rotation
    :param im:
    :param steps:
    :return:
    """

    shifts = numpy.zeros(steps)

    last_signal = None
    last_fft = None

    step = 0

    for n, the_step, imgslice in each_image_slice(im, steps, direction='vertical'):
        step = the_step
        profile = vertical_mean(imgslice)

        profile = hamming_smooth(profile, smoothing_signal_length)
        profile = numpy.diff(profile)

        if n == 0:
            last_signal = profile
            last_fft = None
            continue

        shift, current_fft = find_phase(last_signal, profile, last_fft, return2=True)
        last_signal, last_fft = profile, current_fft

        shifts[n] = shift

    shifts = remove_outliers(shifts)

    return math.atan(numpy.mean(shifts) / step) * 180.0 / math.pi


try:
    # noinspection PyUnresolvedReferences
    import cv2

    def rotate_image(image, angle):
        """

        :param image:
        :param angle:
        :return:
        """
        return cv2.warpAffine(image,
                              cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), angle, 1.0),
                              (image.shape[1], image.shape[0]))
except ImportError:
    try:
        # noinspection PyUnresolvedReferences
        from scipy.misc import imrotate

        def rotate_image(image, angle):
            """

            :param image:
            :param angle:
            :return:
            """

            return imrotate(image, angle)
    except ImportError:
        from scipy.ndimage.interpolation import rotate

        def rotate_image(image, angle):
            """

            :param image:
            :param angle:
            :return:
            """
            return rotate(image, angle=angle, reshape=False)


def calculate_crop_for_angle(image, angle):
    """

    :param image:
    :param angle:
    :return:
    """
    wd = (image.shape[0] * 0.5) * math.tan(angle / (180.0 / math.pi))
    hd = (image.shape[1] * 0.5) * math.tan(angle / (180.0 / math.pi))
    hd, wd = int(abs(hd)), int(abs(wd))
    return hd, wd


def apply_rotate_and_cleanup(img, angle):
    """

    :param img:
    :param angle:
    :return:
    """
    newimg = rotate_image(img, angle)
    h, w = calculate_crop_for_angle(img, angle)
    lh, rh = (h, -h) if h else (None, None)
    lw, rw = (w, -w) if w else (None, None)
    newimg = newimg[lh:rh, lw:rw]
    return newimg, angle, h, w


def rotate_and_cleanup(img):
    """

    :param img:
    :return:
    """
    angle = find_rotation(img)
    return apply_rotate_and_cleanup(img, angle)

