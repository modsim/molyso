from __future__ import division, unicode_literals, print_function
import math

import numpy

from .util import *
from .signal import find_phase


def translation_2x1d(imA=None, imB=None, fftsA=(), fftsB=(), returnA=False, returnB=False):
    if fftsA != ():
        fftAv, fftAh = fftsA
        signalAv, signalAh = None, None
    else:
        fftAv, fftAh = None, None
        signalAv, signalAh = vertical_mean(imA), horizontal_mean(imB)

    if fftsB != ():
        fftBv, fftBh = fftsB
        signalBv, signalBh = None, None
    else:
        fftBv, fftBh = None, None
        signalBv, signalBh = vertical_mean(imB), horizontal_mean(imB)

    v, fftAv, fftBv = find_phase(signal1=signalAv, signal2=signalBv,
                                 fft1=fftAv, fft2=fftBv, return1=True, return2=True)

    h, fftAh, fftBh = find_phase(signal1=signalAh, signal2=signalBh,
                                 fft1=fftAh, fft2=fftBh, return1=True, return2=True)

    result = ([float(-v), float(-h)],)

    if returnA:
        result += ((fftAv, fftAh),)

    if returnB:
        result += ((fftBv, fftBh),)

    return result


def shift_image(img, shift):
    a, b = shift
    aa, bb = img.shape
    if a < 0:
        fromam = -a
        fromat = aa
        toam = 0
        toat = a
    else:
        fromam = 0
        fromat = aa - a
        toam = a
        toat = aa
    if b < 0:
        frombm = -b
        frombt = bb
        tobm = 0
        tobt = b
    else:
        frombm = 0
        frombt = bb - b
        tobm = b
        tobt = bb
    newimg = img.copy()
    newimg[toam:toat, tobm:tobt] = img[fromam:fromat, frombm:frombt]
    return newimg
