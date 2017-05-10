# -*- coding: utf-8 -*-
"""
The test module at the time only contains a function to get a :py:func:`test_image`.
If called, it will run the doctests.

.. code-block:: bash

    python -m molyso.test

"""
import os

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # code tend to throw warnings because of missing C extensions
    from tifffile import TiffFile

_test_image = None


def test_image():
    """
    Returns a test image (first image of the small dataset).

    :return: image
    :rtype: numpy.ndarray
    """
    global _test_image
    if _test_image is None:
        t = TiffFile(os.path.join(os.path.dirname(__file__), 'example-frame.tif'))
        _test_image = t.pages[0].asarray()
        t.close()

    return _test_image

