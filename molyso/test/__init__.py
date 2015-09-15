import os
from ..imageio.tifffile import TiffFile

_test_image = None


def test_image():
    global _test_image
    if _test_image is None:
        t = TiffFile(os.path.join(os.path.dirname(__file__), 'example-frame.tif'))
        _test_image = t.pages[0].asarray()
        t.close()

    return _test_image

