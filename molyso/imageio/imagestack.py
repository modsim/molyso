# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy as np


class MultiImageStack(object):
    """
    MultiImageStack is the base class for image stack access functions.
    Besides being the parent of particular image access classes,
    it serves as a registry of access classes and defines a common interface
    """

    Phase_Contrast = -1
    DIC = -2
    Fluorescence = -3

    ExtensionRegistry = {}

    def __getitem__(self, *args, **kwargs):
        return self.get_image(*args, **kwargs)

    def get_meta(self, *args, **kwargs):
        return self._get_meta(*args, **kwargs)

    def get_image(self, *args, **kwargs):
        image = self._get_image(*args, **kwargs)

        if "float" in kwargs and kwargs["float"]:
            return image.astype(np.float32)

        if "raw" in kwargs and kwargs["raw"]:
            return image
        return image

    def _get_image(self, *args, **kwargs):
        raise NotImplementedError("Virtual function")

    def _get_meta(self, *args, **kwargs):
        raise NotImplementedError("Virtual function")

    @classmethod
    def open(cls, filename):
        """
        Opens an image stack file.
        Will look up its registry if any image stack class is registered for the extension.
        Raises an exception if no class exists for the supplied file type (by extension)
        :param filename: filename to open
        :return:
        """

        for k, v in cls.ExtensionRegistry.items():
            if filename.lower().endswith(k):
                i = v(filename)
                return i

        raise Exception("Unknown format")

