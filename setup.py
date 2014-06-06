# -*- coding: utf-8 -*-
"""
documentation
"""

import molyso
from distutils.core import setup


setup(
    name='molyso',
      version=molyso.__version__,
      description='MOther machine anaLYsis SOftware',
      long_description='MOther machine anaLYsis SOftware - PyPI site text',
      author='Christian C. Sachs',
      author_email='sachs.christian@gmail.com',
      url='http://example.com',
      packages=[
          'molyso',
          'molyso.debugging',
          'molyso.generic',
          'molyso.imageio',
          'molyso.mm',
      ],
      scripts=['molyso.py'],
      requires=['numpy', 'scipy'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: Other/Proprietary License',
          #'License :: OSI Approved :: MIT License',
          #'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Programming Language :: Python :: 2.7',  # has been developed/tested on
          'Programming Language :: Python :: 3',  # should work with all of
          'Programming Language :: Python :: 3.3',  # has been developed/tested on
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Image Recognition',
      ]
)