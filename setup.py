# -*- coding: utf-8 -*-
"""
documentation
"""

import molyso
from setuptools import setup, find_packages


setup(
    name='molyso',
    version=molyso.__version__,
    description='MOther machine anaLYsis SOftware',
    long_description='MOther machine anaLYsis SOftware - see https://github.com/modsim/molyso for details.',
    author=molyso.__author__,
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/molyso',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'nd2file', 'tifffile', 'tqdm'],
    package_data={
        'molyso': ['test/example-frame.tif'],
    },
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',   # lately no tests
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',    #
        'Programming Language :: Python :: 3.5',  # main focus
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
