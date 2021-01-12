.. image:: docs/_static/molyso-banner.png

molyso Readme
=============

.. image:: https://img.shields.io/pypi/v/molyso.svg
   :target: https://pypi.python.org/pypi/molyso

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://molyso.readthedocs.io/en/latest/

.. image:: https://travis-ci.org/modsim/molyso.svg?branch=master
   :target: https://travis-ci.org/modsim/molyso

.. image:: https://img.shields.io/pypi/l/molyso.svg
   :target: https://opensource.org/licenses/BSD-2-Clause

.. image:: https://img.shields.io/docker/build/modsim/molyso.svg
   :target: https://hub.docker.com/r/modsim/molyso

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.53764.svg
   :target: https://dx.doi.org/10.5281/zenodo.53764

.. image:: https://zenodo.org/badge/doi/10.1371/journal.pone.0163453.svg
   :target: https://dx.doi.org/10.1371/journal.pone.0163453
   

Publication
-----------
When using *molyso* for scientific applications, cite our publication:

    Sachs CC, Grünberger A, Helfrich S, Probst C, Wiechert W, Kohlheyer D, Nöh K (2016)
    Image-Based Single Cell Profiling:
    High-Throughput Processing of Mother Machine Experiments.
    PLoS ONE 11(9): e0163453. doi: 10.1371/journal.pone.0163453

It is available on the PLoS ONE homepage at `DOI: 10.1371/journal.pone.0163453 <https://dx.doi.org/10.1371/journal.pone.0163453>`_

Example Datasets
----------------
You can find example datasets (as used in the publication) deposited at zenodo `DOI: 10.5281/zenodo.53764 <https://dx.doi.org/10.5281/zenodo.53764>`_.

Documentation
-------------
Documentation can be built using sphinx, but is available online as well at `Read the Docs <https://molyso.readthedocs.io/en/latest/>`_.

License
-------
*molyso* is free/libre open source software under the 2-clause BSD License. See :doc:`license`

Prerequisites
-------------
*molyso* needs Python 3, if you don't have a Python installation or are not familiar with installing packages from source, it is suggested
that you use the `Anaconda <https://www.continuum.io/downloads>`_ Python distribution, available for Windows, Linux and macOS.

Ways to install molyso
----------------------

There are different ways to install molyso,
for ease of use it is suggested to use the Anaconda Python distribution and the conda package manager.
Alternatively, you can use molyso inside a Docker container, see the Docker_ section near the end.

With Anaconda
#############

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda config --add channels bioconda
    > conda config --add channels csachs

    > conda install -y molyso


Alternatively, manually from github
###################################

.. code-block:: bash

    > git clone https://github.com/modsim/molyso
    > cd molyso
    > python setup.py install --user

First Steps
-----------
*molyso* is packaged as a Python module, to run it, just use:

.. code-block:: bash

    > python -m molyso

And you will be greeted by the help screen of molyso:

.. code-block:: none


         \   /\  /\  /                             -------------------------
          | | |O| | |    molyso                    Developed  2013 - 2021 by
          | | | | |O|                              Christian   C.  Sachs  at
          |O| |O| |O|    MOther    machine         ModSim / Microscale Group
          \_/ \_/ \_/    anaLYsis SOftware         Research  Center  Juelich
        --------------------------------------------------------------------
        If you use this software in a publication, cite our paper:


        Sachs CC, Grünberger A, Helfrich S, Probst C, Wiechert W, Kohlheyer D, Nöh K (2016)
        Image-Based Single Cell Profiling:
        High-Throughput Processing of Mother Machine Experiments.
        PLoS ONE 11(9): e0163453. doi: 10.1371/journal.pone.0163453

        --------------------------------------------------------------------

    usage: __main__.py [-h] [-m MODULES] [-p] [-gt GROUND_TRUTH] [-ct CACHE_TOKEN]
                       [-tp TIMEPOINTS] [-mp MULTIPOINTS] [-o TABLE_OUTPUT]
                       [-ot TRACKING_OUTPUT] [-nb] [-cpu MP] [-debug] [-do] [-nci]
                       [-cfi] [-ccb CHANNEL_BITS] [-cfb CHANNEL_FLUORESCENCE_BITS]
                       [-q] [-nc [IGNORECACHE]] [-nt] [-t TUNABLES]
                       [-s TUNABLE_LIST TUNABLE_LIST] [-pt] [-rt READ_TUNABLES]
                       [-wt WRITE_TUNABLES]
                       input

    molyso: MOther machine anaLYsis SOftware

    positional arguments:
      input                 input file

    optional arguments:
      -h, --help            show this help message and exit
      -m MODULES, --module MODULES
      -p, --process
      -gt GROUND_TRUTH, --ground-truth GROUND_TRUTH
      -ct CACHE_TOKEN, --cache-token CACHE_TOKEN
      -tp TIMEPOINTS, --timepoints TIMEPOINTS
      -mp MULTIPOINTS, --multipoints MULTIPOINTS
      -o TABLE_OUTPUT, --table-output TABLE_OUTPUT
      -ot TRACKING_OUTPUT, --output-tracking TRACKING_OUTPUT
      -nb, --no-banner
      -cpu MP, --cpus MP
      -debug, --debug
      -do, --detect-once
      -nci, --no-channel-images
      -cfi, --channel-fluorescence-images
      -ccb CHANNEL_BITS, --channel-image-channel-bits CHANNEL_BITS
      -cfb CHANNEL_FLUORESCENCE_BITS, --channel-image-fluorescence-bits CHANNEL_FLUORESCENCE_BITS
      -q, --quiet
      -nc [IGNORECACHE], --no-cache [IGNORECACHE]
      -nt, --no-tracking
      -t TUNABLES, --tunables TUNABLES
      -s TUNABLE_LIST TUNABLE_LIST, --set-tunable TUNABLE_LIST TUNABLE_LIST
      -pt, --print-tunables
      -rt READ_TUNABLES, --read-tunables READ_TUNABLES
      -wt WRITE_TUNABLES, --write-tunables WRITE_TUNABLES

    error: the following arguments are required: input


There are three modes of operation, batch processing, interactive viewer, and ground truth generation.
The most important part for routine use is batch processing, which will process a whole file or selected time/multi points and produce tabular output and/or tracking visualizations.
The interactive viewer can be used to show channel and cell detection on the given dataset, as a first step to check if the settings are applicable.
The ground truth viewer is more of a tool for verification of results, the kymograph of a preanalyzed dataset can be visualized *without* tracking, and individual cell generations can be marked manually, yielding a growth rate which can be compared to the automatic analysis.

To start the interactive viewer, just call molyso without any other parameters:

.. code-block:: bash

    > python -m molyso dataset.ome.tiff

To start batch processing, run molyso with the :code:`-p` option. Give an output file for tabular output with :code:`-o` and/or an output directory for individual tracked kymographs with :code:`-ot`.

Note: While OME-TIFF file contain calibration of time and voxel size, simple :code:`.tif` files may not,
you can tell molyso manually about the calibration by adding comma-delimited parameters after the file name (followed by a question mark):
Example:

.. code-block:: bash

    > python -m molyso "filename.tif?interval=300,calibration=0.08"


Supported are among others: the acquisition :code:`interval` (seconds), and the pixel size :code:`calibration` in um per pixel.
Some older files may have incorrectly labeled axes, since molyso expects the time axis to be correctly labeled, it might be necessary to reorder the axes, this can be done on the fly, by passing e.g. :code:`?swap_axes=Z..T`.
Don't forget to escape/quote the ? in the command line.


.. code-block:: bash

    > python -m molyso dataset.ome.tiff -p -o results.txt -ot dataset_tracking

*molyso* writes cache files in the current directory which contain temporary analysis results. If you want to re-generate tabular output *e.g.*, those files will be read in and already performed analysis steps will be skipped. They are used as well, to show the kymograph for ground truth data mode. They can be kept if you plan any of the mentioned steps, if you are finished with an analysis, they can be deleted as well.

Once *molyso* has run, you will need to post-process the data to extract the information you're interested in.
Take a look at the Jupyter/IPython Notebooks.

Docker
------

`Docker <https://www.docker.com/>`_ is a containerization platform allowing for applications to be run with bundled dependencies without explicit
installation steps.

You can use the following commands to run molyso in lieu of the aforementioned calls, e.g. for analysis:

.. code-block:: bash

   > docker run --tty --interactive --rm --volume "`pwd`:/data" --user `id -u` modsim/molyso -p <parameters ...>

And to run interactive mode (display on local X11, under Linux):

.. code-block:: bash

   > docker run --tty --interactive --rm --volume "`pwd`:/data" --user `id -u` --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix modsim/molyso <parameters ...>

Docker usage has just been tested with Linux host systems.

Third Party Licenses
--------------------
Note that this software contains the following portions from other authors, under the following licenses (all BSD-flavoured):

molyso/generic/otsu.py:
    functions threshold_otsu and histogram by the scikit-image team, licensed BSD (see file head).
        Copyright (C) 2011, the scikit-image team
