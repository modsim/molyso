molyso2vizardous
================

You can convert *molyso*'s tabular output format to PhyloXML/MetaXML lineage trees for viewing and analysis with
*Vizardous* see:

    Helfrich, S. *et al.*, **2015**.
    "Vizardous: Interactive Analysis of Microbial Populations with Single Cell Resolution"
    *Bioinformatics* (Oxford, England). DOI: 10.1093/bioinformatics/btv468

You can download *Vizardous* at https://github.com/modsim/vizardous .

The appropriate tool is embedded in molyso in the molyso.util.molyso2vizardous package.

.. code-block:: bash

    > python3 -m molyso.util.molyso2vizardous

    usage: __main__.py [-h] [-o OUTPUT] [-d MINIMUM_DEPTH] [-q] input

    molyso2vizardous molyso-tabular data format to Vizardous metaXML/phyloXML
    converter

    positional arguments:
      input                 input file

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
      -d MINIMUM_DEPTH, --minimum-depth MINIMUM_DEPTH
      -q, --quiet

    error: the following arguments are required: input


.. code-block:: bash

    > python3 -m molyso.util.molyso2vizardous results.txt


The tool will then generate many individual files for each found track (you can filter out too short tracks by using
the -d MINIMUM_DEPTH option). Note that the internal XML representation is very memory consuming.




