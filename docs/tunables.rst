Tunables
=========

Introduction
------------

Tunables are configuration parameters for the analysis.
You can either let *molyso* read or write all tunables from/to a file, or you can set individual tunables per command
line. Note that tunables are defined where they are used, and to *collect* all tunables, a typical run has to be
performed (use -cpu 0 to disable parallelism!).

Tunables are are read/written in JSON. JSON is used as well to set tunables on the command line, e.g.:

.. code-block:: bash

    > python -m molyso -t '{"cells.empty_channel.skipping":true}'

See below for a table of tunables. Note that for most data sets it is not necessary to modify tunables, and their
particular action is best understood by looking up their usage in the source code ...

Various tunables will as well affect processing speed.

Table of Tunables
-----------------

+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| **Name**                                            | **Default** | **Type** | **Description**                                                                |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.empty_channel.skipping                        | False       | bool     | For empty channel detection, whether it is enabled.                            |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.empty_channel.skipping.outlier_times_sigma    | 2.0         | float    | For empty channel detection, maximum sigma used for thresholding the profile.  |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.extrema.order                                 | 15          | int      | For cell detection, window width of the local extrema detector.                |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.filtering.maximum_brightness                  | 0.5         | float    | For cell detection, maximum brightness a cell may have.                        |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.filtering.minimum_prominence                  | 10.0        | float    | For cell detection, minimum prominence a cell must have.                       |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.minimal_length.in_mu                          | 1.0         | float    | The minimal allowed cell size (Smaller cells will be filtered out).            |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.otsu_bias                                     | 1.0         | float    | Bias factor for the cell detection Otsu image.                                 |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| cells.smoothing.length                              | 10          | int      | Length of smoothing Hamming window for cell detection.                         |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.fft_oversampling                | 8           | int      | For channel detection, FFT oversampling factor.                                |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.noise_suppression_factor.lower  | 0.1         | float    | For channel detection, lower profile, noise reduction, reduction factor.       |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.noise_suppression_factor.upper  | 0.1         | float    | For channel detection, upper profile, noise reduction, reduction factor.       |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.noise_suppression_range.lower   | 0.5         | float    | For channel detection, lower profile, noise reduction, reduction range.        |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.noise_suppression_range.upper   | 0.5         | float    | For channel detection, upper profile, noise reduction, reduction range.        |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.profile_smoothing_width.lower   | 5           | int      | For channel detection, lower profile, smoothing window width.                  |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.profile_smoothing_width.upper   | 5           | int      | For channel detection, upper profile, smoothing window width.                  |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.horizontal.threshold_factor                | 0.2         | float    | For channel detection, threshold factor for l/r border determination.          |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.vertical.alternate.delta                   | 5           | int      | For channel detection (alternate, vertical), acceptable delta.                 |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.vertical.alternate.fft_smoothing_width     | 3           | int      | For channel detection (alternate, vertical), spectrum smoothing width.         |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.vertical.alternate.split_factor            | 60          | int      | For channel detection (alternate, vertical), split factor.                     |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| channels.vertical.method                            | alternate   | str      | For channel detection, vertical method to use (either alternate or recursive). |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.cell                                         | #005b82     | str      | For debug output, cell color.                                                  |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.channel                                      | #e7af12     | str      | For debug output, channel color.                                               |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.visualization.track.alpha                    | 0.3         | float    | Track alpha for visualization.                                                 |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.visualization.track.color                    | #005B82     | str      | Track color for visualization.                                                 |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.visualization.track.random                   | 1           | int      | Randomize tracking color palette?                                              |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| colors.visualization.track.random.seed              | 3141592653  | int      | Random seed for tracking visualization.                                        |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| orientation-detection.strips                        | 10          | int      | Number of strips for orientation correction.                                   |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+
| tracking.empty_channel_filtering.minimum_mean_cells | 2.0         | float    | For empty channel removal, minimum of cell mean per channel.                   |
+-----------------------------------------------------+-------------+----------+--------------------------------------------------------------------------------+

