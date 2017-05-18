# -*- coding: utf-8 -*-
"""
The test module's __main__ contains the main() function to run the doctests.
"""

import sys
import doctest


def main():
    """
    Runs all the doctests.

    """
    import molyso.generic.etc
    import molyso.generic.fft
    import molyso.generic.registration
    import molyso.generic.rotation
    import molyso.generic.signal
    import molyso.generic.smoothing
    import molyso.generic.tunable

    import molyso.imageio.imagestack
    import molyso.imageio.imagestack_ometiff
    import molyso.imageio.imagestack_czi
    import molyso.imageio.imagestack_nd2

    import molyso.debugging.debugplot

    import molyso.mm.cell_detection
    import molyso.mm.channel_detection
    import molyso.mm.fluorescence
    import molyso.mm.highlevel
    import molyso.mm.highlevel_interactive_ground_truth
    import molyso.mm.highlevel_interactive_viewer
    import molyso.mm.image
    import molyso.mm.tracking
    import molyso.mm.tracking_infrastructure
    import molyso.mm.tracking_output

    modules_to_test = [
        molyso.generic.rotation,
        #
        molyso.generic.etc,
        molyso.generic.fft,
        #
        molyso.generic.registration,
        molyso.generic.rotation,
        molyso.generic.signal,
        molyso.generic.smoothing,
        molyso.generic.tunable,
        #
        molyso.imageio.imagestack,
        molyso.imageio.imagestack_ometiff,
        molyso.imageio.imagestack_czi,
        molyso.imageio.imagestack_nd2,
        #
        molyso.debugging.debugplot,
        #
        molyso.mm.cell_detection,
        molyso.mm.channel_detection,
        molyso.mm.fluorescence,
        molyso.mm.highlevel,
        molyso.mm.highlevel_interactive_ground_truth,
        molyso.mm.highlevel_interactive_viewer,
        molyso.mm.image,
        molyso.mm.tracking,
        molyso.mm.tracking_infrastructure,
        molyso.mm.tracking_output
    ]

    total_failures, total_tests = 0, 0

    for a_module in modules_to_test:
        failures, tests = doctest.testmod(a_module)
        total_failures += failures
        total_tests += tests

    print("Run %d tests in total." % (total_tests,))

    if total_failures > 0:
        print("Test failures occurred, exiting with non-zero status.")
        sys.exit(1)


if __name__ == '__main__':
    main()
