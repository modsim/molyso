# -*- coding: utf-8 -*-
"""
The test module's __main__ contains the main() function to run the doctests.
"""


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

    import doctest

    doctest.testmod(molyso.generic.rotation)

    doctest.testmod(molyso.generic.etc)
    doctest.testmod(molyso.generic.fft)

    doctest.testmod(molyso.generic.registration)
    doctest.testmod(molyso.generic.rotation)
    doctest.testmod(molyso.generic.signal)
    doctest.testmod(molyso.generic.smoothing)
    doctest.testmod(molyso.generic.tunable)

    doctest.testmod(molyso.imageio.imagestack)
    doctest.testmod(molyso.imageio.imagestack_ometiff)

    doctest.testmod(molyso.debugging.debugplot)

    doctest.testmod(molyso.mm.cell_detection)
    doctest.testmod(molyso.mm.channel_detection)
    doctest.testmod(molyso.mm.fluorescence)
    doctest.testmod(molyso.mm.highlevel)
    doctest.testmod(molyso.mm.highlevel_interactive_ground_truth)
    doctest.testmod(molyso.mm.highlevel_interactive_viewer)
    doctest.testmod(molyso.mm.image)
    doctest.testmod(molyso.mm.tracking)
    doctest.testmod(molyso.mm.tracking_infrastructure)
    doctest.testmod(molyso.mm.tracking_output)

if __name__ == '__main__':
    main()
