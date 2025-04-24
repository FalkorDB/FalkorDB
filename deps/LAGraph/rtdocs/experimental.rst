Experimental Algorithms
=======================

LAGraph includes a set of experimental algorithms and utilities, in the
LAGraph/experimental folder.  The include file appears in
LAGraph/include/LAGraphX.h.  These methods are in various states
of development, and their C APIs are not guaranteed to be stable.
They are not guaranteed to have all of their performance issues resolved.
However, they have been tested, debugged, and mostly benchmarked.

New algorithms and utilities can be contributed by placing them in
the experimental/algorithm or experimental/utility folder.
Tests for new algorithms or utilities should be placed in the
experimental/test folder, and benchmark programs that exercise
their performance (and typically check results) should be placed in
the experimental/benchmark folder.

An simple example algorithm and its test and benchmark is provided,
which serves as a template for creating new algorithms:

    * `algorithm/LAGraph_HelloWorld.c`_
        a simple "algorithm" that merely creates a copy of the G->A
        adjacency matrix.

    * `benchmark/helloworld2_demo.c`_
        a benchmark program illustrates how to
        write a main program that loads in a graph, calls an algorithm, and
        checks and prints the result.  If any file appears in the
        benchmark folder with a name ending in _demo.c, then the CMake
        script will find it and compile it.

    * `benchmark/helloworld_demo.c`_
        another benchmark program.
        This one relies on internal utilities.  See the file for
        details.

    * `test/test_HelloWorld.c`_
        a test program, using the acutest
        test suite.  If any file appears in experimenta/test with
        the prefix test_*, the CMake script will compile it and
        include it in the "make test" target.

.. _algorithm/LAGraph_HelloWorld.c: https://github.com/GraphBLAS/LAGraph/blob/stable/experimental/algorithm/LAGraph_HelloWorld.c
.. _benchmark/helloworld2_demo.c: https://github.com/GraphBLAS/LAGraph/blob/stable/experimental/benchmark/helloworld2_demo.c
.. _benchmark/helloworld_demo.c: https://github.com/GraphBLAS/LAGraph/blob/stable/experimental/benchmark/helloworld_demo.c
.. _test/test_HelloWorld.c: https://github.com/GraphBLAS/LAGraph/blob/stable/experimental/test/test_HelloWorld.c
