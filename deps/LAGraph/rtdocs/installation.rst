Installation
============

LAGraph is available at `<https://github.com/GraphBLAS/LAGraph>`_.  Be sure to
check out the default ``stable`` branch, or use one of the stable releases.
LAGraph requires SuiteSparse:GraphBLAS, available at
`<https://github.com/DrTimothyAldenDavis/GraphBLAS>`_.

To compile and install LAGraph, you must first compile and install a recent
version of SuiteSparse:GraphBLAS.  Place LAGraph and GraphBLAS in the same
folder, side-by-side.  Compile and (optionally) install SuiteSparse:GraphBLAS
(see the documentation in SuiteSparse:GraphBLAS for details).  At least on
Linux or Mac, if GraphBLAS is not installed system-wide, LAGraph can find it if
GraphBLAS appears in the same folder as LAGraph, so you do not need system
privileges to use GraphBLAS.

LAGraph includes a `CMakeLists.txt` file that does the bulk of the work to
build the package.  Also included is a very simple `Makefile` that simplifies
the use of `make` for Linux and MacOS.  In Linux or Mac, you can use it to run
these commands::

    cd LAGraph
    make
    make test

If you have system admin privileges, you can then install LAGraph::

    sudo make install

On Windows, the `CMakeLists.txt` file can be imported into MS Visual Studio,
and LAGraph can be built directly from there.

