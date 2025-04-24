[![Build Status](https://github.com/GraphBLAS/LAGraph/workflows/LAGraph%20CI/badge.svg)](https://github.com/GraphBLAS/LAGraph/actions)
[![Documentation Status](https://readthedocs.org/projects/lagraph/badge/?version=latest)](https://lagraph.readthedocs.io/en/latest/?badge=latest)

# LAGraph

LAGraph is a library plus a test harness for collecting algorithms that
use GraphBLAS.

See <https://github.com/GraphBLAS/LAGraph> for the source code for LAGraph,
Documenation is at <https://lagraph.readthedocs.org>.
Test coverage results are at <https://graphblas.org/LAGraph>.

Currently, SuiteSparse:GraphBLAS v7.0.0 or later is required.  However, use the
latest stable release of SuiteSparse:GraphBLAS for best results.
See <https://github.com/DrTimothyAldenDavis/GraphBLAS>

A simple Makefile is provided but its use is optional. It simplifies the
use of cmake, which is the primary build system for LAGraph.  For Windows,
import the CMakeLists.txt into MS Visual Studio instead.

To compile, run the tests, and install (Linux, Mac):
```
        make
        make test
        sudo make install
```

To compile/install for elsewhere (for example, in /home/me/mystuff/lib
and /home/me/mystuff/include), do not use this Makefile.  Instead, do:
```
        cd build
        cmake -DCMAKE_INSTALL_PREFIX="/home/me/mystuff" ..
        make
        make install
```

To clean up the files:
```
        make clean
```

To uninstall:
```
        make uninstall
```

To compile and run test coverage: use "make cov".  Next, open your browser to
your local file, LAGraph/build/test_coverage/index.html.  Be sure to do "make
clean" afterwards, and then "make" to compile without test coverage.

The test coverage of the latest [CI build](https://github.com/GraphBLAS/LAGraph/actions) is deployed to <https://graphblas.github.io/LAGraph/>.

To run the GAP benchmarks, see the instructions in this file:
```
./src/benchmark/README.md
```

# LAGraph contents

LAGraph contains the following files and folders:

    CMakeLists.txt: a CMake script for compiling/installing/testing LAGraph.

    cmake_modules:  helper scripts for CMake to find GraphBLAS and to provide
        test coverage

    data: small test matrices for the continuous integration tests

    deps: 3rd party dependencies

    doc: documentation

    include: contains the LAGraph.h and LAGraphX.h files
        Do not edit include/LAGraph.h, since it is constructed
        from config/LAGraph.h.in.

    LICENSE: BSD 2-clause license

    README.md: this file

    src: stable source code for the LAGraph library (LAGraph.h)

        * algorithms: graph algorithms such as BFS, connected components,
            centrality, etc

        * utilities: read/write a graph from a file, etc

    experimental*: draft code under development: (LAGraphX.h)
        do not benchmark without asking the LAGraph authors first

        * algorithms: draft graph algorithms such as Maximal Independent Set

        * utilities: draft utilities

    build: initially empty

    Acknowledgments.txt

    ChangeLog: changes since LAGraph v1.0

    config: LAGraph.h.in, for constructing include/LAGraph.h

    CONTRIBUTING.md: how to contributed to LAGraph

    CODE_OF_CONDUCT.md: code of conduct from
        https://www.contributor-covenant.org/version/2/1/code_of_conduct.html

    Contributors.txt: list of contributors

    Dockerfile

    Makefile: simple scripts that rely on cmake

    papers: papers on LAGraph

    rtdocs: source for the LAGraph documentation

# LAGraph and GraphBLAS

To link against GraphBLAS, first install whatever GraphBLAS library you wish to
use.  LAGraph will use -lgraphblas and will include the GraphBLAS.h file
from its installed location.  Alternatively, the CMakeLists.txt script can use
a relative directory:

    ../GraphBLAS: any GraphBLAS implementation.

So that LAGraph and GraphBLAS reside in the same parent folder.  The include
file for GraphBLAS will be assumed to appear in ../GraphBLAS/Include, and the
compiled GraphBLAS library is assumed to appear in ../GraphBLAS/build.  The
CMake should find GraphBLAS, but if you use a GraphBLAS library that uses a
different structure, then edit the CMakeLists.txt file to point to right
location.

# Authors

    See the Contributors.txt file

