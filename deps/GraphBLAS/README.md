# SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

SPDX-License-Identifier: Apache-2.0

VERSION 10.3.0, Dec 3, 2025

SuiteSparse:GraphBLAS is a complete implementation of the GraphBLAS standard,
which defines a set of sparse matrix operations on an extended algebra of
semirings using an almost unlimited variety of operators and types.  When
applied to sparse adjacency matrices, these algebraic operations are equivalent
to computations on graphs.  GraphBLAS provides a powerful and expressive
framework for creating graph algorithms based on the elegant mathematics of
sparse matrix operations on a semiring.

SuiteSparse:GraphBLAS is used heavily in production.  It appears as the
underlying graph engine in the RedisGraph database by Redis, and as the
built-in sparse matrix multiply in MATLAB R2021a, where `C=A*B` is now up to
30x faster than in prior versions of MATLAB (on my 20-core NVIDIA DGX Station).

The development of this package is supported by Intel, NVIDIA (including the
donation of the 20-core DGX Station), Redis, MIT Lincoln Lab, MathWorks,
IBM, and Julia Computing.

See the user guide in `Doc/GraphBLAS_UserGuide.pdf` for documentation on the
SuiteSparse implementation of GraphBLAS, and how to use it in your
applications.

See http://graphblas.org for more information on GraphBLAS, including the
GraphBLAS C API.  See https://github.com/GraphBLAS/GraphBLAS-Pointers for
additional resources on GraphBLAS.

QUICK START: To compile and install, do these commands in this directory:

    make
    sudo make install

Please be patient; some files can take several minutes to compile.  Requires a
C11 compiler, so cmake will fail if your compiler is not C11 compliant.
See the User Guide PDF in Doc/ for directions on how to use another compiler.

For faster compilation, do this instead of just "make", which uses 32
parallel threads to compile the package:

    make JOBS=32

The output of the demo programs will be compared with their expected output.

To remove all compiled files:

    make clean

To compile and run the demos:

    make demos

See the GraphBLAS/ subfolder for the Octave/MATLAB interface, which contains a
README.md file with further details.

--------------------------------------------------------------------------------
## Files and folders in this GraphBLAS directory:

CMakeLists.txt:  cmake instructions to compile GraphBLAS

`cmake_modules`:  additional cmake files

Config:         version-dependent files used by CMake

Demo:           a set of demos on how to use GraphBLAS

Doc:            SuiteSparse:GraphBLAS User Guide and license

GraphBLAS:      the @GrB Octave/MATLAB interface, including its test suite and
                demos.  This folder is called 'GraphBLAS' so that typing 'help
                graphblas' or 'doc graphblas' in the Octave or MATLAB Command
                Window can locate the Contents.m file.

Include:        user-accessible include file, GraphBLAS.h

Makefile:       to compile the SuiteSparse:GraphBLAS library and demos

README.md:      this file

Source:         source files of the SuiteSparse:GraphBLAS library.

Tcov:           test coverage, requires Octave or MATLAB.  See the log...txt
                files, which certify 100% test coverage.

Test:           Extensive tests, not meant for general usage.  To compile and
                run, go to this directory and type make;testall in Octave or
                MATLAB.  Requires Octave or MATLAB

build:          build directory for CMake, initially empty

CUDA:           GPU interface, a work in progress.  This is being developed in
                collaboration with Joe Eaton, Corey Nolet and others at NVIDIA,
                with support from NVIDIA.  It appears in this release but the
                CUDA folder is a draft that isn't ready to use yet.

CONTRIBUTOR-LICENSE.txt:    how to contribute to SuiteSparse:GraphBLAS

`cpu_features`: (c) Google.com, Apache 2.0 license.

logo:           the (awesome!) GraphBLAS logo by Jakab Rokob, CC BY 4.0 license

lz4:            LZ4 compression, (c) 2011-2016, Yann Collet, BSD2 license
zstd:           ZSTD compression, (c) Meta, by Yann Collet, BSD3 license
xxHash:         xxHash code, (c) 2012-2021, Yann Collet

`rmm_wrap`:     Rapids Memory Manager, (c) NVIDIA, to use with CUDA.
                (draft; not yet in use)


JITpackage:     a small program that packages the GraphBLAS source code into
                the GraphBLAS library itself so it can compile the JIT kernels.
                If you edit the GraphBLAS source code, see the README.txt file
                in this director for instructions.

LICENSE:        licenses for GraphBLAS and its 3rd party dependencies

PreJIT:         a folder for JIT kernels that are to be integrated into the
                compiled GraphBLAS library.

--------------------------------------------------------------------------------

## GraphBLAS C API Specification:

Versions v5.2.0 and earlier conform to the version 1.3.0 (Sept 25, 2019) of the
GraphBLAS C API Specification.  Versions v6.0.0 and later conform to the
version 2.0.0 (Nov, 2021) of the GraphBLAS C API Specification.   Versions
9.0.0 and later conform to the v2.1.0 C API.  This library also includes
several additional functions and features as extensions to the spec.

All functions, objects, and macros with the prefix GxB are extensions to
the spec.  Functions, objects, and macros with prefix GB must not be accessed
by user code.  They are for internal use in GraphBLAS only.

--------------------------------------------------------------------------------

## About Benchmarking

Do not use the demos in GraphBLAS/Demos for benchmarking or in production.
Those are simple methods for illustration only, and can be slow.  Use LAGraph
for benchmarking and production uses.

I have tested this package extensively on multicore single-socket systems, but
have not yet optimized it for multi-socket systems with a NUMA architecture.
That will be done in a future release.  If you publish benchmarks
with this package, please state the SuiteSparse:GraphBLAS version, and a caveat
if appropriate.  If you see significant performance issues when going from a
single-socket to multi-socket system, I would like to hear from you so I can
look into it.

Contact me at davis@tamu.edu for any questions about benchmarking
SuiteSparse:GraphBLAS and LAGraph.

--------------------------------------------------------------------------------

## Contributing to SuiteSparse:GraphBLAS

To add an issue for a bug report (gasp!) or a feature request,
you can use the issue tracker on github.com, at
[`https://github.com/DrTimothyAldenDavis/GraphBLAS/issues`]
(https://github.com/DrTimothyAldenDavis/GraphBLAS/issues) or
[`https://github.com/DrTimothyAldenDavis/SuiteSparse/issues`]
(https://github.com/DrTimothyAldenDavis/SuiteSparse/issues).

To contribute code, you can submit a pull request.  To do so,
you must first agree to the Contributor License Agreement
[`CONTRIBUTOR-LICENSE.txt`](CONTRIBUTOR-LICENSE.txt).
Print a copy of the txt file (as a PDF), sign and date it,
and email it to me at DrTimothyAldenDavis@gmail.com.  Pull
requests will only be included into SuiteSparse after I receive
your email with the signed PDF.

--------------------------------------------------------------------------------

## Licensing and supporting SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS is released primarily under the Apache-2.0 license,
because of how the project is supported by many organizations (NVIDIA, Redis,
MIT Lincoln Lab, Intel, IBM, and Julia Computing), primarily through gifts to
the Texas A&M Foundation.  Because of this support, and to facilitate the
wide-spread use of GraphBLAS, the decision was made to give this library a
permissive open-source license (Apache-2.0).  Currently all source code
required to create the C-callable library libgraphblas.so is licensed with
Apache-2.0, and there are no plans to change this.

However, just because this code is free to use doesn't make it zero-cost to
create.  If you are using GraphBLAS in a commercial closed-source product and
are not supporting its development, please consider supporting this project
to ensure that it will continue to be developed and enhanced in the future.

To support the development of GraphBLAS, contact the author (davis@tamu.edu) or
the Texas A&M Foundation (True Brown, tbrown@txamfoundation.com; or Kevin
McGinnis, kmcginnis@txamfoundation.com) for details.

SuiteSparse:GraphBLAS, is copyrighted by Timothy A. Davis, (c) 2017-2025, All
Rights Reserved.  davis@tamu.edu.

-----------------------------------------------------------------------------

## For distro maintainers (Linux, homebrew, spack, R, Octave, Trilinos, ...):

Thanks for packaging SuiteSparse!  Here are some suggestions:

    * GraphBLAS takes a long time to compile because it creates many fast
        "FactoryKernels" at compile-time.  If you want to reduce the compile
        time and library size, enable the GRAPHBLAS_COMPACT mode, but keep the
        JIT enabled.  Then GraphBLAS will compile the kernels it needs at
        run-time, via its JIT.  Performance will be the same as the
        FactoryKernels once the JIT kernels are compiled.  User compiled
        kernels are placed in ~/.SuiteSparse, by default.  You do not need to
        distribute the source for GraphBLAS to enable the JIT: just
        libgraphblas.so and GraphBLAS.h is enough.

    * GraphBLAS needs OpenMP!  It's fundamentally a parallel code so please
        distribute it with OpenMP enabled.  Performance will suffer
        otherwise.

--------------------------------------------------------------------------------

## References:

To cite this package, please use the following:

    T. A. Davis. Algorithm 1037: SuiteSparse:GraphBLAS: Parallel Graph
    Algorithms in the Language of Sparse Linear Algebra. ACM Trans. Math.
    Softw. 49, 3, Article 28 (September 2023), 30 pages.
    https://doi.org/10.1145/3577195

    T. Davis, Algorithm 1000: SuiteSparse:GraphBLAS: graph algorithms in
    the language of sparse linear algebra, ACM Trans on Mathematical
    Software, vol 45, no 4, Dec. 2019, Article No 44.
    https://doi.org/10.1145/3322125.

--------------------------------------------------------------------------------
## Software Acknowledgements

SuiteSparse:GraphBLAS relies on the following packages (details in the LICENSE
file, and in the GraphBLAS User Guide):

(1) LZ4, xxHash, and ZSTD by Yann Collet, appearing here under the
BSD2 or BSD3 licenses.

(2) `cpu_features` (c) Google, Apache 2.0 license with components (c) IBM and
Intel (also Apache 2.0), and the `cpu_featurer/ndk_compat` component (c)
The Android Open Source Project (BSD-2-clause)

