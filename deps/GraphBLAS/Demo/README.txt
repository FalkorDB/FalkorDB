SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

This is the GraphBLAS/Demo folder.  It contains a set of simple demo programs
that illustrate the use of GraphBLAS.  To compile and run the demos, see
../README.txt.

vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
>>> DO NOT BENCHMARK ANY OF THESE PROGRAMS <<<
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    These methods are simple ones, meant for illustration only.  They can be
    slow.  Use LAGraph for benchmarking and use in production.  Do not use any
    of these methods in user applications.  Eventually, this Demo folder will
    be removed, and its purpose will be achieved in LAGraph instead.

--------------------------------------------------------------------------------
Files in this folder:

    README.txt              this file
    demo                    run all demos

--------------------------------------------------------------------------------
in Demo/Program:
--------------------------------------------------------------------------------

    complex_demo.c          demo program to test complex type
    import_demo.c           demo program to test import/export
    kron_demo.c             demo program to test GrB_kronecker
    simple_demo.c           demo program to test simple_rand
    wildtype_demo.c         demo program, arbitrary struct as user-defined type
    openmp_demo.c           demo program using OpenMP

--------------------------------------------------------------------------------
in Demo/Output:
--------------------------------------------------------------------------------

    complex_demo_out.m  output of complex_demo
    simple_test.out     output of simple_demo
    tri_demo.out        output of tri_demo
    wildtype_demo.out   output of wildtype_demo
    import_demo.out     output of import_demo

--------------------------------------------------------------------------------
in Demo/Include:
--------------------------------------------------------------------------------

    graphblas_demos.h       include file for all demos
    simple_rand.h           include file for simple_rand.c
    usercomplex.h           include file for usercomplex.h

    get_matrix.c            get a matrix (file, Wathen, or random)
    random_matrix.c         create a random matrix
    read_matrix.c           read a matrix from a file (Matrix/*)
    simple_rand.c           a very simple random number generator
    usercomplex.c           user-defined double complex type
    wathen.c                GraphBLAS version of wathen.m
    import_test.c           test import/export
    isequal.c               test if 2 matrices are equal

--------------------------------------------------------------------------------
in Demo/Matrix:
--------------------------------------------------------------------------------

folder with test matrices, with 0-based indices. Many of these are derived from
the Harwell/Boeing matrix collection.  Contains:

    2blocks
    ash219
    bcsstk01
    eye3
    fs_183_1
    huge
    ibm32a
    ibm32b
    lp_afiro
    mbeacxc
    t1
    t2
    west0067

