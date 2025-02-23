SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

This is the GraphBLAS/Demo folder.  It contains a set of simple demo programs
that illustrate the use of GraphBLAS.  To compile and run the demos, see
../README.txt.  These methods are not meant as benchmarks; use LAGraph for
that.

--------------------------------------------------------------------------------
Files in this folder:

    README.txt              this file
    demo                    run all demos
    vdemo                   run all demos with valgrind
    wdemo                   run the wathen_demo with larger matrices

--------------------------------------------------------------------------------
in Demo/Program:
--------------------------------------------------------------------------------

    complex_demo.c          a user-defined complex number type
    context_demo.c          how to use the GxB_Context for nested parallelism
    gauss_demo.c            the Gaussian integer: an int with real/imag parts
    gauss.m                 the same as gauss_demo.c, but in MATLAB
    grow_demo.c             growing a matrix, one or many rows at a time
    simple_demo.c           a test for simple_rand
    wathen_demo.c           creating a finite element matrix
    wildtype_demo.c         an arbitrary user-defined type

--------------------------------------------------------------------------------
in Demo/Output:
--------------------------------------------------------------------------------

    complex_demo_out2.m     output of complex_demo
    complex_demo_out.m      output of complex_demo
    context_demo.out        output of context_demo
    gauss_demo1.out         output of gauss_demo
    gauss_demo.out          output of gauss_demo
    simple_demo.out         output of simple_demo
    wathen_demo.out         output of wathen_demo
    wildtype_demo.out       output of wildtype_demo

--------------------------------------------------------------------------------
in Demo/Include:
--------------------------------------------------------------------------------

    get_matrix.c            get a matrix (file, Wathen, or random)
    graphblas_demos.h       include file for all demos
    isequal.c               test if 2 matrices are equal
    random_matrix.c         create a random matrix
    read_matrix.c           read a matrix from a file (Matrix/*)
    simple_rand.h           simple random number generator
    usercomplex.c           user-defined double complex type
    usercomplex.h           include file for usercomplex.h
    wathen.c                GraphBLAS version of wathen.m

--------------------------------------------------------------------------------
in Demo/Matrix:
--------------------------------------------------------------------------------

folder with test matrices, with 0-based indices. Many of these are derived from
the Harwell/Boeing matrix collection.  Contains:

    2blocks
    ash219
    bcsstk01
    cover.mtx
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

