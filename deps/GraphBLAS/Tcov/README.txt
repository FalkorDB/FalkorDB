SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

GraphBLAS/Tcov: statement coverage tests

Requirements:  the mex command must use a C compiler supporting C11.
Microft Visual Studio does not support C11 so this test is not available
on Windows unless you use another compiler.

Since nearly all GraphBLAS tests are in *.m files, I have taken the unusual
step of creating a statement coverage mechanism to use within a mexFunction.
To compile GraphBLAS for statement coverage testing, and to run the tests, type
this in the Command Window.

    grbcov

If you get a linking problem on linux, add this directory to your
LD_LIBRARY_PATH, so that the libgraphblas_tcov.so constructed by grbmake can be
found by the mexFunctions.

Statement coverage tests results will be saved in Tcov/log.txt.

The lines covered by the test are marked in each file in tmp_cover/.

To remove all compiled files, type this in the Unix/Linux shell:

    make distclean

Or, delete these files manually:

    *.o *.obj *.mex* cover_*.c errlog*.txt grbstat.mat tmp*/*

To also remove the log.txt file:

    make purge

--------------------------------------------------------------------------------
Files in GraphBLAS/Tcov:
--------------------------------------------------------------------------------

    Contents.m     for 'help Tcov'; list of files

    grbcov.m        makes the tests, runs them, and lists the test coverage
    grbcover.m      compile GraphBLAS for statement coverage testing
    grbcover_edit.m create a version of GraphBLAS for statement coverage tests
    testcov.m       run all GraphBLAS tests, with statement coverage
    grbshow.m       create a test coverage report in tmp_cover/
    Makefile        just for 'make clean' and 'make purge'
    README.txt      this file

    log_*.txt           100% test coverage certificates

    tmp_cover       where coverage reports are placed
    tmp_include     for include files augmented with coverate tests
    tmp_source      for source files augmented with coverate tests

