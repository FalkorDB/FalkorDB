# SuiteSparse/GraphBLAS/GraphBLAS/test

This folder, GraphBLAS/GraphBLAS/test, provides a 100% test for the
interface to GraphBLAS.  For a full test of the GraphBLAS library itself,
see GraphBLAS/Test and GraphBLAS/Tcov.

To run this test, first compile the GraphBLAS library by typing 'make' in the
top-level GraphBLAS folder, in your system shell.  That statement will use
cmake to compile GraphBLAS.  Use 'make JOBS=40' to compile in parallel (replace
'40' with the number of cores in your system).  Next, go to the
GraphBLAS/GraphBLAS/@GrB/private folder, and type the following in the
Command Window.

# Usage:

   cd GraphBLAS/GraphBLAS
   help GraphBLAS      % short description of the interface to GraphBLAS
   addpath (pwd) ;
   savepath ;          % if this fails, edit your startup.m file instead
   cd @GrB/private
   gbmake ;            % compile the interface to GraphBLAS
   cd ../../test       % this folder
   gbtest

If the savepath command fails, then add the following line to your startup.m:

    addpath ('/your/path/to/SuiteSparse/GraphBLAS/GraphBLAS')

where "/your/path/to/" should be replaced with the actual path of where
GraphBLAS resides in your file system.

See the tcov subfolder to run the test with statement coverage of the
C mexFunctions and utility routines.

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

