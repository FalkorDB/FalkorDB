# GraphBLAS/GraphBLAS: MATLAB/Octave interface for SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The @GrB class provides an easy-to-use interface to SuiteSparse:GraphBLAS.

To install it for use in MATLAB/Octave, first compile the GraphBLAS library,
-lgraphblas (for Octave) or -lgraphblas_matlab (for MATLAB).  See the
instructions in the top-level GraphBLAS folder for details.  Be sure to use
OpenMP for best performance.

MATLAB (not Octave) the gbmake script will link against the library
-lgraphblas_matlab, not -lgraphblas, because that version of MATLAB includes
its own version of SuiteSparse:GraphBLAS (v3.3.3, an earlier one).  To avoid a
name conflict, you must compile the -lgraphblas_matlab library in
/home/me/SuiteSparse/GraphBLAS/GraphBLAS/build.

On Windows 10, on the Search bar type env and hit enter; (or you can
right-click My Computer or This PC and select Properties, and then select
Advanced System Settings).  Select "Edit the system environment variables",
then "Environment Variables".  Under "System Variables" select "Path" and click
"Edit".  These "New" to add a path and then "Browse".  Browse to the folder
(for example: C:/Users/me/Documents/SuiteSparse/GraphBLAS/build/Release) and
add it to your path.  For MATLAB, you must use the
libgraphblas_matlab.dll, in:
/User/me/SuiteSparse/GraphBLAS/GraphBLAS/build/Release instead.  Then close the
editor, sign out of Windows and sign back in again.

Next, start MATLAB/Octave and go to this GraphBLAS/GraphBLAS folder.  Type

    addpath (pwd)

to add the GraphBLAS interface to your path.  Then do

    savepath

Or, if that function is not allowed because of file permissions, add this
command to your startup.m file:

    % add the MATLAB/Octave interface to the MATLAB/Octave path
    addpath ('/home/me/SuiteSparse/GraphBLAS/GraphBLAS') :

where the path /home/me/SuiteSparse/GraphBLAS/GraphBLAS is the full path to
this folder.

The name "GraphBLAS/GraphBLAS" is used for this folder so that this can be done
in MATLAB/Octave:

    help GraphBLAS

To get additional help, type:

    methods GrB
    help GrB

Next, go to the GraphBLAS/GraphBLAS/@GrB/private folder and compile the
MATLAB/Octave mexFunctions.  Assuming your working directory is
GraphBLAS/GraphBLAS (where this README.md file is located), do the following:

    cd @GrB/private
    gbmake

To run the demos, go to the GraphBLAS/GraphBLAS/demo folder and type:

    gbdemo
    gbdemo2

The output of these demos on a Dell XPS 13 laptop and an NVIDIA DGX Station can
also be found in GraphBLAS/GraphBLAS/demo/html, in both PDF and HTML formats.

To test your installation, go to GraphBLAS/GraphBLAS/test and type:

    gbtest

If everything is successful, it should report 'gbtest: all tests passed'.  Note
that gbtest tests all features of the MATLAB/Octave interface to
SuiteSparse/GraphBLAS, including error handling, so you can expect to see error
messages during the test.  This is expected.

# FUTURE: Not yet supported for GrB matrices in MATLAB/Octave:

    linear indexing, except for C=A(:) to index the whole matrix A
        or C(:)=A to index the whole matrix C.
    2nd output for [x,i] = max (...) and [x,i] = min (...):
        use GrB.argmin and GrB.argmax instead.
    'includenan' for min and max
    min and max for complex matrices
    singleton expansion
    saturating element-wise binary and unary operators for integers.
        See also the discussion in the User Guide.

These functions are supported, but are not yet as fast as they could be:
eps, ishermitian, issymmetric, spfun.

