# GraphBLAS/GraphBLAS: MATLAB/Octave interface for SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The @GrB class provides an easy-to-use interface to SuiteSparse:GraphBLAS.
This README.md file explains how to install it for use in MATLAB/Octave on
Linux, Mac, or Windows.

--------------------------------------------------------------------------------
# For Mac
--------------------------------------------------------------------------------

    This can be a little complicated for MATLAB because it does not support the
    use of OpenMP inside compiled mexFunctions.  It doesn't seem to be a
    problem when using Octave on the Mac.

    First, install brew from https://brew.sh.

    If using octave, you should use the octave available via homebrew:

        brew install octave

    For both MATLAB and Octave, must install the OpenMP library from brew
    (this is likely installed by 'brew install octave'):

        brew install libomp

    Next, add the following to your ~/.zshrc file:

        export OpenMP_ROOT=$(brew --prefix)/opt/libomp

    Next, restart your terminal shell before continuing the steps in the
    section "For Linux/Mac" below.

    HOWEVER, this may fail on MATLAB.

    MATLAB on the Mac comes with its own copy of libomp.dylib, typically

        /Applications/MATLAB_R2024b.app/bin/maca64/libomp.dylib

    for R2024b (for example).  GraphBLAS is compiled against the brew
    libomp.dylib but then linked with the above libomp.dylib inside MATLAB,
    since GraphBLAS in MATLAB must use the same OpenMP library as the rest of
    MATLAB.  However, this causes a link error on MacOSx 15.5 (Xcode 16.3),
    since MATLAB R2024b ships with an older and incompatible version of libomp.
    If you get the following error, you cannot use OpenMP in GraphBLAS on the
    Mac:

        Undefined symbols for architecture arm64:
        "___kmpc_dispatch_deinit", referenced from: ...

    There currently is no workaround for this issue, except to compile
    GraphBLAS without OpenMP.  If you encounter this problem, replace the use
    of "graphblas_install" in the instructions in the next section below with

        graphblas_install ('-DGRAPHBLAS_USE_OPENMP=0')

    GraphBLAS will be slower without OpenMP, but it will work.  This issue on
    the Mac does not arise when using GraphBLAS outside of MATLAB.  Octave does
    not have this issue since it relies on the brew-installed libomp.

    This issue does not prohibit the use of OpenMP with GraphBLAS outside of
    MATLAB, which uses the GraphBLAS/build/libgraphblas.dylib compiled library.
    whereas MATLAB uses the GraphBLAS/GraphBLAS/build/libgraphblas_matlab.dylib
    compiled library on the Mac.

--------------------------------------------------------------------------------
# For Linux/Mac
--------------------------------------------------------------------------------

    To install GraphBLAS for use in MATLAB/Octave, do the following inside the
    MATLAB/Octave Command Window:

        cd /home/me/GraphBLAS/GraphBLAS
        graphblas_install
        addpath (pwd)
        cd test
        gbtest

    That should be enough.  However, the above script may fail if the
    graphblas_install script is unable to use "system ('cmake ...')" to
    use cmake to build GraphBLAS.

    If this happens, the script will print a set of commands you can type in
    your system shell to first compile the GraphBLAS library outside of
    MATLAB/Octave, instead of using the graphblas_install.m script.  Use those
    instructions, or continue with the following (both should work OK):

    Suppose your copy of GraphBLAS is in /home/me/GraphBLAS.  For MATLAB on
    Linux/Mac, compile libgraphblas_matlab.so (.dylib on the Mac) with:

        cd /home/me/GraphBLAS/GraphBLAS
        make

    For Octave on Linux/Mac, compile libgraphblas.so (.dylib on the Mac) with:

        cd /home/me/GraphBLAS
        make

    If the 'make' command above fails, do the following instead (assuming you
    are in the /home/me/GraphBLAS/GraphBLAS folder for MATLAB, or
    /home/me/GraphBLAS for Octave), outside of MATLAB/Octave:

        cd build
        cmake  ..
        cmake --build . --config Release -j40

    Then inside MATLAB/Octave, do this:

        cd /home/me/GraphBLAS/GraphBLAS/@GrB/private
        gbmake

--------------------------------------------------------------------------------
# For Windows
--------------------------------------------------------------------------------

    First try the above instructions for Linux/Mac to build GraphBLAS from
    inside MATLAB.  If this doesn't work, try the following:

    On Windows, on the Search bar type env and hit enter; (or you can
    right-click My Computer or This PC and select Properties, and then select
    Advanced System Settings).  Select "Edit the system environment variables",
    then "Environment Variables".  Under "System Variables" select "Path" and
    click "Edit".  These "New" to add a path and then "Browse".  Browse to the
    folder (for example: C:/Users/me/Documents/GraphBLAS/build/Release) and add
    it to your path.  For MATLAB, you must use the libgraphblas_matlab.dll, in:
    /User/me/SuiteSparse/GraphBLAS/GraphBLAS/build/Release instead.  Then close
    the editor, sign out of Windows and sign back in again.

    Then do this inside of MATLAB/Octave:

        cd /home/me/GraphBLAS/GraphBLAS/@GrB/private
        gbmake

--------------------------------------------------------------------------------
# After installation on Linux/Mac/Windows
--------------------------------------------------------------------------------

    Add this command to your startup.m file:

        % add the MATLAB/Octave interface to the MATLAB/Octave path
        addpath ('/home/me/GraphBLAS/GraphBLAS') :

    where the path /home/me/GraphBLAS/GraphBLAS is the full path to this
    folder.

    The name "GraphBLAS/GraphBLAS" is used for this folder so that this can be
    done in MATLAB/Octave:

        help GraphBLAS

    To get additional help, type:

        methods GrB
        help GrB

    To run the demos, go to the GraphBLAS/GraphBLAS/demo folder and type:

        gbdemo
        gbdemo2

    To test your installation, go to GraphBLAS/GraphBLAS/test and type:

        gbtest

    If everything is successful, it should report 'gbtest: all tests passed'.
    Note that gbtest tests all features of the MATLAB/Octave interface to
    SuiteSparse/GraphBLAS, including error handling, so you can expect to see
    error messages during the test.  This is expected.

--------------------------------------------------------------------------------
# MATLAB vs Octave
--------------------------------------------------------------------------------

    You cannot use a single copy of the GraphBLAS source distribution to use in
    both MATLAB and Octave on the same system at the same time.  The .o files
    in GraphBLAS/GraphBLAS/@GrB/private compiled by the graphblas_install.m
    will conflict with each other.  To switch between MATLAB and Octave, use a
    second copy of the GraphBLAS source distribution, or do a clean
    installation (via "make purge" in the GraphBLAS/GraphBLAS/@GrB/private
    folder, outside of MATLAB/Octave) and redo the above instructions.  There
    is no need to recompile the libgraphblas.so (or dylib on the Mac) since
    Octave uses GraphBLAS/build/libgraphblas.so while MATLAB uses
    GraphBLAS/GraphBLAS/build/libgraphblas_matlab.so.  Both MATLAB and Octave
    can share the same compiled JIT kernels.

--------------------------------------------------------------------------------
# FUTURE: Not yet supported for GrB matrices in MATLAB/Octave:
--------------------------------------------------------------------------------

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

