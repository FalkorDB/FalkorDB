#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparsePolicy.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022-2025, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse CMake policies.  The following parameters can be defined prior
# to including this file:
#
#   CMAKE_BUILD_TYPE:   if not set, it is set below to "Release".
#                       To use the "Debug" policy, precede this with
#                       set ( CMAKE_BUILD_TYPE Debug )
#
#   SUITESPARSE_USE_CUDA: if OFF, CUDA is disabled.  if ON, CUDA is enabled,
#                       if available.  Default: ON.
#
#   SUITESPARSE_LOCAL_INSTALL:      if true, "cmake --install" will install
#                       into SuiteSparse/lib and SuiteSparse/include.
#                       if false, "cmake --install" will install into the
#                       default prefix (or the one configured with
#                       CMAKE_INSTALL_PREFIX).  Requires cmake 3.19.
#                       This is ignored when using the root CMakeLists.txt.
#                       Set CMAKE_INSTALL_PREFIX instead.
#                       Default: OFF
#
#   BUILD_SHARED_LIBS:  if true, shared libraries are built.
#                       Default: ON.
#
#   BUILD_STATIC_LIBS:  if true, static libraries are built.
#                       Default: ON, except for GraphBLAS, which
#                       takes a long time to compile so the default for
#                       GraphBLAS is false.
#
#   SUITESPARSE_CUDA_ARCHITECTURES:  a string, such as "all" or
#                       "35;50;75;80" that lists the CUDA architectures to use
#                       when compiling CUDA kernels with nvcc.  The "all"
#                       option requires cmake 3.23 or later.
#                       Default: "52;75;80".
#
#   BLA_VENDOR and BLA_SIZEOF_INTEGER: By default, SuiteSparse searches for
#                       the BLAS library in a specific order.  If you wish to
#                       use a specific BLAS library, set both of these with
#                       (for example):
#                       -DBLA_VENDOR=Intel10_64lp -DBLA_SIZEOF_INTEGER=4
#                       Both settings must appear, or neither.
#                       Default: neither are defined.
#
#   BLA_STATIC:         if ON, use static linkage for BLAS and LAPACK.
#                       Default: not set (that is, the same as OFF)
#
#   SUITESPARSE_USE_64BIT_BLAS    if true, SuiteSparse will search for both
#                       32-bit and 64-bit BLAS.  If false, only 32-bit BLAS
#                       will be searched for.  Ignored if BLA_VENDOR and
#                       BLA_SIZEOF_INTEGER are defined.
#
#   SUITESPARSE_C_TO_FORTRAN:  a string that defines how C calls Fortran.
#                       Defaults to "(name,NAME) name" for Windows (lower case,
#                       no underscore appended to the name), which is the
#                       system that is most likely not to have a Fortran
#                       compiler.  Defaults to "(name,NAME) name##_" otherwise.
#                       This setting is only used if no Fortran compiler is
#                       found.
#
#   SUITESPARSE_USE_FORTRAN:  if OFF, no Fortan files are compiled, and the
#                       Fortran language is not enabled in any cmake scripts.
#                       The built-in cmake script FortranCInterface is skipped.
#                       This will require SUITESPARSE_C_TO_FORTRAN to be
#                       defined explicitly, if the defaults are not appropriate
#                       for your system.
#                       Default: ON
#
#       *** WARNING:  Using the Intel icx compiler, the Intel MKL BLAS, and the
#       GNU Fortran compiler together will result in two OpenMP libraries
#       linked to the UMFPACK, CHOLMOD, ParU, and SPQR libraries.  This will
#       cause serious performance issues.  This cmake file detects if icx and
#       gfortran are in use, and issues a fatal error.  Otherwise, if the C,
#       C++, and Fortran compilers have a different compiler ID, a warning is
#       issued; this is NOT recommended, but it might be OK.  In that case,
#       ensure that you are linking with just one OpenMP library.  If this
#       occurs, disable the use of Fortran by setting SUITESPARSE_USE_FORTRAN
#       to OFF, use a suite of C/C++/Fortran compilers with the same ID,
#       or ensure that all your compiled libraries link against a single
#       OpenMP library.
#
#   SUITESPARSE_PKGFILEDIR: Directory where CMake Config and pkg-config files
#                       will be installed.  By default, CMake Config files will
#                       be installed in the subfolder `cmake` of the directory
#                       where the (static) libraries will be installed (e.g.,
#                       `lib`).  The `.pc` files for pkg-config will be
#                       installed in the subfolder `pkgconfig` of the directory
#                       where the (static) libraries will be installed.
#                       Default: CMAKE_INSTALL_PREFIX, or SuiteSparse/lib if
#                       SUITESPARSE_LOCAL_INSTALL is enabled.
#
#   SUITESPARSE_INCLUDEDIR_POSTFIX : Postfix for installation target of
#                       header from SuiteSparse. Default: suitesparse, so the
#                       default include directory is:
#                       CMAKE_INSTALL_PREFIX/include/suitesparse
#
#   SUITESPARSE_USE_STRICT: SuiteSparse has many user-definable settings of the
#                       form SUITESPARSE_USE_* or (package)_USE_* for some
#                       particular package.  In general, these settings are not
#                       strict.  For example, if SUITESPARSE_USE_OPENMP is
#                       ON then OpenMP is preferred, but SuiteSparse can be
#                       used without OpenMP so no error is generated if OpenMP
#                       is not found.  However, if SUITESPARSE_USE_STRICT is
#                       ON then all *_USE_* settings are treated strictly
#                       and an error occurs if any are set to ON but the
#                       corresponding package or setting is not available.  The
#                       *_USE_SYSTEM_* settings are always treated as strict.
#                       Default: OFF.

message ( STATUS "Source:           ${CMAKE_SOURCE_DIR} ")
message ( STATUS "Build:            ${CMAKE_BINARY_DIR} ")

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0" )
    cmake_policy ( SET CMP0104 NEW )    # initialize CUDA architectures
endif ( )

# SuiteSparse packages have many intentional extra semicolons, for code
# readability (such as "/* do nothing */ ;" in SuiteSparse_config.c).  Disable
# the clang warning for these statements:
if ( CMAKE_C_COMPILER_ID STREQUAL "Clang" )
    set ( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-extra-semi" )
endif ( )
if ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-extra-semi" )
endif ( )

if ( WIN32 )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true )
    add_compile_definitions ( _CRT_SECURE_NO_WARNINGS )
endif ( )

enable_language ( C )
include ( GNUInstallDirs )

# add the cmake_modules folder for this package to the module path
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules )

# Use OpenMP
option ( SUITESPARSE_USE_OPENMP "ON (default): Use OpenMP in libraries by default if available.  OFF: Do not use OpenMP by default." ON )

# control the use of Python interfaces in SuiteSparse packages (currently only
# for SPEX)
option ( SUITESPARSE_USE_PYTHON "ON (default): build Python interfaces for SuiteSparse packages (SPEX).  OFF: do not build Python interfaces for SuiteSparse packages" ON )

# strict usage
option ( SUITESPARSE_USE_STRICT "ON: treat all _USE__ settings as strict if they are ON. OFF (default): consider *_USE_* as preferences, not strict" OFF )

# build the demos
option ( SUITESPARSE_DEMOS "ON: Build the demo programs.  OFF (default): do not build the demo programs." OFF )

# BUILD_SHARED_LIBS and BUILD_STATIC_LIBS options
option ( BUILD_SHARED_LIBS "OFF: do not build shared libraries.  ON (default): build shared libraries" ON )

if ( BUILD_STATIC_LIBS_DEFAULT_OFF )
    option ( BUILD_STATIC_LIBS "OFF (default): do not build static libraries.  ON: build static libraries" OFF )
else ( )
    # For backwards compatibility, use NSTATIC if it is set.
    if ( NSTATIC )
        option ( BUILD_STATIC_LIBS "OFF: do not build static libraries.  ON (default): build static libraries" OFF )
    else ( )
        option ( BUILD_STATIC_LIBS "OFF: do not build static libraries.  ON (default): build static libraries" ON )
    endif ( )
endif ( )

if ( NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS )
    message ( FATAL_ERROR "At least one of BUILD_SHARED_LIBS or BUILD_STATIC_LIBS must be set to ON." )
endif ( )

# installation options
if ( NOT SUITESPARSE_ROOT_CMAKELISTS AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
    # the SUITESPARSE_LOCAL_INSTALL option requires cmake 3.19.0 or later
    option ( SUITESPARSE_LOCAL_INSTALL "Install in SuiteSparse/lib" OFF )
else ( )
    set ( SUITESPARSE_LOCAL_INSTALL OFF )
endif ( )

if ( SUITESPARSE_SECOND_LEVEL )
    # some packages in SuiteSparse are in SuiteSparse/Package/Package
    set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        ${CMAKE_SOURCE_DIR}/../../lib/cmake )
else ( )
    # most packages in SuiteSparse are located in SuiteSparse/Package
    set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        ${CMAKE_SOURCE_DIR}/../lib/cmake )
endif ( )

# allow libraries to "see" each other if they are installed in a non-default LIBRARY_PATH
list ( FIND CMAKE_INSTALL_RPATH "$ORIGIN" _idx )
if ( _idx LESS 0 )
    # "$ORIGIN" not yet included in CMAKE_INSTALL_RPATH
    list ( PREPEND CMAKE_INSTALL_RPATH "$ORIGIN" )
endif ( )

set ( INSIDE_SUITESPARSE OFF )
if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    # determine if this Package is inside the SuiteSparse folder
    if ( SUITESPARSE_LOCAL_INSTALL )
        # if you do not want to install local copies of SuiteSparse
        # packages in SuiteSparse/lib and SuiteSparse/, set
        # SUITESPARSE_LOCAL_INSTALL to false in your CMake options.
        if ( SUITESPARSE_SECOND_LEVEL )
            # the package is normally located at the 2nd level inside SuiteSparse
            # (SuiteSparse/GraphBLAS/GraphBLAS/ for example)
            if ( EXISTS ${CMAKE_SOURCE_DIR}/../../SuiteSparse_config )
                set ( INSIDE_SUITESPARSE true )
            endif ( )
        else ( )
            # typical case, the package is at the 1st level inside SuiteSparse
            # (SuiteSparse/AMD for example)
            if ( EXISTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config )
                set ( INSIDE_SUITESPARSE true )
            endif ( )
        endif ( )

        if ( NOT INSIDE_SUITESPARSE )
            message ( FATAL_ERROR "Unsupported layout for local installation. Correct the directory layout or unset SUITESPARSE_LOCAL_INSTALL." )
        endif ( )

    endif ( )
endif ( )

set ( SUITESPARSE_INCLUDEDIR_POSTFIX "suitesparse" CACHE STRING
    "Postfix for installation target of header from SuiteSparse (default: \"suitesparse\")" )

if ( SUITESPARSE_LOCAL_INSTALL )
    if ( INSIDE_SUITESPARSE )
        # ../lib and ../include exist: the package is inside SuiteSparse.
        # find ( REAL_PATH ...) requires cmake 3.19.
        if ( SUITESPARSE_SECOND_LEVEL )
            file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../..
                SUITESPARSE_LOCAL_PREFIX )
        else ( )
            file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/..
                SUITESPARSE_LOCAL_PREFIX )
        endif ( )
    endif ( )
    set ( SUITESPARSE_LIBDIR ${SUITESPARSE_LOCAL_PREFIX}/lib )
    set ( SUITESPARSE_INCLUDEDIR ${SUITESPARSE_LOCAL_PREFIX}/include/${SUITESPARSE_INCLUDEDIR_POSTFIX} )
    set ( SUITESPARSE_BINDIR ${SUITESPARSE_LOCAL_PREFIX}/bin )
else ( )
    set ( SUITESPARSE_LIBDIR ${CMAKE_INSTALL_LIBDIR} )
    set ( SUITESPARSE_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/${SUITESPARSE_INCLUDEDIR_POSTFIX} )
    set ( SUITESPARSE_BINDIR ${CMAKE_INSTALL_BINDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # append ../lib to the install and build runpaths
    list ( APPEND CMAKE_INSTALL_RPATH ${SUITESPARSE_LIBDIR} )
    list ( APPEND CMAKE_BUILD_RPATH ${SUITESPARSE_LIBDIR} )
endif ( )

if ( APPLE )
    # append /usr/local/lib to the install and build runpaths
    message ( STATUS "Add to Mac rpath: " ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} )
    list ( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} )
    list ( APPEND CMAKE_BUILD_RPATH   ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} )
endif ( )

set ( SUITESPARSE_PKGFILEDIR ${SUITESPARSE_LIBDIR} CACHE STRING
    "Directory where CMake Config and pkg-config files will be installed" )

message ( STATUS "Install lib:      ${SUITESPARSE_LIBDIR}" )
message ( STATUS "Install include:  ${SUITESPARSE_INCLUDEDIR}" )
message ( STATUS "Install bin:      ${SUITESPARSE_BINDIR}" )
message ( STATUS "Install pkg-file: ${SUITESPARSE_PKGFILEDIR}" )
message ( STATUS "Install rpath:    ${CMAKE_INSTALL_RPATH}" )
message ( STATUS "Build   rpath:    ${CMAKE_BUILD_RPATH}" )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:       ${CMAKE_BUILD_TYPE} ")

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

#-------------------------------------------------------------------------------
# Workaround for math.h on old macOS
#-------------------------------------------------------------------------------

if ( APPLE AND CMAKE_SYSTEM_VERSION VERSION_LESS 11 )
    # Older versions of math.h on the Mac define Real and Imag, which
    # conflict with the internal use of those symbols in CHOLMOD, KLU, SPQR,
    # UMFPACK, and ParU.
    # This can be disabled with -D__NOEXTENSIONS__
    message ( STATUS "MacOS: disable extensions in math.h" )
    add_compile_definitions ( "__NOEXTENSIONS__" )
endif ( )

#-------------------------------------------------------------------------------
# check if Fortran is available and enabled
#-------------------------------------------------------------------------------

include ( CheckLanguage )
option ( SUITESPARSE_USE_FORTRAN "ON (default): use Fortran. OFF: do not use Fortran" ON )
if ( SUITESPARSE_USE_FORTRAN )
    message ( STATUS "Checking if Fortran is available and compatible with C/C++" )
    check_language ( Fortran )
    if ( CMAKE_Fortran_COMPILER )
        # Fortran is available; ensure that it is compatible with C/C++
        enable_language ( CXX )
        enable_language ( Fortran )
        set ( SUITESPARSE_HAS_FORTRAN ON )
        if ( NOT "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "${CMAKE_C_COMPILER_ID}" OR
             NOT "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "${CMAKE_CXX_COMPILER_ID}" )
            message ( STATUS " " )
            message ( STATUS "Incompatible Fortran/C/C++ compilers detected:" )
            message ( STATUS "    Fortran:          ${CMAKE_Fortran_COMPILER}" )
            message ( STATUS "    Fortran id:       ${CMAKE_Fortran_COMPILER_ID}" )
            message ( STATUS "    C                 ${CMAKE_C_COMPILER}" )
            message ( STATUS "    C       id:       ${CMAKE_C_COMPILER_ID}" )
            message ( STATUS "    C++               ${CMAKE_CXX_COMPILER}" )
            message ( STATUS "    C++     id:       ${CMAKE_CXX_COMPILER_ID}" )
            if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "IntelLLVM" )
                # icx/icpx cannot be used with gfortran: this is a fatal error
                message ( FATAL_ERROR "ERROR: Using Fortran with SuiteSparse requires that "
                " it has the same compiler ID as the C/C++ compilers."
                "  Use a compatible Fortran compiler, or set SUITESPARSE_USE_FORTRAN to OFF." )
            else ( )
                # other cases: just issue a warning and hope it works.
                message ( WARNING "Warning: Using Fortran with SuiteSparse requires that "
                " it has the same compiler ID as the C/C++ compilers."
                "  Use a compatible Fortran compiler, or set SUITESPARSE_USE_FORTRAN to OFF." )
            endif ( )
        endif ( )
        message ( STATUS "Fortran:          enabled" )
    else ( )
        # Fortran not available:
        set ( SUITESPARSE_HAS_FORTRAN OFF )
        message ( STATUS "Fortran:          not available" )
    endif ( )
else ( )
    message ( STATUS "Fortran:          not enabled" )
    set ( SUITESPARSE_HAS_FORTRAN OFF )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND SUITESPARSE_USE_FORTRAN AND NOT SUITESPARSE_HAS_FORTRAN )
    message ( FATAL_ERROR "Fortran required for SuiteSparse but not found" )
endif ( )

# default C-to-Fortran name mangling if Fortran compiler not found
if ( MSVC )
    # MS Visual Studio Fortran compiler does not mangle the Fortran name
    set ( SUITESPARSE_C_TO_FORTRAN "(name,NAME) name"
        CACHE STRING "C to Fortan name mangling" )
else ( )
    # Other systems (Linux, Mac) typically append an underscore
    set ( SUITESPARSE_C_TO_FORTRAN "(name,NAME) name##_"
        CACHE STRING "C to Fortan name mangling" )
endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

option ( SUITESPARSE_USE_CUDA "ON (default): enable CUDA acceleration for SuiteSparse, OFF: do not use CUDA" ON )

if ( SUITESPARSE_USE_CUDA )

    # try finding CUDA
    check_language ( CUDA )
    # message ( STATUS "Looking for CUDA" )
    if ( CMAKE_CUDA_COMPILER )
        # with CUDA:
        # message ( STATUS "Find CUDA tool kit:" )
        # FindCUDAToolKit needs to have C or CXX enabled first (see above)
        find_package ( CUDAToolkit )
        message ( STATUS "CUDA toolkit :    " ${CUDAToolkit_FOUND} )
        message ( STATUS "CUDA toolkit ver: " ${CUDAToolkit_VERSION} )
        message ( STATUS "CUDA toolkit inc: " ${CUDAToolkit_INCLUDE_DIRS} )
        message ( STATUS "CUDA toolkit lib: " ${CUDAToolkit_LIBRARY_DIR} )
        if ( CUDAToolkit_VERSION VERSION_LESS "11.2" )
            # CUDA is present but too old
            message ( STATUS "CUDA:               not enabled (CUDA 11.2 or later required)" )
            set ( SUITESPARSE_HAS_CUDA OFF )
        else ( )
            # CUDA 11.2 or later present
            enable_language ( CUDA )
            set ( SUITESPARSE_HAS_CUDA ON )
        endif ( )
    else ( )
        # without CUDA:
        message ( STATUS "CUDA:             not found" )
        set ( SUITESPARSE_HAS_CUDA OFF )
    endif ( )

else ( )

    # CUDA is disabled
    set ( SUITESPARSE_HAS_CUDA OFF )

endif ( )

if ( SUITESPARSE_HAS_CUDA )
    message ( STATUS "CUDA:             enabled" )
    set ( SUITESPARSE_CUDA_ARCHITECTURES "52;75;80" CACHE STRING "CUDA architectures" )
    set ( CMAKE_CUDA_ARCHITECTURES ${SUITESPARSE_CUDA_ARCHITECTURES} )
else ( )
    message ( STATUS "CUDA:             not enabled" )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND SUITESPARSE_USE_CUDA AND NOT SUITESPARSE_HAS_CUDA )
    message ( FATAL_ERROR "CUDA required for SuiteSparse but not found" )
endif ( )

