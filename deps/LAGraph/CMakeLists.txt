#-------------------------------------------------------------------------------
# LAGraph/CMakeLists.txt:  cmake script for LAGraph
#-------------------------------------------------------------------------------

# LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM22-0790

#-------------------------------------------------------------------------------

# CMakeLists.txt: instructions for cmake to build LAGraph.  An ANSI C11
# compiler is required.  First, install any GraphBLAS library.  Alternatively,
# use ../GraphBLAS (see comments below).
#
# To compile the LAGraph library and its tests and benchmarks, and run the
# tests:
#
#   cd build
#   cmake ..
#   make
#   make test
#
# If that fails for any reason, make sure your compiler supports ANSI C11.  You
# could try changing your compiler for the cmake command, for example:
#
#   CC=icc cmake ..
#   CC=xlc cmake ..
#   CC=gcc cmake ..

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # LAGraph can be built stand-alone

# version of LAGraph
set ( LAGraph_DATE "Feb 20, 2025" )
set ( LAGraph_VERSION_MAJOR 1 CACHE STRING "" FORCE )
set ( LAGraph_VERSION_MINOR 1 CACHE STRING "" FORCE )
set ( LAGraph_VERSION_SUB   5 CACHE STRING "" FORCE )

message ( STATUS "Building LAGraph version: v"
    ${LAGraph_VERSION_MAJOR}.
    ${LAGraph_VERSION_MINOR}.
    ${LAGraph_VERSION_SUB} " (" ${LAGraph_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( LAGraph
          VERSION "${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${PROJECT_SOURCE_DIR}/../cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

enable_language ( C )

# configure LAGraph.h with the LAGraph date and version
configure_file (
    "config/LAGraph.h.in"
    "${PROJECT_SOURCE_DIR}/include/LAGraph.h"
    NEWLINE_STYLE LF )

include ( CheckSymbolExists )
check_symbol_exists ( fmax "math.h" NO_LIBM )

#-------------------------------------------------------------------------------
# code coverage and build type
#-------------------------------------------------------------------------------

# To compile with test coverage:
#   cd build
#   cmake -DCOVERAGE=1 ..
#   make -j8
#   make test_coverage

if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if ( COVERAGE )
        message ( STATUS "============== Code coverage enabled ===============" )
        set ( CMAKE_BUILD_TYPE Debug )
        # On the Mac, you need gcov-11 from homebrew (part of gcc-11):
        # and uncomment this line:
        # set ( GCOV_PATH /usr/local/bin/gcov-11)
        include ( CodeCoverage )

        append_coverage_compiler_flags ( )

        # turn off optimization for non-skewed coverage reports
        set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -DCOVERAGE" )

        setup_target_for_coverage_lcov (
            NAME test_coverage
            EXECUTABLE ctest
            DEPENDENCIES ${PROJECT_NAME}
            BASE_DIRECTORY "."
            NO_DEMANGLE TRUE
            EXCLUDE "*/benchmark/*" "deps/json*/*" "src/test/include/acutest.h"
            )
    endif ( )
endif ( )

# For development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Find the GraphBLAS installation
#-------------------------------------------------------------------------------

if ( SUITESPARSE_ROOT_CMAKELISTS )

    if ( TARGET GraphBLAS )
        add_library ( GraphBLAS::GraphBLAS ALIAS GraphBLAS )
    else ( )
        add_library ( GraphBLAS::GraphBLAS ALIAS GraphBLAS_static )
    endif ( )
    if ( TARGET GraphBLAS_static )
        add_library ( GraphBLAS::GraphBLAS_static ALIAS GraphBLAS_static )
    endif ( )

else ( )

    # If GraphBLAS is not in a standard installation location, either
    #     export GRAPHBLAS_ROOT <path>
    # or
    #     GRAPHBLAS_ROOT=<path> cmake ..
    # or uncomment the next line:
    # set ( ENV{GRAPHBLAS_ROOT} ${PROJECT_SOURCE_DIR}/../GraphBLAS )

    # The ../GraphBLAS folder is considered by default, if it exists.

#   message ( STATUS "GraphBLAS_ROOT: ${GraphBLAS_ROOT} $ENV{GraphBLAS_ROOT}" )
#   message ( STATUS "GRAPHBLAS_ROOT: ${GRAPHBLAS_ROOT} $ENV{GRAPHBLAS_ROOT}" )

    # No package version is explicitly stated here; an arbitrary GraphBLAS
    # library can have any version number.  For SuiteSparse:GraphBLAS, LAGraph
    # requires v7.1.0 or later, which is checked in LAGraph.h.
    find_package ( GraphBLAS MODULE REQUIRED )

endif ( )

#-------------------------------------------------------------------------------
# determine what user threading model to use
#-------------------------------------------------------------------------------

option ( LAGRAPH_USE_OPENMP "ON: Use OpenMP in LAGraph if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( COVERAGE )
    set ( LAGRAPH_USE_OPENMP "OFF" CACHE STRING "" FORCE ) # OK: test coverage is enabled
    message ( STATUS "OpenMP disabled for test coverage" )
else ( )
    if ( LAGRAPH_USE_OPENMP )
        if ( CMAKE_VERSION VERSION_LESS 3.24 )
            find_package ( OpenMP COMPONENTS C )
        else ( )
            find_package ( OpenMP COMPONENTS C GLOBAL )
        endif ( )
        if ( OpenMP_C_FOUND AND BUILD_STATIC_LIBS )
            list ( APPEND LAGRAPH_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
        endif ( )
        find_package ( Threads )
    else ( )
        set ( OpenMP_C_FOUND OFF )
    endif ( )
endif ( )

if ( LAGRAPH_USE_OPENMP AND OpenMP_C_FOUND )
    set ( LAGRAPH_HAS_OPENMP ON )
else ( )
    set ( LAGRAPH_HAS_OPENMP OFF )
endif ( )
message ( STATUS "LAGraph has OpenMP: ${LAGRAPH_HAS_OPENMP}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND LAGRAPH_USE_OPENMP AND NOT LAGRAPH_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP required for LAGraph but not found" )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE build type:          " ${CMAKE_BUILD_TYPE} )
message ( STATUS "CMAKE source directory:    " ${PROJECT_SOURCE_DIR} )
message ( STATUS "CMAKE build directory:     " ${PROJECT_BINARY_DIR} )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message ( STATUS "CMAKE C Flags debug:       " ${CMAKE_C_FLAGS_DEBUG} )
else ( )
    message ( STATUS "CMAKE C Flags release:     " ${CMAKE_C_FLAGS_RELEASE} )
endif ( )

message ( STATUS "CMAKE compiler ID:         " ${CMAKE_C_COMPILER_ID} )
message ( STATUS "CMAKE thread library:      " ${CMAKE_THREAD_LIBS_INIT} )
message ( STATUS "CMAKE have pthreads:       " ${CMAKE_USE_PTHREADS_INIT}  )
message ( STATUS "CMAKE have Win32 pthreads: " ${CMAKE_USE_WIN32_THREADS_INIT} )
message ( STATUS "CMAKE have OpenMP:         " ${OPENMP_C_FOUND} )

#-------------------------------------------------------------------------------
# include directories for LAGraph library
#-------------------------------------------------------------------------------

include_directories ( ${PROJECT_SOURCE_DIR}/include
                      ${PROJECT_SOURCE_DIR}/src/utility
                      ${PROJECT_SOURCE_DIR}/test/include )

# tell LAGraph where to find its own source (for LAGraph/data files)
set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLGDIR=${PROJECT_SOURCE_DIR}" )
# set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O" )

#-------------------------------------------------------------------------------
# compiler options
#-------------------------------------------------------------------------------

# check which compiler is being used.  If you need to make
# compiler-specific modifications, here is the place to do it.
if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # check all warnings:
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Werror " )
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9 )
    #    message ( FATAL_ERROR "gcc version must be at least 4.9" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Intel" )
    # note that -g can be used, for VTune.  Comment out the following line
    # to compile without -g.
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g " )
    # check all warnings:
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -w3 " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 18.0 )
    #    message ( FATAL_ERROR "icc version must be at least 18.0" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" AND NOT "${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC" )
    # options for clang
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3 " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 3.3 )
    #    message ( FATAL_ERROR "clang version must be at least 3.3" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC" )
    # options for MicroSoft Visual Studio
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "PGI" )
    # options for PGI pgcc compiler
    set ( CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -acc -Minfo=accel -Mcuda -Mnoopenmp -noswitcherror -fPIC " )
endif ( )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}" )
endif ( )

#-------------------------------------------------------------------------------
# force use of vanilla code
#-------------------------------------------------------------------------------

if ( LAGRAPH_VANILLA )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DLAGRAPH_VANILLA=1 " )
    message ( STATUS "Vanilla build: not relying on any GxB extensions" )
else ( )
    message ( STATUS "GxB build: relying on SuiteSparse GxB extensions" )
endif ( )

#-------------------------------------------------------------------------------
# print final C flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE C flags: " ${CMAKE_C_FLAGS} )

#-------------------------------------------------------------------------------
# enable testing and add subdirectories
#-------------------------------------------------------------------------------

include ( CTest )

add_subdirectory ( src )
add_subdirectory ( experimental )
if ( DEFINED ENV{READTHEDOCS} )
    add_subdirectory ( rtdocs )
endif ( )

#-------------------------------------------------------------------------------
# LAGraph installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS LAGraph LAGraphX
        EXPORT LAGraphTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

if ( BUILD_STATIC_LIBS )
    install ( TARGETS LAGraph_static LAGraphX_static
        EXPORT LAGraphTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )


# create (temporary) export target file during build
export ( EXPORT LAGraphTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/LAGraphTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT LAGraphTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/LAGraph )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    config/LAGraphConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LAGraphConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/LAGraphConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    config/LAGraphConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/LAGraphConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/LAGraph )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/LAGraphConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/LAGraphConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LAGraphConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindGraphBLAS.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/LAGraph )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    if ( BUILD_STATIC_LIBS )
        if ( NOT NO_LIBM )
            list ( APPEND LAGRAPH_STATIC_LIBS "m" )
        endif ( )
    endif ( )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( LAGRAPH_STATIC_LIBS_LIST ${LAGRAPH_STATIC_LIBS} )
    set ( LAGRAPH_STATIC_LIBS "" )
    foreach ( _lib ${LAGRAPH_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( LAGRAPH_STATIC_LIBS "${LAGRAPH_STATIC_LIBS} -l${_lib}" )
            continue ()
        endif ( )
        set ( _kinds "SHARED" "STATIC" )
        if ( WIN32 )
            list ( PREPEND _kinds "IMPORT" )
        endif ( )
        foreach ( _kind IN LISTS _kinds )
            set ( _regex ".*\\/(lib)?([^\\.]*)(${CMAKE_${_kind}_LIBRARY_SUFFIX})" )
            if ( ${_lib} MATCHES ${_regex} )
                string ( REGEX REPLACE ${_regex} "\\2" _libname ${_lib} )
                if ( NOT "${_libname}" STREQUAL "" )
                    set ( LAGRAPH_STATIC_LIBS "${LAGRAPH_STATIC_LIBS} -l${_libname}" )
                    break ()
                endif ( )
            endif ( )
        endforeach ( )
    endforeach ( )
    if ( BUILD_STATIC_LIBS )
        set ( LAGRAPH_STATIC_LIBS "-l$<TARGET_FILE_BASE_NAME:GraphBLAS::GraphBLAS> ${LAGRAPH_STATIC_LIBS}" )
    endif ( )

    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${SUITESPARSE_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${SUITESPARSE_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${SUITESPARSE_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${SUITESPARSE_INCLUDEDIR}")
    endif ( )
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:LAGraph> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:LAGraph_static> )
    endif ( )
    configure_file (
        config/LAGraph.pc.in
        LAGraph.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT LAGraph.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/LAGraph.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/LAGraph.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )
