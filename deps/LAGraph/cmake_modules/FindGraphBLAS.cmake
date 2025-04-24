#[=======================================================================[.rst:
FindGraphBLAS
--------

The following copyright and license applies to just this file only, not to
the GraphBLAS library itself:
LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
SPDX-License-Identifier: BSD-2-Clause
See additional acknowledgments in the LICENSE file,
or contact permission@sei.cmu.edu for the full terms.

Find the native GraphBLAS includes and library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``GraphBLAS::GraphBLAS``, if
GraphBLAS has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

::

  GRAPHBLAS_INCLUDE_DIR    - where to find GraphBLAS.h, etc.
  GRAPHBLAS_LIBRARY        - dynamic GraphBLAS library
  GRAPHBLAS_STATIC         - static GraphBLAS library
  GRAPHBLAS_LIBRARIES      - List of libraries when using GraphBLAS.
  GRAPHBLAS_FOUND          - True if GraphBLAS found.

::

Hints
^^^^^

A user may set ``GraphBLAS_ROOT`` to a GraphBLAS installation root to tell this
module where to look (for cmake 3.27, you may also use ``GRAPHBLAS_ROOT``).

First, a GraphBLASConfig.cmake file is searched in the LAGraph/build or
../GraphBLAS/build folder.  If that is not found, a search is made for
GraphBLASConfig.cmake in the standard places that CMake looks.
SuiteSparse::GraphBLAS v8.2.0 and following create a GraphBLASConfig.cmake
file, and other GraphBLAS libraries may do the same.

If the GraphBLASConfig.cmake file is not found, the GraphBLAS.h include file
and compiled GraphBLAS library are searched for in the places given by
GraphBLAS_ROOT, GRAPHBLAS_ROOT, LAGraph/.., LAGraph/../GraphBLAS, or
LAGraph/../SuiteSparse/GraphBLAS.  This will find SuiteSparse:GraphBLAS
versions 8.0.x and later, or it may find another GraphBLAS library that does
not provide a GraphBLASConfig.cmake file.

If SuiteSparse:GraphBLAS is used, all the *Config.cmake files in SuiteSparse
are installed by 'make install' into the default location when compiling
SuiteSparse.  CMake should know where to find them.

SuiteSparse:GraphBLAS also comes in many Linux distros, spack, brew, conda,
etc. Try:

    apt search libsuitesparse-dev
    spack info suite-sparse
    brew info suite-sparse
    conda search -c conda-forge graphblas

If GraphBLAS is not found, or if a different version is found than what was
expected, you can enable the LAGRAPH_DUMP option to display the places where
CMake looks.

#]=======================================================================]

option ( LAGRAPH_DUMP "ON: display list of places to search. OFF (default): no debug output" OFF )

# NB: this is built around assumptions about one particular GraphBLAS
# installation (SuiteSparse:GraphBLAS). As other installations become available
# changes to this will likely be required.

#-------------------------------------------------------------------------------
# find GraphBLASConfig.make in ../GraphBLAS (typically inside SuiteSparse):
#-------------------------------------------------------------------------------

if ( LAGRAPH_DUMP AND NOT GraphBLAS_DIR )
    message ( STATUS "(1) Looking for GraphBLASConfig.cmake in... :
    CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}
    PROJECT_SOURCE_DIR/../GraphBLAS/build: ${PROJECT_SOURCE_DIR}/../GraphBLAS/build" )
endif ( )

find_package ( GraphBLAS ${GraphBLAS_FIND_VERSION} CONFIG
    PATHS ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/../GraphBLAS/build
    NO_DEFAULT_PATH )

set ( _lagraph_gb_common_tree ON )

#-------------------------------------------------------------------------------
# if not yet found, look for GraphBLASConfig.cmake in the standard places
#-------------------------------------------------------------------------------

if ( LAGRAPH_DUMP AND NOT TARGET SuiteSparse::GraphBLAS )
message ( STATUS "(2) Looking for GraphBLASConfig.cmake in these places (in order):
    GraphBLAS_ROOT: ${GraphBLAS_ROOT}" )
if ( CMAKE_VERSION VERSION_GREATER_EQUAL "3.27" )
    message ( STATUS "
    GRAPHBLAS_ROOT: ${GRAPHBLAS_ROOT}" )
endif ( )
    message ( STATUS "
    ENV GraphBLAS_ROOT: $ENV{GraphBLAS_ROOT}" )
if ( CMAKE_VERSION VERSION_GREATER_EQUAL "3.27" )
    message ( STATUS "
    ENV GRAPHBLAS_ROOT: $ENV{GRAPHBLAS_ROOT}" )
endif ( )
message ( STATUS "
    CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}
    CMAKE_FRAMEWORK_PATH: ${CMAKE_FRAMEWORK_PATH}
    CMAKE_APPBUNDLE_PATH: ${CMAKE_APPBUNDLE_PATH}
    ENV GraphBLAS_DIR: $ENV{GraphBLAS_DIR}
    ENV CMAKE_PREFIX_PATH: $ENV{CMAKE_PREFIX_PATH}
    ENV CMAKE_FRAMEWORK_PATH: $ENV{CMAKE_FRAMEWORK_PATH}
    ENV CMAKE_APPBUNDLE_PATH: $ENV{CMAKE_APPBUNDLE_PATH}
    ENV PATH: $ENV{PATH}
    CMake user package registry: (see cmake documentation)
    CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}
    CMAKE_SYSTEM_FRAMEWORK_PATH: ${CMAKE_SYSTEM_FRAMEWORK_PATH}
    CMAKE_SYSTEM_APPBUNDLE_PATH: ${CMAKE_SYSTEM_APPBUNDLE_PATH}
    CMake system package registry: (see cmake documentation)"
    )
endif ( )

if ( NOT TARGET SuiteSparse::GraphBLAS )
    find_package ( GraphBLAS ${GraphBLAS_FIND_VERSION} CONFIG )
    set ( _lagraph_gb_common_tree OFF )
endif ( )

if ( GraphBLAS_FOUND )
    if ( TARGET SuiteSparse::GraphBLAS )
        # It's not possible to create an alias of an alias.
        get_property ( _graphblas_aliased TARGET SuiteSparse::GraphBLAS
            PROPERTY ALIASED_TARGET )
        if ( GRAPHBLAS_VERSION LESS "8.3.0" AND _lagraph_gb_common_tree )
            # workaround for incorrect INTERFACE_INCLUDE_DIRECTORIES of
            # SuiteSparse:GraphBLAS 8.2.x before installation
            # (did not have "/Include")
            get_property ( _inc TARGET SuiteSparse::GraphBLAS PROPERTY
                INTERFACE_INCLUDE_DIRECTORIES )
            if ( IS_DIRECTORY ${_inc}/Include )
                if ( "${_graphblas_aliased}" STREQUAL "" )
                    target_include_directories ( SuiteSparse::GraphBLAS INTERFACE
                        ${_inc}/Include )
                else ( )
                    target_include_directories ( ${_graphblas_aliased} INTERFACE
                        ${_inc}/Include )
                endif ( )
                message ( STATUS "additional include: ${_inc}/Include" )
            endif ( )
        endif ( )
        if ( "${_graphblas_aliased}" STREQUAL "" )
            add_library ( GraphBLAS::GraphBLAS ALIAS SuiteSparse::GraphBLAS )
        else ( )
            add_library ( GraphBLAS::GraphBLAS ALIAS ${_graphblas_aliased} )
        endif ( )
    endif ( )
    if ( TARGET SuiteSparse::GraphBLAS_static )
        # It's not possible to create an alias of an alias.
        get_property ( _graphblas_aliased TARGET SuiteSparse::GraphBLAS_static
            PROPERTY ALIASED_TARGET )
        if ( GRAPHBLAS_VERSION LESS "8.3.0" AND _lagraph_gb_common_tree )
            # workaround for incorrect INTERFACE_INCLUDE_DIRECTORIES of
            # SuiteSparse:GraphBLAS 8.2.x before installation
            # (did not have "/Include")
            get_property ( _inc TARGET SuiteSparse::GraphBLAS_static PROPERTY
                INTERFACE_INCLUDE_DIRECTORIES )
            if ( IS_DIRECTORY ${_inc}/Include )
                if ( "${_graphblas_aliased}" STREQUAL "" )
                    target_include_directories ( SuiteSparse::GraphBLAS_static INTERFACE
                        ${_inc}/Include )
                else ( )
                    target_include_directories ( ${_graphblas_aliased} INTERFACE
                        ${_inc}/Include )
                endif ( )
                message ( STATUS "additional include: ${_inc}/Include" )
            endif ( )
        endif ( )
        if ( "${_graphblas_aliased}" STREQUAL "" )
            add_library ( GraphBLAS::GraphBLAS_static ALIAS SuiteSparse::GraphBLAS_static )
        else ( )
            add_library ( GraphBLAS::GraphBLAS_static ALIAS ${_graphblas_aliased} )
        endif ( )
    endif ( )
    return ( )
endif ( )

#-------------------------------------------------------------------------------
# if still not found, look for GraphBLAS.h and compiled libraries directly
#-------------------------------------------------------------------------------

# Older versions of SuiteSparse GraphBLAS (8.0 or older) or GraphBLAS libraries
# not from SuiteSparse.

if ( LAGRAPH_DUMP )
    message ( STATUS "Looking for vanilla GraphBLAS (or older SuiteSparse),
    GraphBLAS.h and the compiled GraphBLAS library in:
    GraphBLAS_ROOT: ${GraphBLAS_ROOT}
    ENV GraphBLAS_ROOT $ENV{GraphBLAS_ROOT}
    GRAPHBLAS_ROOT: ${GRAPHBLAS_ROOT}
    ENV GRAPHBLAS_ROOT ENV${GRAPHBLAS_ROOT}
    PROJECT_SOURCE_DIR/..: ${PROJECT_SOURCE_DIR}/..
    PROJECT_SOURCE_DIR/../GraphBLAS: ${PROJECT_SOURCE_DIR}/../GraphBLAS
    PROJECT_SOURCE_DIR/../SuiteSparse/GraphBLAS: ${PROJECT_SOURCE_DIR}/../SuiteSparse/GraphBLAS"
    )
endif ( )

# "Include" for SuiteSparse:GraphBLAS
find_path ( GRAPHBLAS_INCLUDE_DIR
  NAMES GraphBLAS.h
  HINTS ${GraphBLAS_ROOT}
  HINTS ENV GraphBLAS_ROOT
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${PROJECT_SOURCE_DIR}/..
  HINTS ${PROJECT_SOURCE_DIR}/../GraphBLAS
  HINTS ${PROJECT_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES include Include
  NO_DEFAULT_PATH )

# dynamic SuiteSparse:GraphBLAS library
find_library ( GRAPHBLAS_LIBRARY
  NAMES graphblas
  HINTS ${GraphBLAS_ROOT}
  HINTS ENV GraphBLAS_ROOT
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${PROJECT_SOURCE_DIR}/..
  HINTS ${PROJECT_SOURCE_DIR}/../GraphBLAS
  HINTS ${PROJECT_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES lib build alternative
  NO_DEFAULT_PATH )

if ( MSVC )
    set ( STATIC_NAME graphblas_static )
else ( )
    set ( STATIC_NAME graphblas )
endif ( )

# static SuiteSparse:GraphBLAS library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
find_library ( GRAPHBLAS_STATIC
  NAMES ${STATIC_NAME}
  HINTS ${GraphBLAS_ROOT}
  HINTS ENV GraphBLAS_ROOT
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${PROJECT_SOURCE_DIR}/..
  HINTS ${PROJECT_SOURCE_DIR}/../GraphBLAS
  HINTS ${PROJECT_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES lib build alternative
  NO_DEFAULT_PATH )

set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
if ( MINGW AND GRAPHBLAS_STATIC MATCHES ".*\.dll\.a" )
    set ( GRAPHBLAS_STATIC "" )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( GRAPHBLAS_LIBRARY  ${GRAPHBLAS_LIBRARY} REALPATH )
get_filename_component ( GRAPHBLAS_FILENAME ${GRAPHBLAS_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    GRAPHBLAS_VERSION
    ${GRAPHBLAS_FILENAME}
  )

if ( GRAPHBLAS_VERSION )
    if ( ${GRAPHBLAS_VERSION} MATCHES "([0-9]+).[0-9]+.[0-9]+" )
        set ( GraphBLAS_VERSION_MAJOR ${CMAKE_MATCH_1} )
    endif ( )
    if ( ${GRAPHBLAS_VERSION} MATCHES "[0-9]+.([0-9]+).[0-9]+" )
        set ( GraphBLAS_VERSION_MINOR ${CMAKE_MATCH_1} )
    endif ( )
    if ( ${GRAPHBLAS_VERSION} MATCHES "[0-9]+.[0-9]+.([0-9]+)" )
        set ( GraphBLAS_VERSION_PATCH ${CMAKE_MATCH_1} )
    endif ( )
    if ( LAGRAPH_DUMP )
        message ( STATUS "major: ${GraphBLAS_VERSION_MAJOR}" )
        message ( STATUS "minor: ${GraphBLAS_VERSION_MINOR}" )
        message ( STATUS "patch: ${GraphBLAS_VERSION_PATCH}" )
    endif ( )
endif ( )

# set ( GRAPHBLAS_VERSION "" )
if ( EXISTS "${GRAPHBLAS_INCLUDE_DIR}" AND NOT GRAPHBLAS_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${GRAPHBLAS_INCLUDE_DIR}/GraphBLAS.h GRAPHBLAS_MAJOR_STR
        REGEX "define GxB_IMPLEMENTATION_MAJOR" )
    file ( STRINGS ${GRAPHBLAS_INCLUDE_DIR}/GraphBLAS.h GRAPHBLAS_MINOR_STR
        REGEX "define GxB_IMPLEMENTATION_MINOR" )
    file ( STRINGS ${GRAPHBLAS_INCLUDE_DIR}/GraphBLAS.h GRAPHBLAS_PATCH_STR
        REGEX "define GxB_IMPLEMENTATION_SUB" )
    if ( LAGRAPH_DUMP )
        message ( STATUS "major: ${GRAPHBLAS_MAJOR_STR}" )
        message ( STATUS "minor: ${GRAPHBLAS_MINOR_STR}" )
        message ( STATUS "patch: ${GRAPHBLAS_PATCH_STR}" )
    endif ( )
    string ( REGEX MATCH "[0-9]+" GraphBLAS_VERSION_MAJOR ${GRAPHBLAS_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" GraphBLAS_VERSION_MINOR ${GRAPHBLAS_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" GraphBLAS_VERSION_PATCH ${GRAPHBLAS_PATCH_STR} )
    set (GRAPHBLAS_VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_PATCH}")
endif ( )

set ( GRAPHBLAS_LIBRARIES ${GRAPHBLAS_LIBRARY} )

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args(
  GraphBLAS
  REQUIRED_VARS GRAPHBLAS_LIBRARIES GRAPHBLAS_INCLUDE_DIR
  VERSION_VAR GRAPHBLAS_VERSION
  )

mark_as_advanced(
  GRAPHBLAS_INCLUDE_DIR
  GRAPHBLAS_LIBRARY
  GRAPHBLAS_STATIC
  GRAPHBLAS_LIBRARIES
  )

if ( GRAPHBLAS_FOUND )
    message ( STATUS "GraphBLAS version: ${GRAPHBLAS_VERSION}" )
    message ( STATUS "GraphBLAS include: ${GRAPHBLAS_INCLUDE_DIR}" )
    message ( STATUS "GraphBLAS library: ${GRAPHBLAS_LIBRARY}" )
    message ( STATUS "GraphBLAS static:  ${GRAPHBLAS_STATIC}" )
else ( )
    message ( STATUS "GraphBLAS not found" )
    set ( GRAPHBLAS_INCLUDE_DIR "" )
    set ( GRAPHBLAS_LIBRARIES "" )
    set ( GRAPHBLAS_LIBRARY "" )
    set ( GRAPHBLAS_STATIC "" )
endif ( )

# Create target from information found

if ( GRAPHBLAS_LIBRARY )
    message ( STATUS "Create target GraphBLAS::GraphBLAS" )
    # Get library name from filename of library
    # This might be something like:
    #   /usr/lib/libgraphblas.so or /usr/lib/libgraphblas.a or graphblas64
    # convert to library name that can be used with -l flags for pkg-config
    set ( GRAPHBLAS_LIBRARY_TMP ${GRAPHBLAS_LIBRARY} )
    string ( FIND ${GRAPHBLAS_LIBRARY} "." _pos REVERSE )
    if ( ${_pos} EQUAL "-1" )
        set ( _graphblas_library_name ${GRAPHBLAS_LIBRARY} )
    else ( )
      set ( _kinds "SHARED" "STATIC" )
      if ( WIN32 )
          list ( PREPEND _kinds "IMPORT" )
      endif ( )
      foreach ( _kind IN LISTS _kinds )
          set ( _regex ".*\\/(lib)?([^\\.]*)(${CMAKE_${_kind}_LIBRARY_SUFFIX})" )
          if ( ${GRAPHBLAS_LIBRARY} MATCHES ${_regex} )
              string ( REGEX REPLACE ${_regex} "\\2" _libname ${GRAPHBLAS_LIBRARY} )
              if ( NOT "${_libname}" STREQUAL "" )
                  set ( _graphblas_library_name ${_libname} )
                  break ()
              endif ( )
          endif ( )
      endforeach ( )
    endif ( )

    add_library ( GraphBLAS::GraphBLAS UNKNOWN IMPORTED )
    set_target_properties ( GraphBLAS::GraphBLAS PROPERTIES
        IMPORTED_LOCATION "${GRAPHBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GRAPHBLAS_INCLUDE_DIR}"
        OUTPUT_NAME ${_graphblas_library_name} )
endif ( )

if ( GRAPHBLAS_STATIC )
    message ( STATUS "Create target GraphBLAS::GraphBLAS_static" )
    # Get library name from filename of library
    # This might be something like:
    #   /usr/lib/libgraphblas.so or /usr/lib/libgraphblas.a or graphblas64
    # convert to library name that can be used with -l flags for pkg-config
    set ( GRAPHBLAS_LIBRARY_TMP ${GRAPHBLAS_STATIC} )
    string ( FIND ${GRAPHBLAS_STATIC} "." _pos REVERSE )
    if ( ${_pos} EQUAL "-1" )
        set ( _graphblas_library_name ${GRAPHBLAS_STATIC} )
    else ( )
      set ( _kinds "SHARED" "STATIC" )
      if ( WIN32 )
          list ( PREPEND _kinds "IMPORT" )
      endif ( )
      foreach ( _kind IN LISTS _kinds )
          set ( _regex ".*\\/(lib)?([^\\.]*)(${CMAKE_${_kind}_LIBRARY_SUFFIX})" )
          if ( ${GRAPHBLAS_STATIC} MATCHES ${_regex} )
              string ( REGEX REPLACE ${_regex} "\\2" _libname ${GRAPHBLAS_STATIC} )
              if ( NOT "${_libname}" STREQUAL "" )
                  set ( _graphblas_library_name ${_libname} )
                  break ()
              endif ( )
          endif ( )
      endforeach ( )
    endif ( )

    add_library ( GraphBLAS::GraphBLAS_static UNKNOWN IMPORTED )
    set_target_properties ( GraphBLAS::GraphBLAS_static PROPERTIES
        IMPORTED_LOCATION "${GRAPHBLAS_STATIC}"
        INTERFACE_INCLUDE_DIRECTORIES "${GRAPHBLAS_INCLUDE_DIR}"
        OUTPUT_NAME ${_graphblas_library_name} )
endif ( )
