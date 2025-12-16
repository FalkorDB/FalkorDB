#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLAS_version.cmake: define the GraphBLAS version
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# version of SuiteSparse:GraphBLAS
set ( GraphBLAS_DATE "Dec 3, 2025" )
set ( GraphBLAS_VERSION_MAJOR 10 CACHE STRING "" FORCE )
set ( GraphBLAS_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( GraphBLAS_VERSION_SUB   0 CACHE STRING "" FORCE )

# GraphBLAS C API Specification version, at graphblas.org
set ( GraphBLAS_API_DATE "Dec 22, 2023" )
set ( GraphBLAS_API_VERSION_MAJOR 2 )
set ( GraphBLAS_API_VERSION_MINOR 1 )
set ( GraphBLAS_API_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse:GraphBLAS version: v"
    ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    ", date: " ${GraphBLAS_DATE} )

message ( STATUS "GraphBLAS C API: v"
    ${GraphBLAS_API_VERSION_MAJOR}.${GraphBLAS_API_VERSION_MINOR}
    ", date: ${GraphBLAS_API_DATE}" )

# Notes from Sebastien Villemot (sebastien@debian.org):
# SOVERSION policy: if a binary compiled against the old version of the shared
# library needs recompiling in order to work with the new version, then a
# SO_VERSION increase # is needed. Otherwise not.  Examples of the changes that
# require a SO_VERSION increase:
#
#   - a public function or static variable is removed
#   - the prototype of a public function changes
#   - the integer value attached to a public #define or enum changes
#   - the fields of a public structure are modified
#
# Examples of changes that do not require a SO_VERSION increase:
#
#   - a new public function or static variable is added
#   - a private function or static variable is removed or modified
#   - changes in the internals of a structure that is opaque to the calling
#       program (i.e. is only a pointer manipulated through public functions of
#       the library)
#   - a public enum is extended (by adding a new item at the end, but without
#       changing the already existing items)
