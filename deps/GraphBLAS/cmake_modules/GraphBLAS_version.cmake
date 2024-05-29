#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLAS_version.cmake: define the GraphBLAS version
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# version of SuiteSparse:GraphBLAS
set ( GraphBLAS_DATE "May 22, 2024" )
set ( GraphBLAS_VERSION_MAJOR 9 CACHE STRING "" FORCE )
set ( GraphBLAS_VERSION_MINOR 2 CACHE STRING "" FORCE )
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

