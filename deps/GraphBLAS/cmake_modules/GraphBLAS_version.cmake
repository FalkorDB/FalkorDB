#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLAS_version.cmake: define the GraphBLAS version
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# version of SuiteSparse:GraphBLAS
set ( GraphBLAS_DATE "Oct 7, 2023" )
set ( GraphBLAS_VERSION_MAJOR 9 )
set ( GraphBLAS_VERSION_MINOR 0 )
set ( GraphBLAS_VERSION_SUB   0 )

# GraphBLAS C API Specification version, at graphblas.org
set ( GraphBLAS_API_DATE "Oct 7, 2023" )
set ( GraphBLAS_API_VERSION_MAJOR 2 )
set ( GraphBLAS_API_VERSION_MINOR 1 )
set ( GraphBLAS_API_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse:GraphBLAS version: v"
    ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    ", date: " ${GraphBLAS_DATE} )

message ( STATUS "GraphBLAS C API: v"
    ${GraphBLAS_API_VERSION_MAJOR}.${GraphBLAS_API_VERSION_MINOR}
    ", date: ${GraphBLAS_API_DATE}" )

