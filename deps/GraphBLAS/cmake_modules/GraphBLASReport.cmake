#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLASReport.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2012-2025, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------
# report status and compile flags
#-------------------------------------------------------------------------------

message ( STATUS "------------------------------------------------------------------------" )
message ( STATUS "CMAKE report for: ${PROJECT_NAME}" )
message ( STATUS "------------------------------------------------------------------------" )
if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    message ( STATUS "inside common SuiteSparse root:  ${INSIDE_SUITESPARSE}" )
    message ( STATUS "install in SuiteSparse/lib and SuiteSparse/include: ${SUITESPARSE_LOCAL_INSTALL}" )
endif ( )
message ( STATUS "build type:           ${CMAKE_BUILD_TYPE}" )
message ( STATUS "BUILD_SHARED_LIBS:    ${BUILD_SHARED_LIBS}" )
message ( STATUS "BUILD_STATIC_LIBS:    ${BUILD_STATIC_LIBS}" )
message ( STATUS "C compiler:           ${CMAKE_C_COMPILER} ")
message ( STATUS "C flags:              ${CMAKE_C_FLAGS}" )
if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "C Flags debug:        ${CMAKE_C_FLAGS_DEBUG} ")
else ( )
    message ( STATUS "C Flags release:      ${CMAKE_C_FLAGS_RELEASE} ")
endif ( )
get_property ( CDEFN DIRECTORY PROPERTY COMPILE_DEFINITIONS )
message ( STATUS "compile definitions:  ${CDEFN}")
message ( STATUS "GraphBLAS has CUDA:   ${GRAPHBLAS_HAS_CUDA}")

if ( GRAPHBLAS_HAS_CUDA )
    message ( STATUS "CUDA compiler:        ${CMAKE_CUDA_COMPILER} ")
    message ( STATUS "CUDA flags:           ${CMAKE_CUDA_FLAGS} ")
    message ( STATUS "CUDA release:         ${CMAKE_CUDA_FLAGS_RELEASE} ")
    message ( STATUS "CUDA flags debug:     ${CMAKE_CUDA_FLAGS_DEBUG} ")
endif ( )

message ( STATUS "------------------------------------------------------------------------" )
