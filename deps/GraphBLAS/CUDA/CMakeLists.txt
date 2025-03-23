#-------------------------------------------------------------------------------
# GraphBLAS/CUDA/CMakeLists.txt:  cmake script for GraphBLAS/CUDA
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

# Some files in this folder are (c) NVIDIA or (c) Google.  Please refer
# to their individual licenses (Apache, BSD, or others).
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone

project ( GRAPHBLAS_CUDA
    VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}"
    LANGUAGES CXX CUDA )

if ( CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 )
    # requires cmake 3.24:
    cmake_policy ( SET CMP0135 NEW )    # URL download timestamp policy
endif ( )

set ( CMAKE_CXX_STANDARD 17 )

set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo " )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC " )

add_compile_definitions ( GBNCPUFEAT )

message ( STATUS "C++ flags for CUDA: ${CMAKE_CXX_FLAGS}" )

file ( GLOB GRAPHBLAS_CUDA_SOURCES "*.cu" "*.c" "*.cpp" )

add_library ( GraphBLAS_CUDA SHARED ${GRAPHBLAS_CUDA_SOURCES} )

set_target_properties ( GraphBLAS_CUDA PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    OUTPUT_NAME graphblascuda
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON )

#-------------------------------------------------------------------------------
# GraphBLAS_CUDA properties
#-------------------------------------------------------------------------------

target_include_directories ( GraphBLAS_CUDA PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/../
        include
        ../Include
        ../rmm_wrap
        ../Source
        ../Source/hyper
        ../Source/builtin
        )

set_target_properties ( GraphBLAS_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( GraphBLAS_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
# FIXME: use SUITESPARSE_CUDA_ARCHITECTURES
set_target_properties ( GraphBLAS_CUDA PROPERTIES CUDA_ARCHITECTURES "52;75;80" )

target_link_libraries ( GraphBLAS_CUDA PRIVATE CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver )

if ( TARGET CUDA::nvToolsExt )
    target_link_libraries ( GraphBLAS_CUDA PRIVATE CUDA::nvToolsExt )
endif ( )

if ( TARGET CUDA::nvtx3 )
    target_link_libraries ( GraphBLAS_CUDA PRIVATE CUDA::nvtx3 )
    target_compile_definitions ( GraphBLAS_CUDA PRIVATE GBNVTX )
endif ( )

target_compile_definitions ( GraphBLAS_CUDA PUBLIC "GRAPHBLAS_HAS_CUDA" )

if ( OpenMP_CXX_FOUND )
    target_include_directories ( GraphBLAS_CUDA PRIVATE OpenMP::OpenMP_CXX )
endif ( )

target_include_directories ( GraphBLAS_CUDA
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS GraphBLAS_CUDA
    EXPORT GraphBLAS_CUDATargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

# create (temporary) export target file during build
export ( EXPORT GraphBLAS_CUDATargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDATargets.cmake )

# install export target and config for find_package
install ( EXPORT GraphBLAS_CUDATargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )

configure_package_config_file (
    Config/GraphBLAS_CUDAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
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
    configure_file (
        Config/GraphBLAS_CUDA.pc.in
        GraphBLAS_CUDA.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDA.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# report
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES} ")
message ( STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER} ")
message ( STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS} ")
message ( STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE} ")
message ( STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG} ")


