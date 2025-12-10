#-------------------------------------------------------------------------------
# rmm_wrap/CMakeLists.txt: cmake script for building rmm_wrap
#-------------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone

project ( rmm_wrap VERSION 0.1 )

# This build depends upon having RMM cmake installed from https://github.com/rapidsai/rmm.git

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_C_STANDARD 99 )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )

find_package ( CUDAToolkit REQUIRED )
message ( STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS} " )

set ( EXTERNAL_INCLUDES_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes )

if ( NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY} )
    file ( MAKE_DIRECTORY ${EXTERNAL_INCLUDES_DIRECTORY} )
endif ( )

if ( NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY}/spdlog )
    message ( STATUS "cloning spdlog v1.10.0" )
    execute_process (
        COMMAND git clone "https://github.com/gabime/spdlog" --branch v1.10.0 --recursive spdlog
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes )
endif ( )

set ( SPDLOG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external_includes/spdlog/include )
include_directories ( ${SPDLOG_INCLUDE_DIR} )

if ( NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY}/rmm )
    message ( STATUS "cloning rmm branch-21.10" )
    execute_process (
        COMMAND git clone "https://github.com/rapidsai/rmm" --branch branch-21.10 --recursive rmm
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes )
endif()

set ( RMM_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external_includes/rmm/include )
include_directories ( ${RMM_INCLUDE_DIR} )

add_library ( RMM_wrap rmm_wrap.cpp rmm_wrap.hpp rmm_wrap.h )

set_target_properties ( RMM_wrap PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    OUTPUT_NAME rmm_wrap
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    PUBLIC_HEADER "rmm_wrap.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

add_executable ( rmm_wrap_test rmm_wrap_test.c rmm_wrap.cpp rmm_wrap.hpp rmm_wrap.h )

set ( RMM_WRAP_LIBS ${EXTRA_LIBS} CUDA::cudart_static )
set ( RMM_WRAP_INCLUDES ${PROJECT_BINARY_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${RMM_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CONDA_PREFIX}/include
    ${PROJECT_SOURCE_DIR}/include )

target_link_libraries ( RMM_wrap PRIVATE ${RMM_WRAP_LIBS} )
target_include_directories ( RMM_wrap PRIVATE "${RMM_WRAP_INCLUDES}" )

target_link_libraries ( rmm_wrap_test PUBLIC ${RMM_WRAP_LIBS} )
target_include_directories ( rmm_wrap_test PUBLIC "${RMM_WRAP_INCLUDES}" )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

install ( TARGETS RMM_wrap
    EXPORT GraphBLASTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

