#-------------------------------------------------------------------------------
# GraphBLAS/CUDA/CMakeLists.txt:  cmake script for GraphBLAS/CUDA
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.

# Some files in this folder are (c) NVIDIA or (c) Google.  Please refer
# to their individual licenses (Apache, BSD, or others).
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

# CMake build for generating googletest c++ files that can be compiled and
# executed in parallel.  Build can be customized to speed up development by
# allowing the targeting of specific specific parameters. The output of this
# build is an executable that can be used to run the gtests.

project ( GRAPHBLAS_CUDA
    VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}"
    LANGUAGES CXX CUDA )

cmake_policy ( SET CMP0135 NEW )    # URL download timestamp policy

set ( CMAKE_CXX_STANDARD 17 )

set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo " )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC " )

add_compile_definitions ( GBNCPUFEAT )
add_compile_definitions ( GBCUDA_CPLUSPLUS )

message ( STATUS "C++ flags for CUDA: ${CMAKE_CXX_FLAGS}" )

file ( GLOB GRAPHBLAS_CUDA_SOURCES "*.cu" "*.c" "*.cpp" )

add_library ( GraphBLAS_CUDA SHARED ${GRAPHBLAS_CUDA_SOURCES} )

set_target_properties ( GraphBLAS_CUDA PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    OUTPUT_NAME graphblascuda
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON )

# find rmm_wrap, the malloc/calloc/realloc/free wrapper for the Rapids
# memory manager
set ( RMM_WRAP_INCLUDES "../rmm_wrap" )

message ( STATUS "RMM_WRAP_INCLUDES: ${RMM_WRAP_INCLUDES}" )
set ( GRAPHBLAS_CUDA_INCLUDES
        ${RMM_WRAP_INCLUDES}
        ../Source
        ../Source/Shared
        ../Source/Template
        ../Source/Factories
        ../Include
        ../CUDA )

message ( STATUS "GraphBLAS CUDA includes: ${GRAPHBLAS_CUDA_INCLUDES}" )

#-------------------------------------------------------------------------------
# GraphBLAS_CUDA properties
#-------------------------------------------------------------------------------

target_include_directories(GraphBLAS_CUDA PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GRAPHBLAS_CUDA_INCLUDES})
set_target_properties(GraphBLAS_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(GraphBLAS_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# FIXME: use SUITESPARSE_CUDA_ARCHITECTURES
set_target_properties(GraphBLAS_CUDA PROPERTIES CUDA_ARCHITECTURES "52;75;80" )

target_link_libraries(GraphBLAS_CUDA CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver CUDA::nvToolsExt )

target_compile_definitions ( GraphBLAS_CUDA PUBLIC "SUITESPARSE_CUDA" )

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
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GraphBLAS )

configure_package_config_file (
    Config/GraphBLAS_CUDAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GraphBLAS )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS_CUDAConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GraphBLAS )

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
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# test suite for the CUDA kernels
#-------------------------------------------------------------------------------

# 1. Execute enumify/stringify/jitify logic to compile ptx kernels and
# compile/link w/ relevant *.cu files.

# TODO: Need to do this piece in cmake

# 2. Generate test .cu files named "{semiring_operation}_test_instances.hpp"
set ( CUDA_TEST_SUITES
    AxB_dot3
#    reduce_to_scalar
)

#
set ( CUDA_TEST_MONOIDS PLUS MIN MAX) # TIMES ANY )
set ( CUDA_TEST_BINOPS TIMES PLUS MIN MAX DIV ) #MINUS RDIV RMINUS FIRST SECOND PAIR )
set ( CUDA_TEST_SEMIRINGS PLUS_TIMES MIN_PLUS MAX_PLUS )
set ( CUDA_TEST_DATATYPES int32_t int64_t uint32_t uint64_t float double )
set ( CUDA_TEST_KERNELS vsvs) # mp vsvs dndn spdn vssp )
set ( CUDA_TEST_FORMATS sparse dense sparse_dense reduce )

# TODO: Update testGen.py to accept the above CUDA_TEST_* params as arguments

# Note: I don't believe there's a way to do this particular piece in parallel but
# once all the files are written, we should be able to compile them in parallel

# Separate individual kernels from larger "overview" test (e.g. 2-level testing structure)
# We want to test all the *_cuda versions

# TODO: make this a shorter test
set(CUDA_TEST_CPP_FILES "")
if ( FALSE ) # TODO: use a cmake option
    foreach(var ${CUDA_TEST_SUITES})
        foreach(semiring ${CUDA_TEST_SEMIRINGS})
            foreach(kernel ${CUDA_TEST_KERNELS})
                foreach(format ${CUDA_TEST_FORMATS})
                    # TODO: Have Python script also build separate cudaTest.cpp (named something
                    # like AxB_dot3_cuda_tests.cpp) for each suite. This way we should be able to
                    # easily ignore them from the build
                    add_custom_command(
                            OUTPUT
                            ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${format}_test_instances.hpp
                            ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${format}_cuda_tests.cpp
#                            DEPENDS
#                            jitFactory.hpp
                            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/testGen_cmake.py "\"${CMAKE_CURRENT_SOURCE_DIR}\"" "\"${var}\"" "\"${CUDA_TEST_MONOIDS}\""
                                "\"${CUDA_TEST_BINOPS}\"" "\"${semiring}\"" "\"${CUDA_TEST_DATATYPES}\""
                                "\"${kernel}\""
                    )
                    # Construct final list of files to compile (in parallel)
                    list(APPEND CUDA_TEST_CPP_FILES ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${format}_cuda_tests.cpp)
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endif ( )

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

#FetchContent_MakeAvailable(googletest EC)


#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes)
#execute_process(
#        COMMAND git clone "https://github.com/google/googletest.git" googletest
#        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes)
#
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/external_includes/googletest/googletest/include)

#add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/external_includes/googletest/googletest/)

# 3. Compile/link individual {test_suite_name}_cuda_tests.cpp files into a gtest executable
set(GRAPHBLAS_CUDA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/test)

message(STATUS "CUDA tests files: " "${CUDA_TEST_CPP_FILES}")

add_executable(graphblascuda_test ${CUDA_TEST_CPP_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.cpp)

set_target_properties(graphblascuda_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(graphblascuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(graphblascuda_test PROPERTIES CUDA_ARCHITECTURES "52;75;80" )

include(GoogleTest)

add_dependencies(graphblascuda_test GraphBLAS)
add_dependencies(graphblascuda_test GraphBLAS_CUDA)
add_dependencies(graphblascuda_test gtest_main)
add_dependencies(graphblascuda_test rmm_wrap)

target_link_libraries(graphblascuda_test
        PUBLIC
        GraphBLAS
        GraphBLAS_CUDA
        rmm_wrap
        CUDA::cudart_static
        CUDA::nvrtc
        ${ADDITIONAL_DEPS}
        PRIVATE
        gtest_main)

target_include_directories(graphblascuda_test
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../Include
        ${CMAKE_CURRENT_SOURCE_DIR}/../rmm_wrap
        rmm_wrap
        ${ADDITIONAL_INCLUDES}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GRAPHBLAS_CUDA_INCLUDES})

