#-------------------------------------------------------------------------------
# GraphBLAS/CMakeLists.txt:  cmake script for GraphBLAS
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# See the User Guide for details on how to compile SuiteSparse:GraphBLAS.

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( GraphBLAS LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules )

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

include ( GraphBLAS_version )

set ( PROJECT_VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

# GraphBLAS takes a long time to build, so do not build the static library
# by default
if ( NOT BUILD_SHARED_LIBS )
    set ( BUILD_STATIC_LIBS_DEFAULT_OFF ON )
endif ( )

# CUDA is under development for now, and not deployed in production:
  set ( GRAPHBLAS_USE_CUDA OFF )    # turn CUDA off in production
# set ( GRAPHBLAS_USE_CUDA ON )     # use this for CUDA development only

include ( SuiteSparsePolicy )

if ( NOT GRAPHBLAS_BUILD_STATIC_LIBS AND BUILD_SHARED_LIBS )
    # ignore current value of BUILD_STATIC_LIBS and set it to OFF
    set ( BUILD_STATIC_LIBS OFF )
endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

# in the future, when GraphBLAS can use CUDA in production:
# option ( GRAPHBLAS_USE_CUDA "ON (default): enable CUDA acceleration for GraphBLAS, OFF: do not use CUDA" ${SUITESPARSE_USE_CUDA} )

if ( SUITESPARSE_HAS_CUDA AND GRAPHBLAS_USE_CUDA )
    # FOR NOW: do not compile FactoryKernels when developing the CUDA kernels
    set ( GRAPHBLAS_COMPACT ON )
    message ( STATUS "GraphBLAS CUDA JIT: enabled")
    set ( CMAKE_CXX_STANDARD 17 )
    enable_language ( CXX CUDA )
    set ( GRAPHBLAS_HAS_CUDA ON )
else ( )
    message ( STATUS "GraphBLAS CUDA JIT: disabled")
    set ( GRAPHBLAS_HAS_CUDA OFF )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND GRAPHBLAS_USE_CUDA AND NOT GRAPHBLAS_HAS_CUDA )
    message ( FATAL_ERROR "CUDA required for GraphBLAS but not found" )
endif ( )

include ( GraphBLAS_JIT_paths )

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_HAS_CUDA )

    # with CUDA, and supporting libraries: Rapids Memory Manager (rmm), and spdlog
    cmake_policy ( SET CMP0135 NEW )    # URL download timestamp policy
    find_package ( CUDAToolkit REQUIRED )
    set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -Wno-deprecated-gpu-targets " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC " )
    message ( STATUS "C++ flags for CUDA: ${CMAKE_CXX_FLAGS}" )
    message ( STATUS "CUDA include dirs:  ${CUDA_INCLUDE_DIRS}" )
    message ( STATUS "CUDA libraries:     ${CUDA_LIBRARIES}" )
    set ( GB_CUDA ${CUDA_LIBRARIES} )

    include_directories ( "rmm_wrap" "CUDA/include" "CUDA" ${CUDA_INCLUDE_DIRS} )

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

    if ( NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY}/rmm )
        message ( STATUS "cloning rmm branch-21.10" )
        execute_process (
            COMMAND git clone "https://github.com/rapidsai/rmm" --branch branch-21.10 --recursive rmm
#           COMMAND git clone "https://github.com/rapidsai/rmm" --branch branch-25.10 --recursive rmm
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes )
    endif()

    set ( RMM_WRAP_INCLUDES ${PROJECT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/external_includes/spdlog/include
# for v21.10:
        ${CMAKE_CURRENT_BINARY_DIR}/external_includes/rmm/include
# for v25.10:
#       ${CMAKE_CURRENT_BINARY_DIR}/external_includes/rmm/cpp/include
        ${CUDA_INCLUDE_DIRS} )

else ( )

    # without CUDA
    set ( GB_CUDA )
    set ( RMM_WRAP_INCLUDES )

endif ( )

#-------------------------------------------------------------------------------
# OpenMP
#-------------------------------------------------------------------------------

option ( GRAPHBLAS_USE_OPENMP "ON: Use OpenMP in GraphBLAS if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( GRAPHBLAS_USE_OPENMP )
    if ( CMAKE_VERSION VERSION_LESS 3.24 )
        if ( GRAPHBLAS_HAS_CUDA )
            find_package ( OpenMP COMPONENTS C CXX )
        else ( )
            find_package ( OpenMP COMPONENTS C )
        endif ( )
    else ( )
        if ( GRAPHBLAS_HAS_CUDA )
            find_package ( OpenMP COMPONENTS C CXX GLOBAL )
        else ( )
            find_package ( OpenMP COMPONENTS C GLOBAL )
        endif ( )
    endif ( )
else ( )
    # OpenMP has been disabled.
    set ( OpenMP_C_FOUND OFF )
    set ( OpenMP_CXX_FOUND OFF )
endif ( )

if ( GRAPHBLAS_USE_OPENMP AND OpenMP_C_FOUND )
    set ( GRAPHBLAS_HAS_OPENMP ON )
else ( )
    set ( GRAPHBLAS_HAS_OPENMP OFF )
endif ( )
message ( STATUS "GraphBLAS has OpenMP: ${GRAPHBLAS_HAS_OPENMP}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND GRAPHBLAS_USE_OPENMP AND NOT GRAPHBLAS_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP required for GraphBLAS but not found" )
endif ( )

#-------------------------------------------------------------------------------
# cpu_features
#-------------------------------------------------------------------------------

if ( NOT GBNCPUFEAT )
    # default: enable Google's cpu_features package
    message ( STATUS "cpu_features (by google.com): enabled " )
    include_directories ( "cpu_features/include" "cpu_features" "cpu_features/src" "cpu_features/include/internal" )
else ( )
    # disable Google's cpu_features package
    message ( STATUS "cpu_features (by google.com): disabled" )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBNCPUFEAT " )
endif ( )

if ( DEFINED GBX86 )
    # default: this is detected automatically, but can be set here also
    if ( GBX86 )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBX86=1 " )
    else ( )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBX86=0 " )
    endif ( )
endif ( )

if ( DEFINED GBAVX2 )
    # default: this is detected automatically, but can be set here also
    if ( GBAVX2 )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBAVX2=1 " )
    else ( )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBAVX2=0 " )
    endif ( )
endif ( )

if ( DEFINED GBAVX512F )
    # default: this is detected automatically, but can be set here also
    if ( GBAVX512F )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBAVX512F=1 " )
    else ( )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBAVX512F=0 " )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# RISC-V
#-------------------------------------------------------------------------------

if ( DEFINED GBRISCV64 )
    if ( GBRISCV64 )
    # default: this is detected automatically, but can be set here also
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBRISCV64=1 " )
    else ( )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBRISCV64=0 " )
    endif ( )
endif ( )

if ( DEFINED GBRVV )
    if ( GBRVV )
    # default: this is detected automatically, but can be set here also
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBRVV=1 " )
    else ( )
        set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBRVV=0 " )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# check compiler support for the complex data types
#-------------------------------------------------------------------------------

include ( GraphBLAS_complex )

#-------------------------------------------------------------------------------
# determine build type
#-------------------------------------------------------------------------------

# select "ON" to build both dynamic and static libraries:
# set ( BUILD_STATIC_LIBS ON )
# set ( BUILD_STATIC_LIBS OFF )
# or use cmake with -DGRAPHBLAS_BUILD_STATIC_LIBS=ON

if ( BUILD_STATIC_LIBS AND BUILD_SHARED_LIBS )
    message ( STATUS "Building all GraphBLAS libraries (static and dynamic)" )
elseif ( BUILD_SHARED_LIBS )
    message ( STATUS "Building dynamic GraphBLAS library (no static)" )
else ( )
    message ( STATUS "Building static GraphBLAS library (no shared)" )
endif ( )

#-------------------------------------------------------------------------------
# Configure Include/GraphBLAS.h and documentation with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/GraphBLAS.h.in"
    "${PROJECT_SOURCE_DIR}/Include/GraphBLAS.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/GraphBLAS_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/GraphBLAS_version.tex"
    NEWLINE_STYLE LF )
configure_file ( "Config/GraphBLAS_API_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/GraphBLAS_API_version.tex"
    NEWLINE_STYLE LF )
configure_file ( "Config/README.md.in"
    "${PROJECT_SOURCE_DIR}/README.md"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories for both GraphBLAS and the demos
#-------------------------------------------------------------------------------

include_directories ( ${PROJECT_SOURCE_DIR} Source Include Config
    xxHash lz4 zstd zstd/zstd_subset JITpackage Demo/Include rmm_wrap
    # include all Source/* folders that have include/ or template/ subfolders:
    Source/add
    Source/apply
    Source/assign
    Source/builder
    Source/builtin
    Source/callback
    Source/concat
    Source/convert
    Source/cumsum
    Source/emult
    Source/ewise
    Source/extract
    Source/hyper
    Source/ij
    Source/jit_kernels
    Source/kronecker
    Source/mask
    Source/math
    Source/matrix
    Source/memory
    Source/monoid
    Source/mxm
    Source/ok
    Source/omp
    Source/print
    Source/reduce
    Source/select
    Source/split
    Source/slice
    Source/sort
    Source/transpose
    Source/type
    Source/wait
    Source/werk
    # include all CUDA/* folders that have include/ or template/ subfolders:
    CUDA
    CUDA/apply
    CUDA/cumsum
    CUDA/device
    CUDA/include
    CUDA/init
    CUDA/matrix
    CUDA/monoid
    CUDA/mxm
    CUDA/reduce
    CUDA/select
    CUDA/slice
    CUDA/type
    )

#-------------------------------------------------------------------------------
# compiler options
#-------------------------------------------------------------------------------

include ( GraphBLAS_compiler_options )

#-------------------------------------------------------------------------------
# GraphBLAS source code
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_HAS_CUDA )
    file ( GLOB GRAPHBLAS_SOURCES
        "rmm_wrap/rmm_wrap.cpp"
        "CUDA/*/*.cu" "CUDA/*/*.c" "CUDA/*/*.cpp"
        "PreJIT/*.c" "Config/*.c" "Source/*/*.c" )
else ( )
    file ( GLOB GRAPHBLAS_SOURCES "PreJIT/*.c" "Config/*.c" "Source/*/*.c" )
endif ( )

if ( NOT GRAPHBLAS_COMPACT )
    # compile the FactoryKernels
    file ( GLOB GRAPHBLAS_FACTORYKERNELS "FactoryKernels/*.c" )
    list ( APPEND GRAPHBLAS_SOURCES ${GRAPHBLAS_FACTORYKERNELS} )
endif ( )

#-------------------------------------------------------------------------------
# GraphBLAS JITpackage
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_USE_JIT )
    # generate compressed JIT sources to create GB_JITpackage.c
    message ( STATUS "Creating the GraphBLAS/JITpackage:" )
    add_subdirectory ( JITpackage )
    # GLOB does not include files that do not yet exist, so prepend
    # the GB_JITpackage.c file here:
    list ( PREPEND GRAPHBLAS_SOURCES
        $<TARGET_PROPERTY:GB_JITpackage,GENERATED_FILE> )
else ( )
    # create an empty GB_JITpackage.c file:
    message ( STATUS "Creating an empty GraphBLAS/JITpackage (JIT disabled):" )
    configure_file ( "Config/GB_JITpackage_NJIT.c.in"
        "${PROJECT_SOURCE_DIR}/JITpackage/GB_JITpackage.c"
        NEWLINE_STYLE LF )
    list ( PREPEND GRAPHBLAS_SOURCES "JITpackage/GB_JITpackage.c")
endif ( )

#-------------------------------------------------------------------------------
# dynamic GraphBLAS library properties
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )

    # create the dynamic GraphBLAS library and set its properties
    add_library ( GraphBLAS SHARED ${GRAPHBLAS_SOURCES} )

    set_target_properties ( GraphBLAS PROPERTIES
        VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
        OUTPUT_NAME graphblas
        SOVERSION ${GraphBLAS_VERSION_MAJOR}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        PUBLIC_HEADER "Include/GraphBLAS.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( GRAPHBLAS_HAS_CUDA )
        set_target_properties ( GraphBLAS PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON )
        set_target_properties ( GraphBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( GraphBLAS PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( GraphBLAS
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

    if ( GRAPHBLAS_HAS_CUDA )
        target_compile_definitions ( GraphBLAS PRIVATE "GRAPHBLAS_HAS_CUDA" )
        target_include_directories ( GraphBLAS PRIVATE "${RMM_WRAP_INCLUDES}" )
# required for rmm 25.10:
#       target_compile_definitions ( GraphBLAS PRIVATE "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE" )
    endif ( )

    if ( WIN32 )
        # Compiling the GraphBLAS dll on Windows: export the dll symbols
        target_compile_definitions ( GraphBLAS PRIVATE GB_DLL_EXPORT )
    endif ( )

    if ( GRAPHBLAS_USE_JIT )
        # Make sure compressed sources are built before library
        add_dependencies ( GraphBLAS GB_JITpackage )
    endif ( )

    get_target_property ( CSTD GraphBLAS C_STANDARD )
    message ( STATUS "GraphBLAS C standard: ${CSTD}" )
endif ( )

#-------------------------------------------------------------------------------
# static GraphBLAS library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )

    # create the static GraphBLAS library and set its properties
    add_library ( GraphBLAS_static STATIC ${GRAPHBLAS_SOURCES} )

    set_target_properties ( GraphBLAS_static PROPERTIES
        OUTPUT_NAME graphblas
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        PUBLIC_HEADER "Include/GraphBLAS.h" )

    if ( GRAPHBLAS_HAS_CUDA )
        set_target_properties ( GraphBLAS_static PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON )
        set_target_properties ( GraphBLAS_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    endif ( )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( GraphBLAS_static PROPERTIES
            OUTPUT_NAME graphblas_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( GraphBLAS_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( GraphBLAS_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

    if ( GRAPHBLAS_HAS_CUDA )
        target_compile_definitions ( GraphBLAS_static PRIVATE "GRAPHBLAS_HAS_CUDA" )
# required for rmm 25.10:
#       target_compile_definitions ( GraphBLAS_static PRIVATE "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE" )
        target_include_directories ( GraphBLAS_static PRIVATE "${RMM_WRAP_INCLUDES}" )
    endif ( )

    if ( WIN32 )
        # On Windows, don't use __declspec ( dllimport ) for static library.
        target_compile_definitions ( GraphBLAS_static PUBLIC GB_STATIC )
    endif ( )

    if ( GRAPHBLAS_USE_JIT )
        # Make sure compressed sources are built before library
        add_dependencies ( GraphBLAS_static GB_JITpackage )
    endif ( )

    get_target_property ( CSTD GraphBLAS_static C_STANDARD )
    message ( STATUS "GraphBLAS C standard: ${CSTD}" )
endif ( )

#-------------------------------------------------------------------------------
# cpu_features settings
#-------------------------------------------------------------------------------

include ( CheckSymbolExists )
if ( NOT GBNCPUFEAT )
    if ( UNIX )
        # look for requirements for cpu_features/src/hwcaps.c
        include ( CheckIncludeFile )
        check_include_file ( dlfcn.h HAVE_DLFCN_H )
        if ( HAVE_DLFCN_H )
            message ( STATUS "cpu_feautures has dlfcn.h" )
            if ( BUILD_SHARED_LIBS )
                target_compile_definitions ( GraphBLAS PRIVATE HAVE_DLFCN_H )
            endif ( )
            if ( BUILD_STATIC_LIBS )
                target_compile_definitions ( GraphBLAS_static PRIVATE HAVE_DLFCN_H )
            endif ( )
        else ( )
            message ( STATUS "cpu_feautures without dlfcn.h" )
        endif ( )
        check_symbol_exists ( getauxval "sys/auxv.h" HAVE_STRONG_GETAUXVAL )
        if ( HAVE_STRONG_GETAUXVAL )
            message ( STATUS "cpu_feautures has getauxval from sys/auxv.h" )
            if ( BUILD_SHARED_LIBS )
                target_compile_definitions ( GraphBLAS PRIVATE HAVE_STRONG_GETAUXVAL )
            endif ( )
            if ( BUILD_STATIC_LIBS )
                target_compile_definitions ( GraphBLAS_static PRIVATE HAVE_STRONG_GETAUXVAL )
            endif ( )
        else ( )
            message ( STATUS "cpu_feautures doesn't have getauxval from sys/auxv.h" )
        endif ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# determine which basic libraries are needed (m, dl, and atomic)
#-------------------------------------------------------------------------------

# libm:
check_symbol_exists ( fmax "math.h" NO_LIBM )
if ( NOT NO_LIBM )
    set ( GB_M "m" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( GraphBLAS PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND GRAPHBLAS_STATIC_LIBS "m" )
        target_link_libraries ( GraphBLAS_static PUBLIC m )
    endif ( )
endif ( )

# libdl
if ( NOT "${CMAKE_DL_LIBS}" STREQUAL "" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( GraphBLAS PRIVATE ${CMAKE_DL_LIBS} )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND GRAPHBLAS_STATIC_LIBS ${CMAKE_DL_LIBS} )
        target_link_libraries ( GraphBLAS_static PUBLIC ${CMAKE_DL_LIBS} )
    endif ( )
endif ( )

# atomic
include ( SuiteSparseAtomic )
if ( LIBATOMIC_REQUIRED )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( GraphBLAS PRIVATE atomic )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND GRAPHBLAS_STATIC_LIBS "atomic" )
        target_link_libraries ( GraphBLAS_static PUBLIC atomic )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# get the current library list, before linking with OpenMP, CUDA and rmm
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    get_target_property ( GB_CMAKE_LIBRARIES GraphBLAS LINK_LIBRARIES )
else ( )
    get_target_property ( GB_CMAKE_LIBRARIES GraphBLAS_static LINK_LIBRARIES )
endif ( )
if ( NOT GB_CMAKE_LIBRARIES )
    set ( GB_CMAKE_LIBRARIES "" )
endif ( )

#-------------------------------------------------------------------------------
# add the OpenMP, IPP, CUDA, BLAS, etc libraries
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_HAS_OPENMP )
    message ( STATUS "CMAKE OpenMP libraries:    ${OpenMP_C_LIBRARIES}" )
    message ( STATUS "CMAKE OpenMP include:      ${OpenMP_C_INCLUDE_DIRS}" )
    list ( APPEND GB_CMAKE_LIBRARIES ${OpenMP_C_LIBRARIES} )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( GraphBLAS PRIVATE OpenMP::OpenMP_C )
        if ( GRAPHBLAS_HAS_CUDA )
            target_link_libraries ( GraphBLAS PRIVATE OpenMP::OpenMP_CXX )
        endif ( )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND GRAPHBLAS_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
        target_link_libraries ( GraphBLAS_static PRIVATE OpenMP::OpenMP_C )
        if ( GRAPHBLAS_HAS_CUDA )
            target_link_libraries ( GraphBLAS_static PRIVATE OpenMP::OpenMP_CXX )
        endif ( )
    endif ( )
    message ( STATUS "CMAKE OpenMP C flags:      ${OpenMP_C_FLAGS}" )
    set ( GB_OPENMP_C_FLAGS "${OpenMP_C_FLAGS}" )
else ( )
    message ( WARNING
    "WARNING:  OpenMP was not found (or was disabled with "
    "GRAPHBLAS_USE_OPENMP).  See the GraphBLAS user guide on the consequences "
    "of compiling GraphBLAS without OpenMP. GraphBLAS will work but may not be "
    "thread-safe, since it relies on '#pragma omp flush' to ensure the work "
    "performed by one user thread is available to another, in GrB_wait.  If "
    "OpenMP is not in use, the thread-safety of GrB_wait becomes the "
    "responsibilty of the user application (perhaps through a pthreads "
    "construct).  Compiling GraphBLAS without OpenMP is not recommended for "
    "installation in a package manager (Linux, conda-forge, spack, brew, "
    "vcpkg, etc). " )
    # check for __thread, __declspec(thread), and _Thread_local keywords
    include ( SuiteSparse__thread )
    if ( NOT HAVE_KEYWORD__DECLSPEC_THREAD AND NOT HAVE_KEYWORD__THREAD
        AND NOT HAVE_KEYWORD__THREAD_LOCAL )
        message ( WARNING
        "The C compiler does not support thread-local-storage; "
        "GxB_Context_engage will return GrB_NOT_IMPLEMENTED." )
    endif ( )
    set ( GB_OPENMP_C_FLAGS "" )
endif ( )

#-------------------------------------------------------------------------------
# link CUDA libraries
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_HAS_CUDA )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( GraphBLAS PRIVATE ${GB_CUDA} )
        target_link_libraries ( GraphBLAS PRIVATE CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver )
        if ( TARGET CUDA::nvToolsExt )
            target_link_libraries ( GraphBLAS PRIVATE CUDA::nvToolsExt )
        endif ( )
        if ( TARGET CUDA::nvtx3 )
            target_link_libraries ( GraphBLAS PRIVATE CUDA::nvtx3 )
            target_compile_definitions ( GraphBLAS PRIVATE GBNVTX )
        endif ( )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( GraphBLAS_static PUBLIC ${GB_CUDA} )
        target_link_libraries ( GraphBLAS_static PRIVATE CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver )
        if ( TARGET CUDA::nvToolsExt )
            target_link_libraries ( GraphBLAS_static PRIVATE CUDA::nvToolsExt )
        endif ( )
        if ( TARGET CUDA::nvtx3 )
            target_link_libraries ( GraphBLAS_static PRIVATE CUDA::nvtx3 )
            target_compile_definitions ( GraphBLAS_static PRIVATE GBNVTX )
        endif ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# print final C flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE C flags: ${CMAKE_C_FLAGS} " )

#-------------------------------------------------------------------------------
# Demo programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    message ( STATUS "Also compiling the demos in GraphBLAS/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( complex_demo  "Demo/Program/complex_demo.c" )
    add_executable ( simple_demo   "Demo/Program/simple_demo.c" )
    add_executable ( wildtype_demo "Demo/Program/wildtype_demo.c" )
    add_executable ( wathen_demo   "Demo/Program/wathen_demo.c" )
    add_executable ( context_demo  "Demo/Program/context_demo.c" )
    add_executable ( gauss_demo    "Demo/Program/gauss_demo.c" )
    add_executable ( grow_demo     "Demo/Program/grow_demo.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( complex_demo PUBLIC GraphBLAS )
        target_link_libraries ( simple_demo PUBLIC GraphBLAS )
        target_link_libraries ( wildtype_demo PUBLIC GraphBLAS )
        target_link_libraries ( wathen_demo PUBLIC GraphBLAS )
        target_link_libraries ( context_demo PUBLIC GraphBLAS )
        target_link_libraries ( gauss_demo PUBLIC GraphBLAS )
        target_link_libraries ( grow_demo PUBLIC GraphBLAS )
    else ( )
        target_link_libraries ( complex_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( simple_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( wildtype_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( wathen_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( context_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( gauss_demo PUBLIC GraphBLAS_static )
        target_link_libraries ( grow_demo PUBLIC GraphBLAS_static )
    endif ( )

    target_link_libraries ( complex_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( simple_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( wildtype_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( wathen_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( context_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( gauss_demo PUBLIC ${GB_M} ${GB_CUDA} )
    target_link_libraries ( grow_demo PUBLIC ${GB_M} ${GB_CUDA} )

    if ( GRAPHBLAS_HAS_OPENMP )
        target_link_libraries ( wathen_demo PUBLIC OpenMP::OpenMP_C )
        target_link_libraries ( context_demo PUBLIC OpenMP::OpenMP_C )
        target_link_libraries ( grow_demo PUBLIC OpenMP::OpenMP_C )
        if ( GRAPHBLAS_HAS_CUDA )
            target_link_libraries ( wathen_demo PUBLIC OpenMP::OpenMP_CXX )
            target_link_libraries ( context_demo PUBLIC OpenMP::OpenMP_CXX )
            target_link_libraries ( grow_demo PUBLIC OpenMP::OpenMP_CXX )
        endif ( )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in GraphBLAS/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS GraphBLAS
        EXPORT GraphBLASTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

    # create (temporary) export target file during build
    export ( EXPORT GraphBLASTargets
        NAMESPACE SuiteSparse::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASTargets.cmake )

    # install export target and config for find_package
    install ( EXPORT GraphBLASTargets
        NAMESPACE SuiteSparse::
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )
endif ( )

if ( BUILD_STATIC_LIBS )
    install ( TARGETS GraphBLAS_static
        EXPORT GraphBLASTargets_static
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

    # create (temporary) export target file during build
    export ( EXPORT GraphBLASTargets_static
        NAMESPACE SuiteSparse::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASTargets_static.cmake )

    # install export target and config for find_package
    install ( EXPORT GraphBLASTargets_static
        NAMESPACE SuiteSparse::
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )
endif ( )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/GraphBLASConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/GraphBLASConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/GraphBLASConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/GraphBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GraphBLASConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GraphBLAS )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( GRAPHBLAS_STATIC_LIBS_LIST ${GRAPHBLAS_STATIC_LIBS} )
    set ( GRAPHBLAS_STATIC_LIBS "" )
    foreach ( _lib ${GRAPHBLAS_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( GRAPHBLAS_STATIC_LIBS "${GRAPHBLAS_STATIC_LIBS} -l${_lib}" )
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
                    set ( GRAPHBLAS_STATIC_LIBS "${GRAPHBLAS_STATIC_LIBS} -l${_libname}" )
                    break ()
                endif ( )
            endif ( )
        endforeach ( )
    endforeach ( )

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
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:GraphBLAS> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:GraphBLAS_static> )
    endif ( )
    configure_file (
        Config/GraphBLAS.pc.in
        GraphBLAS.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT GraphBLAS.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GraphBLAS.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# configure the JIT
#-------------------------------------------------------------------------------

include ( GraphBLAS_JIT_configure )

configure_file ( "Config/GB_config.h.in"
    "${PROJECT_SOURCE_DIR}/Config/GB_config.h"
    NEWLINE_STYLE LF )

include ( GraphBLAS_PreJIT )

#-------------------------------------------------------------------------------
# dump all variables, for debugging only
#-------------------------------------------------------------------------------

# uncomment this line or add -DDUMP=true to your cmake call to dump all cmake variables:
# set ( DUMP true )

if ( DUMP )
    message("====")
    message("Begin dumping all CMake variables in alphabetic order")
    message("====")
    get_cmake_property ( _variableNames VARIABLES )
    list ( SORT _variableNames )
    foreach ( _variableName ${_variableNames} )
        message ( STATUS "${_variableName}=${${_variableName}}" )
    endforeach ( )
    message("====")
    message("End dumping all CMake variables in alphabetic order")
    message("====")
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( GraphBLASReport )
