#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS/CMakeLists.txt:  build GraphBLAS for use in MATLAB
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

# CMakeLists.txt: instructions for cmake to build GraphBLAS for use in MATLAB.
# GraphBLAS is built into MATLAB as libmwgraphblas.  Using another version
# of GraphBLAS causes a naming conflict, which this cmake handles.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../cmake_modules )

include ( GraphBLAS_version )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( graphblas_matlab
    VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( SUITESPARSE_SECOND_LEVEL true )

# CUDA is under development for now, and not deployed in production:
set ( GRAPHBLAS_USE_CUDA OFF )

set ( GBMATLAB on )
set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBMATLAB=1 " )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

# in the future, when GraphBLAS can use CUDA in production:
# option ( GRAPHBLAS_USE_CUDA "ON (default): enable CUDA acceleration for GraphBLAS, OFF: do not use CUDA" ${SUITESPARSE_USE_CUDA} )

set ( GRAPHBLAS_HAS_CUDA OFF )
message ( STATUS "GraphBLAS CUDA JIT: disabled for MATLAB" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND GRAPHBLAS_USE_CUDA AND NOT GRAPHBLAS_HAS_CUDA )
    message ( FATAL_ERROR "CUDA required for GraphBLAS but not found" )
endif ( )

include ( GraphBLAS_JIT_paths )

#-------------------------------------------------------------------------------
# find OpenMP
#-------------------------------------------------------------------------------

option ( GRAPHBLAS_USE_OPENMP "ON: Use OpenMP in GraphBLAS if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( GRAPHBLAS_USE_OPENMP )
    find_package ( OpenMP COMPONENTS C )
else ( )
    # OpenMP has been disabled
    set ( OpenMP_C_FOUND OFF )
endif ( )

if ( OpenMP_C_FOUND )
    set ( GRAPHBLAS_HAS_OPENMP ON )
else ( )
    set ( GRAPHBLAS_HAS_OPENMP OFF )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND GRAPHBLAS_USE_OPENMP AND NOT GRAPHBLAS_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP required for GraphBLAS but not found" )
endif ( )

#-------------------------------------------------------------------------------
# find cpu_features
#-------------------------------------------------------------------------------

if ( NOT GBNCPUFEAT )
    # default: enable Google's cpu_features package
    message ( STATUS "cpu_features (by google.com): enabled " )
    include_directories ( "../cpu_features/include" "../cpu_features" "../cpu_features/src" "../cpu_features/include/internal" )
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
# determine build type
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE build type:          " ${CMAKE_BUILD_TYPE} )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "CMAKE C Flags debug:       " ${CMAKE_C_FLAGS_DEBUG} )
else ( )
    message ( STATUS "CMAKE C Flags release:     " ${CMAKE_C_FLAGS_RELEASE} )
endif ( )

message ( STATUS "C compiler:                 ${CMAKE_C_COMPILER} ")
message ( STATUS "CMAKE have OpenMP:          ${OpenMP_C_FOUND}" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

include_directories ( ${PROJECT_SOURCE_DIR}/../ ../Source ../Include Config
    ../xxHash ../lz4 ../zstd ../zstd/zstd_subset ../JITpackage
    rename
    # include all Source/* folders that have include/ or template/ subfolders:
    ../Source/add
    ../Source/apply
    ../Source/assign
    ../Source/builder
    ../Source/builtin
    ../Source/callback
    ../Source/concat
    ../Source/convert
    ../Source/cumsum
    ../Source/emult
    ../Source/ewise
    ../Source/extract
    ../Source/hyper
    ../Source/ij
    ../Source/jit_kernels
    ../Source/kronecker
    ../Source/mask
    ../Source/math
    ../Source/matrix
    ../Source/memory
    ../Source/monoid
    ../Source/mxm
    ../Source/ok
    ../Source/omp
    ../Source/print
    ../Source/reduce
    ../Source/select
    ../Source/split
    ../Source/slice
    ../Source/sort
    ../Source/transpose
    ../Source/type
    ../Source/wait
    ../Source/werk
    )

#-------------------------------------------------------------------------------
# compiler options:
#-------------------------------------------------------------------------------

include ( GraphBLAS_compiler_options )

#-------------------------------------------------------------------------------
# dynamic graphblas_matlab library properties
#-------------------------------------------------------------------------------

file ( GLOB GRAPHBLAS_SOURCES "../PreJIT/*.c" "../Config/*.c"
    "../Source/*/*.c" )

if ( NOT GRAPHBLAS_COMPACT )
    # compile the FactoryKernels
    file ( GLOB GRAPHBLAS_FACTORYKERNELS "../FactoryKernels/*.c" )
    list ( APPEND GRAPHBLAS_SOURCES ${GRAPHBLAS_FACTORYKERNELS} )
endif ( )

if ( GRAPHBLAS_USE_JIT )
    # generate compressed JIT sources to create GB_JITpackage.c
    message ( STATUS "Creating the GraphBLAS/JITpackage:" )
    add_subdirectory ( ../JITpackage build )
    # GLOB does not include files that do not yet exist, so prepend
    # the GB_JITpackage.c file here:
    list ( PREPEND GRAPHBLAS_SOURCES
        $<TARGET_PROPERTY:GB_JITpackage,GENERATED_FILE> )
else ( )
    # create an empty GB_JITpackage.c file:
    message ( STATUS "Creating an empty GraphBLAS/JITpackage (JIT disabled):" )
    configure_file ( "../Config/GB_JITpackage_NJIT.c.in"
        "../JITpackage/GB_JITpackage.c"
        NEWLINE_STYLE LF )
    list ( PREPEND GRAPHBLAS_SOURCES "../JITpackage/GB_JITpackage.c")
endif ( )

add_library ( graphblas_matlab SHARED ${GRAPHBLAS_SOURCES} )
set_target_properties ( graphblas_matlab PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    PUBLIC_HEADER "../Include/GraphBLAS.h" )

if ( GRAPHBLAS_USE_JIT )
    # Make sure compressed sources are built before library
    add_dependencies ( graphblas_matlab GB_JITpackage )
endif ( )

#-------------------------------------------------------------------------------
# select the math library (not required for Microsoft Visual Studio)
#-------------------------------------------------------------------------------

# libm:
if ( NOT WIN32 )
    target_link_libraries ( graphblas_matlab PUBLIC m )
endif ( )

# libdl
if ( NOT WIN32 )
    target_link_libraries ( graphblas_matlab PUBLIC dl )
endif ( )

# include ( SuiteSparseAtomic )
# if ( LIBATOMIC_REQUIRED )
#     target_link_libraries ( graphblas_matlab PUBLIC atomic )
# endif ( )

#-------------------------------------------------------------------------------
# add library dependencies
#-------------------------------------------------------------------------------

if ( GRAPHBLAS_HAS_OPENMP )
    message ( STATUS "CMAKE OpenMP libraries:    " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "CMAKE OpenMP include:      " ${OpenMP_C_INCLUDE_DIRS} )
#   target_link_libraries ( graphblas_matlab PUBLIC ${OpenMP_C_LIBRARIES} )
    include ( GraphBLAS_MATLAB_OpenMP.cmake )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
else ( )
    # check for __thread, __declspec(thread), and _Thread_local keywords
    include ( SuiteSparse__thread )
endif ( )

#-------------------------------------------------------------------------------
# cpu_features settings
#-------------------------------------------------------------------------------

if ( NOT GBNCPUFEAT )
    if ( UNIX )
        # look for requirements for cpu_features/src/hwcaps.c
        include ( CheckIncludeFile )
        include ( CheckSymbolExists )
        check_include_file ( dlfcn.h HAVE_DLFCN_H )
        if ( HAVE_DLFCN_H )
            target_compile_definitions ( graphblas_matlab PRIVATE HAVE_DLFCN_H )
        endif ( )
        check_symbol_exists ( getauxval "sys/auxv.h" HAVE_STRONG_GETAUXVAL )
        if ( HAVE_STRONG_GETAUXVAL )
            target_compile_definitions ( graphblas_matlab PRIVATE HAVE_STRONG_GETAUXVAL )
        endif ( )
    endif ( )
endif ( )

# get the current library list
get_target_property ( GB_CMAKE_LIBRARIES graphblas_matlab LINK_LIBRARIES )
if ( NOT GB_CMAKE_LIBRARIES )
    set ( GB_CMAKE_LIBRARIES "" )
endif ( )

#-------------------------------------------------------------------------------
# determine complex type
#-------------------------------------------------------------------------------

include ( GraphBLAS_complex )
if ( GxB_HAVE_COMPLEX_C99 )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGxB_HAVE_COMPLEX_C99=1 " )
endif ( )
if ( GxB_HAVE_COMPLEX_MSVC )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGxB_HAVE_COMPLEX_MSVC=1 " )
endif ( )

#-------------------------------------------------------------------------------
# print final C flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE C flags: " ${CMAKE_C_FLAGS} )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

install ( TARGETS graphblas_matlab
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

#-------------------------------------------------------------------------------
# configure the JITs
#-------------------------------------------------------------------------------

include ( GraphBLAS_JIT_configure )

configure_file ( "../Config/GB_config.h.in"
    "${PROJECT_SOURCE_DIR}/Config/GB_config.h"
    NEWLINE_STYLE LF )

