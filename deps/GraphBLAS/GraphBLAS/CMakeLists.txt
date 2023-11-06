#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS/CMakeLists.txt:  build GraphBLAS for use in MATLAB
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

# CMakeLists.txt: instructions for cmake to build GraphBLAS for use in MATLAB.
# GraphBLAS is built into MATLAB as libmwgraphblas.  Using another version
# of GraphBLAS causes a naming conflict, which this cmake handles.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.16 )

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../cmake_modules )

include ( GraphBLAS_version )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( SUITESPARSE_SECOND_LEVEL true )

# CUDA is under development for now, and not deployed in production:
set ( SUITESPARSE_CUDA off )

set ( GBMATLAB on )
set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DGBMATLAB=1 " )

include ( SuiteSparsePolicy )
include ( GraphBLAS_JIT_paths )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( graphblas_matlab
    VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" LANGUAGES C )

#-------------------------------------------------------------------------------
# find OpenMP and cpu_features
#-------------------------------------------------------------------------------

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" off )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

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
# determine build type
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE build type:          " ${CMAKE_BUILD_TYPE} )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "CMAKE C Flags debug:       " ${CMAKE_C_FLAGS_DEBUG} )
else ( )
    message ( STATUS "CMAKE C Flags release:     " ${CMAKE_C_FLAGS_RELEASE} )
endif ( )

message ( STATUS "C compiler:                 ${CMAKE_C_COMPILER} ")
message ( STATUS "CMAKE have OpenMP:          ${OPENMP_FOUND}" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

if ( COMPACT )
    # do not compile the FactoryKernels
    include_directories ( ../Source/Template ../Source ../Include rename
        ../Source/Shared Config ../Source/JitKernels ../JITpackage
        ../lz4 ../zstd ../zstd/zstd_subset ../xxHash ../Source/Factories )
else ( )
    include_directories ( ../Source/Template ../Source ../Include rename
        ../Source/Shared Config ../Source/JitKernels ../JITpackage
        ../Source/FactoryKernels
        ../lz4 ../zstd ../zstd/zstd_subset ../xxHash ../Source/Factories )
endif ( )

#-------------------------------------------------------------------------------
# compiler options:
#-------------------------------------------------------------------------------

include ( GraphBLAS_compiler_options )

#-------------------------------------------------------------------------------
# dynamic graphblas_matlab library properties
#-------------------------------------------------------------------------------

if ( NJIT )
    if ( COMPACT )
        # no JIT, do not compile the FactoryKernels
        file ( GLOB GRAPHBLAS_SOURCES "../Source/*.c"
            "../PreJIT/*.c" "../Config/GB_prejit.c" )
    else ( )
        # no JIT, but do compile the FactoryKernels
        file ( GLOB GRAPHBLAS_SOURCES "../Source/*.c" 
            "../PreJIT/*.c" "../Config/GB_prejit.c"
            "../Source/FactoryKernels/*.c" )
    endif ( )
else ( )
    if ( COMPACT )
        # with the JIT; but do not compile the FactoryKernels
        file ( GLOB GRAPHBLAS_SOURCES "../Source/*.c" 
            "../PreJIT/*.c" "../Config/GB_prejit.c" "../JITpackage/*.c" )
    else ( )
        # with the JIT and the FactoryKernels
        file ( GLOB GRAPHBLAS_SOURCES "../Source/*.c" 
            "../PreJIT/*.c" "../Config/GB_prejit.c" "../JITpackage/*.c"
            "../Source/FactoryKernels/*.c" )
    endif ( )
endif ( )

add_library ( graphblas_matlab SHARED ${GRAPHBLAS_SOURCES} )
set_target_properties ( graphblas_matlab PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    PUBLIC_HEADER "../Include/GraphBLAS.h" )

#-------------------------------------------------------------------------------
# select the threading library
#-------------------------------------------------------------------------------

if ( OPENMP_FOUND )
    set ( USE_OPENMP true )
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

include ( SuiteSparseAtomic )
if ( LIBATOMIC_REQUIRED )
    target_link_libraries ( graphblas_matlab PUBLIC atomic )
endif ( )

#-------------------------------------------------------------------------------
# add library dependencies
#-------------------------------------------------------------------------------

if ( USE_OPENMP )
    message ( STATUS "CMAKE OpenMP libraries:    " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "CMAKE OpenMP include:      " ${OpenMP_C_INCLUDE_DIRS} )
    target_link_libraries ( graphblas_matlab PUBLIC ${OpenMP_C_LIBRARIES} )
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

