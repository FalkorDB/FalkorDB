#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_JIT_paths.cmake:  configure the JIT paths
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------
# define the source and cache paths
#-------------------------------------------------------------------------------

# The GraphBLAS CPU and CUDA JITs need to know where the GraphBLAS source is
# located, and where to put the compiled libraries.

# set the GRAPHBLAS_CACHE_PATH for compiled JIT kernels
if ( DEFINED ENV{GRAPHBLAS_CACHE_PATH} )
    # use the GRAPHBLAS_CACHE_PATH environment variable
    set ( GRAPHBLAS_CACHE_PATH "$ENV{GRAPHBLAS_CACHE_PATH}" )
elseif ( DEFINED ENV{HOME} )
    # use the current HOME environment variable from cmake (for Linux, Unix, Mac)
    set ( GRAPHBLAS_CACHE_PATH "$ENV{HOME}/.SuiteSparse/GrB${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )
    if ( GBMATLAB AND APPLE )
        # MATLAB on the Mac is a non-native application so the compiled JIT
        # kernels are compiled to x86 assembly.  The primary libgraphblas.dylib
        # called from a C application would likely be native, in ARM assembly.
        # So use a different JIT folder for MATLAB.
        set ( GRAPHBLAS_CACHE_PATH "${GRAPHBLAS_CACHE_PATH}_matlab" )
    endif ( )
elseif ( WIN32 )
    # use LOCALAPPDATA for Windows
    set ( GRAPHBLAS_CACHE_PATH "$ENV{LOCALAPPDATA}/SuiteSparse/GrB${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )
else ( )
    # otherwise the cache path must be set at run time by GB_jitifyer_init
    set ( GRAPHBLAS_CACHE_PATH "" )
endif ( )

#-------------------------------------------------------------------------------
# NJIT and COMPACT options
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    # FOR NOW: do not compile FactoryKernels when developing the CUDA kernels
    set ( COMPACT on )
endif ( )

option ( COMPACT "ON: do not compile FactoryKernels.  OFF (default): compile FactoryKernels" off )
option ( NJIT "ON: do not use the CPU JIT.  OFF (default): enable the CPU JIT" off )

if ( NJIT )
    # disable the CPU JIT (but keep any PreJIT kernels enabled)
    add_compile_definitions ( NJIT )
    message ( STATUS "GraphBLAS CPU JIT: disabled (any PreJIT kernels will still be enabled)")
else ( )
    message ( STATUS "GraphBLAS CPU JIT: enabled")
endif ( )

if ( COMPACT )
    add_compile_definitions ( GBCOMPACT )
    message ( STATUS "GBCOMPACT: enabled; FactoryKernels will not be built" )
endif ( )

set ( JITINIT 4
    CACHE STRING "Initial JIT control 4:on, 3:load, 2:run, 1:pause, 0:off (default 4)" )
if ( NOT ( ${JITINIT} EQUAL 4 ))
    add_compile_definitions ( JITINIT=${JITINIT} )
endif ( )

