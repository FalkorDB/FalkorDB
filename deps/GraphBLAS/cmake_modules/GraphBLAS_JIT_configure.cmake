#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_JIT_configure.cmake:  configure the JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------

# construct the JIT compiler/link strings
set ( GB_C_COMPILER  "${CMAKE_C_COMPILER}" )
set ( GB_C_FLAGS "${CMAKE_C_FLAGS}" )
set ( GB_C_LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
set ( GB_LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}" )
set ( GB_LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}" )

# construct the C flags and link flags
if ( APPLE )
    # MacOS
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    if ( NOT GBMATLAB )
        # MATLAB on the Mac is not a native application
        set ( GB_C_FLAGS "${GB_C_FLAGS} -arch ${CMAKE_HOST_SYSTEM_PROCESSOR} " )
    endif ( )
    set ( GB_C_FLAGS "${GB_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT} " )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -dynamiclib " )
    set ( GB_OBJ_SUFFIX ".o" )
elseif ( WIN32 )
    # Windows
    set ( GB_OBJ_SUFFIX ".obj" )
else ( )
    # Linux / Unix
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -shared " )
    set ( GB_OBJ_SUFFIX ".o" )
endif ( )

string ( REPLACE "\"" "\\\"" GB_C_FLAGS ${GB_C_FLAGS} )

# construct the -I list for OpenMP
if ( OPENMP_FOUND )
    set ( GB_OMP_INC_DIRS ${OpenMP_C_INCLUDE_DIRS} )
    set ( GB_OMP_INC ${OpenMP_C_INCLUDE_DIRS} )
    list ( TRANSFORM GB_OMP_INC PREPEND " -I" )
else ( )
    set ( GB_OMP_INC_DIRS "" )
    set ( GB_OMP_INC "" )
endif ( )

# construct the library list
string ( REPLACE "." "\\." LIBSUFFIX1 ${GB_LIB_SUFFIX} )
string ( REPLACE "." "\\." LIBSUFFIX2 ${CMAKE_STATIC_LIBRARY_SUFFIX} )
set ( GB_C_LIBRARIES "" )
foreach ( LIB_NAME ${GB_CMAKE_LIBRARIES} )
    if (( LIB_NAME MATCHES ${LIBSUFFIX1} ) OR ( LIB_NAME MATCHES ${LIBSUFFIX2} ))
        string ( APPEND GB_C_LIBRARIES " " ${LIB_NAME} )
    else ( )
        string ( APPEND GB_C_LIBRARIES " -l" ${LIB_NAME} )
    endif ( )
endforeach ( )

if ( NOT NJIT OR ENABLE_CUDA )
    message ( STATUS "------------------------------------------------------------------------" )
    message ( STATUS "JIT configuration:" )
    message ( STATUS "------------------------------------------------------------------------" )
    # one or both JITs are enabled; make sure the cache path exists
    message ( STATUS "JIT C compiler: ${GB_C_COMPILER}" )
    message ( STATUS "JIT C flags:    ${GB_C_FLAGS}" )
    message ( STATUS "JIT link flags: ${GB_C_LINK_FLAGS}" )
    message ( STATUS "JIT lib prefix: ${GB_LIB_PREFIX}" )
    message ( STATUS "JIT lib suffix: ${GB_LIB_SUFFIX}" )
    message ( STATUS "JIT obj suffix: ${GB_OBJ_SUFFIX}" )
    message ( STATUS "JIT cache:      ${GRAPHBLAS_CACHE_PATH}" )
    message ( STATUS "JIT openmp inc: ${GB_OMP_INC}" )
    message ( STATUS "JIT openmp dirs ${GB_OMP_INC_DIRS}" )
    message ( STATUS "JIT libraries:  ${GB_C_LIBRARIES}" )
    message ( STATUS "JIT cmake libs: ${GB_CMAKE_LIBRARIES}" )
endif ( )

# create the JIT cache directories
file ( MAKE_DIRECTORY ${GRAPHBLAS_CACHE_PATH} )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/src" )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/lib" )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/tmp" )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/lock" )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/c" )
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/cu" )


