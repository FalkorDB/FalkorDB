#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_JIT_configure.cmake:  configure the JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------

# construct the JIT compiler/link strings
if ( MINGW )
    execute_process ( COMMAND cygpath -u "${CMAKE_C_COMPILER}"
        OUTPUT_VARIABLE C_COMPILER_BINARY OUTPUT_STRIP_TRAILING_WHITESPACE )
else ( )
    set ( C_COMPILER_BINARY "${CMAKE_C_COMPILER}" )
endif ( )
set ( GB_C_COMPILER  "${C_COMPILER_BINARY}" )
set ( GB_C_FLAGS "${CMAKE_C_FLAGS}" )
set ( GB_C_LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
set ( GB_LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}" )
set ( GB_LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}" )

# construct the C flags and link flags
if ( APPLE )
    # MacOS
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    if ( NOT GBMATLAB )
        # MATLAB on the Mac might not be a native application
        set ( GB_C_FLAGS "${GB_C_FLAGS} -arch ${CMAKE_HOST_SYSTEM_PROCESSOR} " )
    endif ( )
    if ( DEFINED CMAKE_OSX_SYSROOT AND NOT "${CMAKE_OSX_SYSROOT}" STREQUAL "")
        set ( GB_C_FLAGS "${GB_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT} " )
    endif ( )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -dynamiclib " )
    set ( GB_OBJ_SUFFIX ".o" )
elseif ( MSVC )
    # Microsoft compiler
    set ( GB_OBJ_SUFFIX ".obj" )
else ( )
    # Linux / Unix
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -shared " )
    set ( GB_OBJ_SUFFIX ".o" )
endif ( )

string ( REPLACE "\"" "\\\"" GB_C_FLAGS ${GB_C_FLAGS} )

# construct the -I list for OpenMP
if ( OpenMP_C_FOUND )
    set ( GB_OMP_INC_DIRS ${OpenMP_C_INCLUDE_DIRS} )
    set ( GB_OMP_INC ${OpenMP_C_INCLUDE_DIRS} )
    list ( TRANSFORM GB_OMP_INC PREPEND " -I" )
else ( )
    set ( GB_OMP_INC_DIRS "" )
    set ( GB_OMP_INC "" )
endif ( )

# construct the library list
if ( APPLE )
    set ( default_jit_enable_relocate OFF )
else ( )
    set ( default_jit_enable_relocate ON )
endif ( )

option ( GRAPHBLAS_JIT_ENABLE_RELOCATE
    "ON: Enable relocation of libraries for JIT. OFF: Keep libraries with full path for JIT."
    ${default_jit_enable_relocate} )

if ( GRAPHBLAS_JIT_ENABLE_RELOCATE )

    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags to avoid relocation issues, i.e.: "-lgomp -lpthread -lm"
    set ( GB_C_LIBRARIES "" )
    foreach ( _lib ${GB_CMAKE_LIBRARIES} )

        # skip CUDA::cuda_driver, etc
        if ( ${_lib} MATCHES "CUDA::" )
            continue ( )
        endif ( )

        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( GB_C_LIBRARIES "${GB_C_LIBRARIES} -l${_lib}" )
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
                    set ( GB_C_LIBRARIES "${GB_C_LIBRARIES} -l${_libname}" )
                    break ()
                endif ( )
            endif ( )
        endforeach ( )
    endforeach ( )

else ( )

    # keep full paths to libraries
    string ( REPLACE "." "\\." LIBSUFFIX1 ${GB_LIB_SUFFIX} )
    string ( REPLACE "." "\\." LIBSUFFIX2 ${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set ( GB_C_LIBRARIES "" )
    foreach ( LIB_NAME ${GB_CMAKE_LIBRARIES} )
        message ( STATUS "lib: ${LIB_NAME} " )
        if ( LIB_NAME MATCHES "CUDA::" )
            continue ( )
        elseif (( LIB_NAME MATCHES ${LIBSUFFIX1} ) OR ( LIB_NAME MATCHES ${LIBSUFFIX2} ))
            string ( APPEND GB_C_LIBRARIES " " ${LIB_NAME} )
        else ( )
            string ( APPEND GB_C_LIBRARIES " -l" ${LIB_NAME} )
        endif ( )

    endforeach ( )

endif ( )

if ( GRAPHBLAS_USE_JIT OR GRAPHBLAS_HAS_CUDA )
    message ( STATUS "------------------------------------------------------------------------" )
    message ( STATUS "JIT configuration:" )
    message ( STATUS "------------------------------------------------------------------------" )
    # one or both JITs are enabled; make sure the cache path exists
    message ( STATUS "JIT C compiler: ${GB_C_COMPILER}" )
    message ( STATUS "JIT C flags:    ${GB_C_FLAGS} ${GB_OPENMP_C_FLAGS}" )
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
file ( MAKE_DIRECTORY "${GRAPHBLAS_CACHE_PATH}/c" )

