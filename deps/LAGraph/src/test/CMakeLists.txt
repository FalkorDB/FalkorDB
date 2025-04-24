#-------------------------------------------------------------------------------
# LAGraph/src/test/CMakeLists.txt:  cmake script for LAGraph/src/test
#-------------------------------------------------------------------------------

# LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM22-0790

#-------------------------------------------------------------------------------

# Most tests can be done with any GraphBLAS library, but extensive ("brutal")
# memory test can only be done with SuiteSparse:GraphBLAS.

#-------------------------------------------------------------------------------
# build the lagraphtest library
#-------------------------------------------------------------------------------

include_directories ( ${PROJECT_SOURCE_DIR}/src/test/include
    ${PROJECT_SOURCE_DIR}/src/algorithm )

file ( GLOB LAGRAPHTEST_LIB_SOURCES "LG_*.c" )

# Uncomment this line for for development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )
if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
endif ( )

#-------------------------------------------------------------------------------
# dynamic lagraphtest library properties
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    add_library ( lagraphtest SHARED ${LAGRAPHTEST_LIB_SOURCES} )
    set_target_properties ( lagraphtest PROPERTIES
        VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
        SOVERSION ${LAGraph_VERSION_MAJOR}
        C_STANDARD_REQUIRED ON
        C_STANDARD 11
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dlls )
    target_link_libraries ( lagraphtest PRIVATE LAGraph GraphBLAS::GraphBLAS )
    target_compile_definitions ( lagraphtest PRIVATE LG_TEST_LIBRARY )
    target_compile_definitions ( lagraphtest PUBLIC LG_TEST_DLL )
endif ( )

#-------------------------------------------------------------------------------
# static lagraphtest library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( lagraphtest_static STATIC ${LAGRAPHTEST_LIB_SOURCES} )
    set_target_properties ( lagraphtest_static PROPERTIES
        VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
        OUTPUT_NAME lagraphtest
        POSITION_INDEPENDENT_CODE OFF
        SOVERSION ${LAGraph_VERSION_MAJOR}
        C_STANDARD_REQUIRED ON
        C_STANDARD 11 )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( lagraphtest_static PROPERTIES
            OUTPUT_NAME lagraphtest_static )
    endif ( )

    target_link_libraries ( lagraphtest_static PRIVATE LAGraph_static GraphBLAS::GraphBLAS )
endif ( )

#-------------------------------------------------------------------------------
# add OpenMP
#-------------------------------------------------------------------------------

if ( LAGRAPH_HAS_OPENMP )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( lagraphtest PRIVATE OpenMP::OpenMP_C )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( lagraphtest_static PRIVATE OpenMP::OpenMP_C )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# This will only build tests from files with the name "test_*.c"
#-------------------------------------------------------------------------------

file( GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.c )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".c" "" testname ${justname} )
#   message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile})
    set_target_properties ( ${testname} PROPERTIES
        C_STANDARD_REQUIRED ON
        C_STANDARD 11 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries( ${testname} LAGraph LAGraphX lagraphtest GraphBLAS::GraphBLAS )
    else ( )
        target_link_libraries( ${testname} LAGraph_static LAGraphX_static lagraphtest_static GraphBLAS::GraphBLAS )
    endif ( )
    string( REPLACE "test_" "LAGraph_" ctestname ${testname})
    add_test( NAME ${ctestname} COMMAND $<TARGET_FILE:${testname}> )
#   add_test( NAME ${ctestname} COMMAND valgrind $<TARGET_FILE:${testname}> )
    if (WIN32)
        if ( BUILD_SHARED_LIBS )
            set_tests_properties ( ${ctestname} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:GraphBLAS::GraphBLAS>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lagraphtest>" )
        else ( )
            set_tests_properties ( ${ctestname} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:GraphBLAS::GraphBLAS>" )
        endif ( )
    endif ( )
endforeach( testsourcefile ${TEST_SOURCES} )

