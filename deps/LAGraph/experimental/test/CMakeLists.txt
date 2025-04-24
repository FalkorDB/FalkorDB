#-------------------------------------------------------------------------------
# LAGraph/experimental/test/CMakeLists.txt
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

#-------------------------------------------------------------------------------
# build the lagraphxtest library
#-------------------------------------------------------------------------------

include_directories ( ${PROJECT_SOURCE_DIR}/src/test/include
    ${PROJECT_SOURCE_DIR}/src/test/include
    ${PROJECT_SOURCE_DIR}/experimental/test/include )

file ( GLOB LAGRAPHXTEST_LIB_SOURCES "LG_*.c" )

# Uncomment this line for for development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )
if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
endif ( )

#-------------------------------------------------------------------------------
# dynamic lagraphxtest library properties
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    add_library ( lagraphxtest SHARED ${LAGRAPHXTEST_LIB_SOURCES} )
    set_target_properties ( lagraphxtest PROPERTIES
        VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
        SOVERSION ${LAGraph_VERSION_MAJOR}
        C_STANDARD_REQUIRED ON
        C_STANDARD 11
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dlls )
    target_link_libraries ( lagraphxtest PRIVATE LAGraph lagraphtest GraphBLAS::GraphBLAS )
    target_compile_definitions ( lagraphxtest PRIVATE LGX_TEST_LIBRARY )
    target_compile_definitions ( lagraphxtest PUBLIC LGX_TEST_DLL )
endif ( )

#-------------------------------------------------------------------------------
# static lagraphxtest library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( lagraphxtest_static STATIC ${LAGRAPHXTEST_LIB_SOURCES} )
    set_target_properties ( lagraphxtest_static PROPERTIES
        VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
        OUTPUT_NAME lagraphxtest
        POSITION_INDEPENDENT_CODE OFF
        SOVERSION ${LAGraph_VERSION_MAJOR}
        C_STANDARD_REQUIRED ON
        C_STANDARD 11 )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( lagraphxtest_static PROPERTIES
            OUTPUT_NAME lagraphxtest_static )
    endif ( )

    target_link_libraries ( lagraphxtest_static PRIVATE LAGraph_static lagraphtest_static GraphBLAS::GraphBLAS )
endif ( )

#-------------------------------------------------------------------------------
# add OpenMP
#-------------------------------------------------------------------------------

if ( LAGRAPH_HAS_OPENMP )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( lagraphxtest PRIVATE OpenMP::OpenMP_C )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( lagraphxtest_static PRIVATE OpenMP::OpenMP_C )
    endif ( )
endif ( )


#-------------------------------------------------------------------------------
# This will only build tests from files with the name "test_*.c"
#-------------------------------------------------------------------------------

file( GLOB TEST_SOURCES LIST_DIRECTORIES false *_test.c test*.c )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".c" "" testname ${justname} )
#   message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} )
    set_target_properties ( ${testname} PROPERTIES
        C_STANDARD_REQUIRED ON
        C_STANDARD 11 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries( ${testname}
            LAGraphX LAGraph lagraphxtest lagraphtest GraphBLAS::GraphBLAS )
    else ( )
        target_link_libraries( ${testname}
            LAGraphX_static LAGraph_static lagraphxtest_static lagraphtest_static GraphBLAS::GraphBLAS )
    endif ( )
    string( REPLACE "test_" "LAGraphX_" ctestname ${testname})
    add_test( NAME ${ctestname} COMMAND $<TARGET_FILE:${testname}> )
#   add_test( NAME ${ctestname} COMMAND valgrind $<TARGET_FILE:${testname}> )
    if (WIN32)
        if ( BUILD_SHARED_LIBS )
            set_tests_properties ( ${ctestname} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:GraphBLAS::GraphBLAS>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lagraphxtest>" )
        else ( )
            set_tests_properties ( ${ctestname} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:GraphBLAS::GraphBLAS>" )
        endif ( )
    endif ( )
endforeach( testsourcefile ${TEST_SOURCES} )

