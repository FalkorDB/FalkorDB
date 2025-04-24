#-------------------------------------------------------------------------------
# LAGraph/src/benchmark/CMakeLists.txt:  cmake script for LAGraph/src/benchmark
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

include_directories ( ${PROJECT_SOURCE_DIR}/src/test/include
    ${PROJECT_SOURCE_DIR}/src/algorithm )

file( GLOB DEMO_SOURCES LIST_DIRECTORIES false *_demo.c )
foreach( demosourcefile ${DEMO_SOURCES} )
    get_filename_component(justname ${demosourcefile} NAME)
    string( REPLACE ".c" "" demoname ${justname} )
#   message("Adding: ${demoname}")
    add_executable( ${demoname} ${demosourcefile} )
    set_target_properties ( ${demoname} PROPERTIES
        C_STANDARD_REQUIRED ON
        C_STANDARD 11 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries( ${demoname} LAGraph LAGraphX lagraphtest GraphBLAS::GraphBLAS )
    else ( )
        target_link_libraries( ${demoname} LAGraph_static LAGraphX_static lagraphtest_static GraphBLAS::GraphBLAS )
    endif ( )
endforeach( demosourcefile ${DEMO_SOURCES} )
