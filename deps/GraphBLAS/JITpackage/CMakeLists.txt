#-------------------------------------------------------------------------------
# GraphBLAS/JITpackage/CMakeLists.txt:  package source for the GraphBLAS JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone
project ( jitpackage LANGUAGES C )

set ( GRAPHBLAS_CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING
    "list of configuration flags used for building native generator binaries when cross-compiling" )

if ( CMAKE_CROSSCOMPILING )

    cmake_path ( GET PROJECT_BINARY_DIR FILENAME _subdir )
    if ( ${_subdir} STREQUAL "native" )
        message ( FATAL_ERROR "Native toolchain not configured correctly" )
    endif ( )

    include ( ExternalProject )

    # Try to call CMake on this file with a native toolchain
    ExternalProject_Add ( jitpackage_generator_native
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        BINARY_DIR ${PROJECT_BINARY_DIR}/native
        CMAKE_ARGS ${GRAPHBLAS_CROSS_TOOLCHAIN_FLAGS_NATIVE}
        INSTALL_COMMAND "" )

    # Overriding this variable seems to be necessary if users set
    # CMAKE_FIND_ROOT_PATH_MODE_PACKAGE to ONLY.
    # See: https://github.com/DrTimothyAldenDavis/SuiteSparse/issues/729
    set( GrBJITPackageGenerator_DIR ${PROJECT_BINARY_DIR}/native )

    # The following package won't be found on the first run (before the native
    # grb_jitpackage has been built). But it should be found after the above
    # external project has built it.
    find_package ( GrBJITPackageGenerator
        PATHS ${PROJECT_BINARY_DIR}/native NO_DEFAULT_PATH )

    if ( NOT TARGET grb_jitpackage )
        # target to rerun cmake after jitpackage_generator_native was built
        # fixme: This likely won't work with the Ninja generator or other
        #        generators that lock their own files. But it should work
        #        with a "* Makefiles" generator (which is the default on
        #        most platforms apart from Windows).
        add_custom_target ( Reconfigure
            ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
            DEPENDS jitpackage_generator_native )
    endif ( )

else ( )

    # Native build rules for the generator
    add_executable ( grb_jitpackage "Source/grb_jitpackage.c" )
    target_include_directories ( grb_jitpackage PRIVATE
        ../zstd/zstd_subset ../zstd )
    if ( NOT WIN32 )
        target_link_libraries ( grb_jitpackage PRIVATE m )
    endif ( )

    # Generate an import target to be able to run the native executable when
    # cross-compiling.
    export ( TARGETS grb_jitpackage FILE
        "${CMAKE_BINARY_DIR}/GrBJITPackageGeneratorConfig.cmake")

endif ( )

if ( TARGET grb_jitpackage )
    # This target might not exist on the first configuration run when
    # cross-compiling. But is should exist for native builds and on the second
    # configuration run when cross-compiling.

    file ( GLOB GRB_SOURCE_FILES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../Include/GraphBLAS.h"
        "../CUDA/*/template/*"
        "../CUDA/*/include/*"
        "../CUDA/include/*"
        "../Source/*/template/*"
        "../Source/*/include/*"
        "../Source/include/*" )

    # write list of source files for generator to file
    file ( WRITE ${PROJECT_BINARY_DIR}/grb_source_file_list "" )
    foreach ( src_file ${GRB_SOURCE_FILES} )
        file ( APPEND ${PROJECT_BINARY_DIR}/grb_source_file_list ${src_file}\n )
    endforeach ( )

    add_custom_command ( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/GB_JITpackage.c
        COMMAND $<TARGET_FILE:grb_jitpackage>
        ARGS @${PROJECT_BINARY_DIR}/grb_source_file_list
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${GRB_SOURCE_FILES} ${PROJECT_BINARY_DIR}/grb_source_file_list
        COMMENT "Generating compressed sources for JIT compiler..." )

    # target to make sure the file exists when building libgraphblas
    add_custom_target ( GB_JITpackage
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/GB_JITpackage.c )
    set_target_properties ( GB_JITpackage PROPERTIES
        GENERATED_FILE ${CMAKE_CURRENT_SOURCE_DIR}/GB_JITpackage.c )

else ( )

    # dummy target when cross-compiling before the (native) grb_jitpackage
    # has been built
    add_custom_target ( GB_JITpackage )
    set_target_properties ( GB_JITpackage PROPERTIES
        GENERATED_FILE "" )

    if ( CMAKE_CROSSCOMPILING )
        # configure again after the native grb_jitpackage has been built
        add_dependencies ( GB_JITpackage Reconfigure )
    endif ( )
endif ( )

