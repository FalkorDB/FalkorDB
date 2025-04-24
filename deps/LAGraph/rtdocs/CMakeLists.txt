#-------------------------------------------------------------------------------
# LAGraph/rtdocs/CMakeLists.txt:  cmake script for LAGraph/src/test
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
find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Find all the public headers
get_target_property(LAGRAPH_PUBLIC_HEADER_DIR "lagraph" INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE LAGRAPH_PUBLIC_HEADERS
  ${LAGRAPH_PUBLIC_HEADER_DIR}/*)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Set the Doxygen input and output directories in the Doxyfile
configure_file ( ${DOXYFILE_IN} ${DOXYFILE_OUT}
    @ONLY
    NEWLINE_STYLE LF )

# Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Only regenerate Doxygen when the Doxyfile or public headers change
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${LAGRAPH_PUBLIC_HEADERS}
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
            COMMENT "Generating docs"
        VERBATIM)

# Nice named target so we can run the job easily
add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
#  - Doxygen has rerun
#  - Our doc files have been updated
#  - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
        ${SPHINX_EXECUTABLE} -b html
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.lagraph=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        # Other docs files you want to track should go here (or in some variable)
        ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
        ${DOXYGEN_INDEX_FILE}
    MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
    COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

include(GNUInstallDirs)
install(DIRECTORY ${SPHINX_BUILD}
    DESTINATION ${CMAKE_INSTALL_DOCDIR})
