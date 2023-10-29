 
# Read the package version number specified as the second argument
# to the AC_INIT macro in a GNU Autoconf configure.ac file.
#
# Define the following variables:
# VERSION_STRING:  The second argument to AC_INIT
# MAJOR_VERSION:   For a version string of the form m.n.p..., m
# MINOR_VERSION:   For a version string of the form m.n.p..., n
# PATCH_VERSION:   For a version string of the form m.n.p..., p
# DEVELOPMENT_VERSION:   For a version string of the form m.n.p..., p

macro ( get_ac_init_version )

file (STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/configure.ac" CONFIGURE_AC REGEX "AC_INIT\\(.*\\)" )
string(REGEX REPLACE "AC_INIT\\(\\[.*\\], \\[([0-9]+\\.[0-9]+\\.[0-9]+(~devel)?)\\]\\)" "\\1" PACKAGE_VERSION ${CONFIGURE_AC})
string(REPLACE "AC_INIT([libcypher-parser],[" "" VERSION ${PACKAGE_VERSION})
string(REPLACE "])" "" VERSION ${VERSION})
string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.[0-9]+(~devel)?" "\\1" DEVELOPMENT_VERSION ${VERSION})
string(REGEX REPLACE "([0-9]+\\.[0-9]+\\.[0-9]+)(~devel)?" "\\1" VERSION ${VERSION})
string(REGEX REPLACE "([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" MAJOR_VERSION ${VERSION})
string(REGEX REPLACE "[0-9]+\\.([0-9])+\\.[0-9]+" "\\1" MINOR_VERSION ${VERSION})
string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" PATCH_VERSION ${VERSION})
message(STATUS "Parsed libcypher-parser version: ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}${DEVELOPMENT_VERSION}")
string(REPLACE "~" "" DEVELOPMENT_VERSION ${DEVELOPMENT_VERSION})

endmacro( get_ac_init_version )