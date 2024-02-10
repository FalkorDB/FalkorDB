cmake_minimum_required ( VERSION 3.20 )

project ( falkordb )

file ( GLOB_RECURSE SOURCES "src/*.c" )
include_directories ( "deps/rax" "deps/xxHash" "deps/GraphBLAS/Include" "src" )

add_library ( falkordb SHARED ${SOURCES} )

################################################################################
# libcypher-parser
################################################################################

include ( ExternalProject )
ExternalProject_Add ( parser
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libcypher-parser
    CONFIGURE_COMMAND ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/autogen.sh && ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/configure
    BUILD_COMMAND make CFLAGS="-Wno-error=ignored-qualifiers"
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE TRUE
)

# Define library libcypher-parser target
add_library(libcypher-parser STATIC IMPORTED)
add_dependencies(libcypher-parser parser)

# Set properties for library B
set_target_properties(libcypher-parser PROPERTIES
	IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/lib/src/libcypher-parser.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/lib/src
)

################################################################################
# GraphBLAS
################################################################################

add_custom_target ( GraphBLAS
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/GraphBLAS
	COMMENT "Building GraphBLAS"
)

################################################################################
# oniguruma
################################################################################

add_subdirectory ( ${CMAKE_SOURCE_DIR}/deps/oniguruma oniguruma )

################################################################################
# rax
################################################################################

add_custom_target ( rax
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/rax
	COMMENT "Building rax"
)

################################################################################
# RediSearch
################################################################################

add_custom_target ( RediSearch
	COMMAND ${CMAKE_MAKE_PROGRAM} STATIC=1 -C ${CMAKE_SOURCE_DIR}/deps/RediSearch
	COMMENT "Building RediSearch"
)

################################################################################
# utf8proc
################################################################################

add_custom_target ( utf8proc
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/utf8proc
	COMMENT "Building utf8proc"
)

################################################################################
# xxHash
################################################################################

add_custom_target ( xxHash
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/xxHash
	COMMENT "Building xxHash"
)

add_dependencies ( falkordb libcypher-parser GraphBLAS onig rax RediSearch utf8proc xxHash )

# Link with libcypher-parser
target_link_libraries ( falkordb PRIVATE libcypher-parser )

# Link with GraphBLAS
target_link_libraries ( falkordb PRIVATE libcypher-parser )

# Set linker flags for static linking
target_link_options ( falkordb PUBLIC "-static" )
