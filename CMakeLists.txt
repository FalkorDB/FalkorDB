cmake_minimum_required ( VERSION 3.20 )

project ( falkordb )

find_package(OpenSSL REQUIRED)

file ( GLOB_RECURSE SOURCES "src/*.c" )
include_directories ( "src" "deps/rax" "deps/xxHash" "deps/oniguruma/src" )

add_library ( falkordb SHARED ${SOURCES} )
set_target_properties(falkordb PROPERTIES LIBRARY_OUTPUT_NAME falkordb)

# define REDIS_MODULE_TARGET
add_definitions ( -DREDIS_MODULE_TARGET )

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

# Set properties for library libcypher-parser
set_target_properties(libcypher-parser PROPERTIES
	IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/lib/src/libcypher-parser.la
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/lib/src
)

################################################################################
# GraphBLAS
################################################################################

add_custom_target ( GraphBLAS
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/GraphBLAS
	COMMENT "Building GraphBLAS"
)

# Define library lib-GraphBLAS target
add_library(lib-GraphBLAS STATIC IMPORTED)
add_dependencies(lib-GraphBLAS GraphBLAS)

# Set properties for library lib-GraphBLAS
set_target_properties(lib-GraphBLAS PROPERTIES
	IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/GraphBLAS/build/libgraphblas.dylib
	INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/deps/GraphBLAS/Include
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

# Define library lib-RediSearch target
add_library ( lib-RediSearch STATIC IMPORTED )
add_dependencies ( lib-RediSearch RediSearch )

# Set properties for library lib-RediSearch
set_target_properties(lib-RediSearch PROPERTIES
	IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/RediSearch/bin/macos-arm64v8-release/search-static/libredisearch-static.a
	INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/deps/RediSearch/src
)

################################################################################
# utf8proc
################################################################################

add_custom_target ( utf8proc
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/utf8proc
	COMMENT "Building utf8proc"
)

add_library ( lib-utf8proc STATIC IMPORTED )
add_dependencies ( lib-utf8proc utf8proc )

set_target_properties(lib-utf8proc PROPERTIES
	IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/utf8proc/libutf8proc.a
	INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/deps/utf8proc
)

################################################################################
# xxHash
################################################################################

add_custom_target ( xxHash
	COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/deps/xxHash
	COMMENT "Building xxHash"
)

add_dependencies ( falkordb
	rax
	onig
	xxHash
	lib-utf8proc
	lib-GraphBLAS
	lib-RediSearch
	libcypher-parser 
)

# Link with OpenSSL
target_link_libraries ( falkordb PRIVATE OpenSSL::SSL )

# Link with libcypher-parser
target_link_libraries ( falkordb PRIVATE libcypher-parser )

# Link with GraphBLAS
target_link_libraries ( falkordb PRIVATE lib-GraphBLAS )

# Link with RediSearch
target_link_libraries ( falkordb PRIVATE lib-RediSearch )

# Link with utf8proc
target_link_libraries ( falkordb PRIVATE lib-utf8proc )

# Set linker flags for static linking
target_link_options ( falkordb PUBLIC "-static" )
