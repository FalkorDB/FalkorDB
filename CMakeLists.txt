cmake_minimum_required ( VERSION 3.20 )

project ( falkordb )

include_directories ( "deps/rax" "deps/xxHash" "deps/GraphBLAS/Include" "src" )

file ( GLOB_RECURSE SOURCES "src/*.c" )

include(ExternalProject)

# libcypher-parser
ExternalProject_Add ( libcypher-parser
	PREFIX ${CMAKE_BINARY_DIR}/libcypher-parser
	DOWNLOAD_COMMAND ""  # Skip download
    CONFIGURE_COMMAND ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/autogen.sh && ${CMAKE_SOURCE_DIR}/deps/libcypher-parser/configure
    BUILD_COMMAND make CFLAGS="-Wno-error=ignored-qualifiers"
)

# GraphBLAS

add_library ( falkordb SHARED ${SOURCES} )
