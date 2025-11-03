cmake_minimum_required(VERSION 3.13)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
include(${root}/deps/readies/cmake/main)

SET(CMAKE_EXPORT_COMPILE_COMMANDS on)

if (NOT DEFINED UNIT_TESTS)
	set(UNIT_TESTS OFF)
endif()

if (NOT DEFINED UNIT_BENCHMARKS)
	set(UNIT_BENCHMARKS OFF)
endif()

project(falkordb)

setup_cc_options()

file(GLOB_RECURSE SOURCES "src/*.c")

add_library(falkordb SHARED ${SOURCES})

include_directories(
	${root}
	${root}/src
	${root}/deps
	${root}/deps/rax
	${root}/deps/xxHash
	${root}/deps/utf8proc
	${root}/deps/oniguruma
	${root}/deps/RediSearch/src
	${root}/deps/LAGraph/include
	${root}/deps/GraphBLAS/Include
	${root}/deps/libcurl/include/curl
	${root}/deps/libcypher-parser/lib/src
	$ENV{LIBCYPHER_PARSER_BINDIR}/lib/src)

add_compile_definitions(
	_GNU_SOURCE
	REDIS_MODULE_TARGET
	REDISMODULE_EXPERIMENTAL_API
	XXH_STATIC_LINKING_ONLY)

if (DEBUG)
	add_compile_definitions(RG_DEBUG)
endif()

if (MEMCHECK)
	add_compile_definitions(MEMCHECK)
endif()

setup_shared_object_target(falkordb "")

set_target_properties(falkordb PROPERTIES LINKER_LANGUAGE CXX)

set(FALKORDB_OBJECTS $<TARGET_OBJECTS:falkordb>)

find_package(OpenSSL)

lists_from_env(GRAPHBLAS LAGRAPH LAGRAPHX LIBXXHASH LIBCURL LIBCSV RAX LIBCYPHER_PARSER REDISEARCH_LIBS UTF8PROC ONIGURUMA FalkorDBRS)
set(FALKORDB_LIBS ${FalkorDBRS} ${LAGRAPHX} ${LAGRAPH} ${GRAPHBLAS} ${LIBXXHASH} ${LIBCURL} ${LIBCSV} ${RAX} ${LIBCYPHER_PARSER} ${REDISEARCH_LIBS} ${UTF8PROC} ${ONIGURUMA} OpenSSL::SSL)

target_link_options(falkordb PRIVATE ${CMAKE_LD_FLAGS_LIST} ${CMAKE_SO_LD_FLAGS_LIST})
target_link_libraries(falkordb PRIVATE ${FALKORDB_LIBS} ${CMAKE_LD_LIBS} ${CoreFoundation_Library} ${SystemConfiguration_Library})

extract_debug_symbols(falkordb)

if (APPLE)
	add_library(falkordb_static STATIC $<TARGET_OBJECTS:falkordb>)
endif()

if (UNIT_TESTS)
	add_subdirectory(${root}/tests/unit tests/unit)
endif()

if (UNIT_BENCHMARKS)
    add_subdirectory(${root}/tests/unit_benchmarks tests/unit_benchmarks)
endif()