cmake_minimum_required(VERSION 3.13)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

SET(CMAKE_EXPORT_COMPILE_COMMANDS on)

# Set CMake policies
if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Unit tests option
if(NOT DEFINED UNIT_TESTS)
    set(UNIT_TESTS OFF)
endif()

project(falkordb)

#-----------------------------------------------------------------------------
# Platform detection
#-----------------------------------------------------------------------------

if(NOT DEFINED OS)
    set(OS $ENV{OS})
endif()

if(NOT DEFINED OSNICK)
    set(OSNICK $ENV{OSNICK})
endif()

if(NOT DEFINED ARCH)
    set(ARCH $ENV{ARCH})
endif()

message("# OS=${OS}")
message("# OSNICK=${OSNICK}")
message("# ARCH=${ARCH}")

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

function(add_target_linker_flags TARGET FLAGS)
    get_target_property(LD_FLAGS ${TARGET} LINK_FLAGS)
    if(LD_FLAGS)
        set(FLAGS "${LD_FLAGS} ${FLAGS}")
    endif()
    set_target_properties(${TARGET} PROPERTIES LINK_FLAGS ${FLAGS})
endfunction()

function(setup_shared_object_target TARGET OUTPUT_NAME)
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
    set_target_properties(${TARGET} PROPERTIES SUFFIX ".so")

    if(APPLE)
        # macOS uses two-level namespace by default, which prevents symbol interposition
        target_link_options(${TARGET} PRIVATE "-undefined" "dynamic_lookup")
    else()
        target_link_options(${TARGET} PRIVATE "-Wl,--enable-new-dtags" "-Wl,-rpath,\$ORIGIN")
    endif()

    if("${OUTPUT_NAME}" STREQUAL "")
        set(OUTPUT_NAME "${TARGET}")
    endif()
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "${OUTPUT_NAME}")
endfunction()

#-----------------------------------------------------------------------------
# Compiler and linker flags from environment
#-----------------------------------------------------------------------------

# Get compiler flags from environment (as strings)
set(CMAKE_CC_FLAGS "$ENV{CMAKE_CC_FLAGS}")
set(CMAKE_CC_C_FLAGS "$ENV{CMAKE_CC_C_FLAGS}")
set(CMAKE_CC_CXX_FLAGS "$ENV{CMAKE_CC_CXX_FLAGS}")
set(CMAKE_LD_FLAGS "$ENV{CMAKE_LD_FLAGS}")
set(CMAKE_SO_LD_FLAGS "$ENV{CMAKE_SO_LD_FLAGS}")
set(CMAKE_EXE_LD_FLAGS "$ENV{CMAKE_EXE_LD_FLAGS}")

# Get linker flags as lists (convert space-separated env vars to CMake lists)
string(REPLACE " " ";" CMAKE_LD_FLAGS_LIST "$ENV{CMAKE_LD_FLAGS_LIST}")
string(REPLACE " " ";" CMAKE_SO_LD_FLAGS_LIST "$ENV{CMAKE_SO_LD_FLAGS_LIST}")
string(REPLACE " " ";" CMAKE_EXE_LD_FLAGS_LIST "$ENV{CMAKE_EXE_LD_FLAGS_LIST}")
string(REPLACE " " ";" CMAKE_LD_LIBS "$ENV{CMAKE_LD_LIBS}")

if("$ENV{VERBOSE}" STREQUAL "1")
    message("# CC=$ENV{CC}")
    message("# CXX=$ENV{CXX}")
    message("# CMAKE_CC_FLAGS=${CMAKE_CC_FLAGS}")
    message("# CMAKE_CC_C_FLAGS=${CMAKE_CC_C_FLAGS}")
    message("# CMAKE_CC_CXX_FLAGS=${CMAKE_CC_CXX_FLAGS}")
    message("# CMAKE_LD_FLAGS=${CMAKE_LD_FLAGS}")
    message("# CMAKE_EXE_LD_FLAGS=${CMAKE_EXE_LD_FLAGS}")
    message("# CMAKE_SO_LD_FLAGS=${CMAKE_SO_LD_FLAGS}")
    message("# CMAKE_LD_LIBS=${CMAKE_LD_LIBS}")
endif()

if(DEFINED ENV{SAN})
    set(SANITIZER $ENV{SAN})
endif()

#-----------------------------------------------------------------------------
# Setup compiler options
#-----------------------------------------------------------------------------

message("# CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CC_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CC_CXX_FLAGS}")

# Add compile options from CMAKE_CC_FLAGS
if(CMAKE_CC_FLAGS)
    string(REPLACE " " ";" CMAKE_CC_FLAGS_LIST ${CMAKE_CC_FLAGS})
    add_compile_options(${CMAKE_CC_FLAGS_LIST})
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_LD_FLAGS} ${CMAKE_EXE_LD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_LD_FLAGS} ${CMAKE_SO_LD_FLAGS}")

# -Bsymbolic: Resolve internal references to global symbols locally, preventing
# symbol interposition from the host process (Redis). This is critical because
# RediSearch has its own dict implementation with the same symbol names as Redis's
# internal dict (dictCreate, dictAdd, etc.), but with a different dictType structure.
# Without -Bsymbolic, RediSearch's code would call Redis's dict functions instead
# of its own, causing crashes due to ABI incompatibility (Redis 8.x dictType has
# more fields than RediSearch's dictType).
if(NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")
endif()

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)

# Enable OpenMP if available
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Workaround for RediSearch macro compatibility with newer Clang
# Disable warnings for GNU zero-variadic-macro-arguments completely
# Using -Wno-vla-extension instead of -Wno-vla-cxx-extension for compatibility with Clang < 18
if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments -Wno-error=vla-extension")
endif()

# Remove NDEBUG to match RediSearch's RS_LOG_ASSERT macro signature (expects 2 args without NDEBUG)
# The RS_LOG_ASSERT macro in RediSearch has different signatures depending on NDEBUG:
#   - With NDEBUG: RS_LOG_ASSERT(ctx, condition, fmt, ...) - 4 arguments
#   - Without NDEBUG: RS_LOG_ASSERT(condition, str) - 2 arguments
# The code uses the 2-argument version, so we must ensure NDEBUG is not defined.
# Using -UNDEBUG explicitly undefines it, overriding any -DNDEBUG from CMake defaults.
add_compile_options(-UNDEBUG)

#-----------------------------------------------------------------------------
# Build RediSearch as subdirectory
#-----------------------------------------------------------------------------

# Enable Position Independent Code for all targets - required for shared library on ARM64 Linux
# This must be set before adding RediSearch subdirectory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set RediSearch build options - use static library without coordinator
set(BUILD_STATIC ON CACHE BOOL "Build RediSearch as static library" FORCE)
set(BUILD_COORDINATOR OFF CACHE BOOL "Disable coordinator" FORCE)
set(BUILD_SEARCH_UNIT_TESTS OFF CACHE BOOL "Don't build RediSearch unit tests" FORCE)

# RediSearch expects BINROOT to be set
if(NOT DEFINED BINROOT)
    set(BINROOT ${CMAKE_BINARY_DIR})
endif()

# Add RediSearch subdirectory
add_subdirectory(${root}/deps/RediSearch ${CMAKE_BINARY_DIR}/search-static)

#-----------------------------------------------------------------------------
# Source files and library
#-----------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES "src/*.c")

add_library(falkordb SHARED ${SOURCES})

include_directories(
    ${root}
    ${root}/src
    ${root}/deps
    ${root}/deps/rax
    ${root}/deps/xxHash
    ${root}/deps/quickjs
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

if(DEBUG)
    add_compile_definitions(RG_DEBUG)
endif()

if(MEMCHECK)
    add_compile_definitions(MEMCHECK)
endif()

setup_shared_object_target(falkordb "")

set_target_properties(falkordb PROPERTIES LINKER_LANGUAGE CXX)

set(FALKORDB_OBJECTS $<TARGET_OBJECTS:falkordb>)

find_package(OpenSSL)

# Find macOS frameworks if needed
if(APPLE)
    find_library(CoreFoundation_Library CoreFoundation)
    find_library(SystemConfiguration_Library SystemConfiguration)
endif()

# Get external library paths from environment
set(GRAPHBLAS $ENV{GRAPHBLAS})
set(LAGRAPH $ENV{LAGRAPH})
set(LAGRAPHX $ENV{LAGRAPHX})
set(QUICKJS $ENV{QUICKJS})
set(LIBXXHASH $ENV{LIBXXHASH})
set(LIBCURL $ENV{LIBCURL})
set(LIBCSV $ENV{LIBCSV})
set(RAX $ENV{RAX})
set(LIBCYPHER_PARSER $ENV{LIBCYPHER_PARSER})
set(UTF8PROC $ENV{UTF8PROC})
set(ONIGURUMA $ENV{ONIGURUMA})
set(FalkorDBRS $ENV{FalkorDBRS})

# Build library list - redisearch-static and VectorSimilarity are CMake targets
# Note: VectorSimilarity must be explicitly linked because redisearch-static (a static library)
# doesn't transitively link its dependencies. Only the shared redisearch library links VectorSimilarity,
# but we can't use that because it would make falkordb.so depend on redisearch.so at runtime.
set(FALKORDB_LIBS
    ${FalkorDBRS}
    ${LAGRAPHX}
    ${LAGRAPH}
    ${QUICKJS}
    ${GRAPHBLAS}
    ${LIBXXHASH}
    ${LIBCURL}
    ${LIBCSV}
    ${RAX}
    ${LIBCYPHER_PARSER}
    ${UTF8PROC}
    ${ONIGURUMA}
    redisearch-static
    VectorSimilarity
    OpenSSL::SSL
    ${CMAKE_DL_LIBS}
)

if(APPLE)
    list(APPEND FALKORDB_LIBS ${CoreFoundation_Library} ${SystemConfiguration_Library})
endif()

# Ensure RediSearch is built before falkordb
add_dependencies(falkordb redisearch-static)

target_link_options(falkordb PRIVATE ${CMAKE_LD_FLAGS_LIST} ${CMAKE_SO_LD_FLAGS_LIST})
target_link_libraries(falkordb PRIVATE ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})

if(APPLE)
    add_library(falkordb_static STATIC $<TARGET_OBJECTS:falkordb>)
endif()

if(UNIT_TESTS)
    add_subdirectory(${root}/tests/unit tests/unit)
endif()
