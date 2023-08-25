cmake_minimum_required (VERSION 3.5)

project (libcypher-parser C)

set (MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.cmake/Modules")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MODULE_DIR})

include (CheckCCompilerFlag)
include (CheckSymbolExists)
include (CheckCSourceCompiles)
include (GNUInstallDirs)
include (CMakeParseArguments)
include (CheckFunctionExists)
include (CheckIncludeFile)
include (Checks)
include (CrLinkLibraries)
include (ExtractMakeVariable)
include (GetAcInitVersion)
include (MsvcRuntime)

get_ac_init_version()

set (PACKAGE "libcypher-parser")
set (PACKAGE_MAJOR_VERSION ${MAJOR_VERSION})
set (PACKAGE_MINOR_VERSION ${MINOR_VERSION})
set (PACKAGE_PATCH_VERSION ${PATCH_VERSION})
set (VERSION "${PACKAGE_MAJOR_VERSION}.${PACKAGE_MINOR_VERSION}.${PACKAGE_PATCH_VERSION}")
set (PACKAGE_DEVELOPMENT_VERSION "devel")
set (PACKAGE_NAME "${PACKAGE}")
set (PACKAGE_TARNAME "${PACKAGE}")
set (PACKAGE_VERSION "${VERSION}")
set (PACKAGE_STATUS_VERSION 1)
set (PACKAGE_STRING "${PACKAGE} ${VERSION}")
set (PACKAGE_URL "https://github.com/cleishm/libcypher-parser")
set (PACKAGE_BUGREPORT "https://github.com/cleishm/libcypher-parser/issues")

set (CYPHER_PARSER_VERSION "${VERSION}")
set (CYPHER_PARSER_MAJOR_VERSION "${PACKAGE_MAJOR_VERSION}")
set (CYPHER_PARSER_MINOR_VERSION "${PACKAGE_MINOR_VERSION}")
set (CYPHER_PARSER_PATCH_VERSION "${PACKAGE_PATCH_VERSION}")
set (CYPHER_PARSER_DEVELOPMENT_VERSION "${PACKAGE_DEVELOPMENT_VERSION}")

set (configure_input "config.h - Generated from config.h.in by cmake.")
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/.cmake/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/lib/src/cypher-parser.h.in ${CMAKE_CURRENT_SOURCE_DIR}/lib/src/cypher-parser.h)

check_c_compiler_flag ("-fPIC" CC_HAVE_FPIC)
check_c_compiler_flag ("-fvisibility=hidden" CC_HAVE_VISIBILITY)

# Linker flags are ignored by check_c_compiler_flag, so we have
# to use this workaround
set (CMAKE_REQUIRED_FLAGS "-Wl,--exclude-libs=ALL")
check_c_compiler_flag ("" LINKER_HAS_EXCLUDE_LIBS)
unset (CMAKE_REQUIRED_FLAGS)

if (LINKER_HAS_EXCLUDE_LIBS)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--exclude-libs=ALL")
endif (LINKER_HAS_EXCLUDE_LIBS)

if (CC_HAVE_FPIC)
  set (PIC_C_FLAGS "-fPIC")
endif (CC_HAVE_FPIC)

if (CC_HAVE_VISIBILITY)
  set (VISI_C_FLAGS "-fvisibility=hidden")
endif (CC_HAVE_VISIBILITY)

find_package (check)
find_package (fmem)
find_package (leg)
find_package (getopt)

list (APPEND CMAKE_REQUIRED_DEFINITIONS
  -D_GNU_SOURCE
  -D_CRT_RAND_S
  -DVC_EXTRALEAN
  -DWIN32_LEAN_AND_MEAN)

extract_make_variable (lib/src/Makefile.am libcypher_parser_la_SOURCES SOURCES lib/src/)
add_library (libcypher-parser ${SOURCES})
target_compile_options (libcypher-parser PUBLIC /wd4267 /wd4244 /wd4047)

extract_make_variable (lib/test/Makefile.am check_libcypher_parser_CHECKS CHECKS lib/test/)
extract_make_variable (lib/test/Makefile.am check_libcypher_parser_SOURCES TESTS lib/test/)
list (REMOVE_ITEM TESTS "lib/test/\${check_libcypher_parser_CHECKS}")
list (APPEND TESTS ${CHECKS})

foreach (item IN LISTS CHECKS)
  string (REPLACE "check_" "" item ${item})
  get_filename_component(name ${item} NAME_WE)
  list (APPEND cases "TCase* ${name}_tcase(void)")
  list (APPEND add_cases "suite_add_tcase(s, ${name}_tcase())")
endforeach (item)

list (JOIN cases ";\n" cases)
list (JOIN add_cases ";\n    " add_cases)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/.cmake/check_libcypher-parser_suite.c.in 
                ${CMAKE_CURRENT_SOURCE_DIR}/lib/test/check_libcypher-parser_suite.c)

add_executable (tests ${TESTS})
target_link_libraries (tests libcypher-parser)

list (APPEND LINTER "linter/src/cypher-lint.c")
add_executable (cypher-linter ${LINTER})
target_link_libraries (cypher-linter libcypher-parser)

if (LEG_FOUND)
    add_custom_command (
        TARGET libcypher-parser
        PRE_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/src
        COMMAND ${LEG_PROGRAMS} parser.leg > parser_leg.c)
    add_custom_command (
        TARGET libcypher-parser
        PRE_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/src
        COMMAND ${LEG_PROGRAMS} quick_parser.leg > quick_parser_leg.c)
endif (LEG_FOUND)

if (CHECK_FOUND AND FMEM_FOUND)
  target_include_directories (tests PUBLIC ${FMEM_INCLUDE_DIRS} ${CHECK_INCLUDE_DIRS} ${CHECK_INCLUDE_DIRS}/..)
  target_link_libraries (tests ${CHECK_LIBRARIES} ${FMEM_LIBRARIES})
endif (CHECK_FOUND AND FMEM_FOUND)

if (GETOPT_FOUND)
  target_include_directories (cypher-linter PUBLIC ${GETOPT_INCLUDE_DIRS} "lib/src/")
  target_link_libraries (cypher-linter ${GETOPT_LIBRARIES})
endif (GETOPT_FOUND)

if (HAVE_PTHREADS)
  target_link_libraries (libcypher-parser ${CMAKE_THREAD_LIBS_INIT})
endif (HAVE_PTHREADS)

install (TARGETS libcypher-parser tests
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
