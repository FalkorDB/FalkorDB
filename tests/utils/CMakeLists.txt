file(GLOB UTILS_SOURCES LIST_DIRECTORIES false *.c)

add_library(falkordb_test_utils STATIC ${UTILS_SOURCES})
set_target_properties(falkordb_test_utils PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(falkordb_test_utils PRIVATE falkordb ${FALKORDB_LIBS})
