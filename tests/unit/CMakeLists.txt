# Define the shared test utility files
set(TEST_UTILS
    ${CMAKE_CURRENT_SOURCE_DIR}/mock_log.c
    # Add other mock files here as needed
)

file(GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.c)

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test ${test_src} NAME_WE)
	add_executable(${test} ${TEST_UTILS} ${test_src})
	set_target_properties(${test} PROPERTIES LINKER_LANGUAGE CXX)
	if (NOT APPLE)
		target_link_libraries(${test} PRIVATE falkordb ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	else()
		target_link_libraries(${test} PRIVATE ${FALKORDB_OBJECTS} ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	endif()
endforeach()
