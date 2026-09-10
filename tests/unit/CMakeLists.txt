
file(GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.c)

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test ${test_src} NAME_WE)
	add_executable(${test} ${test_src})
	set_target_properties(${test} PROPERTIES LINKER_LANGUAGE CXX)

	# absolute path to the effects v3 conformance corpus, so the tests that read
	# it do not depend on the directory they are run from. inert in every other
	# test; the environment variable of the same name overrides it at runtime.
	target_compile_definitions(${test} PRIVATE
		EFFECTS_V3_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/effects_v3")
	if (NOT APPLE)
		target_link_libraries(${test} PRIVATE falkordb ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	else()
		target_link_libraries(${test} PRIVATE ${FALKORDB_OBJECTS} ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	endif()
endforeach()
