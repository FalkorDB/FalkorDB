
file(GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.c)

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test ${test_src} NAME_WE)
	add_executable(${test} ${test_src})
	set_target_properties(${test} PROPERTIES LINKER_LANGUAGE CXX)
	# Match the rmalloc.h dual-guard contract enforced for the falkordb
	# target: define __REDISEARCH_ALLOC__ so RediSearch's rmalloc.h skips
	# its body, and force FalkorDB's rmalloc.h to be included first.
	target_compile_definitions(${test} PRIVATE __REDISEARCH_ALLOC__)
	target_compile_options(${test} PRIVATE -include "${root}/src/util/rmalloc.h")
	if (NOT APPLE)
		target_link_libraries(${test} PRIVATE falkordb ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	else()
		target_link_libraries(${test} PRIVATE ${FALKORDB_OBJECTS} ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	endif()
endforeach()
