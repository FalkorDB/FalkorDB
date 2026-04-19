
file(GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.c)

# test_indexer_regression.c must NOT link against falkordb (embeds its own sync
# code) and requires GNU ld --wrap flags.  Exclude it from the standard loop.
list(FILTER TEST_SOURCES EXCLUDE REGEX "test_indexer_regression\\.c$")

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test ${test_src} NAME_WE)
	add_executable(${test} ${test_src})
	set_target_properties(${test} PROPERTIES LINKER_LANGUAGE CXX)
	if (NOT APPLE)
		target_link_libraries(${test} PRIVATE falkordb ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	else()
		target_link_libraries(${test} PRIVATE ${FALKORDB_OBJECTS} ${FALKORDB_LIBS} ${CMAKE_LD_LIBS})
	endif()
endforeach()

# Regression tests for C-9 (lost-wakeup) and C-10 (uninit pthread destroy).
# GNU ld --wrap intercepts pthread calls so we can inject spurious wakeups and
# force/track init failures without modifying the code under test.
# Only supported on Linux (--wrap is a GNU ld extension).
if (NOT APPLE)
	add_executable(test_indexer_regression test_indexer_regression.c alloc_stub.c)
	set_target_properties(test_indexer_regression PROPERTIES LINKER_LANGUAGE C)
	target_include_directories(test_indexer_regression PRIVATE
		${CMAKE_SOURCE_DIR}/src
		${CMAKE_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/deps/RedisModulesSDK
		${CMAKE_CURRENT_SOURCE_DIR}
	)
	target_link_options(test_indexer_regression PRIVATE
		-Wl,--wrap=pthread_cond_wait
		-Wl,--wrap=pthread_mutex_init
		-Wl,--wrap=pthread_mutex_destroy
		-Wl,--wrap=pthread_cond_destroy
		-Wl,--wrap=pthread_cond_init
		-Wl,--wrap=pthread_attr_init
		-Wl,--wrap=pthread_attr_destroy
	)
	target_link_libraries(test_indexer_regression PRIVATE pthread)
endif()
