find_package(benchmark REQUIRED)

file(GLOB BENCH_SOURCES LIST_DIRECTORIES false benchmark_*.cpp)


foreach(bench_src ${BENCH_SOURCES})
	get_filename_component(bench ${bench_src} NAME_WE)
	add_executable(${bench} ${bench_src})
	set_target_properties(${bench} PROPERTIES LINKER_LANGUAGE CXX)
	if (NOT APPLE)
		target_link_libraries(${bench} PRIVATE falkordb ${FALKORDB_LIBS} benchmark::benchmark ${CMAKE_LD_LIBS})
	else()
		target_link_libraries(${bench} PRIVATE ${FALKORDB_OBJECTS} ${FALKORDB_LIBS} benchmark::benchmark ${CMAKE_LD_LIBS})
	endif()
endforeach()
