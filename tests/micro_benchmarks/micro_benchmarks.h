#pragma once

#include <benchmark/benchmark.h>

// Define C++ things
#define restrict
extern "C" {
	#include "RG.h"
	#include "rax.h"
	#include "LAGraph.h"
	#include "LAGraphX.h"
	#include "src/globals.h"
	#include "src/query_ctx.h"
	#include "src/graph/graphcontext.h"
	#include "src/graph/tensor/tensor.h"
	#include "src/configuration/config.h"
	#include "tests/utils/tensor_random.h"
	#include "src/graph/delta_matrix/delta_utils.h"
	#include "src/arithmetic/algebraic_expression.h"
}
#undef restrict

#define FDB_BENCHMARK_MAIN()                                                   \
	int main(int argc, char** argv) {                                          \
		RedisModule_Alloc   = malloc;                                          \
		RedisModule_Realloc = realloc;                                         \
		RedisModule_Calloc  = calloc;                                          \
		RedisModule_Free    = free;                                            \
		RedisModule_Strdup  = strdup;                                          \
		                                                                       \
		/* initialize LAGraph */                                               \
		if (LAGraph_Init(NULL) != GrB_SUCCESS) {                               \
			fprintf(stderr, "Fatal: LAGraph_Init failed\n");                   \
			return 1;                                                          \
		}                                                                      \
		                                                                       \
		Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES, "100000", NULL);   \
		GrB_Global_set_INT32(GrB_GLOBAL, GxB_JIT_OFF, GxB_JIT_C_CONTROL);      \
		GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);                   \
		GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);                         \
		Global_GrB_Ops_Init();                                                 \
		                                                                       \
		/* Google Benchmark */                                                 \
		::benchmark::Initialize(&argc, argv);                                  \
		if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;    \
		::benchmark::RunSpecifiedBenchmarks();                                 \
		::benchmark::Shutdown();                                               \
		                                                                       \
		/* teardown */                                                         \
		Global_GrB_Ops_Free();                                                 \
		LAGraph_Finalize(NULL);                                                \
		return 0;                                                              \
	}

