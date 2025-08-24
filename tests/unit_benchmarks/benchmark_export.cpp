#include "tests/unit_benchmarks/create_random.h"

void rg_setup(const benchmark::State &state) {
	// Initialize GraphBLAS.
	RedisModule_Alloc   = malloc;
	RedisModule_Realloc = realloc;
	RedisModule_Calloc  = calloc;
	RedisModule_Free    = free;
	RedisModule_Strdup  = strdup;
	LAGraph_Init(NULL);

	Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES,
			"100000", NULL);
    GrB_Global_set_INT32(GrB_GLOBAL, GxB_JIT_OFF, GxB_JIT_C_CONTROL);
    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // all matrices in CSR format
}

void rg_teardown(const benchmark::State &state) {
    GrB_finalize();
    // GrB_OK((GrB_Info) LAGraph_Finalize(NULL));
}

static void BM_export(benchmark::State &state) {
    Delta_Matrix A = NULL;
	GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, 1E-10, 1E-10, seed);

    for (auto _ : state) {
		Delta_Matrix_export(&C, A);
        GrB_Matrix_free(&C);
    }

    Delta_Matrix_free(&A);
}

BENCHMARK(BM_export)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
