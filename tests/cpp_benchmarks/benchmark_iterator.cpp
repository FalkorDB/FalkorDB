#include "tests/cpp_benchmarks/create_random.h"

void rg_setup(const benchmark::State &state) {
	static bool allocators_initialized = false;
	if (!allocators_initialized) {
		RedisModule_Alloc   = malloc;
		RedisModule_Realloc = realloc;
		RedisModule_Calloc  = calloc;
		RedisModule_Free    = free;
		RedisModule_Strdup  = strdup;
		allocators_initialized = true;
	}

	printf("Starting benchmark ...\n");
	LAGraph_Init(NULL);

	Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES, "100000", NULL);
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
	Global_GrB_Ops_Init();
}

void rg_teardown(const benchmark::State &state) {
	Global_GrB_Ops_Free();
	GrB_finalize();
	// GrB_OK((GrB_Info) LAGraph_Finalize(NULL));
}

static void BM_delta_iterator(benchmark::State &state) {
	Delta_Matrix   A     = NULL;
	Delta_MatrixTupleIter it;
	uint64_t       n     = 10000000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, 1E-10, 1E-10, seed);
	Delta_MatrixTupleIter_attach(&it, A);

	for (auto _ : state) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it, NULL, NULL, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_free(&A);
}

static void BM_tensor_iterator(benchmark::State &state) {
	Tensor         A     = NULL;
	TensorIterator it;
	uint64_t       n     = 10000000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Random_Tensor(&A, n, 5E-7, 1E-10, 1E-10, seed);
	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	for (auto _ : state) {
		bool found = TensorIterator_next(&it, NULL, NULL, NULL, NULL);
		ASSERT(found);
	}

	Tensor_free(&A);
}

BENCHMARK(BM_delta_iterator)->Setup(rg_setup)->Teardown(rg_teardown)
	->Iterations(5000000);
BENCHMARK(BM_tensor_iterator)->Setup(rg_setup)->Teardown(rg_teardown)
	->Iterations(5000000);
BENCHMARK_MAIN();
