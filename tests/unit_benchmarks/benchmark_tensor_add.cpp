#include "tests/unit_benchmarks/create_random.h"
#include <vector>
#include <algorithm>

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
    Global_GrB_Ops_Init();
}

void rg_teardown(const benchmark::State &state) {
    Global_GrB_Ops_Free();
    GrB_finalize();
    // GrB_OK((GrB_Info) LAGraph_Finalize(NULL));
}

static void BM_single_add(benchmark::State &state) {
    Tensor         A     = NULL;
    uint64_t       n     = 100;
    Delta_Matrix_new(&A, GrB_UINT64, n, n, false);

    uint64_t i = 0;
    for (auto _ : state) {
        Tensor_SetElement(A, rand() % 100, rand() % 100, i);
        ++ i;
    }
    Tensor_free(&A);
}

static void BM_batch_add(benchmark::State &state) {
    Tensor         A     = NULL;
    uint64_t       n     = 100;
    Delta_Matrix_new(&A, GrB_UINT64, n, n, false);
    std::vector<uint64_t> batch(20000);
    std::array<uint64_t, 10000> batch_id;

    uint64_t i = 0;
    for (auto _ : state) {
        state.PauseTiming();
        for(uint64_t &n: batch)
            n = rand() % 100;
        for(uint64_t &id: batch_id)
            id = i++;
        std::sort(batch.begin(), batch.end());
        state.ResumeTiming();
        
        Tensor_SetElements(A, batch.data(), batch.data() + 10000, 
            batch_id.data(), 10000);
    }
    Tensor_free(&A);
}

BENCHMARK(BM_single_add)->Setup(rg_setup)->Teardown(rg_teardown);
BENCHMARK(BM_batch_add)->Setup(rg_setup)->Teardown(rg_teardown);
BENCHMARK_MAIN();
