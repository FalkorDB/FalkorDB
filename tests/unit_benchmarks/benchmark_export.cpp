#include "tests/unit_benchmarks/create_random.h"
#include <benchmark/benchmark.h>

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

static void ArgGenerator(benchmark::internal::Benchmark* b) {
    std::vector<int> values = {0, 100, 10000};

    for (int x : values) {
        for (int y: values) {
            b->Args({x, y});
        }
    }
}

static void BM_export(benchmark::State &state) {
    Delta_Matrix A = NULL;
	GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = additions / ((double) n * (double) n);
    double del_density = deletions / ((double) n * (double) n); 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);

    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

    for (auto _ : state) {
		Delta_Matrix_export(&C, A, GrB_BOOL);
        GrB_Matrix_wait(C, GrB_MATERIALIZE);
        GrB_Matrix_free(&C);
    }

    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

    Delta_Matrix_free(&A);
}

static void BM_wait(benchmark::State &state) {
    Delta_Matrix A = NULL;
    Delta_Matrix C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = additions / ((double) n * (double) n);
    double del_density = deletions / ((double) n * (double) n); 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);


    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

    for (auto _ : state) {
		if((additions | deletions) == 0) state.SkipWithMessage("nothing to sync");
		state.PauseTiming();
		Delta_Matrix_free(&C);
		Delta_Matrix_dup(&C, A);
		state.ResumeTiming();

		Delta_Matrix_wait(C, true);
    }

    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	Delta_Matrix_free(&C);
    Delta_Matrix_free(&A);
}

// BENCHMARK(BM_export)->Setup(rg_setup)->Teardown(rg_teardown)
// 	->Unit(benchmark::kMillisecond)->Apply(ArgGenerator);
BENCHMARK(BM_wait)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMillisecond)->Apply(ArgGenerator);
BENCHMARK_MAIN();
