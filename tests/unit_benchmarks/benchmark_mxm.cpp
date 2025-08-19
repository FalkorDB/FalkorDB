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

static void BM_mxm_all_V1(benchmark::State &state) {
    GrB_Matrix   A_M   = NULL;
    Delta_Matrix A     = NULL;
    Delta_Matrix B     = NULL;
    Delta_Matrix C     = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(Delta_Matrix_new(&A, GrB_BOOL, n, n, false));
    GrB_OK(Delta_Matrix_new(&C, GrB_BOOL, n, n, false));

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    LAGraph_Random_Matrix(&A_M, GrB_BOOL, n, n, 5E-7, seed, NULL);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 5E-7, add_density, del_density, seed + 1);

    GrB_OK(Delta_Matrix_setM(A, A_M));
    A_M = NULL;
    Delta_Matrix_wait(B, false);

    for (auto _ : state) {
        Delta_mxm(C, GxB_ANY_PAIR_BOOL, A, B);
    }

    Delta_Matrix_free(&A);
    Delta_Matrix_free(&B);
    Delta_Matrix_free(&C);
}

// simulate matching 
static void BM_mxm_chain_V1(benchmark::State &state) {
    Delta_Matrix A     = NULL;
    Delta_Matrix C     = NULL;
    GrB_Matrix   C_M   = NULL;
    GrB_Matrix   C_cpy = NULL;
    uint64_t     n     = 10000000;
    uint64_t     m     = 16;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C_cpy, GrB_BOOL, m, n));
    for(int i = 0; i < m; i++){
        GrB_Matrix_setElement_BOOL(C_cpy, true, i, i);
    }
    GrB_OK(GrB_Matrix_wait(C_cpy, GrB_MATERIALIZE));
    GrB_OK (Delta_Matrix_new(&C, GrB_BOOL, m, n, false));

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = additions / ((double) n * (double) n);
    double del_density = deletions / ((double) n * (double) n); 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);
    GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
    // Delta_Matrix_print(A, GxB_SUMMARY);

    for (auto _ : state) {
        state.PauseTiming();
        GrB_OK(GrB_Matrix_dup(&C_M, C_cpy));
        Delta_Matrix_clear(C);
        Delta_Matrix_setM(C, C_M);
        C_M = NULL;
        state.ResumeTiming();

        for(int i = 0; i < 5; i++) {
            Delta_mxm(C, GxB_ANY_PAIR_BOOL, C, A);
        }
    }

    Delta_Matrix_free(&A);
    Delta_Matrix_free(&C);
}

BENCHMARK(BM_mxm_all_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    // ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
    ->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
    ->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
    ->Args({100, 0});
BENCHMARK(BM_mxm_chain_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    // ->Unit(benchmark::kMicrosecond)->Args({10000, 10000})->Threads(1);
    ->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
    ->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
    ->Args({100, 0});
BENCHMARK_MAIN();
