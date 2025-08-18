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
    GrB_Matrix   A = NULL;
    Delta_Matrix B = NULL;
    GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, n, n));

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    LAGraph_Random_Matrix(&A, GrB_BOOL, n, n, 5E-7, seed, NULL);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 5E-7, add_density, del_density, seed + 1);
    Delta_Matrix_wait(B, false);

    for (auto _ : state) {
        Delta_mxm_identity(C, GrB_LOR_LAND_SEMIRING_BOOL, A, B);
    }

    GrB_Matrix_free(&A);
    Delta_Matrix_free(&B);
    GrB_Matrix_free(&C);
}

static void BM_mxm_all_V2(benchmark::State &state) {
    GrB_Matrix   A = NULL;
    Delta_Matrix B = NULL;
    GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, n, n));

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    LAGraph_Random_Matrix(&A, GrB_BOOL, n, n, 5E-7, seed, NULL);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 5E-7, add_density, del_density, seed + 1);
    Delta_Matrix_wait(B, false);

    for (auto _ : state) {
        Delta_mxm_count(C, GxB_PLUS_PAIR_UINT64, A, B);
    }

    GrB_Matrix_free(&A);
    Delta_Matrix_free(&B);
    GrB_Matrix_free(&C);
}

// simulate matching 
static void BM_mxm_chain_V1(benchmark::State &state) {
    Delta_Matrix A = NULL;
    GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     m     = 16;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, m, n));
    for(int i = 0; i < m; i++){
        GrB_Matrix_setElement_BOOL(C, i, i, true);
    }

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);

    for (auto _ : state) {
        for(int i = 0; i < 5; i++) {
            Delta_mxm_identity(C, GrB_LOR_LAND_SEMIRING_BOOL, C, A);
        }

        // clean up
        state.PauseTiming();
        GrB_Matrix_clear(C);
        for(int i = 0; i < m; i++){
            GrB_Matrix_setElement_BOOL(C, i, i, true);
        }
    }

    Delta_Matrix_free(&A);
    GrB_Matrix_free(&C);
}

// simulate matching 
static void BM_mxm_chain_V2(benchmark::State &state) {
    Delta_Matrix A = NULL;
    GrB_Matrix   C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     m     = 16;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, m, n));
    for(int i = 0; i < m; i++){
        GrB_Matrix_setElement_BOOL(C, i, i, true);
    }

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);

    for (auto _ : state) {
        for(int i = 0; i < 5; i++) {
            Delta_mxm_count(C, GxB_PLUS_PAIR_UINT64, C, A);
        }

        // clean up
        state.PauseTiming();
        GrB_Matrix_clear(C);
        for(int i = 0; i < m; i++){
            GrB_Matrix_setElement_BOOL(C, i, i, true);
        }
    }

    Delta_Matrix_free(&A);
    GrB_Matrix_free(&C);
}

BENCHMARK(BM_mxm_all_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
BENCHMARK(BM_mxm_all_V2)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
BENCHMARK(BM_mxm_chain_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
BENCHMARK(BM_mxm_chain_V2)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
BENCHMARK_MAIN();
