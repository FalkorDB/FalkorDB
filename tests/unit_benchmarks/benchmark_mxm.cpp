#include "RG.h"
#include <LAGraphX.h>
#include "GraphBLAS.h"
#include <benchmark/benchmark.h>
extern "C" {
#include "src/configuration/config.h"
#include "src/graph/delta_matrix/delta_utils.h"
}

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
    uint64_t     n     = 10000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, n, n));

    LAGraph_Random_Matrix(&A, GrB_BOOL, n, n, 0.01, seed, NULL);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 0.01, 0.00001, 0.00001, seed + 1);

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
    uint64_t     n     = 10000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, n, n));

    LAGraph_Random_Matrix(&A, GrB_BOOL, n, n, 0.01, seed, NULL);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 0.01, 0.00001, 0.00001, seed + 1);

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
    uint64_t     n     = 100000;
    uint64_t     m     = 10;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(GrB_Matrix_new(&C, GrB_BOOL, m, n));
    for(int i = 0; i < m; i++){
        GrB_Matrix_setElement_BOOL(C, i, i, true);
    }

    Delta_Random_Matrix(&A, GrB_BOOL, n, 0.01, 0.00001, 0.00001, seed);

    for (auto _ : state) {
        for(int i = 0; i < 5; i++) {
            Delta_mxm_identity(C, GrB_LOR_LAND_SEMIRING_BOOL, C, A);
        }
        GrB_Matrix_clear(C);
        for(int i = 0; i < m; i++){
            GrB_Matrix_setElement_BOOL(C, i, i, true);
        }
    }

    Delta_Matrix_free(&A);
    GrB_Matrix_free(&C);
}

BENCHMARK(BM_mxm_all_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_mxm_all_V2)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_mxm_chain_V1)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();