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

static void BM_add_all(benchmark::State &state) {
    Delta_Matrix A = NULL;
    Delta_Matrix B = NULL;
    Delta_Matrix C = NULL;
    uint64_t     n     = 500000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(Delta_Matrix_new(&C, GrB_BOOL, n, n, false));

    
    Delta_Random_Matrix(&A, GrB_BOOL, n, 0.001, 0.0000001, 0.0000001, seed);
    Delta_Random_Matrix(&B, GrB_BOOL, n, 0.001, 0.0000001, 0.0000001, seed + 7);


    for (auto _ : state) {
        Delta_eWiseAdd_OLD(C, GxB_ANY_PAIR_BOOL, A, B);
    }

    Delta_Matrix_free(&A);
    Delta_Matrix_free(&B);
    Delta_Matrix_free(&C);
}

static void BM_add_chain(benchmark::State &state) {
    Delta_Matrix Cs[5];
    uint64_t     n     = 50000;
    uint64_t     seed  = 870713428976ul;    

    for(int i = 0; i < 5; i++) {
        Delta_Random_Matrix(&Cs[i], GrB_BOOL, n, 0.01, 0.00001, 0.00001, seed + 7);
    }

    for (auto _ : state) {
        for(int i = 0; i < 4; i++) {
            Delta_eWiseAdd(Cs[i], GrB_LOR, Cs[i], Cs[i + 1]);
        }
    }

    for(int i = 0; i < 5; i++) {
        Delta_Matrix_free(&Cs[i]);
    }
}

BENCHMARK(BM_add_all)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_add_chain)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();