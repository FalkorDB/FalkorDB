#include "tests/unit_benchmarks/create_random.h"

void rg_setup(const benchmark::State &state) {
	// Initialize GraphBLAS.
    printf("Starting union benchmark ...\n");
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
    printf("Ending union benchmark ...\n");
    GrB_finalize();
    // GrB_OK((GrB_Info) LAGraph_Finalize(NULL));
}

static void BM_union_all(benchmark::State &state) {
    Delta_Matrix A = NULL;
    Delta_Matrix B = NULL;
    Delta_Matrix C = NULL;
    GrB_Scalar   a = NULL;
    uint64_t     n     = 500000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(Delta_Matrix_new(&C, GrB_BOOL, n, n, false));
    GrB_OK(GrB_Scalar_new(&a, GrB_UINT64));
    GrB_OK(GrB_Scalar_setElement_BOOL(a, U64_ZOMBIE));
    
    Delta_Random_Matrix(&A, GrB_UINT64, n, 0.001, 0.0000001, 0.0000001, seed);
    Delta_Random_Matrix(&B, GrB_UINT64, n, 0.001, 0.0000001, 0.0000001, seed + 7);

    for (auto _ : state) {
        Delta_eWiseUnion(C, GrB_LT_UINT64, A, a, B, a);
    }

    GrB_Scalar_free(&a);
    Delta_Matrix_free(&A);
    Delta_Matrix_free(&B);
    Delta_Matrix_free(&C);
}

static void BM_union_chain(benchmark::State &state) {
    Delta_Matrix C = NULL;
    Delta_Matrix As[5];
    GrB_Scalar   a = NULL;
    GrB_Scalar   b = NULL;
    uint64_t     n     = 50000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(Delta_Matrix_new(&C, GrB_BOOL, n, n, false));
    GrB_OK(GrB_Scalar_new(&a, GrB_UINT64));
    GrB_OK(GrB_Scalar_setElement_BOOL(a, U64_ZOMBIE));
    GrB_OK(GrB_Scalar_new(&b, GrB_BOOL));
    GrB_OK(GrB_Scalar_setElement_BOOL(b, BOOL_ZOMBIE));


    for(int i = 0; i < 5; i++) {
        Delta_Random_Matrix(&As[i], GrB_BOOL, n, 0.01, 0.00001, 0.00001, seed + 7 * i);
    }

    for (auto _ : state) {
        for(int i = 0; i < 5; i++) {
            Delta_eWiseUnion(C, GrB_LOR, C, a, As[i], b);
        }
        Delta_Matrix_clear(C);
    }

    for(int i = 0; i < 5; i++) {
        Delta_Matrix_free(&As[i]);
    }
    Delta_Matrix_free(&C);
    GrB_Scalar_free(&a);
}

BENCHMARK(BM_union_all)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_union_chain)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();