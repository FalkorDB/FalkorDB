
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

static void BM_delete_from_m(benchmark::State &state) {
    Delta_Matrix A     = NULL;
    uint64_t     n     = 500000;
    uint64_t     seed  = 870713428976ul;    
    GrB_Index    nvals = 0;

    Delta_Random_Matrix(&A, GrB_BOOL, n, 0.001, 0.0000001, 0.0000001, seed);

    GrB_Matrix m = DELTA_MATRIX_M(A);

    GrB_Matrix_nvals(&nvals, m);

    GrB_Index *i_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));
    GrB_Index *j_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));
    bool      *x_v = (bool *)     malloc(nvals * sizeof(bool));

    GrB_Matrix_extractTuples_BOOL(i_v, j_v, x_v, &nvals, m);
    for (auto _ : state) {
        for(int i = 0; i < 1; ++i) {
            uint64_t idx = state.iterations() * 1000 + i;
            if(x_v[idx]){
                Delta_Matrix_removeElement_BOOL(A, i_v[idx], j_v[idx]);
            }
        }
    }
    Delta_Matrix_free(&A);
    free(i_v);
    free(j_v);
    free(x_v);
}

BENCHMARK(BM_delete_from_m)->Setup(rg_setup)->Teardown(rg_teardown)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();