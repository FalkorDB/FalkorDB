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
    Global_GrB_Ops_Init();
}

void rg_teardown(const benchmark::State &state) {
    GrB_finalize();
    Global_GrB_Ops_Free();
    // GrB_OK((GrB_Info) LAGraph_Finalize(NULL));
}

static void BM_add_all(benchmark::State &state) {
    Delta_Matrix A = NULL;
    Delta_Matrix B = NULL;
    Delta_Matrix C = NULL;
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;    

    GrB_OK(Delta_Matrix_new(&C, GrB_BOOL, n, n, false));

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = 1e-14 * additions;
    double del_density = 1e-14 * deletions; 

    Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed) ;
    Delta_Random_Matrix(&B, GrB_BOOL, n, 5E-7, add_density, del_density, seed+1) ;


    for (auto _ : state) {
        Delta_add(C, A, B);
    }

    Delta_Matrix_free(&A);
    Delta_Matrix_free(&B);
    Delta_Matrix_free(&C);
}

static void BM_add_chain(benchmark::State &state) {
    Delta_Matrix Cs[5];
    uint64_t     n     = 10000000;
    uint64_t     seed  = 870713428976ul;    
    Delta_Matrix C     = NULL;

    int additions = state.range(0);
    int deletions = state.range(1);
    double add_density = additions / ((double) n * (double) n);
    double del_density = deletions / ((double) n * (double) n); 

    for(int i = 0; i < 5; i++) {
        Delta_Random_Matrix(&Cs[i], GrB_BOOL, n, 5E-7, add_density, del_density, seed + 7 * i);
    }

    for (auto _ : state) {
        state.PauseTiming();
        Delta_Matrix_free(&C);
        Delta_Matrix_new(&C, GrB_BOOL, n, n, false);
        state.ResumeTiming();

        for(int i = 0; i < 5; i++) {
            Delta_add(C, C, Cs[i]);
        }
    }

    for(int i = 0; i < 5; i++) {
        Delta_Matrix_free(&Cs[i]);
    }
    Delta_Matrix_free(&C);
}

BENCHMARK(BM_add_all)->Setup(rg_setup)->Teardown(rg_teardown)
    // ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
    ->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
    ->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
    ->Args({100, 0});
BENCHMARK(BM_add_chain)->Setup(rg_setup)->Teardown(rg_teardown)
    // ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
    ->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
    ->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
    ->Args({100, 0});
BENCHMARK_MAIN();
