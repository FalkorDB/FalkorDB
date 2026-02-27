#include "micro_benchmarks.h"

extern "C" {
#include "LAGraphX.h"
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
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

	GrB_OK(Delta_Matrix_setM(A, &A_M));
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
	Delta_Matrix A   = NULL;
	Delta_Matrix C   = NULL;
	GrB_Matrix C_M   = NULL;
	GrB_Matrix C_cpy = NULL;
	uint64_t   n     = 10000000;
	uint64_t   m     = 16;
	uint64_t   seed  = 870713428976ul;

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
		Delta_Matrix_setM(C, &C_M);
		C_M = NULL;
		state.ResumeTiming();

		for(int i = 0; i < 5; i++) {
			Delta_mxm(C, GxB_ANY_PAIR_BOOL, C, A);
		}
	}

	Delta_Matrix_free(&A);
	Delta_Matrix_free(&C);
	GrB_Matrix_free(&C_cpy);
}

FDB_BENCHMARK_ARGS(BM_mxm_all_V1);
FDB_BENCHMARK_ARGS(BM_mxm_chain_V1);
FDB_BENCHMARK_MAIN()
