#include "micro_benchmarks.h"

extern "C" {
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
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
	double add_density = additions / ((double) n * (double) n);
	double del_density = deletions / ((double) n * (double) n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);
	Delta_Random_Matrix(&B, GrB_BOOL, n, 5E-7, add_density, del_density, seed+1);


	for (auto _ : state) {
		Delta_eWiseAdd(C, GxB_ANY_PAIR_BOOL, A, B);
	}

	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&C);
}

static void BM_add_chain(benchmark::State &state) {
	Delta_Matrix Cs[5];
	uint64_t     n     = 10000000;
	uint64_t     seed  = 870713428976ul;

	Delta_Matrix C = NULL;
	Delta_Matrix_new(&C, GrB_BOOL, n, n, false);

	int additions = state.range(0);
	int deletions = state.range(1);
	double add_density = additions / ((double) n * (double) n);
	double del_density = deletions / ((double) n * (double) n);

	for(int i = 0; i < 5; i++) {
		Delta_Random_Matrix(&Cs[i], GrB_BOOL, n, 5E-7, add_density, del_density, seed + 7 * i);
	}

	for (auto _ : state) {
		for(int i = 0; i < 5; i++) {
			Delta_eWiseAdd(C, GxB_ANY_PAIR_BOOL, C, Cs[i]);
		}
		Delta_Matrix_clear(C);
	}

	for(int i = 0; i < 5; i++) {
		Delta_Matrix_free(&Cs[i]);
	}
	Delta_Matrix_free(&C);
}

FDB_BENCHMARK_ARGS(BM_add_all);
FDB_BENCHMARK_ARGS(BM_add_chain);

FDB_BENCHMARK_MAIN()
