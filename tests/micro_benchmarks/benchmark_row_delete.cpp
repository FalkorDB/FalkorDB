#include "micro_benchmarks.h"

static void BM_Delta_row_delete(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	uint64_t     n    = 500000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	int    transpose   = state.range(2);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-4, add_density, del_density, seed);
	Delta_Matrix_cacheTranspose(A);
	GrB_Descriptor desc = transpose ? GrB_DESC_T0 : NULL;

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	GrB_Index i = 0;
	for (auto _ : state) {
		ASSERT(i < n);
		Delta_Matrix_removeRow(A, i, desc);
		++i;
	}
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	Delta_Matrix_free(&A);
}

static void BM_Delta_rows_delete(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	uint64_t     n    = 500000;
	uint64_t     seed = 870713428976ul;
	GrB_Vector   rows = NULL;

	int additions  = state.range(0);
	int deletions  = state.range(1);
	int batch_size = state.range(2);
	int transpose  = state.range(3);

	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-5, add_density, del_density, seed);
	Delta_Matrix_cacheTranspose(A);
	GrB_Descriptor desc = transpose ? GrB_DESC_T0 : NULL;

	// Set up rows vector to be [0, 1, ..., batch_size-1]
	GrB_OK(GrB_Vector_new(&rows, GrB_INT32, batch_size));
	GrB_OK(GrB_Vector_assign_INT32(rows, NULL, NULL, 0, GrB_ALL, 0, NULL));
	GrB_OK(GrB_Vector_apply_IndexOp_INT32(
	    rows, NULL, NULL, GrB_ROWINDEX_INT32, rows, 0, NULL));
	GrB_OK(GrB_Vector_resize(rows, n));

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	int i = 0;
	for (auto _ : state) {
		ASSERT((i += batch_size) <= n);
		// iterate to next row batch
		GrB_OK(GrB_Vector_apply_BinaryOp1st_INT32(
		    rows, NULL, NULL, GrB_PLUS_INT32, batch_size, rows, NULL));
		GrB_OK(Delta_Matrix_removeRows(A, rows, desc));
	}
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	Delta_Matrix_free(&A);
}

static void BM_Delta_row_delete_manual(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	uint64_t     n    = 500000;
	uint64_t     seed = 870713428976ul;

	int       additions   = state.range(0);
	int       deletions   = state.range(1);
	int       batch_size  = state.range(2);
	int       transpose   = state.range(3);
	double    add_density = additions / ((double)n * (double)n);
	double    del_density = deletions / ((double)n * (double)n);

	GrB_Index *tuples = new GrB_Index[1000*batch_size];
	uint64_t n_tups = 0;

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-5, add_density, del_density, seed);
	Delta_Matrix_cacheTranspose(A);
	Delta_MatrixTupleIter it = {0};
	GrB_OK(Delta_MatrixTupleIter_attach(&it, A));
	GrB_Info info = GrB_SUCCESS;

	// Initialize the SuiteSparse iterator
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	GrB_Index i = 0;
	for (auto _ : state) {
		ASSERT(i < n);
		info = Delta_MatrixTupleIter_iterate_range(&it, i, i + batch_size);
		i += batch_size;

		n_tups = 0;
		while (info == GrB_SUCCESS) {
			GrB_Index r, c;
			info = Delta_MatrixTupleIter_next_BOOL(&it, &r, &c, NULL);
			tuples[n_tups++] = r;
			tuples[n_tups++] = c;
		}
		ASSERT(info >= 0);

		for (int j = 0; j < n_tups; j += 2) {
			Delta_Matrix_removeElement(A, tuples[j], tuples[j + 1]);
		}
	}

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	// Cleanup
	Delta_Matrix_free(&A);
	delete[] tuples;
}

BENCHMARK(BM_Delta_row_delete)
    ->Unit(benchmark::kMicrosecond)
    ->Args({10000, 10000, false})
    ->Args({10000, 10000, true});
BENCHMARK(BM_Delta_row_delete_manual)
    ->Unit(benchmark::kMicrosecond)
    ->Args({10000, 10000, 10000, false})
    ->Args({10000, 10000, 10000, true});
BENCHMARK(BM_Delta_rows_delete)
    ->Unit(benchmark::kMicrosecond)
    ->Args({10000, 10000, 10000, false})
    ->Args({10000, 10000, 10000, true});
// ->Iterations(2);

FDB_BENCHMARK_MAIN()
