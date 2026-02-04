#include "GraphBLAS.h"
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

	GrB_Global_set_INT32(GrB_GLOBAL, true, GxB_BURBLE);
	GrB_Index i = 0;
	for (auto _ : state) {
		ASSERT(i < n);
		Delta_Matrix_removeRow(A, i, desc);
		++i;
	}
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	Delta_Matrix_free(&A);
}

static void BM_Delta_row_delete_manual(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	GrB_Matrix   GBA  = NULL;
	GxB_Iterator it   = NULL;
	uint64_t     n    = 500000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	int    transpose   = state.range(2);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-4, add_density, del_density, seed);
	Delta_Matrix_cacheTranspose(A);

	// Export A to a standard GraphBLAS matrix to use its iterator
	Delta_Matrix_export(&GBA, A, GrB_BOOL);

	// Initialize the SuiteSparse iterator
	GxB_Iterator_new(&it);
	GxB_rowIterator_attach(it, GBA, NULL);
	GxB_rowIterator_seekRow(it, 0);

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	GrB_Index i = 0;
	for (auto _ : state) {
		ASSERT(i < n);

		GrB_Info info = GrB_SUCCESS;
		info = GxB_rowIterator_nextRow(it);

		while (info == GrB_SUCCESS) {
			int c = GxB_rowIterator_getColIndex (it);
			info = GxB_rowIterator_nextCol(it);

			Delta_Matrix_removeElement(A, i, c);
		}

		i++;
	}
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	// Cleanup
	GxB_Iterator_free(&it);
	GrB_Matrix_free(&GBA);
	Delta_Matrix_free(&A);
}

BENCHMARK(BM_Delta_row_delete)
    ->Unit(benchmark::kMicrosecond)
    ->Args({10000, 10000, false})
    ->Args({10000, 10000, true})
	->Iterations(2);
BENCHMARK(BM_Delta_row_delete_manual)
    ->Unit(benchmark::kMicrosecond)
    ->Args({10000, 10000, false})
    ->Args({10000, 10000, true})
	->Iterations(2);

FDB_BENCHMARK_MAIN()
