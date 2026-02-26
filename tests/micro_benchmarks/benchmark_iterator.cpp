#include "micro_benchmarks.h"

extern "C" {
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
}

static void BM_delta_iterator(benchmark::State &state) {
	Delta_Matrix   A     = NULL;
	Delta_MatrixTupleIter it;
	uint64_t       n     = 10000000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, 1E-10, 1E-10, seed);
	Delta_MatrixTupleIter_attach(&it, A);

	for (auto _ : state) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it, NULL, NULL, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_free(&A);
}

static void BM_tensor_iterator(benchmark::State &state) {
	Tensor         A     = NULL;
	TensorIterator it;
	uint64_t       n     = 10000000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Random_Tensor(&A, n, 5E-7, 1E-10, 1E-10, seed);
	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	for (auto _ : state) {
		bool found = TensorIterator_next(&it, NULL, NULL, NULL, NULL);
		ASSERT(found);
	}

	Tensor_free(&A);
}

BENCHMARK(BM_delta_iterator)->Iterations(5000000);
BENCHMARK(BM_tensor_iterator)->Iterations(5000000);
FDB_BENCHMARK_MAIN()
