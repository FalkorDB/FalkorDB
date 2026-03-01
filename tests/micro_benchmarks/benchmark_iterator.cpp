#include "micro_benchmarks.h"

extern "C" {
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
}

static void BM_delta_iterator(benchmark::State &state) {
	Delta_Matrix      A  = NULL;
	Delta_MatrixTupleIter it;
	uint64_t          n  = 10000000;
	uint64_t          seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);
	Delta_MatrixTupleIter_attach(&it, A);

	for (auto _ : state) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it, NULL, NULL, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_free(&A);
}

static void BM_tensor_iterator(benchmark::State &state) {
	Tensor         A    = NULL;
	TensorIterator it;
	uint64_t       n    = 10000000;
	uint64_t       seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Random_Tensor(&A, n, 5E-7, add_density, del_density, seed);
	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	for (auto _ : state) {
		bool found = TensorIterator_next(&it, NULL, NULL, NULL, NULL);
		ASSERT(found);
	}

	Tensor_free(&A);
}

BENCHMARK(BM_delta_iterator)->Apply(ArgGenerator)->ArgNames({"adds", "dels"})->Unit(benchmark::kNanosecond)->Iterations(5000000);
BENCHMARK(BM_tensor_iterator)->Apply(ArgGenerator)->ArgNames({"adds", "dels"})->Unit(benchmark::kNanosecond)->Iterations(5000000);
FDB_BENCHMARK_MAIN()
