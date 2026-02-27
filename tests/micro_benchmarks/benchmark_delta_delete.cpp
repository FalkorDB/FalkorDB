#include "micro_benchmarks.h"

extern "C" {
#include "src/graph/delta_matrix/delta_matrix.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "tests/utils/tensor_random.h"
}

// Delete all edges from a Delta_Matrix, iterating over every element present
// at construction time (captured via export). Measures average deletion cost
// regardless of whether the element lives in M or DP.
static void BM_delete(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	GrB_Matrix   snap = NULL;
	uint64_t     n    = 50000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 0.0001, add_density, del_density, seed);

	// Snapshot all currently-present elements before we start deleting
	Delta_Matrix_export(&snap, A, GrB_BOOL);

	GrB_Index nvals = 0;
	GrB_Matrix_nvals(&nvals, snap);

	GrB_Index* i_v = (GrB_Index*)malloc(nvals * sizeof(GrB_Index));
	GrB_Index* j_v = (GrB_Index*)malloc(nvals * sizeof(GrB_Index));
	GrB_OK (GrB_Matrix_extractTuples_BOOL(i_v, j_v, NULL, &nvals, snap));
	GrB_OK (GrB_Matrix_free(&snap));

	GrB_Index idx = 0;
	for (auto _ : state) {
		if (idx >= nvals)
			break;
		GrB_OK (Delta_Matrix_removeElement(A, i_v[idx], j_v[idx]));
		idx++;
	}

	Delta_Matrix_free(&A);
	free(i_v);
	free(j_v);
}

static void BM_tensor_delete(benchmark::State& state) {
	Tensor         A = NULL;
	TensorIterator it;
	uint64_t       n    = 500000;
	uint64_t       seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Random_Tensor(&A, n, 0.001, add_density, del_density, seed);

	Edge arr[40000];
	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	for (int i = 0; i < 40000; ++i) {
		bool found = TensorIterator_next(
		    &it, &arr[i].src_id, &arr[i].dest_id, &arr[i].id, NULL);
		ASSERT(found);
	}

	int j = 0;
	for (auto _ : state) {
		ASSERT(j < 40000);
		Tensor_RemoveElements(A, arr + j, 1, NULL);
		j++;
	}
	Tensor_free(&A);
}

static void BM_tensor_delete_batch(benchmark::State& state) {
	Tensor         A = NULL;
	TensorIterator it;
	uint64_t       n    = 500000;
	uint64_t       seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Random_Tensor(&A, n, 0.001, add_density, del_density, seed);

	Edge arr[40000];
	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	for (int i = 0; i < 40000; ++i) {
		bool found = TensorIterator_next(
		    &it, &arr[i].src_id, &arr[i].dest_id, &arr[i].id, NULL);
		ASSERT(found);
	}

	int batch_size = 10000;
	int j          = 0;
	for (auto _ : state) {
		ASSERT(j + batch_size <= 40000);
		Tensor_RemoveElements(A, arr + j, batch_size, NULL);
		j += batch_size;
	}

	Tensor_free(&A);
}

BENCHMARK(BM_delete)
    ->Apply(ArgGenerator)
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(40000);
BENCHMARK(BM_tensor_delete)
    ->Apply(ArgGenerator)
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(40000);
BENCHMARK(BM_tensor_delete_batch)
    ->Apply(ArgGenerator)
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(4);
FDB_BENCHMARK_MAIN()
