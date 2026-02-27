#include "micro_benchmarks.h"

extern "C" {
#include "src/graph/delta_matrix/delta_matrix.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/util/simple_rand.h"
#include "tests/utils/tensor_random.h"
}

// Insert a random element into a Delta_Matrix — may hit M or DP depending on
// whether the coordinate already exists. Measures average insertion cost.
static void BM_insert_new(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	uint64_t     n    = 500000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 0.0001, add_density, del_density, seed);

	uint64_t rng = seed + 1;
	for (auto _ : state) {
		GrB_Index i = simple_rand(&rng) % n;
		GrB_Index j = simple_rand(&rng) % n;
		Delta_Matrix_setElement_BOOL(A, i, j);
	}

	Delta_Matrix_free(&A);
}

static void BM_tensor_insert_existing(benchmark::State& state) {
	Tensor         A = NULL;
	TensorIterator it;
	uint64_t       n    = 500000;
	uint64_t       seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Random_Tensor(&A, n, 0.0001, add_density, del_density, seed);

	// Collect existing edges to re-insert
	const GrB_Index max_candidates = 50000;
	GrB_Index* rows  = (GrB_Index*)malloc(max_candidates * sizeof(GrB_Index));
	GrB_Index* cols  = (GrB_Index*)malloc(max_candidates * sizeof(GrB_Index));
	uint64_t*  vals  = (uint64_t*)malloc(max_candidates * sizeof(uint64_t));
	GrB_Index  count = 0;

	TensorIterator_ScanRange(&it, A, 0, n - 1, false);
	while (count < max_candidates) {
		GrB_Index r, c;
		uint64_t  v;
		bool      found = TensorIterator_next(&it, &r, &c, &v, NULL);
		if (!found)
			break;
		rows[count] = r;
		cols[count] = c;
		vals[count] = v;
		count++;
	}

	GrB_Index idx = 0;
	for (auto _ : state) {
		if (idx >= count)
			idx = 0;
		Tensor_SetElement(A, rows[idx], cols[idx], vals[idx]);
		idx++;
	}

	Tensor_free(&A);
	free(rows);
	free(cols);
	free(vals);
}

static void BM_tensor_insert_new(benchmark::State& state) {
	Tensor   A    = NULL;
	uint64_t n    = 500000;
	uint64_t seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Random_Tensor(&A, n, 0.0001, add_density, del_density, seed);

	uint64_t rng = seed + 1;
	uint64_t val = 1;
	for (auto _ : state) {
		GrB_Index i = simple_rand(&rng) % n;
		GrB_Index j = simple_rand(&rng) % n;
		Tensor_SetElement(A, i, j, val++);
	}

	Tensor_free(&A);
}

BENCHMARK(BM_insert_new)
    ->Args({10000, 10000})
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(5000);
BENCHMARK(BM_tensor_insert_existing)
    ->Args({10000, 10000})
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(500);
BENCHMARK(BM_tensor_insert_new)
    ->Args({10000, 10000})
    ->ArgNames({"adds", "dels"})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(500);
FDB_BENCHMARK_MAIN()
