#include "micro_benchmarks.h"

extern "C" {
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
}

static void BM_delete_from_m(benchmark::State &state) {
	Delta_Matrix A     = NULL;
	Tensor T = NULL;
	uint64_t     n     = 500000;
	uint64_t     seed  = 870713428976ul;
	GrB_Index    nvals = 0;

	Delta_Random_Matrix(&A, GrB_BOOL, n, 0.001, 0.0000001, 0.0000001, seed);
	GrB_Matrix m = Delta_Matrix_M(A);

	GrB_Matrix_nvals(&nvals, m);

	GrB_Index *i_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));
	GrB_Index *j_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));
	bool      *x_v = (bool *)      malloc(nvals * sizeof(bool));

	GrB_OK(GrB_Matrix_extractTuples_BOOL(i_v, j_v, x_v, &nvals, m));

	uint64_t i = 0;
	for (auto _ : state) {
		if (i >= nvals) break;
		if(x_v[i]){
			GrB_Info info = Delta_Matrix_removeElement(A, i_v[i], j_v[i]);
			ASSERT(info == GrB_SUCCESS);
		}
		i++;
	}
	Delta_Matrix_free(&A);
	free(i_v);
	free(j_v);
	free(x_v);
}

static void BM_delete_from_dp(benchmark::State &state) {
	Delta_Matrix A     = NULL;
	uint64_t     n     = 500000;
	uint64_t     seed  = 870713428976ul;
	GrB_Index    nvals = 0;

	Delta_Random_Matrix(&A, GrB_BOOL, n, 0.001, 0.0000001, 0.0000001, seed);

	GrB_Matrix dp = Delta_Matrix_DP(A);

	GrB_Matrix_nvals(&nvals, dp);

	GrB_Index *i_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));
	GrB_Index *j_v = (GrB_Index *) malloc(nvals * sizeof(GrB_Index));

	GrB_OK(GrB_Matrix_extractTuples_BOOL(i_v, j_v, NULL, &nvals, dp));

	int idx = 0;
	for (auto _ : state) {
		if (idx >= nvals) break;
		GrB_Info info = Delta_Matrix_removeElement(A, i_v[idx], j_v[idx]);
		ASSERT(info == GrB_SUCCESS);
		++idx;
	}

	Delta_Matrix_free(&A);
	free(i_v);
	free(j_v);
}

static void BM_tensor_delete(benchmark::State &state) {
	Tensor         A     = NULL;
	TensorIterator it;
	uint64_t       n     = 500000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Random_Tensor(&A, n, 0.001, 0.0000001, 0.0000001, seed);

	Edge arr[40000];

	TensorIterator_ScanRange(&it, A, 0, n - 1, false);

	for (int i = 0; i < 40000; ++i) {
		Edge &e = arr[i];

		bool found = TensorIterator_next(&it, &e.src_id, &e.dest_id, &e.id, NULL);
		ASSERT(found);
	}

	int j = 0;
	for (auto _ : state) {
		ASSERT(j < 40000);
		Edge *e = arr + j;
		Tensor_RemoveElements(A, e, 1, NULL);
		j++;
	}
	Tensor_free(&A);
}

static void BM_tensor_delete_batch(benchmark::State &state) {
	Tensor         A     = NULL;
	TensorIterator it;
	uint64_t       n     = 500000;
	uint64_t       seed  = 870713428976ul;
	GrB_Index      nvals = 0;

	Random_Tensor(&A, n, 0.001, 0.0000001, 0.0000001, seed);

	Edge arr[40000];

	TensorIterator_ScanRange(&it, A, 0, n - 1, false);

	for (int i = 0; i < 40000; ++i) {
		Edge &e = arr[i];

		bool found = TensorIterator_next(&it, &e.src_id, &e.dest_id, &e.id, NULL);
		ASSERT(found);
	}

	int batch_size = 10000;
	int j = 0;
	for (auto _ : state) {
		ASSERT(j + batch_size < 40000);
		Edge *e = arr + j;
		Tensor_RemoveElements(A, e, batch_size, NULL);
		j += batch_size;
	}

	Tensor_free(&A);
}

BENCHMARK(BM_delete_from_m);
BENCHMARK(BM_delete_from_dp)->Iterations(20000);
BENCHMARK(BM_tensor_delete)->Iterations(40000);
BENCHMARK(BM_tensor_delete_batch)->Iterations(4);
FDB_BENCHMARK_MAIN()
