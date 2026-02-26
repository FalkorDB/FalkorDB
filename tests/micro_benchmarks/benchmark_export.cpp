#include "micro_benchmarks.h"

extern "C" {
#include "tests/utils/tensor_random.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include "src/graph/delta_matrix/delta_matrix.h"
}

static void ArgGenerator(benchmark::internal::Benchmark* b) {
	std::vector<int> values = {0, 100, 10000};

	for (int x : values) {
		for (int y : values) {
			if (x == 0 && y == 0) {
				continue;
			}
			b->Args({x, y});
		}
	}
}

static void BM_export(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	GrB_Matrix   C    = NULL;
	uint64_t     n    = 10000000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	for (auto _ : state) {
		Delta_Matrix_export(&C, A, GrB_BOOL);
		GrB_Matrix_wait(C, GrB_MATERIALIZE);
		GrB_Matrix_free(&C);
	}

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	Delta_Matrix_free(&A);
}

static void BM_wait(benchmark::State& state) {
	Delta_Matrix A    = NULL;
	Delta_Matrix C    = NULL;
	uint64_t     n    = 10000000;
	uint64_t     seed = 870713428976ul;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double)n * (double)n);
	double del_density = deletions / ((double)n * (double)n);

	Delta_Random_Matrix(&A, GrB_BOOL, n, 5E-7, add_density, del_density, seed);

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	if(Delta_Matrix_Synced(A)) {
		printf ("Error! Got a fully synced matrix. Exiting.");
		return ;
	}

	for (auto _ : state) {
		state.PauseTiming();
		GrB_OK (Delta_Matrix_free(&C));
		GrB_OK (Delta_Matrix_dup(&C, A));
		state.ResumeTiming();

		GrB_OK (Delta_Matrix_wait(C, true));
	}

	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	GrB_OK (Delta_Matrix_free(&C));
	GrB_OK (Delta_Matrix_free(&A));
}

BENCHMARK(BM_export)->Unit(benchmark::kMillisecond)->Apply(ArgGenerator);
BENCHMARK(BM_wait)->Unit(benchmark::kMillisecond)->Apply(ArgGenerator);
FDB_BENCHMARK_MAIN()
