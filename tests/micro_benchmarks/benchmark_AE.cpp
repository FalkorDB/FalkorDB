#include "micro_benchmarks.h"

static void _fake_graph_context() {
	GraphContext *gc = (GraphContext *)calloc(1, sizeof(GraphContext));

	gc->g = Graph_New(16, 16);

	gc->ref_count        = 1;
	gc->index_count      = 0;
	gc->graph_name       = strdup("G");
	gc->attributes       = raxNew();
	gc->string_mapping   = (char**)array_new(char*, 64);
	gc->node_schemas     = (Schema**)array_new(Schema*, GRAPH_DEFAULT_LABEL_CAP);
	gc->relation_schemas = (Schema**)array_new(Schema*, GRAPH_DEFAULT_RELATION_TYPE_CAP);
	gc->queries_log      = QueriesLog_New();

	pthread_rwlock_init(&gc->_attribute_rwlock,  NULL);

	GraphContext_AddSchema(gc, "Person", SCHEMA_NODE);
	GraphContext_AddSchema(gc, "City", SCHEMA_NODE);
	GraphContext_AddSchema(gc, "friend", SCHEMA_EDGE);
	GraphContext_AddSchema(gc, "visit", SCHEMA_EDGE);
	GraphContext_AddSchema(gc, "war", SCHEMA_EDGE);

	int res = QueryCtx_Init();
	ASSERT(res);
	QueryCtx_SetGraphCtx(gc);
}

void rg_setup(const benchmark::State &state) {
	_fake_graph_context();
}

void rg_teardown(const benchmark::State &state) {
	QueryCtx_Free();
}

void BM_eval_add_chain(benchmark::State &state) {
	Delta_Matrix Cs[5];
	rax         *matrices = raxNew();
	uint64_t     n    = 10000000;
	uint64_t     seed = 870713428976ul;
	Delta_Matrix res  = NULL;

	int additions = state.range(0);
	int deletions = state.range(1);
	double add_density = additions / ((double) n * (double) n);
	double del_density = deletions / ((double) n * (double) n); 
	unsigned char names[5][16] = {"C0", "C1", "C2", "C3", "C4"};


	for(int i = 0; i < 5; i++) {
		Delta_Random_Matrix(&Cs[i], GrB_BOOL, n, 5E-7, add_density, del_density, seed + 7 * i);
		Delta_Matrix_wait(Cs[i], false);

		raxInsert(matrices, names[i], strlen((char *) names[i]), Cs[i], NULL);
	}

	AlgebraicExpression *exp = AlgebraicExpression_FromString("C0+C1+C2+C3+C4", matrices);

	for (auto _ : state) {
		state.PauseTiming();
		if(res) Delta_Matrix_free(&res);
		Delta_Matrix_new(&res, GrB_BOOL, n, n, false);
		Delta_Matrix_wait(res, true);
		state.ResumeTiming();

		AlgebraicExpression_Eval(exp, res);
	}

	AlgebraicExpression_Free(exp);

	for(int i = 0; i < 5; i++) {
		Delta_Matrix_free(&Cs[i]);
	}
	Delta_Matrix_free(&res);
}

void BM_eval(benchmark::State &state, const char *expression) {
	Delta_Matrix F;
	Delta_Matrix Cs[5];
	rax			 *matrices = raxNew();
	uint64_t	 n		   = 10000000;
	uint64_t	 m		   = 16;
	uint64_t	 seed	   = 870713428976ul;	
	Delta_Matrix res	   = NULL;

	int additions = state.range(0);
	int deletions = state.range(1);
	double add_density = additions / ((double) n * (double) n);
	double del_density = deletions / ((double) n * (double) n); 
	unsigned char names[5][16] = {"C0", "C1", "C2", "C3", "C4"};


	for(int i = 0; i < 5; i++) {
		Delta_Random_Matrix(&Cs[i], GrB_BOOL, n, 5E-7, add_density, del_density, seed + 7 * i);
		
		Delta_Matrix_wait(Cs[i], false);

		raxInsert(matrices, names[i], strlen((char *) names[i]), Cs[i], NULL);
	}

	Delta_Matrix_new (&F, GrB_BOOL, m, n, false);
	for(int i = 0; i < m; i++) {
		Delta_Matrix_setElement_BOOL(F, i, i);
	}
	Delta_Matrix_wait(F, true);
	raxInsert(matrices, (unsigned char *) "F", strlen("F"), F, NULL);

	AlgebraicExpression *exp = AlgebraicExpression_FromString(expression, matrices);

	for (auto _ : state) {
		state.PauseTiming();
		if(res) Delta_Matrix_free(&res);
		Delta_Matrix_new(&res, GrB_BOOL, m, n, false);
		Delta_Matrix_wait(res, true);
		state.ResumeTiming();

		AlgebraicExpression_Eval(exp, res);
	}

	AlgebraicExpression_Free(exp);

	for(int i = 0; i < 5; i++) {
		Delta_Matrix_free(&Cs[i]);
	}
	Delta_Matrix_free(&res);
	Delta_Matrix_free(&F);
}

void BM_eval_mul_chain(benchmark::State &state) {
	Delta_Matrix C;
	rax          *matrices = raxNew();
	Delta_Matrix F         = NULL;
	uint64_t     n         = 10000000;
	uint64_t     m         = 16;
	uint64_t     seed      = 870713428976ul;
	Delta_Matrix res       = NULL;

	int    additions   = state.range(0);
	int    deletions   = state.range(1);
	double add_density = additions / ((double) n * (double) n);
	double del_density = deletions / ((double) n * (double) n); 

	Delta_Random_Matrix(&C, GrB_BOOL, n, 5E-7, add_density, del_density, seed);
	
	Delta_Matrix_wait(C, false);

	Delta_Matrix_new (&F, GrB_BOOL, m, n, false);
	for(int i = 0; i < m; i++) {
		Delta_Matrix_setElement_BOOL(F, i, i);
	}
	Delta_Matrix_wait(F, true);
	raxInsert(matrices, (unsigned char *) "C", strlen("C"), C, NULL);
	raxInsert(matrices, (unsigned char *) "F", strlen("F"), F, NULL);

	AlgebraicExpression *exp = AlgebraicExpression_FromString("F*C*C*C*C*C", matrices);

	for (auto _ : state) {
		state.PauseTiming();
		if(res) Delta_Matrix_free(&res);
		Delta_Matrix_new(&res, GrB_BOOL, m, n, false);
		state.ResumeTiming();

		AlgebraicExpression_Eval(exp, res);
	}

	AlgebraicExpression_Free(exp);
	Delta_Matrix_free(&res);
	Delta_Matrix_free(&C);
}


BENCHMARK(BM_eval_add_chain)->Setup(rg_setup)->Teardown(rg_teardown)
	// ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
	->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
	->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
	->Args({100, 0});
BENCHMARK(BM_eval_mul_chain)->Setup(rg_setup)->Teardown(rg_teardown)
	// ->Unit(benchmark::kMicrosecond)->Args({10000, 10000});
	->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
	->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
	->Args({100, 0});
// BENCHMARK_CAPTURE(BM_eval, (F*C0*C1*C2) + (F*C1*C2*C3) + (F*C2*C3*C4), 
//	   "(F*C0*C1*C2)+(F*C0)")
//	   ->Setup(rg_setup)->Teardown(rg_teardown)
//	   ->Unit(benchmark::kMicrosecond)->Args({10000, 10000})->Iterations(1);
//	   // ->Unit(benchmark::kMillisecond)->Args({0, 0})->Args({10000, 10000})
//	   // ->Args({0, 10000})->Args({10000, 0})->Args({100, 100})->Args({0, 100})
//	   // ->Args({100, 0});
FDB_BENCHMARK_MAIN()
