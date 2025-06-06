/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/value.h"
#include "src/util/arr.h"
#include "src/query_ctx.h"
#include "src/redismodule.h"
#include "src/graph/graph.h"
#include "src/util/rmalloc.h"
#include "src/util/thpool/pools.h"
#include "src/graph/query_graph.h"
#include "src/graph/graphcontext.h"
#include "src/util/simple_timer.h"
#include "src/configuration/config.h"
#include "src/execution_plan/execution_plan.h"
#include "src/arithmetic/algebraic_expression.h"
#include "src/arithmetic/algebraic_expression/utils.h"
#include "GraphBLAS/Include/GraphBLAS.h"

#include <assert.h>
#include <stdlib.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();

#include "acutest.h"

extern AR_ExpNode **_BuildReturnExpressions(const cypher_astnode_t *ret_clause, AST *ast);

QueryGraph *qg;

// Matrices.
Delta_Matrix mat_p;
Delta_Matrix mat_ef;
Delta_Matrix mat_tef;
Delta_Matrix mat_f;
Delta_Matrix mat_ev;
Delta_Matrix mat_tev;
Delta_Matrix mat_c;
Delta_Matrix mat_ew;
Delta_Matrix mat_tew;
Delta_Matrix mat_e;
rax *_matrices;

const char *query_no_intermidate_return_nodes =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p, e";
const char *query_one_intermidate_return_nodes =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p, c, e";
const char *query_multiple_intermidate_return_nodes =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p, f, c, e";
const char *query_return_first_edge =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ef";
const char *query_return_intermidate_edge =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ev";
const char *query_return_last_edge =
	"MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ew";

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

	TEST_ASSERT(QueryCtx_Init());
	QueryCtx_SetGraphCtx(gc);
}

/* Create a graph containing:
	* Entities: 'people' and 'countries'.
	* Relations: 'friend', 'visit' and 'war'. */
static void _build_graph() {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	Node n;
	Graph *g = gc->g;
	Graph_AcquireWriteLock(g);

	size_t city_count = 2;
	size_t person_count = 2;
	size_t node_count = person_count + city_count;

	// Introduce person and country labels.
	int city_label[1] = {GraphContext_GetSchema(gc, "City", SCHEMA_NODE)->id};
	int person_label[1] = {GraphContext_GetSchema(gc, "Person", SCHEMA_NODE)->id};
	Graph_AllocateNodes(g, node_count);

	for(int i = 0; i < person_count; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, person_label, 1);
	}

	for(int i = 0; i < city_count; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, city_label, 1);
	}

	// Creates a relation matrices.
	GrB_Index war_relation_id = GraphContext_GetSchema(gc, "war", SCHEMA_EDGE)->id;
	GrB_Index visit_relation_id = GraphContext_GetSchema(gc, "visit", SCHEMA_EDGE)->id;
	GrB_Index friend_relation_id = GraphContext_GetSchema(gc, "friend", SCHEMA_EDGE)->id;

	// Introduce relations, connect nodes.
	Edge e;
	Graph_CreateEdge(g, 0, 1, friend_relation_id, &e);
	Graph_CreateEdge(g, 1, 0, friend_relation_id, &e);

	Graph_CreateEdge(g, 0, 2, visit_relation_id, &e);
	Graph_CreateEdge(g, 0, 3, visit_relation_id, &e);
	Graph_CreateEdge(g, 1, 2, visit_relation_id, &e);

	Graph_CreateEdge(g, 2, 3, war_relation_id, &e);
	Graph_CreateEdge(g, 3, 2, war_relation_id, &e);

	Graph_ApplyAllPending(g, true);
	Graph_ReleaseLock(g);
}

static void _bind_matrices() {
	int mat_id;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;

	mat_id = GraphContext_GetSchema(gc, "Person", SCHEMA_NODE)->id;
	mat_p = Graph_GetLabelMatrix(g, mat_id);
	mat_f = Graph_GetLabelMatrix(g, mat_id);

	mat_id = GraphContext_GetSchema(gc, "City", SCHEMA_NODE)->id;
	mat_c = Graph_GetLabelMatrix(g, mat_id);
	mat_e = Graph_GetLabelMatrix(g, mat_id);

	mat_id = GraphContext_GetSchema(gc, "friend", SCHEMA_EDGE)->id;
	mat_ef = Graph_GetRelationMatrix(g, mat_id, false);
	mat_tef = Graph_GetRelationMatrix(g, mat_id, true);

	mat_id = GraphContext_GetSchema(gc, "visit", SCHEMA_EDGE)->id;
	mat_ev = Graph_GetRelationMatrix(g, mat_id, false);
	mat_tev = Graph_GetRelationMatrix(g, mat_id, true);

	mat_id = GraphContext_GetSchema(gc, "war", SCHEMA_EDGE)->id;
	mat_ew = Graph_GetRelationMatrix(g, mat_id, false);
	mat_tew = Graph_GetRelationMatrix(g, mat_id, true);

	GrB_Matrix dummy_matrix;
	GrB_Matrix_new(&dummy_matrix, GrB_BOOL, 2, 2);
	_matrices = raxNew();
	raxInsert(_matrices, (unsigned char *)"p", strlen("p"), mat_p, NULL);
	raxInsert(_matrices, (unsigned char *)"f", strlen("f"), mat_f, NULL);
	raxInsert(_matrices, (unsigned char *)"c", strlen("c"), mat_c, NULL);
	raxInsert(_matrices, (unsigned char *)"e", strlen("e"), mat_e, NULL);
	raxInsert(_matrices, (unsigned char *)"F", strlen("F"), mat_ef, NULL);
	raxInsert(_matrices, (unsigned char *)"V", strlen("V"), mat_ev, NULL);
	raxInsert(_matrices, (unsigned char *)"W", strlen("W"), mat_ew, NULL);
	raxInsert(_matrices, (unsigned char *)"tF", strlen("tF"), mat_tef, NULL);
	raxInsert(_matrices, (unsigned char *)"tV", strlen("tV"), mat_tev, NULL);
	raxInsert(_matrices, (unsigned char *)"tW", strlen("tW"), mat_tew, NULL);
	raxInsert(_matrices, (unsigned char *)"A", strlen("A"), dummy_matrix, NULL);
	raxInsert(_matrices, (unsigned char *)"B", strlen("B"), dummy_matrix, NULL);
	raxInsert(_matrices, (unsigned char *)"C", strlen("C"), dummy_matrix, NULL);
}

AlgebraicExpression **build_algebraic_expression(const char *query, AST **master_ast) {
	cypher_parse_result_t *parse_result = cypher_parse(query, NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);
	*master_ast = AST_Build(parse_result);
	AST *ast = AST_NewSegment(*master_ast, 0, cypher_ast_query_nclauses((*master_ast)->root));
	QueryGraph *qg = BuildQueryGraph(ast);
	AlgebraicExpression **ae = AlgebraicExpression_FromQueryGraph(qg);
	_AlgebraicExpression_RemoveRedundentOperands(ae, qg);

	uint exp_count = array_len(ae);
	for(uint i = 0; i < exp_count; i++) {
		AlgebraicExpression_Optimize(ae + i);
	}

	QueryGraph_Free(qg);
	AST_Free(ast);

	return ae;
}

void _print_matrix(GrB_Matrix mat) {
	GrB_Index ncols, nrows, nvals;
	GrB_Matrix_ncols(&ncols, mat);
	GrB_Matrix_nrows(&nrows, mat);
	GrB_Matrix_nvals(&nvals, mat);
#ifdef __aarch64__
	printf("ncols: %llu, nrows: %llu, nvals: %llu\n", ncols, nrows, nvals);
#else
    printf("ncols: %lu, nrows: %lu, nvals: %lu\n", ncols, nrows, nvals);
#endif

	GrB_Index II[nvals];    // array for returning row indices of tuples
	GrB_Index J[nvals];     // array for returning col indices of tuples
	bool X[nvals];          // array for returning values of tuples

	GrB_Matrix_extractTuples_BOOL(II, J, X, &nvals, mat);
	for(int i = 0; i < nvals; i++) {
		printf("[%lu,%lu,%d]\n", II[i], J[i], X[i]);
	}
}

bool _compare_matrices(GrB_Matrix expected, Delta_Matrix actual) {
	GrB_Matrix a = expected;
	GrB_Matrix b = NULL;
	Delta_Matrix_export(&b, actual, GrB_BOOL);

	GrB_Index acols, arows, avals;
	GrB_Index bcols, brows, bvals;

	GrB_Matrix_ncols(&acols, a);
	GrB_Matrix_nrows(&arows, a);
	GrB_Matrix_nvals(&avals, a);
	GrB_Matrix_ncols(&bcols, b);
	GrB_Matrix_nrows(&brows, b);
	GrB_Matrix_nvals(&bvals, b);

	if(acols != bcols || arows != brows || avals != bvals) {
		printf("acols: %lu bcols: %lu\n", acols, bcols);
		printf("arows: %lu brows: %lu\n", arows, brows);
		printf("avals: %lu bvals: %lu\n", avals, bvals);

		GrB_Matrix_free(&b);
		return false;
	}

	GrB_Index aI[avals];     // array for returning row indices of tuples
	GrB_Index aJ[avals];     // array for returning col indices of tuples
	bool aX[avals];          // array for returning values of tuples
	GrB_Index bI[bvals];     // array for returning row indices of tuples
	GrB_Index bJ[bvals];     // array for returning col indices of tuples
	bool bX[bvals];          // array for returning values of tuples

	GrB_Matrix_wait(a, GrB_MATERIALIZE);
	GrB_Matrix_wait(b, GrB_MATERIALIZE);

	GrB_Matrix_extractTuples_BOOL(aI, aJ, aX, &avals, a);
	GrB_Matrix_extractTuples_BOOL(bI, bJ, bX, &bvals, b);

	for(int i = 0; i < avals; i++) {
		if(aI[i] != bI[i] || aJ[i] != bJ[i] || aX[i] != bX[i]) {
			printf("Matrix A \n");
			_print_matrix(a);
			printf("\n\n");
			printf("Matrix B \n");
			_print_matrix(b);
			printf("\n\n");
			GrB_Matrix_free(&b);
			return false;
		}
	}

	GrB_Matrix_free(&b);
	return true;
}

void _compare_algebraic_operand(AlgebraicExpression *a, AlgebraicExpression *b) {
	TEST_ASSERT(a->type == AL_OPERAND);
	TEST_ASSERT(b->type == AL_OPERAND);
	TEST_ASSERT(a->operand.matrix == b->operand.matrix);
}

void compare_algebraic_expression(AlgebraicExpression *a, AlgebraicExpression *b) {
	TEST_ASSERT(a->type == b->type);
	TEST_ASSERT(AlgebraicExpression_ChildCount(a) == AlgebraicExpression_ChildCount(b));
	TEST_ASSERT(AlgebraicExpression_OperandCount(a) == AlgebraicExpression_OperandCount(b));
	// ASSERT_EQ(AlgebraicExpression_OperationCount(a, AL_EXP_ALL), AlgebraicExpression_OperationCount(b, AL_EXP_ALL));
	if(a->type == AL_OPERAND) _compare_algebraic_operand(a, b);

	uint child_count = AlgebraicExpression_ChildCount(a);
	for(uint i = 0; i < child_count; i++) {
		compare_algebraic_expression(a->operation.children[i], b->operation.children[i]);
	}
}

void compare_algebraic_expressions(AlgebraicExpression **actual, AlgebraicExpression **expected,
									uint count) {
	for(uint i = 0; i < count; i++) {
		compare_algebraic_expression(actual[i], expected[i]);
	}
}

void free_algebraic_expressions(AlgebraicExpression **exps, uint count) {
	for(uint i = 0; i < count; i++) {
		AlgebraicExpression *exp = exps[i];
		AlgebraicExpression_Free(exp);
	}
}

void setup() {
	// Use the malloc family for allocations
	Alloc_Reset();

	// Initialize the thread pool.
	TEST_ASSERT(ThreadPools_CreatePools(1, 1, 2));

	// Initialize GraphBLAS.
	GrB_Info info;
	info = GrB_init(GrB_NONBLOCKING);
	TEST_ASSERT(info == GrB_SUCCESS);

	// all matrices in CSR format
	info = GxB_set(GxB_FORMAT, GxB_BY_ROW);
	TEST_ASSERT(info == GrB_SUCCESS);

	// Create a graph
	_fake_graph_context();
	_build_graph();
	_bind_matrices();

	qg = QueryGraph_New(16, 16);
}

void tearDown() {
	TEST_ASSERT(GrB_finalize() == GrB_SUCCESS);
	QueryGraph_Free(qg);
	GraphContext *gc = QueryCtx_GetGraphCtx();
	GraphContext_DecreaseRefCount(gc);
	QueryCtx_Free();
}

void test_algebraicExpression() {
	Delta_Matrix matrix = NULL;
	bool diagonal = false;
	const char *src = "src";
	const char *dest = "dest";
	const char *edge = "edge";
	const char *label = "label";

	AlgebraicExpression *operand = AlgebraicExpression_NewOperand(matrix, diagonal, src, dest, edge,
																  label);
	TEST_ASSERT(operand->type == AL_OPERAND);
	TEST_ASSERT(operand->operand.matrix == matrix);
	TEST_ASSERT(operand->operand.diagonal == diagonal);
	TEST_ASSERT(operand->operand.src == src);
	TEST_ASSERT(operand->operand.dest == dest);
	TEST_ASSERT(operand->operand.edge == edge);
	TEST_ASSERT(operand->operand.label == label);

	TEST_ASSERT(AlgebraicExpression_Src(operand) == src);
	TEST_ASSERT(AlgebraicExpression_Dest(operand) == dest);
	TEST_ASSERT(AlgebraicExpression_Edge(operand) == edge);
	TEST_ASSERT(AlgebraicExpression_ChildCount(operand) == 0);
	TEST_ASSERT(AlgebraicExpression_OperandCount(operand) == 1);
	TEST_ASSERT(AlgebraicExpression_Transposed(operand) == false);

	AL_EXP_OP op = AL_EXP_ADD;
	AlgebraicExpression *operation = AlgebraicExpression_NewOperation(op);
	AlgebraicExpression_AddChild(operation, operand);

	TEST_ASSERT(operation->type == AL_OPERATION);
	TEST_ASSERT(operation->operation.op == op);
	TEST_ASSERT(array_len(operation->operation.children) == 1);

	TEST_ASSERT(AlgebraicExpression_Src(operation) == src);
	TEST_ASSERT(AlgebraicExpression_Dest(operation) == dest);
	TEST_ASSERT(AlgebraicExpression_Edge(operation) == edge);
	TEST_ASSERT(AlgebraicExpression_ChildCount(operation) == 1);
	TEST_ASSERT(AlgebraicExpression_OperandCount(operation) == 1);
	TEST_ASSERT(AlgebraicExpression_Transposed(operation) == false);

	// Will free `operand` aswell.
	AlgebraicExpression_Free(operation);
}

void test_algebraicExpression_domains() {
	GrB_Matrix A;
	GrB_Matrix B;
	const char *src_domain;     // Row domain of expression
	const char *dest_domain;    // Column domain of expression
	AlgebraicExpression *exp;

	GrB_Matrix_new(&A, GrB_BOOL, 2, 2);
	GrB_Matrix_new(&B, GrB_BOOL, 2, 2);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);

	/* Note: addition is performed over matrices with the same source and destination domain
	 * While multiplication takes on source domain of first operand, and destination domain of its last operand
	 * For a given operand A both its source and destination domains are 'A' */

	// A
	// Source(A) = A.
	// Destination(A) = A.
	exp = AlgebraicExpression_FromString("A", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// A+B
	// Addition doesn't modifies column domain.
	// Source(A+B) = A.
	// Destination(A+B) = A.
	exp = AlgebraicExpression_FromString("A+B", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// B+A
	// Addition doesn't modifies column domain.
	// Source(B+A) = B.
	// Destination(B+A) = B.
	exp = AlgebraicExpression_FromString("B+A", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// A*B
	// Source(A*B) = A.
	// Destination(A*B) = B.
	exp = AlgebraicExpression_FromString("A*B", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// B*A
	// Source(B*A) = B.
	// Destination(B*A) = A.
	exp = AlgebraicExpression_FromString("B*A", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(A)
	// Source(Transpose(A)) = A.
	// Destination(Transpose(A)) = A.
	exp = AlgebraicExpression_FromString("T(A)", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose((A+B))
	// Source(Transpose((A+B))) = A.
	// Destination(Transpose((A+B))) = A.
	exp = AlgebraicExpression_FromString("T(A+B)", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose((B+A))
	// Source(Transpose((B+A))) = B.
	// Destination(Transpose((B+A))) = B.
	exp = AlgebraicExpression_FromString("T(B+A)", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose((A*B))
	// Source(Transpose((A*B))) = B.
	// Destination(Transpose((A*B))) = A.
	exp = AlgebraicExpression_FromString("T(A*B)", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose((B*A))
	// Source(Transpose((B*A))) = A.
	// Destination(Transpose((B*A))) = B.
	exp = AlgebraicExpression_FromString("T(B*A)", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(Transpose(A)) = A
	// Source(Transpose(Transpose(A)) = A.
	// Destination(Transpose(Transpose(A)) = A.
	exp = AlgebraicExpression_FromString("T(T(A))", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(Transpose((A+B))) = A+B
	// Source(Transpose(Transpose((A+B)))) = A.
	// Destination(Transpose(Transpose((A+B))) = A.
	exp = AlgebraicExpression_FromString("T(T(A+B))", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(Transpose((B+A))) = B+A
	// Source(Transpose(Transpose((B+A))) = B.
	// Destination(Transpose(Transpose((B+A))) = B.
	exp = AlgebraicExpression_FromString("T(T(B+A))", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(Transpose((A*B))) = A*B
	// Source(Transpose(Transpose((A*B))) = A.
	// Destination(Transpose(Transpose((A*B))) = B.
	exp = AlgebraicExpression_FromString("T(T(A*B))", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "A") == 0);
	TEST_ASSERT(strcmp(dest_domain, "B") == 0);
	AlgebraicExpression_Free(exp);

	// Transpose(Transpose((B*A)) = B*A
	// Source(Transpose(Transpose((B*A))) = B.
	// Destination(Transpose(Transpose((B*A))) = A.
	exp = AlgebraicExpression_FromString("T(T(B*A))", matrices);
	src_domain = AlgebraicExpression_Src(exp);
	dest_domain = AlgebraicExpression_Dest(exp);
	TEST_ASSERT(strcmp(src_domain, "B") == 0);
	TEST_ASSERT(strcmp(dest_domain, "A") == 0);
	AlgebraicExpression_Free(exp);

	raxFree(matrices);
	GrB_Matrix_free(&A);
	GrB_Matrix_free(&B);
}

void test_algebraicExpression_Clone() {
	AlgebraicExpression *exp = NULL;
	AlgebraicExpression *clone = NULL;
	const char *expressions[13] = {"A", "A*B", "A*B*C", "A+B", "A+B+C", "A*B+C", "A+B*C",
								   "T(A)", "T(A*B)", "T(A*B*C)", "T(A+B)", "T(A+B+C)", "T(T(A))"
								  };

	for(uint i = 0; i < 13; i++) {
		exp = AlgebraicExpression_FromString(expressions[i], _matrices);
		clone = AlgebraicExpression_Clone(exp);
		compare_algebraic_expression(exp, clone);
		AlgebraicExpression_Free(clone);
		AlgebraicExpression_Free(exp);
	}
}

void test_algebraicExpression_Transpose() {
	AlgebraicExpression *exp = NULL;
	AlgebraicExpression *transposed_exp = NULL;

	const char *expressions[6] = {"A", "A*B", "A*B*C", "A+B", "A+B+C", "T(A)"};
	const char *transposed_expressions[6] = {"T(A)", "T(B)*T(A)", "T(C)*T(B)*T(A)", "T(A)+T(B)", "T(A)+T(B)+T(C)", "A"};

	for(uint i = 0; i < 6; i++) {
		exp = AlgebraicExpression_FromString(expressions[i], _matrices);
		AlgebraicExpression_Transpose(&exp);
		AlgebraicExpression_Optimize(&exp); // Perform transpose optimizations.

		transposed_exp = AlgebraicExpression_FromString(transposed_expressions[i], _matrices);
		compare_algebraic_expression(exp, transposed_exp);

		AlgebraicExpression_Free(exp);
		AlgebraicExpression_Free(transposed_exp);
	}
}

void test_Exp_OP_ADD() {
	// Exp = A + B
	Delta_Matrix A;
	Delta_Matrix B;
	Delta_Matrix res;
	GrB_Matrix expected;

	// A
	// 1 1
	// 0 0
	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(A, 0, 0);
	Delta_Matrix_setElement_BOOL(A, 0, 1);

	// B
	// 0 1
	// 1 1
	Delta_Matrix_new(&B, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(B, 0, 1);
	Delta_Matrix_setElement_BOOL(B, 1, 0);
	Delta_Matrix_setElement_BOOL(B, 1, 1);

	// expected
	// 1 1
	// 1 1
	GrB_Matrix_new(&expected, GrB_BOOL, 2, 2);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 0);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 1);
	GrB_Matrix_setElement_BOOL(expected, true, 1, 0);
	GrB_Matrix_setElement_BOOL(expected, true, 1, 1);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);
	AlgebraicExpression *exp = AlgebraicExpression_FromString("A+B", matrices);

	// Matrix used for intermidate computations of AlgebraicExpression_Eval
	// but also contains the result of expression evaluation.
	Delta_Matrix_new(&res, GrB_BOOL, 2, 2, false);
	AlgebraicExpression_Eval(exp, res);

	// Using the A matrix described above,
	// A + B = C.
	TEST_ASSERT(_compare_matrices(expected, res));

	raxFree(matrices);
	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&res);
	GrB_Matrix_free(&expected);
	AlgebraicExpression_Free(exp);
}

void test_Exp_OP_MUL() {
	// Exp = A * I
	Delta_Matrix A;
	Delta_Matrix i;
	Delta_Matrix res;

	// A
	// 1 1
	// 0 0
	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(A, 0, 0);
	Delta_Matrix_setElement_BOOL(A, 0, 1);
	Delta_Matrix_wait(A, true); // force flush

	// I
	// 1 0
	// 0 1
	Delta_Matrix_new(&i, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(i, 0, 0);
	Delta_Matrix_setElement_BOOL(i, 1, 1);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"I", strlen("I"), i, NULL);
	AlgebraicExpression *exp = AlgebraicExpression_FromString("A*I", matrices);

	// Matrix used for intermidate computations of AlgebraicExpression_Eval
	// but also contains the result of expression evaluation.
	Delta_Matrix_new(&res, GrB_BOOL, 2, 2, false);
	AlgebraicExpression_Eval(exp, res);

	// Using the A matrix described above,
	// A * I = A.
	GrB_Matrix expected;
	Delta_Matrix_export(&expected, A, GrB_BOOL);
	TEST_ASSERT(_compare_matrices(expected, res));

	raxFree(matrices);
	Delta_Matrix_free(&A);
	Delta_Matrix_free(&i);
	Delta_Matrix_free(&res);
	GrB_Matrix_free(&expected);
	AlgebraicExpression_Free(exp);
}

void test_Exp_OP_ADD_Transpose() {
	// Exp = A + Transpose(A)
	Delta_Matrix res;
	GrB_Matrix expected;
	/* We must use matrices that we've added to the graph, or else
	 * the transpose optimization will erroneously use the adjacency matrix
	 * (since there is no schema associated with the operands).

	 * The 'visit' matrix represented by V has the form:
	 * 0 0 1 1
	 * 0 0 1 0
	 * 0 0 0 0
	 * 0 0 0 0
	 *
	 * Its transpose, TV, has the form:
	 * 0 0 0 0
	 * 0 0 0 0
	 * 1 1 0 0
	 * 1 0 0 0
	 *
	 * The expected result of the addition is:
	 * 0 0 1 1
	 * 0 0 1 0
	 * 1 1 0 0
	 * 1 0 0 0
	 */

	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;
	GrB_Index n = Graph_RequiredMatrixDim(g);

	GrB_Matrix_new(&expected, GrB_BOOL, n, n);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 2);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 3);
	GrB_Matrix_setElement_BOOL(expected, true, 1, 2);
	GrB_Matrix_setElement_BOOL(expected, true, 2, 0);
	GrB_Matrix_setElement_BOOL(expected, true, 2, 1);
	GrB_Matrix_setElement_BOOL(expected, true, 3, 0);
	// Matrix used for intermidate computations of AlgebraicExpression_Eval
	// but also contains the result of expression evaluation.
	Delta_Matrix_new(&res, GrB_BOOL, n, n, false);
	AlgebraicExpression *exp = AlgebraicExpression_FromString("V+tV", _matrices);
	AlgebraicExpression_Eval(exp, res);

	// Using the A matrix described above,
	// A + Transpose(A) = B.
	TEST_ASSERT(_compare_matrices(expected, res));

	Delta_Matrix_free(&res);
	GrB_Matrix_free(&expected);
	AlgebraicExpression_Free(exp);
}

void test_Exp_OP_MUL_Transpose() {
	// Exp = Transpose(A) * A
	GrB_Matrix B;
	Delta_Matrix res;

	/* We must use matrices that we've added to the graph, or else
	 * the transpose optimization will erroneously use the adjacency matrix
	 * (since there is no schema associated with the operands).

	 * The 'visit' matrix represented by V has the form:
	 * 0 0 1 1
	 * 0 0 1 0
	 * 0 0 0 0
	 * 0 0 0 0
	 *
	 * Its transpose, TV, has the form:
	 * 0 0 0 0
	 * 0 0 0 0
	 * 1 1 0 0
	 * 1 0 0 0
	 *
	 * The expected result of the multiplication is:
	 * 1 1 0 0
	 * 1 1 0 0
	 * 1 0 0 0
	 * 1 0 0 0
	 */
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;
	GrB_Index n = Graph_RequiredMatrixDim(g);

	GrB_Matrix_new(&B, GrB_BOOL, n, n);
	GrB_Matrix_setElement_BOOL(B, true, 0, 0);
	GrB_Matrix_setElement_BOOL(B, true, 0, 1);
	GrB_Matrix_setElement_BOOL(B, true, 1, 0);
	GrB_Matrix_setElement_BOOL(B, true, 1, 1);

	// Matrix used for intermidate computations of AlgebraicExpression_Eval
	// but also contains the result of expression evaluation.
	Delta_Matrix_new(&res, GrB_BOOL, n, n, false);

	// Transpose(A) * A
	AlgebraicExpression *exp = AlgebraicExpression_FromString("V*tV", _matrices);
	AlgebraicExpression_Eval(exp, res);

	// Using the A matrix described above,
	// Transpose(A) * A = B.
	TEST_ASSERT(_compare_matrices(B, res));

	GrB_Matrix_free(&B);
	Delta_Matrix_free(&res);
	AlgebraicExpression_Free(exp);
}

void test_Exp_OP_A_MUL_B_Plus_C() {

	// Exp = A*(B+C) = A*B + A*C
	Delta_Matrix A;
	Delta_Matrix B;
	Delta_Matrix C;
	Delta_Matrix res;
	GrB_Matrix expected;

	// A
	// 1 1
	// 0 0
	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(A, 0, 0);
	Delta_Matrix_setElement_BOOL(A, 0, 1);
	Delta_Matrix_wait(A, true); // force flush

	// B
	// 1 0
	// 0 0
	Delta_Matrix_new(&B, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(B, 0, 0);

	// C
	// 0 0
	// 0 1
	Delta_Matrix_new(&C, GrB_BOOL, 2, 2, false);
	Delta_Matrix_setElement_BOOL(C, 1, 1);

	// Matrix used for intermidate computations of AlgebraicExpression_Eval
	// but also contains the result of expression evaluation.
	Delta_Matrix_new(&res, GrB_BOOL, 2, 2, false);

	// A * (B+C) = A.rax *matrices = raxNew();
	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);
	raxInsert(matrices, (unsigned char *)"C", strlen("C"), C, NULL);
	AlgebraicExpression *exp = AlgebraicExpression_FromString("A*(B+C)", matrices);
	AlgebraicExpression_Eval(exp, res);

	// expected
	// 1 1
	// 0 0
	GrB_Matrix_new(&expected, GrB_BOOL, 2, 2);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 0);
	GrB_Matrix_setElement_BOOL(expected, true, 0, 1);
	TEST_ASSERT(_compare_matrices(expected, res));

	raxFree(matrices);
	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&C);
	Delta_Matrix_free(&res);
	GrB_Matrix_free(&expected);
	AlgebraicExpression_Free(exp);
}

void test_ExpTransform_A_Times_B_Plus_C() {
	// Test Mul / Add transformation:
	// A*(B+C) -> A*B + A*C
	Delta_Matrix A;
	Delta_Matrix B;
	Delta_Matrix C;

	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&B, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&C, GrB_BOOL, 2, 2, false);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);
	raxInsert(matrices, (unsigned char *)"C", strlen("C"), C, NULL);
	AlgebraicExpression *exp = AlgebraicExpression_FromString("A*(B+C)",
			matrices);

	// Verifications
	// (A*B)+(A*C)
	TEST_ASSERT(exp->type == AL_OPERATION && exp->operation.op == AL_EXP_ADD);

	AlgebraicExpression *rootLeftChild = exp->operation.children[0];
	AlgebraicExpression *rootRightChild = exp->operation.children[1];
	TEST_ASSERT(rootLeftChild && rootLeftChild->type == AL_OPERATION &&
				rootLeftChild->operation.op == AL_EXP_MUL);
	TEST_ASSERT(rootRightChild && rootRightChild->type == AL_OPERATION &&
				rootRightChild->operation.op == AL_EXP_MUL);

	AlgebraicExpression *leftLeft = rootLeftChild->operation.children[0];
	TEST_ASSERT(leftLeft->type == AL_OPERAND && leftLeft->operand.matrix == A);

	AlgebraicExpression *leftRight = rootLeftChild->operation.children[1];
	TEST_ASSERT(leftRight->type == AL_OPERAND && leftRight->operand.matrix == C);

	AlgebraicExpression *rightLeft = rootRightChild->operation.children[0];
	TEST_ASSERT(rightLeft->type == AL_OPERAND && rightLeft->operand.matrix == A);

	AlgebraicExpression *rightRight = rootRightChild->operation.children[1];
	TEST_ASSERT(rightRight->type == AL_OPERAND && rightRight->operand.matrix == B);

	raxFree(matrices);
	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&C);
	AlgebraicExpression_Free(exp);
}

void test_ExpTransform_AB_Times_C_Plus_D() {
	// Test Mul / Add transformation:
	// A*B*(C+D) -> A*B*C + A*B*D
	Delta_Matrix A;
	Delta_Matrix B;
	Delta_Matrix C;
	Delta_Matrix D;

	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&B, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&C, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&D, GrB_BOOL, 2, 2, false);

	// A*B*(C+D) -> A*B*C + A*B*D
	AlgebraicExpression *exp = AlgebraicExpression_NewOperand(C, false, NULL,
			NULL, NULL, NULL);

	// A*B*(C+D)
	AlgebraicExpression_AddToTheRight(&exp, D);
	AlgebraicExpression_MultiplyToTheLeft(&exp, B);
	AlgebraicExpression_MultiplyToTheLeft(&exp, A);
	AlgebraicExpression_Optimize(&exp);

	// Verifications
	// (A*B*C)+(A*B*D)
	TEST_ASSERT(exp->type == AL_OPERATION && exp->operation.op == AL_EXP_ADD);

	AlgebraicExpression *rootLeftChild = exp->operation.children[0];
	TEST_ASSERT(rootLeftChild && rootLeftChild->type == AL_OPERATION &&
				rootLeftChild->operation.op == AL_EXP_MUL);

	AlgebraicExpression *rootRightChild = exp->operation.children[1];
	TEST_ASSERT(rootRightChild && rootRightChild->type == AL_OPERATION &&
				rootRightChild->operation.op == AL_EXP_MUL);

	AlgebraicExpression *leftchild_0 = rootLeftChild->operation.children[0];
	TEST_ASSERT(leftchild_0->type == AL_OPERAND && leftchild_0->operand.matrix == A);

	AlgebraicExpression *leftchild_1 = rootLeftChild->operation.children[1];
	TEST_ASSERT(leftchild_1->type == AL_OPERAND && leftchild_1->operand.matrix == B);

	AlgebraicExpression *leftchild_2 = rootLeftChild->operation.children[2];
	TEST_ASSERT(leftchild_2->type == AL_OPERAND && leftchild_2->operand.matrix == C);

	AlgebraicExpression *rightchild_0 = rootRightChild->operation.children[0];
	TEST_ASSERT(rightchild_0->type == AL_OPERAND && rightchild_0->operand.matrix == A);

	AlgebraicExpression *rightchild_1 = rootRightChild->operation.children[1];
	TEST_ASSERT(rightchild_1->type == AL_OPERAND && rightchild_1->operand.matrix == B);

	AlgebraicExpression *rightchild_2 = rootRightChild->operation.children[2];
	TEST_ASSERT(rightchild_2->type == AL_OPERAND && rightchild_2->operand.matrix == D);


	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&C);
	Delta_Matrix_free(&D);
	AlgebraicExpression_Free(exp);
}

void test_ExpTransform_A_Plus_B_Times_C_Plus_D() {
	// Test Mul / Add transformation:
	// (A+B)*(C+D) -> A*C + A*D + B*C + B*D
	Delta_Matrix A;
	Delta_Matrix B;
	Delta_Matrix C;
	Delta_Matrix D;
	Delta_Matrix_new(&A, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&B, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&C, GrB_BOOL, 2, 2, false);
	Delta_Matrix_new(&D, GrB_BOOL, 2, 2, false);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);
	raxInsert(matrices, (unsigned char *)"C", strlen("C"), C, NULL);
	raxInsert(matrices, (unsigned char *)"D", strlen("D"), D, NULL);

	// (A+B)*(C+D) -> A*C + A*D + B*C + B*D.
	AlgebraicExpression *exp = AlgebraicExpression_FromString("(A+B)*(C+D)", matrices);
	char *exp_str = AlgebraicExpression_ToString(exp);
	/* Verifications
	 * (A*C) + (A*D) + (B*C) + (B*D).
	 *    +
	 *       +
	 *           +
	 *               *
	 *                   A
	 *                   C
	 *               *
	 *                   A
	 *                   D
	 *           *
	 *               B
	 *               C
	 *       *
	 *           B
	 *           D
	 * */

	TEST_ASSERT(strcmp("(((A * C + A * D) + B * C) + B * D)", exp_str) == 0);
	rm_free(exp_str);
	raxFree(matrices);
	Delta_Matrix_free(&A);
	Delta_Matrix_free(&B);
	Delta_Matrix_free(&C);
	Delta_Matrix_free(&D);
	AlgebraicExpression_Free(exp);
}

void test_MultipleIntermidiateReturnNodes() {
	// "MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p, f, c, e";
	const char *q = query_multiple_intermidate_return_nodes;
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	AlgebraicExpression *expected[3];

	// P * EF * F
	expected[0] = AlgebraicExpression_FromString("p*F*f", _matrices);

	// EV * C
	expected[1] = AlgebraicExpression_FromString("V*c", _matrices);

	// EW * E
	expected[2] = AlgebraicExpression_FromString("W*e", _matrices);

	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_OneIntermidiateReturnNode() {
	const char *q = query_one_intermidate_return_nodes;
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 2);

	AlgebraicExpression *expected[2];

	// p*F*f*V*c
	expected[0] = AlgebraicExpression_FromString("p*F*f*V*c", _matrices);

	// W*e
	expected[1] = AlgebraicExpression_FromString("W*e", _matrices);

	compare_algebraic_expressions(actual, expected, 2);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_NoIntermidiateReturnNodes() {
	const char *q = query_no_intermidate_return_nodes;
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	AlgebraicExpression *expected[1];
	// p*F*f*V*c*W*e
	expected[0] = AlgebraicExpression_FromString("p*F*f*V*c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_ONeIntermidiateReturnEdge() {
	const char *q;
	AlgebraicExpression **actual;

	//==============================================================================================
	//=== MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ef
	//==============================================================================================

	q = query_return_first_edge;
	AST *master_ast;
	actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 2);

	AlgebraicExpression *expected[3];
	expected[0] = AlgebraicExpression_FromString("p*F*f", _matrices);
	expected[1] = AlgebraicExpression_FromString("V*c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 2);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	//==============================================================================================
	//=== MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ev
	//==============================================================================================
	q = query_return_intermidate_edge;
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	expected[0] = AlgebraicExpression_FromString("p*F*f", _matrices);
	expected[1] = AlgebraicExpression_FromString("V*c", _matrices);
	expected[2] = AlgebraicExpression_FromString("W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	//==============================================================================================
	//=== MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN ew
	//==============================================================================================
	q = query_return_last_edge;
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 2);


	expected[0] = AlgebraicExpression_FromString("p*F*f*V*c", _matrices);
	expected[1] = AlgebraicExpression_FromString("W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 2);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_BothDirections() {
	const char *q =
		"MATCH (p:Person)-[ef:friend]->(f:Person)<-[ev:visit]-(c:City)-[ew:war]->(e:City) RETURN p,e";
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	AlgebraicExpression *expected[1];
	expected[0] = AlgebraicExpression_FromString("p*F*f*tV*c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_SingleNode() {
	const char *q = "MATCH (p:Person) RETURN p";
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	AlgebraicExpression *expected[1];
	expected[0] = AlgebraicExpression_FromString("p", _matrices);

	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_ShareableEntity() {
	const char *q =
		"MATCH (p:Person)-[ef:friend]->(f:Person) MATCH (f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p,e";
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	AlgebraicExpression *expected[8];
	expected[0] = AlgebraicExpression_FromString("p*F*f*V*c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	exp_count = 0;
	q = "MATCH (p:Person)-[ef:friend]->(f:Person) MATCH (f:Person)<-[ev:visit]-(c:City)<-[ew:war]-(e:City) RETURN p,e";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("e*W*c*V*f*tF*p", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	exp_count = 0;
	q = "MATCH (p:Person)-[ef:friend]->(f:Person) MATCH (f:Person)-[ev:visit]->(c:City) MATCH (c:City)-[ew:war]->(e:City) RETURN p,e";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("p*F*f*V*c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(f:Person) MATCH (b:Person)-[:friend]->(f:Person) RETURN a,b";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("p*F*p*tF*p", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// High incoming degree.
	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(d:Person) MATCH (b:Person)-[:friend]->(d:Person) MATCH (c:Person)-[:friend]->(d:Person) RETURN a";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	expected[0] = AlgebraicExpression_FromString("p*F*p", _matrices);
	expected[1] = AlgebraicExpression_FromString("tF*p", _matrices);
	expected[2] = AlgebraicExpression_FromString("p*F*p", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// High outgoing degree.
	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(b:Person) MATCH (a:Person)-[:friend]->(c:Person) MATCH (a:Person)-[:friend]->(d:Person) RETURN a";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	expected[0] = AlgebraicExpression_FromString("p*tF*p", _matrices);
	expected[1] = AlgebraicExpression_FromString("F*p", _matrices);
	expected[2] = AlgebraicExpression_FromString("F*p", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// Cycle.
	/* TODO: The algebraic expression here can be improved
	 * reducing from 2 expression into a single one
	 * mat_p * mat_ef * mat_p * mat_ef * mat_p
	 * see comment in AlgebraicExpression_FromQueryGraph regarding cycles. */
	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(b:Person)-[:friend]->(a:Person) RETURN a";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("p*F*p*F*p", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// Longer cycle.
	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(b:Person)-[:friend]->(c:Person)-[:friend]->(a:Person) RETURN a";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("p*F*p*F*p*F*p", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// Self pointing node.
	exp_count = 0;
	q = "MATCH (a:Person)-[:friend]->(a) RETURN a";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 1);

	expected[0] = AlgebraicExpression_FromString("p*F*p", _matrices);
	compare_algebraic_expressions(actual, expected, 1);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	//(p1)-[]->(p2)-[]->(p3)-[]->(p2)-[]->(p4)-[]->(p5) RETURN p1
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p2)-[:friend]->(p4)-[:friend]->(p5) RETURN p1";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	expected[0] = AlgebraicExpression_FromString("F", _matrices);
	expected[1] = AlgebraicExpression_FromString("F*F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F*F", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// (p1)-[]->(p2)-[]->(p3)-[]->(p2)-[]->(p4)-[]->(p5) RETURN p1,p2,p3,p4,p5
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p2)-[:friend]->(p4)-[:friend]->(p5) RETURN p1,p2,p3,p4,p5";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 5);

	expected[0] = AlgebraicExpression_FromString("F", _matrices);
	expected[1] = AlgebraicExpression_FromString("F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F", _matrices);
	expected[3] = AlgebraicExpression_FromString("F", _matrices);
	expected[4] = AlgebraicExpression_FromString("F", _matrices);
	compare_algebraic_expressions(actual, expected, 5);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// MATCH (p1)-[]->(p2)-[]->(p3)-[]->(p4)-[]->(p5)-[]->(p2)-[]->(p6)-[]->(p7)-[]->(p3) RETURN p1
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p4)-[:friend]->(p5)-[:friend]->(p2)-[:friend]->(p6)-[:friend]->(p7)-[:friend]->(p3) RETURN p1";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 4);

	expected[0] = AlgebraicExpression_FromString("F", _matrices);
	expected[1] = AlgebraicExpression_FromString("F*F*F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F*F*F", _matrices);
	expected[3] = AlgebraicExpression_FromString("F", _matrices);
	compare_algebraic_expressions(actual, expected, 4);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// MATCH (p1)-[]->(p2)-[]->(p3)-[]->(p4)-[]->(p5)-[]->(p2)-[]->(p6)-[]->(p7)-[]->(p3) RETURN p1,p2,p3,p4,p5,p6,p7
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p4)-[:friend]->(p5)-[:friend]->(p2)-[:friend]->(p6)-[:friend]->(p7)-[:friend]->(p3) RETURN p1,p2,p3,p4,p5,p6,p7";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 8);

	expected[0] = AlgebraicExpression_FromString("F", _matrices);
	expected[1] = AlgebraicExpression_FromString("F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F", _matrices);
	expected[3] = AlgebraicExpression_FromString("F", _matrices);
	expected[4] = AlgebraicExpression_FromString("F", _matrices);
	expected[5] = AlgebraicExpression_FromString("F", _matrices);
	expected[6] = AlgebraicExpression_FromString("F", _matrices);
	expected[7] = AlgebraicExpression_FromString("F", _matrices);
	compare_algebraic_expressions(actual, expected, 8);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// MATCH (p1)-[]->(p2)-[]->(p3)-[]->(p4)-[]->(p1)-[]->(p4),(p4)-[]->(p5) RETURN p1
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p4)-[:friend]->(p1)-[:friend]->(p4)-[:friend]->(p5) RETURN p1";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 4);

	expected[0] = AlgebraicExpression_FromString("tF", _matrices);
	expected[1] = AlgebraicExpression_FromString("F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F*F*F", _matrices);
	expected[3] = AlgebraicExpression_FromString("F", _matrices);
	compare_algebraic_expressions(actual, expected, 4);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// MATCH (p1)-[]->(p2)-[]->(p3)-[]->(p4)-[]->(p1)-[]->(p4),(p4)-[]->(p5) RETURN p1,p2,p3,p4,p5
	exp_count = 0;
	q = "MATCH (p1)-[:friend]->(p2)-[:friend]->(p3)-[:friend]->(p4)-[:friend]->(p1)-[:friend]->(p4)-[:friend]->(p5) RETURN p1,p2,p3,p4,p5";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 6);

	expected[0] = AlgebraicExpression_FromString("tF", _matrices);
	expected[1] = AlgebraicExpression_FromString("F", _matrices);
	expected[2] = AlgebraicExpression_FromString("F", _matrices);
	expected[3] = AlgebraicExpression_FromString("F", _matrices);
	expected[4] = AlgebraicExpression_FromString("F", _matrices);
	expected[5] = AlgebraicExpression_FromString("F", _matrices);
	compare_algebraic_expressions(actual, expected, 6);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_VariableLength() {
	const char *q =
		"MATCH (p:Person)-[ef:friend]->(f:Person)-[:visit*1..3]->(c:City)-[ew:war]->(e:City) RETURN p,e";
	AST *master_ast;
	AlgebraicExpression **actual = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	AlgebraicExpression *expected[3];
	expected[0] = AlgebraicExpression_FromString("p*F*f", _matrices);
	expected[1] = AlgebraicExpression_FromString("V", _matrices);
	expected[2] = AlgebraicExpression_FromString("c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);

	// Transposed variable length.
	exp_count = 0;
	q = "MATCH (p:Person)-[ef:friend]->(f:Person)<-[:visit*1..3]-(c:City)-[ew:war]->(e:City) RETURN p,e";
	actual = build_algebraic_expression(q, &master_ast);
	exp_count = array_len(actual);
	TEST_ASSERT(exp_count == 3);

	expected[0] = AlgebraicExpression_FromString("p*F*f", _matrices);
	expected[1] = AlgebraicExpression_FromString("tV", _matrices);
	expected[2] = AlgebraicExpression_FromString("c*W*e", _matrices);
	compare_algebraic_expressions(actual, expected, 3);

	// Clean up.
	free_algebraic_expressions(actual, exp_count);
	free_algebraic_expressions(expected, exp_count);
	array_free(actual);
	AST_Free(master_ast);
}

void test_ExpressionExecute() {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;

	// "MATCH (p:Person)-[ef:friend]->(f:Person)-[ev:visit]->(c:City)-[ew:war]->(e:City) RETURN p, e"
	const char *q = query_no_intermidate_return_nodes;
	AST *master_ast;
	AlgebraicExpression **ae = build_algebraic_expression(q, &master_ast);
	uint exp_count = array_len(ae);
	TEST_ASSERT(exp_count == 1);

	AlgebraicExpression *exp = ae[0];
	TEST_ASSERT(strcmp(AlgebraicExpression_Src(exp), "p") == 0);
	TEST_ASSERT(strcmp(AlgebraicExpression_Dest(exp), "e") == 0);

	Delta_Matrix res;
	Delta_Matrix_new(&res, GrB_BOOL, Graph_RequiredMatrixDim(g),
			Graph_RequiredMatrixDim(g), false);
	AlgebraicExpression_Eval(exp, res);

	// Validate result matrix.
	GrB_Index ncols, nrows;
	Delta_Matrix_ncols(&ncols, res);
	Delta_Matrix_nrows(&nrows, res);
	assert(ncols == Graph_RequiredMatrixDim(g));
	assert(nrows == Graph_RequiredMatrixDim(g));

	GrB_Matrix expected = NULL;
	GrB_Index expected_entries[6] = {1, 2, 0, 3, 1, 3};
	GrB_Matrix_new(&expected, GrB_BOOL, nrows, ncols);
	for(int i = 0; i < 6; i += 2) {
		GrB_Matrix_setElement_BOOL(expected, true, expected_entries[i],
				expected_entries[i + 1]);
	}

	assert(_compare_matrices(expected, res));

	// Clean up
	Delta_Matrix_free(&res);
	GrB_Matrix_free(&expected);
	free_algebraic_expressions(ae, exp_count);
	array_free(ae);
	AST_Free(master_ast);
}

void test_RemoveOperand() {
	GrB_Matrix A                          = GrB_NULL;
	GrB_Matrix B                          = GrB_NULL;
	AlgebraicExpression *exp              = NULL;
	AlgebraicExpression *expected         = NULL;
	AlgebraicExpression *removed_operand  = NULL;

	GrB_Matrix_new(&A, GrB_BOOL, 2, 2);
	GrB_Matrix_new(&B, GrB_BOOL, 2, 2);

	rax *matrices = raxNew();
	raxInsert(matrices, (unsigned char *)"A", strlen("A"), A, NULL);
	raxInsert(matrices, (unsigned char *)"B", strlen("B"), B, NULL);

	const char *exps[13];
	const char *removed[13];
	const char *expectations[13];

	//--------------------------------------------------------------------------
	// Remove SOURCE operand from expression
	//--------------------------------------------------------------------------

	// Exp: A, remove A, expecting NULL
	exps[0]         = "A";
	removed[0]      = "A";
	expectations[0] = NULL;

	// Exp: A+B, remove A, expecting B
	exps[1]         = "A+B";
	removed[1]      = "A";
	expectations[1] = "B";

	// Exp: A*B, remove A, expecting B
	exps[2]         = "A*B";
	removed[2]      = "A";
	expectations[2] = "B";

	// Exp: T(A), remove A, expecting NULL
	exps[3]         = "T(A)";
	removed[3]      = "A";
	expectations[3] = NULL;

	// Exp: T(T(A)), remove A, expecting NULL
	exps[4]         = "T(T(A))";
	removed[4]      = "A";
	expectations[4] = NULL;

	// Exp: T(A) + B remove A, expecting B
	exps[5]         = "T(A)+B";
	removed[5]      = "A";
	expectations[5] = "B";

	// Exp: T(T(A)) + B remove A, expecting B
	exps[6]         = "T(T(A))+B";
	removed[6]      = "A";
	expectations[6] = "B";

	// Exp: T(A+B), remove A, expecting T(B)
	exps[7]         = "T(A+B)";
	removed[7]      = "A";
	expectations[7] = "T(B)";

	// Exp: T(T(A)+B), remove A, expecting T(B)
	exps[8]         = "T(T(A)+B)";
	removed[8]      = "A";
	expectations[8] = "T(B)";

	// Exp: T(A) * B remove A, expecting B
	exps[9]         = "T(A)*B";
	removed[9]      = "A";
	expectations[9] = "B";

	// Exp: T(T(A)) * B remove A, expecting B
	exps[10]         = "T(T(A))*B";
	removed[10]      = "A";
	expectations[10] = "B";

	// Exp: T(A*B), remove B, expecting T(A)
	exps[11]         = "T(A*B)";
	removed[11]      = "B";
	expectations[11] = "T(A)";

	// Exp: T(T(A)*B), remove B, expecting T(T(A))
	exps[12]         = "T(T(A)*B)";
	removed[12]      = "B";
	expectations[12] = "T(T(A))";

	for(int i = 0; i < 13; i++) {
		exp = AlgebraicExpression_FromString(exps[i], matrices);
		removed_operand = AlgebraicExpression_FromString(removed[i], matrices);

		if(expectations[i] == NULL) expected = NULL;
		else expected = AlgebraicExpression_FromString(expectations[i], matrices);

		/* remove source operand and compare both modified expression
		 * and removed source */
		AlgebraicExpression *src = AlgebraicExpression_RemoveSource(&exp);
		compare_algebraic_expression(src, removed_operand);

		if(expected == NULL) TEST_ASSERT(NULL == exp);
		else compare_algebraic_expression(exp, expected);

		// clean up
		AlgebraicExpression_Free(src);
		AlgebraicExpression_Free(removed_operand);
		if(expected != NULL) {
			AlgebraicExpression_Free(exp);
			AlgebraicExpression_Free(expected);
		}
	}

	//--------------------------------------------------------------------------
	// Remove DESTINATION operand from expression
	//--------------------------------------------------------------------------

	// Exp: A, remove A, expecting NULL
	exps[0]         = "A";
	removed[0]      = "A";
	expectations[0] = NULL;

	// Exp: A+B, remove B, expecting A
	exps[1]         = "A+B";
	removed[1]      = "B";
	expectations[1] = "A";

	// Exp: A*B, remove B, expecting A
	exps[2]         = "A*B";
	removed[2]      = "B";
	expectations[2] = "A";

	// Exp: T(A), remove A, expecting NULL
	exps[3]         = "T(A)";
	removed[3]      = "A";
	expectations[3] = NULL;

	// Exp: T(T(A)), remove A, expecting NULL
	exps[4]         = "T(T(A))";
	removed[4]      = "A";
	expectations[4] = NULL;

	// Exp: T(A) + B remove B, expecting T(A)
	exps[5]         = "T(A)+B";
	removed[5]      = "B";
	expectations[5] = "T(A)";

	// Exp: T(T(A)) + B remove B, expecting T(T(A))
	exps[6]         = "T(T(A))+B";
	removed[6]      = "B";
	expectations[6] = "T(T(A))";

	// Exp: T(A+B), remove B, expecting T(A)
	exps[7]         = "T(A+B)";
	removed[7]      = "B";
	expectations[7] = "T(A)";

	// Exp: T(T(A)+B), remove B, expecting T(T(A))
	exps[8]         = "T(T(A)+B)";
	removed[8]      = "B";
	expectations[8] = "T(T(A))";

	// Exp: T(A) * B remove B, expecting T(A)
	exps[9]         = "T(A)*B";
	removed[9]      = "B";
	expectations[9] = "T(A)";

	// Exp: T(T(A)) * B remove B, expecting T(T(A))
	exps[10]         = "T(T(A))*B";
	removed[10]      = "B";
	expectations[10] = "T(T(A))";

	// Exp: T(A*B), remove A, expecting T(B)
	exps[11]         = "T(A*B)";
	removed[11]      = "A";
	expectations[11] = "T(B)";

	// Exp: T(T(A)*B), remove A, expecting T(B)
	exps[12]         = "T(T(A)*B)";
	removed[12]      = "A";
	expectations[12] = "T(B)";

	for(int i = 0; i < 13; i++) {
		exp = AlgebraicExpression_FromString(exps[i], matrices);
		removed_operand = AlgebraicExpression_FromString(removed[i], matrices);

		if(expectations[i] == NULL) expected = NULL;
		else expected = AlgebraicExpression_FromString(expectations[i], matrices);

		/* remove source operand and compare both modified expression
		 * and removed source */
		AlgebraicExpression *dest = AlgebraicExpression_RemoveDest(&exp);
		compare_algebraic_expression(dest, removed_operand);

		if(expected == NULL) TEST_ASSERT(NULL == exp);
		else compare_algebraic_expression(exp, expected);

		// clean up
		AlgebraicExpression_Free(dest);
		AlgebraicExpression_Free(removed_operand);
		if(expected != NULL) {
			AlgebraicExpression_Free(exp);
			AlgebraicExpression_Free(expected);
		}
	}

	raxFree(matrices);
}

void test_LocateOperand() {
	// construct algebraic expression
	bool                 located  =  false;
	Delta_Matrix         mat      =  NULL;
	AlgebraicExpression  *A       =  NULL;
	AlgebraicExpression  *B       =  NULL;
	AlgebraicExpression  *r       =  NULL;
	AlgebraicExpression  *op      =  NULL;
	AlgebraicExpression  *p       =  NULL;

	// ( T(A) * B )
	A = AlgebraicExpression_NewOperand(mat, false, "a", "b", "e0", NULL);
	B = AlgebraicExpression_NewOperand(mat, false, "b", "c", "e1", NULL);
	AlgebraicExpression_Transpose(&A);
	r = AlgebraicExpression_NewOperation(AL_EXP_MUL);
	AlgebraicExpression_AddChild(r, A);
	AlgebraicExpression_AddChild(r, B);

	// search for A operand
	located = AlgebraicExpression_LocateOperand(r, &op, &p, "a", "b", "e0", NULL);

	// validate located operand
	TEST_ASSERT(located);
	TEST_ASSERT(op->type == AL_OPERAND);
	TEST_ASSERT(strcmp(op->operand.src, "a") == 0);
	TEST_ASSERT(strcmp(op->operand.dest, "b") == 0);
	TEST_ASSERT(strcmp(op->operand.edge, "e0") == 0);

	TEST_ASSERT(p->type == AL_OPERATION);
	TEST_ASSERT(p->operation.op == AL_EXP_TRANSPOSE);

	// search for none existing operand
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, "x", "b", "e0", NULL));
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, "a", "x", "e0", NULL));
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, "a", "b", "x", NULL));
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, NULL, "b", "e0", NULL));
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, "a", NULL, "e0", NULL));
	TEST_ASSERT(!AlgebraicExpression_LocateOperand(r, &op, &p, "a", "b", NULL, NULL));

	AlgebraicExpression_Free(r);
}

TEST_LIST = {
	{"algebraicExpression", test_algebraicExpression},
	{"algebraicExpression_domains", test_algebraicExpression_domains},
	{"algebraicExpression_Clone", test_algebraicExpression_Clone},
	{"algebraicExpression_Transpose", test_algebraicExpression_Transpose},
	{"Exp_OP_ADD", test_Exp_OP_ADD},
	{"Exp_OP_MUL", test_Exp_OP_MUL},
	{"Exp_OP_ADD_Transpose", test_Exp_OP_ADD_Transpose},
	{"Exp_OP_MUL_Transpose", test_Exp_OP_MUL_Transpose},
	{"Exp_OP_A_MUL_B_Plus_C", test_Exp_OP_A_MUL_B_Plus_C},
	{"ExpTransform_A_Times_B_Plus_C", test_ExpTransform_A_Times_B_Plus_C},
	{"ExpTransform_AB_Times_C_Plus_D", test_ExpTransform_AB_Times_C_Plus_D},
	{"ExpTransform_A_Plus_B_Times_C_Plus_D", test_ExpTransform_A_Plus_B_Times_C_Plus_D},
	{"MultipleIntermidiateReturnNodes", test_MultipleIntermidiateReturnNodes},
	{"OneIntermidiateReturnNode", test_OneIntermidiateReturnNode},
	{"NoIntermidiateReturnNodes", test_NoIntermidiateReturnNodes},
	{"ONeIntermidiateReturnEdge", test_ONeIntermidiateReturnEdge},
	{"BothDirections", test_BothDirections},
	{"SingleNode", test_SingleNode},
	{"ShareableEntity", test_ShareableEntity},
	{"VariableLength", test_VariableLength},
	{"ExpressionExecute", test_ExpressionExecute},
	{"RemoveOperand", test_RemoveOperand},
	{"LocateOperand", test_LocateOperand},
	{NULL, NULL}
};

