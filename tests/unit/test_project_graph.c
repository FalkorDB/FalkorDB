/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/value.h"
#include "src/query_ctx.h"
#include "src/util/arr.h"
#include "src/graph/graph.h"
#include "src/util/rmalloc.h"
#include "src/graph/graphcontext.h"
#include "src/graph/graphcontext_struct.h"
#include "src/procedures/utility/internal.h"
#include "GraphBLAS/Include/GraphBLAS.h"
#include <math.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

static GraphContext *gc = NULL;
static Graph *g = NULL;
static LabelID label_a = INVALID_ENTITY_ID;
static LabelID label_b = INVALID_ENTITY_ID;
static RelationID rel_r = INVALID_ENTITY_ID;
static RelationID rel_s = INVALID_ENTITY_ID;
static AttributeID edge_w = ATTRIBUTE_ID_NONE;
static AttributeID node_w = ATTRIBUTE_ID_NONE;

static void _init_graph_context() {
	gc = rm_calloc(1, sizeof(GraphContext));
	gc->g = Graph_New(16, 16);

	gc->ref_count        = 1;
	gc->index_count      = 0;
	gc->graph_name       = rm_strdup("G");
	gc->attributes       = NULL;
	gc->node_schemas     = (Schema**)arr_new(Schema*, 0);
	gc->relation_schemas = (Schema**)arr_new(Schema*, 0);
	gc->queries_log      = QueriesLog_New();
	pthread_rwlock_init(&gc->rwlock, NULL);

	GraphContext_FindOrAddSchema(gc, "A", SCHEMA_NODE, NULL);
	GraphContext_FindOrAddSchema(gc, "B", SCHEMA_NODE, NULL);
	GraphContext_FindOrAddSchema(gc, "R", SCHEMA_EDGE, NULL);
	GraphContext_FindOrAddSchema(gc, "S", SCHEMA_EDGE, NULL);

	label_a = GraphContext_GetSchema(gc, "A", SCHEMA_NODE)->id;
	label_b = GraphContext_GetSchema(gc, "B", SCHEMA_NODE)->id;
	rel_r = GraphContext_GetSchema(gc, "R", SCHEMA_EDGE)->id;
	rel_s = GraphContext_GetSchema(gc, "S", SCHEMA_EDGE)->id;

	edge_w = GraphContext_FindOrAddAttribute(gc, "ew", NULL);
	node_w = GraphContext_FindOrAddAttribute(gc, "nw", NULL);

	TEST_ASSERT(QueryCtx_Init());
	QueryCtx_SetGraphCtx(gc);
	g = GraphContext_GetGraph(gc);
}

static void _build_graph() {
	GraphContext_AcquireWriteLock(gc);

	int la[1] = {label_a};
	int lb[1] = {label_b};

	Node n;
	n = GE_NEW_NODE();
	Graph_CreateNode(g, &n, la, 1);  // id 0
	n = GE_NEW_NODE();
	Graph_CreateNode(g, &n, la, 1);  // id 1
	n = GE_NEW_NODE();
	Graph_CreateNode(g, &n, lb, 1);  // id 2
	n = GE_NEW_NODE();
	Graph_CreateNode(g, &n, lb, 1);  // id 3

	Node n0, n1;
	TEST_ASSERT(Graph_GetNode(g, 0, &n0));
	TEST_ASSERT(Graph_GetNode(g, 1, &n1));
	GraphEntity_AddProperty((GraphEntity *)&n0, node_w, SI_DoubleVal(10));
	GraphEntity_AddProperty((GraphEntity *)&n1, node_w, SI_DoubleVal(20));

	Edge e;
	Graph_CreateEdge(g, 0, 1, rel_r, &e);
	GraphEntity_AddProperty((GraphEntity *)&e, edge_w, SI_DoubleVal(5));

	Graph_CreateEdge(g, 0, 1, rel_r, &e);
	GraphEntity_AddProperty((GraphEntity *)&e, edge_w, SI_DoubleVal(1.5));

	// first edge on (0,2) intentionally has no ew value
	Graph_CreateEdge(g, 0, 2, rel_r, &e);
	Graph_CreateEdge(g, 0, 2, rel_r, &e);
	GraphEntity_AddProperty((GraphEntity *)&e, edge_w, SI_DoubleVal(7));

	Graph_CreateEdge(g, 2, 3, rel_s, &e);
	GraphEntity_AddProperty((GraphEntity *)&e, edge_w, SI_DoubleVal(4));

	Graph_ApplyAllPending(g, true);
	GraphContext_ReleaseLock(gc);
}

static void _assert_bool_entry(GrB_Matrix A, GrB_Index i, GrB_Index j, bool expected) {
	bool v = false;
	GrB_Info info = GrB_Matrix_extractElement_BOOL(&v, A, i, j);
	if(expected) {
		TEST_ASSERT(info == GrB_SUCCESS);
		TEST_ASSERT(v == true);
	} else {
		TEST_ASSERT(info == GrB_NO_VALUE);
	}
}

static void _assert_fp64_entry(GrB_Matrix A, GrB_Index i, GrB_Index j, double expected) {
	double v = 0;
	GrB_Info info = GrB_Matrix_extractElement_FP64(&v, A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(fabs(v - expected) < 1e-9);
}

static void _assert_fp64_missing(GrB_Matrix A, GrB_Index i, GrB_Index j) {
	double v = 0;
	GrB_Info info = GrB_Matrix_extractElement_FP64(&v, A, i, j);
	TEST_ASSERT(info == GrB_NO_VALUE);
}

void setup() {
	Alloc_Reset();
	TEST_ASSERT(GrB_init(GrB_NONBLOCKING) == GrB_SUCCESS);
	TEST_ASSERT(GrB_set(GrB_GLOBAL, GxB_BY_ROW, GxB_FORMAT) == GrB_SUCCESS);
	_init_graph_context();
	_build_graph();
}

void tearDown() {
	GraphContext_DecreaseRefCount(gc);
	QueryCtx_Free();
	gc = NULL;
	g = NULL;
	TEST_ASSERT(GrB_finalize() == GrB_SUCCESS);
}

void test_boolean_projection_default_behavior() {
	GrB_Matrix A = NULL;
	GrB_Vector rows = NULL;
	RelationID rels[] = {rel_r};

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.rels = rels;
	conf.n_rels = 1;

	GrB_Info info = project_graph_to_matrix(&A, &rows, conf);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Type tA = NULL, tR = NULL;
	TEST_ASSERT(GxB_Matrix_type(&tA, A) == GrB_SUCCESS);
	TEST_ASSERT(GxB_Vector_type(&tR, rows) == GrB_SUCCESS);
	TEST_ASSERT(tA == GrB_BOOL);
	TEST_ASSERT(tR == GrB_BOOL);

	_assert_bool_entry(A, 0, 1, true);
	_assert_bool_entry(A, 0, 2, true);
	_assert_bool_entry(A, 2, 3, false);

	GrB_Index rows_nvals = 0;
	TEST_ASSERT(GrB_Vector_nvals(&rows_nvals, rows) == GrB_SUCCESS);
	TEST_ASSERT(rows_nvals == 4);

	GrB_free(&A);
	GrB_free(&rows);
}

void test_edge_default_and_incoming_direction() {
	GrB_Matrix A = NULL;
	RelationID rels[] = {rel_r};

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.rels = rels;
	conf.n_rels = 1;
	conf.default_ew = SI_DoubleVal(2.5);
	conf.strategy = PROJECT_TO_ANY;
	conf.direction = GRAPH_EDGE_DIR_INCOMING;

	GrB_Info info = project_graph_to_matrix(&A, NULL, conf);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Type tA = NULL;
	TEST_ASSERT(GxB_Matrix_type(&tA, A) == GrB_SUCCESS);
	TEST_ASSERT(tA == GrB_FP64);

	_assert_fp64_entry(A, 1, 0, 2.5);
	_assert_fp64_entry(A, 2, 0, 2.5);
	_assert_fp64_missing(A, 0, 1);

	GrB_free(&A);
}

void test_min_max_any_on_multiedge() {
	GrB_Matrix A = NULL;
	RelationID rels[] = {rel_r};
	LabelID lbls[] = {label_a};

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.rels = rels;
	conf.n_rels = 1;
	conf.lbls = lbls;
	conf.n_lbls = 1;
	conf.edge_weight = edge_w;

	conf.strategy = PROJECT_TO_MIN;
	TEST_ASSERT(project_graph_to_matrix(&A, NULL, conf) == GrB_SUCCESS);
	_assert_fp64_entry(A, 0, 1, 1.5);
	GrB_free(&A);

	conf.strategy = PROJECT_TO_MAX;
	TEST_ASSERT(project_graph_to_matrix(&A, NULL, conf) == GrB_SUCCESS);
	_assert_fp64_entry(A, 0, 1, 5);
	GrB_free(&A);

	conf.strategy = PROJECT_TO_ANY;
	TEST_ASSERT(project_graph_to_matrix(&A, NULL, conf) == GrB_SUCCESS);
	_assert_fp64_entry(A, 0, 1, 5);
	GrB_free(&A);
}

void test_any_errors_on_first_invalid_edge() {
	GrB_Matrix A = NULL;
	RelationID rels[] = {rel_r};

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.rels = rels;
	conf.n_rels = 1;
	conf.edge_weight = edge_w;
	conf.strategy = PROJECT_TO_ANY;

	GrB_Info info = project_graph_to_matrix(&A, NULL, conf);
	TEST_ASSERT(info != GrB_SUCCESS);
	GrB_free(&A);
}

void test_weighted_rows_and_compact_projection() {
	GrB_Matrix A = NULL;
	GrB_Vector rows = NULL;
	RelationID rels[] = {rel_r};
	LabelID lbls[] = {label_a};

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.rels = rels;
	conf.n_rels = 1;
	conf.lbls = lbls;
	conf.n_lbls = 1;
	conf.default_ew = SI_DoubleVal(9);
	conf.node_weight = node_w;
	conf.compact = true;

	TEST_ASSERT(project_graph_to_matrix(&A, &rows, conf) == GrB_SUCCESS);

	GrB_Index nrows = 0, ncols = 0;
	TEST_ASSERT(GrB_Matrix_nrows(&nrows, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_ncols(&ncols, A) == GrB_SUCCESS);
	TEST_ASSERT(nrows == 2 && ncols == 2);

	GrB_Type tR = NULL;
	TEST_ASSERT(GxB_Vector_type(&tR, rows) == GrB_SUCCESS);
	TEST_ASSERT(tR == GrB_FP64);

	double w0 = 0, w1 = 0;
	TEST_ASSERT(GrB_Vector_extractElement_FP64(&w0, rows, 0) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Vector_extractElement_FP64(&w1, rows, 1) == GrB_SUCCESS);
	TEST_ASSERT(w0 == 10);
	TEST_ASSERT(w1 == 20);

	GrB_free(&A);
	GrB_free(&rows);
}

void test_rows_attr_missing_value_errors_without_default() {
	GrB_Matrix A = NULL;
	GrB_Vector rows = NULL;

	PGTM_config conf = DEFAULT_PGTM_CONFIG;
	conf.g = g;
	conf.node_weight = node_w;

	GrB_Info info = project_graph_to_matrix(&A, &rows, conf);
	TEST_ASSERT(info != GrB_SUCCESS);

	GrB_free(&A);
	GrB_free(&rows);
}

TEST_LIST = {
	{"boolean projection fallback", test_boolean_projection_default_behavior},
	{"edge default and incoming", test_edge_default_and_incoming_direction},
	{"min max any multiedge", test_min_max_any_on_multiedge},
	{"any first invalid edge errors", test_any_errors_on_first_invalid_edge},
	{"weighted rows and compact", test_weighted_rows_and_compact_projection},
	{"rows missing attr errors", test_rows_attr_missing_value_errors_without_default},
	{NULL, NULL}
};
