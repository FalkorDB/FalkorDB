/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// PR #1907 defensive bounds-check on GraphContext_GetSchemaByID.
//
// Background:
//   On a replica, GRAPH.EFFECT may carry a relation/label schema ID that
//   does not yet exist locally if master/replica state has diverged.
//   Before PR #1907 the lookup performed a raw `schemas[id]` array read
//   with no bounds check, returning garbage that callers then dereferenced
//   as a Schema* (Schema_HasIndices(NULL/garbage) -> SIGSEGV).
//
//   PR #1907 hardens GraphContext_GetSchemaByID to return NULL for any
//   out-of-range ID (negative, >= arr_len(schemas), and the documented
//   sentinel GRAPH_NO_LABEL).  Callers (GraphHub_CreateEdges, ApplyLabels)
//   detect the NULL and call _exit(1) instead of crashing.
//
// Why a unit test:
//   With current master both historical desync mechanisms are closed
//   (PR #1815 - failed-query orphan schema rollback;  PR #1877 - Yield
//   no longer fires during normal replication).  No realistic Cypher-
//   level operation can produce the desync from a Python flow test, so
//   this unit test directly drives GraphContext_GetSchemaByID with
//   in-range, out-of-range and sentinel IDs to verify the defensive
//   bounds-check works as documented.

#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/graph/graph.h"
#include "src/graph/graphcontext.h"
#include "src/graph/graphcontext_struct.h"
#include "src/schema/schema.h"
#include "src/queries_log/queries_log.h"
#include "GraphBLAS/Include/GraphBLAS.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

void setup() {
	Alloc_Reset();
	GrB_init(GrB_NONBLOCKING);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
}

void tearDown() {
	GrB_finalize();
}

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

// Build a minimal GraphContext suitable for exercising
// GraphContext_GetSchemaByID without dragging in the full module-load path.
// Mirrors the _fake_graph_context() helper used by test_algebraic_expression.c.
static GraphContext *_fake_graph_context(void) {
	GraphContext *gc = (GraphContext *)calloc(1, sizeof(GraphContext));

	gc->g                = Graph_New(16, 16);
	gc->ref_count        = 1;
	gc->index_count      = 0;
	gc->graph_name       = strdup("G");
	gc->attributes       = raxNew();
	gc->string_mapping   = (char**)arr_new(char*,  64);
	gc->node_schemas     = (Schema**)arr_new(Schema*, GRAPH_DEFAULT_LABEL_CAP);
	gc->relation_schemas = (Schema**)arr_new(Schema*, GRAPH_DEFAULT_RELATION_TYPE_CAP);
	gc->queries_log      = QueriesLog_New();

	pthread_rwlock_init(&gc->_schema_rwlock, NULL);

	return gc;
}

// In-range schema IDs return the expected Schema*.
void test_get_schema_by_id_in_range_returns_schema(void) {
	GraphContext *gc = _fake_graph_context();

	GraphContext_AddSchema(gc, "L0",   SCHEMA_NODE);
	GraphContext_AddSchema(gc, "L1",   SCHEMA_NODE);
	GraphContext_AddSchema(gc, "REL0", SCHEMA_EDGE);
	GraphContext_AddSchema(gc, "REL1", SCHEMA_EDGE);

	Schema *s;

	s = GraphContext_GetSchemaByID(gc, 0, SCHEMA_NODE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "L0") == 0);

	s = GraphContext_GetSchemaByID(gc, 1, SCHEMA_NODE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "L1") == 0);

	s = GraphContext_GetSchemaByID(gc, 0, SCHEMA_EDGE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "REL0") == 0);

	s = GraphContext_GetSchemaByID(gc, 1, SCHEMA_EDGE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "REL1") == 0);
}

// Out-of-range positive IDs (>= arr_len(schemas)) must return NULL,
// not OOB-read the array.
void test_get_schema_by_id_oob_positive_returns_null(void) {
	GraphContext *gc = _fake_graph_context();

	// Two node schemas (ids 0,1) and one edge schema (id 0).
	GraphContext_AddSchema(gc, "L0",   SCHEMA_NODE);
	GraphContext_AddSchema(gc, "L1",   SCHEMA_NODE);
	GraphContext_AddSchema(gc, "REL0", SCHEMA_EDGE);

	// id == arr_len: just past the end (the exact failure mode in the
	// customer crash - master had one more schema than replica).
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 2, SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 1, SCHEMA_EDGE) == NULL);

	// id far past the end.
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 999,    SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 999999, SCHEMA_EDGE) == NULL);
}

// Negative IDs other than the GRAPH_NO_LABEL sentinel must also be
// rejected - signed-to-unsigned coercion of a stray negative id used to
// produce a huge index that read garbage from the schemas array.
void test_get_schema_by_id_negative_returns_null(void) {
	GraphContext *gc = _fake_graph_context();

	GraphContext_AddSchema(gc, "L0",   SCHEMA_NODE);
	GraphContext_AddSchema(gc, "REL0", SCHEMA_EDGE);

	// GRAPH_NO_LABEL == -1 is the documented sentinel; it has always
	// returned NULL via the early-return branch above the bounds check.
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, GRAPH_NO_LABEL, SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, GRAPH_NO_LABEL, SCHEMA_EDGE) == NULL);

	// Other negative IDs must be rejected by the new bounds check rather
	// than coerced to UINT_MAX-ish offsets that index into the schemas
	// array.
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, -2,  SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, -42, SCHEMA_EDGE) == NULL);
}

// On an empty schema array every positive ID must come back NULL.
// This is the literal replica-state-on-first-EFFECT case.
void test_get_schema_by_id_empty_arrays_returns_null(void) {
	GraphContext *gc = _fake_graph_context();

	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 0,   SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 0,   SCHEMA_EDGE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 1,   SCHEMA_NODE) == NULL);
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 100, SCHEMA_EDGE) == NULL);
}

// Boundary: id == arr_len - 1 is the LAST valid index; id == arr_len is
// the FIRST invalid one.  Catches an off-by-one regression in the bounds
// check itself.
void test_get_schema_by_id_boundary(void) {
	GraphContext *gc = _fake_graph_context();

	GraphContext_AddSchema(gc, "L0", SCHEMA_NODE);
	GraphContext_AddSchema(gc, "L1", SCHEMA_NODE);
	GraphContext_AddSchema(gc, "L2", SCHEMA_NODE);

	// last valid id - must succeed
	Schema *s = GraphContext_GetSchemaByID(gc, 2, SCHEMA_NODE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "L2") == 0);

	// first invalid id - must be rejected
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 3, SCHEMA_NODE) == NULL);
}

// SchemaType selection: a relation id that is in range for the EDGE array
// but out of range for the NODE array (or vice versa) must respect the
// selected type.  Prevents a regression where the bounds check looks at
// the wrong schema array.
void test_get_schema_by_id_type_dispatch(void) {
	GraphContext *gc = _fake_graph_context();

	GraphContext_AddSchema(gc, "L0",   SCHEMA_NODE);   // node id 0
	GraphContext_AddSchema(gc, "REL0", SCHEMA_EDGE);   // edge id 0
	GraphContext_AddSchema(gc, "REL1", SCHEMA_EDGE);   // edge id 1

	// edge id 1 is valid against the EDGE array
	Schema *s = GraphContext_GetSchemaByID(gc, 1, SCHEMA_EDGE);
	TEST_ASSERT(s != NULL);
	TEST_ASSERT(strcmp(Schema_GetName(s), "REL1") == 0);

	// edge id 1 is OUT OF RANGE against the NODE array (only 1 node
	// schema exists) - must return NULL, not a stale Schema*.
	TEST_ASSERT(GraphContext_GetSchemaByID(gc, 1, SCHEMA_NODE) == NULL);
}

TEST_LIST = {
	{ "get_schema_by_id_in_range_returns_schema",
	  test_get_schema_by_id_in_range_returns_schema },
	{ "get_schema_by_id_oob_positive_returns_null",
	  test_get_schema_by_id_oob_positive_returns_null },
	{ "get_schema_by_id_negative_returns_null",
	  test_get_schema_by_id_negative_returns_null },
	{ "get_schema_by_id_empty_arrays_returns_null",
	  test_get_schema_by_id_empty_arrays_returns_null },
	{ "get_schema_by_id_boundary",
	  test_get_schema_by_id_boundary },
	{ "get_schema_by_id_type_dispatch",
	  test_get_schema_by_id_type_dispatch },
	{ NULL, NULL }
};
