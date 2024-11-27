/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/arr.h"
#include "src/query_ctx.h"
#include "src/util/rmalloc.h"
#include "src/arithmetic/funcs.h"
#include "src/util/thpool/pools.h"
#include "src/procedures/procedure.h"
#include "src/execution_plan/ops/ops.h"
#include "src/execution_plan/execution_plan_clone.h"
#include "src/execution_plan/optimizations/optimizer.h"
#include "src/execution_plan/execution_plan_build/execution_plan_modify.h"

#include <stdio.h>
#include <string.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();

#include "acutest.h"

static ExecutionPlan *build_fake_plan
(
	const char *query  // [OPTIONAL] initial query
) {
	QueryCtx *ctx = QueryCtx_GetQueryCtx();
	if(query == NULL) query = "RETURN 1";

	ctx->query_data.query_no_params = query;
	cypher_parse_result_t *parse_result =
		cypher_parse(query, NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);
	AST_Build(parse_result);
	return ExecutionPlan_FromTLS_AST();
}

static void build_ast_and_plan
(
	const char *query,
	AST **ast,
	ExecutionPlan **plan
) {
	QueryCtx *ctx = QueryCtx_GetQueryCtx();
	ctx->query_data.query_no_params = query;
	cypher_parse_result_t *parse_result =
		cypher_parse(query, NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);

	*ast = AST_Build(parse_result);
	*plan = ExecutionPlan_FromTLS_AST();
	Optimizer_CompileTimeOptimize(*plan);
	Optimizer_RuntimeOptimize(*plan);
}

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
	QueryCtx_SetGraphCtx(gc);
}

static void ExecutionPlan_OpsEqual
(
	const ExecutionPlan *plan_a,
	const ExecutionPlan *plan_b,
	const OpBase *op_a,
	const OpBase *op_b
) {
	// If both ops are NULL, there is nothing to compare.
	if(op_a == NULL && op_b == NULL) return;
	// In case one of the ops is NULL.
	TEST_ASSERT(op_a && op_b);
	// In case both ops are not in their respective segment, there is nothing to compare.
	if(op_a->plan != plan_a && op_b->plan != plan_b) return;
	TEST_ASSERT(op_a->plan == plan_a && op_b->plan == plan_b);
	TEST_ASSERT(op_a->type == op_b->type);
	TEST_ASSERT(op_a->childCount == op_b->childCount);
	for(uint i = 0; i < op_a->childCount; i++) {
		ExecutionPlan_OpsEqual(plan_a, plan_b, op_a->children[i], op_b->children[i]);
	}
}

static void validate_query_plans_clone
(
	const char **queries
) {
	uint query_count = array_len(queries);
	for(uint i = 0; i < query_count; i++) {
		AST *ast = NULL;
		ExecutionPlan *plan = NULL;
		const char *query = queries[i];

		build_ast_and_plan(query, &ast, &plan);
		TEST_ASSERT(ast != NULL);
		TEST_ASSERT(plan != NULL);

		ExecutionPlan *clone = ExecutionPlan_Clone(plan);
		ExecutionPlan_OpsEqual(plan, clone, plan->root, clone->root);

		AST_Free(ast);
		ExecutionPlan_Free(clone);
		ExecutionPlan_Free(plan);
	}
}

void setup() {
	// skip if memory sanitizer is enabled
	if(getenv("SANITIZER") != NULL || getenv("VALGRIND") != NULL) {
		exit(0);
	}

	// use the malloc family for allocations
	Alloc_Reset();

	// Initialize the thread pool.
	TEST_ASSERT(ThreadPools_CreatePools(1, 1, 2));

	// init query context
	TEST_ASSERT(QueryCtx_Init());

	// initialize GraphBLAS
	GrB_init(GrB_NONBLOCKING);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // all matrices in CSR format

	Proc_Register();     // register procedures
	AR_RegisterFuncs();  // register arithmetic functions

	// create a graphcontext
	_fake_graph_context();
}

void tearDown() {
	TEST_ASSERT(GrB_finalize() == GrB_SUCCESS);
	GraphContext *gc = QueryCtx_GetGraphCtx();
	GraphContext_DecreaseRefCount(gc);
	QueryCtx_Free();
}

void test_createClause() {
	const char **queries = array_new(const char *, 12);
	// anonymous nodes create clauses
	array_append(queries, "CREATE ()");
	array_append(queries, "CREATE (:N)");
	array_append(queries, "CREATE (:N {val:1})");

	// referenced nodes create clauses
	array_append(queries, "CREATE (n) RETURN n");
	array_append(queries, "CREATE (n:N) RETURN n");
	array_append(queries, "CREATE (n:N {val:1}) RETURN n");

	// anonymous edges create clauses
	array_append(queries, "CREATE ()-[]->()");
	array_append(queries, "CREATE ()-[:E]->()");
	array_append(queries, "CREATE ()-[:E {val:1}]->()");

	// referenced edges create clauses
	array_append(queries, "CREATE ()-[e]->() RETURN e");
	array_append(queries, "CREATE ()-[e:E]->() RETURN e");
	array_append(queries, "CREATE ()-[e:E {val:1}]->()");
	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_matchClause() {
	const char **queries = array_new(const char *, 9);
	array_append(queries, "MATCH (n) RETURN n");  // All node scan
	array_append(queries, "MATCH (n:N) RETURN n");    // Label scan
	array_append(queries, "MATCH (n) WHERE id(n) = 0 RETURN n");  // ID Scan
	array_append(queries, "MATCH (n)-[]->() RETURN n");    // Conditional traverse, referenced src node.
	array_append(queries, "MATCH (n)-[e]->() RETURN n");   // Conditional traverse, referenced src node and edge.
	array_append(queries, "MATCH p = ()-[]->() RETURN p"); // Named path, conditional traverse
	array_append(queries, "MATCH (n)-[*]->() RETURN n");   // Variable length traverse.
	array_append(queries, "MATCH p = ()-[*]->() return p");    // Named path, variable length traverse.
	array_append(queries, "MATCH (n) WHERE (n)-[:R]->() AND NOT (n)-[:R2]->() RETURN n");   // Apply ops.

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_updateClause() {
	const char **queries = array_new(const char *, 2);
	array_append(queries, "MATCH (n) SET n.v = 1");
	array_append(queries, "MATCH ()-[e]->() SET e.v = 1");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_deleteClause() {
	const char **queries = array_new(const char *, 2);
	array_append(queries, "MATCH (n) DELETE n");
	array_append(queries, "MATCH ()-[e]->() DELETE e");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_mergeClause() {
	const char **queries = array_new(const char *, 9);
	array_append(queries, "MERGE ()");
	array_append(queries, "MERGE (:N)");
	array_append(queries, "MERGE (:N {val:1})");

	array_append(queries, "MERGE (n) ON MATCH SET n.val2 = 2");
	array_append(queries, "MERGE (n:N) ON MATCH SET n.val2 = 2");
	array_append(queries, "MERGE (n:N {val:1}) ON MATCH SET n.val2 = 2");

	array_append(queries, "MERGE (n) ON CREATE SET n.val2 = 2");
	array_append(queries, "MERGE (n:N) ON CREATE SET n.val2 = 2");
	array_append(queries, "MERGE (n:N {val:1}) ON CREATE SET n.val2 = 2");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_cartesProduct() {
	const char **queries = array_new(const char *, 1);
	array_append(queries, "MATCH (a), (b) RETURN a, b");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_skipLimitSort() {
	const char **queries = array_new(const char *, 5);
	array_append(queries, "MATCH (n) RETURN n SKIP 5");
	array_append(queries, "MATCH (n) RETURN n LIMIT 5");
	array_append(queries, "MATCH (n) RETURN n SKIP 5 LIMIT 5");
	array_append(queries, "MATCH (n) RETURN n ORDER BY n.val");
	array_append(queries, "MATCH (n) RETURN n ORDER BY n.val SKIP 5 LIMIT 5");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_optionalMatch() {
	const char **queries = array_new(const char *, 3);
	array_append(queries, "OPTIONAL MATCH (n) RETURN n");
	array_append(queries, "MATCH (a) OPTIONAL MATCH (b) RETURN a, b");
	array_append(queries, "MATCH (a) OPTIONAL MATCH (a)-[e]->(b) RETURN a, e, b");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_procCall() {
	const char **queries = array_new(const char *, 1);
	array_append(queries,
			"CALL db.idx.fulltext.queryNodes('fruit', 'Orange*') YIELD node RETURN node");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_callSubquery() {
	const char **queries = array_new(const char *, 2);
	array_append(queries, "CALL { MATCH (n) RETURN n } RETURN n");
	array_append(queries, "MATCH (m) CALL { CREATE (:M) } RETURN n");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_unwind() {
	const char **queries = array_new(const char *, 1);
	array_append(queries, "UNWIND [1,2,3] as x RETURN x");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_with() {
	const char **queries = array_new(const char *, 7);
	array_append(queries, "MATCH (n) WITH n RETURN n");
	array_append(queries, "MATCH (n) WITH n AS m RETURN m");
	array_append(queries, "MATCH (n) WITH n AS m SKIP 5 RETURN m");
	array_append(queries, "MATCH (n) WITH n AS m LIMIT 5 RETURN m");
	array_append(queries, "MATCH (n) WITH collect(n) AS ns RETURN ns");
	array_append(queries, "MATCH (n) WITH n AS m ORDER BY n.val RETURN m");
	array_append(queries, "MATCH (n) WITH n AS m WHERE n.val < 5 RETURN m");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_union() {
	const char **queries = array_new(const char *, 1);
	array_append(queries, "MATCH (n) RETURN n UNION MATCH (n) RETURN n");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_foreach() {
	const char **queries = array_new(const char *, 2);
	array_append(queries, "FOREACH (x in [1,2,3] | CREATE (:L {val:x}))");
	array_append(queries, "MATCH (n) WITH collect(n) AS ns FOREACH (x in ns | CREATE (:L {val:x.v}))");

	validate_query_plans_clone(queries);
	array_free(queries);
}

void test_apply_multiplexer() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add apply multiplexer to plan
	OpBase *apply_multiplexer = NewApplyMultiplexerOp(plan, OP_OR);
	ExecutionPlan_UpdateRoot(plan, apply_multiplexer);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_edge_by_index_scan() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add edge_index_scan operation to plan
	QGEdge *e = QGEdge_New("R", "e");
	array_append(e->reltypes, 0);

	Index idx = Index_New("R", 0, GETYPE_EDGE);

	AR_ExpNode *exp = AR_EXP_NewVariableOperandNode("a");
	FT_FilterNode *ft = FilterTree_CreateExpressionFilter(exp);

	OpBase *edge_by_index_scan = NewEdgeIndexScanOp(plan, QueryCtx_GetGraph(),
			e, idx, ft);

	ExecutionPlan_UpdateRoot(plan, edge_by_index_scan);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_expand_into() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan("MATCH (a) RETURN a");
	TEST_ASSERT(plan != NULL);

	// add expand_into operation to plan
	AlgebraicExpression *ae = AlgebraicExpression_NewOperand(NULL, false, "a",
			"a", NULL, "a");

	OpBase *expand_into = NewExpandIntoOp(plan, QueryCtx_GetGraph(), ae);
	ExecutionPlan_UpdateRoot(plan, expand_into);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_node_by_id_seek() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add node_by_id_seek operation to plan
	OpBase *node_by_id_seek = NewNodeByIdSeekOp(plan, "a", NULL);
	ExecutionPlan_UpdateRoot(plan, node_by_id_seek);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_node_by_index_scan() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add node_by_index_scan operation to plan
	Index idx = Index_New("L", 0, GETYPE_NODE);
	NodeScanCtx *n = NodeScanCtx_New("a", "L", 0, QGNode_New("a"));

	AR_ExpNode *exp = AR_EXP_NewVariableOperandNode("a");
	FT_FilterNode *ft = FilterTree_CreateExpressionFilter(exp);

	OpBase *node_by_index_scan = NewIndexScanOp(plan, QueryCtx_GetGraph(), n,
			idx, ft);
	ExecutionPlan_UpdateRoot(plan, node_by_index_scan);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_semi_apply() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add semi_apply operation to plan

	OpBase *semi_apply = NewSemiApplyOp(plan, true);
	ExecutionPlan_UpdateRoot(plan, semi_apply);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

void test_value_hash_join() {
	// build a simple plan just so we can add our desiered operation(s) to
	ExecutionPlan *plan = build_fake_plan(NULL);
	TEST_ASSERT(plan != NULL);

	// add semi_apply operation to plan

	AR_ExpNode *lhs = AR_EXP_NewVariableOperandNode("a");
	AR_ExpNode *rhs = AR_EXP_NewVariableOperandNode("b");
	OpBase *value_hash_join = NewValueHashJoin(plan, lhs, rhs);
	ExecutionPlan_UpdateRoot(plan, value_hash_join);

	// free plan and all of its operations
	ExecutionPlan_Free(plan);
}

TEST_LIST = {
	{"createClause", test_createClause},
	{"matchClause", test_matchClause},
	{"updateClause", test_updateClause},
	{"deleteClause", test_deleteClause},
	{"mergeClause", test_mergeClause},
	{"cartesProduct", test_cartesProduct},
	{"skipLimitSort", test_skipLimitSort},
	{"optionalMatch", test_optionalMatch},
	{"procCall", test_procCall},
	{"callSubquery", test_callSubquery},
	{"unwind", test_unwind},
	{"with", test_with},
	{"union", test_union},
	{"foreach", test_foreach},
	{"apply_multiplexer", test_apply_multiplexer},
	{"edge_by_index_scan", test_edge_by_index_scan},
	{"expand_into", test_expand_into},
	{"node_by_id_seek", test_node_by_id_seek},
	{"node_by_index_scan", test_node_by_index_scan},
	{"semi_apply", test_semi_apply},
	{"value_hash_join", test_value_hash_join},
	{NULL, NULL}
};

