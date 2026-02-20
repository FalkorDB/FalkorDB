/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "mock_log.h"
#include "src/query_ctx.h"
#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/errors/errors.h"
#include "src/filter_tree/filter_tree.h"
#include "src/ast/ast_build_filter_tree.h"
#include "src/arithmetic/funcs.h"

#include <stdio.h>
#include <string.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();

#include "acutest.h"

void _fake_graph_context() {
	// filter tree construction requires access to schemas,
	// those inturn resides within graph context
	// accessible via thread local storage, as such we're creating a
	// fake graph context and placing it within thread local storage
	GraphContext *gc = (GraphContext *)calloc(1, sizeof(GraphContext));
	gc->attributes = raxNew();
	pthread_rwlock_init(&gc->_attribute_rwlock, NULL);
	QueryCtx_SetGraphCtx(gc);
}

void setup() {
	Alloc_Reset () ;
	Logging_Reset () ;
	QueryCtx_Init () ;
	ErrorCtx_Init () ;
	AR_RegisterFuncs () ;
	_fake_graph_context () ;
}

void tearDown() {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	raxFree(gc->attributes);
	free(gc);
	QueryCtx_Free();
}

AST *_build_ast
(
	const char *query
) {
	cypher_parse_result_t *parse_result = cypher_parse(query, NULL, NULL,
			CYPHER_PARSE_ONLY_STATEMENTS);
	AST *ast = AST_Build(parse_result);
	return ast;
}

FT_FilterNode *build_tree_from_query
(
	const char *q
) {
	AST *ast = _build_ast(q);
	FT_FilterNode *tree = AST_BuildFilterTree(ast);
	return tree;
}

FT_FilterNode *_build_simple_const_tree() {
	const char *q = "MATCH (me) WHERE me.age = 34 RETURN me";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_simple_varying_tree() {
	const char *q = "MATCH (me), (him) WHERE me.age > him.age RETURN me, him";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_cond_tree
(
	int cond
) {
	const char *q;
	if(cond == OP_AND) q = "MATCH (me) WHERE me.age > 34 AND me.height <= 188 RETURN me";
	else if(cond == OP_XOR) q = "MATCH (me) WHERE me.age > 34 XOR me.height <= 188 RETURN me";
	else q = "MATCH (me) WHERE me.age > 34 OR me.height <= 188 RETURN me";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_explicit_true_tree
(
	FT_FilterNode **expected
) {
	SIValue true_val = SI_BoolVal(true);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(true_val));
	const char *q = "MATCH (n) WHERE TRUE n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_explicit_false_tree
(
	FT_FilterNode **expected
) {
	SIValue false_val = SI_BoolVal(false);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(false_val));
	const char *q = "MATCH (n) WHERE FALSE n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_true_tree
(
	FT_FilterNode **expected
) {
	SIValue true_val = SI_BoolVal(true);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(true_val));
	const char *q = "MATCH (n) WHERE 1 = 1 RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_compactable_or_true_tree
(
	FT_FilterNode **expected
) {
	SIValue true_val = SI_BoolVal(true);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(true_val));
	const char *q = "MATCH (n) WHERE n.val = n.val2 OR 1 = 1 RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_deep_compactable_or_true_tree
(
	FT_FilterNode **expected
) {
	SIValue true_val = SI_BoolVal(true);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(true_val));
	const char *q = "MATCH (n) WHERE n.val = n.val2 OR ( 1 = 1 AND 1 = 1) RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_deep_non_compactable_or_true_tree
(
	FT_FilterNode **expected,
	AST **ast
) {
	const char *expected_q = "MATCH (n) WHERE n.val = n.val2 OR n.val3 = n.val4 RETURN n";
	*expected = build_tree_from_query(expected_q);
	*ast = QueryCtx_GetAST();
	const char *q = "MATCH (n) WHERE n.val = n.val2 OR ( n.val3 = n.val4 AND 1 = 1) RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_false_tree
(
	FT_FilterNode **expected
) {
	SIValue false_val = SI_BoolVal(false);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(false_val));
	const char *q = "MATCH (n) WHERE 1 > 1 RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_compactable_or_false_tree
(
	FT_FilterNode **expected
) {
	SIValue false_val = SI_BoolVal(false);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(false_val));
	const char *q = "MATCH (n) WHERE 1 = 2 OR 1 > 1 RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_non_compactable_or_false_tree
(
	FT_FilterNode **expected,
	AST **ast
) {
	const char *expected_q = "MATCH (n) WHERE n.val = n.val2 RETURN n";
	*expected = build_tree_from_query(expected_q);
	*ast = QueryCtx_GetAST();
	const char *q = "MATCH (n) WHERE n.val = n.val2 OR 1 > 1 RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_deep_compactable_or_false_tree
(
	FT_FilterNode **expected
) {
	SIValue false_val = SI_BoolVal(false);
	*expected = FilterTree_CreateExpressionFilter(
			AR_EXP_NewConstOperandNode(false_val));

	const char *q = "MATCH (n) WHERE 1 > 1 OR ( n.val3 = n.val4 AND 1 > 1) RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_deep_non_compactable_or_false_tree
(
	FT_FilterNode **expected,
	AST **ast
) {
	const char *expected_q = "MATCH (n) WHERE n.val = n.val2 RETURN n";
	*expected = build_tree_from_query(expected_q);
	*ast = QueryCtx_GetAST();
	const char *q = "MATCH (n) WHERE n.val = n.val2 OR ( n.val3 = n.val4 AND 1 > 1) RETURN n";
	return build_tree_from_query(q);
}

FT_FilterNode *_build_AND_cond_tree() {
	return _build_cond_tree(OP_AND);
}

FT_FilterNode *_build_XOR_cond_tree() {
	return _build_cond_tree(OP_XOR);
}

FT_FilterNode *_build_OR_cond_tree() {
	return _build_cond_tree(OP_OR);
}

FT_FilterNode *_build_deep_tree() {
	// build a tree with an OP_AND at the root
	// OP_AND as a left child
	// OP_OR as a right child
	const char *q =
		"MATCH (me),(he),(she),(theirs) WHERE (me.age > 34 AND he.height <= 188) AND (she.age > 34 OR theirs.height <= 188) RETURN me, he, she, theirs";
	return build_tree_from_query(q);
}

void _compareExpressions
(
	AR_ExpNode *a,
	AR_ExpNode *b
) {
	char *a_variable;
	char *b_variable;
	AR_EXP_ToString(a, &a_variable);
	AR_EXP_ToString(b, &b_variable);
	TEST_ASSERT(strcmp(a_variable, b_variable) == 0);

	free(a_variable);
	free(b_variable);
}

void _compareFilterTreePredicateNode
(
	const FT_FilterNode *a,
	const FT_FilterNode *b
) {
	TEST_ASSERT(a->t == b->t);
	TEST_ASSERT(a->t == FT_N_PRED);
	TEST_ASSERT(a->pred.op == b->pred.op);
	_compareExpressions(a->pred.lhs, b->pred.lhs);
	_compareExpressions(a->pred.rhs, b->pred.rhs);
}

void compareFilterTrees
(
	const FT_FilterNode *a,
	const FT_FilterNode *b
) {
	TEST_ASSERT(a->t == b->t);

	if(a->t == FT_N_PRED) {
		_compareFilterTreePredicateNode(a, b);
	} else if(a->t == FT_N_COND) {
		compareFilterTrees(a->cond.left, b->cond.left);
		compareFilterTrees(a->cond.right, b->cond.right);
	} else {
		_compareExpressions(a->exp.exp, b->exp.exp);
	}
}

void test_subTrees() {
	FT_FilterNode *tree = _build_simple_const_tree();
	const FT_FilterNode **sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 1);

	const FT_FilterNode *sub_tree = sub_trees[0];
	compareFilterTrees(tree, sub_tree);

	FilterTree_Free(tree);
	array_free(sub_trees);
	AST *ast = QueryCtx_GetAST();
	AST_Free(ast);

	//--------------------------------------------------------------------------

	tree = _build_simple_varying_tree();
	sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 1);

	sub_tree = sub_trees[0];
	compareFilterTrees(tree, sub_tree);

	array_free(sub_trees);
	FilterTree_Free(tree);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	//--------------------------------------------------------------------------

	FT_FilterNode *original_tree = _build_AND_cond_tree();
	ast = QueryCtx_GetAST();
	tree = _build_AND_cond_tree();
	sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 2);

	sub_tree = sub_trees[0];
	compareFilterTrees(original_tree->cond.left, sub_tree);

	sub_tree = sub_trees[1];
	compareFilterTrees(original_tree->cond.right, sub_tree);

	array_free(sub_trees);
	FilterTree_Free(tree);
	FilterTree_Free(original_tree);
	AST_Free(ast);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	//--------------------------------------------------------------------------

	tree = _build_XOR_cond_tree();
	sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 1);

	sub_tree = sub_trees[0];
	compareFilterTrees(tree, sub_tree);

	array_free(sub_trees);
	FilterTree_Free(tree);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	//--------------------------------------------------------------------------

	tree = _build_OR_cond_tree();
	sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 1);

	sub_tree = sub_trees[0];
	compareFilterTrees(tree, sub_tree);

	array_free(sub_trees);
	FilterTree_Free(tree);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	//--------------------------------------------------------------------------

	original_tree = _build_deep_tree();
	ast = QueryCtx_GetAST();
	tree = _build_deep_tree();
	sub_trees = FilterTree_SubTrees(tree);
	TEST_ASSERT(array_len(sub_trees) == 3);

	sub_tree = sub_trees[0];
	compareFilterTrees(original_tree->cond.left->cond.left, sub_tree);

	sub_tree = sub_trees[1];
	compareFilterTrees(original_tree->cond.left->cond.right, sub_tree);

	sub_tree = sub_trees[2];
	compareFilterTrees(original_tree->cond.right, sub_tree);

	array_free(sub_trees);
	FilterTree_Free(original_tree);
	FilterTree_Free(tree);
	AST_Free(ast);
	ast = QueryCtx_GetAST();
	AST_Free(ast);
}

void test_collectModified() {
	FT_FilterNode *tree = _build_deep_tree();
	rax *aliases = FilterTree_CollectModified(tree);
	TEST_ASSERT(raxSize(aliases) == 4);

	char *expectation[4] = {"me", "he", "she", "theirs"};
	for(int i = 0; i < 4; i++) {
		char *expected = expectation[i];
		TEST_ASSERT(raxFind(aliases, (unsigned char *)expected,
					strlen(expected)) != raxNotFound);
	}

	// clean up
	raxFree(aliases);
	FilterTree_Free(tree);
	AST *ast = QueryCtx_GetAST();
	AST_Free(ast);
}

void test_NOTReduction() {
	const char *filters[8] = {
		"MATCH (n) WHERE NOT n.v = 1 RETURN n",
		"MATCH (n) WHERE NOT NOT n.v = 1 RETURN n",
		"MATCH (n) WHERE NOT (n.v > 5 AND n.v < 20) RETURN n",
		"MATCH (n) WHERE NOT (n.v <= 5 OR n.v <> 20) RETURN n",
		"MATCH (n) WHERE NOT( NOT( NOT( n.v >= 1 AND (n.v < 5 OR n.v <> 3) ) ) ) RETURN n",
		"MATCH (n) WHERE n.v = 2 AND NOT n.x > 4 RETURN n",
		"MATCH (n) WHERE NOT n.v = 1 MATCH (n2) WHERE ID(n2) = ID(n) RETURN n",
		"MATCH (n) WHERE NOT NOT n.v = 1 MATCH (n2) WHERE ID(n2) = ID(n) RETURN n",
	};

	const char *expected[8] = {
		"MATCH (n) WHERE n.v <> 1 RETURN n",
		"MATCH (n) WHERE n.v = 1 RETURN n",
		"MATCH (n) WHERE n.v <= 5 OR n.v >= 20 RETURN n",
		"MATCH (n) WHERE n.v > 5 AND n.v = 20 RETURN n",
		"MATCH (n) WHERE (n.v < 1 OR (n.v >= 5 AND n.v = 3)) RETURN n",
		"MATCH (n) WHERE n.v = 2 AND n.x <= 4 RETURN n",
		"MATCH (n) WHERE n.v <> 1 MATCH (n2) WHERE ID(n2) = ID(n) RETURN n",
		"MATCH (n) WHERE n.v = 1 MATCH (n2) WHERE ID(n2) = ID(n) RETURN n",
	};

	for(int i = 0; i < 8; i++) {
		const char *actual = filters[i];
		const char *expected_q = expected[i];

		FT_FilterNode *actual_tree = build_tree_from_query(actual);
		AST *ast = QueryCtx_GetAST();
		FT_FilterNode *expected_tree = build_tree_from_query(expected_q);

		compareFilterTrees(actual_tree, expected_tree);

		FilterTree_Free(actual_tree);
		FilterTree_Free(expected_tree);
		AST_Free(ast);
		ast = QueryCtx_GetAST();
		AST_Free(ast);
	}
}

void test_containsFunc() {
	bool found = false;
	FT_FilterNode *node = NULL;
	const char *q = "MATCH (n) WHERE tolower(n.name) = 'alex' RETURN n";
	FT_FilterNode *tree = build_tree_from_query(q);

	found = FilterTree_ContainsFunc(tree, "tolower", &node);
	TEST_ASSERT(found);
	TEST_ASSERT(node != NULL);

	node = NULL;
	found = FilterTree_ContainsFunc(tree, "toupper", &node);
	TEST_ASSERT(!found);
	TEST_ASSERT(node == NULL);

	FilterTree_Free(tree);
	AST *ast = QueryCtx_GetAST();
	AST_Free(ast);
	//--------------------------------------------------------------------------

	q = "MATCH (n) WHERE tolower(toupper(n.name)) = 'alex' RETURN n";
	tree = build_tree_from_query(q);

	found = FilterTree_ContainsFunc(tree, "tolower", &node);
	TEST_ASSERT(found);
	TEST_ASSERT(node != NULL);

	node = NULL;
	found = FilterTree_ContainsFunc(tree, "toupper", &node);
	TEST_ASSERT(found);
	TEST_ASSERT(node != NULL);

	FilterTree_Free(tree);
	ast = QueryCtx_GetAST();
	AST_Free(ast);
}

void test_clone() {
	FT_FilterNode *expected;
	FT_FilterNode *actual;

	expected = _build_simple_const_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	AST *ast = QueryCtx_GetAST();
	AST_Free(ast);

	expected = _build_simple_varying_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	expected = _build_AND_cond_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	expected = _build_XOR_cond_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	expected = _build_OR_cond_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	expected = _build_deep_tree();
	actual = FilterTree_Clone(expected);
	compareFilterTrees(expected, actual);
	FilterTree_Free(expected);
	FilterTree_Free(actual);
	ast = QueryCtx_GetAST();
	AST_Free(ast);
}

void test_compact() {
	FT_FilterNode *actual;
	FT_FilterNode *expected;

	// Compactable 'true' trees.
	actual = _build_explicit_true_tree(&expected);
	TEST_ASSERT(FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	AST *ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_true_tree(&expected);
	TEST_ASSERT(FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_compactable_or_true_tree(&expected);
	TEST_ASSERT(FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_deep_compactable_or_true_tree(&expected);
	TEST_ASSERT(FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	// Non compactable 'true' tree.
	actual = _build_deep_non_compactable_or_true_tree(&expected, &ast);
	TEST_ASSERT(!FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	AST_Free(ast);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	// Compactable 'false' trees.
	actual = _build_explicit_false_tree(&expected);
	FilterTree_Compact(actual);
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_false_tree(&expected);
	FilterTree_Compact(actual);
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_compactable_or_false_tree(&expected);
	FilterTree_Compact(actual);
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_deep_compactable_or_false_tree(&expected);
	FilterTree_Compact(actual);
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	// Non compactable 'false' trees.
	actual = _build_non_compactable_or_false_tree(&expected, &ast);
	TEST_ASSERT(!FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	AST_Free(ast);
	ast = QueryCtx_GetAST();
	AST_Free(ast);

	actual = _build_deep_non_compactable_or_false_tree(&expected, &ast);
	TEST_ASSERT(!FilterTree_Compact(actual));
	compareFilterTrees(expected, actual);
	FilterTree_Free(actual);
	FilterTree_Free(expected);
	AST_Free(ast);
	ast = QueryCtx_GetAST();
	AST_Free(ast);
}

TEST_LIST = {
	{"subTrees", test_subTrees},
	{"collectModified", test_collectModified},
	{"NOTReduction", test_NOTReduction},
	{"containsFunc", test_containsFunc},
	{"clone", test_clone},
	{"compact", test_compact},
	{NULL, NULL}
};
