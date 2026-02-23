/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/arr.h"
#include "src/ast/ast.h"

#include <stdio.h>

void setup() {
	// skip if memory sanitizer is enabled
	if(getenv("SANITIZER") != NULL || getenv("VALGRIND") != NULL) {
		exit(0);
	}
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

// parse query to AST
AST *buildAST
(
	const char *query
) {
	cypher_parse_result_t *parse_result =
		cypher_parse(query, NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);
	return AST_Build(parse_result);
}

uint *getASTSegmentIndices(AST *ast) {
	// retrieve the indices of each WITH clause
	// to properly set the bounds of each segment
	uint *segment_indices = AST_GetClauseIndices(ast, CYPHER_AST_WITH);
	array_append(segment_indices, cypher_ast_query_nclauses(ast->root));
	return segment_indices;
}

void test_match() {
	// test for no inline filters
	char *q = "MATCH (n)";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// Inline node label filter.
	q = "MATCH (n:X)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// Inline node property filter.
	q = "MATCH (n {val:42})-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// Inline edge relationship type.
	q = "MATCH (n)-[e:T]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// Inline edge properties.
	q = "MATCH (n)-[e {val:42}]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// Explicit filters: EQ, NE, GT, LT, IS NULL, IS NOT NULL.
	q = "MATCH (n)-[e]->(m) WHERE n.v = 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m) WHERE n.v <> 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m) WHERE n.v > 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m) WHERE n.v < 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m) WHERE n.v IS NULL 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n)-[e]->(m) WHERE n.v IS NOT NULL 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_set() {
	// simple match and set property
	char *q = "MATCH (n) SET n.v=1";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// path match and set property
	q = "MATCH (n)-[e]->(x)-[]->(m) SET n.v=x.v";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// reference from MATCH and SET caluse
	q = "MATCH (n)-[e:r]->(m) SET n.v=1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_merge() {
	// simple merge
	char *q = "MERGE (n)";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MERGE (n)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// inline node label filter
	q = "MERGE (n:X)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// inline node property filter
	q = "MERGE (n {val:42})-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// inline edge relationship type
	q = "MERGE (n)-[e:T]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// inline edge properties
	q = "MERGE (n)-[e {val:42}]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// on match
	q = "MERGE (n)-[e]->(m) ON MATCH SET n.v=1, m.v=2";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// on create
	q = "MERGE (n)-[e]->(m) ON CREATE SET n.v=1, m.v=2";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	// mixed
	q = "MERGE (n)-[e]->(m) ON CREATE SET n.v=1, m.v=2 ON MATCH SET e.v = 3";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(3 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_unwind() {
	char *q = "UNWIND [1,2] as x";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(0 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_with() {
	char *q = "MATCH (n),(m) WITH n as x RETURN x";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	// two segments: first is the MATCH clause, the second is the WITH clause
	TEST_ASSERT(2 == array_len(segmentIndices));

	// only n is projected from the first segment
	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	AST_Free(astSegment);

	// only x is projected from the second segment
	astSegment = AST_NewSegment(ast, segmentIndices[0], segmentIndices[1]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n),(m) WITH n as x ORDER BY m.value RETURN x";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	// two segments: first is the MATCH clause, the second is the WITH clause
	TEST_ASSERT(2 == array_len(segmentIndices));

	// only n and m projected from the first segment
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	AST_Free(astSegment);

	// only m and x projected from the second segment
	astSegment = AST_NewSegment(ast, segmentIndices[0], segmentIndices[1]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound == raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_return() {
	char *q = "MATCH (n),(m) RETURN n as x";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	// one segment containing MATCH and RETURN clause
	TEST_ASSERT(1 == array_len(segmentIndices));

	// n and x are both accessible in the only segment
	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(2 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH (n),(m) RETURN n as x ORDER BY m.value";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	// one segment containing MATCH and RETURN clause
	TEST_ASSERT(1 == array_len(segmentIndices));

	// all variables are accessible in the only segment
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(3 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"x", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

void test_namePath() {
	char *q = "MATCH p=()";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(1 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"@anon_0", 7));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH p =()-[]-()";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(3 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"@anon_0", 7));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"@anon_1", 7));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"@anon_2", 7));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH p =(n)-[e]-(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(3 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);

	q = "MATCH p =(:Label1 {value:1})-[e:Rel*]-(m:Label2 {value:2})";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	TEST_ASSERT(1 == array_len(segmentIndices));

	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	TEST_ASSERT(3 == raxSize(astSegment->referenced_entities));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"@anon_0", 7));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	TEST_ASSERT(raxNotFound != raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
	array_free(segmentIndices);
	AST_Free(astSegment);
	AST_Free(ast);
}

TEST_LIST = {
	{"match", test_match},
	{"set", test_set},
	{"merge", test_merge},
	{"unwind", test_unwind},
	{"with", test_with},
	{"return", test_return},
	{"namePath", test_namePath},
	{NULL, NULL}
};
