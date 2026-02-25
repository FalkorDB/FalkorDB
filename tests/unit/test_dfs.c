/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"
#include "src/graph/query_graph.h"
#include "src/algorithms/algorithms.h"

void setup() {
	Alloc_Reset();
	Logging_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

static QGNode *A;
static QGNode *B;
static QGNode *C;
static QGNode *D;
static QGEdge *AB;
static QGEdge *BC;
static QGEdge *CD;

static QueryGraph *BuildGraph() {
	// (A)->(B)
	// (B)->(C)
	// (C)->(D)
	size_t node_cap = 4;
	size_t edge_cap = 3;

	// Create nodes.
	const char *relation = "R";

	A = QGNode_New("A");
	B = QGNode_New("B");
	C = QGNode_New("C");
	D = QGNode_New("D");

	AB = QGEdge_New(relation, "AB");
	BC = QGEdge_New(relation, "BC");
	CD = QGEdge_New(relation, "CD");

	QueryGraph *g = QueryGraph_New(node_cap, edge_cap);
	QueryGraph_AddNode(g, A);
	QueryGraph_AddNode(g, B);
	QueryGraph_AddNode(g, C);
	QueryGraph_AddNode(g, D);

	QueryGraph_ConnectNodes(g, A, B, AB);
	QueryGraph_ConnectNodes(g, B, C, BC);
	QueryGraph_ConnectNodes(g, C, D, CD);

	return g;
}

void test_DFSLevels() {
	QGNode *S;      // DFS starts here.
	QGEdge **path;  // Path reached by DFS.
	QueryGraph *g;  // Graph traversed.

	g = BuildGraph();
	S = QueryGraph_GetNodeByAlias(g, "A");

	QGEdge *expected_level_0[0] = {};
	QGEdge *expected_level_1[1] = {AB};
	QGEdge *expected_level_2[2] = {AB, BC};
	QGEdge *expected_level_3[3] = {AB, BC, CD};
	QGEdge *expected_level_4[0] = {};

	QGEdge **expected[5] = {
		expected_level_0,
		expected_level_1,
		expected_level_2,
		expected_level_3,
		expected_level_4
	};

	//------------------------------------------------------------------------------
	// DFS depth 0 - 4
	//------------------------------------------------------------------------------

	for(int level = 0; level < 5; level++) {
		path = DFS(S, level, true);
		QGEdge **expectation = expected[level];

		int edge_count = array_len(path);
		for(int i = 0; i < edge_count; i++) {
			bool found = false;
			for(int j = 0; j < edge_count; j++) {
				if(path[i] == expectation[j]) {
					found = true;
					break;
				}
			}
			TEST_ASSERT(found);
		}
		array_free(path);
	}

	// Clean up.
	QueryGraph_Free(g);
}

TEST_LIST = {
	{"DFSLevels", test_DFSLevels},
	{NULL, NULL}
};
