/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"
#include "src/graph/query_graph.h"
#include "src/graph/entities/qg_node.h"
#include "src/graph/entities/qg_edge.h"
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
static QGEdge *BD;

static QueryGraph *BuildGraph() {
	// (A)->(B)
	// (B)->(C)
	// (B)->(D)
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
	BD = QGEdge_New(relation, "BD");

	QueryGraph *g = QueryGraph_New(node_cap, edge_cap);
	QueryGraph_AddNode(g, A);
	QueryGraph_AddNode(g, B);
	QueryGraph_AddNode(g, C);
	QueryGraph_AddNode(g, D);

	QueryGraph_ConnectNodes(g, A, B, AB);
	QueryGraph_ConnectNodes(g, B, C, BC);
	QueryGraph_ConnectNodes(g, B, D, BD);

	return g;
}

void test_BFSLevels() {
	QGNode *S;                  // BFS starts here.
	QGNode **nodes;             // Nodes reached by BFS.
	QueryGraph *g;              // Graph traversed.
	int level = 0;              // BFS stops when reach level depth.

	g = BuildGraph();
	// S = QueryGraph_GetNodeByID(g, A->id);
	S = A;

	QGNode *expected_level_0[1] = {A};
	QGNode *expected_level_1[1] = {B};
	QGNode *expected_level_2[2] = {C, D};
	QGNode *expected_level_3[0];
	QGNode *expected_level_deepest[2] = {C, D};

	QGNode **expected[4] = {
		expected_level_0,
		expected_level_1,
		expected_level_2,
		expected_level_3
	};

	//------------------------------------------------------------------------------
	// BFS depth 0 - 3
	//------------------------------------------------------------------------------


	for(; level < 4; level++) {
		nodes = BFS(S, &level);
		QGNode **expectation = expected[level];

		int node_count = array_len(nodes);
		for(int i = 0; i < node_count; i++) {
			bool found = false;
			for(int j = 0; j < node_count; j++) {
				if(nodes[i] == expectation[j]) {
					found = true;
					break;
				}
			}
			TEST_ASSERT(found);
		}

		array_free(nodes);
	}

	//------------------------------------------------------------------------------
	// BFS depth BFS_LOWEST_LEVEL
	//------------------------------------------------------------------------------

	level = BFS_LOWEST_LEVEL;
	nodes = BFS(S, &level);

	// Determine number of expected nodes.
	int expected_node_count = sizeof(expected_level_deepest) / sizeof(expected_level_deepest[0]);
	TEST_ASSERT(expected_node_count == array_len(nodes));

	for(int i = 0; i < expected_node_count; i++) {
		bool found = false;
		for(int j = 0; j < expected_node_count; j++) {
			if(nodes[i] == expected_level_deepest[j]) {
				found = true;
				break;
			}
		}
		TEST_ASSERT(found);
	}

	// Clean up.
	array_free(nodes);
	QueryGraph_Free(g);
}

TEST_LIST = {
	{"BFSLevels", test_BFSLevels},
	{NULL, NULL}
};
