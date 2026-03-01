/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"
#include "src/graph/query_graph.h"
#include "src/algorithms/algorithms.h"
#include "GraphBLAS/Include/GraphBLAS.h"

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

QueryGraph *AcyclicBuildGraph() {
	// (A)->(B)->(C)
	size_t node_cap = 3;
	size_t edge_cap = 2;

	// Create nodes.
	const char *relation = "R";

	QGNode *A = QGNode_New("A");
	QGNode *B = QGNode_New("B");
	QGNode *C = QGNode_New("C");

	QGEdge *AB = QGEdge_New(relation, "AB");
	QGEdge *BC = QGEdge_New(relation, "BC");

	QueryGraph *g = QueryGraph_New(node_cap, edge_cap);
	QueryGraph_AddNode(g, A);
	QueryGraph_AddNode(g, B);
	QueryGraph_AddNode(g, C);

	QueryGraph_ConnectNodes(g, A, B, AB);
	QueryGraph_ConnectNodes(g, B, C, BC);

	return g;
}

QueryGraph *CyclicBuildGraph() {
	// (A)->(B)->(C)->(D)->(E)->(B)->(F)->(G)->(C)
	size_t node_cap = 7;
	size_t edge_cap = 8;

	// Create nodes.
	const char *relation = "R";

	QGNode *A = QGNode_New("A");
	QGNode *B = QGNode_New("B");
	QGNode *C = QGNode_New("C");
	QGNode *D = QGNode_New("D");
	QGNode *E = QGNode_New("E");
	QGNode *F = QGNode_New("F");
	QGNode *G = QGNode_New("G");

	// (A)->(B)
	QGEdge *AB = QGEdge_New(relation, "AB");
	// (B)->(C)
	QGEdge *BC = QGEdge_New(relation, "BC");
	// (C)->(D)
	QGEdge *CD = QGEdge_New(relation, "CD");
	// (D)->(E)
	QGEdge *DE = QGEdge_New(relation, "DE");
	// (E)->(B)
	QGEdge *EB = QGEdge_New(relation, "EB");
	// (B)->(F)
	QGEdge *BF = QGEdge_New(relation, "BF");
	// (F)->(G)
	QGEdge *FG = QGEdge_New(relation, "FG");
	// (G)->(C)
	QGEdge *GC = QGEdge_New(relation, "GC");

	QueryGraph *g = QueryGraph_New(node_cap, edge_cap);
	QueryGraph_AddNode(g, A);
	QueryGraph_AddNode(g, B);
	QueryGraph_AddNode(g, C);
	QueryGraph_AddNode(g, D);
	QueryGraph_AddNode(g, E);
	QueryGraph_AddNode(g, F);
	QueryGraph_AddNode(g, G);

	QueryGraph_ConnectNodes(g, A, B, AB);
	QueryGraph_ConnectNodes(g, B, C, BC);
	QueryGraph_ConnectNodes(g, C, D, CD);
	QueryGraph_ConnectNodes(g, D, E, DE);
	QueryGraph_ConnectNodes(g, E, B, EB);
	QueryGraph_ConnectNodes(g, B, F, BF);
	QueryGraph_ConnectNodes(g, F, G, FG);
	QueryGraph_ConnectNodes(g, G, C, GC);

	return g;
}

void setup() {
	// Use the malloc family for allocations
	Alloc_Reset();
	Logging_Reset();

	// Initialize GraphBLAS
	GrB_init(GrB_NONBLOCKING);
	GxB_set(GxB_FORMAT, GxB_BY_ROW);
}

void tearDown() {
	GrB_finalize();
}

void test_acyclicGraph() {
	QueryGraph *g = AcyclicBuildGraph();  // Graph traversed.
	TEST_ASSERT(IsAcyclicGraph(g) == true);

	// clean up
	QueryGraph_Free(g);
}

void test_cyclicGraph() {
	QueryGraph *g = CyclicBuildGraph();  // Graph traversed.
	TEST_ASSERT(IsAcyclicGraph(g) == false);

	// clean up
	QueryGraph_Free(g);
}

TEST_LIST = {
	{ "acyclicGraph", test_acyclicGraph},
	{ "cyclicGraph", test_cyclicGraph},
	{ NULL, NULL }
};

