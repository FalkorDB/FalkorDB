/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/algorithms/algorithms.h"

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();

#include "acutest.h"

static Graph *BuildGraph() {
	Edge e;
	Node n;
	size_t nodeCount = 4;
	size_t edgeCount = 7;
	Graph *g = Graph_New(nodeCount, edgeCount);
	int relation = Graph_AddRelationType(g);
	for(int i = 0; i < 4; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}

	/* Connections:
		* 0 -> 1
		* 0 -> 2
		* 1 -> 0
		* 1 -> 2
		* 2 -> 1
		* 2 -> 3
		* 3 -> 0 */

	// Connections:
	// 0 -> 1
	Graph_CreateEdge(g, 0, 1, relation, &e);
	// 0 -> 2
	Graph_CreateEdge(g, 0, 2, relation, &e);
	// 1 -> 0
	Graph_CreateEdge(g, 1, 0, relation, &e);
	// 1 -> 2
	Graph_CreateEdge(g, 1, 2, relation, &e);
	// 2 -> 1
	Graph_CreateEdge(g, 2, 1, relation, &e);
	// 2 -> 3
	Graph_CreateEdge(g, 2, 3, relation, &e);
	// 3 -> 0
	Graph_CreateEdge(g, 3, 0, relation, &e);
	return g;
}

// This function tests for membership of a path, inside an array of multiple paths.
bool pathArrayContainsPath(NodeID **array, int arrayLen, Path *path) {
	for(int i = 0; i < arrayLen; i++) {
		NodeID *expectedPath = array[i];
		int expectedPathLen = expectedPath[0];
		if(expectedPathLen != Path_NodeCount(path)) {
			continue;
		}
		bool arrayContainsPath = true;
		for(int j = 1; j <= expectedPathLen; j++) {
			Node n = path->nodes[j - 1];
			if(ENTITY_GET_ID(&n) != expectedPath[j]) {
				arrayContainsPath = false;
				break;
			}
		}
		if(arrayContainsPath) return true;
	}
	return false;
}

void setup() {
	// Use the malloc family for allocations
	Alloc_Reset();

	// Initialize GraphBLAS.
	GrB_init(GrB_NONBLOCKING);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);     // all matrices in CSR format
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER); // matrices are never hypersparse
}

void tearDown() {
	GrB_finalize();
}

void test_noPaths() {
	Graph *g = BuildGraph();

	NodeID       src_id = 0;
	unsigned int minLen = 999;
	unsigned int maxLen = minLen + 1;

	Node src;
	Graph_GetNode(g, src_id, &src);

	int relationships[] = {GRAPH_NO_RELATION};
	AllPathsCtx *ctx = AllPathsCtx_New(&src, NULL, g, relationships, 1,
		GRAPH_EDGE_DIR_OUTGOING, minLen, maxLen, NULL, NULL, 0, false);
	Path *p = AllPathsCtx_NextPath(ctx);

	TEST_ASSERT (p == NULL);

	AllPathsCtx_Free(ctx);
	Graph_Free(g);
}

void test_longest_Paths() {
	Graph *g = BuildGraph();

	NodeID src_id = 0;
	Node src;
	Graph_GetNode(g, src_id, &src);

	unsigned int minLen = 0;
	unsigned int maxLen = UINT_MAX - 2;
	int relationships[] = {GRAPH_NO_RELATION};
	AllPathsCtx *ctx = AllPathsCtx_New(&src, NULL, g, relationships, 1,
		GRAPH_EDGE_DIR_OUTGOING, minLen, maxLen, NULL, NULL, 0, false);
	Path *path;

	unsigned int longestPath = 0;
	while((path = AllPathsCtx_NextPath(ctx))) {
		size_t pathLen = Path_Len(path);
		if(longestPath < pathLen) longestPath = pathLen;
	}

	// Under edge-based semantics, the longest path uses all 7 edges:
	// 0->1->2->3->0->2->1->0  (e0,e3,e5,e6,e1,e4,e2)
	TEST_CHECK (longestPath == 7);

	AllPathsCtx_Free(ctx);
	Graph_Free(g);
}

void test_upToThreeLegsPaths() {
	Graph *g = BuildGraph();

	NodeID src_id = 0;
	Node src;
	Graph_GetNode(g, src_id, &src);

	Path *path = NULL;
	unsigned int minLen = 0;
	unsigned int maxLen = 3;
	uint pathsCount = 0;
	int relationships[] = {GRAPH_NO_RELATION};
	AllPathsCtx *ctx = AllPathsCtx_New(&src, NULL, g, relationships, 1,
		GRAPH_EDGE_DIR_OUTGOING, minLen, maxLen, NULL, NULL, 0, false);

	/* Connections:
	 * 0 -> 1
	 * 0 -> 2
	 * 1 -> 0
	 * 1 -> 2
	 * 2 -> 1
	 * 2 -> 3
	 * 3 -> 0 */

	// Zero leg paths.
	NodeID p0[2] = {0, 0};

	// One leg paths.
	NodeID p1[3] = {1, 0, 1};
	NodeID p2[3] = {1, 0, 2};

	// Two leg paths.
	NodeID p3[4] = {2, 0, 1, 0};
	NodeID p4[4] = {2, 0, 1, 2};
	NodeID p5[4] = {2, 0, 2, 1};
	NodeID p6[4] = {2, 0, 2, 3};

	// Three leg paths.
	// Under edge-based semantics, 0->1->0->2 is now valid (edges e0,e2,e1 are distinct).
	NodeID p7[5] = {3, 0, 1, 0, 2};
	NodeID p8[5] = {3, 0, 1, 2, 1};
	NodeID p9[5] = {3, 0, 1, 2, 3};
	NodeID p10[5] = {3, 0, 2, 1, 0};
	NodeID p11[5] = {3, 0, 2, 1, 2};
	NodeID p12[5] = {3, 0, 2, 3, 0};

	NodeID *expectedPaths[13] = {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12};

	while((path = AllPathsCtx_NextPath(ctx))) {
		bool expectedPathFound = false;
		assert(pathsCount < 13);

		// Try to match a path.
		for(int i = 0; i < 13; i++) {
			NodeID *expectedPath = expectedPaths[i];
			size_t expectedPathLen = expectedPath[0];
			expectedPath++; // Skip path length.

			if(Path_Len(path) != expectedPathLen) continue;

			int j = 0;
			for(; j < expectedPathLen; j++) {
				Node n = path->nodes[j];
				if(ENTITY_GET_ID(&n) != expectedPath[j]) break;
			}
			if(j == expectedPathLen) {
				expectedPathFound = true;
				break;
			}
		}
		TEST_CHECK (expectedPathFound);
		pathsCount++;
	}
	TEST_ASSERT (pathsCount == 13);

	AllPathsCtx_Free(ctx);
	Graph_Free(g);
}

void test_twoLegPaths() {
	Graph *g = BuildGraph();

	NodeID src_id = 0;
	Node src;
	Path *path = NULL;
	Graph_GetNode(g, src_id, &src);
	unsigned int minLen = 2;
	unsigned int maxLen = 2;
	unsigned int pathsCount = 0;
	int relationships[] = {GRAPH_NO_RELATION};
	AllPathsCtx *ctx = AllPathsCtx_New(&src, NULL, g, relationships, 1,
		GRAPH_EDGE_DIR_OUTGOING, minLen, maxLen, NULL, NULL, 0, false);
	/* Connections:
	 * 0 -> 1
	 * 0 -> 2
	 * 1 -> 0
	 * 1 -> 2
	 * 2 -> 1
	 * 2 -> 3
	 * 3 -> 0 */
	NodeID p0[3] = {0, 1, 0};
	NodeID p1[3] = {0, 1, 2};
	NodeID p2[3] = {0, 2, 1};
	NodeID p3[3] = {0, 2, 3};
	NodeID *expectedPaths[4] = {p0, p1, p2, p3};

	while((path = AllPathsCtx_NextPath(ctx))) {
		TEST_CHECK (pathsCount < 4);
		TEST_CHECK (Path_Len(path) == 2);
		bool expectedPathFound = false;

		for(int i = 0; i < 4; i++) {
			NodeID *expectedPath = expectedPaths[i];
			int j;
			for(j = 0; j < 3; j++) {
				Node n = path->nodes[j];
				if(ENTITY_GET_ID(&n) != expectedPath[j]) break;
			}
			expectedPathFound = (j == 3);
			if(expectedPathFound) break;
		}

		TEST_CHECK (expectedPathFound);
		pathsCount++;
	}

	TEST_ASSERT (pathsCount == 4);

	AllPathsCtx_Free(ctx);
	Graph_Free(g);
}

// Test all paths from source to a specific destination node.
void test_destinationSpecificPaths() {
	/* Under edge-based semantics, paths from 0 back to 0 (all 15):
	 * len=0: {0}
	 * len=2: {0,1,0}
	 * len=3: {0,2,1,0}, {0,2,3,0}
	 * len=4: {0,1,2,1,0}, {0,1,2,3,0}
	 * len=5: {0,1,0,2,3,0}, {0,2,1,2,3,0}, {0,2,3,0,1,0}
	 * len=7: {0,1,0,2,1,2,3,0}, {0,1,2,1,0,2,3,0}, {0,1,2,3,0,2,1,0},
	 *        {0,2,1,0,1,2,3,0}, {0,2,1,2,3,0,1,0}, {0,2,3,0,1,2,1,0}
	 */
	NodeID p00_0[2]  = {1, 0};
	NodeID p00_1[4]  = {3, 0, 1, 0};
	NodeID p00_2[5]  = {4, 0, 2, 1, 0};
	NodeID p00_3[5]  = {4, 0, 2, 3, 0};
	NodeID p00_4[6]  = {5, 0, 1, 2, 1, 0};
	NodeID p00_5[6]  = {5, 0, 1, 2, 3, 0};
	NodeID p00_6[7]  = {6, 0, 1, 0, 2, 3, 0};
	NodeID p00_7[7]  = {6, 0, 2, 1, 2, 3, 0};
	NodeID p00_8[7]  = {6, 0, 2, 3, 0, 1, 0};
	NodeID p00_9[9]  = {8, 0, 1, 0, 2, 1, 2, 3, 0};
	NodeID p00_10[9] = {8, 0, 1, 2, 1, 0, 2, 3, 0};
	NodeID p00_11[9] = {8, 0, 1, 2, 3, 0, 2, 1, 0};
	NodeID p00_12[9] = {8, 0, 2, 1, 0, 1, 2, 3, 0};
	NodeID p00_13[9] = {8, 0, 2, 1, 2, 3, 0, 1, 0};
	NodeID p00_14[9] = {8, 0, 2, 3, 0, 1, 2, 1, 0};

	NodeID *p00[15] = {
		p00_0, p00_1, p00_2, p00_3, p00_4,
		p00_5, p00_6, p00_7, p00_8, p00_9,
		p00_10, p00_11, p00_12, p00_13, p00_14
	};

	Graph *g = BuildGraph();

	NodeID src_id = 0;
	Node src;
	Path *path = NULL;
	Graph_GetNode(g, src_id, &src);
	unsigned int minLen = 0;
	unsigned int maxLen = UINT_MAX - 2;
	unsigned int pathsCount = 0;
	int relationships[] = {GRAPH_NO_RELATION};
	AllPathsCtx *ctx = AllPathsCtx_New(&src, &src, g, relationships, 1,
		GRAPH_EDGE_DIR_OUTGOING, minLen, maxLen, NULL, NULL, 0, false);

	while((path = AllPathsCtx_NextPath(ctx))) {
		TEST_CHECK(pathsCount < 15);
		TEST_CHECK(pathArrayContainsPath(p00, 15, path));
		pathsCount++;
	}

	TEST_ASSERT (pathsCount == 15);

	AllPathsCtx_Free(ctx);
	Graph_Free(g);
}

TEST_LIST = {
	{"noPaths", test_noPaths},
	{"longest_Paths", test_longest_Paths},
	{"upToThreeLegsPaths", test_upToThreeLegsPaths},
	{"twoLegPaths", test_twoLegPaths},
	{"destinationSpecificPaths", test_destinationSpecificPaths},
	{NULL, NULL}
};
