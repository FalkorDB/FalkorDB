/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "mock_log.h"
#include "src/util/arr.h"
#include "src/graph/graph.h"
#include "src/util/rmalloc.h"
#include "src/util/simple_timer.h"
#include "src/configuration/config.h"
#include "GraphBLAS/Include/GraphBLAS.h"

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

// Console text colors for benchmark printing
#define KGRN "\x1B[32m"
#define KRED "\x1B[31m"
#define KNRM "\x1B[0m"

#define GRAPH_DEFAULT_NODE_CAP 16384
#define GRAPH_DEFAULT_EDGE_CAP 16384

// Encapsulate the essence of an edge.
typedef struct {
	NodeID srcId;       // Source node ID.
	NodeID destId;      // Destination node ID.
	int64_t relationId; // Relation type ID.
} EdgeDesc;

void _test_node_creation(Graph *g, size_t node_count) {
	GrB_Index ncols, nrows, nvals;

	// Create nodes.
	Node n;
	for(uint i = 0; i < node_count; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}

	// Validate nodes creation.
	Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);
	TEST_ASSERT(Delta_Matrix_nrows(&nrows, adj) == GrB_SUCCESS);
	TEST_ASSERT(Delta_Matrix_ncols(&ncols, adj) == GrB_SUCCESS);
	TEST_ASSERT(Delta_Matrix_nvals(&nvals, adj) == GrB_SUCCESS);

	TEST_ASSERT(nvals == 0);                  // No connection were formed.
	TEST_ASSERT(ncols >= Graph_NodeCount(g)); // Graph's adjacency matrix dimensions.
	TEST_ASSERT(nrows >= Graph_NodeCount(g));
	TEST_ASSERT(Graph_NodeCount(g) == node_count);
}

/* TODO benchmark functions are currently not invoked */
void benchmark_node_creation_with_labels() {
	printf("benchmark_node_creation_with_labels\n");
	double tic[2];
	int samples = 64;
	int label_count = 3;
	double timings[samples];
	int outliers = 0;
	float threshold = 0.0018;
	// size_t n = GRAPH_DEFAULT_NODE_CAP;
	size_t n = 1000000;
	Graph *g = Graph_New("g", n, n);
	Graph_AcquireWriteLock(g);

	// Introduce labels and relations to graph.
	for(int i = 0; i < label_count; i++) {
		Graph_AddRelationType(g); // Typed relation.
		Graph_AddLabel(g);    // Typed node.
	}

	Node node;

	// Create N nodes with labels.
	for(int i = 0; i < samples; i++) {
		simple_tic(tic);
		for(unsigned int j = 0; j < n; j++) {
			node = GE_NEW_NODE();
			Graph_CreateNode(g, &node, NULL, 0);
		}
		timings[i] = simple_toc(tic);

		printf("%zu Nodes created, time: %.6f sec\n", n, timings[i]);
		if(timings[i] > threshold)
			outliers++;
	}

	if(outliers > samples * 0.1) {
		printf("Node creation took too long\n");
		for(int i = 0; i < samples; i++) {
			printf("%zu Nodes created, time: %.6f sec\n", n, timings[i]);
		}
		// assert(false);
	}

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

// Test graph creation time.
void benchmark_node_creation_no_labels() {
	printf("benchmark_node_creation_no_labels\n");
	double tic[2];
	int samples = 64;
	double timings[samples];
	int outliers = 0;
	float threshold = 0.000006;
	// size_t n = GRAPH_DEFAULT_NODE_CAP;
	Node node;
	size_t n = 1000000;
	Graph *g = Graph_New("g", n, n);
	Graph_AcquireWriteLock(g);

	for(int i = 0; i < samples; i++) {
		// Create N nodes, don't use labels.
		simple_tic(tic);
		for(int j = 0; j < n; j++) {
			node = GE_NEW_NODE();
			Graph_CreateNode(g, &node, NULL, 0);
		}
		timings[i] = simple_toc(tic);
		printf("%zu Nodes created, time: %.6f sec\n", n, timings[i]);
		if(timings[i] > threshold) outliers++;
	}

	if(outliers > samples * 0.1) {
		printf("Node creation took too long\n");
		for(int i = 0; i < samples; i++) {
			printf("%zu Nodes created, time: %.6f sec\n", n, timings[i]);
		}
		// assert(false);
	}

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

void benchmark_edge_creation_with_relationships() {
	printf("benchmark_edge_creation_with_relationships\n");
	double tic[2];
	int samples = 64;
	double timings[samples];
	int outliers = 0;
	float threshold = 0.002;
	int edge_count = 1000000 * 1.10;
	int node_count = 1000000;
	int relation_count = 3;
	EdgeDesc connections[edge_count];
	Node node;
	Edge edge;
	Graph *g = Graph_New("g", GRAPH_DEFAULT_NODE_CAP, GRAPH_DEFAULT_EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// Introduce relations types.
	for(int i = 0; i < relation_count; i++) Graph_AddRelationType(g);
	for(int i = 0; i < node_count; i++) {
		node = GE_NEW_NODE();
		Graph_CreateNode(g, &node, NULL, 0);
	}
	for(int i = 0; i < samples; i++) {
		// Describe connections;
		// Node I is connected to Node I+1,
		// Connection type is relationships[I%4].
		for(int j = 0; j < edge_count; j++) {
			connections[j].srcId = rand() % node_count;     // Source node id.
			connections[j].destId = rand() % node_count;    // Destination node id.
			connections[j].relationId = i % relation_count; // Relation.
		}

		simple_tic(tic);
		for(int j = 0; j < edge_count; j++) {
			Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, connections[j].relationId,
							   &edge);
		}
		timings[i] = simple_toc(tic);
		printf("%d Formed connections, time: %.6f sec\n", edge_count, timings[i]);
		if(timings[i] > threshold)
			outliers++;
	}

	if(outliers > samples * 0.1) {
		printf("Node creation took too long\n");
		for(int i = 0; i < samples; i++) {
			printf("%d Formed connections, time: %.6f sec\n", edge_count, timings[i]);
		}
		// assert(false);
	}

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

void benchmark_graph() {
	benchmark_node_creation_no_labels();
	benchmark_node_creation_with_labels();
	benchmark_edge_creation_with_relationships();
	printf("%sgraph benchmark - PASS!%s\n", KGRN, KNRM);
}

void setup() {
	// Use the malloc family for allocations
	Alloc_Reset () ;
	Logging_Reset () ;

	// Initialize GraphBLAS.
	GrB_init(GrB_NONBLOCKING);

	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // all matrices in CSR format
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER); // matrices are never hypersparse
	srand(time(NULL));
}

void tearDown() {
	GrB_finalize();
}

// Validate the creation of a graph,
// Make sure graph's defaults are applied.
void test_newGraph() {
	GrB_Index ncols, nrows, nvals;
	Graph *g = Graph_New("g", GRAPH_DEFAULT_NODE_CAP, GRAPH_DEFAULT_EDGE_CAP);
	Graph_AcquireWriteLock(g);
	Delta_Matrix adj_matrix = g->adjacency_matrix;
	TEST_ASSERT(Delta_Matrix_ncols(&ncols, adj_matrix) == GrB_SUCCESS);
	TEST_ASSERT(Delta_Matrix_nrows(&nrows, adj_matrix) == GrB_SUCCESS);
	TEST_ASSERT(Delta_Matrix_nvals(&nvals, adj_matrix) == GrB_SUCCESS);

	TEST_ASSERT(g->nodes != NULL);
	TEST_ASSERT(g->relations != NULL);
	TEST_ASSERT(g->labels != NULL);
	TEST_ASSERT(g->adjacency_matrix != NULL);
	TEST_ASSERT(Graph_NodeCount(g) == 0);

	GrB_Index n = Graph_RequiredMatrixDim(g);
	TEST_ASSERT(nrows == n);
	TEST_ASSERT(ncols == n);
	TEST_ASSERT(nvals == 0);

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

// Tests node and edge creation.
void test_graphConstruction() {
	size_t node_count = GRAPH_DEFAULT_NODE_CAP / 2;
	Graph *g = Graph_New("g", node_count, node_count);
	Graph_AcquireWriteLock(g);
	_test_node_creation(g, node_count);
	Graph_ReleaseLock(g);
	Graph_Free(g);
}

void test_removeNodes() {
	// Construct graph.
	Delta_Matrix M;
	GrB_Index nnz;
	Graph *g = Graph_New("g", 32, 32);
	Graph_AcquireWriteLock(g);

	// Create 3 nodes.
	Node node;
	Edge edge;

	for(int i = 0; i < 3; i++) {
		node = GE_NEW_NODE();
		Graph_CreateNode(g, &node, NULL, 0);
	}
	int r = Graph_AddRelationType(g);

	// Connections:
	// 0 connected to 1
	// 1 connected to 0
	// 1 connected to 2

	// connect 0 to 1
	Graph_CreateEdge(g, 0, 1, r, &edge);

	// connect 1 to 0
	Graph_CreateEdge(g, 1, 0, r, &edge);

	// connect 1 to 2
	Graph_CreateEdge(g, 1, 2, r, &edge);

	// validate graph creation
	TEST_ASSERT(Graph_NodeCount(g) == 3);
	M = Graph_GetRelationMatrix(g, r, false);
	Delta_Matrix_nvals(&nnz, M);
	TEST_ASSERT(nnz == 3);

	M = Graph_GetAdjacencyMatrix(g, false);
	Delta_Matrix_nvals(&nnz, M);
	TEST_ASSERT(nnz == 3);

	Edge *edges = (Edge *)array_new(Edge, 3);

	//--------------------------------------------------------------------------
	// delete node 0
	//--------------------------------------------------------------------------

	// first node should have 2 edges
	Graph_GetNode(g, 0, &node);
	Graph_GetNodeEdges(g, &node, GRAPH_EDGE_DIR_BOTH, GRAPH_NO_RELATION, &edges);
	uint edge_count = array_len(edges);
	TEST_ASSERT(edge_count == 2);

	Graph_DeleteEdges(g, edges, edge_count, false);
	Graph_GetNode(g, 0, &node);
	Graph_DeleteNodes(g, &node, 1);
	
	Graph_ReleaseLock(g);

	array_free(edges);

	TEST_ASSERT(Graph_NodeCount(g) == 2);
	TEST_ASSERT(Graph_EdgeCount(g) == 1);

	Graph_Free(g);
}

void test_getNode() {
	/* Create a graph with nodeCount nodes,
	 * Make sure node retrival works as expected:
	 * try to get nodes (0 - nodeCount) */

	Node n;
	size_t nodeCount = 16;
	Graph *g = Graph_New("g", nodeCount, nodeCount);

	Graph_AcquireWriteLock(g);
	{
		for(int i = 0 ; i < nodeCount; i++) {
			n = GE_NEW_NODE();
			Graph_CreateNode(g, &n, NULL, 0);
		}
	}
	Graph_ReleaseLock(g);

	// Get nodes 0 - nodeCount.
	NodeID i = 0;
	for(; i < nodeCount; i++) {
		Graph_GetNode(g, i, &n);
		TEST_ASSERT(n.attributes != NULL);
	}

	Graph_Free(g);
}

void test_getEdge() {
	// Create a graph with both nodes and edges.
	// Make sure edge retrival works as expected:
	// 1. try to get edges by ID
	// 2. try to get edges connecting source to destination node
	// 3. try to get none existing edges

	Node n;
	Edge e;
	int relationCount = 4;
	size_t nodeCount = 5;
	size_t edgeCount = 5;
	int relations[relationCount];

	Graph *g = Graph_New("g", nodeCount, nodeCount);
	Graph_AcquireWriteLock(g);
	for(int i = 0; i < nodeCount; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}
	for(int i = 0; i < relationCount; i++) relations[i] = Graph_AddRelationType(g);

	// connect nodes:
	// 1. nodes (0-1) will be connected by relation 0
	// 2. nodes (0-1) will be connected by relation 1
	// 3. nodes (1-2) will be connected by relation 1
	// 4. nodes (2-3) will be connected by relation 2
	// 5. nodes (3-4) will be connected by relation 3

	Graph_CreateEdge(g, 0, 1, relations[0], &e);
	Graph_CreateEdge(g, 0, 1, relations[1], &e);
	Graph_CreateEdge(g, 1, 2, relations[1], &e);
	Graph_CreateEdge(g, 2, 3, relations[2], &e);
	Graph_CreateEdge(g, 3, 4, relations[3], &e);

	// validations
	// try to get edges by ID
	for(EdgeID i = 0; i < edgeCount; i++) {
		Graph_GetEdge(g, i, &e);
		TEST_ASSERT(e.attributes != NULL);
		TEST_ASSERT(e.id == i);
	}

	// try to get edges connecting source to destination node.
	NodeID src;
	NodeID dest;
	int relation;
	Edge *edges = (Edge *)array_new(Edge, 4);

	// get all edges connecting node 0 to node 1,
	// expecting 2 edges
	src = 0;
	dest = 1;
	relation = GRAPH_NO_RELATION;   // relation agnostic
	Graph_GetEdgesConnectingNodes(g, src, dest, relation, &edges);
	TEST_ASSERT(array_len(edges) == 2);
	for(int i = 0; i < 2; i++) {
		e = edges[i];
		relation = relations[i];
		TEST_ASSERT(e.attributes != NULL);
		TEST_ASSERT(Edge_GetSrcNodeID(&e) == src);
		TEST_ASSERT(Edge_GetDestNodeID(&e) == dest);
		TEST_ASSERT(Edge_GetRelationID(&e) == relation);
	}
	array_clear(edges);

	// try to get none existing edges:

	// node 0 is not connected to 2
	src = 0;
	dest = 2;
	relation = GRAPH_NO_RELATION;
	Graph_GetEdgesConnectingNodes(g, src, dest, relation, &edges);
	TEST_ASSERT(array_len(edges) == 0);
	array_clear(edges);

	// node 0 is not connected to 1 via relation 2
	src = 0;
	dest = 1;
	relation = relations[2];
	Graph_GetEdgesConnectingNodes(g, src, dest, relation, &edges);
	TEST_ASSERT(array_len(edges) == 0);
	array_clear(edges);

	// node 1 is not connected to 0 via relation 0
	src = 1;
	dest = 0;
	relation = relations[0];
	Graph_GetEdgesConnectingNodes(g, src, dest, relation, &edges);
	TEST_ASSERT(array_len(edges) == 0);
	array_clear(edges);

	// no node connects to itself
	for(NodeID i = 0; i < nodeCount; i++) {
		for(int j = 0; j < relationCount; j++) {
			src = i;
			relation = relations[j];
			Graph_GetEdgesConnectingNodes(g, src, src, relation, &edges);
			TEST_ASSERT(array_len(edges) == 0);
			array_clear(edges); // reset edge count
		}
		relation = GRAPH_NO_RELATION;
		Graph_GetEdgesConnectingNodes(g, src, src, relation, &edges);
		TEST_ASSERT(array_len(edges) == 0);
		array_clear(edges); // reset edge count
	}

	array_free(edges);
	Graph_ReleaseLock(g);
	Graph_Free(g);
}

TEST_LIST = {
	{"newGraph", test_newGraph},
	{"graphConstruction", test_graphConstruction},
	{"removeNodes", test_removeNodes},
	{"getNode", test_getNode},
	{"getEdge", test_getEdge},
	{NULL, NULL}
};
