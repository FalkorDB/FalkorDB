
/* * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/graph/graph.h"
#include "src/procedures/utility/internal.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include <time.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define GRAPH_DEFAULT_NODE_CAP 16384
#define GRAPH_DEFAULT_EDGE_CAP 16384

// Encapsulate the essence of an edge.
typedef struct {
	NodeID srcId;       // Source node ID.
	NodeID destId;      // Destination node ID.
	int64_t relationId; // Relation type ID.
} EdgeDesc;

void setup() {
	// use the malloc family for allocations
	Alloc_Reset();

	// initialize GraphBLAS
	GrB_init(GrB_NONBLOCKING);

	// all matrices in CSR format
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);

	// set delta matrix flush threshold
	Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES, "10000", NULL);
}

void tearDown() {
	GrB_finalize();
}

void CHECK_sub_adjecency_matrix
(
	const GrB_Matrix A,      // matrix
	const GrB_Vector rows,   // filtered rows
	const GrB_Matrix *rels,  // rel adjecency matricies
	unsigned short n_rels,   // number of relationships
	bool symmetric           // build a symmetric matrix
) {
    TEST_ASSERT(A != NULL);
    TEST_ASSERT(rows != NULL);
    TEST_ASSERT(rels != NULL);
    TEST_ASSERT(n_rels > 0);
    
    GrB_Matrix     B     = NULL;
    GrB_Matrix     C     = NULL;
    GrB_Descriptor desc  = NULL;
    GrB_BinaryOp   accum = GrB_ONEB_BOOL; 
    GrB_BinaryOp   eq    = GrB_EQ_BOOL; 
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Index nvals;
    GrB_Index nvals_a;
    GrB_Index nvals_b;
    GrB_Type  t;

    TEST_ASSERT(GrB_Matrix_nrows(&nrows, A) == GrB_SUCCESS);
    TEST_ASSERT(GrB_Matrix_ncols(&ncols, A) == GrB_SUCCESS);
    TEST_ASSERT(GrB_Vector_nvals(&nvals, rows) == GrB_SUCCESS);
    TEST_ASSERT(GxB_Matrix_type(&t, A) == GrB_SUCCESS);
    
    TEST_ASSERT(nrows == nvals);
    
    TEST_ASSERT(GrB_Matrix_new(&B, t, nrows, ncols) == GrB_SUCCESS);
    TEST_ASSERT(GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_new(&desc) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST) == GrB_SUCCESS);
    TEST_ASSERT(GxB_Matrix_extract_Vector(B, NULL, NULL, rels[0], rows, rows, desc) == GrB_SUCCESS);
    
    // add together all relation matrices
    for(int i = 1; i < n_rels; i++){
        TEST_ASSERT(GxB_Matrix_extract_Vector(B, NULL, accum, rels[i], 
            rows, rows, desc) == GrB_SUCCESS);
    }

    // if symmetric, add transpose
    if(symmetric) {
        TEST_ASSERT(GrB_transpose(B, NULL, accum, B, NULL) == GrB_SUCCESS);
    }

    TEST_ASSERT(GrB_Matrix_nvals(&nvals_a, A) == GrB_SUCCESS);
    TEST_ASSERT(GrB_Matrix_nvals(&nvals_b, B) == GrB_SUCCESS);
    TEST_ASSERT(nvals_a == nvals_b);
    TEST_ASSERT(GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, eq, A, B, desc) 
        == GrB_SUCCESS);
    TEST_ASSERT(GrB_Matrix_nvals(&nvals_a, C) == GrB_SUCCESS);
    TEST_ASSERT(nvals_a == nvals_b);
    bool ok = true;
    TEST_ASSERT(GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, C, NULL) 
        == GrB_SUCCESS);
    TEST_ASSERT(ok);
}

void test_sub_adj_matrix(){
	int edge_count = 10000 * 2.3;
	int node_count = 10000;
	int relation_count = 3;
	int label_count = 3;
	EdgeDesc connections[edge_count];
    GrB_Matrix mtx_list[3] = {NULL, NULL, NULL};
	Node node;
	Edge edge;
	Graph *g = Graph_New(GRAPH_DEFAULT_NODE_CAP, GRAPH_DEFAULT_EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// Introduce relations types.
	for(int i = 0; i < relation_count; i++) {
        Graph_AddRelationType(g);
        GrB_Matrix_new(&mtx_list[i], GrB_BOOL, node_count, node_count);
    }

	for(int i = 0; i < label_count; i++) Graph_AddLabel(g);

	for(int i = 0; i < node_count; i++) {
		node = GE_NEW_NODE();
		Graph_CreateNode(g, &node, (LabelID) (rand() % relation_count), 0);
	}

    // Describe connections;
    for(int j = 0; j < edge_count; j++) {
        connections[j].srcId = rand() % node_count;     // Source node id.
        connections[j].destId = rand() % node_count;    // Destination node id.
        connections[j].relationId = rand() % relation_count; // Relation.
    }

    for(int j = 0; j < edge_count; j++) {
        Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, 
            connections[j].relationId, &edge);
        
        GrB_Matrix_setElement_BOOL(mtx_list[connections[j].relationId], 
            true, connections[j].srcId, connections[j].destId);
    }

    GrB_Vector rows = NULL;
    GrB_Matrix A = NULL;
    GrB_Matrix Rs[] = {mtx_list[0], mtx_list[1]};
    RelationID rels[] = {0, 1};
    LabelID lbls[] = {0, 1};
   
    get_sub_adjecency_matrix(&A, &rows, g, lbls, 2, rels, 2, true);
    
    CHECK_sub_adjecency_matrix (A, rows, Rs, 2, true);

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

TEST_LIST = {
    {"test_sub_adj_matrix", test_sub_adj_matrix},
    {NULL, NULL} 
};
