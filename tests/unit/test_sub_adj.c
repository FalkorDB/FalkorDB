
/* * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// TODO: Test all rels or lbls and check rows for correctness
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
	double weight;      // Edge weight.
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

void CHECK_matrix_sub //check A <= B
(
	const GrB_Matrix A,      // matrix
	const GrB_Matrix B,      // matrix
	const GrB_BinaryOp eq
) {
	TEST_ASSERT(A != NULL);
	TEST_ASSERT(B != NULL);
	
	GrB_Matrix C = NULL;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index nvals_a;
	GrB_Index nvals_b;
	GrB_Index nvals_c;
	GrB_Type  t;

	TEST_ASSERT(GrB_Matrix_nrows(&nrows, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_ncols(&ncols, A) == GrB_SUCCESS);

	TEST_ASSERT(GrB_Matrix_nvals(&nvals_a, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_nvals(&nvals_b, B) == GrB_SUCCESS);

	TEST_ASSERT(GxB_Matrix_type(&t, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_new(&C, t, nrows, ncols) == GrB_SUCCESS);

	if(nvals_a > nvals_b)
	{
		GxB_fprint(A, GxB_SHORT, stdout);
		GxB_fprint(B, GxB_SHORT, stdout);
		GrB_Matrix_assign(C, B, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_SC);
		GxB_Matrix_fprint(C, "A - B",GxB_SHORT, stdout);
	}
	TEST_ASSERT(nvals_a <= nvals_b);

	TEST_ASSERT(GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, eq, A, B, NULL) 
		== GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_nvals(&nvals_c, C) == GrB_SUCCESS);
	if(nvals_a != nvals_c)
	{
		GxB_fprint(A, GxB_SHORT, stdout);
		GxB_fprint(B, GxB_SHORT, stdout);
		GrB_Matrix_assign(A, B, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RSC);
		GrB_wait(A, GrB_MATERIALIZE);
		GxB_Matrix_fprint(A, "A - B",GxB_SHORT, stdout);
	}
	TEST_ASSERT(nvals_a == nvals_c);

	bool ok = true;
	TEST_ASSERT(GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, C, 
		NULL) == GrB_SUCCESS);

	if(!ok) {
		GrB_Matrix_select_BOOL(C, NULL, NULL, GrB_VALUEEQ_BOOL, C, false, NULL);
		GxB_fprint(A, GxB_SHORT, stdout);
		GxB_fprint(B, GxB_SHORT, stdout);
		// GrB_Matrix_assign(B, B, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RSC);
		GrB_Matrix_assign(A, C, GrB_MINUS_FP64, B, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RS);
		GrB_wait(A, GrB_MATERIALIZE);
		GxB_Matrix_fprint(A, "A - B",GxB_SHORT, stdout);
	}

	GrB_free(&C);

	TEST_ASSERT(ok);
}

void CHECK_matrix_equal
(
	const GrB_Matrix A,      // matrix
	const GrB_Matrix B,      // matrix
	const GrB_BinaryOp eq
) {
	TEST_ASSERT(A != NULL);
	TEST_ASSERT(B != NULL);
	
	GrB_Matrix C = NULL;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index nvals_a;
	GrB_Index nvals_b;
	GrB_Index nvals_c;
	GrB_Type  t;

	TEST_ASSERT(GrB_Matrix_nrows(&nrows, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_ncols(&ncols, A) == GrB_SUCCESS);

	TEST_ASSERT(GrB_Matrix_nvals(&nvals_a, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_nvals(&nvals_b, B) == GrB_SUCCESS);

	TEST_ASSERT(GxB_Matrix_type(&t, A) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_new(&C, t, nrows, ncols) == GrB_SUCCESS);

	if(nvals_a != nvals_b)
	{
		GxB_fprint(A, GxB_SHORT, stdout);
		GxB_fprint(B, GxB_SHORT, stdout);
		GrB_Matrix_assign(C, A, NULL, B, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_SC);
		GrB_Matrix_assign(C, B, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_SC);
		GxB_Matrix_fprint(C, "A <symetric diff> B",GxB_SHORT, stdout);
	}
	TEST_ASSERT(nvals_a == nvals_b);

	TEST_ASSERT(GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, eq, A, B, NULL) 
		== GrB_SUCCESS);
	TEST_ASSERT(GrB_Matrix_nvals(&nvals_c, C) == GrB_SUCCESS);
	TEST_ASSERT(nvals_a == nvals_c);

	bool ok = true;
	TEST_ASSERT(GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, C, 
		NULL) == GrB_SUCCESS);

	if(!ok) {
		printf("Expected outputs differ on these values: \n\n\n");
		GrB_Matrix_assign(A, C, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RC);
		GrB_Matrix_assign(B, C, NULL, B, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RC);
		GxB_fprint(A, GxB_SHORT, stdout);
		GxB_fprint(B, GxB_SHORT, stdout);
	}

	GrB_free(&C);

	TEST_ASSERT(ok);
}

void CHECK_sub_adjecency_matrix
(
	const GrB_Matrix A,      // matrix
	const GrB_Vector rows,   // filtered rows
	const GrB_BinaryOp comb, // Operator used to combine matrices
	const GrB_Matrix *rels,  // rel adjecency matricies
	unsigned short n_rels,   // number of relationships
	bool symmetric           // build a symmetric matrix
) {
	TEST_ASSERT(A != NULL);
	TEST_ASSERT(rows != NULL);
	TEST_ASSERT(rels != NULL);
	TEST_ASSERT(n_rels > 0);
	
	GrB_Matrix     B     = NULL;
	GrB_Descriptor desc  = NULL;
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
	GrB_BinaryOp eq = (t == GrB_BOOL)? GrB_EQ_BOOL: GrB_EQ_FP64; 
	GrB_BinaryOp ge = (t == GrB_BOOL)? GrB_GE_BOOL: GrB_GE_FP64; 

	TEST_ASSERT(nrows == nvals);

	TEST_ASSERT(GrB_Matrix_new(&B, t, nrows, ncols) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_new(&desc) == GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST) 
		== GrB_SUCCESS);
	TEST_ASSERT(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST) 
		== GrB_SUCCESS);
	TEST_ASSERT(GxB_Matrix_extract_Vector(B, NULL, NULL, rels[0], rows, rows, 
		desc) == GrB_SUCCESS);
	CHECK_matrix_sub(B, A, ge);

	// add together all relation matrice
	for(int i = 1; i < n_rels; i++){
		TEST_ASSERT(GxB_Matrix_extract_Vector(B, NULL, comb, rels[i], 
			rows, rows, desc) == GrB_SUCCESS);
		CHECK_matrix_sub(B, A, ge);
	}

	// if symmetric, add transpose
	if(symmetric) {
		TEST_ASSERT(GrB_transpose(B, NULL, comb, B, NULL) == GrB_SUCCESS);
	}

	CHECK_matrix_equal(A, B, eq);
}

void CHECK_rows(
	const GrB_Vector rows,
	const Graph *g,
	const LabelID *lbls,
	int n_lbls
) {
	Delta_Matrix LD   = Graph_GetNodeLabelMatrix(g);
	GrB_Matrix L      = NULL;
	GrB_Vector res    = NULL;
	GrB_Vector x      = NULL;
	GrB_Index ncols;
	GrB_Index nrows;
	GrB_OK(Delta_Matrix_export(&L, LD));
	GrB_OK(GrB_Vector_size(&nrows, rows));
	GrB_OK(GrB_Matrix_ncols(&ncols, L));
	GrB_OK(GrB_Vector_new(&res, GrB_INT64, ncols));
	GrB_OK(GrB_Vector_new(&x, GrB_INT64, nrows));
	GrB_OK(GrB_Matrix_resize(L, nrows, ncols));
	GrB_OK(GrB_Vector_assign_BOOL(x, NULL, NULL, true, GrB_ALL, nrows, NULL));
	GrB_OK(GrB_vxm(res, NULL, NULL, GxB_PLUS_PAIR_INT64, x, L, NULL));
	GrB_OK(GrB_vxm(res, NULL, GrB_MINUS_INT64, GxB_PLUS_PAIR_INT64, rows, L, 
		NULL));

	for(int i = 0; i < n_lbls; i++) {
		int64_t k;
		GrB_Info info = GrB_Vector_extractElement_INT64(&k, res, lbls[i]);
		// element must be found
		TEST_ASSERT(info == GrB_SUCCESS);
		TEST_ASSERT(k == 0);
	}
}

void test_sub_adj_matrix(){
	int edge_count = 100 * 10;
	int node_count = 100;
	int relation_count = 3;
	int label_count = 3;
	EdgeDesc connections[edge_count];
	GrB_Matrix mtx_list[3] = {NULL, NULL, NULL};
	Node node;
	Edge edge;
	GrB_Info  info;
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
		LabelID labels[] = {rand() % label_count};
		Graph_CreateNode(g, &node, labels, 1);
	}

	// Describe connections;
	for(int j = 0; j < edge_count; j++) {
		connections[j].srcId = rand() % node_count;          // src node id
		connections[j].destId = rand() % node_count;         // dest node id
		connections[j].relationId = rand() % relation_count; // relation
	}

	for(int j = 0; j < edge_count * .9; j++) {
		Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, 
			connections[j].relationId, &edge);
		
		info = GrB_Matrix_setElement_BOOL(mtx_list[connections[j].relationId], 
			true, connections[j].srcId, connections[j].destId);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	Graph_ApplyAllPending(g, true);

	// create pending edges
	for(int j = edge_count * .9; j < edge_count; j++) {
		Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, 
			connections[j].relationId, &edge);
		
		info = GrB_Matrix_setElement_BOOL(mtx_list[connections[j].relationId], 
			true, connections[j].srcId, connections[j].destId);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	// Edge *e_del = rm_malloc(sizeof(Edge) * edge_count / 10);
	// // create pending deletions
	// for(int j = 0; j < edge_count / 10; j++) {
	//     EdgeID edge_id = rand() % edge_count; 

	//     // skip edges that have been deleted
	//     if(connections[j].relationId == GRAPH_NO_RELATION) continue;
		
	//     Graph_GetEdge(g, edge_id, &e_del[j]);
	//     Edge_SetSrcNodeID(&e_del[j], connections[j].srcId);
	//     Edge_SetDestNodeID(&e_del[j], connections[j].destId);
	//     Edge_SetRelationID(&e_del[j], connections[j].relationId);
	//     connections[j].relationId = GRAPH_NO_RELATION; // mark as deleted 

	//     info = GrB_Matrix_removeElement(mtx_list[connections[j].relationId], 
	//         connections[j].srcId, connections[j].destId);
	//     TEST_ASSERT(info == GrB_SUCCESS);
	// }

	// Graph_DeleteEdges(g, e_del, edge_count / 10);
	// Graph_print(g, GxB_SHORT);

	GrB_Vector rows = NULL;
	GrB_Matrix A = NULL;
	RelationID rels[] = {0, 1, 2};
	LabelID lbls[] = {0, 1, 2};

	for(int n_rels = 1; n_rels <= 3; n_rels ++){
		for(int n_lbls = 1; n_lbls <= 3; n_lbls ++){
			get_sub_adjecency_matrix(&A, &rows, g, lbls, n_lbls, rels, n_rels, 
				true);
			CHECK_rows(rows, g, lbls, n_lbls);
			CHECK_sub_adjecency_matrix (A, rows, GrB_ONEB_BOOL, mtx_list, 
				n_rels, true);
			
			GrB_Matrix_free(&A);
			GrB_Vector_free(&rows);
		}
	}

	for(int i = 0; i < relation_count; i++) {
		GrB_Matrix_free(&mtx_list[i]);
	}

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

void test_sub_weight_matrix(){
	int edge_count     = 100 * 90;
	int node_count     = 100;
	int relation_count = 3;
	int label_count    = 3;
	EdgeDesc connections[edge_count];
	GrB_Matrix mtx_list[3] = {NULL, NULL, NULL};
	Node node;
	Edge edge;
	GrB_Info  info;
	Graph *g = Graph_New(GRAPH_DEFAULT_NODE_CAP, GRAPH_DEFAULT_EDGE_CAP);
	Graph_AcquireWriteLock(g);
	
	// Introduce relations types.
	for(int i = 0; i < relation_count; i++) {
		Graph_AddRelationType(g);
		GrB_Matrix_new(&mtx_list[i], GrB_FP64, node_count, node_count);
	}

	for(int i = 0; i < label_count; i++) Graph_AddLabel(g);

	for(int i = 0; i < node_count; i++) {
		node = GE_NEW_NODE();
		LabelID labels[] = {rand() % label_count};
		Graph_CreateNode(g, &node, labels, 1);
	}

	// Describe connections;
	for(int j = 0; j < edge_count; j++) {
		connections[j].srcId = rand() % node_count;          // src node id
		connections[j].destId = rand() % node_count;         // dest node id
		connections[j].relationId = rand() % relation_count; // relation
		connections[j].weight = rand() / (double) RAND_MAX;  // relation
	}

	for(int j = 0; j < edge_count * .9; j++) {
		double currW = 0.0;
		double weight = connections[j].weight;
		
		Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, 
			connections[j].relationId, &edge);

		SIValue w = SI_DoubleVal(weight);
		GraphEntity_AddProperty((GraphEntity *) &edge, 0, w);
		
		info = GrB_Matrix_extractElement_FP64(&currW, 
			mtx_list[connections[j].relationId], connections[j].srcId, 
			connections[j].destId);
	
		weight = (info == GrB_SUCCESS && weight >= currW)? currW: weight;
		
		info = GrB_Matrix_setElement_FP64(mtx_list[connections[j].relationId], 
			weight, connections[j].srcId, connections[j].destId);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	Graph_ApplyAllPending(g, true);

	// create pending edges
	for(int j = edge_count * .9; j < edge_count; j++) {
		double currW = 0.0;
		double weight = connections[j].weight;
		
		Graph_CreateEdge(g, connections[j].srcId, connections[j].destId, 
			connections[j].relationId, &edge);
		SIValue w = SI_DoubleVal(weight);
		GraphEntity_AddProperty((GraphEntity *) &edge, 0, w);

		info = GrB_Matrix_extractElement_FP64(&currW, 
			mtx_list[connections[j].relationId], connections[j].srcId, 
			connections[j].destId);

		weight = (info == GrB_SUCCESS && weight >= currW)? currW: weight;
		
		info = GrB_Matrix_setElement_FP64(mtx_list[connections[j].relationId], 
			weight, connections[j].srcId, connections[j].destId);

		TEST_ASSERT(info == GrB_SUCCESS);
	}

	GrB_Vector  rows   = NULL;
	GrB_Matrix  A      = NULL;
	GrB_Matrix  A_w    = NULL;
	RelationID  rels[] = {0, 1, 2};
	LabelID     lbls[] = {0, 1, 2};
	AttributeID w_att  = 0;
   
	for(int n_rels = 1; n_rels <= 3; n_rels ++){
		for(int n_lbls = 1; n_lbls <= 3; n_lbls ++){
			get_sub_weight_matrix(&A, &A_w, &rows, g, lbls, n_lbls, rels, 
				n_rels, w_att, BWM_MIN, true);
			CHECK_rows(rows, g, lbls, n_lbls);
			CHECK_sub_adjecency_matrix (A_w, rows, GrB_MIN_FP64, mtx_list, 
				n_rels, true);

			GrB_Matrix_free(&A);
			GrB_Matrix_free(&A_w);
			GrB_Vector_free(&rows);
		}
	}

	for(int i = 0; i < relation_count; i++) {
		GrB_Matrix_free(&mtx_list[i]);
	}

	Graph_ReleaseLock(g);
	Graph_Free(g);
}

TEST_LIST = {
	{"test_sub_adj_matrix", test_sub_adj_matrix},
	{"test_sub_weight_matrix", test_sub_weight_matrix},
	{NULL, NULL} 
};
