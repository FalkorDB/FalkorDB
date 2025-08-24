/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "proc_bfs.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"
#include "../configuration/config.h"

// the BFS procedure performs a single source BFS scan
// it's inputs are:
// 1. source node to traverse from
// 2. depth, how deep should the procedure traverse (-1 no limit)
// 3. relationship type to traverse, (NULL for edge type agnostic)
//
// output:
// 1. nodes - an array of reachable nodes
// 2. edges - an array of edges traversed
//
// MATCH (a:User {id: 1}) CALL algo.bfs(a, -1, 'MANAGES') YIELD nodes, edges

typedef struct {
	Graph *g;              // graph scanned
	GrB_Index n;           // total number of results
	bool depleted;         // true if BFS has already been performed for this node
	int reltype_id;        // id of relationship matrix to traverse
	SIValue output[2];     // array with a maximum of 2 entries: [nodes, edges]
	SIValue *yield_nodes;  // yield reachable nodes
	SIValue *yield_edges;  // yield edges traversed
	GrB_Vector nodes;      // vector of reachable nodes
	GrB_Vector parents;    // vector associating each node in the BFS tree with its parent
} BFSCtx;

static void _process_yield
(
	BFSCtx *ctx,
	const char **yield
) {
	ctx->yield_nodes = NULL;
	ctx->yield_edges = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("nodes", yield[i]) == 0) {
			ctx->yield_nodes = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("edges", yield[i]) == 0) {
			ctx->yield_edges = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

static ProcedureResult Proc_BFS_Invoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	// validate inputs
	ASSERT(ctx  != NULL);
	ASSERT(args != NULL);

	if(array_len((SIValue *)args) != 3) {
		return PROCEDURE_ERR;
	}

	if(SI_TYPE(args[0]) != T_NODE                 ||  // source node
	   SI_TYPE(args[1]) != T_INT64                ||  // max level to iterate to, unlimited if 0
	   !(SI_TYPE(args[2]) & (T_NULL | T_STRING)))     // relationship type to traverse if not NULL
		return PROCEDURE_ERR;

	BFSCtx *bfs_ctx = ctx->privateData;
	ASSERT(bfs_ctx != NULL);

	_process_yield(bfs_ctx, yield);

	//--------------------------------------------------------------------------
	// process inputs
	//--------------------------------------------------------------------------

	Node *source_node = args[0].ptrval;
	int64_t max_level = args[1].longval;
	const char *reltype = SIValue_IsNull(args[2]) ? NULL : args[2].stringval;

	// get edge matrix and transpose matrix, if available
	GrB_Matrix    R    = NULL;
	Graph        *g    = QueryCtx_GetGraph();
	GraphContext *gc   = QueryCtx_GetGraphCtx();

	Delta_Matrix D;
	RelationID *rel_id = NULL;
	if(reltype != NULL) {
		Schema *s = GraphContext_GetSchema(gc, reltype, SCHEMA_EDGE);
		// failed to find schema, first step will return NULL
		if(!s) {
			return PROCEDURE_OK;
		}

		bfs_ctx->reltype_id = Schema_GetID(s);
		rel_id = &bfs_ctx->reltype_id;
	}

	GrB_Info info = get_sub_adjecency_matrix(&R, NULL, g, NULL, 0, rel_id,
			(rel_id != NULL) ? 1 : 0, false);
	ASSERT(info == GrB_SUCCESS);

	// if we're not collecting edges, pass a NULL parent pointer
	// so that the algorithm will not perform unnecessary work
	GrB_Vector V    = NULL;  // vector of results
	GrB_Vector PI   = NULL;  // vector backtracking results to their parents
	GrB_Vector temp = NULL;
	GrB_Vector *pPI = (bfs_ctx->yield_edges) ? &PI : NULL;

	char msg[LAGRAPH_MSG_LEN];
	LAGraph_Graph G;

	info = LAGraph_New(&G, &R, LAGraph_ADJACENCY_DIRECTED, msg);
	ASSERT(info == GrB_SUCCESS);

	// find the source node in the compressed graph
	GrB_Index src_id = ENTITY_GET_ID(source_node);

	// FUTURE: handling labels
	// GrB_Index src_loc = 0;

	// struct GB_Iterator_opaque _it;
	// GxB_Iterator it = &_it;
	
	// GrB_OK(GxB_Vector_Iterator_attach(it, rows, NULL));

	// // TODO: probably a better way than linear search but this way we don't care
	// // about how the vector is stored
	// for(; GxB_Vector_Iterator_getIndex(it) < src_id; GxB_Vector_Iterator_next(it)) {
	// 	++src_loc;
	// }

	info = LAGr_BreadthFirstSearch_Extended(&V, pPI, G, src_id, max_level, -1,
			false, msg);
	ASSERT(info == GrB_SUCCESS);

	info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	// remove all values with a level greater than 0
	// values of 0 are not connected to the source
	info = GrB_Vector_select_UINT64(
		V, NULL, NULL, GrB_VALUEGT_UINT64, V, (uint64_t) 0, NULL);
	ASSERT(info == GrB_SUCCESS);

	// get number of entries
	GrB_Index nvals;
	GrB_Vector_nvals(&nvals, V);

	GrB_Vector_set_INT32(V, GxB_SPARSE, GxB_SPARSITY_CONTROL);
	GrB_Vector_set_INT32(PI, GxB_SPARSE, GxB_SPARSITY_CONTROL);

	bfs_ctx->n       = nvals;
	bfs_ctx->nodes   = V;
	bfs_ctx->parents = PI;

	//--------------------------------------------------------------------------
	// FUTURE: map the outputs mack to their respective node ids
	//--------------------------------------------------------------------------
	// GrB_Descriptor desc = NULL;
	// GrB_Type       ty   = NULL;

	// GrB_Descriptor_new(&desc);
	// GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	// GxB_Vector_type(&ty, bfs_ctx->nodes);

	// GrB_Vector_dup(&temp, rows);

	// GxB_Vector_Option_set(temp, GxB_SPARSITY_CONTROL, GxB_SPARSE);
	// GxB_Vector_assign_Vector(temp, NULL, NULL, bfs_ctx->nodes, rows, desc);
	// GrB_Vector_free(&bfs_ctx->nodes);
	// bfs_ctx->nodes = temp; temp = NULL;

	// if(PI) {
	// 	GrB_Vector_dup(&temp, rows);
	// 	GxB_Vector_Option_set(temp, GxB_SPARSITY_CONTROL, GxB_SPARSE);
	// 	GxB_Vector_assign_Vector(temp, NULL, NULL, bfs_ctx->parents, rows, desc);
	// 	GrB_Vector_free(&bfs_ctx->parents);
	// 	bfs_ctx->parents = temp; temp = NULL;
	// }

	// GrB_free(&desc);
	// GrB_free(&rows);

	return PROCEDURE_OK;
}

static SIValue *Proc_BFS_Step
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData);

	BFSCtx *bfs_ctx = (BFSCtx *)ctx->privateData;

	// return NULL if the procedure for this source has already been emitted
	// or there are no connected nodes
	if(bfs_ctx->depleted || bfs_ctx->n == 0) {
		return NULL;
	}

	bool yield_nodes = (bfs_ctx->yield_nodes != NULL);
	bool yield_edges = (bfs_ctx->yield_edges != NULL);

	// build arrays for the outputs the user has requested
	uint n = bfs_ctx->n;
	SIValue nodes, edges;

	if(yield_nodes) {
		nodes = SI_Array(n);
	}

	if(yield_edges) {
		edges = SI_Array(n);
	}

	Edge *edge = array_new(Edge, 1);

	// setup result iterator
	NodeID       id;
	GrB_Info     info;
	GxB_Iterator iter;

	info = GxB_Iterator_new(&iter);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_attach(iter, bfs_ctx->nodes, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_seek(iter, 0);

	while(info == GrB_SUCCESS) {
		id = GxB_Vector_Iterator_getIndex(iter);

		// get the reached node
		if(yield_nodes) {
			// append each reachable node to the nodes output array
			Node n = GE_NEW_NODE();
			bool node_found = Graph_GetNode(bfs_ctx->g, id, &n);
			ASSERT(node_found == true);

			SIArray_Append(&nodes, SI_Node(&n));
		}

		if(yield_edges) {
			array_clear(edge);
			GrB_Index parent_id;
			// find the parent of the reached node
			info = GrB_Vector_extractElement(&parent_id, bfs_ctx->parents, id);
			ASSERT(info == GrB_SUCCESS);

			// retrieve edges connecting the parent node to the current node
			// TODO: we only require a single edge
			// `Graph_GetEdgesConnectingNodes` can return multiple edges
			Graph_GetEdgesConnectingNodes(bfs_ctx->g, parent_id, id,
					bfs_ctx->reltype_id, &edge);

			// append one edge to the edges output array
			SIArray_Append(&edges, SI_Edge(edge));
		}

		info = GxB_Vector_Iterator_next(iter);
	}

	// populate output
	if(yield_nodes) {
		*bfs_ctx->yield_nodes = nodes;
	}

	if(yield_edges) {
		*bfs_ctx->yield_edges = edges;
	}

	// clean up
	array_free(edge);
	GxB_Iterator_free(&iter);
	bfs_ctx->depleted = true;

	return bfs_ctx->output;
}

static ProcedureResult Proc_BFS_Free
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx != NULL);

	// free private data
	BFSCtx *pdata = ctx->privateData;

	if(pdata->nodes   != NULL) GrB_Vector_free(&pdata->nodes);
	if(pdata->parents != NULL) GrB_Vector_free(&pdata->parents);

	rm_free(ctx->privateData);

	return PROCEDURE_OK;
}

static BFSCtx *_Build_Private_Data() {
	// set up the BFS context
	BFSCtx *pdata = rm_calloc(1, sizeof(BFSCtx));

	pdata->g          = QueryCtx_GetGraph();
	pdata->reltype_id = GRAPH_NO_RELATION;

	return pdata;
}

ProcedureCtx *Proc_BFS_Ctx() {
	// construct procedure private data
	void *privdata = _Build_Private_Data();

	// declare possible outputs
	ProcedureOutput *outputs = array_new(ProcedureOutput, 2);
	ProcedureOutput out_nodes = {.name = "nodes", .type = T_ARRAY};
	ProcedureOutput out_edges = {.name = "edges", .type = T_ARRAY};
	array_append(outputs, out_nodes);
	array_append(outputs, out_edges);

	ProcedureCtx *ctx = ProcCtxNew("algo.BFS",
								   3,
								   outputs,
								   Proc_BFS_Step,
								   Proc_BFS_Invoke,
								   Proc_BFS_Free,
								   privdata,
								   true);
	return ctx;
}

