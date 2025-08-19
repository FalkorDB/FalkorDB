/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraph.h"
#include "proc_pagerank.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// CALL algo.pageRank(NULL, NULL)      YIELD node, score
// CALL algo.pageRank('Page', NULL)    YIELD node, score
// CALL algo.pageRank(NULL, 'LINKS')   YIELD node, score
// CALL algo.pageRank('Page', 'LINKS') YIELD node, score

typedef struct {
	Graph *g;               // graph
	GrB_Vector nodes;       // nodes participating in computation
	GrB_Vector centrality;  // nodes centrality
	GrB_Info info;          // iterator state
	GxB_Iterator it;        // nodes iterator
	GxB_Iterator it_cen;    // centrality iterator
	Node node;              // node
	SIValue output[2];      // array with up to 2 entries [node, score]
	SIValue *yield_node;    // yield node
	SIValue *yield_score;   // yield score
} PagerankContext;

static void _process_yield
(
	PagerankContext *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("score", yield[i]) == 0) {
			ctx->yield_score = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

ProcedureResult Proc_PagerankInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	// expecting 2 arguments
	if(array_len((SIValue *)args) != 2) {
		return PROCEDURE_ERR;
	}

	// arg0 and arg1 can be either String or NULL
	SIType arg0_t = SI_TYPE(args[0]);
	SIType arg1_t = SI_TYPE(args[1]);

	if(!(arg0_t & (T_STRING | T_NULL))) {
		return PROCEDURE_ERR;
	}

	if(!(arg1_t & (T_STRING | T_NULL))) {
		return PROCEDURE_ERR;
	}

	// read arguments
	const char   *label    = NULL;
	const char   *relation = NULL;
	Graph        *g        = QueryCtx_GetGraph();
	GraphContext *gc       = QueryCtx_GetGraphCtx();

	LabelID    lbl_id = GRAPH_UNKNOWN_LABEL;
	RelationID rel_id = GRAPH_UNKNOWN_RELATION;

	if(arg0_t == T_STRING) label    = args[0].stringval;
	if(arg1_t == T_STRING) relation = args[1].stringval;

	// get label matrix
	if(label != NULL) {
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		// unknown label, quickly return
		if(!s) {
			return PROCEDURE_ERR;
			return PROCEDURE_OK;
		}

		lbl_id = Schema_GetID(s);
	}

	// get relation matrix
	if(relation != NULL) {
		Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
		// unknown relation, quickly return
		if(!s) {
			return PROCEDURE_ERR;
			return PROCEDURE_OK;
		}

		rel_id = Schema_GetID(s);
	}

	// setup context
	PagerankContext *pdata = rm_calloc(1, sizeof(PagerankContext));
	pdata->g = g;

	_process_yield(pdata, yield);

	ctx->privateData = pdata;


	//--------------------------------------------------------------------------
	// build adjacency matrix
	//--------------------------------------------------------------------------

	unsigned short n_lbls = 0;
	unsigned short n_rels = 0;
	LabelID       *p_lbls = NULL;
	RelationID    *p_rels = NULL;

	if(lbl_id != GRAPH_UNKNOWN_LABEL) {
		p_lbls = &lbl_id;
		n_lbls = 1;
	}

	if(rel_id != GRAPH_UNKNOWN_RELATION) {
		p_rels = &rel_id;
		n_rels = 1;
	}

	GrB_Matrix A;
	GrB_Info info;

	// info = Build_Matrix(&A, &pdata->nodes, g, p_lbls, n_lbls, 
	// 	p_rels, n_rels, false, true);

	info = get_sub_adjecency_matrix(&A, &pdata->nodes, g, p_lbls, n_lbls, 
		p_rels, n_rels, false);

	ASSERT(info         == GrB_SUCCESS);
	ASSERT(A            != NULL);
	ASSERT(pdata->nodes != NULL);

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over participating nodes
	info = GxB_Vector_Iterator_attach(pdata->it, pdata->nodes, NULL);
	ASSERT(info == GrB_SUCCESS);

    pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	// early return if A is empty
	GrB_Index nvals;
	info = GrB_Vector_nvals(&nvals, pdata->nodes);
	ASSERT(info == GrB_SUCCESS);

	if(nvals == 0) {
		// empty matrix
		return PROCEDURE_OK;
	}

	//--------------------------------------------------------------------------
	// run pagerank
	//--------------------------------------------------------------------------

	int         iters   = 0;
	const float tol     = 1e-4;
	const float damping = 0.85;
	const int   itermax = 100; // max iterations

	LAGraph_Graph G;
	char msg[LAGRAPH_MSG_LEN];

	info = LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
	ASSERT(info == GrB_SUCCESS);

	// compute AT, required by algorithm
	info = LAGraph_Cached_AT(G, msg);
	ASSERT(info == GrB_SUCCESS);

	info = LAGraph_Cached_OutDegree(G, msg);
	ASSERT(info == GrB_SUCCESS);

	info = LAGr_PageRank(&pdata->centrality, &iters, G, damping, tol, itermax,
			msg);

	ASSERT(info == GrB_SUCCESS);

	info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	GrB_OK (GxB_Iterator_new(&pdata->it_cen));
	GrB_OK (GxB_Iterator_new(&pdata->it));

	// iterate over participating nodes
	GrB_OK (GxB_Vector_Iterator_attach(pdata->it_cen, pdata->centrality, NULL));
	GrB_OK (GxB_Vector_Iterator_attach(pdata->it, pdata->nodes, NULL));

    GxB_Vector_Iterator_seek(pdata->it_cen, 0);
    pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	return PROCEDURE_OK;
}

SIValue *Proc_PagerankStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	PagerankContext *pdata = (PagerankContext *)ctx->privateData;

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	// retrieve node from graph
	GrB_Index node_id = GxB_Vector_Iterator_getIndex(pdata->it);
	float score = GxB_Iterator_get_FP32(pdata->it_cen);

	ASSERT (Graph_GetNode(pdata->g, node_id, &pdata->node));

	// prep for next call to Proc_BetweennessStep
	pdata->info = GxB_Vector_Iterator_next(pdata->it);
	GrB_Info info = GxB_Vector_Iterator_next(pdata->it_cen);

	ASSERT(info == pdata->info);

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if(pdata->yield_node) {
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	if(pdata->yield_score) {
		*pdata->yield_score = SI_DoubleVal(score);
	}

	return pdata->output;
}

ProcedureResult Proc_PagerankFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		PagerankContext *pdata = ctx->privateData;
		if(pdata->it         != NULL) GrB_free(&pdata->it);
		if(pdata->nodes      != NULL) GrB_free(&pdata->nodes);
		if(pdata->centrality != NULL) GrB_free(&pdata->centrality);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_PagerankCtx() {
	void *privateData = NULL;
	ProcedureOutput *outputs     = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node  = {.name = "node",  .type = T_NODE};
	ProcedureOutput output_score = {.name = "score", .type = T_DOUBLE};

	array_append(outputs, output_node);
	array_append(outputs, output_score);

	ProcedureCtx *ctx = ProcCtxNew("algo.pageRank",
								   2,
								   outputs,
								   Proc_PagerankStep,
								   Proc_PagerankInvoke,
								   Proc_PagerankFree,
								   privateData,
								   true);
	return ctx;
}

