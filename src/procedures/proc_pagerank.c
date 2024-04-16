/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "proc_pagerank.h"
#include "../RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../graph/graphcontext.h"
#include "../algorithms/pagerank.h"

// CALL algo.pageRank(NULL, NULL)      YIELD node, score
// CALL algo.pageRank('Page', NULL)    YIELD node, score
// CALL algo.pageRank(NULL, 'LINKS')   YIELD node, score
// CALL algo.pageRank('Page', 'LINKS') YIELD node, score

typedef struct {
	int n;                          // number of nodes to rank
	int i;                          // current node to return
	Graph *g;                       // graph
	Node node;                      // node
	GrB_Index *mapping;             // mapping between extracted matrix rows and node ids
	LAGraph_PageRank *ranking;      // nodes ranking
	SIValue *output;                // array with up to 2 entries [node, score]
	SIValue *yield_node;            // yield node
	SIValue *yield_score;           // yield score
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
	if(array_len((SIValue *)args) != 2) return PROCEDURE_ERR;

	// arg0 and arg1 can be either String or NULL
	SIType arg0_t = SI_TYPE(args[0]);
	SIType arg1_t = SI_TYPE(args[1]);
	if(!(arg0_t & (T_STRING | T_NULL))) return PROCEDURE_ERR;
	if(!(arg1_t & (T_STRING | T_NULL))) return PROCEDURE_ERR;

	// read arguments
	const char *label = NULL;    // node filter
	const char *relation = NULL; // edge filter
	if(arg0_t == T_STRING) label = args[0].stringval;
	if(arg1_t == T_STRING) relation = args[1].stringval;

	// pagerank config arguments
	int iters;               // iterations performed
	const double tol = 1e-4; // tolerance
	const int itermax = 100; // max iterations

	GrB_Info info;
	UNUSED(info);

	GrB_Index n = 0;               // node count
	GrB_Index nvals;               // number of entries in 'r'
	GrB_Index nrows;               // relation matrix row count
	Schema *s = NULL;
	GrB_Matrix l = NULL;           // label matrix
	GrB_Matrix r = NULL;           // relation matrix
	GrB_Index *mapping = NULL;     // mapping, array for returning row indices of tuples
	Graph *g = QueryCtx_GetGraph();
	LAGraph_PageRank *ranking = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// setup context
	PagerankContext *pdata = rm_malloc(sizeof(PagerankContext));
	pdata->n = n;
	pdata->i = 0;
	pdata->g = g;
	pdata->node = GE_NEW_NODE();
	pdata->mapping = mapping;
	pdata->ranking = ranking;
	pdata->output = array_new(SIValue, 2);
	_process_yield(pdata, yield);

	ctx->privateData = pdata;

	// get label matrix
	if(label) {
		s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		// unknown label, quickly return
		if(!s) return PROCEDURE_OK;
		Delta_Matrix_export(&l, Graph_GetLabelMatrix(g, s->id));
	}

	// get relation matrix
	if(relation) {
		s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
		// unknown relation, quickly return
		if(!s) return PROCEDURE_OK;
		Delta_Matrix_export(&r, Graph_GetRelationMatrix(g, s->id, false));

		// convert the values to true
		info = GrB_Matrix_apply(r, NULL, NULL, GxB_ONE_BOOL, r, GrB_DESC_R);
		ASSERT(info == GrB_SUCCESS);
	} else {
		// relation isn't specified, 'r' is the adjacency matrix
		Delta_Matrix_export(&r, Graph_GetAdjacencyMatrix(g, false));
	}
	// if label is specified:
	// filter 'r' to contain only rows and columns associated with
	// nodes of type 'l'
	if(label != NULL) {
		//----------------------------------------------------------------------
		// create a NxN matrix, one row for each labeled entity
		//----------------------------------------------------------------------
		info = GrB_Matrix_nvals(&n, l);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix reduced; // relation matrix reduced to only 'l' rows/cols
		info = GrB_Matrix_new(&reduced, GrB_BOOL, n, n);
		ASSERT(info == GrB_SUCCESS);

		// discard rows of 'r' associated with nodes of a different type than 'l'
		// this will also perform casting to boolean
		mapping = rm_malloc(sizeof(GrB_Index) * n);
		// extract row indecies from 'l', coresponding to node IDs
		info = GrB_Matrix_extractTuples_BOOL(mapping, GrB_NULL, GrB_NULL, &n, l);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_extract(reduced, GrB_NULL, GrB_NULL, r, mapping, n,
								  mapping, n, GrB_NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_free(&r);
		r = reduced;
	} else {
		// resize to remove unused rows
		n = Graph_UncompactedNodeCount(g);
		GxB_Matrix_resize(r, n, n);
	}

	// invoke Pagerank only if 'r' contains entries
	info = GrB_Matrix_nvals(&nvals, r);
	ASSERT(info == GrB_SUCCESS);

	if(nvals > 0) {
		info = Pagerank(&ranking, r, itermax, tol, &iters);
		ASSERT(info == GrB_SUCCESS);
	}

	// clean up
	GrB_free(&r);
	if(label) {
		GrB_free(&l);
	}

	// update context
	pdata->n        =  n;
	pdata->mapping  =  mapping;
	pdata->ranking  =  ranking;

	return PROCEDURE_OK;
}

SIValue *Proc_PagerankStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData);

	PagerankContext *pdata = (PagerankContext *)ctx->privateData;

	// depleted/no results
	if(pdata->i >= pdata->n || pdata->ranking == NULL) return NULL;

	LAGraph_PageRank rank = pdata->ranking[pdata->i++];
	NodeID node_id = (pdata->mapping) ? pdata->mapping[rank.page] : rank.page;

	Graph_GetNode(pdata->g, node_id, &pdata->node);
	if(pdata->yield_node)   *pdata->yield_node   =  SI_Node(&pdata->node);
	if(pdata->yield_score)  *pdata->yield_score  =  SI_DoubleVal(rank.pagerank);

	return pdata->output;
}

ProcedureResult Proc_PagerankFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		PagerankContext *pdata = ctx->privateData;
		if(pdata->output)   array_free(pdata->output);
		if(pdata->mapping)  rm_free(pdata->mapping);
		if(pdata->ranking)  rm_free(pdata->ranking);
		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_PagerankCtx() {
	void *privateData = NULL;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node = {.name = "node", .type = T_NODE};
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

