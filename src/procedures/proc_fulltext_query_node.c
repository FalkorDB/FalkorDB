/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/index_doc_key.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "proc_fulltext_query.h"
#include "../graph/graphcontext.h"

//------------------------------------------------------------------------------
// fulltext createNodeIndex
//------------------------------------------------------------------------------

// CALL db.idx.fulltext.queryNodes(label, query)

typedef struct {
	Node n;
	Graph *g;
	SIValue output[2];
	Index idx;
	RSIndex *rsIdx;          // strong ref on RediSearch index, held for the procedure's lifetime
	RSResultsIterator *iter;
	SIValue *yield_node;     // yield node
	SIValue *yield_score;    // yield score
} QueryNodeContext;

static void _process_yield
(
	QueryNodeContext *ctx,
	const char **yield
) {
	ctx->yield_node   =    NULL;
	ctx->yield_score  =    NULL;

	int idx = 0;
	for(uint i = 0; i < arr_len(yield); i++) {
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

ProcedureResult Proc_FulltextQueryNodeInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(arr_len((SIValue *)args) != 2) return PROCEDURE_ERR;
	if(!(SI_TYPE(args[0]) & SI_TYPE(args[1]) & T_STRING)) return PROCEDURE_ERR;

	ctx->privateData = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// see if there's a full-text index for given label
	char *err = NULL;
	const char *label = args[0].stringval;
	const char *query = args[1].stringval;

	// get full-text index from schema
	Index idx = GraphContext_GetIndex(gc, label, NULL, 0, INDEX_FLD_ANY,
			SCHEMA_NODE);
	if(!idx) return PROCEDURE_ERR; // TODO: this should cause an error to be emitted

	ctx->privateData = rm_malloc(sizeof(QueryNodeContext));
	QueryNodeContext *pdata = ctx->privateData;

	pdata->g     = GraphContext_GetGraph (gc) ;
	pdata->n     = GE_NEW_NODE();
	pdata->idx   = idx;
	// Strong ref on the RediSearch index for the procedure's lifetime;
	// released in Proc_FulltextQueryNodeFree.
	pdata->rsIdx = Index_AcquireRSIndex(idx);
	if(pdata->rsIdx == NULL) {
		rm_free(pdata);
		ctx->privateData = NULL;
		return PROCEDURE_ERR;
	}

	_process_yield(pdata, yield);

	// execute query
	pdata->iter = Index_Query(pdata->idx, query, &err);

	// raise runtime exception if err != NULL
	if(err) {
		// RediSearch error message is allocated using `rm_strdup`
		// QueryCtx is expecting to free `error` using `free`
		// in which case we have no option but to clone error
		ErrorCtx_SetError(EMSG_REDISEARCH, err);
		rm_free(err);
		// raise the exception, we expect an exception handler to be set
		// as procedure invocation is done at runtime
		ErrorCtx_RaiseRuntimeException(NULL);
	}

	ASSERT(pdata->iter != NULL);

	return PROCEDURE_OK;
}

SIValue *Proc_FulltextQueryNodeStep
(
	ProcedureCtx *ctx
) {
	if(!ctx->privateData) return NULL; // no index was attached to this procedure

	QueryNodeContext *pdata = (QueryNodeContext *)ctx->privateData;
	if(!pdata || !pdata->iter) return NULL;

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0;
	const char *doc_key = (const char *)RediSearch_ResultsIteratorNext(pdata->iter,
			pdata->rsIdx, &len);

	// depleted
	if(!doc_key) return NULL;

	NodeID id;
	IndexDocKey_DecodeNode(doc_key, &id);

	double score = RediSearch_ResultsIteratorGetScore(pdata->iter);

	// get node
	Node *n = &pdata->n;
	Graph_GetNode(pdata->g, id, n);

	if(pdata->yield_node)  *pdata->yield_node  = SI_Node(n);
	if(pdata->yield_score) *pdata->yield_score = SI_DoubleVal(score);

	return pdata->output;
}

ProcedureResult Proc_FulltextQueryNodeFree
(
	ProcedureCtx *ctx
) {
	// Clean up.
	if(!ctx->privateData) return PROCEDURE_OK;

	QueryNodeContext *pdata = ctx->privateData;
	if(pdata->iter) RediSearch_ResultsIteratorFree(pdata->iter);
	// release the strong ref on the RediSearch index, AFTER iter free.
	if(pdata->rsIdx) Index_ReleaseRSIndex(pdata->rsIdx);
	rm_free(pdata);

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_FulltextQueryNodeGen() {
	void *privateData = NULL;
	ProcedureOutput *output   = arr_new(ProcedureOutput, 2);
	ProcedureOutput out_node  = {.name = "node", .type = T_NODE};
	ProcedureOutput out_score = {.name = "score", .type = T_DOUBLE};
	arr_append(output, out_node);
	arr_append(output, out_score);

	ProcedureCtx *ctx = ProcCtxNew("db.idx.fulltext.queryNodes",
								   2,
								   output,
								   Proc_FulltextQueryNodeStep,
								   Proc_FulltextQueryNodeInvoke,
								   Proc_FulltextQueryNodeFree,
								   privateData,
								   true);
	return ctx;
}

