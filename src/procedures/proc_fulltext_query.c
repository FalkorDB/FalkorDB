/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "proc_fulltext_query.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"

//------------------------------------------------------------------------------
// fulltext createNodeIndex
//------------------------------------------------------------------------------

// CALL db.idx.fulltext.queryNodes(label, query)

typedef struct {
	Node n;
	Graph *g;
	SIValue *output;
	Index idx;
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

ProcedureResult Proc_FulltextQueryNodeInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 2) return PROCEDURE_ERR;
	if(!(SI_TYPE(args[0]) & SI_TYPE(args[1]) & T_STRING)) return PROCEDURE_ERR;

	ctx->privateData = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// see if there's a full-text index for given label
	char *err = NULL;
	const char *label = args[0].stringval;
	const char *query = args[1].stringval;

	// get full-text index from schema
	Index idx = GraphContext_GetIndex(gc, label, NULL, 0, IDX_FULLTEXT,
			SCHEMA_NODE);
	if(!idx) return PROCEDURE_ERR; // TODO: this should cause an error to be emitted

	ctx->privateData = rm_malloc(sizeof(QueryNodeContext));
	QueryNodeContext *pdata = ctx->privateData;

	pdata->g      = gc->g;
	pdata->n      = GE_NEW_NODE();
	pdata->idx    = idx;
	pdata->output = array_new(SIValue,  2);

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
	NodeID *id = (NodeID *)RediSearch_ResultsIteratorNext(pdata->iter,
			Index_RSIndex(pdata->idx), &len);

	// depleted
	if(!id) return NULL;

	double score = RediSearch_ResultsIteratorGetScore(pdata->iter);

	// get node
	Node *n = &pdata->n;
	Graph_GetNode(pdata->g, *id, n);

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
	array_free(pdata->output);
	if(pdata->iter) RediSearch_ResultsIteratorFree(pdata->iter);
	rm_free(pdata);

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_FulltextQueryNodeGen() {
	void *privateData = NULL;
	ProcedureOutput *output   = array_new(ProcedureOutput, 2);
	ProcedureOutput out_node  = {.name = "node", .type = T_NODE};
	ProcedureOutput out_score = {.name = "score", .type = T_DOUBLE};
	array_append(output, out_node);
	array_append(output, out_score);

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

