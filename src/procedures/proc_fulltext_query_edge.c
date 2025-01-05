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
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "proc_fulltext_query.h"
#include "../graph/graphcontext.h"

typedef struct {
	Edge e;
	Graph *g;
	Schema *s;
	SIValue *output;
	Index idx;
	RSResultsIterator *iter;
	SIValue *yield_relationship;     // yield relationship
	SIValue *yield_score;            // yield score
} QueryRelationshipContext;

SIValue *Proc_FulltextQueryRelationshipStep
(
	ProcedureCtx *ctx
) {
	if(!ctx->privateData) return NULL; // no index was attached to this procedure

	QueryRelationshipContext *pdata = (QueryRelationshipContext *)ctx->privateData;
	if(!pdata || !pdata->iter) return NULL;

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0;
	const EdgeIndexKey *edge_key = (EdgeIndexKey *)RediSearch_ResultsIteratorNext(pdata->iter,
			Index_RSIndex(pdata->idx), &len);

	// depleted
	if(!edge_key) return NULL;

	double score = RediSearch_ResultsIteratorGetScore(pdata->iter);

	// get edge
	Edge *e = &pdata->e;
	pdata->e.src_id = edge_key->src_id;
	pdata->e.dest_id = edge_key->dest_id;
	pdata->e.relationID = pdata->s->id;
	EntityID edge_id = edge_key->edge_id;
	bool edge_exists = Graph_GetEdge(pdata->g, edge_id, e);
	ASSERT(edge_exists);
	
	if(pdata->yield_relationship)  *pdata->yield_relationship  = SI_Edge(e);
	if(pdata->yield_score) *pdata->yield_score = SI_DoubleVal(score);

	return pdata->output;
}

ProcedureResult Proc_FulltextQueryRelationshipFree
(
	ProcedureCtx *ctx
) {
	// Clean up.
	if(!ctx->privateData) return PROCEDURE_OK;

	QueryRelationshipContext *pdata = ctx->privateData;
	array_free(pdata->output);
	if(pdata->iter) RediSearch_ResultsIteratorFree(pdata->iter);
	rm_free(pdata);

	return PROCEDURE_OK;
}


static void _relationship_process_yield
(
	QueryRelationshipContext *ctx,
	const char **yield
) {
	ctx->yield_relationship = NULL;
	ctx->yield_score        = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("relationship", yield[i]) == 0) {
			ctx->yield_relationship = ctx->output + idx;
			idx++;
			continue;
		}

		else if(strcasecmp("score", yield[i]) == 0) {
			ctx->yield_score = ctx->output + idx;
			idx++;
		}
	}
}

ProcedureResult Proc_FulltextQueryRelationshipInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 2) return PROCEDURE_ERR;
	if(!(SI_TYPE(args[0]) & SI_TYPE(args[1]) & T_STRING)) return PROCEDURE_ERR;

	ctx->privateData = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// see if there's a full-text index for given relationship-type
	char *err            = NULL;
	const char *query    = args[1].stringval;
	const char *relation = args[0].stringval;

	// get full-text index from schema
	Index idx = GraphContext_GetIndex(gc, relation, NULL, 0, INDEX_FLD_ANY,
			SCHEMA_EDGE);
	if(!idx) return PROCEDURE_ERR; // TODO: this should cause an error to be emitted

    Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
	if(s == NULL) return PROCEDURE_ERR;

	ctx->privateData = rm_calloc(1, sizeof(QueryRelationshipContext));
	QueryRelationshipContext *pdata = ctx->privateData;

	// populate context
	pdata->g      = gc->g;
	pdata->s	  = s;
	pdata->idx    = idx;
	pdata->output = array_new(SIValue,  2);

	_relationship_process_yield(pdata, yield);

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

// implement full-text query for relationships
// CALL db.idx.fulltext.queryRelationships(label, query)
// Example:
// GRAPH.QUERY falkor "CREATE ()-[r:WORKS_WITH {title: 'Joe DiMaggio'}]->()"
// GRAPH.QUERY falkor "CALL db.idx.fulltext.queryRelationships('WORKS_WITH', 'Joe')
//                     YIELD relationship
//                     RETURN relationship.title"

ProcedureCtx *Proc_FulltextQueryRelationshipGen() {
	void *privateData = NULL;

	ProcedureOutput *output   = array_new(ProcedureOutput, 2);
	ProcedureOutput out_node  = {.name = "relationship", .type = T_EDGE};
	ProcedureOutput out_score = {.name = "score", .type = T_DOUBLE};
	array_append(output, out_node);
	array_append(output, out_score);

	ProcedureCtx *ctx = ProcCtxNew("db.idx.fulltext.queryRelationships",
								   2,
								   output,
								   Proc_FulltextQueryRelationshipStep,
								   Proc_FulltextQueryRelationshipInvoke,
								   Proc_FulltextQueryRelationshipFree,
								   privateData,
								   true);
	return ctx;
}

