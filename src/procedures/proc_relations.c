/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "proc_labels.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../graph/graphcontext.h"

// CALL db.relationshipTypes()

typedef struct {
	uint schema_id;     // Current schema ID.
	GraphContext *gc;   // Graph context.
	SIValue *output;    // Output label.
} RelationsContext;

ProcedureResult Proc_RelationsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 0) return PROCEDURE_ERR;

	RelationsContext *pdata = rm_malloc(sizeof(RelationsContext));

	pdata->schema_id  =  0;
	pdata->gc         =  QueryCtx_GetGraphCtx();
	pdata->output     =  array_new(SIValue, 1);

	ctx->privateData = pdata;
	return PROCEDURE_OK;
}

SIValue *Proc_RelationsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	RelationsContext *pdata = (RelationsContext *)ctx->privateData;

	// depleted?
	if(pdata->schema_id >= GraphContext_SchemaCount(pdata->gc, SCHEMA_EDGE))
		return NULL;

	// get schema name
	Schema *s = GraphContext_GetSchemaByID(pdata->gc, pdata->schema_id++, SCHEMA_EDGE);
	pdata->output[0] = SI_ConstStringVal(Schema_GetName(s));
	return pdata->output;
}

ProcedureResult Proc_RelationsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		RelationsContext *pdata = ctx->privateData;
		array_free(pdata->output);
		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_RelationsCtx() {
	void *privateData = NULL;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 1);
	ProcedureOutput output = {.name = "relationshipType", .type = T_STRING};
	array_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.relationshipTypes",
								   0,
								   outputs,
								   Proc_RelationsStep,
								   Proc_RelationsInvoke,
								   Proc_RelationsFree,
								   privateData,
								   true);
	return ctx;
}

