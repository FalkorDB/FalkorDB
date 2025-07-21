/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "proc_property_keys.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../graph/graphcontext.h"

// CALL db.propertyKeys()

typedef struct {
	uint prop_id;       // current property ID
	GraphContext *gc;   // graph context
	SIValue *output;    // output label
} RelationsContext;

ProcedureResult Proc_PropKeysInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 0) return PROCEDURE_ERR;

	RelationsContext *pdata = rm_malloc(sizeof(RelationsContext));

	pdata->prop_id  =  0;
	pdata->gc       =  QueryCtx_GetGraphCtx();
	pdata->output   =  array_new(SIValue, 1);

	array_append(pdata->output, SI_ConstStringVal("propertyKey"));

	ctx->privateData = pdata;
	return PROCEDURE_OK;
}

SIValue *Proc_PropKeysStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	RelationsContext *pdata = (RelationsContext *)ctx->privateData;

	// depleted?
	if(pdata->prop_id >= GraphContext_AttributeCount(pdata->gc))
		return NULL;

	// get attribute name
	char *name = (char *)GraphContext_GetAttributeString(pdata->gc, pdata->prop_id++);
	pdata->output[0] = SI_ConstStringVal(name);
	return pdata->output;
}

ProcedureResult Proc_PropKeysFree
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

ProcedureCtx *Proc_PropKeysCtx() {
	void *privateData = NULL;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 1);
	ProcedureOutput output = {.name = "propertyKey", .type = T_STRING};
	array_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.propertyKeys",
								   0,
								   outputs,
								   Proc_PropKeysStep,
								   Proc_PropKeysInvoke,
								   Proc_PropKeysFree,
								   privateData,
								   true);
	return ctx;
}

