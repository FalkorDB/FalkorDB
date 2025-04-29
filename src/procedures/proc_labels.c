/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "proc_labels.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../graph/graphcontext.h"

// CALL db.labels()

typedef struct {
	uint schema_id;     // current schema id
	GraphContext *gc;   // graph context
	SIValue *output;    // output label
} LabelsContext;

ProcedureResult Proc_LabelsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 0) return PROCEDURE_ERR;

	LabelsContext *pdata = rm_malloc(sizeof(LabelsContext));

	pdata->gc        = QueryCtx_GetGraphCtx();
	pdata->output    = array_new(SIValue, 1);
	pdata->schema_id = 0;

	array_append(pdata->output, SI_ConstStringVal("label"));

	ctx->privateData = pdata;
	return PROCEDURE_OK;
}

SIValue *Proc_LabelsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	LabelsContext *pdata = (LabelsContext *)ctx->privateData;

	// depleted?
	if(pdata->schema_id >= GraphContext_SchemaCount(pdata->gc, SCHEMA_NODE))
		return NULL;

	// get schema label
	Schema *s = GraphContext_GetSchemaByID(pdata->gc, pdata->schema_id++, SCHEMA_NODE);
	char *label = (char *)Schema_GetName(s);
	pdata->output[0] = SI_ConstStringVal(label);
	return pdata->output;
}

ProcedureResult Proc_LabelsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		LabelsContext *pdata = ctx->privateData;
		array_free(pdata->output);
		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_LabelsCtx() {
	void *privateData = NULL;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 1);
	ProcedureOutput output = {.name = "label", .type = T_STRING};
	array_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.labels",
								   0,
								   outputs,
								   Proc_LabelsStep,
								   Proc_LabelsInvoke,
								   Proc_LabelsFree,
								   privateData,
								   true);
	return ctx;
}

