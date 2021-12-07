/*
* Copyright 2018-2021 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "proc_list_indexes.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../schema/schema.h"
#include "../datatypes/array.h"

typedef struct {
	SIValue *out;               // outputs
	int node_schema_id;         // current node schema ID
	int edge_schema_id;         // current edge schema ID
	IndexType type;             // current index type to retrieve
	GraphContext *gc;           // graph context
	SIValue *yield_type;        // yield index type
	SIValue *yield_label;       // yield index label
	SIValue *yield_properties;  // yield index properties
} IndexesContext;

static void _process_yield
(
	IndexesContext *ctx,
	const char **yield
) {
	ctx->yield_type        = NULL;
	ctx->yield_label       = NULL;
	ctx->yield_properties  = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("type", yield[i]) == 0) {
			ctx->yield_type = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("label", yield[i]) == 0) {
			ctx->yield_label = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("properties", yield[i]) == 0) {
			ctx->yield_properties = ctx->out + idx;
			idx++;
			continue;
		}
	}
}

// CALL db.indexes()
ProcedureResult Proc_IndexesInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {

	ASSERT(ctx   != NULL);
	ASSERT(args  != NULL);
	ASSERT(yield != NULL);

	// TODO: introduce invoke validation, similar to arithmetic expressions
	// expecting no arguments
	uint arg_count = array_len((SIValue *)args);
	if(arg_count != 0) return PROCEDURE_ERR;

	GraphContext *gc = QueryCtx_GetGraphCtx();

	IndexesContext *pdata    = rm_malloc(sizeof(IndexesContext));
	pdata->gc                = gc;
	pdata->out               = array_new(SIValue, 6);
	pdata->type              = IDX_EXACT_MATCH;
	pdata->node_schema_id    = GraphContext_SchemaCount(gc, SCHEMA_NODE) - 1;
	pdata->edge_schema_id    = GraphContext_SchemaCount(gc, SCHEMA_EDGE) - 1;

	_process_yield(pdata, yield);

	ctx->privateData = pdata;

	return PROCEDURE_OK;
}

static bool _EmitIndex
(
	IndexesContext *ctx,
	const Schema *s,
	IndexType type
) {
	Index *idx = Schema_GetIndex(s, NULL, type);
	if(idx == NULL) return false;

	if(ctx->yield_type != NULL) {
		if(type == IDX_EXACT_MATCH) {
			*ctx->yield_type = SI_ConstStringVal("exact-match");
		} else {
			*ctx->yield_type = SI_ConstStringVal("full-text");
		}
	}

	if(ctx->yield_label) {
		*ctx->yield_label = SI_ConstStringVal((char *)Schema_GetName(s));
	}

	if(ctx->yield_properties) {
		uint fields_count        = Index_FieldsCount(idx);
		const char **fields      = Index_GetFields(idx);
		*ctx->yield_properties   = SI_Array(fields_count);

		for(uint i = 0; i < fields_count; i++) {
			SIArray_Append(ctx->yield_properties,
						   SI_ConstStringVal((char *)fields[i]));
		}
	}

	return true;
}

static SIValue *Schema_Step
(
	int *schema_id,
	SchemaType t,
	IndexesContext *pdata
) {
	Schema *s = NULL;

	// loop over all schemas from last to first
	while(*schema_id >= 0) {
		s = GraphContext_GetSchemaByID(pdata->gc, *schema_id, t);
		if(!Schema_HasIndices(s)) {
			// no indexes found, continue to the next schema
			(*schema_id)--;
			continue;
		}

		// populate index data if one is found
		bool found = _EmitIndex(pdata, s, pdata->type);

		if(pdata->type == IDX_FULLTEXT) {
			// all indexes retrieved; update schema_id, reset schema type
			(*schema_id)--;
			pdata->type = IDX_EXACT_MATCH;
		} else {
			// next iteration will check the same schema for a full-text index
			pdata->type = IDX_FULLTEXT;
		}

		if(found) return pdata->out;
	}

	return NULL;
}

SIValue *Proc_IndexesStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	SIValue *res;
	IndexesContext *pdata = ctx->privateData;

	res = Schema_Step(&pdata->node_schema_id, SCHEMA_NODE, pdata);
	if(res != NULL) return res;

	return Schema_Step(&pdata->edge_schema_id, SCHEMA_EDGE, pdata);
}

ProcedureResult Proc_IndexesFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		IndexesContext *pdata = ctx->privateData;
		array_free(pdata->out);
		rm_free(pdata);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_IndexesCtx() {
	void *privateData = NULL;
	ProcedureOutput output;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 3);

	// index type (exact-match / fulltext)
	output = (ProcedureOutput) {
		.name = "type", .type = T_STRING
	};
	array_append(outputs, output);

	// indexed label
	output = (ProcedureOutput) {
		.name = "label", .type = T_STRING
	};
	array_append(outputs, output);

	// indexed properties
	output = (ProcedureOutput) {
		.name = "properties", .type = T_ARRAY
	};
	array_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.indexes",
								   0,
								   outputs,
								   Proc_IndexesStep,
								   Proc_IndexesInvoke,
								   Proc_IndexesFree,
								   privateData,
								   true);
	return ctx;
}

