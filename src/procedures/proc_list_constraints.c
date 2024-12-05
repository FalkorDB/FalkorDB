/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "proc_list_constraints.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../constraint/constraint.h"
#include "../schema/schema.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"

typedef struct {
	SIValue *out;               // outputs
	SIValue *yield_type;        // yield constraint type
	SIValue *yield_label;       // yield constraint label
	SIValue *yield_properties;  // yield constraint properties
	SIValue *yield_entity_type; // yield constraint entity type
	SIValue *yield_status;      // yield constraint status
	GraphContext *gc;           // graph context
	Constraint *constraints;    // constraints
} ConstraintsContext;

static void _process_yield
(
	ConstraintsContext *ctx,
	const char **yield
) {
	ctx->yield_type        = NULL;
	ctx->yield_label       = NULL;
	ctx->yield_status      = NULL;
	ctx->yield_properties  = NULL;
	ctx->yield_entity_type = NULL;

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

		if(strcasecmp("entitytype", yield[i]) == 0) {
			ctx->yield_entity_type = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("status", yield[i]) == 0) {
			ctx->yield_status = ctx->out + idx;
			idx++;
			continue;
		}
	}
}

static void _EmitConstraint
(
	const Constraint c,
	ConstraintsContext *ctx
) {
    ASSERT(c != NULL);

	GraphContext *gc = ctx->gc;

	//--------------------------------------------------------------------------
	// constraint entity type
	//--------------------------------------------------------------------------

	if(ctx->yield_entity_type != NULL) {
		GraphEntityType t = Constraint_GetEntityType(c);
		if(t == GETYPE_NODE) {
			*ctx->yield_entity_type = SI_ConstStringVal("NODE");
		} else {
			*ctx->yield_entity_type = SI_ConstStringVal("RELATIONSHIP");
		}
	}

	//--------------------------------------------------------------------------
	// constraint status
	//--------------------------------------------------------------------------

	if(ctx->yield_status != NULL) {
		ConstraintStatus status = Constraint_GetStatus(c);
		if(status == CT_ACTIVE) {
			*ctx->yield_status = SI_ConstStringVal("OPERATIONAL");
		} else if (status == CT_PENDING) {
			*ctx->yield_status = SI_ConstStringVal("UNDER CONSTRUCTION");
		} else {
			*ctx->yield_status = SI_ConstStringVal("FAILED");
		}
	}

	//--------------------------------------------------------------------------
	// constraint type
	//--------------------------------------------------------------------------

	if(ctx->yield_type != NULL) {
		ConstraintType t = Constraint_GetType(c);
		if(t == CT_UNIQUE) {
			*ctx->yield_type = SI_ConstStringVal("UNIQUE");
		} else {
			*ctx->yield_type = SI_ConstStringVal("MANDATORY");
		}
	}

	//--------------------------------------------------------------------------
	// constraint label
	//--------------------------------------------------------------------------

	if(ctx->yield_label) {
		SchemaType t = (Constraint_GetEntityType(c) == GETYPE_NODE) ?
			SCHEMA_NODE : SCHEMA_EDGE;
		Schema *s = GraphContext_GetSchemaByID(gc, Constraint_GetSchemaID(c), t);
		*ctx->yield_label = SI_ConstStringVal((char *)Schema_GetName(s));
	}

	//--------------------------------------------------------------------------
	// constraint fields
	//--------------------------------------------------------------------------

	if(ctx->yield_properties) {
		const char **props;
		uint8_t n = Constraint_GetAttributes(c, NULL, &props);
		*ctx->yield_properties = SI_Array(n);

		for(uint8_t i = 0; i < n; i++) {
			SIArray_Append(ctx->yield_properties,
					SI_ConstStringVal((char *)props[i]));
		}
	}
}

SIValue *Proc_ConstraintsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	SIValue *res;
	ConstraintsContext *pdata = ctx->privateData;

	// no more constraints to emit
	if(array_len(pdata->constraints) == 0) {
		return NULL;
	}

	Constraint c = array_pop(pdata->constraints);
	_EmitConstraint(c, pdata);

	return pdata->out;
}

// CALL db.constraints()
ProcedureResult Proc_ConstraintsInvoke
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

	ConstraintsContext *pdata = rm_malloc(sizeof(ConstraintsContext));

	pdata->gc          = gc;
	pdata->out         = array_new(SIValue, 5);
    pdata->constraints = array_new(Constraint, 0);

	_process_yield(pdata, yield);

	ctx->privateData = pdata;

	//--------------------------------------------------------------------------
	// collect constraints
	//--------------------------------------------------------------------------

	SchemaType schema_types[2] = {SCHEMA_NODE, SCHEMA_EDGE};

	// foreach schema type
	for(int i = 0; i < 2; i++) {
		SchemaType schema_type = schema_types[i];
		ushort n = GraphContext_SchemaCount(gc, schema_type);

		// foreach schema
		for(ushort j = 0; j < n; j++) {
			Schema *s = GraphContext_GetSchemaByID(gc, j, schema_type);
			const Constraint *cs = Schema_GetConstraints(s);
			// foreach schema's constraint

			for(uint32_t k = 0; k < array_len((Constraint*)cs); k++) {
				array_append(pdata->constraints, cs[k]);
			}
		}
	}

	return PROCEDURE_OK;
}

ProcedureResult Proc_ConstraintsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		ConstraintsContext *pdata = ctx->privateData;
		array_free(pdata->out);
		array_free(pdata->constraints);
		rm_free(pdata);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_ConstraintsCtx(void) {
	void *privateData = NULL;
	ProcedureOutput output;
	ProcedureOutput *outputs = array_new(ProcedureOutput, 5);

	// constraint type (unique / mandatory)
	output = (ProcedureOutput) {
		.name = "type", .type = T_STRING
	};
	array_append(outputs, output);

	// constraint label
	output = (ProcedureOutput) {
		.name = "label", .type = T_STRING
	};
	array_append(outputs, output);

	// constraint properties
	output = (ProcedureOutput) {
		.name = "properties", .type = T_ARRAY
	};
	array_append(outputs, output);

	// constraint entity type (node / relationship)
	output = (ProcedureOutput) {
		.name = "entitytype", .type = T_STRING
	};
	array_append(outputs, output);

	// constraint status (operational / under construction/ failed)
	output = (ProcedureOutput) {
		.name = "status", .type = T_STRING
	};
	array_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.constraints",
								   0,
								   outputs,
								   Proc_ConstraintsStep,
								   Proc_ConstraintsInvoke,
								   Proc_ConstraintsFree,
								   privateData,
								   true,
                                   "Yield all constraints in the graph, denoting constraint type (UNIQIE/MANDATORY), \
which label/relationship-type and properties each enforces.");
	return ctx;
}

