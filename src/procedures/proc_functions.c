/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "procedure.h"
#include "../util/arr.h"
#include "../datatypes/array.h"
#include "../arithmetic/func_desc.h"
#include "../arithmetic/builtin_funcs_lookup.h"

// CALL dbms.functions()

typedef struct {
	int idx;                     // index into AR_BUILTIN_FUNCS[]
	SIValue *yield_name;         // yield name
	SIValue *yield_retType;      // yield return type
	SIValue *yield_args;         // yield arguments
	SIValue *yield_internal;     // yield internal
	SIValue *yield_reducible;    // yield reducible
	SIValue *yield_aggregation;  // yield aggregation
	SIValue *yield_varLen;       // yield variable len
	SIValue output[7];           // output array
} ProcFunctionsPrivateData;

static void _process_yield
(
	ProcFunctionsPrivateData *ctx,
	const char **yield
) {
	int idx = 0 ;

	for (uint i = 0; i < arr_len (yield); i++) {
		if (strcasecmp ("name", yield [i]) == 0) {
			ctx->yield_name = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("return_type", yield [i]) == 0) {
			ctx->yield_retType = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("arguments", yield [i]) == 0) {
			ctx->yield_args = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("internal", yield [i]) == 0) {
			ctx->yield_internal = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("reducible", yield [i]) == 0) {
			ctx->yield_reducible = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("aggregation", yield [i]) == 0) {
			ctx->yield_aggregation = ctx->output + idx++ ;
			continue ;
		}
		if (strcasecmp ("variable_len", yield [i]) == 0) {
			ctx->yield_varLen = ctx->output + idx++ ;
			continue ;
		}
	}
}

ProcedureResult Proc_FunctionsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if (arr_len ((SIValue *)args) != 0) {
		return PROCEDURE_ERR ;
	}

	ProcFunctionsPrivateData *pdata =
		rm_calloc (1, sizeof (ProcFunctionsPrivateData)) ;

	pdata->idx = 0 ;
	_process_yield (pdata, yield) ;

	ctx->privateData = pdata ;
	return PROCEDURE_OK ;
}

SIValue *Proc_FunctionsStep
(
	ProcedureCtx *ctx
) {
	ASSERT (ctx              != NULL) ;
	ASSERT (ctx->privateData != NULL) ;

	ProcFunctionsPrivateData *pdata = ctx->privateData ;

	if (pdata->idx >= NUM_BUILTIN_FUNCS) {
		return NULL ;
	}

	AR_FuncDesc *f = AR_BUILTIN_FUNCS [pdata->idx++] ;

	if (pdata->yield_name != NULL) {
		*pdata->yield_name = SI_ConstStringVal (f->name) ;
	}

	if (pdata->yield_retType != NULL) {
		char buf [1024] ;
		SIType_ToMultipleTypeString (f->ret_type, buf, sizeof (buf)) ;
		*pdata->yield_retType = SI_DuplicateStringVal (buf) ;
	}

	if (pdata->yield_args != NULL) {
		SIValue types = SI_Array (f->types_len) ;
		char buf [1024] ;
		for (int i = 0 ; i < f->types_len ; i++) {
			SIType_ToMultipleTypeString (f->types [i], buf, sizeof (buf)) ;
			SIArray_Append (&types, SI_ConstStringVal (buf)) ;
		}
		*pdata->yield_args = types ;
	}

	if (pdata->yield_internal    != NULL)
		*pdata->yield_internal    = SI_BoolVal (f->internal) ;
	if (pdata->yield_reducible   != NULL)
		*pdata->yield_reducible   = SI_BoolVal (f->reducible) ;
	if (pdata->yield_aggregation != NULL)
		*pdata->yield_aggregation = SI_BoolVal (f->aggregate) ;
	if (pdata->yield_varLen != NULL)
		*pdata->yield_varLen = SI_BoolVal (f->max_argc == VAR_ARG_LEN) ;

	return pdata->output ;
}

ProcedureResult Proc_FunctionsFree
(
	ProcedureCtx *ctx
) {
	if (ctx->privateData) {
		rm_free (ctx->privateData) ;
		ctx->privateData = NULL ;
	}
	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_FunctionsCtx(void) {
	ProcedureOutput output ;
	ProcedureOutput *outputs = arr_new(ProcedureOutput, 8) ;

	output = (ProcedureOutput){ .name="name",         .type=T_STRING } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="return_type",  .type=T_STRING } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="arguments",    .type=T_ARRAY  } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="internal",     .type=T_BOOL   } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="reducible",    .type=T_BOOL   } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="aggregation",  .type=T_BOOL   } ;
	arr_append (outputs, output) ;
	output = (ProcedureOutput){ .name="variable_len", .type=T_BOOL   } ;
	arr_append (outputs, output) ;

	return ProcCtxNew (
		"dbms.functions",
		0,
		outputs,
		Proc_FunctionsStep,
		Proc_FunctionsInvoke,
		Proc_FunctionsFree,
		NULL,
		true) ;
}
