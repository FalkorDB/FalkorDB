/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "rax.h"
#include "procedure.h"
#include "../util/arr.h"
#include "../datatypes/array.h"
#include "../arithmetic/func_desc.h"

extern FuncsRepo *__aeRegisteredFuncs ;

// CALL dbms.functions()

typedef struct {
	raxIterator iter;            // functions iterator
	SIValue *yield_name;         // yield name
	SIValue *yield_retType;      // yield return type
	SIValue *yield_args;         // yield arguments
	SIValue *yield_internal;     // yield internal
	SIValue *yield_reducible;    // yield reducible
	SIValue *yield_aggregation;  // yield aggregation
	SIValue *yield_udf;          // yield udf
	SIValue *yield_varLen;       // yield variable len
	SIValue output[8];           // output array
} ProcFunctionsPrivateData;

static void _process_yield
(
	ProcFunctionsPrivateData *ctx,
	const char **yield
) {
	int idx = 0 ;

	for (uint i = 0; i < array_len(yield); i++) {
		if (strcasecmp ("name", yield[i]) == 0) {
			ctx->yield_name = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("return_type", yield[i]) == 0) {
			ctx->yield_retType = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("arguments", yield[i]) == 0) {
			ctx->yield_args = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("internal", yield[i]) == 0) {
			ctx->yield_internal = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("reducible", yield[i]) == 0) {
			ctx->yield_reducible = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("aggregation", yield[i]) == 0) {
			ctx->yield_aggregation = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("udf", yield[i]) == 0) {
			ctx->yield_udf = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		if (strcasecmp ("variable_len", yield[i]) == 0) {
			ctx->yield_varLen = ctx->output + idx ;
			idx++ ;
			continue ;
		}
	}
}

ProcedureResult Proc_FunctionsInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting no arguments
	if (array_len ((SIValue *)args) != 0) {
		return PROCEDURE_ERR ;
	}

	ProcFunctionsPrivateData *pdata =
		rm_calloc(1, sizeof(ProcFunctionsPrivateData));

	// initialize an iterator to the rax that contains all functions
	rax *functions = __aeRegisteredFuncs->repo ;
	raxStart (&pdata->iter, functions) ;
	raxSeek (&pdata->iter, "^", NULL, 0) ;
	_process_yield (pdata, yield) ;

	ctx->privateData = pdata ;

	return PROCEDURE_OK ;
}

// advance the iterator to the next function and yield
SIValue *Proc_FunctionsStep
(
	ProcedureCtx *ctx
) {
	ASSERT (ctx              != NULL) ;
	ASSERT (ctx->privateData != NULL) ;

	ProcFunctionsPrivateData *pdata =
		(ProcFunctionsPrivateData *)ctx->privateData;

	// depleted?
	if (!raxNext(&pdata->iter)) {
		return NULL;
	}

	// returns the current function
	AR_FuncDesc *f = pdata->iter.data;

	// yield function name
	if (pdata->yield_name != NULL) {
		*pdata->yield_name = SI_ConstStringVal (f->name) ;
	}

	// yield function return type
	if (pdata->yield_retType != NULL) {
		char buf[1024] ;
		size_t bufferLen = 1024 ;
		SIType_ToMultipleTypeString (f->ret_type, buf, bufferLen) ;
		*pdata->yield_retType = SI_DuplicateStringVal (buf);
	}

	// yield function args
	if (pdata->yield_args != NULL) {
		int l = array_len (f->types) ;
		SIValue types = SI_Array (l) ;

		char buf[1024] ;
		size_t bufferLen = 1024 ;

		for (int i = 0; i < l; i++) {
			SIType t = f->types[i] ;
			SIType_ToMultipleTypeString (f->types[i], buf, bufferLen) ;
			SIArray_Append (&types, SI_ConstStringVal (buf)) ;
		}

		*pdata->yield_args = types ;
	}

	// yield internal
	if (pdata->yield_internal != NULL) {
		*pdata->yield_internal = SI_BoolVal (f->internal) ;
	}

	// yield reducible
	if (pdata->yield_reducible != NULL) {
		*pdata->yield_reducible = SI_BoolVal (f->internal) ;
	}

	// yield aggregation
	if (pdata->yield_aggregation != NULL) {
		*pdata->yield_aggregation = SI_BoolVal (f->aggregate) ;
	}

	// yield udf
	if (pdata->yield_udf != NULL) {
		*pdata->yield_udf = SI_BoolVal (f->udf) ;
	}

	// yield var len
	if (pdata->yield_varLen != NULL) {
		*pdata->yield_varLen = SI_BoolVal (f->max_argc == VAR_ARG_LEN) ;
	}

	return pdata->output;
}

ProcedureResult Proc_FunctionsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if (ctx->privateData) {
		ProcFunctionsPrivateData *pdata = ctx->privateData ;
		raxStop (&pdata->iter) ;
		rm_free (ctx->privateData) ;
	}

	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_FunctionsCtx (void) {
	void *privateData = NULL ;

	ProcedureOutput output ;
	ProcedureOutput *outputs = array_new (ProcedureOutput, 8) ;

	output = (ProcedureOutput) {.name = "name", .type = T_STRING} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "return_type", .type = T_STRING} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "arguments", .type = T_ARRAY} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "internal", .type = T_BOOL} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "reducible", .type = T_BOOL} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "aggregation", .type = T_BOOL} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "variable_len", .type = T_BOOL} ;
	array_append (outputs, output) ;

	output = (ProcedureOutput) {.name = "udf", .type = T_BOOL} ;
	array_append (outputs, output) ;

	ProcedureCtx *ctx = ProcCtxNew (
			"dbms.functions",
			0,
			outputs,
			Proc_FunctionsStep,
			Proc_FunctionsInvoke,
			Proc_FunctionsFree,
			privateData,
			true) ;

	return ctx;
}

