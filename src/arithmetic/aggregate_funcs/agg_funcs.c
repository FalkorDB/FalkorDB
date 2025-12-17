/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../../value.h"
#include "../../util/rmalloc.h"

// create a new aggregation function descriptor
AR_FuncDesc *AR_AggFuncDescNew
(
	const char *name,                   // function name
	AR_Func func,                       // pointer to function
	uint min_argc,                      // minimum number of arguments
	uint max_argc,                      // maximum number of arguments
	SIType *types,                      // acceptable types
	SIType ret_type,                    // return type
	AR_Func_Free free,                  // free aggregation callback
	AR_Func_Finalize finalize,          // finalize aggregation callback
	AR_Func_PrivateData private_data    // generate private data
) {
	AR_FuncDesc *desc = rm_calloc(1, sizeof(AR_FuncDesc));

	desc->name                   = name;
	desc->func                   = func;
	desc->types                  = types;
	desc->ret_type               = ret_type;
	desc->min_argc               = min_argc;
	desc->max_argc               = max_argc;
	desc->internal               = false;
	desc->aggregate              = true;
	desc->reducible              = false;
	desc->deterministic          = true;
	desc->callbacks.free         = free;
	desc->callbacks.finalize     = finalize;
	desc->callbacks.private_data = private_data;

	return desc;
}

// TODO: might be deprecated?
// routine for cloning a generic aggregate function context
void *Aggregate_Clone(void *orig) {
	AggregateCtx *orig_ctx = orig;
	AggregateCtx *ctx_clone = rm_malloc(sizeof(AggregateCtx));
	ctx_clone->result = SI_CloneValue(orig_ctx->result);
	ctx_clone->private_data = NULL;
	return ctx_clone;
}

// finalize the result of an aggregate function
void Aggregate_SetResult
(
	AggregateCtx *ctx,
	SIValue result
) {
	ctx->result = result;
}

void Aggregate_Finalize
(
	AR_FuncDesc *func_desc,
	AggregateCtx *ctx
) {
	ASSERT(func_desc != NULL);
	ASSERT(ctx != NULL);

	if(func_desc->callbacks.finalize) {
		func_desc->callbacks.finalize(ctx);
	}
}

// get aggregated result
SIValue Aggregate_GetResult
(
	AggregateCtx *ctx
) {
	return SI_TransferOwnership(&ctx->result);
}

//------------------------------------------------------------------------------
// Aggregation function registration
//------------------------------------------------------------------------------

// forward declarations
void Register_AVG        (void);
void Register_SUM        (void);
void Register_MAX        (void);
void Register_MIN        (void);
void Register_STD        (void);
void Register_COUNT      (void);
void Register_COLLECT    (void);
void Register_PRECENTILE (void);

// register all aggregation functions
void Register_AggFuncs() {
	Register_AVG();
	Register_SUM();
	Register_MAX();
	Register_MIN();
	Register_STD();
	Register_COUNT();
	Register_COLLECT();
	Register_PRECENTILE();
}

// routine for freeing a generic aggregate function context
void Aggregate_Free
(
	AR_FuncDesc *agg_func,
	AggregateCtx *ctx
) {
	if(ctx == NULL) return;

	SIValue_Free(ctx->result);

	if(ctx->private_data) {
		ASSERT(agg_func->callbacks.free != NULL);
		agg_func->callbacks.free(ctx->private_data);
	}

	rm_free(ctx);
}

