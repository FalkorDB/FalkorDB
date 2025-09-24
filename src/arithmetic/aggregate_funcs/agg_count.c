/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"

//------------------------------------------------------------------------------
// Count
//------------------------------------------------------------------------------

AggregateResult AGG_COUNT(SIValue *argv, int argc, void *private_data) {
	AggregateCtx *ctx = private_data;

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	// Increment the result.
	ctx->result.longval++;

	return AGGREGATE_OK;
}

AggregateCtx *Count_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_LongVal(0);  // count default value is 0
	ctx->private_data = NULL;

	return ctx;
}

void Register_COUNT(void) {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	ret_type = T_INT64;
	func_desc = AR_AggFuncDescNew("count", AGG_COUNT, 1, 1, types, ret_type,
			NULL, NULL, Count_PrivateData);
	AR_RegFunc(func_desc);
}

