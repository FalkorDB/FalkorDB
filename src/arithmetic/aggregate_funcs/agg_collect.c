/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../datatypes/array.h"

//------------------------------------------------------------------------------
// Collect
//------------------------------------------------------------------------------

AggregateResult AGG_COLLECT(SIValue *argv, int argc, void *private_data) {
	AggregateCtx *ctx = private_data;

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	// SIArray_Append will clone the added value, ensuring it can be
	// safely accessed for the lifetime of the Collect context.
	SIArray_Append(&ctx->result, v);

	return AGGREGATE_OK;
}

AggregateCtx *Collect_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_Array(0);  // collect default value is an empty array
	ctx->private_data = NULL;

	return ctx;
}

void Register_COLLECT(void) {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	ret_type = T_NULL | T_ARRAY;
	func_desc = AR_AggFuncDescNew("collect", AGG_COLLECT, 1, 1, types, ret_type,
			NULL, NULL, Collect_PrivateData, "aggregate collect");
	AR_RegFunc(func_desc);
}

