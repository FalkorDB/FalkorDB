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

AggregateResult AGG_COLLECT
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	AggregateCtx *ctx = private_data;

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	// depending on the value's allocation type we'll entier
	// add a clone of it to the array (SIArray_Append clones the value)
	// or transfer owership of the value to the array
	SIAllocation allocation = SI_ALLOCATION(argv);

	switch(allocation) {
		case M_NONE:
		case M_CONST:
		case M_VOLATILE:
			// array will clone the value
			SIArray_Append(&ctx->result, v);
			break;

		case M_SELF:
			// array will take ownership over the value
			SIArray_AppendAsOwner(&ctx->result, argv);
			break;

		default:
			ASSERT(false && "unknown allocation type");
			break;
	}

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
			NULL, NULL, Collect_PrivateData);
	AR_RegFunc(func_desc);
}

