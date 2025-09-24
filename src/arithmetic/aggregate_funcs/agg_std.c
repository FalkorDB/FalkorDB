/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include <math.h>

//------------------------------------------------------------------------------
// Standard deviation
//------------------------------------------------------------------------------

typedef struct {
	double *values;
	double total;
} _agg_StDevCtx;

AggregateResult AGG_STDEV(SIValue *argv, int argc, void *private_data) {
	AggregateCtx *ctx = private_data;
	_agg_StDevCtx *stdev_ctx = ctx->private_data;

	// on the first invocation, initialize the context
	if(stdev_ctx->values == NULL) {
		stdev_ctx->total = 0;
		stdev_ctx->values = array_new(double, 1024);
	}

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	double n;
	SIValue_ToDouble(&v, &n);
	array_append(stdev_ctx->values, n);
	stdev_ctx->total += n;

	return AGGREGATE_OK;
}

void StDevGenericFinalize(AggregateCtx *ctx, int is_sampled) {
	_agg_StDevCtx *stdev_ctx = ctx->private_data;

	if(stdev_ctx == NULL) return;

	uint count = array_len(stdev_ctx->values);
	if(count - is_sampled == 0) {
		Aggregate_SetResult(ctx, SI_DoubleVal(0));
		return;
	}

	double mean = stdev_ctx->total / count;
	long double sum = 0;
	for(int i = 0; i < count; i ++) {
		sum += (long double)(stdev_ctx->values[i] - mean) * (stdev_ctx->values[i] + mean);
	}
	// is_sampled will be equal to 1 in the Stdev case and 0 in the StdevP case
	double variance = sum / (count - is_sampled);
	double stdev = sqrt(variance);

	Aggregate_SetResult(ctx, SI_DoubleVal(stdev));
}

void StDevFinalize(void *ctx_ptr) {
	StDevGenericFinalize(ctx_ptr, 1);
}

void StDevPFinalize(void *ctx_ptr) {
	StDevGenericFinalize(ctx_ptr, 0);
}

void StDev_Free(void *pdata) {
	ASSERT(pdata != NULL);

	_agg_StDevCtx *stdev_ctx = pdata;
	if(stdev_ctx->values != NULL) {
		array_free(stdev_ctx->values);
	}
	rm_free(pdata);
}

AggregateCtx *STD_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_DoubleVal(0);  // STD default value is 0

	// initialize private data
	_agg_StDevCtx *pdata = rm_calloc(1, sizeof(_agg_StDevCtx));
	pdata->values = NULL;
	pdata->total = -1;

	ctx->private_data = pdata;

	return ctx;
}

void Register_STD(void) {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_AggFuncDescNew("stDev", AGG_STDEV, 1, 1, types, ret_type,
			StDev_Free, StDevFinalize, STD_PrivateData);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_AggFuncDescNew("stDevP", AGG_STDEV, 1, 1, types, ret_type,
			StDev_Free, StDevPFinalize, STD_PrivateData);
	AR_RegFunc(func_desc);
}

