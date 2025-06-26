/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../errors/errors.h"
#include <math.h>
#include <stdlib.h>

static inline int _cmp
(
	const double *a,
	const double *b
) {
	return (*a > *b) - (*a < *b);
}

//------------------------------------------------------------------------------
// Precentile
//------------------------------------------------------------------------------

typedef struct {
	double percentile;
	double *values;
} _agg_PercCtx;

// this function is agnostic as to percentile method
AggregateResult AGG_PERC(SIValue *argv, int argc, void *private_data) {
	AggregateCtx *ctx = private_data;
	_agg_PercCtx *perc_ctx = ctx->private_data;

	// on the first invocation, initialize the context
	if(perc_ctx->values == NULL) {
		// the second argument is the requested percentile, which we only
		// need to apply on the first function invocation
		SIValue_ToDouble(&argv[1], &perc_ctx->percentile);
		if(perc_ctx->percentile < 0 || perc_ctx->percentile > 1) {
			ErrorCtx_SetError(EMSG_PREC_INPUT_RANGE, perc_ctx->percentile);
		}
		perc_ctx->values = array_new(double, 1024);
	}

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	double n;
	SIValue_ToDouble(&v, &n);
	array_append(perc_ctx->values, n);

	return AGGREGATE_OK;
}

void PercDiscFinalize(void *ctx_ptr) {
	AggregateCtx *ctx = ctx_ptr;
	_agg_PercCtx *perc_ctx = ctx->private_data;
	if(perc_ctx == NULL) return;

	uint count = array_len(perc_ctx->values);
	if(count == 0) {
		Aggregate_SetResult(ctx, SI_NullVal());
	} else {
		qsort(perc_ctx->values, count, sizeof(double),
				(int(*)(const void*, const void*))_cmp);

		// if perc_ctx->percentile == 0
		// employing this formula would give an index of -1
		int idx = perc_ctx->percentile > 0 ? ceil(perc_ctx->percentile * count) - 1 : 0;
		double n = perc_ctx->values[idx];
		Aggregate_SetResult(ctx, SI_DoubleVal(n));
	}
}

void PercContFinalize(void *ctx_ptr) {
	AggregateCtx *ctx = ctx_ptr;
	_agg_PercCtx *perc_ctx = ctx->private_data;
	if(perc_ctx == NULL) return;

	uint count = array_len(perc_ctx->values);
	if(count == 0) {
		Aggregate_SetResult(ctx, SI_NullVal());
	} else {
		qsort(perc_ctx->values, count, sizeof(double),
				(int(*)(const void*, const void*))_cmp);

		if(perc_ctx->percentile == 1.0 || count == 1) {
			Aggregate_SetResult(ctx, SI_DoubleVal(perc_ctx->values[count - 1]));
		}

		double int_val, fraction_val;
		double float_idx = perc_ctx->percentile * (count - 1);
		// Split the temp value into its integer and fractional values
		fraction_val = modf(float_idx, &int_val);
		int index = int_val; // Casting the integral part of the value to an int for convenience

		if(!fraction_val) {
			// A valid index was requested, so we can directly return a value
			Aggregate_SetResult(ctx, SI_DoubleVal(perc_ctx->values[index]));
			return;
		}

		double lhs, rhs;
		lhs = perc_ctx->values[index] * (1 - fraction_val);
		rhs = perc_ctx->values[index + 1] * fraction_val;

		Aggregate_SetResult(ctx, SI_DoubleVal(lhs + rhs));
	}
}

void Percentile_Free(void *pdata) {
	ASSERT(pdata != NULL);

	_agg_PercCtx *ctx = pdata;
	if(ctx->values != NULL) {
		array_free(ctx->values);
	}
	rm_free(ctx);
}

AggregateCtx *Precentile_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_NullVal();  // precentile default value is NULL

	// initialize private data
	_agg_PercCtx *pdata = rm_calloc(1, sizeof(_agg_PercCtx));
	pdata->percentile= -1; // invalid precentile value
	pdata->values = NULL;

	ctx->private_data = pdata;

	return ctx;
}

void Register_PRECENTILE(void) {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 3);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_AggFuncDescNew("percentileDisc", AGG_PERC, 2, 2, types, ret_type,
			Percentile_Free, PercDiscFinalize, Precentile_PrivateData);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_AggFuncDescNew("percentileCont", AGG_PERC, 2, 2, types, ret_type,
			Percentile_Free, PercContFinalize, Precentile_PrivateData);
	AR_RegFunc(func_desc);
}

