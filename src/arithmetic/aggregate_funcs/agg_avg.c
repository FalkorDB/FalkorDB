/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "agg_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include <math.h>
#include <float.h>

//------------------------------------------------------------------------------
// Avg
//------------------------------------------------------------------------------

// avarage context
typedef struct {
	size_t count;       // number of elements summed
	long double total;  // sum of all elements
	bool overflow;      // track numeric overflow
} AvgCtx;

// return true if adding a and b will overflow
// values have the same MSB, adding will enlarge the total
#define ABOUT_TO_OVERFLOW(a, b) (signbit((a)) == signbit((b)) && \
	   (fabsl((a)) > (DBL_MAX - fabsl((b)))))

// avarage "step" function expects 2 arguments:
// 1. aggregation context
// 2. value to aggregate
AggregateResult AGG_AVG
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	SIValue val = argv[0];
	AggregateCtx *ctx = private_data;
	AvgCtx *avg_ctx = ctx->private_data;

	// On the first invocation, initialize the context.
	if(ctx->private_data == NULL) {
		avg_ctx = ctx->private_data = rm_calloc(1, sizeof(AvgCtx));
	}

	// check input
	if(SI_TYPE(val) == T_NULL) return AGGREGATE_OK;

	long double v = SI_GET_NUMERIC(val);

	avg_ctx->count++; // increment count

	//--------------------------------------------------------------------------
	// check for overflow
	//--------------------------------------------------------------------------

	// if we've already overflowed or adding the current value
	// will cause us to overflow, use the incremental averaging algorithm
	if(avg_ctx->overflow || ABOUT_TO_OVERFLOW(avg_ctx->total, v)) {
		// divide the total by the new count
		long double total = avg_ctx->total /= (long double) avg_ctx->count;
		// if this is not the first call using the incremental algorithm,
		// multiply the total by the previous count
		if(avg_ctx->overflow) total *= (long double)(avg_ctx->count - 1);
		// add v/count to total
		total += (v / (long double)avg_ctx->count);
		avg_ctx->total = total;
		avg_ctx->overflow = true;
	} else {
		// no overflow
		avg_ctx->total += v;
	}

	return AGGREGATE_OK;
}

// finalize aggregation
void Avg_Finalize
(
	void *ctx_ptr
) {
	ASSERT(ctx_ptr != NULL);

	AggregateCtx *ctx = (AggregateCtx*)ctx_ptr;

	AvgCtx *avg_ctx = ctx->private_data;
	if(avg_ctx == NULL) return;

	if(avg_ctx->count > 0) {
		if(avg_ctx->overflow) {
			// used incremental algorithm due to overflow, 'total' is the average
			ctx->result = SI_DoubleVal(avg_ctx->total);
		} else {
			// used traditional algorithm, divide total by count
			ctx->result = SI_DoubleVal(avg_ctx->total / avg_ctx->count);
		}
	}
}

AggregateCtx *Avg_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_NullVal();  // avg default value is NULL
	ctx->private_data = rm_calloc(1, sizeof(AvgCtx));

	return ctx;
}

void Register_AVG(void) {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_INT64 | T_DOUBLE);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_AggFuncDescNew("avg", AGG_AVG, 1, 1, types, ret_type,
			rm_free, Avg_Finalize, Avg_PrivateData);

	AR_RegFunc(func_desc);
}

