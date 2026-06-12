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
// Sum
//------------------------------------------------------------------------------

AggregateResult AGG_SUM(SIValue *argv, int argc, void *private_data) {
	AggregateCtx *ctx = private_data;

	SIValue v = argv[0];
	if(SI_TYPE(v) == T_NULL) return AGGREGATE_OK;

	// Update the total.
	if(SI_TYPE(v) != T_NULL) ctx->result.doubleval += SI_GET_NUMERIC(v);

	return AGGREGATE_OK;
}

AggregateCtx *SUM_PrivateData(void)
{
	AggregateCtx *ctx = rm_malloc(sizeof(AggregateCtx));

	ctx->result = SI_DoubleVal(0);  // SUM default value is 0
	ctx->private_data = NULL;

	return ctx;
}

