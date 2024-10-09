/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "time_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../datatypes/temporal_value.h"

/* returns a timestamp - millis from epoch */
SIValue AR_TIMESTAMP(SIValue *argv, int argc, void *private_data) {
	return SI_LongVal(TemporalValue_NewTimestamp());
}

void Register_TimeFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 0);
	ret_type = T_INT64;
	func_desc = AR_FuncDescNew("timestamp", AR_TIMESTAMP, 0, 0, types, ret_type, false, false, "timestamp()");
	AR_RegFunc(func_desc);
}

