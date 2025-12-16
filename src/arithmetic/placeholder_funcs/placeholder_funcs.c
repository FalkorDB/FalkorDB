/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "placeholder_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"

SIValue AR_PATH_FILTER(SIValue *argv, int argc, void *private_data) {
	ASSERT(false);
	return SI_NullVal();
}

void Register_PlaceholderFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 0);
	ret_type = T_NULL;
	func_desc = AR_FuncDescNew("path_filter", AR_PATH_FILTER, 0, 0, types,
			ret_type, true, false, true);
	AR_RegFunc(func_desc);
}

