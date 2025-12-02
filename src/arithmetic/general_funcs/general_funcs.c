/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "general_funcs.h"
#include "../../value.h"
#include "../func_desc.h"
#include "../../util/arr.h"

typedef struct {
	SIValue value;  // the value to store
} AR_PrevPrivateData;

void AR_PrevPrivateData_Free
(
    void *ctx_ptr
) {
    ASSERT(ctx_ptr != NULL);

	AR_PrevPrivateData *ctx = ctx_ptr;

	SIValue_Free(ctx->value);
	rm_free(ctx);
}

void *AR_PrevPrivateData_Clone
(
    void *orig
) {
	AR_PrevPrivateData *ctx = orig;
	AR_PrevPrivateData *ctx_clone = rm_malloc(sizeof(AR_PrevPrivateData));

	if(ctx == NULL) {
		ctx_clone->value = SI_NullVal();
	} else {
		ctx_clone->value = SI_CloneValue(ctx->value);
	}

	return ctx_clone;
}

// return the previous value of the first argument
// this function is used to store the previous value of an expression
// ex: UNWIND range(1, 5) AS x RETURN x, prev(x)
// the result will be:
// 1, NULL
// 2, 1
// 3, 2
// 4, 3
// 5, 4
SIValue AR_PREV
(
    SIValue *argv,
    int argc,
    void *private_data
) {
	AR_PrevPrivateData *pdata = (AR_PrevPrivateData *)private_data;
	SIValue value = pdata->value;
	pdata->value = SI_CloneValue(argv[0]);
	return value;
}

void Register_GeneralFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | SI_ALL);
	ret_type = SI_ALL;
	func_desc = AR_FuncDescNew("prev", AR_PREV, 1, 1, types, ret_type, false,
			false, true);
	AR_SetPrivateDataRoutines(func_desc, AR_PrevPrivateData_Free, AR_PrevPrivateData_Clone);
	AR_RegFunc(func_desc);
}

