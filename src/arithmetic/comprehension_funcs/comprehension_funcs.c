/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "comprehension_funcs.h"
#include "RG.h"
#include "../../value.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"
#include "../../datatypes/array.h"
#include "../../util/rax_extensions.h"
#include "../../execution_plan/record.h"

// Routine for freeing a comprehension function's private data.
void ListComprehension_Free(void *ctx_ptr) {
	ListComprehensionCtx *ctx = ctx_ptr;

	// if this comprehension has a filter tree, free it
	if(ctx->ft) FilterTree_Free(ctx->ft);

	// if this comprehension has an eval routine, free it
	if(ctx->eval_exp) AR_EXP_Free(ctx->eval_exp);

	if(ctx->local_record) {
		rax *mapping = ctx->local_record->mapping;
		Record_Free(ctx->local_record);
		raxFree(mapping);
	}

	rm_free(ctx);
}

// Routine for cloning a comprehension function's private data.
void *ListComprehension_Clone(void *orig) {
	if(orig == NULL) return NULL;

	ListComprehensionCtx *ctx = orig;
	// Allocate space for the clone.
	ListComprehensionCtx *ctx_clone = rm_malloc(sizeof(ListComprehensionCtx));

	// Clone the variadic node.
	ctx_clone->variable_str = ctx->variable_str;
	ctx_clone->variable_idx = ctx->variable_idx;
	ctx_clone->local_record = NULL;

	// Clone the predicate filter tree, if present.
	ctx_clone->ft = FilterTree_Clone(ctx->ft);
	// Clone the eval routine, if present.
	ctx_clone->eval_exp = AR_EXP_Clone(ctx->eval_exp);

	return ctx_clone;
}

static void _PopulateComprehensionCtx
(
	ListComprehensionCtx *ctx,
	Record outer_record
) {
	rax *local_record_map = raxClone(outer_record->mapping);
	intptr_t id = raxSize(local_record_map);
	raxTryInsert(local_record_map, (unsigned char *)ctx->variable_str,
						  strlen(ctx->variable_str), (void *)id, NULL);
	ctx->local_record = Record_New(local_record_map);

	// this could just be assigned to 'id'
	// but for safety we'll use a Record lookup
	ctx->variable_idx = Record_GetEntryIdx(ctx->local_record, ctx->variable_str, strlen(ctx->variable_str));
	ASSERT(ctx->variable_idx != INVALID_INDEX);
}

SIValue AR_ANY(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	// ANY comprehensions are invoked with three children:
	// The list to iterate over.
	// The current Record.
	// The function context.
	SIValue list = argv[0];
	Record outer_record = argv[1].ptrval;
	ListComprehensionCtx *ctx = private_data;

	// On the first invocation, build the local Record.
	if(ctx->local_record == NULL) _PopulateComprehensionCtx(ctx, outer_record);
	Record r = ctx->local_record;

	// Populate the local Record with the contents of the outer Record.
	Record_Clone(outer_record, r);

	bool is_null = false;
	uint len = SIArray_Length(list);
	for(uint i = 0; i < len; i++) {
		// Retrieve the current element.
		SIValue current_elem = SIArray_Get(list, i);
		// Add the current element to the record at its allocated position.
		Record_AddScalar(r, ctx->variable_idx, current_elem);

		// if any element in an ANY function passes the predicate, return true.
		FT_Result res = FilterTree_applyFilters(ctx->ft, r);
		if(res == FILTER_PASS) return SI_BoolVal(true);
		is_null |= (res == FILTER_NULL);
	}

	if(is_null) {
		return SI_NullVal();
	}
	// No element passed, return false.
	return SI_BoolVal(false);
}

SIValue AR_ALL(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	// ALL comprehensions are invoked with three children:
	// The list to iterate over.
	// The current Record.
	// The function context.
	SIValue list = argv[0];
	Record outer_record = argv[1].ptrval;
	ListComprehensionCtx *ctx = private_data;

	// On the first invocation, build the local Record.
	if(ctx->local_record == NULL) _PopulateComprehensionCtx(ctx, outer_record);
	Record r = ctx->local_record;

	// Populate the local Record with the contents of the outer Record.
	Record_Clone(outer_record, r);

	bool is_null = false;
	uint len = SIArray_Length(list);
	for(uint i = 0; i < len; i++) {
		// Retrieve the current element.
		SIValue current_elem = SIArray_Get(list, i);
		// Add the current element to the record at its allocated position.
		Record_AddScalar(r, ctx->variable_idx, current_elem);

		// If any element in an ALL function does not pass the predicate, return false.
		FT_Result res = FilterTree_applyFilters(ctx->ft, r);
		if(res == FILTER_FAIL) return SI_BoolVal(false);
		is_null |= (res == FILTER_NULL);
	}

	if(is_null) {
		return SI_NullVal();
	}
	// All elements passed, return true.
	return SI_BoolVal(true);
}

SIValue AR_SINGLE(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	// ALL comprehensions are invoked with three children:
	// The list to iterate over.
	// The current Record.
	// The function context.
	SIValue list = argv[0];
	Record outer_record = argv[1].ptrval;
	ListComprehensionCtx *ctx = private_data;

	// On the first invocation, build the local Record.
	if(ctx->local_record == NULL) _PopulateComprehensionCtx(ctx, outer_record);
	Record r = ctx->local_record;

	// Populate the local Record with the contents of the outer Record.
	Record_Clone(outer_record, r);

	bool single = false;
	bool is_null = false;
	uint len = SIArray_Length(list);
	for(uint i = 0; i < len; i++) {
		// Retrieve the current element.
		SIValue current_elem = SIArray_Get(list, i);
		// Add the current element to the record at its allocated position.
		Record_AddScalar(r, ctx->variable_idx, current_elem);

		// If more then 1 element in a SINGLE function pass the predicate, return false.
		FT_Result res = FilterTree_applyFilters(ctx->ft, r);
		if(res == FILTER_PASS) {
			if(single) return SI_BoolVal(false);
			else single = true;
		}
		is_null |= (res == FILTER_NULL);
	}

	if(is_null) {
		return SI_NullVal();
	}
	// If only 1 element in a SINGLE function pass the predicate, return true, otherwise false.
	return SI_BoolVal(single);
}

SIValue AR_NONE(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	// ALL comprehensions are invoked with three children:
	// The list to iterate over.
	// The current Record.
	// The function context.
	SIValue list = argv[0];
	Record outer_record = argv[1].ptrval;
	ListComprehensionCtx *ctx = private_data;

	// On the first invocation, build the local Record.
	if(ctx->local_record == NULL) _PopulateComprehensionCtx(ctx, outer_record);
	Record r = ctx->local_record;

	// Populate the local Record with the contents of the outer Record.
	Record_Clone(outer_record, r);

	bool is_null = false;
	uint len = SIArray_Length(list);
	for(uint i = 0; i < len; i++) {
		// Retrieve the current element.
		SIValue current_elem = SIArray_Get(list, i);
		// Add the current element to the record at its allocated position.
		Record_AddScalar(r, ctx->variable_idx, current_elem);

		// If any element in an NONE function pass the predicate, return false.
		FT_Result res = FilterTree_applyFilters(ctx->ft, r);
		if(res == FILTER_PASS) return SI_BoolVal(false);
		is_null |= (res == FILTER_NULL);
	}

	if(is_null) {
		return SI_NullVal();
	}
	// No elements passed, return true.
	return SI_BoolVal(true);
}


SIValue AR_LIST_COMPREHENSION
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();

	// list comprehensions are invoked with three children:
	// the list to iterate over
	// the current Record
	// the function context
	SIValue list = argv[0];
	Record outer_record = argv[1].ptrval;
	ListComprehensionCtx *ctx = private_data;

	// on the first invocation, build the local Record
	if(ctx->local_record == NULL) _PopulateComprehensionCtx(ctx, outer_record);
	Record r = ctx->local_record;

	// populate the local Record with the contents of the outer Record
	Record_Clone(outer_record, r);
	// instantiate the array to be returned
	SIValue retval = SI_Array(0);

	uint len = SIArray_Length(list);
	for(uint i = 0; i < len; i++) {
		// retrieve the current element
		SIValue current_elem = SIArray_Get(list, i);
		// add the current element to the record at its allocated position
		Record_AddScalar(r, ctx->variable_idx, current_elem);

		// if the comprehension has a filter tree
		// run the current element through it
		// if it does not pass, skip this element
		if(ctx->ft && FilterTree_applyFilters(ctx->ft, r) != FILTER_PASS) {
			continue;
		}

		if(ctx->eval_exp) {
			// compute the current element to append to the return list
			SIValue newval = AR_EXP_Evaluate_NoThrow(ctx->eval_exp, r);
			SIArray_Append(&retval, newval);
			SIValue_Free(newval);
		} else {
			// if the comprehension has no eval routine
			// add each element unmodified
			SIArray_Append(&retval, current_elem);
		}
	}

	return retval;
}

void Register_ComprehensionFuncs() {
	SIType *types;
	SIType ret_type = T_BOOL | T_NULL;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 3);
	array_append(types, T_ARRAY | T_NULL);
	array_append(types, T_PTR);
	func_desc = AR_FuncDescNew("any", AR_ANY, 2, 2, types, ret_type, true, true,
			true);
	AR_SetPrivateDataRoutines(func_desc, ListComprehension_Free, ListComprehension_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_ARRAY | T_NULL);
	array_append(types, T_PTR);
	func_desc = AR_FuncDescNew("all", AR_ALL, 2, 2, types, ret_type, true, true,
			true);
	AR_SetPrivateDataRoutines(func_desc, ListComprehension_Free, ListComprehension_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_ARRAY | T_NULL);
	array_append(types, T_PTR);
	func_desc = AR_FuncDescNew("single", AR_SINGLE, 2, 2, types, ret_type, true,
			true, true);
	AR_SetPrivateDataRoutines(func_desc, ListComprehension_Free, ListComprehension_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_ARRAY | T_NULL);
	array_append(types, T_PTR);
	func_desc = AR_FuncDescNew("none", AR_NONE, 2, 2, types, ret_type, true,
			true, true);
	AR_SetPrivateDataRoutines(func_desc, ListComprehension_Free, ListComprehension_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_ARRAY | T_NULL);
	array_append(types, T_PTR);
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("list_comprehension", AR_LIST_COMPREHENSION, 2,
			2, types, ret_type, true, true, true);
	AR_SetPrivateDataRoutines(func_desc, ListComprehension_Free, ListComprehension_Clone);
	AR_RegFunc(func_desc);
}
