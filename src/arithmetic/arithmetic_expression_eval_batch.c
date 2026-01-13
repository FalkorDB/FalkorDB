/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "rax.h"
#include "value.h"
#include "ast/ast.h"
#include "util/arr.h"
#include "./func_desc.h"
#include "../query_ctx.h"
#include "util/rmalloc.h"
#include "errors/errors.h"
#include "./arithmetic_expression.h"
#include "./arithmetic_expression_eval.h"

// forward declarations

static AR_EXP_Result _AR_EXP_Evaluate_Batch
(
	SIValue *restrict res,        // [input/output] results
	SIType *type,                 // type of results
	SIAllocation *alloc,          // allocation type of results
	AR_ExpNode *restrict root,    // root of expression tree to evaluate
	const Record *restrict recs,  // batch of records
	size_t n,                     // number of records
	uint step                     // step within 'res'
);

static bool _AR_EXP_UpdateEntityIdx
(
	AR_OperandNode *node,
	const Record r
) {
	if(!r) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_MISSING_RECORD, node->variadic.entity_alias);
		return false;
	}
	int entry_alias_idx = Record_GetEntryIdx(r, node->variadic.entity_alias,
			strlen(node->variadic.entity_alias));
	if(entry_alias_idx == INVALID_INDEX) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_MISSING_VALUE, node->variadic.entity_alias);
		return false;
	} else {
		node->variadic.entity_alias_idx = entry_alias_idx;
		return true;
	}
}

// evaluating an expression tree constructs an array of SIValues
// free all of these values
// in case an intermediate node on the tree caused a heap allocation
// for example, in the expression:
// a.first_name + toUpper(a.last_name)
// the result of toUpper() is allocated within this tree
// and will leak if not freed here
static void _AR_EXP_FreeArgsRows
(
	SIValue *args,          // evaluated arguments to free
	SIAllocation *allocs,   // args allocations
	uint argc,              // number of arguments
	uint chunk_size,        // size of chunk
	uint16_t constant_mask  // constants mask
) {
	// free only non cached, heap allocated values
	for (int i = 0; i < chunk_size; i++) {
		if (!(constant_mask & 1ULL << i) && allocs[i] == M_SELF) {
			for (int j = i; j < argc; j+= chunk_size) {
				SIValue_Free (args[j]) ;
			}
		}
	}
}

static void _AR_EXP_FreeArgsColumn
(
	SIValue *args,  //
	SIAllocation *allocs,
	uint16_t n_cols,
	uint16_t n_rows,
	uint16_t constant_mask
) {
	for (uint16_t i = 0 ; i < n_cols ; i++) {
		if (!(constant_mask & 1ULL << i) && allocs[i] == M_SELF) {
			uint16_t offset = n_rows * i ;
			for (uint16_t j = 0 ; j < n_rows ; j++) {
				SIValue_Free (args[offset + j]) ;
			}
		}
	}
}

// validate function arguments
static bool _AR_EXP_ValidateInvocation_Batch
(
	AR_FuncDesc *restrict fdesc,  // function descriptor
	SIType *restrict types,       // argument types
	size_t argc                   // number of arguments
) {
	SIType actual_type ;
	SIType expected_type = T_NULL ;
	SIType expected_types[argc] ;
	uint expected_types_count = array_len (fdesc->types) ;

	//--------------------------------------------------------------------------
	// initialize expected arguments types
	//--------------------------------------------------------------------------

	uint i = 0 ;
	for (; i < expected_types_count; i++) {
		expected_type = fdesc->types[i] ;
		expected_types[i] = expected_type ;
	}

	// a function that accepts a variable number of arguments
	// the last specified type in fdesc->types is repeatable
	for(; i < argc; i++) {
		expected_types[i] = expected_type;
	}

	//--------------------------------------------------------------------------
	// validate arguments types
	//--------------------------------------------------------------------------

	bool valid = true ;
	for (int i = 0; i < argc; i++) {
		SIType expected = expected_types[i] ;
		SIType actual   = types[i] ; 
		if (unlikely ((actual & expected) != actual)) {
			Error_SITypeMismatch (actual, expected_type) ;
			return false ;
		}
	}

	return true ;
}

static AR_EXP_Result _AR_EXP_EvaluateFunctionCall_Batch
(
	SIValue *restrict res,        // [input/output] results
	SIType *type,                 // type of results
	SIAllocation *alloc,          // allocation type of results
	AR_ExpNode *restrict node,    // function node
	const Record *restrict recs,  // records
	size_t n,                     // number of batches
	uint step                     // step within 'res'
) {
	ASSERT (res  != NULL) ;
	ASSERT (node != NULL) ;

	AR_EXP_Result ret = EVAL_OK ;

	uint child_count = NODE_CHILD_COUNT (node) ;
	ASSERT (n <= 64) ;
	uint argc = child_count * n ;

	SIType types[child_count] ;         // value type(s) of sub exps
	SIAllocation allocs[child_count] ;  // value allocation type(s) of sub exps

	bool param_found    = false ;
	bool err_eval_child = false ;
	bool vectorized     = AR_HasBatchVersion (node->op.f) ;
	uint _step          = vectorized ? 1 : child_count ;  // row / col layout

	//--------------------------------------------------------------------------
	// evaluate each child before evaluating current node
	//--------------------------------------------------------------------------

	for (int child_idx = 0 ; child_idx < child_count ; child_idx++) {
		int arg_offset = (vectorized) ?
			child_idx * 64 * vectorized :
			child_idx ;

		// argument already cached
		if (node->op.constant_mask & 1ULL << child_idx) {
			types[child_idx] = SI_TYPE (node->op.args[arg_offset]) ;
			continue ;
		}

		//----------------------------------------------------------------------
		// evaluate child
		//----------------------------------------------------------------------

		AR_ExpNode *child = NODE_CHILD (node, child_idx) ;

		ret = _AR_EXP_Evaluate_Batch (node->op.args + arg_offset,
				types + child_idx, allocs + child_idx, child, recs, n,
				_step) ;

		err_eval_child |= (ret == EVAL_ERR) ;
		param_found    |= (ret == EVAL_FOUND_PARAM) ;
	}

	// error while evaluating children
	if (err_eval_child) {
		ret = EVAL_ERR ;
		goto cleanup ;
	}

	if (param_found) {
		ret = EVAL_FOUND_PARAM;
	}

	// validate before evaluation
	if (!_AR_EXP_ValidateInvocation_Batch (node->op.f, types, child_count)) {
		// the expression tree failed its validations and set an error message
		ret = EVAL_ERR ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// execute function
	//--------------------------------------------------------------------------

	SIType _type = 0 ;
	SIAllocation _alloc = M_NONE ;

	if (vectorized) 
	{
		// set columns
		SIValue *_args [child_count];
		for (uint i = 0; i < child_count; i++) {
			_args[i] = &(node->op.args[i * 64]) ;
		}

		// call vectorized operator
		node->op.f->batch_func (res, _args, types, n, node->op.private_data) ;
	}
	else
	{
		for (uint i = 0; i < n; i++) {
			SIValue *row_args = node->op.args + (i * child_count);
			SIValue v = node->op.f->func (row_args, child_count,
					node->op.private_data) ;

			SIValue_Persist (&v) ;
			res[i * step] = v ;
		}
	}

	// compute results types and allocation types
	for (uint i = 0; i < n; i++) {
		_type  |= SI_TYPE (res[i]) ;
		_alloc |= SI_ALLOCATION (&res[i]) ;

		ASSERT (node->op.f->aggregate ||
				SI_TYPE (res[i]) & AR_FuncDesc_RetType (node->op.f)) ;

		if (unlikely (SIValue_IsNull (res[i]) &&
					  ErrorCtx_EncounteredError ())) {
			// an error was encountered while evaluating this function
			// and has already been set in the QueryCtx
			// exit with an error
			ret = EVAL_ERR ;
			break ;
		}
	}

	// report results types & allocations
	if (type != NULL) {
		*type = _type ;
	}

	if (alloc != NULL) {
		*alloc = _alloc ;
	}

cleanup:
	if (vectorized) {
		_AR_EXP_FreeArgsColumn (node->op.args, allocs, child_count, n,
				node->op.constant_mask) ;
	} else {
		_AR_EXP_FreeArgsRows (node->op.args, allocs, argc, child_count,
				node->op.constant_mask) ;
	}

	return ret ;
}

static inline AR_EXP_Result _AR_EXP_Evaluate_Const_Batch
(
	SIValue *restrict res,            // [input/output] results
	SIType *type,                     // [ignore] type of results
	SIAllocation *alloc,              // [ignore] allocation type of results
	const AR_ExpNode *restrict root,  // root of expression tree to evaluate
	size_t n,                         // number of records
	uint step                         // step within 'res'
) {
	// the value is constant or has been computed elsewhere
	// share with caller
	SIValue c = SI_ShareValue (root->operand.constant) ; 

	// copy const into each result entry
	for (uint i = 0; i < n; i++) {
		res[i * step] = c ;
	}

	return EVAL_OK ;
}

static AR_EXP_Result _AR_EXP_EvaluateVariadic_Batch
(
	SIValue *restrict res,        // [input/output] results
	SIType *type,                 // type of results
	SIAllocation *alloc,          // allocation type of results
	AR_ExpNode *restrict node,    // variadic node
	const Record *restrict recs,  // records
	size_t n,                     // number of records
	uint step                     // step within 'res'
) {
	// make sure entity record index is known
	if (node->operand.variadic.entity_alias_idx == IDENTIFIER_NOT_FOUND) {
		if (!_AR_EXP_UpdateEntityIdx (&node->operand, recs[0])) {
			return EVAL_ERR ;
		}
	}

	SIType _type = 0 ;
	int aliasIdx = node->operand.variadic.entity_alias_idx ;

	for (uint i = 0; i < n; i++) {
		// the value was not created here; share with the caller
		SIValue v = SI_ShareValue (Record_Get (recs[i], aliasIdx)) ;
		res[i * step] = v ;
		_type |= SI_TYPE (v) ;
	}

	if (type != NULL) {
		*type = _type ;
	}

	if (alloc != NULL) {
		// half true, but doesn't matter as long as its not M_SELF
		*alloc = M_VOLATILE ;
	}

	return EVAL_OK ;
}

static AR_EXP_Result _AR_EXP_EvaluateParam_Batch
(
	SIValue *restrict res,      // [input/output] results
	SIType *type,               // type of results
	SIAllocation *alloc,        // allocation type of results
	AR_ExpNode *restrict root,  // root of expression tree to evaluate
	size_t n,                   // number of records
	uint step                   // step within 'res'
) {
	SIValue *param ;
	dict *params = QueryCtx_GetParams();

	if (params) {
		param = HashTableFetchValue (params,
				(unsigned char *)root->operand.param_name) ;
	}

	if (params == NULL || param == NULL) {
		// set the query-level error
		ErrorCtx_SetError (EMSG_MISSING_PARAMETERS) ;
		return EVAL_ERR ;
	}

	//--------------------------------------------------------------------------
	// in place replacement
	//--------------------------------------------------------------------------

	SIType _type = 0 ;
	root->operand.type     = AR_EXP_CONSTANT ;
	root->operand.constant = SI_ShareValue (*param) ;

	// copy const into each result entry
	for(uint i = 0; i < n; i++) {
		SIValue v = root->operand.constant ;
		_type |= SI_TYPE (v) ;
		res[i * step] = v ;
	}

	if (type != NULL) {
		*type = _type ;
	}

	if (alloc != NULL) {
		// half true, but doesn't matter as long as its not M_SELF
		*alloc = M_VOLATILE ;
	}

	return EVAL_FOUND_PARAM ;
}

static AR_EXP_Result _AR_EXP_EvaluateBorrowRecord_Batch
(
	SIValue *restrict res,        // [input/output] results
	SIType *type,                 // type of results
	SIAllocation *alloc,          // allocation type of results
	const Record *restrict recs,  // records
	size_t n,                     // number of records
	uint step                     // step within 'res'
) {
	for (uint i = 0; i < n; i++) {
		// wrap the current Record in an SI pointer
		res[i * step] = SI_PtrVal (recs[i]) ;
	}

	if (alloc != NULL) {
		*alloc = M_NONE ;
	}

	if (type != NULL) {
		*type = T_PTR ;
	}

	return EVAL_OK ;
}

// evaluate an expression tree,
// placing the calculated values in 'res'
// returning whether an error occurred during evaluation
static AR_EXP_Result _AR_EXP_Evaluate_Batch
(
	SIValue *restrict res,        // [input/output] results
	SIType *type,                 // type of results
	SIAllocation *alloc,          // allocation type of results
	AR_ExpNode *restrict root,    // root of expression tree to evaluate
	const Record *restrict recs,  // batch of records
	size_t n,                     // number of records
	uint step                     // step within 'res'
) {
	switch (root->type) {
		case AR_EXP_OP :
			return _AR_EXP_EvaluateFunctionCall_Batch (res, type, alloc, root,
					recs, n, step) ;

		case AR_EXP_OPERAND:
			switch (root->operand.type) {
				case AR_EXP_CONSTANT :
					return _AR_EXP_Evaluate_Const_Batch (res, type, alloc, root,
							n, step) ;

				case AR_EXP_VARIADIC :
					return _AR_EXP_EvaluateVariadic_Batch (res, type, alloc,
							root, recs, n, step) ;

				case AR_EXP_PARAM :
					return _AR_EXP_EvaluateParam_Batch (res, type, alloc, root,
							n, step) ;

				case AR_EXP_BORROW_RECORD :
					return _AR_EXP_EvaluateBorrowRecord_Batch (res, type, alloc,
							recs, n, step) ;

				default :
					ASSERT (false && "Invalid expression type") ;
			}

		default :
			ASSERT (false && "Unknown expression type") ;
	}

	// we shouldn't get here
	return EVAL_ERR ;
}

// evaluate arithmetic exception tree for a number of records
// this function raise exception
// populates res with each evaluated value
void AR_EXP_Evaluate_Batch
(
	SIValue *restrict res,        // [input/output] results
	AR_ExpNode *restrict root,    // root of expression tree to evaluate
	const Record *restrict recs,  // batch of records
	size_t n                      // number of records
) {
	// validations
	ASSERT (n > 0);
	ASSERT (res  != NULL) ;
	ASSERT (root != NULL) ;
	ASSERT (recs != NULL) ;

	AR_EXP_Result ret =
		_AR_EXP_Evaluate_Batch (res, NULL, NULL, root, recs, n, 1);

	if (unlikely (ret == EVAL_ERR)) {
		// jumps if longjump is set
		ErrorCtx_RaiseRuntimeException (NULL) ;

		// otherwise return NULLs
		// the query-level error will be emitted after cleanup

		// Nullify results
		for (uint i = 0; i < n; i++) {
			res[i] = SI_NullVal ();
		}

		return ;
	}

	// at least one param node was encountered during evaluation,
	// tree should be parameters free, try reducing the tree
	if (ret == EVAL_FOUND_PARAM) {
		AR_EXP_ReduceToScalar (root, true, NULL) ;
	}
}

