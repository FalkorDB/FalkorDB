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
static inline void _AR_EXP_FreeArgsArray
(
	SIValue *args,  // evaluated arguments to free
	uint argc,      // number of args in each batch
	uint step,      // batch size
	uint n          // args array length
) {
	for (int i = 0; i < n; i += step) {
		for (int j = 0; j < argc; j++) {
			SIValue_Free (args[i+j]) ;
		}
	}

	// large arrays are heap-allocated, so here is where we free it
	if (n > MAX_ARRAY_SIZE_ON_STACK) {
		rm_free (args) ;
	}
}

// validate function arguments
static bool _AR_EXP_ValidateInvocation_Batch
(
	AR_FuncDesc *restrict fdesc,  // function descriptor
	SIValue *restrict argv,       // arguments
	size_t argc,                  // number of arguments
	size_t n                      // batch size
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
		for (uint j = i; j < n; j+= argc) {
			SIType actual = SI_TYPE (argv[j]) ;
			if (unlikely (!(actual & expected))) {
				Error_SITypeMismatch (argv[j], expected_type) ;
				return false ;
			}
		}
	}

	return true ;
}

static AR_EXP_Result _AR_EXP_EvaluateFunctionCall_Batch
(
	SIValue *restrict res,        // [input/output] results
	AR_ExpNode *restrict node,    // function node
	const Record *restrict recs,  // records
	size_t n,                     // number of batches
	uint step                     // step within 'res'
) {
	AR_EXP_Result ret = EVAL_OK ;

	uint child_count = NODE_CHILD_COUNT (node) ;
	uint argc = child_count * n ;

	// evaluate each child before evaluating current node
	SIValue *args = NULL;

	// if array size is above the threshold
	// we allocate it on the heap (otherwise on stack)
	size_t array_on_stack_size = argc > MAX_ARRAY_SIZE_ON_STACK ? 0 : argc ;
	SIValue sub_trees_on_stack[array_on_stack_size] ;

	if (argc > MAX_ARRAY_SIZE_ON_STACK) {
		args = rm_malloc (argc * sizeof(SIValue)) ;
	} else {
		args = sub_trees_on_stack ;
	}

	bool param_found = false ;
	for (int child_idx = 0 ; child_idx < child_count ; child_idx++) {
		AR_ExpNode *child = NODE_CHILD (node, child_idx) ;
		ret = _AR_EXP_Evaluate_Batch (args + child_idx, child, recs, n,
				child_count) ;

		if (ret == EVAL_ERR) {
			// encountered an error while evaluating a subtree
			// free all arguments generated up to this point
			// and propagate the error upwards
			_AR_EXP_FreeArgsArray (args, child_idx, child_count, argc) ;
			return ret ;
		}

		param_found |= (ret == EVAL_FOUND_PARAM) ;
	}

	if (param_found) {
		ret = EVAL_FOUND_PARAM;
	}

	// validate before evaluation
	if (!_AR_EXP_ValidateInvocation_Batch (node->op.f, args, child_count, argc)) {
		// the expression tree failed its validations and set an error message
		ret = EVAL_ERR ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// execute function
	//--------------------------------------------------------------------------

	for (uint i = 0; i < n; i++) {
		SIValue v = node->op.f->func (args + i * child_count, child_count,
				node->op.private_data) ;

		ASSERT (node->op.f->aggregate || SI_TYPE(v) & AR_FuncDesc_RetType(node->op.f)) ;
		if (SIValue_IsNull (v) && ErrorCtx_EncounteredError ()) {
			// an error was encountered while evaluating this function
			// and has already been set in the QueryCtx
			// exit with an error
			ret = EVAL_ERR ;
		}

		if (res) {
			SIValue_Persist (&v) ;
			res[i * step] = v ;
		}
	}

cleanup:
	_AR_EXP_FreeArgsArray (args, child_count, child_count, argc) ;
	return ret ;
}

static inline AR_EXP_Result _AR_EXP_Evaluate_Const_Batch
(
	SIValue *restrict res,            // [input/output] results
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
	AR_ExpNode *restrict node,    // variadic node
	const Record *restrict recs,  // records
	uint n,                       // number of records
	uint step                     // step within 'res'
) {
	// make sure entity record index is known
	if (node->operand.variadic.entity_alias_idx == IDENTIFIER_NOT_FOUND) {
		if (!_AR_EXP_UpdateEntityIdx (&node->operand, recs[0])) {
			return EVAL_ERR ;
		}
	}

	int aliasIdx = node->operand.variadic.entity_alias_idx ;

	for (uint i = 0; i < n; i++) {
		// the value was not created here; share with the caller
		res[i * step] = SI_ShareValue (Record_Get (recs[i], aliasIdx)) ;
	}

	return EVAL_OK ;
}

static AR_EXP_Result _AR_EXP_EvaluateParam_Batch
(
	SIValue *restrict res,      // [input/output] results
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

	root->operand.type     = AR_EXP_CONSTANT ;
	root->operand.constant = SI_ShareValue (*param) ;

	// copy const into each result entry
	for(uint i = 0; i < n; i++) {
		res[i * step] = root->operand.constant ;
	}

	return EVAL_FOUND_PARAM ;
}

static AR_EXP_Result _AR_EXP_EvaluateBorrowRecord_Batch
(
	SIValue *restrict res,        // [input/output] results
	const Record *restrict recs,  // records
	size_t n,                     // number of records
	uint step                     // step within 'res'
) {
	for (uint i = 0; i < n; i++) {
		// wrap the current Record in an SI pointer
		res[i * step] = SI_PtrVal (recs[i]) ;
	}

	return EVAL_OK ;
}

// evaluate an expression tree,
// placing the calculated values in 'res'
// returning whether an error occurred during evaluation
static AR_EXP_Result _AR_EXP_Evaluate_Batch
(
	SIValue *restrict res,        // [input/output] results
	AR_ExpNode *restrict root,    // root of expression tree to evaluate
	const Record *restrict recs,  // batch of records
	size_t n,                     // number of records
	uint step                     // step within 'res'
) {
	switch (root->type) {
		case AR_EXP_OP :
			return _AR_EXP_EvaluateFunctionCall_Batch (res, root, recs, n,
					step) ;

		case AR_EXP_OPERAND:
			switch (root->operand.type) {
				case AR_EXP_CONSTANT :
					return _AR_EXP_Evaluate_Const_Batch (res, root, n, step) ;

				case AR_EXP_VARIADIC :
					return _AR_EXP_EvaluateVariadic_Batch (res, root, recs, n,
							step) ;

				case AR_EXP_PARAM :
					return _AR_EXP_EvaluateParam_Batch (res, root, n, step) ;

				case AR_EXP_BORROW_RECORD :
					return _AR_EXP_EvaluateBorrowRecord_Batch (res, recs, n,
							step) ;

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

	AR_EXP_Result ret = _AR_EXP_Evaluate_Batch (res, root, recs, n, 1);

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

