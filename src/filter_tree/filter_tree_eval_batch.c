/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "filter_tree.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../ast/ast_shared.h"
#include "../datatypes/array.h"

// pre-calculated truth table for AND
static const FT_Result AND_LUT[4][4] = {
    // R: FAIL, PASS, NULL
    { FILTER_FAIL, FILTER_FAIL, FILTER_FAIL, 0 }, // L: FAIL
    { FILTER_FAIL, FILTER_PASS, FILTER_NULL, 0 }, // L: PASS
    { FILTER_FAIL, FILTER_NULL, FILTER_NULL, 0 }, // L: NULL
    { 0,           0,           0,           0 }  // Padding
};

// pre-calculated truth table for OR
static const FT_Result OR_LUT[4][4] = {
    // L: FAIL(0), PASS(1), NULL(2)
    { FILTER_FAIL, FILTER_PASS, FILTER_NULL, 0 }, // R: FAIL(0)
    { FILTER_PASS, FILTER_PASS, FILTER_PASS, 0 }, // R: PASS(1)
    { FILTER_NULL, FILTER_PASS, FILTER_NULL, 0 }, // R: NULL(2)
    { 0,           0,           0,           0 }  // Padding
};

static inline FT_FilterNode *LeftChild
(
	const FT_FilterNode *node
) {
	return node->cond.left;
}

static inline FT_FilterNode *RightChild
(
	const FT_FilterNode *node
) {
	return node->cond.right;
}

// compares given values, tests if values maintain desired relation (op)
static void _applyBatch_GenericFilterOp
(
	FT_Result *restrict pass,  // result
	SIValue *restrict aVal,    // lhs
	SIValue *restrict bVal,    // rhs
	AST_Operator op,           // op
	uint n                     // number of values
) {
	ASSERT (n > 0) ;
	ASSERT (pass != NULL) ;
	ASSERT (aVal != NULL) ;
	ASSERT (bVal != NULL) ;

	//--------------------------------------------------------------------------
	// Specialized Loop for EQUAL
	//--------------------------------------------------------------------------

	if (op == OP_EQUAL) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare (aVal[i], bVal[i], &state) ;

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL ;
			} else if (unlikely (state != 0)) { // Disjoint or NaN
				pass[i] = FILTER_FAIL ;
			} else {
				pass[i] = (rel == 0) ? FILTER_PASS : FILTER_FAIL ;
			}
		}
	}

	//--------------------------------------------------------------------------
	// Specialized Loop for NEEQUAL
	//--------------------------------------------------------------------------

	else if (op == OP_NEQUAL) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare (aVal[i], bVal[i], &state) ;

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL ;
			} else if (unlikely (state != 0)) { // Disjoint or NaN
				pass[i] = FILTER_FAIL ;
			} else {
				pass[i] = (rel != 0) ? FILTER_PASS : FILTER_FAIL ;
			}
		}
	}

	//--------------------------------------------------------------------------
	// Specialized Loop for GREATER_THAN
	//--------------------------------------------------------------------------

	else if (op == OP_GT) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare(aVal[i], bVal[i], &state);

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL;
			} else if (unlikely (state != 0)) {
				pass[i] = FILTER_FAIL;
			} else {
				pass[i] = (rel > 0) ? FILTER_PASS : FILTER_FAIL;
			}
		}
	}

	//--------------------------------------------------------------------------
	// Specialized Loop for GREATER_EQUAL
	//--------------------------------------------------------------------------

	else if (op == OP_GE) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare(aVal[i], bVal[i], &state);

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL;
			} else if (unlikely (state != 0)) {
				pass[i] = FILTER_FAIL;
			} else {
				pass[i] = (rel >= 0) ? FILTER_PASS : FILTER_FAIL;
			}
		}
	}

	//--------------------------------------------------------------------------
	// Specialized Loop for LESS_THAN
	//--------------------------------------------------------------------------

	else if (op == OP_LT) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare(aVal[i], bVal[i], &state);

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL;
			} else if (unlikely (state != 0)) {
				pass[i] = FILTER_FAIL;
			} else {
				pass[i] = (rel < 0) ? FILTER_PASS : FILTER_FAIL;
			}
		}
	}

	//--------------------------------------------------------------------------
	// Specialized Loop for LESS_EQUAL
	//--------------------------------------------------------------------------

	else if (op == OP_LE) {
		for (uint i = 0 ; i < n; i++) {
			int state = 0 ;
			int rel = SIValue_Compare(aVal[i], bVal[i], &state);

			if (unlikely (state == COMPARED_NULL)) {
				pass[i] = FILTER_NULL;
			} else if (unlikely (state != 0)) {
				pass[i] = FILTER_FAIL;
			} else {
				pass[i] = (rel <= 0) ? FILTER_PASS : FILTER_FAIL;
			}
		}
	}

	else {
		// op should be enforced by AST
		ASSERT(0) ;
	}
}

// helper macro to define the comparison loops to avoid duplication
#define DEFINE_INT64_COMP_LOOP(NAME, OPERATOR) \
static void _comp_int64_##NAME(FT_Result *restrict pass, SIValue *restrict a, SIValue *restrict b, uint n) { \
    for (uint i = 0; i < n; i++) { \
        pass[i] = (a[i].longval OPERATOR b[i].longval) ? FILTER_PASS : FILTER_FAIL; \
    } \
}

DEFINE_INT64_COMP_LOOP(eq,  ==)
DEFINE_INT64_COMP_LOOP(gt,   >)
DEFINE_INT64_COMP_LOOP(ge,  >=)
DEFINE_INT64_COMP_LOOP(lt,   <)
DEFINE_INT64_COMP_LOOP(le,  <=)
DEFINE_INT64_COMP_LOOP(neq, !=)

static void _applyBatchFilter_INT64_INT64
(
	FT_Result *restrict pass,  // result
	SIValue *restrict a,       // lhs
	SIValue *restrict b,       // rhs
	AST_Operator op,           // op
	uint n                     // number of values
) {
	// dispatch to tight, branchless, SIMD-able loops
    switch (op) {
        case OP_EQUAL:   _comp_int64_eq  (pass, a, b, n) ; break ;
        case OP_NEQUAL:  _comp_int64_neq (pass, a, b, n) ; break ;
        case OP_GT:      _comp_int64_gt  (pass, a, b, n) ; break ;
        case OP_GE:      _comp_int64_ge  (pass, a, b, n) ; break ;
        case OP_LT:      _comp_int64_lt  (pass, a, b, n) ; break ;
        case OP_LE:      _comp_int64_le  (pass, a, b, n) ; break ;
        default: ASSERT (0) ;
    }
}

// helper macro to define the comparison loops to avoid duplication
#define DEFINE_DOUBLE_COMP_LOOP(NAME, OPERATOR) \
static void _comp_double_##NAME(FT_Result *restrict pass, SIValue *restrict a, SIValue *restrict b, uint n) { \
    for (uint i = 0; i < n; i++) { \
        pass[i] = (a[i].doubleval OPERATOR b[i].doubleval) ? FILTER_PASS : FILTER_FAIL; \
    } \
}

DEFINE_DOUBLE_COMP_LOOP(eq,  ==)
DEFINE_DOUBLE_COMP_LOOP(gt,   >)
DEFINE_DOUBLE_COMP_LOOP(ge,  >=)
DEFINE_DOUBLE_COMP_LOOP(lt,   <)
DEFINE_DOUBLE_COMP_LOOP(le,  <=)
DEFINE_DOUBLE_COMP_LOOP(neq, !=)

static void _applyBatchFilter_DOUBLE_DOUBLE
(
	FT_Result *restrict pass,  // result
	SIValue *restrict a,       // lhs
	SIValue *restrict b,       // rhs
	AST_Operator op,           // op
	uint n                     // number of values
) {
	// dispatch to tight, branchless, SIMD-able loops
    switch (op) {
        case OP_EQUAL:   _comp_double_eq  (pass, a, b, n) ; break ;
        case OP_NEQUAL:  _comp_double_neq (pass, a, b, n) ; break ;
        case OP_GT:      _comp_double_gt  (pass, a, b, n) ; break ;
        case OP_GE:      _comp_double_ge  (pass, a, b, n) ; break ;
        case OP_LT:      _comp_double_lt  (pass, a, b, n) ; break ;
        case OP_LE:      _comp_double_le  (pass, a, b, n) ; break ;
        default: ASSERT (0) ;
    }
}

// compares given values, tests if values maintain desired relation (op)
void _applyBatchFilterOp
(
	FT_Result *restrict pass,  // result
	SIValue *restrict a,       // lhs
	SIValue *restrict b,       // rhs
	AST_Operator op,           // op
	uint n                     // number of values
) {
	ASSERT (n > 0) ;
	ASSERT (a    != NULL) ;
	ASSERT (b    != NULL) ;
	ASSERT (pass != NULL) ;

	SIType typeA = a[0].type;
    SIType typeB = b[0].type;

	// speculative type check (O(n) but very fast/cache-friendly)
    for (uint i = 0; i < n; i++) {
        if (a[i].type != b[i].type) {
            // if the batch isn't purely T
			// fall back to the generic version for the remainder
            _applyBatch_GenericFilterOp (pass + i, a + i, b + i, op, n - i);

            n = i; // limit the fast path to the homogeneous prefix
            break ;
        }
    }

	if (n == 0) {
		return ;
	}

	// fast path: homogeneous types
	// run a tight loop that only does T comparisons
	// if we find a different type midway, we can fallback
	// but usually types don't switch mid-batch
    if (typeA == T_INT64 && typeB == T_INT64) {
        _applyBatchFilter_INT64_INT64 (pass, a, b, op, n);
        return ;
    }

	else if (typeA == T_DOUBLE && typeB == T_DOUBLE) {
        _applyBatchFilter_DOUBLE_DOUBLE (pass, a, b, op, n);
		return ;
	}

	else {
		// generic version
		_applyBatch_GenericFilterOp (pass, a, b, op, n) ;
	}

	// TODO: add specific filter version the following types:
	// T_BOOL, T_NODE, T_EDGE, T_STRING,
	// T_DATETIME, T_LOCALDATETIME, T_DATE, T_TIME, T_LOCALTIME, T_DURATION,
	// T_POINT, T_VECTOR_F32
}

// apply a predicate filter
// e.g. n.v > 4
void _applyBatchPredicate
(
	FT_Result *restrict pass,            // results
	const FT_FilterNode *restrict root,  // filter
	const Record *restrict records,      // records
	size_t n                             // number of records
) {
	ASSERT (n > 0) ;
	ASSERT (pass    != NULL) ;
	ASSERT (root    != NULL) ;
	ASSERT (records != NULL) ;

	// A op B
	// evaluate the left and right sides of the predicate to obtain
	// comparable SIValues
	SIValue lhs[n] ;
	SIValue rhs[n] ;

	AR_EXP_Evaluate_Batch (lhs, root->pred.lhs, records, n) ;
	AR_EXP_Evaluate_Batch (rhs, root->pred.rhs, records, n) ;

	_applyBatchFilterOp (pass, lhs, rhs, root->pred.op, n) ;

	// clean up
	for (uint i = 0; i < n; i++) {
		SIValue_Free (lhs[i]) ;
		SIValue_Free (rhs[i]) ;
	}
}

static void _applyBatchCondition
(
	FT_Result *restrict pass,            // [input/output] filter results
	const FT_FilterNode *restrict root,  // filter tree root
	const Record *restrict records,      // records to filter
	size_t n                             // number of records
) {
	ASSERT (n > 0) ;
	ASSERT (pass    != NULL) ;
	ASSERT (root    != NULL) ;
	ASSERT (records != NULL) ;

	//--------------------------------------------------------------------------
	// filter left hand side
	//--------------------------------------------------------------------------

	FilterTree_applyBatchFilters (pass, LeftChild (root), records, n) ;

	uint j = 0 ;             // index into rhs_records
	uint idx[n] ;            // mapping failing records and original records 
	Record rhs_records[n] ;  // records passed to the filter right hand side

	//--------------------------------------------------------------------------
	// filter right hand side
	//--------------------------------------------------------------------------

	if (root->cond.op == OP_AND) {
		// AND truth table
		// ------------------------
		// AND  | T     F    NULL |
		// ------------------------
		// T    | T     F    NULL |
		// ------------------------
		// F    | F     F    F    |
		// ------------------------
		// NULL | NULL  F    NULL |
		// ------------------------
		// AND ( F, ? ) == F
		// AND ( T, T ) == T
		// otherwise NULL

		// collect 
		for (uint i = 0; i < n; i++) {
			if (pass[i] != FILTER_FAIL) {
				// either True or NULL
				// remember record index
				idx [j] = i ;
				rhs_records[j] = records[i] ;
				j++ ;
			}
		}

		// process right hand side for none determined records
		if (j > 0) {
			// evaluate right subtree
			FT_Result rhs[j] ;
			FilterTree_applyBatchFilters (rhs, RightChild (root), rhs_records,
					j) ;

			// compute final result
			for (uint i = 0; i < j; i++) {
				uint target_idx = idx[i] ;
				pass[target_idx] = AND_LUT[rhs[i]][pass[target_idx]] ;
			}
		}
	} else if (root->cond.op == OP_OR) {
		// OR truth table
		// ------------------------
		// OR   | T     F    NULL |
		// ------------------------
		// T    | T     T    T    |
		// ------------------------
		// F    | T     F    NULL |
		// ------------------------
		// NULL | T     NULL NULL |
		// ------------------------
		// OR ( T, ? ) == T
		// OR ( F, F ) == F
		// otherwise NULL

		for (uint i = 0; i < n; i++) {
			if (pass[i] != FILTER_PASS) {
				// either False or NULL
				// remember record index
				idx[j]  = i ;
				rhs_records[j] = records[i] ;
				j++ ;
			}
		}

		// process right hand side for none determined records
		if (j > 0) {
			// evaluate right subtree
			FT_Result rhs[j] ;
			FilterTree_applyBatchFilters (rhs, RightChild (root), rhs_records, j) ;

			// compute final result
			for (uint i = 0; i < j; i++) {
				uint target_idx = idx[i] ;
				// lookup result based on RHS result and existing LHS result
				pass[target_idx] = OR_LUT[rhs[i]][pass[target_idx]] ;
			}
		}
	} else if (root->cond.op == OP_XOR) {
		// XOR truth table
		// ------------------------
		// XOR  | T     F    NULL |
		// ------------------------
		// T    | F     T    NULL |
		// ------------------------
		// F    | T     F    NULL |
		// ------------------------
		// NULL | NULL  NULL NULL |
		// ------------------------
		// XOR ( T, F ) == T
		// XOR ( F, T ) == T
		// XOR ( F, F ) == F
		// XOR ( T, T ) == F
		// otherwise NULL

		// evaluate right subtree
		FT_Result rhs[n] ;
		FilterTree_applyBatchFilters (rhs, RightChild (root), records, n) ;

		// compute final result
		for (uint i = 0; i < n; i++) {
			if (pass[i] == FILTER_NULL || rhs[i] == FILTER_NULL) {
				pass[i] = FILTER_NULL ;
			} else {
				pass[i] = (pass[i] == rhs[i]) ? FILTER_FAIL : FILTER_PASS ;
			}
		}
	} else if (root->cond.op == OP_XNOR) {
		// XNOR truth table
		// ------------------------
		// XNOR | T     F    NULL |
		// ------------------------
		// T    | T     F    NULL |
		// ------------------------
		// F    | F     T    NULL |
		// ------------------------
		// NULL | NULL  NULL NULL |
		// ------------------------
		// XOR ( T, F ) == F
		// XOR ( F, T ) == F
		// XOR ( F, F ) == T
		// XOR ( T, T ) == T
		// otherwise NULL

		// evaluate right subtree
		FT_Result rhs[n] ;
		FilterTree_applyBatchFilters (rhs, RightChild (root), records, n);

		// compute final result
		for (uint i = 0; i < n; i++) {
			if (pass[i] == FILTER_NULL || rhs[i] == FILTER_NULL) {
				pass[i] = FILTER_NULL ;
			} else {
				pass[i] = (pass[i] == rhs[i]) ? FILTER_PASS : FILTER_FAIL ;
			}
		}
	} else if (root->cond.op == OP_NOT) {
		// NOT truth table
		// -------------
		// NOT         | 
		// -------------
		// T    | F    |
		// -------------
		// F    | T    |
		// -------------
		// NULL | NULL |
		// -------------

		for (size_t i = 0; i < n; i++) {
			// if value is 2 (NULL), result is 2
			// if value is 0 (FAIL), result is 1
			// if value is 1 (PASS), result is 0
			pass[i] = (pass[i] & 2) | (!pass[i] & 1);
		}
	}
}

// evaluate filter for multiple records
void FilterTree_applyBatchFilters
(
	FT_Result *restrict pass,            // [input/output] filter results
	const FT_FilterNode *restrict root,  // filter tree
	const Record *restrict records,      // records
	size_t n                             // number of records
) {
	// inspect root's type
	switch (root->t) {
		case FT_N_COND: {
			_applyBatchCondition (pass, root, records, n) ;
			break;
		}
		case FT_N_PRED: {
			_applyBatchPredicate (pass, root, records, n) ;
			break;
		}
		case FT_N_EXP: {
			SIValue res[n];
			AR_EXP_Evaluate_Batch (res, root->exp.exp, records, n) ;

			// determine if filter passed for each evaluated expression
			for(uint i = 0; i < n; i++) {
				if(SI_TYPE(res[i]) & T_BOOL) {
					// return false if this boolean value is false
					pass[i] = SIValue_IsTrue(res[i]);
				} else if(SIValue_IsNull(res[i])) { 
					// expression evaluated to NULL should return NULL
					pass[i] = FILTER_NULL;
				} else if(SI_TYPE(res[i]) & T_ARRAY) {
					// an empty array is falsey, all other arrays should return true
					if(SIArray_Length(res[i]) == 0) pass[i] = FILTER_FAIL;
				} else {
					// if the expression node evaluated to an unexpected type:
					// numeric, string, node or edge, emit an error
					Error_SITypeMismatch (SI_TYPE (res[i]), T_BOOL) ;
					pass[i] = FILTER_FAIL;
				}

				SIValue_Free(res[i]); // if res was a heap allocation, free it
			}
			break;
		}
		default:
			ASSERT(false);
			break;
	}
}

