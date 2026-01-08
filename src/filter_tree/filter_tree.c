/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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

// forward declarations
void _FilterTree_DeMorgan
(
	FT_FilterNode **root,
	uint negate_count
);

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

// returns the negated operator of given op
// for example NOT(a > b) === a <= b
static AST_Operator _NegateOperator
(
	AST_Operator op
) {
	switch(op) {
		case OP_AND:
			return OP_OR;
		case OP_XOR:
			return OP_XNOR;
		case OP_XNOR:
			return OP_XOR;
		case OP_OR:
			return OP_AND;
		case OP_EQUAL:
			return OP_NEQUAL;
		case OP_NEQUAL:
			return OP_EQUAL;
		case OP_LT:
			return OP_GE;
		case OP_GT:
			return OP_LE;
		case OP_LE:
			return OP_GT;
		case OP_GE:
			return OP_LT;
		default:
			ASSERT(false);
			return OP_UNKNOWN;
	}
}

// negate expression by wrapping it with a NOT function, NOT(exp)
static void _NegateExpression
(
	AR_ExpNode **exp
) {
	AR_ExpNode *root = AR_EXP_NewOpNode("not", true, 1);
	root->op.children[0] = *exp;
	*exp = root;
}

int IsNodePredicate
(
	const FT_FilterNode *node
) {
	return node->t == FT_N_PRED;
}

FT_FilterNode *FilterTree_AppendLeftChild
(
	FT_FilterNode *root,
	FT_FilterNode *child
) {
	root->cond.left = child;
	return root->cond.left;
}

FT_FilterNode *FilterTree_AppendRightChild
(
	FT_FilterNode *root,
	FT_FilterNode *child
) {
	root->cond.right = child;
	return root->cond.right;
}

FT_FilterNode *FilterTree_CreateExpressionFilter
(
	AR_ExpNode *exp
) {
	ASSERT(exp != NULL);
	FT_FilterNode *node = rm_malloc(sizeof(FT_FilterNode));
	node->t = FT_N_EXP;
	node->exp.exp = exp;
	return node;
}

FT_FilterNode *FilterTree_CreatePredicateFilter
(
	AST_Operator op,
	AR_ExpNode *lhs,
	AR_ExpNode *rhs
) {
	FT_FilterNode *filterNode = rm_malloc(sizeof(FT_FilterNode));
	filterNode->t = FT_N_PRED;
	filterNode->pred.op = op;
	filterNode->pred.lhs = lhs;
	filterNode->pred.rhs = rhs;
	return filterNode;
}

FT_FilterNode *FilterTree_CreateConditionFilter
(
	AST_Operator op
) {
	FT_FilterNode *filterNode = rm_malloc(sizeof(FT_FilterNode));
	filterNode->t = FT_N_COND;
	filterNode->cond.op = op;
	return filterNode;
}

// returns type of filter node
FT_FilterNodeType FilterTree_type
(
	const FT_FilterNode *node  // filter tree node
) {
	ASSERT(node != NULL);

	return node->t;
}

// return filtered expression
// NULL is returned if filter tree isn't of type FT_N_EXP
AR_ExpNode *FilterTree_getExpression
(
	const FT_FilterNode *node  // filter tree node
) {
	if(node->t != FT_N_EXP) {
		return NULL;
	}

	return node->exp.exp;
}

void _FilterTree_SubTrees
(
	const FT_FilterNode *root,
	const FT_FilterNode ***sub_trees
) {
	if(root == NULL) return;

	switch(root->t) {
		case FT_N_EXP:
		case FT_N_PRED:
			// this is a simple predicate tree, can not traverse further
			array_append(*sub_trees, root);
			break;
		case FT_N_COND:
			switch(root->cond.op) {
				case OP_AND:
					// break AND down to its components
					_FilterTree_SubTrees(root->cond.left, sub_trees);
					_FilterTree_SubTrees(root->cond.right, sub_trees);
					break;
				case OP_OR:
				case OP_XOR:
				case OP_XNOR:
					// OR, XOR, and XNOR trees must be return as is
					array_append(*sub_trees, root);
					break;
				default:
					ASSERT(0);
					break;
			}
			break;
		default:
			ASSERT(0);
			break;
	}
}

// combine filters into a single filter tree using AND conditions
// filters[0] AND filters[1] AND ... filters[count]
FT_FilterNode *FilterTree_Combine
(
	const FT_FilterNode **filters,
	uint count
) {
	ASSERT(filters != NULL);

	FT_FilterNode *root = NULL;

	if(count > 0) {
		root = FilterTree_Clone(filters[0]);
		for(uint i = 1; i < count; i++) {
			FT_FilterNode *and = FilterTree_CreateConditionFilter(OP_AND);
			FilterTree_AppendLeftChild(and, root);
			FilterTree_AppendRightChild(and, FilterTree_Clone(filters[i]));
			root = and;
		}
	}

	return root;
}

const FT_FilterNode **FilterTree_SubTrees
(
	const FT_FilterNode *root
) {
	const FT_FilterNode **sub_trees = array_new(const FT_FilterNode *, 1);
	_FilterTree_SubTrees(root, &sub_trees);
	return sub_trees;
}

// applies a single filter to a single result
// compares given values, tests if values maintain desired relation (op)
FT_Result _applyFilter
(
	SIValue *aVal,
	SIValue *bVal,
	AST_Operator op
) {
	int disjointOrNull = 0;
	int rel = SIValue_Compare(*aVal, *bVal, &disjointOrNull);
	// if there was null comparison, return false
	if(disjointOrNull == COMPARED_NULL) return FILTER_NULL;
	// values are of disjoint types
	if(disjointOrNull == DISJOINT || disjointOrNull == COMPARED_NAN) {
		// the filter passes if we're testing for inequality, and fails otherwise
		return (op == OP_NEQUAL);
	}

	switch(op) {
		case OP_EQUAL:
			return rel == 0;
		case OP_NEQUAL:
			return rel != 0;
		case OP_GT:
			return rel > 0;
		case OP_GE:
			return rel >= 0;
		case OP_LT:
			return rel < 0;
		case OP_LE:
			return rel <= 0;
		default:
			// op should be enforced by AST
			ASSERT(0);
			break;
	}

	// we shouldn't reach this point
	return 0;
}

FT_Result _applyPredicateFilters
(
	const FT_FilterNode *root,
	const Record r
) {
	// A op B
	// Evaluate the left and right sides of the predicate to obtain
	// comparable SIValues
	SIValue lhs = AR_EXP_Evaluate(root->pred.lhs, r);
	SIValue rhs = AR_EXP_Evaluate(root->pred.rhs, r);

	FT_Result ret = _applyFilter(&lhs, &rhs, root->pred.op);

	SIValue_Free(lhs);
	SIValue_Free(rhs);

	return ret;
}

static FT_Result _applyCondition
(
	const FT_FilterNode *root,
	const Record r
) {
	// root->t == FT_N_COND, visit left subtree
	FT_Result rhs_pass;
	FT_Result lhs_pass = FilterTree_applyFilters(LeftChild(root), r);
	FT_Result pass = lhs_pass;

	if(root->cond.op == OP_AND && lhs_pass != FILTER_FAIL) {
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

		// evaluate right subtree
		rhs_pass = FilterTree_applyFilters(RightChild(root), r);
		if(lhs_pass == FILTER_PASS && rhs_pass == FILTER_PASS) {
			// true && true == true
			pass = FILTER_PASS;
		} else if(rhs_pass == FILTER_FAIL) {
			// ? && false == false
			pass = FILTER_FAIL;
		} else {
			// otherwise NULL
			pass = FILTER_NULL;
		}
	} else if(root->cond.op == OP_OR && lhs_pass != FILTER_PASS) {
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

		// visit right subtree
		rhs_pass = FilterTree_applyFilters(RightChild(root), r);
		if(rhs_pass == FILTER_PASS) {
			// ? || true == true
			pass = FILTER_PASS;
		} else if(lhs_pass == FILTER_FAIL && rhs_pass == FILTER_FAIL) {
			// false || false == false
			pass = FILTER_FAIL;
		} else {
			// otherwise NULL
			pass = FILTER_NULL;
		}
	} else if(root->cond.op == OP_XOR && lhs_pass != FILTER_NULL) {
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

		// visit right subtree
		rhs_pass = FilterTree_applyFilters(RightChild(root), r);
		if(rhs_pass == FILTER_NULL) {
			pass = FILTER_NULL;
		} else {
			pass = (lhs_pass == rhs_pass) ? FILTER_FAIL : FILTER_PASS;
		}
	} else if(root->cond.op == OP_XNOR && lhs_pass != FILTER_NULL) {
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

		// visit right subtree
		rhs_pass = FilterTree_applyFilters(RightChild(root), r);
		if(rhs_pass == FILTER_NULL) {
			pass = FILTER_NULL;
		} else {
			pass = (lhs_pass == rhs_pass) ? FILTER_PASS : FILTER_FAIL;
		}
	} else if(root->cond.op == OP_NOT && lhs_pass != FILTER_NULL) {
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

		pass = lhs_pass == FILTER_PASS ? FILTER_FAIL : FILTER_PASS;
	}

	return pass;
}

FT_Result FilterTree_applyFilters
(
	const FT_FilterNode *root,
	const Record r
) {
	switch(root->t) {
		case FT_N_COND: {
			return _applyCondition(root, r);
		}
		case FT_N_PRED: {
			return _applyPredicateFilters(root, r);
		}
		case FT_N_EXP: {
			FT_Result retval = FILTER_PASS;
			SIValue res = AR_EXP_Evaluate(root->exp.exp, r);
			if(SIValue_IsNull(res)) {
				// expression evaluated to NULL should return NULL
				retval = FILTER_NULL;
			} else if(SI_TYPE(res) & T_BOOL) {
				// return false if this boolean value is false
				if(SIValue_IsFalse(res)) retval = FILTER_FAIL;
			} else if(SI_TYPE(res) & T_ARRAY) {
				// an empty array is falsey, all other arrays should return true
				if(SIArray_Length(res) == 0) retval = FILTER_FAIL;
			} else {
				// if the expression node evaluated to an unexpected type:
				// numeric, string, node or edge, emit an error
				Error_SITypeMismatch (SI_TYPE (res), T_BOOL) ;
				retval = FILTER_FAIL;
			}

			SIValue_Free(res); // if res was a heap allocation, free it
			return retval;
		}
		default:
			ASSERT(false);
			break;
	}

	// we shouldn't be here
	ASSERT(false);
	return FILTER_FAIL;
}

void _FilterTree_CollectModified
(
	const FT_FilterNode *root,
	rax *modified
) {
	if(root == NULL) return;

	switch(root->t) {
		case FT_N_COND: {
			_FilterTree_CollectModified(root->cond.left, modified);
			_FilterTree_CollectModified(root->cond.right, modified);
			break;
		}
		case FT_N_PRED: {
			// traverse left and right-hand expressions,
			// adding all encountered modified to the triemap
			// we'll typically encounter 0 or 1 modified in each expression,
			// but there are multi-argument exceptions
			AR_EXP_CollectEntities(root->pred.lhs, modified);
			AR_EXP_CollectEntities(root->pred.rhs, modified);
			break;
		}
		case FT_N_EXP: {
			// traverse expression, adding all encountered modified to the triemap
			AR_EXP_CollectEntities(root->exp.exp, modified);
			break;
		}
		default: {
			ASSERT(0);
			break;
		}
	}
}

rax *FilterTree_CollectModified
(
	const FT_FilterNode *root
) {
	rax *modified = raxNew();
	_FilterTree_CollectModified(root, modified);

	return modified;
}

// collect filtered attribute for a given entity
// e.g. person.first_name = 'a' OR person.last_name = 'b'
// will collect both 'first_name' and 'last_name'
// if entity is 'person'
static void _FilterTree_CollectAttributes
(
	const FT_FilterNode *root,  // filter tree
	const char *entity,         // filtered entity
	rax *attributes             // collected attributes
) {
	ASSERT(entity     != NULL);
	ASSERT(attributes != NULL);

	if(root == NULL) return;

	switch(root->t) {
		case FT_N_COND: {
			_FilterTree_CollectAttributes(root->cond.left,  entity, attributes);
			_FilterTree_CollectAttributes(root->cond.right, entity, attributes);
			break;
		}

		case FT_N_PRED: {
			// traverse left and right-hand expressions,
			// adding all encountered attributes to the triemap
			AR_EXP_CollectAttributes(root->pred.lhs, entity, attributes);
			AR_EXP_CollectAttributes(root->pred.rhs, entity, attributes);
			break;
		}

		case FT_N_EXP: {
			AR_EXP_CollectAttributes(root->exp.exp, entity, attributes);
			break;
		}

		default: {
			ASSERT(0);
			break;
		}
	}
}

// collect filtered attribute for a given entity
// e.g. person.first_name = 'a' OR person.last_name = 'b'
// will collect both 'first_name' and 'last_name'
// if entity is 'person'
rax *FilterTree_CollectAttributes
(
	const FT_FilterNode *root,  // filter tree
	const char *entity          // filtered entity
) {
	ASSERT(root   != NULL);
	ASSERT(entity != NULL);

	rax *attributes = raxNew();

	_FilterTree_CollectAttributes(root, entity, attributes);

	return attributes;
}

// check if any of the filtered variable refers to a projection alias
bool FilterTree_FiltersAlias
(
	const FT_FilterNode *root,   // filter tree root
	const cypher_astnode_t *ast  // AST
) {
	// collect all filtered variables
	rax *filtered_variables = FilterTree_CollectModified(root);

	raxIterator it;
	raxStart(&it, filtered_variables);

	// iterate over all keys in the rax
	bool alias_is_filtered = false;

	raxSeek(&it, "^", NULL, 0);
	while(raxNext(&it)) {
		// build string on the stack and add null terminator
		char variable[it.key_len + 1];
		memcpy(variable, it.key, it.key_len);
		variable[it.key_len] = 0;

		// check if the filtered variable is an alias
		if(AST_IdentifierIsAlias(ast, variable)) {
			alias_is_filtered = true;
			break;
		}
	}

	raxStop(&it);
	raxFree(filtered_variables);

	return alias_is_filtered;
}

bool FilterTree_containsOp
(
	const FT_FilterNode *root,
	AST_Operator op
) {
	switch(root->t) {
		case FT_N_COND:
			if(FilterTree_containsOp(root->cond.left, op)) return true;
			if(FilterTree_containsOp(root->cond.right, op)) return true;
			return false;
		case FT_N_EXP:
			return false;
		case FT_N_PRED:
			return (root->pred.op == op);
		default:
			ASSERT(false);
			return false;
	}
}

bool _FilterTree_ContainsFunc
(
	const FT_FilterNode *root,
	const char *func,
	FT_FilterNode **node
) {
	if(root == NULL) return false;
	switch(root->t) {
		case FT_N_COND: {
			return FilterTree_ContainsFunc(root->cond.left, func, node) ||
				   FilterTree_ContainsFunc(root->cond.right, func, node);
		}
		case FT_N_PRED: {
			if(AR_EXP_ContainsFunc(root->pred.lhs, func) || AR_EXP_ContainsFunc(root->pred.rhs, func)) {
				*node = (FT_FilterNode *)root;
				return true;
			}
			return false;
		}
		case FT_N_EXP: {
			if(AR_EXP_ContainsFunc(root->exp.exp, func)) {
				*node = (FT_FilterNode *) root;
				return true;
			}
			return false;
		}
		default:
			ASSERT("Unkown filter tree node type" && false);
			break;
	}
	return false;
}

bool FilterTree_ContainsFunc
(
	const FT_FilterNode *root,
	const char *func,
	FT_FilterNode **node
) {
	ASSERT(root && func && node);
	*node = NULL;
	return _FilterTree_ContainsFunc(root, func, node);
}

void _FilterTree_ApplyNegate
(
	FT_FilterNode **root,
	uint negate_count
) {
	switch((*root)->t) {
		case FT_N_EXP:
			if(negate_count % 2 == 1) {
				_NegateExpression(&((*root)->exp.exp));
			}
			break;
		case FT_N_PRED:
			if(negate_count % 2 == 1) {
				(*root)->pred.op = _NegateOperator((*root)->cond.op);
			}
			break;
		case FT_N_COND:
			if((*root)->cond.op == OP_NOT) {
				// _FilterTree_DeMorgan will increase negate_count by 1
				_FilterTree_DeMorgan(root, negate_count);
			} else {
				if(negate_count % 2 == 1) {
					(*root)->cond.op = _NegateOperator((*root)->cond.op);
				}
				_FilterTree_ApplyNegate(&(*root)->cond.left, negate_count);
				_FilterTree_ApplyNegate(&(*root)->cond.right, negate_count);
			}
			break;
		default:
			ASSERT(false);
			break;
	}
}

// if a filter node that's not a child of a predicate is an expression,
// it should resolve to a boolean value
static inline bool _FilterTree_ValidExpressionNode
(
	const FT_FilterNode *root
) {
	bool valid = AR_EXP_ReturnsBoolean(root->exp.exp);
	if(!valid) ErrorCtx_SetError(EMSG_EXPECTED_BOOLEAN_PREDICATE);
	return valid;
}

bool FilterTree_Valid
(
	const FT_FilterNode *root
) {
	// An empty tree is has a valid structure.
	if(!root) return true;

	switch(root->t) {
		case FT_N_EXP:
			return _FilterTree_ValidExpressionNode(root);
			break;
		case FT_N_PRED:
			// Empty or semi empty predicate, invalid structure.
			if((!root->pred.lhs || !root->pred.rhs)) {
				ErrorCtx_SetError(EMSG_FILTER_INVALID_COMPARISON);
				return false;
			}
			break;
		case FT_N_COND:
			// Empty condition, invalid structure.
			// OR, AND should utilize both left and right children
			// NOT utilize only the left child.
			if(!root->cond.left && !root->cond.right) {
				ErrorCtx_SetError(EMSG_EMPTY_FILTER);
				return false;
			}
			if(root->cond.op == OP_NOT && root->cond.right) {
				ErrorCtx_SetError(EMSG_INVALID_NOT_USAGE);
				return false;
			}
			if(!FilterTree_Valid(root->cond.left)) return false;
			if(!FilterTree_Valid(root->cond.right)) return false;
			break;
		default:
			ASSERT("Unknown filter tree node" && false);
			break;
	}
	return true;
}

void _FilterTree_DeMorgan
(
	FT_FilterNode **root,
	uint negate_count
) {
	// search for NOT nodes and reduce using DeMorgan
	if(*root == NULL || (*root)->t == FT_N_PRED || (*root)->t == FT_N_EXP) return;

	// node is of type condition
	if((*root)->cond.op == OP_NOT) {
		ASSERT((*root)->cond.right == NULL);
		_FilterTree_ApplyNegate(&(*root)->cond.left, negate_count + 1);
		// replace NOT node with only child
		FT_FilterNode *child = (*root)->cond.left;
		(*root)->cond.left = NULL;
		FilterTree_Free(*root);
		*root = child;
	} else {
		FilterTree_DeMorgan(&((*root)->cond.left));
		FilterTree_DeMorgan(&((*root)->cond.right));
	}
}

void FilterTree_DeMorgan
(
	FT_FilterNode **root
) {
	_FilterTree_DeMorgan(root, 0);
}

// return if this node can be used in compression - constant expression
static inline bool _FilterTree_Compact_Exp
(
	FT_FilterNode *node
) {
	return AR_EXP_IsConstant(node->exp.exp) || AR_EXP_IsParameter(node->exp.exp);
}

// in place set an existing filter tree node to expression node
static inline void _FilterTree_In_Place_Set_Exp
(
	FT_FilterNode *node,
	SIValue v
) {
	node->t = FT_N_EXP;
	node->exp.exp = AR_EXP_NewConstOperandNode(v);
}

// compacts 'AND' condition node
static bool _FilterTree_Compact_And
(
	FT_FilterNode *node
) {
	// try to compact left and right children
	bool is_lhs_const = FilterTree_Compact(node->cond.left);
	bool is_rhs_const = FilterTree_Compact(node->cond.right);

	// if both are not compactable, this node is not compactable
	if(!is_lhs_const && !is_rhs_const) return false;

	// in every case from now, there will be a reduction
	// save the children in local placeholders
	// for current node in-place modifications
	FT_FilterNode *lhs = node->cond.left;
	FT_FilterNode *rhs = node->cond.right;

	// both children are constants
	// this node can be set as a constant expression
	if(is_lhs_const && is_rhs_const) {
		// both children are contant expressions
		// we can evaluate and compact
		// final value is AND operation on lhs and rhs - reducing an AND node
		SIValue  final_value  =  SI_NullVal();
		SIValue  lhs_value    =  AR_EXP_Evaluate(lhs->exp.exp,  NULL);
		SIValue  rhs_value    =  AR_EXP_Evaluate(rhs->exp.exp,  NULL);

		if(!SIValue_IsNull(lhs_value) && !SIValue_IsNull(rhs_value)) {
			// both lhs and rhs are NOT NULL
			final_value = SI_BoolVal(SIValue_IsTrue(lhs_value) && SIValue_IsTrue(rhs_value));
		} else if((!SIValue_IsNull(lhs_value) && SIValue_IsFalse(lhs_value)) ||
				  (!SIValue_IsNull(rhs_value) && SIValue_IsFalse(rhs_value))) {
			// FALSE AND NULL is NULL
			final_value =  SI_BoolVal(false);
		}

		// in place set the node to be an expression node
		_FilterTree_In_Place_Set_Exp(node, final_value);
		FilterTree_Free(lhs);
		FilterTree_Free(rhs);
		return true;
	} else {
		// only one of the nodes is constant, find and evaluate
		FT_FilterNode *const_node = is_lhs_const ? lhs : rhs;
		FT_FilterNode *non_const_node = is_lhs_const ? rhs : lhs;

		// evaluate constant
		SIValue const_value = AR_EXP_Evaluate(const_node->exp.exp, NULL);
		// if constant is null, no compaction
		if(SIValue_IsNull(const_value)) return false;
		
		// if consant is false, everything is false
		if(SIValue_IsFalse(const_value)) {
			*node = *const_node;
			// free const node allocation, without free the data
			rm_free(const_node);
			// free non const node completely
			FilterTree_Free(non_const_node);
			return true;
		} else {
			// const value is either true
			// current node should be replaced with the non const node
			*node = *non_const_node;
			// free non const node allocation, without free the data
			rm_free(non_const_node);
			// free const node completely
			FilterTree_Free(const_node);
			return false;
		}
	}
}

// Compacts 'OR' condition node.
static bool _FilterTree_Compact_Or
(
	FT_FilterNode *node
) {
	// try to compact left and right children
	bool is_lhs_const = FilterTree_Compact(node->cond.left);
	bool is_rhs_const = FilterTree_Compact(node->cond.right);
	// if both are not compactable, this node is not compactable
	if(!is_lhs_const && !is_rhs_const) return false;

	// in every case from now, there will be a reduction,
	// save the children in local placeholders for current node in-place modifications
	FT_FilterNode *lhs = node->cond.left;
	FT_FilterNode *rhs = node->cond.right;
	// both children are constants. This node can be set as constant expression
	if(is_lhs_const && is_rhs_const) {
		// both children are now contant expressions, evaluate and compact
		SIValue final_value = SI_NullVal();
		SIValue lhs_value = AR_EXP_Evaluate(rhs->exp.exp, NULL);
		SIValue rhs_value = AR_EXP_Evaluate(lhs->exp.exp, NULL);
		if(!SIValue_IsNull(lhs_value) && !SIValue_IsNull(rhs_value)) {
			final_value =  SI_BoolVal(SIValue_IsTrue(lhs_value) || SIValue_IsTrue(rhs_value));
		} else if((!SIValue_IsNull(lhs_value) && SIValue_IsTrue(lhs_value)) ||
				  (!SIValue_IsNull(rhs_value) && SIValue_IsTrue(rhs_value))) {
			final_value =  SI_BoolVal(true);
		}

		// final value is OR operation on lhs and rhs - reducing an OR node
		// in place set the node to be an expression node
		_FilterTree_In_Place_Set_Exp(node, final_value);
		FilterTree_Free(lhs);
		FilterTree_Free(rhs);
		return true;
	} else {
		// only one of the nodes is constant, find and evaluate
		FT_FilterNode *const_node = is_lhs_const ? lhs : rhs;
		FT_FilterNode *non_const_node = is_lhs_const ? rhs : lhs;

		// evaluate constant
		SIValue const_value = AR_EXP_Evaluate(const_node->exp.exp, NULL);
		if(SIValue_IsNull(const_value)) return false;

		// if consant is true, everything is true
		if(SIValue_IsTrue(const_value)) {
			*node = *const_node;
			// free const node allocation, without free the data
			rm_free(const_node);
			// free non const node completely
			FilterTree_Free(non_const_node);
			return true;
		} else {
			// const value is false, current node should be replaced with the non const node
			*node = *non_const_node;
			// free non const node allocation, without free the data
			rm_free(non_const_node);
			// free const node completely
			FilterTree_Free(const_node);
			return false;
		}
	}
}

// compacts 'XOR' and 'XNOR' condition nodes.
static bool _FilterTree_Compact_XOr
(
	FT_FilterNode *node,
	bool xnor
) {
	// try to compact left and right children
	bool is_lhs_const = FilterTree_Compact(node->cond.left);
	bool is_rhs_const = FilterTree_Compact(node->cond.right);

	// if both are not compactable, this node is not compactable
	if(!is_lhs_const && !is_rhs_const) return false;

	// in every case from now, there will be a reduction,
	// save the children in local placeholders for current node in-place modifications
	SIValue final_value = SI_NullVal();
	FT_FilterNode *lhs = node->cond.left;
	FT_FilterNode *rhs = node->cond.right;

	// both children are constants
	// this node can be set as constant expression
	if(is_lhs_const && is_rhs_const) {
		// both children are now contant expressions, evaluate and compact
		SIValue lhs_value = AR_EXP_Evaluate(lhs->exp.exp, NULL);
		SIValue rhs_value = AR_EXP_Evaluate(rhs->exp.exp, NULL);
		if(!SIValue_IsNull(lhs_value) && !SIValue_IsNull(rhs_value)) {
			// lhs and rhs are NOT NULL
			if(xnor) {
				final_value = SI_BoolVal(SIValue_IsTrue(lhs_value) == SIValue_IsTrue(rhs_value));
			} else {
				final_value = SI_BoolVal(SIValue_IsTrue(lhs_value) != SIValue_IsTrue(rhs_value));
			}
		}

		// final value is XOR operation on lhs and rhs - reducing an XOR node
		// in place set the node to be an expression node
		_FilterTree_In_Place_Set_Exp(node, final_value);
		FilterTree_Free(lhs);
		FilterTree_Free(rhs);
		return true;
	} else if(is_lhs_const || is_rhs_const) {
		// either lhs or rhs is const
		SIValue value;
		if(is_lhs_const) {
			value = AR_EXP_Evaluate(lhs->exp.exp, NULL);
		} else {
			value = AR_EXP_Evaluate(rhs->exp.exp, NULL);
		}

		// reduce to NULL if `value` is NULL
		// ? XOR NULL == ? XNOR NULL == NULL
		if(SIValue_IsNull(value)) {
			_FilterTree_In_Place_Set_Exp(node, value);
			FilterTree_Free(lhs);
			FilterTree_Free(rhs);
			return true;
		}
	}

	return false;
}

// compacts a condition node if possible
static inline bool _FilterTree_Compact_Cond
(
	FT_FilterNode *node
) {
	switch(node->cond.op) {
		case OP_AND:
			return _FilterTree_Compact_And(node);
		case OP_OR:
			return _FilterTree_Compact_Or(node);
		case OP_XOR:
			return _FilterTree_Compact_XOr(node, false);
		case OP_XNOR:
			return _FilterTree_Compact_XOr(node, true);
		default:
			ASSERT(false && "_FilterTree_Compact_Cond: Unkown filter operator to compact");
			return false;
	}
}

// compacts a predicate node if possible
static bool _FilterTree_Compact_Pred
(
	FT_FilterNode *node
) {
	// check if both sides are constant expressions
	if((AR_EXP_IsConstant(node->pred.lhs) || AR_EXP_IsParameter(node->pred.lhs)) &&
	   (AR_EXP_IsConstant(node->pred.rhs) || AR_EXP_IsParameter(node->pred.rhs))) {
		// Evaluate expressions.
		SIValue lhs = AR_EXP_Evaluate(node->pred.lhs, NULL);
		SIValue rhs = AR_EXP_Evaluate(node->pred.rhs, NULL);
		// Evalute result.
		FT_Result ret = _applyFilter(&lhs, &rhs, node->pred.op);
		// Result can be NULL like WHERE null <> true otherwise it's bool
		SIValue v = ret == FILTER_NULL ? SI_NullVal() : SI_BoolVal(ret);
		// Free resources and do in place replacment.
		AR_EXP_Free(node->pred.lhs);
		AR_EXP_Free(node->pred.rhs);
		_FilterTree_In_Place_Set_Exp(node, v);
		return true;
	}
	return false;
}

bool FilterTree_Compact
(
	FT_FilterNode *root
) {
	if(!root) return true;
	switch(root->t) {
		case FT_N_EXP:
			return _FilterTree_Compact_Exp(root);
		case FT_N_COND:
			return _FilterTree_Compact_Cond(root);
		case FT_N_PRED:
			return _FilterTree_Compact_Pred(root);
		default:
			ASSERT(false && "FilterTree_Compact: Unkown filter tree node to compect");
			return false;
	}
}

//------------------------------------------------------------------------------
// Resolve unknows
//------------------------------------------------------------------------------

static void _FilterTree_ResolveVariables
(
	FT_FilterNode *root,
	const Record r
) {
	ASSERT(root != NULL);

	switch(root->t) {
		case FT_N_EXP:
			AR_EXP_ResolveVariables(root->exp.exp, r);
			break;
		case FT_N_COND:
			_FilterTree_ResolveVariables(root->cond.left, r);
			_FilterTree_ResolveVariables(root->cond.right, r);
			break;
		case FT_N_PRED:
			AR_EXP_ResolveVariables(root->pred.lhs, r);
			AR_EXP_ResolveVariables(root->pred.rhs, r);
			break;
		default:
			ASSERT(false && "_FilterTree_ResolveVariables: Unkown filter tree node to compect");
			break;
	}
}

void FilterTree_ResolveVariables
(
	FT_FilterNode *root,
	const Record r
) {
	_FilterTree_ResolveVariables(root, r);
	FilterTree_Compact(root);
}

// clone an expression node
static inline FT_FilterNode *_FilterTree_Clone_Exp
(
	const FT_FilterNode *node
) {
	AR_ExpNode *exp_clone = AR_EXP_Clone(node->exp.exp);
	return FilterTree_CreateExpressionFilter(exp_clone);
}

// clones a condition node
static inline FT_FilterNode *_FilterTree_Clone_Cond
(
	const FT_FilterNode *node
) {
	FT_FilterNode *clone = FilterTree_CreateConditionFilter(node->cond.op);
	FT_FilterNode *left_child_clone = FilterTree_Clone(node->cond.left);
	FilterTree_AppendLeftChild(clone, left_child_clone);
	FT_FilterNode *right_child_clone = FilterTree_Clone(node->cond.right);
	FilterTree_AppendRightChild(clone, right_child_clone);
	return clone;
}

// clones a predicate node
static inline FT_FilterNode *_FilterTree_Clone_Pred
(
	const FT_FilterNode *node
) {
	AST_Operator op = node->pred.op;
	AR_ExpNode *lhs_exp_clone = AR_EXP_Clone(node->pred.lhs);
	AR_ExpNode *rhs_exp_clone = AR_EXP_Clone(node->pred.rhs);
	return FilterTree_CreatePredicateFilter(op, lhs_exp_clone, rhs_exp_clone);
}

FT_FilterNode *FilterTree_Clone
(
	const FT_FilterNode *root
) {
	if(!root) return NULL;
	switch(root->t) {
		case FT_N_EXP:
			return _FilterTree_Clone_Exp(root);
		case FT_N_COND:
			return _FilterTree_Clone_Cond(root);
		case FT_N_PRED:
			return _FilterTree_Clone_Pred(root);
		default:
			ASSERT(false && "Unkown filter tree node to clone");
			return NULL;
	}
}

void _FilterTree_Print
(
	const FT_FilterNode *root,
	int ident
) {
	char *exp = NULL;
	char *left = NULL;
	char *right = NULL;

	if(root == NULL) return;
	// ident
	printf("%*s", ident, "");

	switch(root->t) {
		case FT_N_EXP:
			AR_EXP_ToString(root->exp.exp, &exp);
			printf("%s\n",  exp);
			rm_free(exp);
			break;
		case FT_N_PRED:
			AR_EXP_ToString(root->pred.lhs, &left);
			AR_EXP_ToString(root->pred.rhs, &right);
			printf("%s %d %s\n",  left, root->pred.op, right);
			rm_free(left);
			rm_free(right);
			break;
		case FT_N_COND:
			printf("%d\n", root->cond.op);
			_FilterTree_Print(LeftChild(root), ident + 4);
			_FilterTree_Print(RightChild(root), ident + 4);
			break;
		default:
			ASSERT(false);
			break;
	}
}

void FilterTree_Print
(
	const FT_FilterNode *root
) {
	if(root == NULL) {
		printf("empty filter tree\n");
		return;
	}
	_FilterTree_Print(root, 0);
}

void FilterTree_Free
(
	FT_FilterNode *root
) {
	if(root == NULL) return;
	switch(root->t) {
		case FT_N_EXP:
			AR_EXP_Free(root->exp.exp);
			break;
		case FT_N_PRED:
			AR_EXP_Free(root->pred.lhs);
			AR_EXP_Free(root->pred.rhs);
			break;
		case FT_N_COND:
			FilterTree_Free(root->cond.left);
			FilterTree_Free(root->cond.right);
			break;
		default:
			ASSERT(false);
			break;
	}

	rm_free(root);
}

