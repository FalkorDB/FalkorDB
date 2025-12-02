/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./arithmetic_expression.h"

#include "RG.h"
#include "funcs.h"
#include "rax.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../graph/graph.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "../datatypes/temporal_value.h"
#include "../datatypes/array.h"
#include "../ast/ast_shared.h"

#include <ctype.h>

// returns true if given node 'n' represents an aggregation expression
#define AGGREGATION_NODE(n) ((AR_EXP_IsOperation(n)) && (n)->op.f->aggregate)

// return number of child nodes of 'n'
#define NODE_CHILD_COUNT(n) (n)->op.child_count

// return child at position 'idx' of 'n'
#define NODE_CHILD(n, idx) (n)->op.children[(idx)]

// maximum size for which an array of SIValue will be stack-allocated, otherwise it will be heap-allocated.
#define MAX_ARRAY_SIZE_ON_STACK 32

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

static AR_EXP_Result _AR_EXP_EvaluateVariadic(AR_ExpNode *node, const Record r,
		SIValue *result);

static AR_EXP_Result _AR_EXP_Evaluate(AR_ExpNode *root, const Record r,
									  SIValue *result);

static void _AR_EXP_ResolveVariables(AR_ExpNode *root, const Record r);

// Clear an op node internals, without freeing the node allocation itself.
static void _AR_EXP_FreeOpInternals(AR_ExpNode *op_node);

static void _AR_EXP_FreeInternals (AR_ExpNode *node) ;

inline bool AR_EXP_IsConstant(const AR_ExpNode *exp) {
	return exp->type == AR_EXP_OPERAND && exp->operand.type == AR_EXP_CONSTANT;
}

inline bool AR_EXP_IsVariadic(const AR_ExpNode *exp) {
	return exp->type == AR_EXP_OPERAND && exp->operand.type == AR_EXP_VARIADIC;
}

inline bool AR_EXP_IsParameter(const AR_ExpNode *exp) {
	return exp->type == AR_EXP_OPERAND && exp->operand.type == AR_EXP_PARAM;
}

inline bool AR_EXP_IsOperation(const AR_ExpNode *exp) {
	return exp->type == AR_EXP_OP;
}

bool AR_EXP_IsAttribute(const AR_ExpNode *exp, char **attr) {
	ASSERT(exp != NULL);

	// an arithmetic expression performs attribute extraction
	// if it applys the "property" function, in which case the left-handside
	// child represents the graph entity from which we access the attribute
	// while the right-handside represents the attribute name

	if(exp->type != AR_EXP_OP) return false;
	if(strcmp(AR_EXP_GetFuncName(exp), "property") != 0) return false;

	if(attr != NULL) {
		AR_ExpNode *r = exp->op.children[1];
		ASSERT(AR_EXP_IsConstant(r));
		SIValue v = r->operand.constant;
		ASSERT(SI_TYPE(v) == T_STRING);
		*attr = v.stringval;
	}

	return true;
}

// returns true if `exp` is an aggregation
bool AR_EXP_IsAggregation
(
	const AR_ExpNode *exp  // expression to inspect
) {
	ASSERT (exp != NULL) ;
	return AGGREGATION_NODE (exp) ;
}

bool AR_EXP_PerformsDistinct(AR_ExpNode *exp) {
	return AR_EXP_ContainsFunc(exp, "distinct");
}

// repurpose node to a constant expression
static void _AR_EXP_InplaceRepurposeConstant(AR_ExpNode *node, SIValue v) {
	// free node internals
	if(AR_EXP_IsOperation(node)) _AR_EXP_FreeOpInternals(node);
	else if(AR_EXP_IsConstant(node)) SIValue_Free(node->operand.constant);

	// repurpose as constant operand
	node->type              =  AR_EXP_OPERAND;
	node->operand.type      =  AR_EXP_CONSTANT;
	node->operand.constant  =  v;
}

static AR_ExpNode *_AR_EXP_CloneOperand
(
	const AR_ExpNode *exp
) {
	AR_ExpNode *clone = rm_calloc(1, sizeof(AR_ExpNode));
	clone->type = AR_EXP_OPERAND;

	switch(exp->operand.type) {
	case AR_EXP_CONSTANT:
		clone->operand.type = AR_EXP_CONSTANT;
		clone->operand.constant = SI_ShallowCloneValue(exp->operand.constant);
		break;
	case AR_EXP_VARIADIC:
		clone->operand.type = exp->operand.type;
		clone->operand.variadic.entity_alias = exp->operand.variadic.entity_alias;
		clone->operand.variadic.entity_alias_idx = exp->operand.variadic.entity_alias_idx;
		break;
	case AR_EXP_PARAM:
		clone->operand.type = AR_EXP_PARAM;
		clone->operand.param_name = exp->operand.param_name;
		break;
	case AR_EXP_BORROW_RECORD:
		clone->operand.type = AR_EXP_BORROW_RECORD;
		break;
	default:
		ASSERT(false);
		break;
	}

	return clone;
}

static AR_ExpNode *_AR_EXP_NewOpNode(uint child_count) {
	AR_ExpNode *node = rm_calloc(1, sizeof(AR_ExpNode));

	node->type           = AR_EXP_OP;
	node->op.children    = rm_calloc (child_count, sizeof (AR_ExpNode *)) ;
	node->op.child_count = child_count;

	return node;
}

static AR_ExpNode *_AR_EXP_CloneOp
(
	const AR_ExpNode *exp
) {
	const char *func_name = exp->op.f->name ;
	bool include_internal = exp->op.f->internal ;
	uint child_count = exp->op.child_count ;
	AR_ExpNode *clone =
		AR_EXP_NewOpNode (func_name, include_internal, child_count) ;

	AR_Func_Clone clone_cb = clone->op.f->callbacks.clone ;
	void *pdata = exp->op.private_data ;
	if (clone_cb != NULL) {
		// clone callback specified, use it to duplicate function's private data
		clone->op.private_data = clone_cb (exp->op.private_data) ;
	}

	// clone child nodes
	for (uint i = 0; i < exp->op.child_count; i++) {
		AR_ExpNode *child = AR_EXP_Clone (exp->op.children[i]) ;
		clone->op.children[i] = child ;
	}

	return clone ;
}

static void _AR_EXP_ValidateArgsCount
(
	AR_FuncDesc *fdesc,
	uint argc
) {
	// Make sure number of arguments is as expected.
	if(fdesc->min_argc > argc) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_FUNCTION_MIN_ARGS, argc,
						  fdesc->name, fdesc->min_argc);
	} else if(fdesc->max_argc < argc) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_FUNCTION_MAX_ARGS, argc,
						  fdesc->name, fdesc->max_argc);
	}
}

AR_ExpNode *AR_EXP_NewOpNode
(
	const char *func_name,
	bool include_internal,
	uint child_count
) {
	// retrieve function
	AR_FuncDesc *func = AR_GetFunc(func_name, include_internal);
	AR_ExpNode  *node = _AR_EXP_NewOpNode(child_count);

	if(!func->internal) _AR_EXP_ValidateArgsCount(func, child_count);

	ASSERT(func != NULL);
	node->op.f = func;

	// add aggregation context as function private data
	if(func->aggregate) {
		// generate aggregation context and store it in node's private data
		ASSERT(func->callbacks.private_data != NULL);
		node->op.private_data = func->callbacks.private_data();
	}

	// mark query as non deterministic if function is non deterministic
	if (unlikely (!func->deterministic)) {
		QueryCtx_SetNonDeterministic () ;
	}

	return node;
}

static inline AR_ExpNode *_AR_EXP_InitializeOperand(AR_OperandNodeType type) {
	AR_ExpNode *node = rm_calloc(1, sizeof(AR_ExpNode));
	node->type = AR_EXP_OPERAND;
	node->operand.type = type;
	return node;
}

AR_ExpNode *AR_EXP_NewVariableOperandNode(const char *alias) {
	AR_ExpNode *node = _AR_EXP_InitializeOperand(AR_EXP_VARIADIC);
	node->operand.variadic.entity_alias = alias;
	node->operand.variadic.entity_alias_idx = IDENTIFIER_NOT_FOUND;

	return node;
}

AR_ExpNode *AR_EXP_NewAttributeAccessNode(AR_ExpNode *entity,
										  const char *attr) {

	ASSERT(attr != NULL);
	ASSERT(entity != NULL);

	// use property index when possible, prop_idx is set to ATTRIBUTE_NOTFOUND
	// if the graph is not aware of it in which case we'll try to resolve
	// the property using its string representation

	GraphContext *gc = QueryCtx_GetGraphCtx();
	SIValue prop_idx = SI_LongVal(ATTRIBUTE_ID_NONE);
	SIValue prop_name = SI_ConstStringVal((char *)attr);
	AttributeID idx = GraphContext_GetAttributeID(gc, attr);

	if(idx != ATTRIBUTE_ID_NONE) prop_idx = SI_LongVal(idx);

	// entity is an expression which should be evaluated to a graph entity
	// attr is the name of the attribute we want to extract from entity
	AR_ExpNode *root = AR_EXP_NewOpNode("property", true, 3);
	root->op.children[0] = entity;
	root->op.children[1] = AR_EXP_NewConstOperandNode(prop_name);
	root->op.children[2] = AR_EXP_NewConstOperandNode(prop_idx);

	return root;
}

AR_ExpNode *AR_EXP_NewConstOperandNode(SIValue constant) {
	AR_ExpNode *node = _AR_EXP_InitializeOperand(AR_EXP_CONSTANT);
	node->operand.constant = constant;
	return node;
}

AR_ExpNode *AR_EXP_NewParameterOperandNode
(
	const char *param_name
) {
	ASSERT(param_name != NULL);

	AR_ExpNode *node = _AR_EXP_InitializeOperand(AR_EXP_PARAM);
	node->operand.param_name = param_name;

	return node;
}

AR_ExpNode *AR_EXP_NewRecordNode() {
	return _AR_EXP_InitializeOperand(AR_EXP_BORROW_RECORD);
}

void AR_SetPrivateData
(
	AR_ExpNode *node, void *pdata
) {
	// validations
	// node must be an operation
	// operation private data must be NULL
	ASSERT(node != NULL);
	ASSERT(node->type == AR_EXP_OP);
	ASSERT(node->op.private_data == NULL);

	node->op.private_data = pdata;
}

// compact tree by evaluating constant expressions
// e.g. MINUS(X) where X is a constant number will be reduced to
// a single node with the value -X
// PLUS(MINUS(A), B) will be reduced to a single constant: B-A
bool AR_EXP_ReduceToScalar
(
	AR_ExpNode *root,
	bool reduce_params,
	SIValue *val
) {
	if(val != NULL) {
		*val = SI_NullVal();
	}

	if(root->type == AR_EXP_OPERAND) {
		// in runtime, parameters are set so they can be evaluated
		if(reduce_params && AR_EXP_IsParameter(root)) {
			SIValue v = AR_EXP_Evaluate(root, NULL);
			if(val != NULL) {
				*val = v;
			}
			return true;
		}
		if(AR_EXP_IsConstant(root)) {
			// Root is already a constant
			if(val != NULL) *val = root->operand.constant;
			return true;
		}
		// Root is variadic, no way to reduce
		return false;
	} else {
		// root represents an operation
		ASSERT(AR_EXP_IsOperation(root));

		// see if we're able to reduce each child of root
		// if so we'll be able to reduce root
		bool reduce_children = true;
		for(int i = 0; i < root->op.child_count; i++) {
			if(!AR_EXP_ReduceToScalar(root->op.children[i], reduce_params, NULL)) {
				// root reduce is not possible, but continue to reduce every reducable child
				reduce_children = false;
			}
		}

		// can't reduce root as one of its children is not a constant
		if(!reduce_children) {
			return false;
		}

		// all child nodes are constants, make sure function is marked as reducible
		if(!root->op.f->reducible) {
			return false;
		}

		// evaluate function
		SIValue v = AR_EXP_Evaluate(root, NULL);
		if(val != NULL) *val = v;
		if(SIValue_IsNull(v)) {
			return false;
		}

		// reduce
		// clear children and function context
		_AR_EXP_FreeOpInternals(root);

		// in-place update, set as constant
		root->type = AR_EXP_OPERAND;
		root->operand.type = AR_EXP_CONSTANT;
		root->operand.constant = v;

		return true;
	}
}

static void _AR_EXP_OpResolveVariables(AR_ExpNode *node, const Record r) {
	ASSERT(node->type == AR_EXP_OP);

	for(uint i = 0; i < node->op.child_count; i++) {
		AR_ExpNode *child = node->op.children[i];
		_AR_EXP_ResolveVariables(child, r);
	}
}

void _AR_EXP_OperandResolveVariables(AR_ExpNode *node, const Record r) {
	ASSERT(node->type == AR_EXP_OPERAND);

	// return if this is not a variadic node
	if(!AR_EXP_IsVariadic(node)) return;

	// see if record contains a value for this variadic
	const char *alias = node->operand.variadic.entity_alias;
	uint idx = Record_GetEntryIdx(r, alias, strlen(alias));
	ASSERT(idx != INVALID_INDEX);
	if(!Record_ContainsEntry(r, idx)) return;

	// replace variadic with constant
	SIValue v = SI_CloneValue(Record_Get(r, idx));
	_AR_EXP_InplaceRepurposeConstant(node, v);
}

static void _AR_EXP_ResolveVariables(AR_ExpNode *root, const Record r) {
	ASSERT(r != NULL);

	if(root == NULL) return;

	switch(root->type) {
		case AR_EXP_OP:
			_AR_EXP_OpResolveVariables(root, r);
			break;
		case AR_EXP_OPERAND:
			_AR_EXP_OperandResolveVariables(root, r);
			break;
		default:
			ASSERT(false && "unknown arithmetic expression node type");
	}
}

void AR_EXP_ResolveVariables(AR_ExpNode *root, const Record r) {
	_AR_EXP_ResolveVariables(root, r);
	AR_EXP_ReduceToScalar(root, true, NULL);
}

static bool _AR_EXP_ValidateInvocation
(
	AR_FuncDesc *fdesc,
	SIValue *argv,
	uint argc
) {
	SIType actual_type;
	SIType expected_type = T_NULL;

	uint expected_types_count = array_len(fdesc->types);
	for(int i = 0; i < argc; i++) {
		actual_type = SI_TYPE(argv[i]);
		/* For a function that accepts a variable number of arguments.
		* the last specified type in fdesc->types is repeatable. */
		if(i < expected_types_count) {
			expected_type = fdesc->types[i];
		}
		if(!(actual_type & expected_type)) {
			Error_SITypeMismatch(argv[i], expected_type);
			return false;
		}
	}

	return true;
}

/* Evaluating an expression tree constructs an array of SIValues.
 * Free all of these values, in case an intermediate node on the tree caused a heap allocation.
 * For example, in the expression:
 * a.first_name + toUpper(a.last_name)
 * the result of toUpper() is allocated within this tree, and will leak if not freed here. */
static inline void _AR_EXP_FreeResultsArray(SIValue *results, int count) {
	for(int i = 0; i < count; i ++) {
		SIValue_Free(results[i]);
	}
	// Large arrays are heap-allocated, so here is where we free it.
	if (count > MAX_ARRAY_SIZE_ON_STACK) {
		rm_free(results);
	}
}

static AR_EXP_Result _AR_EXP_EvaluateFunctionCall
(
	AR_ExpNode *node,
	const Record r,
	SIValue *result
) {
	AR_EXP_Result res = EVAL_OK;

	int child_count = NODE_CHILD_COUNT(node);

	// evaluate each child before evaluating current node
	SIValue *sub_trees = NULL;
	// if array size is above the threshold, we allocate it on the heap (otherwise on stack)
	size_t array_on_stack_size = child_count > MAX_ARRAY_SIZE_ON_STACK ? 0 : child_count;
	SIValue sub_trees_on_stack[array_on_stack_size];
	if (child_count > MAX_ARRAY_SIZE_ON_STACK) {
		sub_trees = rm_malloc(child_count * sizeof(SIValue));
	} else {
		sub_trees = sub_trees_on_stack;
	}
	bool param_found = false;
	for(int child_idx = 0; child_idx < child_count; child_idx++) {
		SIValue v;
		AR_ExpNode *child = NODE_CHILD(node, child_idx);
		res = _AR_EXP_Evaluate(child, r, &v);

		if(res == EVAL_ERR) {
			// encountered an error while evaluating a subtree
			// free all values generated up to this point
			// and propagate the error upwards
			_AR_EXP_FreeResultsArray(sub_trees, child_idx);
			return res;
		}

		param_found |= (res == EVAL_FOUND_PARAM);
		sub_trees[child_idx] = v;
	}

	if(param_found) res = EVAL_FOUND_PARAM;

	// validate before evaluation
	if(!_AR_EXP_ValidateInvocation(node->op.f, sub_trees, child_count)) {
		// the expression tree failed its validations and set an error message
		res = EVAL_ERR;
		goto cleanup;
	}

	// evaluate self
	SIValue v = node->op.f->func(sub_trees, child_count, node->op.private_data);
	ASSERT(node->op.f->aggregate || SI_TYPE(v) & AR_FuncDesc_RetType(node->op.f));
	if(SIValue_IsNull(v) && ErrorCtx_EncounteredError()) {
		// an error was encountered while evaluating this function,
		// and has already been set in the QueryCtx
		// exit with an error
		res = EVAL_ERR;
	}
	if(result) {
		SIValue_Persist(&v);
		*result = v;
	}

cleanup:
	_AR_EXP_FreeResultsArray(sub_trees, node->op.child_count);
	return res;
}

static bool _AR_EXP_UpdateEntityIdx(AR_OperandNode *node, const Record r) {
	if(!r) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_MISSING_RECORD, node->variadic.entity_alias);
		return false;
	}
	int entry_alias_idx = Record_GetEntryIdx(r, node->variadic.entity_alias, strlen(node->variadic.entity_alias));
	if(entry_alias_idx == INVALID_INDEX) {
		// Set the query-level error.
		ErrorCtx_SetError(EMSG_MISSING_VALUE, node->variadic.entity_alias);
		return false;
	} else {
		node->variadic.entity_alias_idx = entry_alias_idx;
		return true;
	}
}

static AR_EXP_Result _AR_EXP_EvaluateVariadic(AR_ExpNode *node, const Record r, SIValue *result) {
	// Make sure entity record index is known.
	if(node->operand.variadic.entity_alias_idx == IDENTIFIER_NOT_FOUND) {
		if(!_AR_EXP_UpdateEntityIdx(&node->operand, r)) return EVAL_ERR;
	}

	int aliasIdx = node->operand.variadic.entity_alias_idx;

	// the value was not created here; share with the caller
	*result = SI_ShareValue(Record_Get(r, aliasIdx));

	return EVAL_OK;
}

static AR_EXP_Result _AR_EXP_EvaluateParam
(
	AR_ExpNode *node,
	SIValue *result
) {
	SIValue *param;
	dict *params = QueryCtx_GetParams();

	if(params) {
		param = HashTableFetchValue(params,
				(unsigned char *)node->operand.param_name);
	}

	if(params == NULL || param == NULL) {
		// set the query-level error
		ErrorCtx_SetError(EMSG_MISSING_PARAMETERS);
		return EVAL_ERR;
	}

	//--------------------------------------------------------------------------
	// in place replacement
	//--------------------------------------------------------------------------

	node->operand.type     = AR_EXP_CONSTANT;
	node->operand.constant = SI_ShareValue(*param);

	*result = node->operand.constant;

	return EVAL_FOUND_PARAM;
}

static inline AR_EXP_Result _AR_EXP_EvaluateBorrowRecord(AR_ExpNode *node, const Record r,
														 SIValue *result) {
	// Wrap the current Record in an SI pointer.
	*result = SI_PtrVal(r);
	return EVAL_OK;
}

// evaluate an expression tree,
// placing the calculated value in 'result'
// and returning whether an error occurred during evaluation
static AR_EXP_Result _AR_EXP_Evaluate
(
	AR_ExpNode *root,
	const Record r,
	SIValue *result
) {
	AR_EXP_Result res = EVAL_OK;

	switch(root->type) {
	case AR_EXP_OP:
		return _AR_EXP_EvaluateFunctionCall(root, r, result);
	case AR_EXP_OPERAND:
		switch(root->operand.type) {
		case AR_EXP_CONSTANT:
			// the value is constant or has been computed elsewhere
			// share with caller
			*result = SI_ShareValue(root->operand.constant);
			return res;
		case AR_EXP_VARIADIC:
			return _AR_EXP_EvaluateVariadic(root, r, result);
		case AR_EXP_PARAM:
			return _AR_EXP_EvaluateParam(root, result);
		case AR_EXP_BORROW_RECORD:
			return _AR_EXP_EvaluateBorrowRecord(root, r, result);
		default:
			ASSERT(false && "Invalid expression type");
		}
	default:
		ASSERT(false && "Unknown expression type");
	}

	return res;
}

SIValue AR_EXP_Evaluate_NoThrow(AR_ExpNode *root, const Record r) {
	SIValue result;
	AR_EXP_Result res = _AR_EXP_Evaluate(root, r, &result);

	if(res == EVAL_ERR) {
		return SI_NullVal(); // Otherwise return NULL; the query-level error will be emitted after cleanup.
	}

	// at least one param node was encountered during evaluation,
	// tree should be parameters free, try reducing the tree
	if(res == EVAL_FOUND_PARAM) {
		AR_EXP_ReduceToScalar(root, true, NULL);
	}

	return result;
}

SIValue AR_EXP_Evaluate
(
	AR_ExpNode *root,
	const Record r
) {
	SIValue result;
	AR_EXP_Result res = _AR_EXP_Evaluate(root, r, &result);

	if(res == EVAL_ERR) {
		ErrorCtx_RaiseRuntimeException(NULL);
		// otherwise return NULL;
		// the query-level error will be emitted after cleanup
		return SI_NullVal();
	}

	// at least one param node was encountered during evaluation,
	// tree should be parameters free, try reducing the tree
	if(res == EVAL_FOUND_PARAM) {
		AR_EXP_ReduceToScalar(root, true, NULL);
	}

	return result;
}

void AR_EXP_Aggregate(AR_ExpNode *root, const Record r) {
	if(AGGREGATION_NODE(root)) {
		AR_EXP_Result res = _AR_EXP_EvaluateFunctionCall(root, r, NULL);
		if(res == EVAL_ERR) {
			ErrorCtx_RaiseRuntimeException(NULL);  // Raise an exception if we're in a run-time context.
			return;
		}
	} else if(AR_EXP_IsOperation(root)) {
		// keep searching for aggregation nodes
		for(int i = 0; i < root->op.child_count; i++) {
			AR_ExpNode *child = root->op.children[i];
			AR_EXP_Aggregate(child, r);
		}
	}
}

// finalize an aggregation node
// returns the node's aggregated value
static SIValue _AR_EXP_FinalizeAggregation
(
	AR_ExpNode *root  // aggregation node
) {
	ASSERT(AGGREGATION_NODE(root));

	AggregateCtx *ctx = root->op.private_data;

	Aggregate_Finalize(root->op.f, ctx);

	SIValue v = Aggregate_GetResult(ctx);
	ASSERT(SI_TYPE(v) & AR_FuncDesc_RetType(root->op.f));

	return v;
}

void _AR_EXP_FinalizeAggregations
(
	AR_ExpNode *root
) {
	//--------------------------------------------------------------------------
	// finalize aggregation node
	//--------------------------------------------------------------------------

	if(AGGREGATION_NODE(root)) {
		SIValue v = _AR_EXP_FinalizeAggregation(root);

		// free node internals
		_AR_EXP_FreeOpInternals(root);

		// replace root with constant node
		root->type             = AR_EXP_OPERAND;
		root->operand.type     = AR_EXP_CONSTANT;
		root->operand.constant = v;

		// return, aggregation nodes cannot contain nested aggregation nodes
		return;
	}

	//--------------------------------------------------------------------------
	// recursively traverse child nodes
	//--------------------------------------------------------------------------

	if(AR_EXP_IsOperation(root)) {
		for(int i = 0; i < NODE_CHILD_COUNT(root); i++) {
			AR_ExpNode *child = NODE_CHILD(root, i);
			_AR_EXP_FinalizeAggregations(child);
		}
	}
}

SIValue AR_EXP_FinalizeAggregations
(
	AR_ExpNode *root,
	const Record r
) {
	ASSERT(root != NULL);

	// incase root is an aggregation node
	// e.g.
	// RETURN a, SUM(a.value)
	// simply return the aggregated value
	//
	// otherwise, root is not an aggregation node
	// e.g.
	// RETURN a, SUM(a.value) / 2
	// finalize the aggregation and eveluate the entire expression

	// note that an aggregation function can not be nested within
	// another aggregation function
	if(AGGREGATION_NODE(root)) {
		return _AR_EXP_FinalizeAggregation(root);
	} else {
		_AR_EXP_FinalizeAggregations(root);
		return AR_EXP_Evaluate(root, r);
	}
}

// set the ith child of parent
// asserts if idx is out of bounds
// returns previous child at `idx` position
AR_ExpNode *AR_EXP_setChild
(
	AR_ExpNode *parent,  // parent node
	AR_ExpNode *child,   // child to add
	uint idx             // child index
) {
	ASSERT (parent       != NULL) ;
	ASSERT (child        != NULL) ;
	ASSERT (child        != parent) ;
	ASSERT (parent->type == AR_EXP_OP) ;
	ASSERT (idx < parent->op.child_count) ;

	AR_ExpNode *prev_child = parent->op.children[idx] ;
	parent->op.children[idx] = child ;

	return prev_child ;
}

// get the ith child of root
// in case root isn't a parent or idx > number of children NULL is returned
AR_ExpNode *AR_EXP_getChild
(
	const AR_ExpNode *root,  // arithmetic expression node
	uint i                   // child index to return
) {
	ASSERT(root != NULL);

	if(root->type == AR_EXP_OP) {
		if(i < root->op.child_count) {
			return root->op.children[i];
		}
	}

	return NULL;
}

// traverse an expression tree and add all entity aliases to a rax
void AR_EXP_CollectEntities
(
	const AR_ExpNode *root,  // expression root node
	rax *aliases             // collected aliases
) {
	ASSERT (root    != NULL) ;
	ASSERT (aliases != NULL) ;

	if(AR_EXP_IsOperation(root)) {
		for(int i = 0; i < root->op.child_count; i++) {
			AR_EXP_CollectEntities(root->op.children[i], aliases);
		}
	} else { // type == AR_EXP_OPERAND
		if(root->operand.type == AR_EXP_VARIADIC) {
			const char *entity = root->operand.variadic.entity_alias;
			raxInsert(aliases, (unsigned char *)entity, strlen(entity), NULL,
					NULL);
		}
	}
}

// collect accessed attribute for a given entity
// e.g. person.first_name + person.last_name
// will populate attributes with both 'first_name' and 'last_name'
// if entity is 'person'
void AR_EXP_CollectAttributes
(
	AR_ExpNode *root,    // expression to collect attributes from
	const char *entity,  // accessed entity
	rax *attributes      // collected attributes
) {
	ASSERT(root       != NULL);
	ASSERT(entity     != NULL);
	ASSERT(attributes != NULL);

	// search for the 'property' operation which indicates attribute access
	// e.g. person.first_name
	if(AR_EXP_IsOperation(root)) {
		if(strcmp(AR_EXP_GetFuncName(root), "property") == 0) {
			// attribute access function
			// property('person', 'first_name')
			AR_ExpNode *child = NODE_CHILD(root, 0);

			// make sure the given entity is being accessed
			if(AR_EXP_IsVariadic(child) &&
			   strcmp(child->operand.variadic.entity_alias, entity) == 0) {

				// get accessed attribute name
				AR_ExpNode *arg = NODE_CHILD(root, 1);

				// expecting arg to be a const string
				ASSERT(AR_EXP_IsConstant(arg));
				ASSERT(SI_TYPE(arg->operand.constant) == T_STRING);

				// collect attribute
				const char *attr = arg->operand.constant.stringval;
				raxInsert(attributes, (unsigned char *)attr, strlen(attr), NULL,
						NULL);
			}
		}

		// continue scanning expression
		for(int i = 0; i < root->op.child_count; i ++) {
			AR_EXP_CollectAttributes(NODE_CHILD(root,i), entity, attributes);
		}
	}
}

// collect each variable operand node expands from root
AR_ExpNode **AR_EXP_CollectVariableOperands
(
	AR_ExpNode *root  // expression root
) {
	ASSERT (root != NULL) ;
	uint i = 0 ;
	AR_ExpNode **nodes = array_new (AR_ExpNode*, 1) ;

	array_append (nodes, root) ;
	while (i < array_len(nodes)) {
		AR_ExpNode *node = nodes[i] ;

		switch (node->type) {
			case AR_EXP_OPERAND:
				switch (node->operand.type) {
					case AR_EXP_PARAM:
					case AR_EXP_CONSTANT:
					case AR_EXP_BORROW_RECORD:
						array_del_fast (nodes, i) ;
						break ;

					case AR_EXP_VARIADIC:
						i++ ;
						break;

					default:
						ASSERT (false) ;
						break ;
				}
				break;

			case AR_EXP_OP:
				for (uint j = 0; j < node->op.child_count; j++) {
					AR_ExpNode *child = node->op.children[j] ;

					if (AR_EXP_IsConstant (child) ||
						AR_EXP_IsParameter (child) ) {
						continue ;
					}

					array_append (nodes, child) ;
				}

				array_del_fast (nodes, i) ;
				break;

			default:
				ASSERT(false);
				break;
		}
	}

	return nodes ;
}

AR_ExpNode **AR_EXP_CollectFunctions
(
	AR_ExpNode *root
) {
	ASSERT (root != NULL) ;

	AR_ExpNode **funcs = array_new (AR_ExpNode*, 0) ;
	AR_ExpNode **nodes = array_new (AR_ExpNode*, 1) ;

	array_append (nodes, root) ;

	while (array_len (nodes) > 0) {
		AR_ExpNode *node = array_pop (nodes) ;

		if (node->type == AR_EXP_OP) {
			array_append (funcs, node) ;

			for (int i = 0; i < NODE_CHILD_COUNT (node); i++) {
				AR_ExpNode *child = NODE_CHILD (node, i) ;
				if (child->type == AR_EXP_OP) {
					array_append (nodes, child) ;
				}
			}
		}
	}

	array_free (nodes) ;

	return funcs ;
}

// collect every aggregation node within expression tree
// returns: dynamically allocated array of AR_ExpNode pointers caller must free
// with array_free()
AR_ExpNode **AR_EXP_CollectAggregations
(
	AR_ExpNode *root  // expression root node
) {
	ASSERT (root != NULL) ;

	AR_ExpNode **nodes        = array_new (AR_ExpNode*, 1) ;
	AR_ExpNode **aggregations = array_new (AR_ExpNode*, 1) ;

	array_append (nodes, root) ;

	while (array_len (nodes) > 0) {
		AR_ExpNode *node = array_pop (nodes) ;

		if (AGGREGATION_NODE (node)) {
			// found an aggregation node, as aggregation functions can not be
			// nested, we can simply continue
			array_append (aggregations, node) ;
			continue ;
		}

		// inspect only operation nodes
		if (node->type == AR_EXP_OP) {
			for (uint i = 0; i < NODE_CHILD_COUNT (node); i++) {
				AR_ExpNode *child = NODE_CHILD (node, i) ;
				if (child->type == AR_EXP_OP) {
					array_append (nodes, child) ;
				}
			}
		}
	}

	array_free (nodes) ;

	return aggregations ;
}

bool AR_EXP_ContainsAggregation
(
	const AR_ExpNode *root
) {
	if (AGGREGATION_NODE (root)) {
		return true;
	}

	if (AR_EXP_IsOperation (root)) {
		for (int i = 0; i < root->op.child_count; i++) {
			AR_ExpNode *child = root->op.children[i] ;
			if (AR_EXP_ContainsAggregation (child)) {
				return true ;
			}
		}
	}

	return false ;
}

bool AR_EXP_ContainsFunc(const AR_ExpNode *root, const char *func) {
	if(root == NULL) return false;
	if(AR_EXP_IsOperation(root)) {
		if(strcasecmp(AR_EXP_GetFuncName(root), func) == 0) return true;
		for(int i = 0; i < root->op.child_count; i++) {
			if(AR_EXP_ContainsFunc(root->op.children[i], func)) return true;
		}
	}
	return false;
}

bool AR_EXP_ContainsVariadic(const AR_ExpNode *root) {
	if(root == NULL) return false;
	if(AR_EXP_IsOperation(root)) {
		for(int i = 0; i < root->op.child_count; i++) {
			if(AR_EXP_ContainsVariadic(root->op.children[i])) return true;
		}
	} else if(AR_EXP_IsVariadic(root)) {
		return true;
	}
	return false;
}

// return type of expression
// e.g. the expression: `1+3` return type is SI_NUMERIC
// e.g. the expression : `ToString(4+3)` return type is T_STRING
SIType AR_EXP_ReturnType
(
	const AR_ExpNode *exp  // expression to query
)
{
	// validation
	ASSERT(exp != NULL);
	ASSERT(exp->type != AR_EXP_UNKNOWN);

	SIType t = T_NULL; // returned type

	if(exp->type == AR_EXP_OP) {
		// expression is a function call
		// get function return type
		t = AR_FuncDesc_RetType(exp->op.f);
	} else {
		// expression is an operand
		if (exp->operand.type == AR_EXP_CONSTANT) {
			t = exp->operand.constant.type;
		}
	}

	return t;
}

bool AR_EXP_ReturnsBoolean
(
	const AR_ExpNode *exp
) {
	ASSERT(exp != NULL && exp->type != AR_EXP_UNKNOWN);

	SIType t = AR_EXP_ReturnType(exp);

	// return true if `t` is either Boolean or NULL
	// in case `exp` is a variable or parameter
	// whether it evaluates to boolean cannot be determined at this point
	return t & T_BOOL || t == T_NULL;
}

void _AR_EXP_ToString
(
	const AR_ExpNode *root,
	char **str,
	size_t *str_size,
	size_t *bytes_written
) {
	// make sure there are at least 64 bytes in str
	if(*str == NULL) {
		*bytes_written = 0;
		*str_size = 128;
		*str = rm_calloc(*str_size, sizeof(char));
	}

	if((*str_size - strlen(*str)) < 64) {
		*str_size += 128;
		*str = rm_realloc(*str, sizeof(char) * *str_size);
	}
	/* Concat Op. */
	if(AR_EXP_IsOperation(root)) {
		/* Binary operation? */
		char binary_op = 0;

		const char *func_name = AR_EXP_GetFuncName(root);
		if(strcmp(func_name, "ADD") == 0) binary_op = '+';
		else if(strcmp(func_name, "SUB") == 0) binary_op = '-';
		else if(strcmp(func_name, "MUL") == 0) binary_op = '*';
		else if(strcmp(func_name, "DIV") == 0)  binary_op = '/';

		if(binary_op) {
			_AR_EXP_ToString(root->op.children[0], str, str_size, bytes_written);

			/* Make sure there are at least 64 bytes in str. */
			if((*str_size - strlen(*str)) < 64) {
				*str_size += 128;
				*str = rm_realloc(*str, sizeof(char) * *str_size);
			}

			*bytes_written += sprintf((*str + *bytes_written), " %c ", binary_op);

			_AR_EXP_ToString(root->op.children[1], str, str_size, bytes_written);
		} else {
			/* Operation isn't necessarily a binary operation, use function call representation. */
			*bytes_written += sprintf((*str + *bytes_written), "%s(", AR_EXP_GetFuncName(root));

			for(int i = 0; i < root->op.child_count ; i++) {
				_AR_EXP_ToString(root->op.children[i], str, str_size, bytes_written);

				/* Make sure there are at least 64 bytes in str. */
				if((*str_size - strlen(*str)) < 64) {
					*str_size += 128;
					*str = rm_realloc(*str, sizeof(char) * *str_size);
				}
				if(i < (root->op.child_count - 1)) {
					*bytes_written += sprintf((*str + *bytes_written), ",");
				}
			}
			*bytes_written += sprintf((*str + *bytes_written), ")");
		}
	} else {
		// Concat Operand node.
		if(root->operand.type == AR_EXP_CONSTANT) {
			SIValue_ToString(root->operand.constant, str, str_size, bytes_written);
		} else {
			*bytes_written += sprintf((*str + *bytes_written), "%s", root->operand.variadic.entity_alias);
		}
	}
}

void AR_EXP_ToString
(
	const AR_ExpNode *root,
	char **str
) {
	size_t str_size = 0;
	size_t bytes_written = 0;
	*str = NULL;
	_AR_EXP_ToString(root, str, &str_size, &bytes_written);
}

inline const char *AR_EXP_GetFuncName(const AR_ExpNode *exp) {
	ASSERT(exp != NULL);
	ASSERT(exp->type == AR_EXP_OP);

	return exp->op.f->name;
}

AR_ExpNode *AR_EXP_Clone
(
	const AR_ExpNode *exp
) {
	if (exp == NULL) {
		return NULL ;
	}

	AR_ExpNode *clone = NULL;

	switch (exp->type) {
		case AR_EXP_OPERAND:
			clone = _AR_EXP_CloneOperand (exp) ;
			break ;

		case AR_EXP_OP:
			clone = _AR_EXP_CloneOp (exp) ;
			break ;

		default:
			ASSERT (false) ;
			break ;
	}

	clone->resolved_name = exp->resolved_name ;

	return clone ;
}

// copies the content of `src` into `dest`
void AR_EXP_Overwrite
(
	AR_ExpNode *dest,      // node whose content will be replaced
	const AR_ExpNode *src  // node to overwrite with
) {
	ASSERT (src  != NULL) ;
	ASSERT (dest != NULL) ;

	// free any dynamically allocated internals within dest
	_AR_EXP_FreeInternals (dest) ;

	// create a deep copy of src
	AR_ExpNode *clone = AR_EXP_Clone (src) ;

	// copy the cloned content into dest's memory location
	*dest = *clone ;

	// free the clone's outer shell without freeing its internals
	// (since those internals now belong to dest)
	rm_free (clone) ;
}

static inline void _AR_EXP_FreeOpInternals
(
	AR_ExpNode *op_node
) {
	if(AGGREGATION_NODE(op_node)) {
		AR_FuncDesc *agg_func = op_node->op.f;

		AggregateCtx *ctx = op_node->op.private_data;
		ASSERT(ctx != NULL);

		Aggregate_Free(agg_func, ctx);
	} else if(op_node->op.f->callbacks.free && op_node->op.private_data) {
		op_node->op.f->callbacks.free(op_node->op.private_data);
	}

	for(int child_idx = 0; child_idx < op_node->op.child_count; child_idx++) {
		AR_EXP_Free(op_node->op.children[child_idx]);
	}

	rm_free(op_node->op.children);
}

static void _AR_EXP_FreeInternals
(
	AR_ExpNode *node
) {
	if (AR_EXP_IsOperation (node)) {
		_AR_EXP_FreeOpInternals (node) ;
	} else if (AR_EXP_IsConstant (node)) {
		SIValue_Free (node->operand.constant) ;
	}
}

inline void AR_EXP_Free
(
	AR_ExpNode *root
) {
	_AR_EXP_FreeInternals (root) ;
	rm_free (root) ;
}

