/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "arithmetic_expression_construct.h"
#include "RG.h"
#include "funcs.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../datatypes/array.h"
#include "../configuration/config.h"
#include "../ast/ast_build_filter_tree.h"

// Forward declaration
static AR_ExpNode *_AR_EXP_FromASTNode(const cypher_astnode_t *expr);
static AR_ExpNode *_AR_ExpNodeFromGraphEntity(const cypher_astnode_t *entity);
static AR_ExpNode *_AR_ExpFromNamedPath(const cypher_astnode_t *path);

static bool __AR_EXP_ContainsNestedAgg(const AR_ExpNode *root, bool in_agg) {
	// Is this an aggregation node?
	bool agg_node = (root->type == AR_EXP_OP && root->op.f->aggregate == true);
	// Aggregation node nested within another aggregation node.
	if(agg_node && in_agg) return true;

	if(root->type == AR_EXP_OP) {
		// Scan child nodes.
		for(int i = 0; i < root->op.child_count; i++) {
			AR_ExpNode *child = root->op.children[i];
			// Cary on `in_agg`.
			if(__AR_EXP_ContainsNestedAgg(child, agg_node | in_agg)) return true;
		}
	}

	return false;
}

/* Return true if expression contains nested calls to aggregation functions
 * e.g. MAX(MIN(1)) */
static bool _AR_EXP_ContainsNestedAgg(const AR_ExpNode *exp) {
	bool in_agg = false;
	return __AR_EXP_ContainsNestedAgg(exp, in_agg);
}

#define OP_COUNT 25
// The OpName array is strictly parallel with the AST_Operator enum.
static const char *OpName[OP_COUNT] = {
	"UNKNOWN", "NULL", "OR", "XOR", "AND", "NOT", "EQ", "NEQ", "LT", "GT", "LE",  "GE",
	"ADD", "SUB", "MUL", "DIV", "MOD", "POW", "CONTAINS", "STARTS WITH",
	"ENDS WITH", "IN", "IS NULL", "IS NOT NULL", "XNOR"
};

static inline const char *_ASTOpToString(AST_Operator op) {
	ASSERT(op < OP_COUNT);
	return OpName[op];
}

static AR_ExpNode *AR_EXP_NewOpNodeFromAST(AST_Operator op, uint child_count) {
	const char *func_name = _ASTOpToString(op);
	return AR_EXP_NewOpNode(func_name, true, child_count);
}

static AR_ExpNode *_AR_EXP_FromApplyExpression(const cypher_astnode_t *expr) {
	AR_ExpNode *op;

	bool                    distinct    =  cypher_ast_apply_operator_get_distinct(expr);
	uint                    arg_count   =  cypher_ast_apply_operator_narguments(expr);
	const cypher_astnode_t  *func_node  =  cypher_ast_apply_operator_get_func_name(expr);
	const char              *func_name  =  cypher_ast_function_name_get_value(func_node);
	bool                    aggregate   =  AR_FuncIsAggregate(func_name);

	op = AR_EXP_NewOpNode(func_name, false, arg_count);

	if(ErrorCtx_EncounteredError()) {
		// no children to free
		op->op.child_count = 0;
		AR_EXP_Free(op);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	if(op->op.f->internal) {
		ErrorCtx_SetError(EMSG_ACCESS_VAR);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	for(unsigned int i = 0; i < arg_count; i ++) {
		const cypher_astnode_t *arg = cypher_ast_apply_operator_get_argument(expr, i);
		// Recursively convert arguments
		op->op.children[i] = _AR_EXP_FromASTNode(arg);
	}

	if(aggregate && distinct) {
		// when we aggregating distinct values
		// for example COUNT(DISTINCT x)
		// we use distinct function
		ASSERT(arg_count == 1);
		AR_ExpNode *distinct = AR_EXP_NewOpNode("distinct", true, arg_count);
		// move x to be child of distinct
		distinct->op.children[0] = op->op.children[0];
		// distinct is child of COUNT
		op->op.children[0] = distinct;
	}

	return op;
}

static AR_ExpNode *_AR_EXP_FromApplyAllExpression(const cypher_astnode_t *expr) {
	// ApplyAll operators use accessors similar to normal Apply operators with the exception
	// that they have no argument accessors - by definition, they have one argument (all/STAR).
	const cypher_astnode_t *func_node = cypher_ast_apply_all_operator_get_func_name(expr);
	const char *func_name = cypher_ast_function_name_get_value(func_node);
	AR_ExpNode *op = AR_EXP_NewOpNode(func_name, false, 1);

	// Introduce a fake child constant so that the function always operates on something.
	op->op.children[0] = AR_EXP_NewConstOperandNode(SI_BoolVal(1));

	return op;
}

static AR_ExpNode *_AR_EXP_FromIdentifierExpression(const cypher_astnode_t *expr) {
	// Identifier referencing another entity
	const char *alias = cypher_ast_identifier_get_name(expr);
	return AR_EXP_NewVariableOperandNode(alias);
}

static AR_ExpNode *_AR_EXP_FromIdentifier(const cypher_astnode_t *expr) {
	AST *ast = QueryCtx_GetAST();
	if(ast == NULL) {
		/* Attempted to access the AST before it has been constructed.
		 * This can occur in scenarios like parameter evaluation:
		 * CYPHER param=[a] MATCH (a) RETURN a */
		ErrorCtx_SetError(EMSG_ACCESS_VAR);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	// check if the identifier is a named path identifier
	AnnotationCtx *named_paths_ctx =
		AST_AnnotationCtxCollection_GetNamedPathsCtx(ast->anot_ctx_collection);

	const cypher_astnode_t *named_path_annotation =
		cypher_astnode_get_annotation(named_paths_ctx, expr);

	// if the identifier is a named path identifier,
	// evaluate the path expression accordingly
	if(named_path_annotation) return _AR_ExpFromNamedPath(named_path_annotation);
	// else, evalute the identifier
	return _AR_EXP_FromIdentifierExpression(expr);
}

static AR_ExpNode *_AR_EXP_FromPropertyExpression(const cypher_astnode_t *expr) {
	// the property expression is constructed of two parts:
	// 1. an expression evaluating to a graph entity, in the future a map type
	// 2. a property name string.
	// examples: a.v, arr[0].v

	//--------------------------------------------------------------------------
	// Extract entity and property name expressions.
	//--------------------------------------------------------------------------

	const cypher_astnode_t *prop_expr = cypher_ast_property_operator_get_expression(expr);
	const cypher_astnode_t *prop_name_node = cypher_ast_property_operator_get_prop_name(expr);
	const char *prop_name = cypher_ast_prop_name_get_value(prop_name_node);

	AR_ExpNode *entity = _AR_EXP_FromASTNode(prop_expr);
	AR_ExpNode *root = AR_EXP_NewAttributeAccessNode(entity, prop_name);

	return root;
}

static SIValue _AR_EXP_FromIntegerString(const char *value_str) {
	char *endptr = NULL;
	errno = 0;
	int64_t l = strtol(value_str, &endptr, 0);
	if(endptr[0] != 0) {
		// Failed to convert integer value; set compile-time error to be raised later.
		ErrorCtx_SetError(EMSG_INVALID_NUMERIC, value_str);
		return SI_NullVal();
	}
	if(errno == ERANGE) {
		ErrorCtx_SetError(EMSG_INTEGER_OVERFLOW, value_str);
		return SI_NullVal();
	}
	SIValue converted = SI_LongVal(l);
	return converted;
}

static AR_ExpNode *_AR_EXP_FromIntegerExpression(const cypher_astnode_t *expr) {
	const char *value_str = cypher_ast_integer_get_valuestr(expr);
	SIValue converted = _AR_EXP_FromIntegerString(value_str);
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromFloatExpression(const cypher_astnode_t *expr) {
	const char *value_str = cypher_ast_float_get_valuestr(expr);
	char *endptr = NULL;
	double d = strtod(value_str, &endptr);
	if(endptr[0] != 0) {
		// Failed to convert integer value; set compile-time error to be raised later.
		ErrorCtx_SetError(EMSG_INVALID_NUMERIC, value_str);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}
	if(errno == ERANGE) {
		ErrorCtx_SetError(EMSG_FLOAT_OVERFLOW, value_str);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}
	SIValue converted = SI_DoubleVal(d);
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromStringExpression(const cypher_astnode_t *expr) {
	const char *value_str = cypher_ast_string_get_value(expr);
	SIValue converted = SI_ConstStringVal((char *)value_str);
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromTrueExpression() {
	SIValue converted = SI_BoolVal(true);
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromFalseExpression() {
	SIValue converted = SI_BoolVal(false);
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromNullExpression() {
	SIValue converted = SI_NullVal();
	return AR_EXP_NewConstOperandNode(converted);
}

static AR_ExpNode *_AR_EXP_FromUnaryOpExpression(const cypher_astnode_t *expr) {
	AR_ExpNode *op = NULL;
	const cypher_astnode_t *arg = cypher_ast_unary_operator_get_argument(expr); // CYPHER_AST_EXPRESSION
	const cypher_operator_t *operator = cypher_ast_unary_operator_get_operator(expr);

	if(operator == CYPHER_OP_UNARY_MINUS) {
		// This expression can be something like -3 or -a.val
		if(cypher_astnode_type(arg) == CYPHER_AST_INTEGER) {
			const char *value_str = cypher_ast_integer_get_valuestr(arg);
			char *minus_str = rm_malloc(sizeof(char) * strlen(value_str) + 2);
			memcpy(minus_str + 1, value_str, strlen(value_str));
			minus_str[0] = '-';
			minus_str[strlen(value_str) + 1] = '\0';
			SIValue converted = _AR_EXP_FromIntegerString(minus_str);
			op = AR_EXP_NewConstOperandNode(converted);
			rm_free(minus_str);
		} else {
			op = AR_EXP_NewOpNodeFromAST(OP_MULT, 2);
			op->op.children[0] = AR_EXP_NewConstOperandNode(SI_LongVal(-1));
			op->op.children[1] = _AR_EXP_FromASTNode(arg);
		}
	} else if(operator == CYPHER_OP_UNARY_PLUS) {
		/* This expression is something like +3 or +a.val.
		 * I think the + can always be safely ignored. */
		op = _AR_EXP_FromASTNode(arg);
	} else if(operator == CYPHER_OP_NOT) {
		op = AR_EXP_NewOpNodeFromAST(OP_NOT, 1);
		op->op.children[0] = _AR_EXP_FromASTNode(arg);
	} else if(operator == CYPHER_OP_IS_NULL) {
		op = AR_EXP_NewOpNodeFromAST(OP_IS_NULL, 1);
		op->op.children[0] = _AR_EXP_FromASTNode(arg);
	} else if(operator == CYPHER_OP_IS_NOT_NULL) {
		op = AR_EXP_NewOpNodeFromAST(OP_IS_NOT_NULL, 1);
		op->op.children[0] = _AR_EXP_FromASTNode(arg);
	} else {
		// No supported operator found.
		ASSERT(false);
	}
	return op;
}

static AR_ExpNode *_AR_EXP_FromBinaryOpExpression(const cypher_astnode_t *expr) {
	const cypher_operator_t *operator = cypher_ast_binary_operator_get_operator(expr);
	AST_Operator operator_enum = AST_ConvertOperatorNode(operator);
	// Arguments are of type CYPHER_AST_EXPRESSION
	AR_ExpNode *op = AR_EXP_NewOpNodeFromAST(operator_enum, 2);
	const cypher_astnode_t *lhs_node = cypher_ast_binary_operator_get_argument1(expr);
	op->op.children[0] = _AR_EXP_FromASTNode(lhs_node);
	const cypher_astnode_t *rhs_node = cypher_ast_binary_operator_get_argument2(expr);
	op->op.children[1] = _AR_EXP_FromASTNode(rhs_node);
	return op;
}

static AR_ExpNode *_AR_EXP_FromComparisonExpression(const cypher_astnode_t *expr) {
	// 1 < 2 = 2 <= 4
	AR_ExpNode *op;
	uint length = cypher_ast_comparison_get_length(expr);
	if(length > 1) {
		op = AR_EXP_NewOpNodeFromAST(OP_AND, length);
		for(uint i = 0; i < length; i++) {
			const cypher_operator_t *operator = cypher_ast_comparison_get_operator(expr, i);
			AST_Operator operator_enum = AST_ConvertOperatorNode(operator);
			const cypher_astnode_t *lhs_node = cypher_ast_comparison_get_argument(expr, i);
			const cypher_astnode_t *rhs_node = cypher_ast_comparison_get_argument(expr, i + 1);

			AR_ExpNode *inner_op = AR_EXP_NewOpNodeFromAST(operator_enum, 2);
			inner_op->op.children[0] = _AR_EXP_FromASTNode(lhs_node);
			inner_op->op.children[1] = _AR_EXP_FromASTNode(rhs_node);
			op->op.children[i] = inner_op;
		}
	} else {
		const cypher_operator_t *operator = cypher_ast_comparison_get_operator(expr, 0);
		AST_Operator operator_enum = AST_ConvertOperatorNode(operator);
		op = AR_EXP_NewOpNodeFromAST(operator_enum, 2);
		const cypher_astnode_t *lhs_node = cypher_ast_comparison_get_argument(expr, 0);
		const cypher_astnode_t *rhs_node = cypher_ast_comparison_get_argument(expr, 1);
		op->op.children[0] = _AR_EXP_FromASTNode(lhs_node);
		op->op.children[1] = _AR_EXP_FromASTNode(rhs_node);
	}
	return op;
}

static AR_ExpNode *_AR_EXP_FromCaseExpression(const cypher_astnode_t *expr) {
	//Determin number of child expressions:
	unsigned int arg_count;
	const cypher_astnode_t *expression = cypher_ast_case_get_expression(expr);
	unsigned int alternatives = cypher_ast_case_nalternatives(expr);

	/* Simple form: 2 * alternatives + default
	 * Generic form: 2 * alternatives + default */
	if(expression) arg_count = 1 + 2 * alternatives + 1;
	else arg_count = 2 * alternatives + 1;

	// Create Expression and child expressions
	AR_ExpNode *op = AR_EXP_NewOpNode("case", true, arg_count);

	// Value to compare against
	int offset = 0;
	if(expression != NULL) {
		op->op.children[offset++] = _AR_EXP_FromASTNode(expression);
	}

	// Alternatives
	for(uint i = 0; i < alternatives; i++) {
		const cypher_astnode_t *predicate = cypher_ast_case_get_predicate(expr, i);
		op->op.children[offset++] = _AR_EXP_FromASTNode(predicate);
		const cypher_astnode_t *value = cypher_ast_case_get_value(expr, i);
		op->op.children[offset++] = _AR_EXP_FromASTNode(value);
	}

	// Default value.
	const cypher_astnode_t *deflt = cypher_ast_case_get_default(expr);
	if(deflt == NULL) {
		// Default not specified, use NULL.
		op->op.children[offset] = AR_EXP_NewConstOperandNode(SI_NullVal());
	} else {
		op->op.children[offset] = _AR_EXP_FromASTNode(deflt);
	}

	return op;
}

static AR_ExpNode *_AR_ExpFromCollectionExpression(const cypher_astnode_t *expr) {
	uint expCount = cypher_ast_collection_length(expr);
	AR_ExpNode *op = AR_EXP_NewOpNode("tolist", true, expCount);
	for(uint i = 0; i < expCount; i ++) {
		const cypher_astnode_t *exp_node = cypher_ast_collection_get(expr, i);
		op->op.children[i] = AR_EXP_FromASTNode(exp_node);
	}
	return op;
}

static AR_ExpNode *_AR_ExpFromMapExpression(const cypher_astnode_t *expr) {
	/* create a new map construction expression
	 * determine number of elements in map
	 * double argument count to accommodate both key and value */
	uint element_count = cypher_ast_map_nentries(expr);
	AR_ExpNode *op = AR_EXP_NewOpNode("tomap", true, element_count * 2);

	// process each key value pair
	for(uint i = 0; i < element_count; i++) {
		const cypher_astnode_t *key_node = cypher_ast_map_get_key(expr, i);
		const char *key = cypher_ast_prop_name_get_value(key_node);
		const cypher_astnode_t *val = cypher_ast_map_get_value(expr, i);
		// this is a bit of an overkill, as key is supposed to be a const string
		op->op.children[i * 2] = AR_EXP_NewConstOperandNode(SI_ConstStringVal((char *)key));
		op->op.children[i * 2 + 1] = AR_EXP_FromASTNode(val);
	}

	return op;
}

static AR_ExpNode *_AR_ExpFromMapProjection(const cypher_astnode_t *expr) {
	// MATCH (n) RETURN n { .name, .age, scores: collect(m.score) }
	// MATCH (n) RETURN n { .* }

	cypher_astnode_type_t t;
	const cypher_astnode_t *identifier = cypher_ast_map_projection_get_expression(expr);
	// Return an error if the identifier is not a string literal, like 5 in:
	// RETURN 5 {v: 'b'}
	if(cypher_astnode_type(identifier) != CYPHER_AST_IDENTIFIER) {
		ErrorCtx_SetError(EMSG_UNHANDLED_TYPE_MAP_PROJECTION);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}
	const char *entity_name = cypher_ast_identifier_get_name(identifier);

	const cypher_astnode_t *selector = NULL;
	unsigned int n_selectors = cypher_ast_map_projection_nselectors(expr);

	AR_ExpNode *tomapOp = NULL;
	AR_ExpNode *propertiesOp = NULL;

	// Count the number of selectors of type CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES, because these are not children of tomap OpNode
	uint allProps_selectors = 0;
	for(uint i = 0; i < n_selectors; i++) {
		selector = cypher_ast_map_projection_get_selector(expr, i);
		t = cypher_astnode_type(selector);
		if(t == CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES) {
			// { .* }
			allProps_selectors++;
		}
	}
	if(allProps_selectors > 0) {
		// { .* }
		// Use properties() to get a map with all properties
		propertiesOp = AR_EXP_NewOpNode("properties", false, 1);
		propertiesOp->op.children[0] = AR_EXP_NewVariableOperandNode(entity_name);

		if(n_selectors == allProps_selectors) {
			return propertiesOp;
		}
	}

	tomapOp = AR_EXP_NewOpNode("tomap", true, (n_selectors - allProps_selectors) * 2);

	uint j = 0;
	for(uint i = 0; i < n_selectors; i++) {
		selector = cypher_ast_map_projection_get_selector(expr, i);

		const char *prop_name = NULL;
		const cypher_astnode_t *prop = NULL;
		t = cypher_astnode_type(selector);

		if(t == CYPHER_AST_MAP_PROJECTION_PROPERTY) {
			// { .name }
			prop = cypher_ast_map_projection_property_get_prop_name(selector);
			prop_name = cypher_ast_prop_name_get_value(prop);
			tomapOp->op.children[j * 2] = AR_EXP_NewConstOperandNode(SI_ConstStringVal((char *)prop_name));
			AR_ExpNode *entity = AR_EXP_NewVariableOperandNode(entity_name);
			tomapOp->op.children[j * 2 + 1] = AR_EXP_NewAttributeAccessNode(entity, prop_name);
			j++;
		} else if(t == CYPHER_AST_MAP_PROJECTION_LITERAL) {
			// { v: n.v }
			prop = cypher_ast_map_projection_literal_get_prop_name(selector);
			prop_name = cypher_ast_prop_name_get_value(prop);
			const cypher_astnode_t *literal_exp =
				cypher_ast_map_projection_literal_get_expression(selector);
			tomapOp->op.children[j * 2] = AR_EXP_NewConstOperandNode(SI_ConstStringVal((char *)prop_name));
			tomapOp->op.children[j * 2 + 1] = AR_EXP_FromASTNode(literal_exp);
			j++;
		} else if(t == CYPHER_AST_MAP_PROJECTION_IDENTIFIER) {
			// { v }
			prop = cypher_ast_map_projection_identifier_get_identifier(selector);
			prop_name = cypher_ast_identifier_get_name(prop);
			tomapOp->op.children[j * 2] = AR_EXP_NewConstOperandNode(SI_ConstStringVal((char *)prop_name));
			tomapOp->op.children[j * 2 + 1] = AR_EXP_NewVariableOperandNode(prop_name);
			j++;
		} else if(t == CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES) {
			continue;
		} else {
			ASSERT("Unexpected AST node type" && false);
		}
	}

	if(propertiesOp) {
		// to support case like: CREATE (a:A {z:1}) RETURN a{.*, .undefinedProp}
		AR_ExpNode *mergemapOp = AR_EXP_NewOpNode("merge_maps", true, 2);
		mergemapOp->op.children[0] = tomapOp;
		mergemapOp->op.children[1] = propertiesOp;
		return mergemapOp;
	} else {
		return tomapOp;
	}
}

static AR_ExpNode *_AR_ExpFromSubscriptExpression(const cypher_astnode_t *expr) {
	AR_ExpNode *op = AR_EXP_NewOpNode("subscript", true, 2);
	const cypher_astnode_t *exp_node = cypher_ast_subscript_operator_get_expression(expr);
	op->op.children[0] = AR_EXP_FromASTNode(exp_node);
	const cypher_astnode_t *subscript_node = cypher_ast_subscript_operator_get_subscript(expr);
	op->op.children[1] = AR_EXP_FromASTNode(subscript_node);
	return op;
}

static AR_ExpNode *_AR_ExpFromSliceExpression(const cypher_astnode_t *expr) {
	AR_ExpNode *op = AR_EXP_NewOpNode("slice", true, 3);
	const cypher_astnode_t *exp_node = cypher_ast_slice_operator_get_expression(expr);
	const cypher_astnode_t *start_node = cypher_ast_slice_operator_get_start(expr);
	const cypher_astnode_t *end_node = cypher_ast_slice_operator_get_end(expr);

	op->op.children[0] = AR_EXP_FromASTNode(exp_node);

	if(start_node) op->op.children[1] = AR_EXP_FromASTNode(start_node);
	else op->op.children[1] = AR_EXP_NewConstOperandNode(SI_LongVal(0));

	if(end_node) op->op.children[2] = AR_EXP_FromASTNode(end_node);
	else op->op.children[2] = AR_EXP_NewConstOperandNode(SI_LongVal(INT32_MAX));

	return op;
}

static AR_ExpNode *_AR_ExpFromNamedPath(const cypher_astnode_t *path) {
	uint path_len = cypher_ast_pattern_path_nelements(path);
	/* The method TO_PATH accepts as its first parameter the ast node which represents the path.
	 * The other parameters are the graph entities (node, edge, path) which the path builder implemented
	 * in TO_PATH requires in order to build a complete path. The order of the evaluated graph entities
	 * is the same order in which they apeare in the AST.*/
	AR_ExpNode *op = AR_EXP_NewOpNode("topath", true, 1 + path_len);
	// Set path AST as first paramerter.
	op->op.children[0] = AR_EXP_NewConstOperandNode(SI_PtrVal((void *)path));
	for(uint i = 0; i < path_len; i ++)
		// Set graph entities as parameters, ordered according to the path AST.
		op->op.children[i + 1] = _AR_EXP_FromASTNode(cypher_ast_pattern_path_get_element(path, i));
	return op;
}

static AR_ExpNode *_AR_ExpFromShortestPath
(
	const cypher_astnode_t *path
) {
	// allShortestPaths is handled separately
	ASSERT(cypher_ast_shortest_path_is_single(path) == true);

	uint path_len = cypher_ast_pattern_path_nelements(path);
	if(path_len != 3) {
		ErrorCtx_SetError(EMSG_SHORTESTPATH_SINGLE_RELATIONSHIP);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	// Retrieve the minimum and maximum number of hops, if specified.
	uint start = 1;
	uint end = EDGE_LENGTH_INF;
	const cypher_astnode_t *edge = cypher_ast_pattern_path_get_element(path, 1);
	const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(edge);
	if(range == NULL) {
		end = 1; // Not a variable-length edge
	} else {
		const cypher_astnode_t *range_start = cypher_ast_range_get_start(range);
		if(range_start) {
			// If specified, the edge's minimum hop value must be 0 or 1
			start = AST_ParseIntegerNode(range_start);
			if(start > 1) {
				ErrorCtx_SetError(EMSG_SHORTESTPATH_MINIMAL_LENGTH);
				return AR_EXP_NewConstOperandNode(SI_NullVal());
			}
		}
		const cypher_astnode_t *range_end = cypher_ast_range_get_end(range);
		if(range_end) end = AST_ParseIntegerNode(range_end);

		if(end != EDGE_LENGTH_INF && end < start) {
			ErrorCtx_SetError(EMSG_SHORTESTPATH_MAX_HOPS);
			return AR_EXP_NewConstOperandNode(SI_NullVal());
		}
	}

	enum cypher_rel_direction dir = cypher_ast_rel_pattern_get_direction(edge);
	if(dir == CYPHER_REL_BIDIRECTIONAL) {
		ErrorCtx_SetError(EMSG_SHORTESTPATH_UNDIRECTED);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	if(cypher_ast_rel_pattern_get_properties(edge)) {
		ErrorCtx_SetError(EMSG_SHORTESTPATH_RELATIONSHIP_FILTERS);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	if(cypher_ast_node_pattern_get_properties(cypher_ast_pattern_path_get_element(path, 0)) ||
	   cypher_ast_node_pattern_get_properties(cypher_ast_pattern_path_get_element(path, 2))) {
		ErrorCtx_SetError(EMSG_SHORTESTPATH_NODE_FILTERS);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	// Collect the IDs of all relationship types
	uint reltype_count = cypher_ast_rel_pattern_nreltypes(edge);
	const char **reltype_names = NULL;
	if(reltype_count > 0) {
		reltype_names = array_new(const char *, reltype_count);
		for(uint i = 0; i < reltype_count; i ++) {
			const char *reltype = cypher_ast_reltype_get_name(cypher_ast_rel_pattern_get_reltype(edge, i));
			array_append(reltype_names, reltype);
		}
	}

	AR_ExpNode *op = AR_EXP_NewOpNode("shortestpath", true, 2);

	// Instantiate a context struct with traversal details.
	ShortestPathCtx *ctx = rm_malloc(sizeof(ShortestPathCtx));
	ctx->R              =  GrB_NULL;
	ctx->minHops        =  start;
	ctx->maxHops        =  end;
	ctx->reltypes       =  NULL;
	ctx->reltype_names  =  reltype_names;
	ctx->reltype_count  =  array_len(reltype_names);
	ctx->free_matrices  =  false;

	AR_SetPrivateData(op, ctx);
	AR_ExpNode *src;
	AR_ExpNode *dest;
	const cypher_astnode_t *ast_src = cypher_ast_pattern_path_get_element(path, 0);
	const cypher_astnode_t *ast_dest = cypher_ast_pattern_path_get_element(path, 2);
	if(dir == CYPHER_REL_OUTBOUND) {
		// Standard traversal
		src = _AR_ExpNodeFromGraphEntity(ast_src);
		dest = _AR_ExpNodeFromGraphEntity(ast_dest);
	} else {
		// Inbound traversal, swap source and dest
		dest = _AR_ExpNodeFromGraphEntity(ast_src);
		src = _AR_ExpNodeFromGraphEntity(ast_dest);
	}
	op->op.children[0] = src;
	op->op.children[1] = dest;

	return op;
}

static AR_ExpNode *_AR_ExpNodeFromGraphEntity(const cypher_astnode_t *entity) {
	const char *alias = AST_ToString(entity);
	return AR_EXP_NewVariableOperandNode(alias);
}

static AR_ExpNode *_AR_ExpNodeFromParameter(const cypher_astnode_t *param) {
	const char *identifier = cypher_ast_parameter_get_name(param);
	return AR_EXP_NewParameterOperandNode(identifier);
}

static AR_ExpNode *_AR_ExpNodeFromComprehensionFunction
(
	const cypher_astnode_t *comp_exp,
	cypher_astnode_type_t type
) {
	// set the appropriate function name according to the node type
	const char *func_name;
	if(type == CYPHER_AST_ANY)         func_name = "ANY";
	else if(type == CYPHER_AST_ALL)    func_name = "ALL";
	else if(type == CYPHER_AST_SINGLE) func_name = "SINGLE";
	else if(type == CYPHER_AST_NONE)   func_name = "NONE";
	else                               func_name = "LIST_COMPREHENSION";

	// using the sample query:
	// WITH [1,2,3] AS arr
	// RETURN [val IN arr WHERE val % 2 = 1 | val * 2] AS comp

	// the comprehension's local variable, WHERE expression, and eval routine
	// do not change for each invocation
	// so are bundled together in the function's context
	ListComprehensionCtx *ctx = rm_malloc(sizeof(ListComprehensionCtx));
	ctx->ft           = NULL;
	ctx->eval_exp     = NULL;
	ctx->local_record = NULL;
	ctx->variable_str = NULL;
	ctx->variable_idx = INVALID_INDEX;

	// retrieve the variable name introduced in this context to iterate over
	// list elements in the above query, this is 'val'
	const cypher_astnode_t *variable_node =
		cypher_ast_list_comprehension_get_identifier(comp_exp);
	ASSERT(cypher_astnode_type(variable_node) == CYPHER_AST_IDENTIFIER);

	// retrieve the variable string for the local variable
	ctx->variable_str = cypher_ast_identifier_get_name(variable_node);

	// the predicate node is the set of WHERE conditions in the comprehension
	// if any
	const cypher_astnode_t *predicate_node =
		cypher_ast_list_comprehension_get_predicate(comp_exp);

	// build a FilterTree to represent this predicate
	if(predicate_node) {
		AST_ConvertFilters(&ctx->ft, predicate_node);
	} else if(type != CYPHER_AST_LIST_COMPREHENSION) {
		// functions like any() and all() must have a predicate node
		ErrorCtx_SetError(EMSG_FUNCTION_REQUIER_PREDICATE, func_name);
		rm_free(ctx);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	// construct the operator node that will generate updated values,
	// if one is provided
	// in the above query, this will be an operation node representing "val * 2"
	// this will always be NULL for comprehensions like any() and all()
	const cypher_astnode_t *eval_node =
		cypher_ast_list_comprehension_get_eval(comp_exp);
	if(eval_node) ctx->eval_exp = _AR_EXP_FromASTNode(eval_node);

	// 'arr' is the list expression
	// note that this value could resolve to an alias, a literal array,
	// a function call, and so on
	const cypher_astnode_t *list_node =
		cypher_ast_list_comprehension_get_expression(comp_exp);
	AR_ExpNode *list = AR_EXP_FromASTNode(list_node);

	// no evaluation expression
	// no filter
	// e.g.
	// RETURN [x IN [1,2,3]] -> [1,2,3]
	// in this case we can simply use the internal array expression
	if(eval_node == NULL && predicate_node == NULL) {
		ASSERT(type == CYPHER_AST_LIST_COMPREHENSION);
		rm_free(ctx);
		return list;
	}

	// build an operation node to represent the list comprehension
	AR_ExpNode *op = AR_EXP_NewOpNode(func_name, true, 2);

	// add the context as function's private data
	AR_SetPrivateData(op, ctx);

	// the list expression is the function's first child
	op->op.children[0] = list;

	// the second child will be a pointer to the Record being evaluated
	op->op.children[1] = AR_EXP_NewRecordNode();

	return op;
}

static AR_ExpNode *_AR_ExpNodeFromReduceFunction
(
	const cypher_astnode_t *reduce_exp
) {
	// reduce(sum = 0, n IN [1,2,3] | sum + n)

	ListReduceCtx *ctx = rm_malloc(sizeof(ListReduceCtx));

	ctx->exp              =  NULL;
	ctx->record           =  NULL;
	ctx->variable         =  NULL;
	ctx->accumulator      =  NULL;
	ctx->variable_idx     =  INVALID_INDEX;
	ctx->accumulator_idx  =  INVALID_INDEX;

	// retrieve the accumulator string `sum`
	const cypher_astnode_t *accumulator_node = cypher_ast_reduce_get_accumulator(reduce_exp);
	ASSERT(cypher_astnode_type(accumulator_node) == CYPHER_AST_IDENTIFIER);
	ctx->accumulator = cypher_ast_identifier_get_name(accumulator_node);

	// retrieve the variable name `n`
	const cypher_astnode_t *identifier_node = cypher_ast_reduce_get_identifier(reduce_exp);
	ASSERT(cypher_astnode_type(identifier_node) == CYPHER_AST_IDENTIFIER);
	ctx->variable = cypher_ast_identifier_get_name(identifier_node);

	//--------------------------------------------------------------------------
	// sub expressions
	//--------------------------------------------------------------------------

	// accumulator init exp
	const cypher_astnode_t *init_exp = cypher_ast_reduce_get_init(reduce_exp);
	AR_ExpNode *init_val = AR_EXP_FromASTNode(init_exp);

	// array exp
	const cypher_astnode_t *exp_node = cypher_ast_reduce_get_expression(reduce_exp);
	AR_ExpNode *list = AR_EXP_FromASTNode(exp_node);

	// eval exp
	const cypher_astnode_t *eval_node = cypher_ast_reduce_get_eval(reduce_exp);
	ctx->exp = AR_EXP_FromASTNode(eval_node);

	// build an operation node to represent the reduction
	AR_ExpNode *reduce = AR_EXP_NewOpNode("REDUCE", true, 3);

	// add the context as function's private data
	AR_SetPrivateData(reduce, ctx);

	//--------------------------------------------------------------------------
	// set expression child nodes
	//--------------------------------------------------------------------------

	// accumulator init value
	reduce->op.children[0] = init_val;

	// list to reduce
	reduce->op.children[1] = list;

	// record
	reduce->op.children[2] = AR_EXP_NewRecordNode();

	return reduce;
}

static AR_ExpNode *_AR_ExpFromLabelsOperatorFunction(const cypher_astnode_t *exp) {
	const char *func_name = "hasLabels";

	// create node expression
	const cypher_astnode_t *node = cypher_ast_labels_operator_get_expression(exp);
	AR_ExpNode *node_exp = _AR_EXP_FromASTNode(node);

	// create labels expression
	uint nlabels = cypher_ast_labels_operator_nlabels(exp);
	SIValue labels = SI_Array(nlabels);
	for(uint i = 0; i < nlabels; i++) {
		const cypher_astnode_t *label = cypher_ast_labels_operator_get_label(exp, i);
		const char *label_str = cypher_ast_label_get_name(label);
		SIArray_Append(&labels, SI_ConstStringVal((char *)label_str));
	}
	AR_ExpNode *labels_exp = AR_EXP_NewConstOperandNode(labels);

	// create func expression
	AR_ExpNode *op = AR_EXP_NewOpNode(func_name, true, 2);

	// set function arguments
	op->op.children[0] = node_exp;
	op->op.children[1] = labels_exp;

	return op;
}

static AR_ExpNode *_AR_EXP_FromASTNode(const cypher_astnode_t *expr) {

	const cypher_astnode_type_t t = cypher_astnode_type(expr);

	// function invocations
	if(t == CYPHER_AST_APPLY_OPERATOR) {
		return _AR_EXP_FromApplyExpression(expr);
		// function invocations with STAR projections
	} else if(t == CYPHER_AST_APPLY_ALL_OPERATOR) {
		return _AR_EXP_FromApplyAllExpression(expr);
		// variables (full nodes and edges, UNWIND artifacts
	} else if(t == CYPHER_AST_IDENTIFIER) {
		return _AR_EXP_FromIdentifier(expr);
		// entity-property pair
	} else if(t == CYPHER_AST_PROPERTY_OPERATOR) {
		return _AR_EXP_FromPropertyExpression(expr);
		// sIValue constant types
	} else if(t == CYPHER_AST_INTEGER) {
		return _AR_EXP_FromIntegerExpression(expr);
	} else if(t == CYPHER_AST_FLOAT) {
		return _AR_EXP_FromFloatExpression(expr);
	} else if(t == CYPHER_AST_STRING) {
		return _AR_EXP_FromStringExpression(expr);
	} else if(t == CYPHER_AST_TRUE) {
		return _AR_EXP_FromTrueExpression();
	} else if(t == CYPHER_AST_FALSE) {
		return _AR_EXP_FromFalseExpression();
	} else if(t == CYPHER_AST_NULL) {
		return _AR_EXP_FromNullExpression();
		// handling for unary operators (-5, +a.val)
	} else if(t == CYPHER_AST_UNARY_OPERATOR) {
		return _AR_EXP_FromUnaryOpExpression(expr);
	} else if(t == CYPHER_AST_BINARY_OPERATOR) {
		return _AR_EXP_FromBinaryOpExpression(expr);
	} else if(t == CYPHER_AST_COMPARISON) {
		return _AR_EXP_FromComparisonExpression(expr);
	} else if(t == CYPHER_AST_CASE) {
		return _AR_EXP_FromCaseExpression(expr);
	} else if(t == CYPHER_AST_COLLECTION) {
		return _AR_ExpFromCollectionExpression(expr);
	} else if(t == CYPHER_AST_SUBSCRIPT_OPERATOR) {
		return _AR_ExpFromSubscriptExpression(expr);
	} else if(t == CYPHER_AST_SLICE_OPERATOR) {
		return _AR_ExpFromSliceExpression(expr);
	} else if(t == CYPHER_AST_NAMED_PATH) {
		return _AR_ExpFromNamedPath(expr);
	} else if(t == CYPHER_AST_SHORTEST_PATH) {
		return _AR_ExpFromShortestPath(expr);
	} else if(t == CYPHER_AST_NODE_PATTERN || t == CYPHER_AST_REL_PATTERN) {
		return _AR_ExpNodeFromGraphEntity(expr);
	} else if(t == CYPHER_AST_PARAMETER) {
		return _AR_ExpNodeFromParameter(expr);
	} else if(t == CYPHER_AST_LIST_COMPREHENSION ||
			  t == CYPHER_AST_ANY ||
			  t == CYPHER_AST_ALL ||
			  t == CYPHER_AST_SINGLE ||
			  t == CYPHER_AST_NONE) {
		return _AR_ExpNodeFromComprehensionFunction(expr, t);
	} else if(t == CYPHER_AST_MAP) {
		return _AR_ExpFromMapExpression(expr);
	} else if(t == CYPHER_AST_MAP_PROJECTION) {
		return _AR_ExpFromMapProjection(expr);
	} else if(t == CYPHER_AST_LABELS_OPERATOR) {
		return _AR_ExpFromLabelsOperatorFunction(expr);
	} else if(t == CYPHER_AST_REDUCE) {
		return _AR_ExpNodeFromReduceFunction(expr);
	} else if(t == CYPHER_AST_PATTERN_PATH || t == CYPHER_AST_PATTERN_COMPREHENSION) {
		// this variable is assign by operitions that created in build_pattern_comprehension_ops.c
		const char *alias = AST_ToString(expr);
		return AR_EXP_NewVariableOperandNode(alias);
	} else {
		/*
		   Unhandled types:
		*/
		Error_UnsupportedASTNodeType(expr);
		return AR_EXP_NewConstOperandNode(SI_NullVal());
	}

	ASSERT(false);
	return NULL;
}

AR_ExpNode *AR_EXP_FromASTNode(const cypher_astnode_t *expr) {
	AR_ExpNode *root = _AR_EXP_FromASTNode(expr);
	AR_EXP_ReduceToScalar(root, false, NULL);

	/* Make sure expression doesn't contain nested aggregation functions
	 * count(max(n.v)) */
	if(_AR_EXP_ContainsNestedAgg(root)) {
		// Set error (compile-time), this error will be raised later on.
		ErrorCtx_SetError(EMSG_NESTED_AGGREGATION);
	}

	return root;
}

