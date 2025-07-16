/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "ast_build_filter_tree.h"
#include "ast_shared.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../arithmetic/arithmetic_expression_construct.h"

// Forward declaration
FT_FilterNode *_FilterNode_FromAST(const cypher_astnode_t *expr);

FT_FilterNode *_CreatePredicateFilterNode(AST_Operator op, const cypher_astnode_t *lhs,
										  const cypher_astnode_t *rhs) {
	return FilterTree_CreatePredicateFilter(op, AR_EXP_FromASTNode(lhs), AR_EXP_FromASTNode(rhs));
}

void _FT_Append(FT_FilterNode **root_ptr, FT_FilterNode *child) {
	ASSERT(child);

	FT_FilterNode *root = *root_ptr;
	// If the tree is uninitialized, its root is the child
	if(root == NULL) {
		*root_ptr = child;
		return;
	}

	if(root->t == FT_N_COND) {
		if(root->cond.left == NULL) {
			FilterTree_AppendLeftChild(root, child);
			return;
		}

		// NOT condition nodes should always have a NULL right child; do not replace it.
		if(root->cond.right == NULL && root->cond.op != OP_NOT) {
			FilterTree_AppendRightChild(root, child);
			return;
		}
	}

	FT_FilterNode *new_root = FilterTree_CreateConditionFilter(OP_AND);
	FilterTree_AppendLeftChild(new_root, root);
	FilterTree_AppendRightChild(new_root, child);
	*root_ptr = new_root;
}

FT_FilterNode *_CreateFilterSubtree(AST_Operator op, const cypher_astnode_t *lhs,
									const cypher_astnode_t *rhs) {
	FT_FilterNode *filter = NULL;
	switch(op) {
	case OP_OR:
	case OP_AND:
	case OP_XOR:
		filter = FilterTree_CreateConditionFilter(op);
		FilterTree_AppendLeftChild(filter, _FilterNode_FromAST(lhs));
		FilterTree_AppendRightChild(filter, _FilterNode_FromAST(rhs));
		return filter;
	case OP_NOT:
		filter = FilterTree_CreateConditionFilter(op);
		FilterTree_AppendLeftChild(filter, _FilterNode_FromAST(lhs));
		FilterTree_AppendRightChild(filter, NULL);
		return filter;
	case OP_EQUAL:
	case OP_NEQUAL:
	case OP_LT:
	case OP_LE:
	case OP_GT:
	case OP_GE:
		return _CreatePredicateFilterNode(op, lhs, rhs);
	default:
		/* Probably an invalid query
		 * e.g. MATCH (u) where u.v NOT NULL RETURN u
		 * this will cause the constructed tree to form an illegal structure
		 * which will be caught later on by `FilterTree_Valid`
		 * and set a compile-time error. */
		return NULL;
	}
}

// AND, OR, XOR (others?)
/* WHERE (condition) AND (condition),
 * WHERE a.val = b.val */
static FT_FilterNode *_convertBinaryOperator(const cypher_astnode_t *op_node) {
	const cypher_operator_t *operator = cypher_ast_binary_operator_get_operator(op_node);
	AST_Operator op = AST_ConvertOperatorNode(operator);
	const cypher_astnode_t *lhs;
	const cypher_astnode_t *rhs;

	switch(op) {
	case OP_OR:
	case OP_AND:
	case OP_XOR:
	case OP_EQUAL:
	case OP_NEQUAL:
	case OP_LT:
	case OP_LE:
	case OP_GT:
	case OP_GE:
		// Arguments are of type CYPHER_AST_EXPRESSION
		lhs = cypher_ast_binary_operator_get_argument1(op_node);
		rhs = cypher_ast_binary_operator_get_argument2(op_node);
		return _CreateFilterSubtree(op, lhs, rhs);
	case OP_NOT:
		ErrorCtx_SetError(EMSG_INVALIDE_NOT_USAGE);
		return NULL;
	default:
		return FilterTree_CreateExpressionFilter(AR_EXP_FromASTNode(op_node));
	}
}

static FT_FilterNode *_convertUnaryOperator(const cypher_astnode_t *op_node) {
	const cypher_operator_t *operator = cypher_ast_unary_operator_get_operator(op_node);
	// Argument is of type CYPHER_AST_EXPRESSION
	const cypher_astnode_t *arg = cypher_ast_unary_operator_get_argument(op_node);
	AST_Operator op = AST_ConvertOperatorNode(operator);
	switch(op) {
		case OP_NOT:
			return _CreateFilterSubtree(op, arg, NULL);
		default:
			return FilterTree_CreateExpressionFilter(AR_EXP_FromASTNode(op_node));
	}
}

static FT_FilterNode *_convertOperator(const cypher_astnode_t *expr) {
	AR_ExpNode *exp = AR_EXP_FromASTNode(expr);
	return FilterTree_CreateExpressionFilter(exp);
}

/* A comparison node contains two arrays - one of operators, and one of expressions.
 * Most comparisons will only have one operator and two expressions, but Cypher
 * allows more complex formulations like "x < y <= z".
 * A comparison takes a form such as "WHERE a.val < y.val". */
static FT_FilterNode *_convertComparison(const cypher_astnode_t *comparison_node) {
	// "x < y <= z"
	uint nelems = cypher_ast_comparison_get_length(comparison_node);
	FT_FilterNode **filters = array_new(FT_FilterNode *, nelems);

	// Create and accumulate simple predicates x < y.
	for(int i = 0; i < nelems; i++) {
		const cypher_operator_t *operator = cypher_ast_comparison_get_operator(comparison_node, i);
		const cypher_astnode_t *lhs = cypher_ast_comparison_get_argument(comparison_node, i);
		const cypher_astnode_t *rhs = cypher_ast_comparison_get_argument(comparison_node, i + 1);

		AST_Operator op = AST_ConvertOperatorNode(operator);
		FT_FilterNode *filter = _CreatePredicateFilterNode(op, lhs, rhs);
		array_append(filters, filter);
	}

	// Reduce by anding.
	while(array_len(filters) > 1) {
		FT_FilterNode *a = array_pop(filters);
		FT_FilterNode *b = array_pop(filters);
		FT_FilterNode *intersec = FilterTree_CreateConditionFilter(OP_AND);

		FilterTree_AppendLeftChild(intersec, a);
		FilterTree_AppendRightChild(intersec, b);

		array_append(filters, intersec);
	}

	FT_FilterNode *root = array_pop(filters);
	array_free(filters);
	return root;
}

static FT_FilterNode *_convertInlinedProperties(const cypher_astnode_t *entity,
												GraphEntityType type) {
	const cypher_astnode_t *props = NULL;
	const cypher_astnode_t *ast_identifer;
	if(type == GETYPE_NODE) {
		props = cypher_ast_node_pattern_get_properties(entity);
	} else { // relation
		props = cypher_ast_rel_pattern_get_properties(entity);
	}

	if(!props) return NULL;

	// Retrieve the entity's alias.
	const char *alias = AST_ToString(entity);

	FT_FilterNode *root = NULL;
	uint nelems = cypher_ast_map_nentries(props);
	for(uint i = 0; i < nelems; i ++) {
		// key is of type CYPHER_AST_PROP_NAME
		const char *prop = cypher_ast_prop_name_get_value(cypher_ast_map_get_key(props, i));

		AR_ExpNode *graph_entity = AR_EXP_NewVariableOperandNode(alias);
		AR_ExpNode *lhs = AR_EXP_NewAttributeAccessNode(graph_entity, prop);
		// val is of type CYPHER_AST_EXPRESSION
		const cypher_astnode_t *val = cypher_ast_map_get_value(props, i);
		AR_ExpNode *rhs = AR_EXP_FromASTNode(val);
		/* TODO In a query like:
		 * "MATCH (r:person {name:"Roi"}) RETURN r"
		 * (note the repeated double quotes) - this creates a variable rather than a scalar.
		 * Can we use this to handle escape characters or something? How does it work? */
		FT_FilterNode *t = FilterTree_CreatePredicateFilter(OP_EQUAL, lhs, rhs);
		_FT_Append(&root, t);
	}
	return root;
}

static FT_FilterNode *_convertPatternPath(const cypher_astnode_t *entity) {
	// Collect aliases specified in pattern.
	const char **aliases = array_new(const char *, 1);
	AST_CollectAliases(&aliases, entity);
	uint alias_count = array_len(aliases);

	/* Create a function call expression
	 * First argument is a pointer to the original AST pattern node.
	 * argument 1..alias_count are the referenced aliases,
	 * required for filter positioning when constructing an execution plan. */
	AR_ExpNode *exp = AR_EXP_NewOpNode("path_filter", true, 1 + alias_count);
	exp->op.children[0] = AR_EXP_NewConstOperandNode(SI_PtrVal((void *)entity));
	for(uint i = 0; i < alias_count; i++) {
		AR_ExpNode *child = AR_EXP_NewVariableOperandNode(aliases[i]);
		exp->op.children[1 + i] = child;
	}
	array_free(aliases);
	return FilterTree_CreateExpressionFilter(exp);
}

FT_FilterNode *_FilterNode_FromAST(const cypher_astnode_t *expr) {
	ASSERT(expr);
	cypher_astnode_type_t type = cypher_astnode_type(expr);

	if(type == CYPHER_AST_COMPARISON) {
		return _convertComparison(expr);
	} else if(type == CYPHER_AST_BINARY_OPERATOR) {
		return _convertBinaryOperator(expr);
	} else if(type == CYPHER_AST_UNARY_OPERATOR) {
		return _convertUnaryOperator(expr);
	} else if(type == CYPHER_AST_PATTERN_PATH) {
		return _convertPatternPath(expr);
	} else {
		return _convertOperator(expr);
	}
}

void _AST_ConvertGraphPatternToFilter(const AST *ast, FT_FilterNode **root,
									  const cypher_astnode_t *pattern) {
	if(!pattern) return;
	FT_FilterNode *ft_node = NULL;
	uint npaths = cypher_ast_pattern_npaths(pattern);
	// Go over each path in the pattern.
	for(uint i = 0; i < npaths; i++) {
		const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, i);
		// Go over each element in the path pattern and check if there is an inline filter.
		uint nelements = cypher_ast_pattern_path_nelements(path);
		// Nodes are in even places.
		for(uint n = 0; n < nelements; n += 2) {
			const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, n);
			ft_node = _convertInlinedProperties(node, GETYPE_NODE);
			if(ft_node) _FT_Append(root, ft_node);
		}
		// Edges are in odd places.
		for(uint e = 1; e < nelements; e += 2) {
			const cypher_astnode_t *edge = cypher_ast_pattern_path_get_element(path, e);
			ft_node = _convertInlinedProperties(edge, GETYPE_EDGE);
			if(ft_node) _FT_Append(root, ft_node);
		}
	}
}

void AST_ConvertFilters(FT_FilterNode **root, const cypher_astnode_t *entity) {
	if(!entity) return;

	cypher_astnode_type_t type = cypher_astnode_type(entity);

	FT_FilterNode *node = NULL;

	if(type == CYPHER_AST_PATTERN_PATH) {
		node = _convertPatternPath(entity);
	} else if(type == CYPHER_AST_COMPARISON) {
		node = _convertComparison(entity);
	} else if(type == CYPHER_AST_BINARY_OPERATOR) {
		node = _convertBinaryOperator(entity);
	} else if(type == CYPHER_AST_UNARY_OPERATOR) {
		node = _convertUnaryOperator(entity);
	} else {
		node = _convertOperator(entity);
	}
	if(node) _FT_Append(root, node);
}

FT_FilterNode *AST_BuildFilterTree(AST *ast) {
	FT_FilterNode *filter_tree = NULL;
	const cypher_astnode_t **match_clauses = AST_GetClauses(ast, CYPHER_AST_MATCH);
	if(match_clauses) {
		uint match_count = array_len(match_clauses);
		for(uint i = 0; i < match_count; i ++) {
			// Optional match clauses are handled separately.
			if(cypher_ast_match_is_optional(match_clauses[i])) continue;
			const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match_clauses[i]);
			_AST_ConvertGraphPatternToFilter(ast, &filter_tree, pattern);
			const cypher_astnode_t *predicate = cypher_ast_match_get_predicate(match_clauses[i]);
			if(predicate) AST_ConvertFilters(&filter_tree, predicate);
		}
		array_free(match_clauses);
	}

	const cypher_astnode_t **with_clauses = AST_GetClauses(ast, CYPHER_AST_WITH);
	if(with_clauses) {
		uint with_count = array_len(with_clauses);
		for(uint i = 0; i < with_count; i ++) {
			const cypher_astnode_t *predicate = cypher_ast_with_get_predicate(with_clauses[i]);
			if(predicate) AST_ConvertFilters(&filter_tree, predicate);
		}
		array_free(with_clauses);
	}

	const cypher_astnode_t **call_clauses = AST_GetClauses(ast, CYPHER_AST_CALL);
	if(call_clauses) {
		uint call_count = array_len(call_clauses);
		for(uint i = 0; i < call_count; i ++) {
			const cypher_astnode_t *where_predicate = cypher_ast_call_get_predicate(call_clauses[i]);
			if(where_predicate) AST_ConvertFilters(&filter_tree, where_predicate);
		}
		array_free(call_clauses);
	}

	if(!FilterTree_Valid(filter_tree)) {
		// Invalid filter tree structure, a compile-time error has been set.
		FilterTree_Free(filter_tree);
		return NULL;
	}

	// Apply De Morgan's laws
	FilterTree_DeMorgan(&filter_tree);

	return filter_tree;
}

FT_FilterNode *AST_BuildFilterTreeFromClauses
(
	const AST *ast,
	const cypher_astnode_t **clauses,
	uint count
) {
	cypher_astnode_type_t type;
	FT_FilterNode *filter_tree = NULL;
	const cypher_astnode_t *predicate = NULL;

	for(uint i = 0; i < count; i ++) {
		type = cypher_astnode_type(clauses[i]);
		if(type == CYPHER_AST_MATCH) {
			const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(clauses[i]);
			_AST_ConvertGraphPatternToFilter(ast, &filter_tree, pattern);
			predicate = cypher_ast_match_get_predicate(clauses[i]);
		} else if(type == CYPHER_AST_WITH) {
			predicate = cypher_ast_with_get_predicate(clauses[i]);
		} else if(type == CYPHER_AST_CALL) {
			predicate = cypher_ast_call_get_predicate(clauses[i]);
		} else {
			ASSERT(false);
		}

		if(predicate) AST_ConvertFilters(&filter_tree, predicate);
	}

	if(!FilterTree_Valid(filter_tree)) {
		// Invalid filter tree structure, a compile-time error has been set.
		FilterTree_Free(filter_tree);
		return NULL;
	}

	// Apply De Morgan's laws
	FilterTree_DeMorgan(&filter_tree);

	return filter_tree;
}

