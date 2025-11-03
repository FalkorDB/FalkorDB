/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ast.h"
#include "util.h"
#include "astnode.h"
#include "ast_shared.h"
#include "../util/arr.h"
#include "ast_visitor.h"
#include "../errors/errors.h"
#include "../util/rax_extensions.h"
#include "../procedures/procedure.h"
#include "../execution_plan/ops/op.h"
#include "../arithmetic/arithmetic_expression.h"

typedef enum {
	NOT_DEFINED = -0x01,  // Yet to be defined
	REGULAR = 0x00,       // UNION
	ALL = 0x01            // UNION ALL
} is_union_all;

typedef struct {
	rax *defined_identifiers;      // identifiers environment
	cypher_astnode_type_t clause;  // top-level clause type
	is_union_all union_all;        // union type (regular or ALL)
} validations_ctx;

// ast validation visitor mappings
// number of ast-node types: _MAX_VT_OFF = sizeof(struct cypher_astnode_vts) / sizeof(struct cypher_astnode_vt *) = 116
static visit validations_mapping[116];

// find an identifier in the environment
static void *_IdentifiersFind
(
	validations_ctx *vctx,  // validations context
	const char *identifier  // identifier to find
) {
	ASSERT(vctx       != NULL);
	ASSERT(identifier != NULL);

	return raxFind(vctx->defined_identifiers, (unsigned char *)identifier,
			strlen(identifier));
}

// introduce an identifier to the environment
static int _IdentifierAdd
(
	validations_ctx *vctx,   // validations context
	const char *identifier,  // identifier to introduce
	void *value              // value to associate with the identifier
) {
	ASSERT(vctx       != NULL);
	ASSERT(identifier != NULL);

	return raxInsert(vctx->defined_identifiers, (unsigned char *)identifier,
			strlen(identifier), value, NULL);
}

// remove an identifier from the environment
static void _IdentifierRemove
(
	validations_ctx *vctx,  // validations context
	const char *identifier  // identifier to remove
) {
	ASSERT(vctx       != NULL);
	ASSERT(identifier != NULL);

	raxRemove(vctx->defined_identifiers, (unsigned char *)identifier,
			strlen(identifier), NULL);
}

// return the number of identifiers in the environment
static int _IdentifierCount
(
	validations_ctx *vctx  // validations context
) {
	ASSERT(vctx != NULL);
	return raxSize(vctx->defined_identifiers);
}

// validate that allShortestPaths is in a supported place
static bool _ValidateAllShortestPaths
(
	const cypher_astnode_t *root // root to validate
) {
	ASSERT(root != NULL);

	cypher_astnode_type_t t = cypher_astnode_type(root);
	// if we found allShortestPaths in invalid parent return true
	if(t == CYPHER_AST_SHORTEST_PATH &&
	   !cypher_ast_shortest_path_is_single(root)) {
		return false;
	}

	// allShortestPaths is invalid in the MATCH predicate
	if(t == CYPHER_AST_MATCH) {
		const cypher_astnode_t *predicate = cypher_ast_match_get_predicate(root);
		return predicate == NULL || _ValidateAllShortestPaths(predicate);
	}

	// recursively traverse all children
	uint nchildren = cypher_astnode_nchildren(root);
	for(uint i = 0; i < nchildren; i ++) {
		const cypher_astnode_t *child = cypher_astnode_get_child(root, i);
		if(!_ValidateAllShortestPaths(child)) {
			return false;
		}
	}

	return true;
}

// validate that shortestPaths is in a supported place
static bool _ValidateShortestPaths
(
	const cypher_astnode_t *root // root to validate
) {
	ASSERT(root != NULL);

	cypher_astnode_type_t t = cypher_astnode_type(root);
	// if we found allShortestPaths in invalid parent return true
	if(t == CYPHER_AST_SHORTEST_PATH &&
	   cypher_ast_shortest_path_is_single(root)) {
		return false;
	}

	// shortestPaths is invalid in the MATCH pattern
	if(t == CYPHER_AST_MATCH) {
		const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(root);
		return _ValidateShortestPaths(pattern);
	}

	if(t == CYPHER_AST_WITH || t == CYPHER_AST_RETURN) {
		return true;
	}

	// recursively traverse all children
	uint nchildren = cypher_astnode_nchildren(root);
	for(uint i = 0; i < nchildren; i ++) {
		const cypher_astnode_t *child = cypher_astnode_get_child(root, i);
		if(!_ValidateShortestPaths(child)) {
			return false;
		}
	}

	return true;
}

// introduce aliases of a WITH clause to the bound vars
// return true if no errors where encountered, false otherwise
static bool _AST_GetWithAliases
(
	const cypher_astnode_t *node,  // ast-node from which to retrieve the aliases
	validations_ctx *vctx
) {
	if(!node || (cypher_astnode_type(node) != CYPHER_AST_WITH)) {
		return false;
	}

	// local env to check for duplicate column names
	rax *local_env = raxNew();

	// traverse the projections
	uint num_with_projections = cypher_ast_with_nprojections(node);
	for(uint i = 0; i < num_with_projections; i ++) {
		const cypher_astnode_t *child = cypher_ast_with_get_projection(node, i);
		const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(child);
		const char *alias;
		if(alias_node) {
			// Retrieve "a" from "WITH [1, 2, 3] as a"
			alias = cypher_ast_identifier_get_name(alias_node);
		} else {
			// Retrieve "a" from "WITH a"
			const cypher_astnode_t *expr = cypher_ast_projection_get_expression(child);
			if(cypher_astnode_type(expr) != CYPHER_AST_IDENTIFIER) {
				ErrorCtx_SetError(EMSG_WITH_PROJ_MISSING_ALIAS);
				raxFree(local_env);
				return false;	
			}
			alias = cypher_ast_identifier_get_name(expr);
		}
		_IdentifierAdd(vctx, alias, NULL);

		// check for duplicate column names (other than internal representation
		// of outer-context variables)
		if(raxTryInsert(local_env, (unsigned char *)alias, strlen(alias), NULL,
			NULL) == 0 &&
			alias[0] != '@') {
				ErrorCtx_SetError(EMSG_SAME_RESULT_COLUMN_NAME);
				raxFree(local_env);
				return false;
		}
	}

	raxFree(local_env);
	return true;
}

// Extract identifiers / aliases from a procedure call.
static void _AST_GetProcCallAliases
(
	const cypher_astnode_t *node,  // ast-node to validate
	validations_ctx *vctx
) {
	// CALL db.labels() yield label
	// CALL db.labels() yield label as l
	ASSERT(node);
	ASSERT(cypher_astnode_type(node) == CYPHER_AST_CALL);

	// traverse projections, collecting the identifiers and expressions
	uint projection_count = cypher_ast_call_nprojections(node);
	for(uint i = 0; i < projection_count; i++) {
		const char *identifier = NULL;
		const cypher_astnode_t *proj_node = cypher_ast_call_get_projection(node, i);
		const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(proj_node);
		if(alias_node) {
			// Alias is given: YIELD label AS l.
			identifier = cypher_ast_identifier_get_name(alias_node);
			_IdentifierAdd(vctx, identifier, NULL);
		}
		// Introduce expression-identifier as well
		// Example: YIELD label --> label is introduced (removed outside of scope)
		const cypher_astnode_t *exp_node = cypher_ast_projection_get_expression(proj_node);
		identifier = cypher_ast_identifier_get_name(exp_node);
		_IdentifierAdd(vctx, identifier, NULL);
	}
}

// make sure multi-hop traversals has length greater than or equal to zero
static AST_Validation _ValidateMultiHopTraversal
(
	const cypher_astnode_t *edge,  // ast-node to validate
	const cypher_astnode_t *range  // varlength range
) {
	int start = 1;
	int end = INT_MAX - 2;

	const cypher_astnode_t *range_start = cypher_ast_range_get_start(range);
	if(range_start) {
		start = AST_ParseIntegerNode(range_start);
	}

	const cypher_astnode_t *range_end = cypher_ast_range_get_end(range);
	if(range_end) {
		end = AST_ParseIntegerNode(range_end);
	}

	// Validate specified range
	if(start > end) {
		ErrorCtx_SetError(EMSG_VAR_LEN_INVALID_RANGE);
		return AST_INVALID;
	}
	
	return AST_VALID;
}

// Verify that MERGE doesn't redeclare bound relations, that one reltype is specified for unbound relations, 
// and that the entity is not a variable length pattern
static AST_Validation _ValidateMergeRelation
(
	const cypher_astnode_t *entity,  // ast-node (rel-pattern)
	validations_ctx *vctx
) {
	// Verify that this is not a variable length relationship
	const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(entity);
	if(range) {
		ErrorCtx_SetError(EMSG_VAR_LEN, "MERGE");
		return AST_INVALID;
	}

	const cypher_astnode_t *identifier = cypher_ast_rel_pattern_get_identifier(entity);
	const char *alias = NULL;
	if(identifier) {
		alias = cypher_ast_identifier_get_name(identifier);
		// verify that we're not redeclaring a bound variable
		if(_IdentifiersFind(vctx, alias) != raxNotFound) {
			ErrorCtx_SetError(EMSG_REDECLARE, "variable", alias, "MERGE");
			return AST_INVALID;
		}
	}

	// Exactly one reltype should be specified for the introduced edge
	uint reltype_count = cypher_ast_rel_pattern_nreltypes(entity);
	if(reltype_count != 1) {
		ErrorCtx_SetError(EMSG_ONE_RELATIONSHIP_TYPE, "MERGE");
		return AST_INVALID;
	}

	// We don't need to validate the MERGE edge's direction, as an undirected edge
	// in MERGE should result a single outgoing edge to be created.

	return AST_VALID;
}

// Verify that MERGE does not introduce labels or properties to bound nodes
static AST_Validation _ValidateMergeNode
(
	const cypher_astnode_t *entity,  // ast-node
	validations_ctx *vctx
) {
	if(_IdentifierCount(vctx) == 0) {
		return AST_VALID;
	}

	const cypher_astnode_t *identifier = cypher_ast_node_pattern_get_identifier(entity);
	if(!identifier) {
		return AST_VALID;
	}

	const char *alias = cypher_ast_identifier_get_name(identifier);
	// if the entity is unaliased or not previously bound, it cannot be redeclared
	if(_IdentifiersFind(vctx, alias) == raxNotFound) {
		return AST_VALID;
	}

	// If the entity is already bound, the MERGE pattern should not introduce labels or properties
	if(cypher_ast_node_pattern_nlabels(entity) ||
	   cypher_ast_node_pattern_get_properties(entity)) {
		ErrorCtx_SetError(EMSG_REDECLARE, "node", alias, "MERGE");
		return AST_INVALID;
	}

	return AST_VALID;
}

// validate that the relation alias of an edge is not bound
static AST_Validation _ValidateCreateRelation
(
	const cypher_astnode_t *entity,  // ast-node
	validations_ctx *vctx
) {
	const cypher_astnode_t *identifier = cypher_ast_rel_pattern_get_identifier(entity);
	if(identifier) {
		const char *alias = cypher_ast_identifier_get_name(identifier);
		if(_IdentifiersFind(vctx, alias) != raxNotFound) {
			ErrorCtx_SetError(EMSG_REDECLARE, "variable", alias, "CREATE");
			return AST_INVALID;
		}
	}

	return AST_VALID;
}

// validate each entity referenced in a single path of a CREATE clause
static AST_Validation _Validate_CREATE_Entities
(
	const cypher_astnode_t *path,  // ast-node (pattern-path)
	validations_ctx *vctx
) {
	uint nelems = cypher_ast_pattern_path_nelements(path);
	 // redeclaration of a node is not allowed only when the path is of length 0
	 // as in: MATCH (a) CREATE (a)
	 // otherwise, using a defined alias of a node is allowed
	 // as in: MATCH (a) CREATE (a)-[:E]->(:B)
	if(nelems == 1) {
		const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
		const cypher_astnode_t *identifier = cypher_ast_node_pattern_get_identifier(node);
		if(identifier) {
			const char *alias = cypher_ast_identifier_get_name(identifier);
			if(_IdentifiersFind(vctx, alias) != raxNotFound) {
				ErrorCtx_SetError(EMSG_REDECLARE, "variable", alias, "CREATE");
				return AST_INVALID;
			}
		}
	}

	return AST_VALID;
}

// make sure an identifier is bound
static AST_Validation _Validate_referred_identifier
(
	validations_ctx *vctx,
	const char *identifier     // identifier to check
) {
	if(_IdentifiersFind(vctx, identifier) == raxNotFound) {
		int len = strlen(identifier);
		ErrorCtx_SetError(EMSG_NOT_DEFINED_LEN, len, identifier);
		return AST_INVALID;
	}

	return AST_VALID;
}

// validate a list comprehension
static VISITOR_STRATEGY _Validate_list_comprehension
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	// we enter ONLY when start=true, so no check is needed

	const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(n);
	const char *identifier = cypher_ast_identifier_get_name(id);
	bool is_new = (_IdentifiersFind(vctx, identifier) == raxNotFound);

	// introduce local identifier if it is not yet introduced
	if(is_new) {
		_IdentifierAdd(vctx, identifier, NULL);
	}

	// Visit expression-children
	// Visit expression
	const cypher_astnode_t *exp = cypher_ast_list_comprehension_get_expression(n);
	if(exp) {
		AST_Visitor_visit(exp, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// Visit predicate
	const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(n);
	if(pred) {
		AST_Visitor_visit(pred, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// Visit eval
	const cypher_astnode_t *eval = cypher_ast_list_comprehension_get_eval(n);
	if(eval) {
		AST_Visitor_visit(eval, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// list comprehension identifier is no longer bound, remove it from bound vars
	// if it was introduced
	if(is_new) _IdentifierRemove(vctx, identifier);

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate a pattern comprehension
static VISITOR_STRATEGY _Validate_pattern_comprehension
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	// we enter ONLY when start=true, so no check is needed

	const cypher_astnode_t *id = cypher_ast_pattern_comprehension_get_identifier(n);
	bool is_new;
	const char *identifier;
	if(id) {
		identifier = cypher_ast_identifier_get_name(id);
		is_new = (_IdentifiersFind(vctx, identifier) == raxNotFound);
	}
	else {
		is_new = false;
	}

	// introduce local identifier if it is not yet introduced
	if(is_new) _IdentifierAdd(vctx, identifier, NULL);

	// Visit expression-children
	// Visit pattern
	const cypher_astnode_t *pattern = cypher_ast_pattern_comprehension_get_pattern(n);
	if(pattern) {
		AST_Visitor_visit(pattern, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// Visit predicate
	const cypher_astnode_t *pred = cypher_ast_pattern_comprehension_get_predicate(n);
	if(pred) {
		AST_Visitor_visit(pred, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// Visit eval
	const cypher_astnode_t *eval = cypher_ast_pattern_comprehension_get_eval(n);
	if(eval) {
		AST_Visitor_visit(eval, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// pattern comprehension identifier is no longer bound, remove it from bound vars
	// if it was introduced
	if(is_new) _IdentifierRemove(vctx, identifier);

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate LOAD CSV clause
static VISITOR_STRATEGY _Validate_load_csv
(
	const cypher_astnode_t *n,  // ast-node (LOAD CSV)
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	const cypher_astnode_t *node = cypher_ast_load_csv_get_identifier(n);
	const char *alias = cypher_ast_identifier_get_name(node);

	_IdentifierAdd(vctx, alias, NULL);

	return VISITOR_CONTINUE;
}

// validate that an identifier is bound
static VISITOR_STRATEGY _Validate_identifier
(
	const cypher_astnode_t *n,  // ast-node (identifier)
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if (!start) {
		return VISITOR_CONTINUE;
	}

	const char *identifier = cypher_ast_identifier_get_name(n);
	if(_Validate_referred_identifier(vctx, identifier) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	return VISITOR_RECURSE;
}

// validate the values of a map
static VISITOR_STRATEGY _Validate_map
(
	const cypher_astnode_t *n,  // ast-node (map)
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	// we enter ONLY when start=true, so no check is needed

	// traverse the entries of the map
	uint nentries = cypher_ast_map_nentries(n);
	for (uint i = 0; i < nentries; i++) {
		const cypher_astnode_t *exp = cypher_ast_map_get_value(n, i);
		AST_Visitor_visit(exp, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate a projection
static VISITOR_STRATEGY _Validate_projection
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	// we enter ONLY when start=true, so no check is needed

	const cypher_astnode_t *exp = cypher_ast_projection_get_expression(n);
	AST_Visitor_visit(exp, visitor);
	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate a function-call
static AST_Validation _ValidateFunctionCall
(
	const char *funcName,    // function name
	bool include_aggregates  // are aggregations allowed
) {
	// check existence of the function-name
	if(!AR_FuncExists(funcName)) {
		ErrorCtx_SetError(EMSG_UNKNOWN_FUNCTION, funcName);
		return AST_INVALID;
	}

	if(!include_aggregates && AR_FuncIsAggregate(funcName)) {
		// Provide a unique error for using aggregate functions from inappropriate contexts
		ErrorCtx_SetError(EMSG_INVALID_USE_OF_AGGREGATION_FUNCTION, funcName);
		return AST_INVALID;
	}

	return AST_VALID;
}

// validate an apply-all operator
static VISITOR_STRATEGY _Validate_apply_all_operator
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	// Working with a function call that has * as its argument.
	const cypher_astnode_t *func = cypher_ast_apply_all_operator_get_func_name(n);
	const char *func_name = cypher_ast_function_name_get_value(func);

	// Verify that this is a COUNT call.
	if(strcasecmp(func_name, "COUNT")) {
		ErrorCtx_SetError(EMSG_INVALID_USAGE_OF_STAR_PARAMETER);
		return VISITOR_BREAK;
	}

	// Verify that DISTINCT is not specified.
	if(cypher_ast_apply_all_operator_get_distinct(n)) {
		// TODO consider opening a parser error, this construction is invalid in Neo's parser.
		ErrorCtx_SetError(EMSG_INVALID_USAGE_OF_DISTINCT_STAR_PARAMETER);
		return VISITOR_BREAK;
	}

	return VISITOR_RECURSE;
}

// validate an apply operator
static VISITOR_STRATEGY _Validate_apply_operator
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	// Collect the function name.
	const cypher_astnode_t *func = cypher_ast_apply_operator_get_func_name(n);
	const char *func_name = cypher_ast_function_name_get_value(func);
	if(_ValidateFunctionCall(func_name, (vctx->clause == CYPHER_AST_WITH ||
										vctx->clause == CYPHER_AST_RETURN)) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	return VISITOR_RECURSE;
}

// validate reduce
static VISITOR_STRATEGY _Validate_reduce
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	cypher_astnode_type_t orig_clause = vctx->clause;
	// change clause type of vctx so that function-validation will work properly
	// (include-aggregations should be set to false)
	vctx->clause = CYPHER_AST_REDUCE;

	// A reduce call has an accumulator and a local list variable that should
	// only be accessed within its scope;
	// do not leave them in the identifiers map
	// example: reduce(sum=0, n in [1,2] | sum+n)
	//  the reduce function is composed of 5 components:
	//     1. accumulator                  `sum`
	//     2. accumulator init expression  `0`
	//     3. list expression              `[1,2,3]`
	//     4. variable                     `n`
	//     5. eval expression              `sum + n`
	
	// make sure that the init expression is a known var or valid exp.
	const cypher_astnode_t *init_node = cypher_ast_reduce_get_init(n);
	if(cypher_astnode_type(init_node) == CYPHER_AST_IDENTIFIER) {
		// check if the variable has already been introduced
		const char *var_str = cypher_ast_identifier_get_name(init_node);
		if(_IdentifiersFind(vctx, var_str) == raxNotFound) {
			ErrorCtx_SetError(EMSG_NOT_DEFINED, var_str);
			return VISITOR_BREAK;
		}
	}
	else {
		AST_Visitor_visit(init_node, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// make sure that the list expression is a list (or list comprehension) or an 
	// alias of an existing one.
	const cypher_astnode_t *list_var = cypher_ast_reduce_get_expression(n);
	if(cypher_astnode_type(list_var) == CYPHER_AST_IDENTIFIER) {
		const char *list_var_str = cypher_ast_identifier_get_name(list_var);
		if(_IdentifiersFind(vctx, list_var_str) == raxNotFound) {
			ErrorCtx_SetError(EMSG_NOT_DEFINED, list_var_str);
			return VISITOR_BREAK;
		}
	}

	// Visit the list expression (no need to introduce local vars)
	AST_Visitor_visit(list_var, visitor);
	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	// make sure that the eval-expression exists
	const cypher_astnode_t *eval_node = cypher_ast_reduce_get_eval(n);
	if(!eval_node) {
		ErrorCtx_SetError(EMSG_MISSING_EVAL_EXP_IN_REDUCE);
		return VISITOR_BREAK;
	}

	// If accumulator is already in the environment, don't reintroduce it
	const cypher_astnode_t *accum_node =cypher_ast_reduce_get_accumulator(n);
	const char *accum_str = cypher_ast_identifier_get_name(accum_node);
	bool introduce_accum = (_IdentifiersFind(vctx, accum_str) == raxNotFound);
	if(introduce_accum) _IdentifierAdd(vctx, accum_str, NULL);

	// same for the list var
	const cypher_astnode_t *list_var_node =cypher_ast_reduce_get_identifier(n);
	const char *list_var_str = cypher_ast_identifier_get_name(list_var_node);
	bool introduce_list_var = (_IdentifiersFind(vctx, list_var_str) == raxNotFound);
	if(introduce_list_var) _IdentifierAdd(vctx, list_var_str, NULL);
	
	// visit eval expression
	const cypher_astnode_t *eval_exp = cypher_ast_reduce_get_eval(n);
	AST_Visitor_visit(eval_exp, visitor);
	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	// change clause type back
	vctx->clause = orig_clause;

	// Remove local vars\aliases if introduced
	if(introduce_accum) _IdentifierRemove(vctx, accum_str);
	if(introduce_list_var) _IdentifierRemove(vctx, list_var_str);

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate the property maps used in node/edge patterns in MATCH, and CREATE clauses
static AST_Validation _ValidateInlinedProperties
(
	const cypher_astnode_t *props  // ast-node representing the map
) {
	if(props == NULL) {
		return AST_VALID;
	}

	// emit an error if the properties are not presented as a map, as in:
	// MATCH (p {invalid_property_construction}) RETURN p
	if(cypher_astnode_type(props) != CYPHER_AST_MAP) {
		ErrorCtx_SetError(EMSG_UNHANDLED_TYPE_INLINE_PROPERTIES);
		return AST_INVALID;
	}

	// traverse map entries
	uint prop_count = cypher_ast_map_nentries(props);
	for(uint i = 0; i < prop_count; i++) {
		const cypher_astnode_t *prop_val = cypher_ast_map_get_value(props, i);
		const cypher_astnode_t **patterns = AST_GetTypedNodes(prop_val, CYPHER_AST_PATTERN_PATH);
		uint patterns_count = array_len(patterns);
		array_free(patterns);
		if(patterns_count > 0) {
			// encountered query of the form
			// MATCH (a {prop: ()-[]->()}) RETURN a
			ErrorCtx_SetError(EMSG_UNHANDLED_TYPE_INLINE_PROPERTIES);
			return AST_INVALID;
		}
	}

	return AST_VALID;
}

// validate a relation-pattern
static VISITOR_STRATEGY _Validate_rel_pattern
(
	const cypher_astnode_t *n,  // ast-node (rel-pattern)s
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(n);
	if(vctx->clause == CYPHER_AST_CREATE) {
		// validate that the relation alias is not bound
		if(_ValidateCreateRelation(n, vctx) == AST_INVALID) {
			return VISITOR_BREAK;
		}

		// Validate that each relation has exactly one type
		uint reltype_count = cypher_ast_rel_pattern_nreltypes(n);
		if(reltype_count != 1) {
			ErrorCtx_SetError(EMSG_ONE_RELATIONSHIP_TYPE, "CREATE");
			return VISITOR_BREAK;
		}

		// Validate that each relation being created is directed
		if(cypher_ast_rel_pattern_get_direction(n) == CYPHER_REL_BIDIRECTIONAL) {
			ErrorCtx_SetError(EMSG_CREATE_DIRECTED_RELATIONSHIP);
			return VISITOR_BREAK;
		}

		// Validate that each relation being created is not variable length relationship
		if(range) {
			ErrorCtx_SetError(EMSG_VAR_LEN, "CREATE");
			return VISITOR_BREAK;
		}
	}

	if(_ValidateInlinedProperties(cypher_ast_rel_pattern_get_properties(n)) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	if(vctx->clause == CYPHER_AST_MERGE &&
		_ValidateMergeRelation(n, vctx) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	const cypher_astnode_t *alias_node = cypher_ast_rel_pattern_get_identifier(n);
	if(!alias_node && !range) {
		return VISITOR_RECURSE; // Skip unaliased, single-hop entities.
	}

	// If this is a multi-hop traversal, validate it accordingly
	if(range && _ValidateMultiHopTraversal(n, range) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	if(alias_node) {
		const char *alias = cypher_ast_identifier_get_name(alias_node);
		void *alias_type = _IdentifiersFind(vctx, alias);
		if(alias_type == raxNotFound) {
			_IdentifierAdd(vctx, alias, (void*)T_EDGE);
			return VISITOR_RECURSE;
		}
			
		if(alias_type != (void *)T_EDGE && alias_type != NULL) {
			ErrorCtx_SetError(EMSG_SAME_ALIAS_NODE_RELATIONSHIP, alias);
			return VISITOR_BREAK;
		}

		if(vctx->clause == CYPHER_AST_MATCH && alias_type != NULL) {
			ErrorCtx_SetError(EMSG_SAME_ALIAS_MULTIPLE_PATTERNS, alias);
			return VISITOR_BREAK;
		}
	}

	return VISITOR_RECURSE;
}

// validate a node-pattern expression
static VISITOR_STRATEGY _Validate_node_pattern
(
	const cypher_astnode_t *n,  // ast-node (node-pattern)
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	if(_ValidateInlinedProperties(cypher_ast_node_pattern_get_properties(n)) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	const cypher_astnode_t *alias_node = cypher_ast_node_pattern_get_identifier(n);
	if(!alias_node) {
		return VISITOR_RECURSE;
	}

	const char *alias = cypher_ast_identifier_get_name(alias_node);
	if(vctx->clause == CYPHER_AST_MERGE) {
		if(_ValidateMergeNode(n, vctx) == AST_INVALID) {
			return VISITOR_BREAK;
		}
	} else {
		void *alias_type = _IdentifiersFind(vctx, alias);
		if(alias_type != raxNotFound && alias_type != NULL && alias_type != (void *)T_NODE) {
			ErrorCtx_SetError(EMSG_SAME_ALIAS_NODE_RELATIONSHIP, alias);
			return VISITOR_BREAK;
		}
	}
	_IdentifierAdd(vctx, alias, (void*)T_NODE);

	return VISITOR_RECURSE;
}

// validate a shortest-path expression
static VISITOR_STRATEGY _Validate_shortest_path
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	if(cypher_ast_shortest_path_is_single(n)) {
		const cypher_astnode_t *path = cypher_ast_shortest_path_get_path(n);
		int elements = cypher_ast_pattern_path_nelements(path);
		const cypher_astnode_t *start = cypher_ast_node_pattern_get_identifier(cypher_ast_pattern_path_get_element(path, 0));
		const cypher_astnode_t *end = cypher_ast_node_pattern_get_identifier(cypher_ast_pattern_path_get_element(path, elements - 1));
		if(start == NULL || end == NULL) {
			ErrorCtx_SetError(EMSG_SHORTESTPATH_BOUND_NODES);
			return VISITOR_BREAK;
		}
		const char *start_id = cypher_ast_identifier_get_name(start);
		const char *end_id = cypher_ast_identifier_get_name(end);
		if(_IdentifiersFind(vctx, start_id) == raxNotFound ||
		   _IdentifiersFind(vctx, end_id) == raxNotFound) {
			ErrorCtx_SetError(EMSG_SHORTESTPATH_BOUND_NODES);
			return VISITOR_BREAK;
		}
		return VISITOR_RECURSE;
	} else {
		// MATCH (a), (b), p = allShortestPaths((a)-[*2..]->(b)) RETURN p
		// validate rel pattern range doesn't contains a minimum > 1
		const cypher_astnode_t **ranges = AST_GetTypedNodes(n, CYPHER_AST_RANGE);
		int range_count = array_len(ranges);
		for(int i = 0; i < range_count; i++) {
			long min_hops = 1;
			const cypher_astnode_t *r = ranges[i];
			const cypher_astnode_t *start = cypher_ast_range_get_start(r);
			if(start) min_hops = AST_ParseIntegerNode(start);
			if(min_hops != 1) {
				ErrorCtx_SetError(EMSG_ALLSHORTESTPATH_MINIMAL_LENGTH);
				break;
			}
		}
		array_free(ranges);
	}

	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	return VISITOR_RECURSE;
}

// validate a named path
static VISITOR_STRATEGY _Validate_named_path
(
	const cypher_astnode_t *n,  // ast-node (named path)
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	// introduce identifiers to bound variables environment
	const cypher_astnode_t *alias_node = cypher_ast_named_path_get_identifier(n);
	const char *alias = cypher_ast_identifier_get_name(alias_node);
	_IdentifierAdd(vctx, alias, NULL);

	return VISITOR_RECURSE;
}

// validate limit and skip modifiers
static AST_Validation _Validate_LIMIT_SKIP_Modifiers
(
	const cypher_astnode_t *limit,  // limit ast-node
	const cypher_astnode_t *skip    // skip ast-node
) {
	if(limit) {
		// Handle non-integer or non parameter types specified as LIMIT value
		// The value validation of integer node or parameter node is done in run time evaluation.
		if(cypher_astnode_type(limit) != CYPHER_AST_INTEGER &&
			cypher_astnode_type(limit) != CYPHER_AST_PARAMETER) {
			ErrorCtx_SetError(EMSG_LIMIT_MUST_BE_NON_NEGATIVE);
			return AST_INVALID;
		}
	}

	if(skip) {
		// Handle non-integer or non parameter types specified as skip value
		// The value validation of integer node or parameter node is done in run time evaluation.
		if(cypher_astnode_type(skip) != CYPHER_AST_INTEGER &&
			cypher_astnode_type(skip) != CYPHER_AST_PARAMETER) {
			ErrorCtx_SetError(EMSG_SKIP_MUST_BE_NON_NEGATIVE);
			return AST_INVALID;
		}
	}

	return AST_VALID;
}

// validate UNION clauses
static AST_Validation _ValidateUnion_Clauses
(
	const AST *ast  // ast-node
) {
	AST_Validation res = AST_VALID;

	uint *union_indices = AST_GetClauseIndices(ast, CYPHER_AST_UNION);
	uint union_clause_count = array_len(union_indices);
	array_free(union_indices);

	if(union_clause_count != 0) {
		// Require all RETURN clauses to perform the exact same projection
		uint *return_indices = AST_GetClauseIndices(ast, CYPHER_AST_RETURN);
		uint return_clause_count = array_len(return_indices);

		// We should have one more RETURN clauses than we have UNION clauses.
		if(return_clause_count != union_clause_count + 1) {
			ErrorCtx_SetError(EMSG_UNION_MISSING_RETURNS, union_clause_count,
							return_clause_count);
			array_free(return_indices);
			return AST_INVALID;
		}

		const cypher_astnode_t *return_clause = cypher_ast_query_get_clause(ast->root, return_indices[0]);
		uint proj_count = cypher_ast_return_nprojections(return_clause);
		const char *projections[proj_count];

		for(uint j = 0; j < proj_count; j++) {
			const cypher_astnode_t *proj = cypher_ast_return_get_projection(return_clause, j);
			const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(proj);
			if(alias_node == NULL)  {
				// The projection was not aliased, so the projection itself must be an identifier.
				alias_node = cypher_ast_projection_get_expression(proj);
				ASSERT(cypher_astnode_type(alias_node) == CYPHER_AST_IDENTIFIER);
			}
			const char *alias = cypher_ast_identifier_get_name(alias_node);
			projections[j] = alias;
		}

		for(uint i = 1; i < return_clause_count; i++) {
			return_clause = cypher_ast_query_get_clause(ast->root, return_indices[i]);
			if(proj_count != cypher_ast_return_nprojections(return_clause)) {
				ErrorCtx_SetError(EMSG_UNION_MISMATCHED_RETURNS);
				res = AST_INVALID;
				goto cleanup;
			}

			for(uint j = 0; j < proj_count; j++) {
				const cypher_astnode_t *proj = cypher_ast_return_get_projection(return_clause, j);
				const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(proj);
				if(alias_node == NULL)  {
					// The projection was not aliased, so the projection itself must be an identifier.
					alias_node = cypher_ast_projection_get_expression(proj);
					ASSERT(cypher_astnode_type(alias_node) == CYPHER_AST_IDENTIFIER);
				}
				const char *alias = cypher_ast_identifier_get_name(alias_node);
				if(strcmp(projections[j], alias) != 0) {
					ErrorCtx_SetError(EMSG_UNION_MISMATCHED_RETURNS);
					res = AST_INVALID;
					goto cleanup;
				}
			}
		}

	cleanup:
		array_free(return_indices);
		if(res == AST_INVALID) {
			return res;
		}
	}

	// validate union clauses of subqueries
	uint *call_subquery_indices = AST_GetClauseIndices(ast,
		CYPHER_AST_CALL_SUBQUERY);
	uint n_subqueries = array_len(call_subquery_indices);

	for(uint i = 0; i < n_subqueries; i++) {
		AST subquery_ast = {
			.root = cypher_ast_call_subquery_get_query(
				cypher_ast_query_get_clause(ast->root,
					call_subquery_indices[i])),
		};

		if(_ValidateUnion_Clauses(&subquery_ast) == AST_INVALID) {
			res = AST_INVALID;
			break;
		}
	}
	array_free(call_subquery_indices);

	return res;
}

// validate a CALL clause
static VISITOR_STRATEGY _Validate_CALL_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(start) {
		vctx->clause = cypher_astnode_type(n);
		// introduce aliases in the clause to the bounded vars environment
		_AST_GetProcCallAliases(n, vctx);

		/* Make sure procedure calls are valid:
		* 1. procedure exists
		* 2. number of arguments to procedure is as expected
		* 3. yield refers to procedure output */

		ProcedureCtx *proc = NULL;
		rax *rax = NULL;

		// Make sure procedure exists.
		const char *proc_name = cypher_ast_proc_name_get_value(cypher_ast_call_get_proc_name(n));
		proc = Proc_Get(proc_name);

		if(proc == NULL) {
			ErrorCtx_SetError(EMSG_PROCEDURE_NOT_REGISTERED, proc_name);
			goto cleanup;
		}

		// Validate num of arguments.
		if(proc->argc != PROCEDURE_VARIABLE_ARG_COUNT) {
			unsigned int given_arg_count = cypher_ast_call_narguments(n);
			if(Procedure_Argc(proc) != given_arg_count) {
				ErrorCtx_SetError(EMSG_PROCEDURE_INVALID_ARGUMENTS, proc_name, proc->argc,
									given_arg_count);
				goto cleanup;
			}
		}

		rax = raxNew();

		// validate projections
		uint proj_count = cypher_ast_call_nprojections(n);
		// collect call projections
		for(uint j = 0; j < proj_count; j++) {
			const cypher_astnode_t *proj = cypher_ast_call_get_projection(n, j);
			const cypher_astnode_t *ast_exp = cypher_ast_projection_get_expression(proj);
			ASSERT(cypher_astnode_type(ast_exp) == CYPHER_AST_IDENTIFIER);
			const char *identifier = cypher_ast_identifier_get_name(ast_exp);

			// make sure each yield output is mentioned only once
			if(!raxInsert(rax, (unsigned char*)identifier, strlen(identifier), NULL, NULL)) {
				ErrorCtx_SetError(EMSG_VAIABLE_ALREADY_DECLARED, identifier);
				goto cleanup;
			}

			// make sure procedure is aware of output
			if(!Procedure_ContainsOutput(proc, identifier)) {
				ErrorCtx_SetError(EMSG_PROCEDURE_INVALID_OUTPUT,
									proc_name, identifier);
				goto cleanup;
			}
		}

cleanup:
		if(proc) {
			Proc_Free(proc);
		}
		if(rax) {
			raxFree(rax);
		}
		return !ErrorCtx_EncounteredError() ? VISITOR_RECURSE : VISITOR_BREAK;
	}

	// end handling

	uint proj_count = cypher_ast_call_nprojections(n);
	// remove expression identifiers from bound vars if an alias exists
	for(uint j = 0; j < proj_count; j++) {
		const cypher_astnode_t *proj = cypher_ast_call_get_projection(n, j);
		const cypher_astnode_t *ast_exp = cypher_ast_projection_get_expression(proj);
		ASSERT(cypher_astnode_type(ast_exp) == CYPHER_AST_IDENTIFIER);
		const char *identifier = cypher_ast_identifier_get_name(ast_exp);
		if(cypher_ast_projection_get_alias(proj)){
			_IdentifierRemove(vctx, identifier);
		}
	}

	return VISITOR_CONTINUE;
}

// validates a leading `WITH` clause of a subquery
static bool _ValidateCallInitialWith
(
	const cypher_astnode_t *with_clause,  // `WITH` clause to validate
	validations_ctx *vctx                 // validation context
) {
	// in case WITH contains *
	// skip this test, wait for second pass (after * AST rewrite)
	if (!cypher_ast_with_has_include_existing (with_clause)) {
		uint n = cypher_ast_with_nprojections (with_clause);
		for (uint i = 0; i < n; i++) {
			const cypher_astnode_t *curr_proj =
				cypher_ast_with_get_projection (with_clause, i) ;
			const cypher_astnode_t *exp =
				cypher_ast_projection_get_expression (curr_proj) ;
			const cypher_astnode_type_t t = cypher_astnode_type (exp) ;

			// special case (empty star import)
			// CALL { WITH * ... }
			if (unlikely (t == CYPHER_AST_NULL)) {
				continue ;
			}

			// fail if WITH expression isn't a simple import expression:
			// WITH a, b, c...
			if (t != CYPHER_AST_IDENTIFIER ||
				cypher_ast_projection_get_alias (curr_proj) != NULL) {
				return false ;
			}
		}
	}

	// order by, predicates, limit and skips are not valid
	if (cypher_ast_with_get_skip(with_clause)      != NULL ||
		cypher_ast_with_get_limit(with_clause)     != NULL ||
		cypher_ast_with_get_order_by(with_clause)  != NULL ||
		cypher_ast_with_get_predicate(with_clause) != NULL) {

		return false ;
	}

	return true ;
}

// return true only if each returned expression is associated with as alias
// e.g.
// RETURN a AS b, 2 AS c...
static bool _ValidateCallReturn
(
	const cypher_astnode_t *return_clause,
	validations_ctx *vctx
) {
	uint n = cypher_ast_return_nprojections (return_clause) ;

	for (uint i = 0; i < n; i++) {
		const cypher_astnode_t *proj =
			cypher_ast_return_get_projection (return_clause, i) ;

		// projecting an identifier
		const cypher_astnode_t *exp =
			cypher_ast_projection_get_expression (proj) ;
		if (cypher_astnode_type (exp) == CYPHER_AST_IDENTIFIER) {
			continue ;
		}

		// projecting a more complex expression, make sure it's aliased
		const cypher_astnode_t *ast_alias =
			cypher_ast_projection_get_alias (proj) ;

		// projection isn't aliased
		if (ast_alias == NULL) {
			return false ;
		}
	}

	// all projected expressions are aliased
	return true ;
}

// validate a CALL {} (subquery) clause
static VISITOR_STRATEGY _Validate_call_subquery
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext (visitor) ;

	vctx->clause = cypher_astnode_type (n) ;

	// create a query astnode with the body of the subquery as its body
	cypher_astnode_t *body = cypher_ast_call_subquery_get_query (n) ;
	uint nclauses = cypher_ast_query_nclauses (body) ;

	// clone the bound vars context
	rax *in_env = raxClone (vctx->defined_identifiers) ;

	// if there are no imports, clear the env
	const cypher_astnode_t *first_clause = cypher_ast_query_get_clause (body, 0) ;
	if (cypher_astnode_type (first_clause) != CYPHER_AST_WITH) {
		raxFree (vctx->defined_identifiers) ;
		vctx->defined_identifiers = raxNew () ;
	} else {
		// validate that the WITH imports are "simple" expressions
		// i.e., 'WITH a, b, c'
		if (!_ValidateCallInitialWith (first_clause, vctx)) {
			raxFree (in_env) ;
			ErrorCtx_SetError (EMSG_CALLSUBQUERY_INVALID_REFERENCES) ;
			return VISITOR_BREAK ;
		}
	}

	// save current state
	is_union_all union_all = vctx->union_all ;
	// reset state
	vctx->union_all = NOT_DEFINED ;

	// visit the subquery clauses
	bool after_union = false ;
	for (uint i = 0; i < nclauses; i++) {
		const cypher_astnode_t *clause = cypher_ast_query_get_clause (body, i) ;
		cypher_astnode_type_t type = cypher_astnode_type (clause) ;

		// if the current clause is a `UNION` clause, it has reset the bound
		// vars env to the empty env
		// we compensate for that in case there is no initial `WITH` clause
		if (after_union && type == CYPHER_AST_WITH) {
			// set the env of bound-vars to the input env
			raxFree (vctx->defined_identifiers) ;
			vctx->defined_identifiers = raxClone (in_env) ;

			// validate that the WITH imports are "simple" expressions
			// i.e., 'WITH a, b, c'
			if (!_ValidateCallInitialWith (clause, vctx)) {
				raxFree (in_env) ;
				ErrorCtx_SetError (EMSG_CALLSUBQUERY_INVALID_REFERENCES) ;
				return VISITOR_BREAK ;
			}
		}

		AST_Visitor_visit (clause, visitor) ;
		if (ErrorCtx_EncounteredError ()) {
			raxFree (in_env) ;
			return VISITOR_BREAK ;
		}

		if (type == CYPHER_AST_UNION) {
			after_union = true ;
			continue ;
		}

		if (type == CYPHER_AST_RETURN) {
			if (!_ValidateCallReturn (clause, vctx)) {
				raxFree (in_env) ;
				ErrorCtx_SetError (EMSG_CALLSUBQUERY_ALIAS_EXPRESSION) ;
				return VISITOR_BREAK ;
			}
		}

		after_union = false ;
	}

	// restore state
	vctx->union_all = union_all ;

	// free the temporary environment
	raxFree (vctx->defined_identifiers) ;

	// restored original env
	vctx->defined_identifiers = in_env ;

	const cypher_astnode_t *last_clause =
		cypher_ast_query_get_clause (body, nclauses-1) ;
	bool is_returning = cypher_astnode_type (last_clause) == CYPHER_AST_RETURN ;

	if (is_returning) {
		// merge projected aliases from in_env into vctx->defined_identifiers
		// make sure no returned aliases are bound
		// notice: this can be done only once for the last branch of a UNION
		// since the returned aliases are always the same

		const cypher_astnode_t *return_clause = last_clause ;

		uint n = cypher_ast_return_nprojections (return_clause) ;
		for (uint i = 0; i < n; i++) {
			const cypher_astnode_t *proj =
				cypher_ast_return_get_projection (return_clause, i) ;
			const cypher_astnode_t *ast_alias =
				cypher_ast_projection_get_alias (proj) ;

			const char *name = NULL ;

			if (ast_alias == NULL) {
				// projecting an identifier
				const cypher_astnode_t *exp =
					cypher_ast_projection_get_expression (proj) ;
				ASSERT (cypher_astnode_type (exp) == CYPHER_AST_IDENTIFIER) ;
				name = cypher_ast_identifier_get_name (exp) ;
			} else {
				name = cypher_ast_identifier_get_name (ast_alias) ;
			}

			ASSERT (name != NULL) ;

			if (!raxTryInsert (vctx->defined_identifiers,
						(unsigned char *)name, strlen (name), NULL, NULL)) {
				ErrorCtx_SetError (
						EMSG_VAIABLE_ALREADY_DECLARED_IN_OUTER_SCOPE,
						name) ;
				return VISITOR_BREAK ;
			}
		}
	}

	// don't traverse children
	return VISITOR_CONTINUE ;
}

// returns true if the clause is an updating clause
#define UPDATING_CALUSE(t)           \
	(type == CYPHER_AST_CREATE ||    \
	 type == CYPHER_AST_MERGE  ||    \
	 type == CYPHER_AST_DELETE ||    \
	 type == CYPHER_AST_SET    ||    \
	 type == CYPHER_AST_REMOVE ||    \
	 type == CYPHER_AST_FOREACH)

// validate a WITH clause
static VISITOR_STRATEGY _Validate_WITH_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	vctx->clause = cypher_astnode_type(n);

	if(_Validate_LIMIT_SKIP_Modifiers(cypher_ast_with_get_limit(n),
		cypher_ast_with_get_skip(n)) == AST_INVALID) {
		return VISITOR_BREAK;
	}

	// manually traverse children. order by and predicate should be aware of the
	// vars introduced in the with projections, but the projections should not
	for(uint i = 0; i < cypher_ast_with_nprojections(n); i++) {
		// visit the projection
		const cypher_astnode_t *proj = cypher_ast_with_get_projection(n, i);
		AST_Visitor_visit(proj, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// introduce WITH aliases to the bound vars context
	if(!_AST_GetWithAliases(n, vctx)) {
		return VISITOR_BREAK;
	}

	// visit predicate clause
	const cypher_astnode_t *predicate = cypher_ast_with_get_predicate(n);
	if(predicate) {
		AST_Visitor_visit(predicate, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// visit ORDER BY clause
	const cypher_astnode_t *order_by = cypher_ast_with_get_order_by(n);
	if(order_by) {
		AST_Visitor_visit(order_by, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// if one of the 'projections' is a star -> proceed with current env
	// otherwise build a new environment using the new column names (aliases)
	if(!cypher_ast_with_has_include_existing(n)) {
		// free old env, set new one
		raxFree(vctx->defined_identifiers);
		vctx->defined_identifiers = raxNew();

		// introduce the WITH aliases to the bound vars context
		for(uint i = 0; i < cypher_ast_with_nprojections(n); i++) {
			const cypher_astnode_t *proj = cypher_ast_with_get_projection(n, i);
			const cypher_astnode_t *ast_alias =
				cypher_ast_projection_get_alias(proj);
			if(!ast_alias) {
				ast_alias = cypher_ast_projection_get_expression(proj);
			}
			const char *alias = cypher_ast_identifier_get_name(ast_alias);
			_IdentifierAdd(vctx, alias, NULL);
		}
	}

	return VISITOR_CONTINUE;
}

// validate a DELETE clause
static VISITOR_STRATEGY _Validate_DELETE_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);
	uint expression_count = cypher_ast_delete_nexpressions(n);
	for(uint i = 0; i < expression_count; i++) {
		const cypher_astnode_t *exp = cypher_ast_delete_get_expression(n, i);
		cypher_astnode_type_t type = cypher_astnode_type(exp);
		// expecting an identifier or a function call
		// identifiers and calls that don't resolve to a node, path or edge
		// will raise an error at run-time
		if(type != CYPHER_AST_IDENTIFIER         &&
		   type != CYPHER_AST_APPLY_OPERATOR     &&
		   type != CYPHER_AST_APPLY_ALL_OPERATOR &&
		   type != CYPHER_AST_SUBSCRIPT_OPERATOR) {
			ErrorCtx_SetError(EMSG_DELETE_INVALID_ARGUMENTS);
			return VISITOR_BREAK;
		}
	}

	return VISITOR_RECURSE;
}

static VISITOR_STRATEGY _Validate_REMOVE_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);

	// make sure each attribute removal is of the form:
	// identifier . propery
	unsigned int l = cypher_ast_remove_nitems(n);
	for(unsigned int i = 0; i < l; i++) {
		const cypher_astnode_t *item = cypher_ast_remove_get_item(n, i);
		cypher_astnode_type_t t = cypher_astnode_type(item);
		if(t == CYPHER_AST_REMOVE_PROPERTY) {
			const cypher_astnode_t *prop =
				cypher_ast_remove_property_get_property(item);
			const cypher_astnode_t *exp =
				cypher_ast_property_operator_get_expression(prop);

			if(cypher_astnode_type(exp) != CYPHER_AST_IDENTIFIER) {
				ErrorCtx_SetError(EMSG_REMOVE_INVALID_INPUT);
				return VISITOR_BREAK;
			}
		}
	}

	return VISITOR_RECURSE;
}

// checks if a set property contains non-aliased references in its lhs
static VISITOR_STRATEGY _Validate_set_property
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	if(!start) {
		return VISITOR_CONTINUE;
	}

	const cypher_astnode_t *ast_prop = cypher_ast_set_property_get_property(n);
	const cypher_astnode_t *ast_entity = cypher_ast_property_operator_get_expression(ast_prop);
	if(cypher_astnode_type(ast_entity) != CYPHER_AST_IDENTIFIER) {
		ErrorCtx_SetError(EMSG_SET_LHS_NON_ALIAS);
		return VISITOR_BREAK;
	}

	return VISITOR_RECURSE;
}

// validate a SET clause
static VISITOR_STRATEGY _Validate_SET_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);
	return VISITOR_RECURSE;
}

// validate a UNION clause
static VISITOR_STRATEGY _Validate_UNION_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	// make sure all UNIONs specify ALL or none of them do
	if(vctx->union_all == NOT_DEFINED) {
		vctx->union_all = cypher_ast_union_has_all(n);
	} else if(vctx->union_all != cypher_ast_union_has_all(n)) {
		ErrorCtx_SetError(EMSG_UNION_COMBINATION);
		return VISITOR_BREAK;
	}

	// free old bounded vars environment, create a new one
	vctx->clause = cypher_astnode_type(n);
	raxFree(vctx->defined_identifiers);
	vctx->defined_identifiers = raxNew();

	return VISITOR_RECURSE;
}

// validate a CREATE clause
static VISITOR_STRATEGY _Validate_CREATE_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	VISITOR_STRATEGY res = VISITOR_CONTINUE; // optimistic
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	// set current clause
	vctx->clause = cypher_astnode_type(n);

	// track new entities (identifier + type) introduced by CREATE clause
	const char **new_identifiers = array_new(const char*, 1);

	// manual traverse validation of the CREATE clause
	// this is done primarily because of identifiers scoping
	// the CREATE isn't allowed access to its own identifiers
	// e.g. CREATE (a {v:b.v}), (b {v:a.v})
	// but while the AST is traversed, we visit the created entity IDENTIFIER
	// AST node which tests to see if the identifier is within scope and fails
	// if it isn't
	// by manually traversing the AST, we can avoid this issue

	const cypher_astnode_t *pattern = cypher_ast_create_get_pattern(n);
	uint npaths = cypher_ast_pattern_npaths(pattern);

	//--------------------------------------------------------------------------
	// validate CREATE patterns
	//--------------------------------------------------------------------------

	// CREATE (a)-[:R]->(b), (b)-[:R]->(c)
	// path 1: (a)-[:R]->(b)
	//    elements:
	//    (a)
	//    [:R]
	//    (b)
	//
	// path 2: (b)-[:R]->(c)
	//   elements:
	//   (b)
	//   [:R]
	//   (c)

	for(uint i = 0; i < npaths; i++) {
		const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, i);
		// make sure CREATE actually creates something
		// e.g. MATCH (a) CREATE (a) doesn't create anything
		if(_Validate_CREATE_Entities(path, vctx) == AST_INVALID) {
			res = VISITOR_BREAK;
			goto cleanup;
		}

		// validate individual path elements
		uint nelems = cypher_ast_pattern_path_nelements(path);
		for(uint j = 0; j < nelems; j++) {
			const cypher_astnode_t *e = cypher_ast_pattern_path_get_element(path, j);
			SIType t;
			const cypher_astnode_t *id;

			if(j % 2 == 0) {
				id = cypher_ast_node_pattern_get_identifier(e);
				t = T_NODE;
			} else {
				id = cypher_ast_rel_pattern_get_identifier(e);
				t = T_EDGE;
			}

			bool hide = false;
			const char *alias = NULL;

			// hide created entity identifier from scope once processed
			// the CREATE clause is not allowed to access its own entities
			// e.g.
			// CREATE (a {v:1}), (b {v: a.v+1})
			// is invalid because 'b' is trying to access 'a' which is created
			// within the same clause
			if(id != NULL) {
				alias = cypher_ast_identifier_get_name(id);
				// hide if identifier is new
				hide = (_IdentifiersFind(vctx, alias) == raxNotFound);
			}

			// validate AST expand from current element
			AST_Visitor_visit(e, visitor);
			if(ErrorCtx_EncounteredError()) {
				res = VISITOR_BREAK;
				goto cleanup;
			}

			// remove identifier from scope
			if(hide) {
				_IdentifierRemove(vctx, alias);
				array_append(new_identifiers, alias);
				array_append(new_identifiers, (char*)t); // note identifier type
			}
		}
	}

	//--------------------------------------------------------------------------
	// introduce identifiers to scope
	//--------------------------------------------------------------------------

	uint l = array_len(new_identifiers);
	for(uint i = 0; i < l; i+=2) {
		const char *alias = new_identifiers[i];
		SIType t = (SIType)new_identifiers[i+1];

		// fail on duplicate identifier
		if(_IdentifierAdd(vctx, alias, (void*)t) == 0 && t == T_EDGE) {
			ErrorCtx_SetError(EMSG_VAIABLE_ALREADY_DECLARED, alias);
			res = VISITOR_BREAK;
			break;
		}
	}

	cleanup:
	array_free(new_identifiers);
	return res;
}

// validate a MERGE clause
static VISITOR_STRATEGY _Validate_MERGE_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);
	return VISITOR_RECURSE;
}

// validate an UNWIND clause
static VISITOR_STRATEGY _Validate_UNWIND_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	// set current clause
	vctx->clause = cypher_astnode_type(n);

	//--------------------------------------------------------------------------
	// validate unwind collection
	//--------------------------------------------------------------------------

	const cypher_astnode_t *collection = cypher_ast_unwind_get_expression(n);

	AST_Visitor_visit(collection, visitor);
	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	// introduce UNWIND alias to scope
	// fail if alias is already defined
	// e.g. MATCH (n) UNWIND [0,1] AS n RETURN n
	const cypher_astnode_t *alias = cypher_ast_unwind_get_alias(n);
	const char *identifier = cypher_ast_identifier_get_name(alias);

	if(_IdentifierAdd(vctx, identifier, NULL) == 0) {
		ErrorCtx_SetError(EMSG_VAIABLE_ALREADY_DECLARED, identifier);
		return VISITOR_BREAK;
	}

	return VISITOR_CONTINUE;
}

// validate a FOREACH clause
// MATCH (n) FOREACH(x in [1,2,3] | CREATE (n)-[:R]->({v:x}))
static VISITOR_STRATEGY _Validate_FOREACH_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = visitor->ctx;

	// we enter ONLY when start=true, so no check is needed

	// build a new environment of bounded vars from the current one to be
	// used in the traversal of the visitor in the clauses of the FOREACH
	// clause - as they are local to the FOREACH clause
	rax *orig_env = vctx->defined_identifiers;
	rax *scoped_env = raxClone(orig_env);
	vctx->defined_identifiers = scoped_env;

	// set the clause of the context
	vctx->clause = CYPHER_AST_FOREACH;

	// visit FOREACH array expression
	const cypher_astnode_t *list_node =
		cypher_ast_foreach_get_expression(n);
	AST_Visitor_visit(list_node, visitor);

	// introduce loop variable to bound vars
	const cypher_astnode_t *identifier_node =
		cypher_ast_foreach_get_identifier(n);

	const char *identifier =
		cypher_ast_identifier_get_name(identifier_node);

	_IdentifierAdd(vctx, identifier, NULL);

	// visit FOREACH loop body clauses
	uint nclauses = cypher_ast_foreach_nclauses(n);
	for(uint i = 0; i < nclauses; i++) {
		const cypher_astnode_t *clause = cypher_ast_foreach_get_clause(n, i);
		// make sure it is an updating clause
		cypher_astnode_type_t child_clause_type =
			cypher_astnode_type(clause);
		if(child_clause_type != CYPHER_AST_CREATE  &&
			child_clause_type != CYPHER_AST_SET     &&
			child_clause_type != CYPHER_AST_REMOVE  &&
			child_clause_type != CYPHER_AST_MERGE   &&
			child_clause_type != CYPHER_AST_DELETE  &&
			child_clause_type != CYPHER_AST_FOREACH
		) {
			ErrorCtx_SetError(EMSG_FOREACH_INVALID_BODY);
			break;
		}

		// visit the clause
		AST_Visitor_visit(clause, visitor);
	}

	// restore original environment of bounded vars
	vctx->defined_identifiers = orig_env;
	raxFree(scoped_env);

	// check for errors
	if(ErrorCtx_EncounteredError()) {
		return VISITOR_BREAK;
	}

	// do not traverse children
	return VISITOR_CONTINUE;
}

// validate a RETURN clause
static VISITOR_STRATEGY _Validate_RETURN_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	vctx->clause = cypher_astnode_type(n);
	uint num_return_projections = cypher_ast_return_nprojections(n);

	// visit LIMIT and SKIP
	if(_Validate_LIMIT_SKIP_Modifiers(cypher_ast_return_get_limit(n), cypher_ast_return_get_skip(n))
		== AST_INVALID) {
			return VISITOR_BREAK;
	}

	if(!cypher_ast_return_has_include_existing(n)) {
		// check for duplicate column names
		rax           *rax          = raxNew();
		const char   **columns      = AST_BuildReturnColumnNames(n);
		uint           column_count = array_len(columns);

		for (uint i = 0; i < column_count; i++) {
			// column with same name is invalid
			if(raxTryInsert(rax, (unsigned char *)columns[i], strlen(columns[i]), NULL, NULL) == 0) {
				ErrorCtx_SetError(EMSG_SAME_RESULT_COLUMN_NAME);
				break;
			}
		}

		raxFree(rax);
		array_free(columns);
	}

	// manually traverse children. order by and predicate should be aware of the
	// vars introduced in the with projections, but the projections should not
	for(uint i = 0; i < cypher_ast_return_nprojections(n); i++) {
		// visit the projection
		const cypher_astnode_t *proj = cypher_ast_return_get_projection(n, i);
		AST_Visitor_visit(proj, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// introduce bound vars
	for(uint i = 0; i < num_return_projections; i ++) {
		const cypher_astnode_t *child = cypher_ast_return_get_projection(n, i);
		const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(child);
		if(alias_node == NULL) {
			continue;
		}
		const char *alias = cypher_ast_identifier_get_name(alias_node);
		_IdentifierAdd(vctx, alias, NULL);
	}

	// visit ORDER BY clause
	const cypher_astnode_t *order_by = cypher_ast_return_get_order_by(n);
	if(order_by) {
		AST_Visitor_visit(order_by, visitor);
		if(ErrorCtx_EncounteredError()) {
			return VISITOR_BREAK;
		}
	}

	// do not traverse children
	return !ErrorCtx_EncounteredError() ? VISITOR_CONTINUE : VISITOR_BREAK;
}

// validate a MATCH clause
static VISITOR_STRATEGY _Validate_MATCH_Clause
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);
	
	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);
	return VISITOR_RECURSE;
}

// validate index creation
static VISITOR_STRATEGY _Validate_index_creation
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);

	const cypher_astnode_t *id = cypher_ast_create_pattern_props_index_get_identifier(n);
	const char *name = cypher_ast_identifier_get_name(id);
	_IdentifierAdd(vctx, name, NULL);
	return VISITOR_RECURSE;
}

// validate index deletion
static VISITOR_STRATEGY _Validate_index_deletion
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	validations_ctx *vctx = AST_Visitor_GetContext(visitor);

	if(!start) {
		return VISITOR_CONTINUE;
	}

	vctx->clause = cypher_astnode_type(n);

	const cypher_astnode_t *id = cypher_ast_drop_pattern_props_index_get_identifier(n);
	const char *name = cypher_ast_identifier_get_name(id);
	_IdentifierAdd(vctx, name, NULL);
	return VISITOR_RECURSE;
}

// A query must end in a RETURN clause, a procedure, or an updating clause
// (CREATE, MERGE, DELETE, SET, REMOVE, FOREACH or CALL {})
static AST_Validation _ValidateQueryTermination
(
	const AST *ast  // ast
) {
	ASSERT(ast != NULL);

	const cypher_astnode_t *root = ast->root;
	uint clause_idx = 0;
	const cypher_astnode_t *return_clause = NULL;
	const cypher_astnode_t *following_clause = NULL;
	uint clause_count = cypher_ast_query_nclauses(root);

	const cypher_astnode_t *last_clause = cypher_ast_query_get_clause(root,
		clause_count - 1);
	cypher_astnode_type_t type = cypher_astnode_type(last_clause);
	if(type != CYPHER_AST_RETURN         &&
	   type != CYPHER_AST_CREATE         &&
	   type != CYPHER_AST_MERGE          &&
	   type != CYPHER_AST_DELETE         &&
	   type != CYPHER_AST_SET            &&
	   type != CYPHER_AST_CALL           &&
	   type != CYPHER_AST_CALL_SUBQUERY  &&
	   type != CYPHER_AST_REMOVE         &&
	   type != CYPHER_AST_FOREACH
	  ) {
			ErrorCtx_SetError(EMSG_QUERY_INVALID_LAST_CLAUSE,
						cypher_astnode_typestr(type));
		return AST_INVALID;
	}

	// if the last clause is CALL {}, it must be non-returning
	if(type == CYPHER_AST_CALL_SUBQUERY) {
		cypher_astnode_t *query =
			cypher_ast_call_subquery_get_query(last_clause);
		if(cypher_astnode_type(cypher_ast_query_get_clause(query,
				cypher_ast_query_nclauses(query)-1)) ==
		   CYPHER_AST_RETURN) {
			ErrorCtx_SetError(EMSG_QUERY_INVALID_LAST_CLAUSE,
						"a returning subquery");

			return AST_INVALID;
		}
	}

	// validate that `UNION` is the only clause following a `RETURN` clause, and
	// termination of embedded call {} clauses
	bool last_was_return = false;
	for(uint i = 0; i < clause_count; i++) {
		const cypher_astnode_t *clause = cypher_ast_query_get_clause(root, i);
		cypher_astnode_type_t type = cypher_astnode_type(clause);
		if(type != CYPHER_AST_UNION && last_was_return) {
			// unexpected clause following RETURN
			ErrorCtx_SetError(EMSG_UNEXPECTED_CLAUSE_FOLLOWING_RETURN);
			return AST_INVALID;
		} else if(type == CYPHER_AST_RETURN) {
			last_was_return = true;
		} else if(type == CYPHER_AST_CALL_SUBQUERY) {
			AST subquery_ast = {
				.root = cypher_ast_call_subquery_get_query(clause)
			};
			if(_ValidateQueryTermination(&subquery_ast) != AST_VALID) {
				return AST_INVALID;
			}
			last_was_return = false;
		} else {
			last_was_return = false;
		}
	}

	return AST_VALID;
}

// default visit function
VISITOR_STRATEGY _default_visit
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	ASSERT(n != NULL);

	return VISITOR_RECURSE;
}

// Perform validations not constrained to a specific scope
static AST_Validation _ValidateQuerySequence
(
	const AST *ast
) {
	// Validate the final clause
	if(_ValidateQueryTermination(ast) != AST_VALID) {
		return AST_INVALID;
	}

	// The query cannot begin with a "WITH *" projection.
	const cypher_astnode_t *start_clause = cypher_ast_query_get_clause(ast->root, 0);
	if(cypher_astnode_type(start_clause) == CYPHER_AST_WITH &&
	   cypher_ast_with_has_include_existing(start_clause)) {
		ErrorCtx_SetError(EMSG_QUERY_CANNOT_BEGIN_WITH, "WITH");
		return AST_INVALID;
	}

	// The query cannot begin with a "RETURN *" projection.
	if(cypher_astnode_type(start_clause) == CYPHER_AST_RETURN &&
	   cypher_ast_return_has_include_existing(start_clause)) {
		ErrorCtx_SetError(EMSG_QUERY_CANNOT_BEGIN_WITH, "RETURN");
		return AST_INVALID;
	}

	return AST_VALID;
}

// in any given query scope, reading clauses (MATCH, UNWIND, and InQueryCall)
// cannot follow updating clauses (CREATE, MERGE, DELETE, SET, REMOVE, FOREACH).
// https://s3.amazonaws.com/artifacts.opencypher.org/railroad/SinglePartQuery.html
// Additionally, a MATCH clause cannot follow an OPTIONAL MATCH clause
static AST_Validation _ValidateClauseOrder
(
	const AST *ast  // ast
) {
	ASSERT(ast != NULL);

	uint clause_count                = cypher_ast_query_nclauses(ast->root);
	bool encountered_optional_match  = false;
	bool encountered_updating_clause = false;

	for(uint i = 0; i < clause_count; i++) {
		const cypher_astnode_t *clause =
			cypher_ast_query_get_clause(ast->root, i);
		cypher_astnode_type_t type = cypher_astnode_type(clause);

		if(encountered_updating_clause && (type == CYPHER_AST_MATCH          ||
										   type == CYPHER_AST_UNWIND         ||
										   type == CYPHER_AST_CALL           ||
										   type == CYPHER_AST_CALL_SUBQUERY)) {
			ErrorCtx_SetError(EMSG_MISSING_WITH, cypher_astnode_typestr(type));
			return AST_INVALID;
		}

		encountered_updating_clause |= UPDATING_CALUSE(type);

		if(type == CYPHER_AST_MATCH) {
			// check whether this match is optional
			bool current_clause_is_optional = cypher_ast_match_is_optional(clause);
			// if the current clause is non-optional but we have already
			// encountered an optional match, emit an error
			if(!current_clause_is_optional && encountered_optional_match) {
				ErrorCtx_SetError(EMSG_MISSING_WITH_AFTER_MATCH);
				return AST_INVALID;
			}
			encountered_optional_match |= current_clause_is_optional;
		} else if(type == CYPHER_AST_WITH || type == CYPHER_AST_UNION) {
			// reset scope on WITH / UNION clauses
			encountered_optional_match  = false;
			encountered_updating_clause = false;
		} else if(type == CYPHER_AST_CALL_SUBQUERY) {
			AST subquery_ast = {
				.root = cypher_ast_call_subquery_get_query(clause)
			};
			if(_ValidateClauseOrder(&subquery_ast) != AST_VALID) {
				return AST_INVALID;
			}
		}
	}

	return AST_VALID;
}

// break visitor traversal, resulting in a fast-fold
static VISITOR_STRATEGY _visit_break
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	Error_UnsupportedASTNodeType(n);
	return VISITOR_BREAK;
}

// visit a binary operator, break if it is unsupported
static VISITOR_STRATEGY _visit_binary_op
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
) {
	const cypher_operator_t *op = cypher_ast_binary_operator_get_operator(n);
	if(op == CYPHER_OP_SUBSCRIPT ||
	   op == CYPHER_OP_MAP_PROJECTION ||
	   op == CYPHER_OP_REGEX) {
		Error_UnsupportedASTOperator(op);
		return VISITOR_BREAK;
	}
	return VISITOR_RECURSE;
}

// validate a query
static AST_Validation _ValidateScopes
(
	AST *ast  // ast
) {
	// create a context for the traversal
	validations_ctx ctx = {
		.union_all = NOT_DEFINED,
		.defined_identifiers = raxNew()
	};

	// create a visitor
	ast_visitor visitor = {&ctx, validations_mapping};

	// visit (traverse) the ast
	AST_Visitor_visit(ast->root, &visitor);
	
	// cleanup
	raxFree(ctx.defined_identifiers);

	return !ErrorCtx_EncounteredError() ? AST_VALID : AST_INVALID;
}

// build the global mapping from ast-node-type to visiting functions
bool AST_ValidationsMappingInit(void) {
	// create a mapping for the validations

	// set default entries
	for(uint i = 0; i < 116; i++) {
		validations_mapping[i] = _default_visit;
	}

	// populate the mapping with validation functions

	//--------------------------------------------------------------------------
	// register supported types
	//--------------------------------------------------------------------------

	validations_mapping[CYPHER_AST_SET]                        = _Validate_SET_Clause;
	validations_mapping[CYPHER_AST_MAP]                        = _Validate_map;
	validations_mapping[CYPHER_AST_ANY]                        = _Validate_list_comprehension;
	validations_mapping[CYPHER_AST_ALL]                        = _Validate_list_comprehension;
	validations_mapping[CYPHER_AST_CALL]                       = _Validate_CALL_Clause;
	validations_mapping[CYPHER_AST_WITH]                       = _Validate_WITH_Clause;
	validations_mapping[CYPHER_AST_NONE]                       = _Validate_list_comprehension;
	validations_mapping[CYPHER_AST_UNION]                      = _Validate_UNION_Clause;
	validations_mapping[CYPHER_AST_MATCH]                      = _Validate_MATCH_Clause;
	validations_mapping[CYPHER_AST_MERGE]                      = _Validate_MERGE_Clause;
	validations_mapping[CYPHER_AST_SINGLE]                     = _Validate_list_comprehension;
	validations_mapping[CYPHER_AST_RETURN]                     = _Validate_RETURN_Clause;
	validations_mapping[CYPHER_AST_UNWIND]                     = _Validate_UNWIND_Clause;
	validations_mapping[CYPHER_AST_CREATE]                     = _Validate_CREATE_Clause;
	validations_mapping[CYPHER_AST_DELETE]                     = _Validate_DELETE_Clause;
	validations_mapping[CYPHER_AST_REMOVE]                     = _Validate_REMOVE_Clause;
	validations_mapping[CYPHER_AST_REDUCE]                     = _Validate_reduce;
	validations_mapping[CYPHER_AST_FOREACH]                    = _Validate_FOREACH_Clause;
	validations_mapping[CYPHER_AST_LOAD_CSV]                   = _Validate_load_csv;
	validations_mapping[CYPHER_AST_IDENTIFIER]                 = _Validate_identifier;
	validations_mapping[CYPHER_AST_PROJECTION]                 = _Validate_projection;
	validations_mapping[CYPHER_AST_NAMED_PATH]                 = _Validate_named_path;
	validations_mapping[CYPHER_AST_REL_PATTERN]                = _Validate_rel_pattern;
	validations_mapping[CYPHER_AST_SET_PROPERTY]               = _Validate_set_property;
	validations_mapping[CYPHER_AST_NODE_PATTERN]               = _Validate_node_pattern;
	validations_mapping[CYPHER_AST_CALL_SUBQUERY]              = _Validate_call_subquery;
	validations_mapping[CYPHER_AST_SHORTEST_PATH]              = _Validate_shortest_path;
	validations_mapping[CYPHER_AST_APPLY_OPERATOR]             = _Validate_apply_operator;
	validations_mapping[CYPHER_AST_BINARY_OPERATOR]            = _visit_binary_op;
	validations_mapping[CYPHER_AST_APPLY_ALL_OPERATOR]         = _Validate_apply_all_operator;
	validations_mapping[CYPHER_AST_LIST_COMPREHENSION]         = _Validate_list_comprehension;
	validations_mapping[CYPHER_AST_PATTERN_COMPREHENSION]      = _Validate_pattern_comprehension;
	validations_mapping[CYPHER_AST_DROP_PATTERN_PROPS_INDEX]   = _Validate_index_deletion;	
	validations_mapping[CYPHER_AST_CREATE_PATTERN_PROPS_INDEX] = _Validate_index_creation;

	//--------------------------------------------------------------------------
	// register unsupported types
	//--------------------------------------------------------------------------

	validations_mapping[CYPHER_AST_START]                       = _visit_break;
	validations_mapping[CYPHER_AST_FILTER]                      = _visit_break;
	validations_mapping[CYPHER_AST_EXTRACT]                     = _visit_break;
	validations_mapping[CYPHER_AST_COMMAND]                     = _visit_break;
	validations_mapping[CYPHER_AST_MATCH_HINT]                  = _visit_break;
	validations_mapping[CYPHER_AST_USING_JOIN]                  = _visit_break;
	validations_mapping[CYPHER_AST_USING_SCAN]                  = _visit_break;
	validations_mapping[CYPHER_AST_INDEX_NAME]                  = _visit_break;
	validations_mapping[CYPHER_AST_REL_ID_LOOKUP]               = _visit_break;
	validations_mapping[CYPHER_AST_ALL_RELS_SCAN]               = _visit_break;
	validations_mapping[CYPHER_AST_USING_INDEX]                 = _visit_break;
	validations_mapping[CYPHER_AST_START_POINT]                 = _visit_break;
	validations_mapping[CYPHER_AST_REMOVE_ITEM]                 = _visit_break;
	validations_mapping[CYPHER_AST_QUERY_OPTION]                = _visit_break;
	validations_mapping[CYPHER_AST_REL_INDEX_QUERY]             = _visit_break;
	validations_mapping[CYPHER_AST_EXPLAIN_OPTION]              = _visit_break;
	validations_mapping[CYPHER_AST_PROFILE_OPTION]              = _visit_break;
	validations_mapping[CYPHER_AST_SCHEMA_COMMAND]              = _visit_break;
	validations_mapping[CYPHER_AST_NODE_ID_LOOKUP]              = _visit_break;
	validations_mapping[CYPHER_AST_ALL_NODES_SCAN]              = _visit_break;
	validations_mapping[CYPHER_AST_REL_INDEX_LOOKUP]            = _visit_break;
	validations_mapping[CYPHER_AST_NODE_INDEX_QUERY]            = _visit_break;
	validations_mapping[CYPHER_AST_NODE_INDEX_LOOKUP]           = _visit_break;
	validations_mapping[CYPHER_AST_USING_PERIODIC_COMMIT]       = _visit_break;
	validations_mapping[CYPHER_AST_DROP_REL_PROP_CONSTRAINT]    = _visit_break;
	validations_mapping[CYPHER_AST_DROP_NODE_PROP_CONSTRAINT]   = _visit_break;
	validations_mapping[CYPHER_AST_CREATE_REL_PROP_CONSTRAINT]  = _visit_break;
	validations_mapping[CYPHER_AST_CREATE_NODE_PROP_CONSTRAINT] = _visit_break;

	return true;
}

// Checks to see if libcypher-parser reported any errors.
bool AST_ContainsErrors
(
	const cypher_parse_result_t *result  // parse-result checked for errors
) {
	return cypher_parse_result_nerrors(result) > 0;
}

/* This function checks for the existence a valid root in the query.
 * As cypher_parse_result_t can have multiple roots such as comments,
 * only a query that has a root with type CYPHER_AST_STATEMENT is considered valid.
 * Comment roots are ignored. */
AST_Validation AST_Validate_ParseResultRoot
(
	const cypher_parse_result_t *result,
	int *index
) {
	// Check for failures in libcypher-parser
	ASSERT(AST_ContainsErrors(result) == false);

	uint nroots = cypher_parse_result_nroots(result);
	for(uint i = 0; i < nroots; i++) {
		const cypher_astnode_t *root = cypher_parse_result_get_root(result, i);
		cypher_astnode_type_t root_type = cypher_astnode_type(root);
		if(root_type == CYPHER_AST_LINE_COMMENT || root_type == CYPHER_AST_BLOCK_COMMENT ||
		   root_type == CYPHER_AST_COMMENT) {
			continue;
		} else if(root_type != CYPHER_AST_STATEMENT) {
			ErrorCtx_SetError(EMSG_UNSUPPORTED_QUERY_TYPE, cypher_astnode_typestr(root_type));
			return AST_INVALID;
		} else {
			// We got a statement.
			*index = i;
			return AST_VALID;
		}
	}

	// query with no roots like ';'
	if(nroots == 0) {
		ErrorCtx_SetError(EMSG_EMPTY_QUERY);
	}

	return AST_INVALID;
}

// validate a query
AST_Validation AST_Validate_Query
(
	const cypher_astnode_t *root  // query to validate
) {
	const cypher_astnode_t *body = cypher_ast_statement_get_body(root);
	AST ast; // Build a fake AST with the correct AST root
	ast.root = body;

	cypher_astnode_type_t body_type = cypher_astnode_type(body);

	if(body_type == CYPHER_AST_CREATE_NODE_PROP_CONSTRAINT ||
	   body_type == CYPHER_AST_CREATE_REL_PROP_CONSTRAINT ||
	   body_type == CYPHER_AST_DROP_NODE_PROP_CONSTRAINT ||
	   body_type == CYPHER_AST_DROP_REL_PROP_CONSTRAINT) {
		ErrorCtx_SetError(EMSG_INVALID_CONSTRAINT_COMMAND);
		return AST_INVALID;
	}

	if(body_type == CYPHER_AST_CREATE_NODE_PROPS_INDEX    ||
	   body_type == CYPHER_AST_CREATE_PATTERN_PROPS_INDEX ||
	   body_type == CYPHER_AST_DROP_PROPS_INDEX           ||
	   body_type == CYPHER_AST_DROP_PATTERN_PROPS_INDEX) {
		return _ValidateScopes(&ast);
	}

	// Verify that the RETURN clause and terminating clause do not violate scoping rules.
	if(_ValidateQuerySequence(&ast) != AST_VALID) {
		return AST_INVALID;
	}

	// Verify that the clause order in the scope is valid.
	if(_ValidateClauseOrder(&ast) != AST_VALID) {
		return AST_INVALID;
	}

	// Verify that the clauses surrounding UNION return the same column names.
	if(_ValidateUnion_Clauses(&ast) != AST_VALID) {
		return AST_INVALID;
	}

	// validate positions of allShortestPaths
	if(!_ValidateAllShortestPaths(body)) {
		ErrorCtx_SetError(EMSG_ALLSHORTESTPATH_SUPPORT);
		return AST_INVALID;
	}

	if(!_ValidateShortestPaths(body)) {
		ErrorCtx_SetError(EMSG_SHORTESTPATH_SUPPORT);
		return AST_INVALID;
	}

	// check for invalid queries not captured by libcypher-parser
	return _ValidateScopes(&ast);
}

// report encountered errors by libcypher-parser
void AST_ReportErrors
(
	const cypher_parse_result_t *result  // parse-result
) {
	ASSERT(cypher_parse_result_nerrors(result) > 0);

	// report first encountered error
	const cypher_parse_error_t *error =
		cypher_parse_result_get_error(result, 0);

	// Get the position of an error.
	struct cypher_input_position errPos = cypher_parse_error_position(error);

	// Get the error message of an error.
	const char *errMsg = cypher_parse_error_message(error);

	// Get the error context of an error.
	// This returns a pointer to a null-terminated string, which contains a
	// section of the input around where the error occurred, that is limited
	// in length and suitable for presentation to a user.
	const char *errCtx = cypher_parse_error_context(error);

	// Get the offset into the context of an error.
	// Identifies the point of the error within the context string, allowing
	// this to be reported to the user, typically with an arrow pointing to the
	// invalid character.
	size_t errCtxOffset = cypher_parse_error_context_offset(error);
	ErrorCtx_SetError(EMSG_PARSER_ERROR, errMsg, errPos.line, errPos.column,
			errPos.offset, errCtx, errCtxOffset);
}

