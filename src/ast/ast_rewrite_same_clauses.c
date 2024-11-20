/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "ast.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/qsort.h"
#include "../util/sds/sds.h"
#include "../errors/errors.h"
#include "../procedures/procedure.h"

// forward declarations
static inline bool is_compressible(const cypher_astnode_t *clause,
	cypher_astnode_type_t type);
static bool _compress_clauses(const cypher_astnode_t *node);

typedef void (*replace_func)(cypher_astnode_t *, cypher_astnode_t *, unsigned int, unsigned int);
typedef const cypher_astnode_t * (*get_clause_func)(const cypher_astnode_t *, unsigned int);

// compresses multiple consecutive MATCH clauses into a single MATCH clause
static void _replace_match_clause
(
	cypher_astnode_t *root,      // ast root
	cypher_astnode_t **clauses,  // clause being replaced
	int scope_start,             // beginning of scope
	int scope_end,               // ending of scope
	replace_func replace         // replace function pointer
) {
	uint count = array_len(clauses);

	cypher_astnode_t *predicate = NULL;
	struct cypher_input_range range = cypher_astnode_range(clauses[0]);
	cypher_astnode_t **paths = array_new(cypher_astnode_t *, count);

	// collect MATCH patterns and predicates
	for (uint i = 0; i < count; i++) {
		const cypher_astnode_t *pattern =
			cypher_ast_match_get_pattern(clauses[i]);
		uint npaths = cypher_ast_pattern_npaths(pattern);

		for (uint j = 0; j < npaths; j++) {
			const cypher_astnode_t *path =
				cypher_ast_pattern_get_path(pattern, j);
			array_append(paths, cypher_ast_clone(path));	
		}

		// combine MATCH predicates into a single predicate using AND connectors
		cypher_astnode_t *new_predicate =
			(cypher_astnode_t *)cypher_ast_match_get_predicate(clauses[i]);
		if(new_predicate != NULL) {
			new_predicate = cypher_ast_clone(new_predicate);
			if(predicate == NULL) {
				predicate = new_predicate;
			} else {
				// concat using AND
				cypher_astnode_t *children[] = {predicate, new_predicate};
				predicate = cypher_ast_binary_operator(CYPHER_OP_AND, predicate,
						new_predicate, children, 2, range);
			}
		}
	}
	
	// build the replacement pattern
	cypher_astnode_t *pattern =
		cypher_ast_pattern(paths, array_len(paths), paths, array_len(paths),
				range);
	
	// build the replacement clause
	cypher_astnode_t *children[] = {pattern, predicate};
	cypher_astnode_t *new_clause = cypher_ast_match(false, pattern, NULL, 0,
			predicate, children, predicate == NULL ? 1 : 2, range);

	// replace original clause with the new one
	replace(root, new_clause, scope_start, scope_end);
	
	array_free(paths);
}

// compresses multiple consecutive DELETE clauses into a single DELETE clause
static void _replace_delete_clause
(
	cypher_astnode_t *root,      // ast root
	cypher_astnode_t **clauses,  // clause being replaced
	int scope_start,             // beginning of scope
	int scope_end,               // ending of scope
	replace_func replace         // replace function pointer
) {
	uint count = array_len(clauses);

	struct cypher_input_range range = cypher_astnode_range(clauses[0]);
	cypher_astnode_t **exps = array_new(cypher_astnode_t *, count);
	rax *identifiers = raxNew();

	// collect expressions
	for (uint i = 0; i < count; i++) {
		uint nexps = cypher_ast_delete_nexpressions(clauses[i]);
		for(uint j = 0; j < nexps; j++) {
			const cypher_astnode_t *exp =
				cypher_ast_delete_get_expression(clauses[i], j);
			
			// do not aggregate multiple appearances of an identifier
			if(cypher_astnode_type(exp) == CYPHER_AST_IDENTIFIER) {
				const char *identifier = cypher_ast_identifier_get_name(exp);
				if(raxTryInsert(identifiers, (unsigned char *)identifier, strlen(identifier), NULL, NULL)) {
					array_append(exps, cypher_ast_clone(exp));
				}
			} else {
				// "MATCH p1=(n:N), p2=(m:M) DELETE nodes(p1)[0] DELETE nodes(p2)[0]"
				array_append(exps, cypher_ast_clone(exp));
			}
		}
	}
	raxFree(identifiers);

	// build the replacement clause
	cypher_astnode_t *new_clause = cypher_ast_delete(false, exps,
			array_len(exps), exps, array_len(exps), range);

	// replace original clause with the new one
	replace(root, new_clause, scope_start, scope_end);
	
	array_free(exps);
}

// compresses multiple consecutive SET clauses into a single SET clause
static void _replace_set_clause
(
	cypher_astnode_t *root,      // ast root
	cypher_astnode_t **clauses,  // clause being replaced
	int scope_start,             // beginning of scope
	int scope_end,               // ending of scope
	replace_func replace         // replace function pointer
) {
	uint count = array_len(clauses);

	struct cypher_input_range range = cypher_astnode_range(clauses[0]);
	cypher_astnode_t **items = array_new(cypher_astnode_t *, count);

	for (uint i = 0; i < count; i++) {
		uint nitems = cypher_ast_set_nitems(clauses[i]);
		for(uint j = 0; j < nitems; j++) {
			const cypher_astnode_t *item =
				cypher_ast_set_get_item(clauses[i], j);
			array_append(items, cypher_ast_clone(item));
		}
	}
	
	// build the replacement clause
	cypher_astnode_t *new_clause = cypher_ast_set(items, array_len(items),
			items, array_len(items), range);

	// replace original clause with the new one
	replace(root, new_clause, scope_start, scope_end);
	
	array_free(items);
}

// compresses multiple consecutive REMOVE clauses into a single REMOVE clause
static void _replace_remove_clause
(
	cypher_astnode_t *root,      // ast root
	cypher_astnode_t **clauses,  // clause being replaced
	int scope_start,             // beginning of scope
	int scope_end,               // ending of scope
	replace_func replace         // replace function pointer
) {
	uint count = array_len(clauses);

	struct cypher_input_range range = cypher_astnode_range(clauses[0]);
	cypher_astnode_t **items = array_new(cypher_astnode_t *, count);

	for (uint i = 0; i < count; i++) {
		uint nitems = cypher_ast_remove_nitems(clauses[i]);
		for(uint j = 0; j < nitems; j++) {
			const cypher_astnode_t *item =
				cypher_ast_remove_get_item(clauses[i], j);
			array_append(items, cypher_ast_clone(item));
		}
	}
	
	// build the replacement clause
	cypher_astnode_t *new_clause =
		cypher_ast_remove(items, array_len(items), items, array_len(items),
				range);

	// replace original clause with fully populated one
	replace(root, new_clause, scope_start, scope_end);
	
	array_free(items);
}

// compresses multiple consecutive MERGE clauses into a single MERGE clause
static void _replace_merge_clause
(
	cypher_astnode_t *root,      // ast root
	cypher_astnode_t **clauses,  // clause being replaced
	int scope_start,             // beginning of scope
	int scope_end,               // ending of scope
	replace_func replace         // replace function pointer
) {
	uint count = array_len(clauses);

	struct cypher_input_range range = cypher_astnode_range(clauses[0]);
	cypher_astnode_t **paths = array_new(cypher_astnode_t *, count);

	// collect MERGE patterns
	for (uint i = 0; i < count; i++) {
		const cypher_astnode_t *path =
			cypher_ast_merge_get_pattern_path(clauses[i]);
		array_append(paths, cypher_ast_clone(path));
	}
	
	// build the replacement clause
	cypher_astnode_t *new_clause = cypher_ast_merge(paths, array_len(paths),
			paths, array_len(paths), range);

	// replace original clause with the new one
	replace(root, new_clause, scope_start, scope_end);
	
	array_free(paths);
}

// tries to compress clauses of a query or a foreach clause
// returns true if a rewrite occurred
static bool _compress_clauses
(
	const cypher_astnode_t *node  // node containing clauses to compress
) {
	bool rewritten = false;
	
	cypher_astnode_type_t type = cypher_astnode_type(node);

	ASSERT(type == CYPHER_AST_QUERY || type == CYPHER_AST_FOREACH);

	cypher_astnode_t **clauses = array_new(cypher_astnode_t *, 0);
	// is the node representing a FOREACH clause
	uint            clause_count  = 0;
	replace_func    replace_func  = NULL;
	get_clause_func get_clause    = NULL;
	// use appropriate function to get a positioned clause
	if(type == CYPHER_AST_FOREACH) {
		get_clause   = cypher_ast_foreach_get_clause;
		clause_count = cypher_ast_foreach_nclauses(node);
		replace_func = cypher_ast_foreach_replace_clauses;
	} else {
		get_clause   = cypher_ast_query_get_clause;
		replace_func = cypher_ast_query_replace_clauses;
		clause_count = cypher_ast_query_nclauses(node);
	}

	for(uint i = 0; i < clause_count; i++) {
		const cypher_astnode_t *clause = get_clause(node, i);
		cypher_astnode_type_t t = cypher_astnode_type(clause);

		// try compressing the inner-clauses of foreach clause and call subquery
		if(t == CYPHER_AST_FOREACH) {
			rewritten |= _compress_clauses(clause);
			continue;
		} else if(t == CYPHER_AST_CALL_SUBQUERY) {
			rewritten |=
				_compress_clauses(cypher_ast_call_subquery_get_query(clause));
			continue;
		}

		// check compressibility, move on if not compressible
		if(!is_compressible(clause, t)) {
			continue;
		}

		//----------------------------------------------------------------------
		// collect clauses of the same type as current clause
		//----------------------------------------------------------------------

		for (uint j = i; j < clause_count; j++) {
			clause = get_clause(node, j);
			cypher_astnode_type_t t2 = cypher_astnode_type(clause);
			if(t2 != t || !is_compressible(clause, t2)) {
				break;
			}
			array_append(clauses, (cypher_astnode_t *)clause);
		}

		//----------------------------------------------------------------------
		// compress clauses
		//----------------------------------------------------------------------

		uint s = i;
		uint e = s + array_len(clauses) - 1;
		if(array_len(clauses) > 1) {
			// multiple consecutive clauses of the same type
			// compress them
			if(t == CYPHER_AST_MATCH) {
				_replace_match_clause((cypher_astnode_t *)node, clauses, s, e, replace_func);
			} else if(t == CYPHER_AST_DELETE) {
				_replace_delete_clause((cypher_astnode_t *)node, clauses, s, e, replace_func);
			} else if(t == CYPHER_AST_SET) {
				_replace_set_clause((cypher_astnode_t *)node, clauses, s, e, replace_func);
			} else if(t == CYPHER_AST_REMOVE) {
				_replace_remove_clause((cypher_astnode_t *)node, clauses, s, e, replace_func);
			} else if(t == CYPHER_AST_MERGE) {
				_replace_merge_clause((cypher_astnode_t *)node, clauses, s, e, replace_func);
			}

			rewritten = true;

			// update clause count, skip compressed clauses
			clause_count -= array_len(clauses) - 1;
		}

		array_clear(clauses);
	}

	array_free(clauses);
	return rewritten;
}

// returns true if AST clause type is compressible
static inline bool is_compressible
(
	const cypher_astnode_t *clause,  // clause to check
	cypher_astnode_type_t type       // type of clause
) {
	// compressible clauses:
	// 1. Non-OPTIONAL MATCH
	// 2. SET
	// 3. DELETE
	// 4. REMOVE
	// 5. MERGE
	return ((type == CYPHER_AST_MATCH &&
			  !cypher_ast_match_is_optional(clause)) ||
			  type == CYPHER_AST_SET                 ||
			  type == CYPHER_AST_DELETE              ||
			  type == CYPHER_AST_REMOVE              ||
			  type == CYPHER_AST_MERGE);
}

// rewrite result by compressing consecutive clauses of the same type
// to a single clause, returning true if the rewrite has been performed
bool AST_RewriteSameClauses
(
	const cypher_astnode_t *root // root of AST
) {
	bool rewritten = false;

	if(cypher_astnode_type(root) != CYPHER_AST_STATEMENT) {
		return rewritten;
	}

	// retrieve the root's body
	const cypher_astnode_t *body = cypher_ast_statement_get_body(root);
	if(cypher_astnode_type(body) != CYPHER_AST_QUERY) {
		return rewritten;
	}

	// traverse clauses
	// compress consecutive clauses
	return _compress_clauses(body);
}
