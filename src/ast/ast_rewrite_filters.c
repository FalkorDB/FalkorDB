/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "rax.h"
#include "../util/arr.h"
#include "ast_rewrites.h"

static void collectProjectedIdentifiers
(
	const cypher_astnode_t *with, // WITH clause
	rax *exps,                     // projected expressions
	rax *aliases                   // projected aliases
) {
	ASSERT(with    != NULL);
	ASSERT(exps    != NULL);
	ASSERT(aliases != NULL);

	// collect projected identifiers & aliases
	uint projs = cypher_ast_with_nprojections(with);
	for (uint i = 0; i < projs; i++) {
		const char *alias_name = NULL;
		const cypher_astnode_t *proj =
			cypher_ast_with_get_projection(with, i);
		const cypher_astnode_t *alias =
			cypher_ast_projection_get_alias(proj);
		const cypher_astnode_t *exp =
			cypher_ast_projection_get_expression(proj);

		if(alias != NULL) {
			alias_name = cypher_ast_identifier_get_name(alias);
			raxInsert(aliases, (unsigned char*)alias_name,
					strlen(alias_name), NULL, NULL);
		}

		if(cypher_astnode_type(exp) == CYPHER_AST_IDENTIFIER) {
			const char *exp_name = cypher_ast_identifier_get_name(exp);
			if(alias_name != NULL) {
				// WITH a AS b
				raxInsert(exps, (unsigned char*)exp_name, strlen(exp_name),
						(void*)alias_name, NULL);
			} else {
				// WITH a == WITH a AS a
				raxInsert(aliases, (unsigned char*)exp_name, strlen(exp_name),
						NULL, NULL);
			}
		}
	}
}

// WITH a AS b WHERE a.v = 1 -> WITH a AS b WHERE b.v = 1
bool adjustPredicateIdentifiers
(
	const cypher_astnode_t *root,  // predicate current inspected root
	rax *exps,                     // projected expressions
	rax *aliases                   // projected aliases
) {
	bool res = false;

	// scan through predicate identifiers
	int n = cypher_astnode_nchildren(root);

	for(int i = 0; i < n; i++) {
		const cypher_astnode_t *child = cypher_astnode_get_child(root, i);

		if(cypher_astnode_type(child) != CYPHER_AST_IDENTIFIER) {
			// recursively scan through predicate
			adjustPredicateIdentifiers(child, exps, aliases);
			continue;
		}

		const char *name = cypher_ast_identifier_get_name(child);
		size_t len = strlen(name);

		// continue if identifier is aliased
		if(raxFind(aliases, (unsigned char*)name, len) != raxNotFound) {
			continue;
		}

		// see if identifier is projected
		const char *alternative = (const char*) raxFind(exps,
				(unsigned char*)name, len);

		// unknown identifier
		if(alternative == raxNotFound) continue;

		// replace identifier with alias
		struct cypher_input_range range = {0};
		cypher_astnode_t *new_identifier = cypher_ast_identifier(alternative,
			strlen(alternative), range);
		cypher_astnode_set_child((cypher_astnode_t*)root, new_identifier, i);
		res = true;
		//cypher_astnode_free((cypher_astnode_t*)child);
	}

	return res;
}

static bool _AST_RewriteFilters
(
	const cypher_astnode_t *with // WITH clause
) {
	// collect all WITH clauses
	rax *exps = raxNew();
	rax *aliases = raxNew();

	// check if WITH caluse has filters
	const cypher_astnode_t *pred = cypher_ast_with_get_predicate(with);
	if(pred == NULL) return false;

	//--------------------------------------------------------------------------
	// collect projected identifiers & aliases
	//--------------------------------------------------------------------------

	// WITH a AS b, c
	// exps will contain: 'a', 'c'
	// aliases will contain: 'b', 'c'
	collectProjectedIdentifiers(with, exps, aliases);

	//--------------------------------------------------------------------------
	// replace predicate identifiers
	//--------------------------------------------------------------------------

	// WHERE a.v = 1
	// identifiers will contain: 'a'
	bool res = adjustPredicateIdentifiers(pred, exps, aliases);

	raxFree(exps);
	raxFree(aliases);

	return res;
}

// AST_RewriteFilters rewrites filters which tries to access a variable which is
// aliased but is no longer available
//
// e.g.
//
// WITH a AS b WHERE a.v = 1
// ->
// WITH a AS b WHERE b.v = 1
//
// from user perspective 'a' still exists, but internally only 'b' is available
//
// in case such as:
// WITH a AS b, c AS a WHERE a.v = 1
// we do not perform the rewrite as 'a' is defined and acceessible
bool AST_RewriteFilters
(
	const cypher_astnode_t *root // root of AST
) {
	return false;
	ASSERT(root != NULL);

	bool res = false;

	if(cypher_astnode_type(root) != CYPHER_AST_STATEMENT) {
		return res;
	}

	// retrieve the root's body
	cypher_astnode_t *query =
		(cypher_astnode_t *)cypher_ast_statement_get_body(root);

	if(cypher_astnode_type(query) != CYPHER_AST_QUERY) {
		return res;
	}

	uint nclauses = cypher_ast_query_nclauses(query);
	for(uint i = 0; i < nclauses; i++) {
		cypher_astnode_t *clause =
			(cypher_astnode_t *) cypher_ast_query_get_clause(query, i);
		cypher_astnode_type_t type = cypher_astnode_type(clause);
		if(type == CYPHER_AST_WITH) {
			res |= _AST_RewriteFilters(clause);
		}
	}

	return res;
}

