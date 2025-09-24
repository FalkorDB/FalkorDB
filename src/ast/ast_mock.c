/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "ast_mock.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"

AST *AST_MockMatchClause(AST *master_ast, cypher_astnode_t *node, bool node_is_path) {
	/* The AST node and its children must be reused directly,
	 * as cloning causes annotations (entity names) to be lost.
	 * TODO consider updating parser to improve this. */
	AST *ast = rm_malloc(sizeof(AST));
	ast->referenced_entities = master_ast->referenced_entities;
	ast->anot_ctx_collection = master_ast->anot_ctx_collection;
	ast->free_root = true;
	cypher_astnode_t *pattern;
	struct cypher_input_range range = {0};
	const cypher_astnode_t *predicate = NULL;
	uint child_count = (node_is_path) ? 1 : cypher_astnode_nchildren(node);
	cypher_astnode_t *children[child_count];

	if(node_is_path) {
		/* MERGE clauses and path filters are comprised of a single path.
		 * Construct a pattern node containing just this path. */
		pattern = cypher_ast_pattern(&node, 1, &node, 1, range);
		children[0] = pattern;
	} else {
		/* OPTIONAL MATCH clauses contain a pattern node
		 * and possibly a predicate node (containing WHERE conditions). */
		ASSERT(cypher_astnode_type(node) == CYPHER_AST_MATCH);
		pattern = (cypher_astnode_t *)cypher_ast_match_get_pattern(node);
		predicate = cypher_ast_match_get_predicate(node);

		// Explicitly collect all child nodes from the clause.
		for(uint i = 0; i < child_count; i ++) {
			children[i] = (cypher_astnode_t *)cypher_astnode_get_child(node, i);
		}
	}

	// Build a new match clause that holds this pattern.
	cypher_astnode_t *match_clause = cypher_ast_match(false, pattern, NULL, 0, predicate,
													  children, child_count, range);

	// Build a query node holding this clause.
	ast->root = cypher_ast_query(NULL, 0, &match_clause, 1, &match_clause, 1, range);

	QueryCtx_SetAST(ast); // Update the TLS.

	return ast;
}

void AST_MockFree(AST *ast, bool free_pattern) {
	/* When freeing the mock AST, we have to be careful to not free any shared node
	 * or its annotations. We'll free every surrounding layer explicitly - the MATCH
	 * pattern (if we constructed it), the MATCH clause, and finally the AST root. */
	const cypher_astnode_t *clause = cypher_ast_query_get_clause(ast->root, 0);
	ASSERT(cypher_astnode_type(clause) == CYPHER_AST_MATCH);
	if(free_pattern) {
		const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(clause);
		cypher_astnode_free((cypher_astnode_t *)pattern);
	}
	cypher_astnode_free((cypher_astnode_t *)clause);
	cypher_astnode_free((cypher_astnode_t *)ast->root);
	rm_free(ast);
}

