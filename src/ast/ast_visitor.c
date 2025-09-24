/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ast_visitor.h"
#include "util/rmalloc.h"
#include "ast_validations.h"

// recursively visit an ast-node
static VISITOR_STRATEGY _AST_Visitor_visit
(
	const cypher_astnode_t *node,
	ast_visitor *visitor
) {
	ASSERT(node    != NULL);
	ASSERT(visitor != NULL);

	// get ast node type
	cypher_astnode_type_t node_type = cypher_astnode_type(node);

	//--------------------------------------------------------------------------
	// opening call
	//--------------------------------------------------------------------------

	// first visit of the node
	VISITOR_STRATEGY state = visitor->mapping[node_type](node, true, visitor);
	if(state != VISITOR_RECURSE) {
		// do not visit children
		return state;
	}

	//--------------------------------------------------------------------------
	// recall for each child node
	//--------------------------------------------------------------------------

	uint nchildren = cypher_astnode_nchildren(node);
	for (uint i = 0; i < nchildren; i++) {
		if(_AST_Visitor_visit(cypher_astnode_get_child(node, i), visitor) ==
			VISITOR_BREAK) {
				// error occurred, fast fold
				return VISITOR_BREAK;
		}
	}

	//--------------------------------------------------------------------------
	// closing call
	//--------------------------------------------------------------------------

	return visitor->mapping[node_type](node, false, visitor);
}

// get the context of a visitor
void *AST_Visitor_GetContext
(
	const ast_visitor *visitor  // visitor
) {
	return visitor->ctx;
}

// visit an ast-node
void AST_Visitor_visit
(
	const cypher_astnode_t *node,  // node to visit
	ast_visitor *visitor           // visitor
) {
	ASSERT(node    != NULL);
	ASSERT(visitor != NULL);

	_AST_Visitor_visit(node, visitor);
}

// registers a function to a mapping
void AST_Visitor_mapping_register
(
	ast_visitor_mapping mapping, // visitor to register the function to
	cypher_astnode_type_t type,  // type of ast-node
	visit cb                     // visit function to register
) {
	ASSERT(cb      != NULL);
	ASSERT(mapping != NULL);
	ASSERT(mapping[type] == _default_visit);  // override default cb

	mapping[type] = cb;
}
