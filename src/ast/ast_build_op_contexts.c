/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "ast_build_op_contexts.h"
#include "ast_shared.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../util/rax_extensions.h"
#include "../arithmetic/arithmetic_expression_construct.h"
#include "../query_ctx.h"

static inline EdgeCreateCtx _NewEdgeCreateCtx
(
	GraphContext *gc,
	const QGEdge *e,
	const cypher_astnode_t *edge
) {
	const cypher_astnode_t *props = cypher_ast_rel_pattern_get_properties(edge);

	EdgeCreateCtx new_edge = {  .alias = e->alias,
								.relation = e->reltypes[0],
								.reltypeId = e->reltypeIDs[0],
								.properties = PropertyMap_New(props),
								.src = e->src->alias,
								.dest = e->dest->alias
							 };
	return new_edge;
}

static inline NodeCreateCtx _NewNodeCreateCtx
(
	GraphContext *gc,
	const QGNode *n,
	const cypher_astnode_t *ast_node
) {
	const cypher_astnode_t *ast_props = cypher_ast_node_pattern_get_properties(ast_node);

	NodeCreateCtx new_node;
	new_node.alias = n->alias;
	new_node.properties = PropertyMap_New(ast_props);
	array_clone(new_node.labels, n->labels);
	array_clone(new_node.labelsId, n->labelsID);

	return new_node;
}

// updates a single property or label
static void _ConvertUpdateItem
(
	GraphContext *gc,
	rax *updates,
	const cypher_astnode_t *update_item
) {
	ASSERT(gc          != NULL);
	ASSERT(updates     != NULL);
	ASSERT(update_item != NULL);

	const char                  *alias     = NULL;  // entity being updated
	const char                  *attribute = NULL;  // attribute being set
	const cypher_astnode_t      *prop_expr = NULL;
	const cypher_astnode_t      *ast_prop  = NULL;
	const cypher_astnode_t      *ast_key   = NULL;  // AST node attribute set
	const cypher_astnode_t      *ast_value = NULL;  // AST node value set
	const cypher_astnode_type_t type       = cypher_astnode_type(update_item);

	bool         set_labels    = false;
	bool         remove_labels = false;
	bool         remove_attr   = false;  // REMOVE n.v
	UPDATE_MODE  update_mode   = UPDATE_MERGE;

	//--------------------------------------------------------------------------
	// determine the type of assignment
	//--------------------------------------------------------------------------
	// 1. override  - MATCH (a) SET a = {v: 5}
	// 2. merge     - MATCH (a) SET a += {v: 5}
	// 3. update    - MATCH (a) SET a.v = 5
	// 4. label set - MATCH (a) SET a:Label1:Label2
	// 5. label del - MATCH (a) REMOVE a:Label1:Label2
	// 6. attr del  - MATCH (a) REMOVE a.v
	//--------------------------------------------------------------------------

	if(type == CYPHER_AST_SET_ALL_PROPERTIES) {
		// MATCH (a) SET a = {v: 5}
		update_mode = UPDATE_REPLACE;

		// alias
		prop_expr = cypher_ast_set_all_properties_get_identifier(update_item);
		ASSERT(cypher_astnode_type(prop_expr) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(prop_expr);
		// value
		ast_value = cypher_ast_set_all_properties_get_expression(update_item);
	} else if(type == CYPHER_AST_MERGE_PROPERTIES) {
		// MATCH (a) SET a += {v: 5}
		// alias
		prop_expr = cypher_ast_merge_properties_get_identifier(update_item);
		ASSERT(cypher_astnode_type(prop_expr) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(prop_expr);
		// value
		ast_value = cypher_ast_merge_properties_get_expression(update_item);
	} else if(type == CYPHER_AST_SET_PROPERTY) {
		// MATCH (a) SET a.v = 5

		// alias
		ast_prop = cypher_ast_set_property_get_property(update_item);
		prop_expr = cypher_ast_property_operator_get_expression(ast_prop);
		ASSERT(cypher_astnode_type(prop_expr) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(prop_expr);

		// attribute
		ast_key = cypher_ast_property_operator_get_prop_name(ast_prop);
		attribute = cypher_ast_prop_name_get_value(ast_key);

		// updated value
		ast_value = cypher_ast_set_property_get_expression(update_item);
	} else if(type == CYPHER_AST_SET_LABELS) {
		// MATCH (a) SET a:Label1:Label2
		const cypher_astnode_t *id =
			cypher_ast_set_labels_get_identifier(update_item);
		ASSERT(cypher_astnode_type(id) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(id);
		set_labels = true;
	} else if(type == CYPHER_AST_REMOVE_LABELS) {
		// MATCH (a) REMOVE a:Label1:Label2
		const cypher_astnode_t *id =
			cypher_ast_remove_labels_get_identifier(update_item);
		ASSERT(cypher_astnode_type(id) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(id);
		remove_labels = true;
	} else if(type == CYPHER_AST_REMOVE_PROPERTY) {
		// MATCH (a) REMOVE a.v
		remove_attr = true ;

		// alias
		ast_prop = cypher_ast_remove_property_get_property(update_item);
		prop_expr = cypher_ast_property_operator_get_expression(ast_prop);
		ASSERT(cypher_astnode_type(prop_expr) == CYPHER_AST_IDENTIFIER);
		alias = cypher_ast_identifier_get_name(prop_expr);

		// attribute
		ast_key = cypher_ast_property_operator_get_prop_name(ast_prop);
		attribute = cypher_ast_prop_name_get_value(ast_key);
	} else {
		ASSERT(false);
	}

	// see if we need to create an update context for updated entity
	int len = strlen(alias);
	EntityUpdateEvalCtx *ctx = raxFind(updates, (unsigned char *)alias, len);
	if(ctx == raxNotFound) {
		ctx = UpdateCtx_New(alias);
		raxInsert(updates, (unsigned char *)alias, len, ctx, NULL);
	} 

	//--------------------------------------------------------------------------
	// collect update information
	//--------------------------------------------------------------------------
	// 1. set of labels to add
	// 2. set of labels to remove
	// 3. attribute to set
	//--------------------------------------------------------------------------

	if(set_labels) {
		rax *labels      = raxNew();
		uint label_count = cypher_ast_set_labels_nlabels(update_item);

		if(ctx->add_labels == NULL) {
			ctx->add_labels = array_new(const char *, label_count);
		} else {
			uint count = array_len(ctx->add_labels);
			for (uint i = 0; i < count; i++) {
				const char *label = ctx->add_labels[i];
				raxInsert(labels, (unsigned char *)label, strlen(label)+1,
					NULL, NULL);
			}
		}

		for (uint i = 0; i < label_count; i++) {
			const cypher_astnode_t *label_node =
				cypher_ast_set_labels_get_label(update_item, i);
			const char* label = cypher_ast_label_get_name(label_node);
			if(raxTryInsert(labels, (unsigned char *)label, strlen(label)+1,
					NULL, NULL) != 0) {
				// mark label for addition
				array_append(ctx->add_labels, label);
			}
		}

		raxFree(labels);
	} else if(remove_labels) {
		rax *labels      = raxNew();
		uint label_count = cypher_ast_remove_labels_nlabels(update_item);

		if(ctx->remove_labels == NULL) {
			ctx->remove_labels = array_new(const char *, label_count);
		} else {
			uint count = array_len(ctx->remove_labels);
			for (uint i = 0; i < count; i++) {
				const char *label = ctx->remove_labels[i];
				raxInsert(labels, (unsigned char *)label, strlen(label)+1,
					NULL, NULL);
			}
		}

		for (uint i = 0; i < label_count; i++) {
			const cypher_astnode_t * label_node =
				cypher_ast_remove_labels_get_label(update_item, i);
			const char* label = cypher_ast_label_get_name(label_node);
			if(raxTryInsert(labels, (unsigned char *)label, strlen(label)+1,
					NULL, NULL) != 0) {
				// mark label for removal
				array_append(ctx->remove_labels, label);
			}
		}

		raxFree(labels);
	} else {
		if (update_mode == UPDATE_REPLACE) {
			UpdateCtx_Clear (ctx) ;
		}

		// updated value
		AR_ExpNode *exp ;

		if (ast_value != NULL) {
			// remove redundant updates
			// e.g. SET n.v = 1, n.v = 2
			if (attribute != NULL) {
				for (int i = 0; i < array_len (ctx->properties); i++) {
					const char *updated_attr = ctx->properties[i].attr_name ;
					if (updated_attr != NULL &&
						strcmp (updated_attr, attribute) == 0) {
						AR_EXP_Free (ctx->properties[i].exp) ;
						array_del_fast (ctx->properties, i) ;
						break ;
					}
				}
			}

			exp = AR_EXP_FromASTNode (ast_value) ;
		} else {
			ASSERT (remove_attr == true) ;
			// remove an attribute e.g. REMOVE a.v
			// this is done by performing a.v = NULL
			exp = AR_EXP_NewConstOperandNode (SI_NullVal ()) ;
		}

		PropertySetCtx update = {
			.exp       = exp,
			.mode      = update_mode,
			.attr_id   = ATTRIBUTE_ID_NONE,
			.attr_name = attribute
		};

		array_append (ctx->properties, update) ;
	}
}

void AST_PreparePathCreation
(
	const cypher_astnode_t *path,
	const QueryGraph *qg,
	rax *bound_vars,
	NodeCreateCtx **nodes,
	EdgeCreateCtx **edges
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	QueryGraph *g = QueryGraph_ExtractPaths(qg, &path, 1);
	uint path_elem_count = cypher_ast_pattern_path_nelements(path);
	for(uint i = 0; i < path_elem_count; i ++) {
		/* See if current entity needs to be created:
		 * 1. Current entity is NOT bound in a previous clause.
		 * 2. We have yet to account for this entity. */
		const cypher_astnode_t *elem = cypher_ast_pattern_path_get_element(path, i);
		const char *alias = AST_ToString(elem);

		// Skip entities defined in previous clauses or already represented in our nodes/edges arrays.
		int rc = raxTryInsert(bound_vars, (unsigned char *)alias, strlen(alias), NULL, NULL);
		if(rc == 0) continue;

		if((i % 2) == 1) {
			// relation
			QGEdge *e = QueryGraph_GetEdgeByAlias(g, alias);
			EdgeCreateCtx new_edge = _NewEdgeCreateCtx(gc, e, elem);
			array_append(*edges, new_edge);
		} else {
			// node
			QGNode *n = QueryGraph_GetNodeByAlias(g, alias);
			NodeCreateCtx new_node = _NewNodeCreateCtx(gc, n, elem);
			array_append(*nodes, new_node);
		}
	}

	QueryGraph_Free(g);
}

//------------------------------------------------------------------------------
// SORT operation
//------------------------------------------------------------------------------

// get direction of each sort operation
// append to an array
// return the array in the form of out parameter
void AST_PrepareSortOp
(
	const cypher_astnode_t *order_clause,
	int **sort_directions
) {
	ASSERT (order_clause    != NULL) ;
	ASSERT (sort_directions != NULL) ;

	unsigned int nitems = cypher_ast_order_by_nitems (order_clause) ;
	int *directions = array_new (int, nitems) ;

	for (unsigned int i = 0; i < nitems; i ++) {
		const cypher_astnode_t *item =
			cypher_ast_order_by_get_item (order_clause, i) ;

		int dir = cypher_ast_sort_item_is_ascending (item) ?
			DIR_ASC  :
			DIR_DESC ;

		array_append (directions, dir) ;
	}

	*sort_directions = directions ;
}

//------------------------------------------------------------------------------
// UNWIND operation
//------------------------------------------------------------------------------

AST_UnwindContext AST_PrepareUnwindOp
(
	const cypher_astnode_t *unwind_clause
) {
	const cypher_astnode_t *collection = cypher_ast_unwind_get_expression(unwind_clause);
	AR_ExpNode *exp = AR_EXP_FromASTNode(collection);
	exp->resolved_name = cypher_ast_identifier_get_name(cypher_ast_unwind_get_alias(unwind_clause));

	AST_UnwindContext ctx = { .exp = exp };
	return ctx;
}

//------------------------------------------------------------------------------
// DELETE operation
//------------------------------------------------------------------------------

AR_ExpNode **AST_PrepareDeleteOp
(
	const cypher_astnode_t *delete_clause
) {
	uint delete_count = cypher_ast_delete_nexpressions(delete_clause);
	AR_ExpNode **exps = array_new(AR_ExpNode *, delete_count);

	for(uint i = 0; i < delete_count; i ++) {
		const cypher_astnode_t *ast_expr =
			cypher_ast_delete_get_expression(delete_clause, i);
		AR_ExpNode *exp = AR_EXP_FromASTNode(ast_expr);
		array_append(exps, exp);
	}

	return exps;
}

//------------------------------------------------------------------------------
// MERGE operation
//------------------------------------------------------------------------------

AST_MergeContext AST_PrepareMergeOp
(
	const cypher_astnode_t *merge_clause,
	GraphContext *gc,
	QueryGraph *qg,
	rax *bound_vars
) {
	AST_MergeContext merge_ctx = { .nodes_to_merge = NULL,
								   .edges_to_merge = NULL,
								   .on_match = NULL,
								   .on_create = NULL
								 };

	// Prepare all create contexts for nodes and edges on Merge path.
	rax *on_match_items = NULL;
	rax *on_create_items = NULL;
	NodeCreateCtx *nodes_to_merge = array_new(NodeCreateCtx, 1);
	EdgeCreateCtx *edges_to_merge = array_new(EdgeCreateCtx, 1);
	const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(merge_clause);

	// Shouldn't operate on the original bound variables map, as this function may insert aliases.
	rax *bound_and_introduced_entities = (bound_vars) ? raxClone(bound_vars) : raxNew();
	AST_PreparePathCreation(path, qg, bound_and_introduced_entities, &nodes_to_merge, &edges_to_merge);
	raxFree(bound_and_introduced_entities);


	// Convert any ON MATCH and ON CREATE directives.
	uint directive_count = cypher_ast_merge_nactions(merge_clause);

	for(uint i = 0; i < directive_count; i ++) {
		const cypher_astnode_t *directive = cypher_ast_merge_get_action(merge_clause, i);
		cypher_astnode_type_t type = cypher_astnode_type(directive);

		if(type == CYPHER_AST_ON_CREATE) {
			uint create_prop_count = cypher_ast_on_create_nitems(directive);
			if(on_create_items == NULL) on_create_items = raxNew();
			for(uint j = 0; j < create_prop_count; j ++) {
				const cypher_astnode_t *create_item = cypher_ast_on_create_get_item(directive, j);
				_ConvertUpdateItem(gc, on_create_items, create_item);
			}
		} else if(type == CYPHER_AST_ON_MATCH) {
			uint match_prop_count = cypher_ast_on_match_nitems(directive);
			if(on_match_items == NULL) on_match_items = raxNew();
			for(uint j = 0; j < match_prop_count; j ++) {
				const cypher_astnode_t *match_item = cypher_ast_on_match_get_item(directive, j);
				_ConvertUpdateItem(gc, on_match_items, match_item);
			}
		} else {
			ASSERT(false);
		}
	}

	merge_ctx.on_match = on_match_items;
	merge_ctx.on_create = on_create_items;
	merge_ctx.edges_to_merge = edges_to_merge;
	merge_ctx.nodes_to_merge = nodes_to_merge;
	return merge_ctx;
}

//------------------------------------------------------------------------------
// UPDATE operation
//------------------------------------------------------------------------------

rax *AST_PrepareUpdateOp
(
	GraphContext *gc,
	const cypher_astnode_t *clause
) {
	cypher_astnode_type_t type = cypher_astnode_type(clause);
	ASSERT(type == CYPHER_AST_SET || type == CYPHER_AST_REMOVE);

	rax *updates = raxNew(); // entity alias -> EntityUpdateEvalCtx
	if(type == CYPHER_AST_SET) {
		uint nitems = cypher_ast_set_nitems(clause);
		for(uint i = 0; i < nitems; i++) {
			const cypher_astnode_t *set_item =
				cypher_ast_set_get_item(clause, i);
			_ConvertUpdateItem(gc, updates, set_item);
		}
	} else {
		uint nitems = cypher_ast_remove_nitems(clause);
		for(uint i = 0; i < nitems; i++) {
			const cypher_astnode_t *remove_item =
				cypher_ast_remove_get_item(clause, i);
			_ConvertUpdateItem(gc, updates, remove_item);
		}
	}

	return updates;
}

//------------------------------------------------------------------------------
// CREATE operation
//------------------------------------------------------------------------------

AST_CreateContext AST_PrepareCreateOp
(
	QueryGraph *qg,
	rax *bound_vars,
	const cypher_astnode_t *clause
) {
	// shouldn't operate on the original bound variables map
	// as this function may insert aliases
	rax *bound_and_introduced_entities = raxClone(bound_vars);

	NodeCreateCtx *nodes_to_create = array_new(NodeCreateCtx, 1);
	EdgeCreateCtx *edges_to_create = array_new(EdgeCreateCtx, 1);

	const cypher_astnode_t *pattern = cypher_ast_create_get_pattern(clause);
	uint npaths = cypher_ast_pattern_npaths(pattern);

	for(uint j = 0; j < npaths; j++) {
		const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, j);
		AST_PreparePathCreation(path, qg, bound_and_introduced_entities,
				&nodes_to_create, &edges_to_create);
	}

	raxFree(bound_and_introduced_entities);

	AST_CreateContext ctx = { .nodes_to_create = nodes_to_create,
		.edges_to_create = edges_to_create };

	return ctx;
}

