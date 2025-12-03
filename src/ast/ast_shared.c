/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ast_shared.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../graph/graph_hub.h"
#include "../util/rax_extensions.h"
#include "../arithmetic/arithmetic_expression_construct.h"

AST_Operator AST_ConvertOperatorNode
(
	const cypher_operator_t *op
) {
	// TODO: ordered by precedence
	// which I don't know if we're managing properly right now
	if(op == CYPHER_OP_OR) {
		return OP_OR;
	} else if(op == CYPHER_OP_XOR) {
		return OP_XOR;
	} else if(op == CYPHER_OP_AND) {
		return OP_AND;
	} else if(op == CYPHER_OP_NOT) {
		// Unary, maybe doesn't belong here
		return OP_NOT;
	} else if(op == CYPHER_OP_EQUAL) {
		return OP_EQUAL;
	} else if(op == CYPHER_OP_NEQUAL) {
		return OP_NEQUAL;
	} else if(op == CYPHER_OP_LT) {
		return OP_LT;
	} else if(op == CYPHER_OP_GT) {
		return OP_GT;
	} else if(op == CYPHER_OP_LTE) {
		return OP_LE;
	} else if(op == CYPHER_OP_GTE) {
		return OP_GE;
	} else if(op == CYPHER_OP_PLUS) {
		return OP_PLUS;
	} else if(op == CYPHER_OP_MINUS) {
		return OP_MINUS;
	} else if(op == CYPHER_OP_MULT) {
		return OP_MULT;
	} else if(op == CYPHER_OP_DIV) {
		return OP_DIV;
	} else if(op == CYPHER_OP_MOD) {
		return OP_MOD;
	} else if(op == CYPHER_OP_POW) {
		return OP_POW;
	} else if(op == CYPHER_OP_CONTAINS) {
		return OP_CONTAINS;
	} else if(op == CYPHER_OP_STARTS_WITH) {
		return OP_STARTSWITH;
	} else if(op == CYPHER_OP_ENDS_WITH) {
		return OP_ENDSWITH;
	} else if(op == CYPHER_OP_IN) {
		return OP_IN;
	} else if(op == CYPHER_OP_IS_NULL) {
		return OP_IS_NULL;
	} else if(op == CYPHER_OP_IS_NOT_NULL) {
		return OP_IS_NOT_NULL;
	}

	return -1;
}

// convert a map of properties from the AST into a set of attribute ID keys
// and AR_ExpNode values
PropertyMap *PropertyMap_New
(
	const cypher_astnode_t *props  // AST properties
) {
	// no properties
	if (props == NULL) {
		return NULL;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	// TODO: add parameter support
	ASSERT (cypher_astnode_type (props) == CYPHER_AST_MAP) ;

	uint prop_count = cypher_ast_map_nentries (props) ;

	PropertyMap *m = rm_malloc (sizeof (PropertyMap)) ;

	// allocate arrays
	m->keys     = array_new (const char *, prop_count) ;
	m->values   = array_new (AR_ExpNode *, prop_count) ;
	m->attr_ids = array_new (AttributeID,  prop_count) ;

	//--------------------------------------------------------------------------
	// process AST props
	//--------------------------------------------------------------------------

	for (uint prop_idx = 0; prop_idx < prop_count; prop_idx++) {
		uint insert_idx = prop_idx;

		const cypher_astnode_t *ast_key =
			cypher_ast_map_get_key (props, prop_idx) ;

		const char *attr =
			cypher_ast_prop_name_get_value (ast_key) ;

		const cypher_astnode_t *ast_value =
			cypher_ast_map_get_value (props, prop_idx) ;

		AR_ExpNode *exp =
			AR_EXP_FromASTNode (ast_value) ;

		//----------------------------------------------------------------------
		// handle duplicates
		//----------------------------------------------------------------------

		uint count = array_len (m->keys) ;
		for (uint i = 0; i < count; i++) {
			if (strcmp (attr, m->keys[i]) == 0) {
				// duplicate!
				insert_idx = i ;
				break ;
			}
		}

		if (insert_idx == prop_idx) {
			// new entry
			array_append (m->keys, attr) ;
			array_append (m->values, exp) ;
			//array_append (m->attr_ids, GraphContext_GetAttributeID (gc, attr)) ;

			// NOTE: might introduce a change to the graph schema!
			array_append (m->attr_ids,
					GraphHub_FindOrAddAttribute (gc, attr, true)) ;

		} else {
			// replace duplicate
			AR_EXP_Free (m->values[insert_idx]) ;
			m->values[insert_idx] = exp ;
		}
	}

	return m ;
}

// clone a property map
static PropertyMap *_PropertyMap_Clone
(
	const PropertyMap *map
) {
	PropertyMap *clone = rm_malloc (sizeof (PropertyMap)) ;

	array_clone (clone->keys, map->keys) ;
	array_clone (clone->attr_ids, map->attr_ids) ;
	array_clone_with_cb (clone->values, map->values, AR_EXP_Clone) ;

	return clone ;
}

void PropertyMap_Free
(
	PropertyMap *map
) {
	if (map == NULL) {
		return ;
	}

	array_free (map->keys) ;
	array_free (map->attr_ids) ;
	array_free_cb (map->values, AR_EXP_Free) ;

	rm_free (map) ;
}

NodeCreateCtx NodeCreateCtx_Clone
(
	NodeCreateCtx ctx
) {
	NodeCreateCtx clone = ctx ;
	array_clone (clone.labels, ctx.labels) ;
	array_clone (clone.labelsId, ctx.labelsId) ;

	if (ctx.properties != NULL) {
		clone.properties = _PropertyMap_Clone (ctx.properties) ;
	}

	return clone ;
}

void NodeCreateCtx_Free
(
	NodeCreateCtx ctx
) {
	array_free (ctx.labels) ;
	array_free (ctx.labelsId) ;
	PropertyMap_Free (ctx.properties) ;
}

EdgeCreateCtx EdgeCreateCtx_Clone
(
	EdgeCreateCtx ctx
) {
	EdgeCreateCtx clone = ctx ;

	if(ctx.properties != NULL) {
		clone.properties = _PropertyMap_Clone (ctx.properties) ;
	}

	return clone ;
}

EntityUpdateEvalCtx *UpdateCtx_New
(
	const char *alias
) {
	EntityUpdateEvalCtx *ctx = rm_malloc(sizeof(EntityUpdateEvalCtx));

	ctx->alias         = alias;
	ctx->record_idx    = INVALID_INDEX;
	ctx->properties    = array_new(PropertySetCtx, 1);
	ctx->add_labels    = NULL;
	ctx->remove_labels = NULL;

	return ctx;
}

EntityUpdateEvalCtx *UpdateCtx_Clone
(
	const EntityUpdateEvalCtx *orig
) {
	EntityUpdateEvalCtx *clone = rm_malloc(sizeof(EntityUpdateEvalCtx));

	uint count = array_len(orig->properties);

	clone->alias         = orig->alias;
	clone->record_idx    = orig->record_idx;
	clone->properties    = array_new(PropertySetCtx, count);
	clone->add_labels    = NULL;
	clone->remove_labels = NULL;
	if(orig->add_labels != NULL) {
		array_clone(clone->add_labels, orig->add_labels);
	}
	if(orig->remove_labels != NULL) {
		array_clone(clone->remove_labels, orig->remove_labels);
	}

	for(uint i = 0; i < count; i ++) {
		PropertySetCtx update = {
			.exp       = AR_EXP_Clone(orig->properties[i].exp),
			.mode      = orig->properties[i].mode,
			.attr_id   = orig->properties[i].attr_id,
			.attr_name = orig->properties[i].attr_name,
		};
		array_append(clone->properties, update);
	}

	return clone;
}

void UpdateCtx_Clear
(
	EntityUpdateEvalCtx *ctx
) {
	uint count = array_len(ctx->properties);
	for(uint i = 0; i < count; i ++) AR_EXP_Free(ctx->properties[i].exp);
	array_clear(ctx->properties);
}

void UpdateCtx_Free
(
	EntityUpdateEvalCtx *ctx
) {
	uint count = array_len(ctx->properties);
	for(uint i = 0; i < count; i ++) {
		AR_EXP_Free(ctx->properties[i].exp);
	}

	array_free(ctx->properties);
	if(ctx->add_labels != NULL) array_free(ctx->add_labels);
	if(ctx->remove_labels != NULL) array_free(ctx->remove_labels);

	rm_free(ctx);
}

static void _collect_aliases_in_path
(
	const cypher_astnode_t *path,
	rax *identifiers
) {
	// collect path name if, if exists
	if(cypher_astnode_type(path) == CYPHER_AST_NAMED_PATH) {
		const cypher_astnode_t *ast_alias =
			cypher_ast_named_path_get_identifier(path);
		if(ast_alias != NULL) {
			const char *identifier = cypher_ast_identifier_get_name(ast_alias);
			raxTryInsert(identifiers, (unsigned char *)identifier,
				strlen(identifier), (void *)ast_alias, NULL);
		}
	}

	uint path_len = cypher_ast_pattern_path_nelements(path);
	// every even offset corresponds to a node
	for(uint i = 0; i < path_len; i += 2) {
		const cypher_astnode_t *ast_node =
			cypher_ast_pattern_path_get_element(path, i);
		const cypher_astnode_t *ast_alias =
			cypher_ast_node_pattern_get_identifier(ast_node);

		if(ast_alias == NULL) continue;  // unaliased node, do nothing

		// add node alias to projection rax
		const char *identifier = cypher_ast_identifier_get_name(ast_alias);
		raxTryInsert(identifiers, (unsigned char *)identifier,
			strlen(identifier), (void *)ast_alias, NULL);
	}

	// every odd offset corresponds to an edge
	for(uint i = 1; i < path_len; i += 2) {
		const cypher_astnode_t *ast_edge =
			cypher_ast_pattern_path_get_element(path, i);
		const cypher_astnode_t *ast_alias =
			cypher_ast_rel_pattern_get_identifier(ast_edge);

		if(ast_alias == NULL) continue;  // unaliased edge, do nothing

		// add edge alias to projection rax
		const char *identifier = cypher_ast_identifier_get_name(ast_alias);
		raxTryInsert(identifiers, (unsigned char *)identifier,
			strlen(identifier), (void *)ast_alias, NULL);
	}
}

static void _collect_aliases_in_pattern
(
	const cypher_astnode_t *pattern,
	rax *identifiers
) {
	uint path_count = cypher_ast_pattern_npaths(pattern);
	for(uint i = 0; i < path_count; i ++) {
		_collect_aliases_in_path(cypher_ast_pattern_get_path(pattern, i),
			identifiers);
	}
}

static void _collect_with_projections
(
	const cypher_astnode_t *with_clause,
	rax *identifiers
) {
	uint projection_count = cypher_ast_with_nprojections(with_clause);
	for(uint i = 0; i < projection_count; i ++) {
		const cypher_astnode_t *projection =
			cypher_ast_with_get_projection(with_clause, i);
		const cypher_astnode_t *identifier_node =
			cypher_ast_projection_get_alias(projection);
		if(identifier_node == NULL) {
			// the projection was not aliased
			// so the projection itself must be an identifier
			identifier_node = cypher_ast_projection_get_expression(projection);
			ASSERT(cypher_astnode_type(identifier_node) == CYPHER_AST_IDENTIFIER);
		} else {
			// do not include empty projections, which may have been made to
			// handle the MATCH () WITH * case
			if(!strcmp(cypher_ast_identifier_get_name(identifier_node), "")) continue;
		}
		const char *identifier = cypher_ast_identifier_get_name(identifier_node);
		raxTryInsert(identifiers, (unsigned char *)identifier,
			strlen(identifier), (void *)identifier_node, NULL);
	}
}

static void _collect_call_projections(
	const cypher_astnode_t *call_clause,
	rax *identifiers
) {
	uint yield_count = cypher_ast_call_nprojections(call_clause);

	if(yield_count == 0) {
		// error if this is a RETURN clause with no aliases
		// e.g.
		// CALL db.indexes() RETURN *
		ErrorCtx_SetError(EMSG_RETURN_STAR_NO_VARIABLES);
		return;
	}

	for(uint i = 0; i < yield_count; i ++) {
		const cypher_astnode_t *projection = cypher_ast_call_get_projection(call_clause, i);
		const cypher_astnode_t *ast_exp = cypher_ast_projection_get_expression(projection);

		const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(projection);
		if(alias_node == NULL) alias_node = ast_exp;

		const char *identifier = cypher_ast_identifier_get_name(alias_node);
		raxTryInsert(identifiers, (unsigned char *)identifier,
			strlen(identifier), (void *)alias_node, NULL);
	}
}

// collect aliases from a CALL {} clause
static void _collect_call_subquery_projections
(
	const cypher_astnode_t *clause,
	rax *identifiers
) {
	// collect returned aliases from the subquery, if there are any
	const cypher_astnode_t *query =
		cypher_ast_call_subquery_get_query(clause);
	uint nclauses = cypher_ast_query_nclauses(query);
	const cypher_astnode_t *last_clause =
		cypher_ast_query_get_clause(query, nclauses - 1);
	bool is_returning = (cypher_astnode_type(last_clause) == CYPHER_AST_RETURN);
	if(!is_returning) {
		return;
	}

	// collect aliases from the RETURN clause
	uint projection_count = cypher_ast_return_nprojections(last_clause);
	for(uint i = 0; i < projection_count; i ++) {
		const cypher_astnode_t *projection =
			cypher_ast_return_get_projection(last_clause, i);
		const cypher_astnode_t *identifier_node =
			cypher_ast_projection_get_alias(projection);
		if(identifier_node == NULL) {
			// the projection was not aliased
			// so the projection itself must be an identifier
			identifier_node = cypher_ast_projection_get_expression(projection);
			ASSERT(cypher_astnode_type(identifier_node) == CYPHER_AST_IDENTIFIER);
		} else {
			// do not include empty projections, which may have been made to
			// handle the MATCH () WITH * case
			if(!strcmp(cypher_ast_identifier_get_name(identifier_node), "")) continue;
		}
		const char *identifier = cypher_ast_identifier_get_name(identifier_node);
		raxTryInsert(identifiers, (unsigned char *)identifier,
			strlen(identifier), (void *)identifier_node, NULL);
	}
}

// collect aliases defined in a scope bounded by scope_start and scope_end
void collect_aliases_in_scope
(
	const cypher_astnode_t *root,  // the query root
	uint scope_start,              // start index of scope
	uint scope_end,                // end index of scope
	rax *identifiers               // rax to populate with identifiers
) {
	ASSERT(scope_start != scope_end);
	ASSERT(identifiers != NULL);

	for(uint i = scope_start; i < scope_end; i ++) {
		const cypher_astnode_t *clause = cypher_astnode_type(root) ==
										 CYPHER_AST_QUERY ?
			cypher_ast_query_get_clause(root, i) :
			cypher_ast_query_get_clause(
				cypher_ast_call_subquery_get_query(root), i);
		cypher_astnode_type_t type = cypher_astnode_type(clause);

		if(type == CYPHER_AST_WITH) {
			// the WITH clause contains either
			// aliases or its own STAR projection
			_collect_with_projections(clause, identifiers);
		} else if(type == CYPHER_AST_MATCH) {
			// the MATCH clause contains one pattern of N paths
			const cypher_astnode_t *pattern =
				cypher_ast_match_get_pattern(clause);
			_collect_aliases_in_pattern(pattern, identifiers);
		} else if(type == CYPHER_AST_CREATE) {
			// the CREATE clause contains one pattern of N paths
			const cypher_astnode_t *pattern =
				cypher_ast_create_get_pattern(clause);
			_collect_aliases_in_pattern(pattern, identifiers);
		} else if(type == CYPHER_AST_MERGE) {
			// the MERGE clause contains one path
			const cypher_astnode_t *path =
				cypher_ast_merge_get_pattern_path(clause);
			_collect_aliases_in_path(path, identifiers);
		} else if(type == CYPHER_AST_UNWIND) {
			// the UNWIND clause introduces one alias
			const cypher_astnode_t *unwind_alias =
				cypher_ast_unwind_get_alias(clause);
			const char *identifier =
				cypher_ast_identifier_get_name(unwind_alias);
			raxTryInsert(identifiers, (unsigned char *)identifier,
				strlen(identifier), (void *)unwind_alias, NULL);
		} else if(type == CYPHER_AST_CALL) {
			_collect_call_projections(clause, identifiers);
		} else if(type == CYPHER_AST_CALL_SUBQUERY) {
			_collect_call_subquery_projections(clause, identifiers);
		}
	}
}
