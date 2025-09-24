/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "annotate_projected_named_paths.h"
#include "../../util/arr.h"

static void _annotate_projected_named_path(AST *ast);

//------------------------------------------------------------------------------
//  Annotation context - Attach named path projection identifiers their respective ast structure
//------------------------------------------------------------------------------

static AnnotationCtx *_AST_NewProjectNamedPathContext(void) {
	AnnotationCtx *project_all_ctx = cypher_ast_annotation_context();
	return project_all_ctx;
}

// annotate identifier expression with a pointer to its referenced path
static void _attach_identifier
(
	rax *identifier_map,
	AnnotationCtx *named_paths_ctx,
	const cypher_astnode_t *path_identifier,
	const cypher_astnode_t *path
) {
	const char *path_name = cypher_ast_identifier_get_name(path_identifier);
	const cypher_astnode_t **exp_arr = raxFind(identifier_map,
		(unsigned char *)path_name, strlen(path_name));
	if(exp_arr != raxNotFound) {
		uint arrayLen = array_len(exp_arr);
		for(uint i = 0; i < arrayLen; i++)
			cypher_astnode_attach_annotation(named_paths_ctx, exp_arr[i],
				(void *)path, NULL);
	}
}

// recursively annotate all named paths in pattern comprehension nodes
static void _annotate_named_paths_in_expression
(
	AST *ast,
	rax *identifier_map,
	AnnotationCtx *named_paths_ctx,
	const cypher_astnode_t *exp
) {
	if(cypher_astnode_type(exp) == CYPHER_AST_PATTERN_COMPREHENSION) {
		const cypher_astnode_t *path_identifier =
		   	cypher_ast_pattern_comprehension_get_identifier(exp);
		if(path_identifier != NULL) {
			const cypher_astnode_t *path =
			   	cypher_ast_pattern_comprehension_get_pattern(exp);
			_attach_identifier(identifier_map, named_paths_ctx,
					path_identifier, path);
		}
	}

	// recursively visit children
	uint child_count = cypher_astnode_nchildren(exp);
	for(uint i = 0; i < child_count; i++) {
		const cypher_astnode_t *child = cypher_astnode_get_child(exp, i);
		_annotate_named_paths_in_expression(ast, identifier_map,
				named_paths_ctx, child);
	}
}


static void _annotate_relevant_projected_named_path_identifier
(
	AST *ast,
	rax *identifier_map,
	uint scope_start,
	uint scope_end
) {
	AnnotationCtx *named_paths_ctx =
		AST_AnnotationCtxCollection_GetNamedPathsCtx(ast->anot_ctx_collection);

	for(uint clause_iter = scope_start; clause_iter <= scope_end;
		clause_iter++) {
		const cypher_astnode_t *clause =
			cypher_ast_query_get_clause(ast->root, clause_iter);
		const cypher_astnode_type_t clause_type = cypher_astnode_type(clause);
		if(clause_type == CYPHER_AST_MATCH) {
			const cypher_astnode_t *pattern =
				cypher_ast_match_get_pattern(clause);
			uint path_count = cypher_ast_pattern_npaths(pattern);
			for(uint i = 0; i < path_count; i++) {
				const cypher_astnode_t *path =
					cypher_ast_pattern_get_path(pattern, i);
				if(cypher_astnode_type(path) == CYPHER_AST_NAMED_PATH) {
					const cypher_astnode_t *path_identifier =
						cypher_ast_named_path_get_identifier(path);
					_attach_identifier(identifier_map, named_paths_ctx,
						   	path_identifier, path);
				}
			}
		} else if(clause_type == CYPHER_AST_MERGE) {
			const cypher_astnode_t *path =
				cypher_ast_merge_get_pattern_path(clause);
			if(cypher_astnode_type(path) == CYPHER_AST_NAMED_PATH) {
				const cypher_astnode_t *path_identifier =
					cypher_ast_named_path_get_identifier(path);
				const char *path_name =
					cypher_ast_identifier_get_name(path_identifier);
				_attach_identifier(identifier_map, named_paths_ctx,
					path_identifier, path);
			}
		} else if(clause_type == CYPHER_AST_RETURN) {
			uint return_projection_count =
				cypher_ast_return_nprojections(clause);
			for(uint i = 0; i < return_projection_count; i++) {
				const cypher_astnode_t *projection =
					cypher_ast_return_get_projection(clause, i);
				const cypher_astnode_t *exp =
					cypher_ast_projection_get_expression(projection);
				_annotate_named_paths_in_expression(ast, identifier_map,
													named_paths_ctx, exp);
			}
		} else if(clause_type == CYPHER_AST_WITH) {
			uint with_projection_count = cypher_ast_with_nprojections(clause);
			for(uint i = 0; i < with_projection_count; i++) {
				const cypher_astnode_t *projection =
					cypher_ast_with_get_projection(clause, i);
				const cypher_astnode_t *exp =
					cypher_ast_projection_get_expression(projection);
				_annotate_named_paths_in_expression(ast, identifier_map,
													named_paths_ctx, exp);
			}
		}
	}
}

void _collect_projected_identifier
(
	const cypher_astnode_t *ast_exp,
	rax *identifier_map
) {
	cypher_astnode_type_t type = cypher_astnode_type(ast_exp);
	// In case of identifier.
	if(type == CYPHER_AST_IDENTIFIER) {
		const char *identifier = cypher_ast_identifier_get_name(ast_exp);
		const cypher_astnode_t **exp_arr =
			raxFind(identifier_map, (unsigned char *)identifier,
					strlen(identifier));
		// Use array in case of multiple projections for the same named path.
		if(exp_arr == raxNotFound) exp_arr =
			array_new(const cypher_astnode_t *, 1);
		array_append(exp_arr, ast_exp);
		raxInsert(identifier_map, (unsigned char *)identifier,
			strlen(identifier), (void *)exp_arr, NULL);
	} else {
		// Recurse over children.
		uint child_count = cypher_astnode_nchildren(ast_exp);
		for(uint i = 0; i < child_count; i++) {
			const cypher_astnode_t *child =
				cypher_astnode_get_child(ast_exp, i);
			// Recursively continue mapping.
			_collect_projected_identifier(child, identifier_map);
		}
	}
}

static void _annotate_with_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *with_clause,
	uint scope_start,
	uint scope_end
) {
	// Iterate over the projections and find their identifiers.
	rax *identifier_map = raxNew();
	uint with_projection_count = cypher_ast_with_nprojections(with_clause);
	for(uint projection_iter = 0; projection_iter < with_projection_count;
		projection_iter++) {
		const cypher_astnode_t *projection =
			cypher_ast_with_get_projection(with_clause, projection_iter);
		_collect_projected_identifier(projection, identifier_map);
	}
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_delete_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *delete_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();
	uint delete_exp_count = cypher_ast_delete_nexpressions(delete_clause);
	for(uint exp_iter = 0; exp_iter < delete_exp_count; exp_iter++) {
		const cypher_astnode_t *exp =
			cypher_ast_delete_get_expression(delete_clause, exp_iter);
		_collect_projected_identifier(exp, identifier_map);
	}
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_unwind_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *unwind_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();
	const cypher_astnode_t *exp =
		cypher_ast_unwind_get_expression(unwind_clause);
	_collect_projected_identifier(exp, identifier_map);

	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_return_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *return_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();
	uint return_projection_count =
		cypher_ast_return_nprojections(return_clause);
	for(uint projection_iter = 0; projection_iter < return_projection_count;
		projection_iter++) {
		const cypher_astnode_t *projection =
			cypher_ast_return_get_projection(return_clause, projection_iter);
		_collect_projected_identifier(projection, identifier_map);
	}
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_match_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *match_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();
	_collect_projected_identifier(match_clause, identifier_map);
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_foreach_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *foreach_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();
	// collect identifiers from list expression
	const cypher_astnode_t *exp =
		cypher_ast_foreach_get_expression(foreach_clause);
	_collect_projected_identifier(exp, identifier_map);

	// annotate named paths for the list expression only, as it can be defined
	// before the FOREACH clause only.
	_annotate_relevant_projected_named_path_identifier(ast,
		identifier_map, scope_start, scope_end);
	// remove the list expression from the identifier_map
	raxFreeWithCallback(identifier_map, array_free);
	identifier_map = raxNew();

	cypher_astnode_t **clauses = array_new(cypher_astnode_t *, 1);
	uint nclauses = cypher_ast_foreach_nclauses(foreach_clause);
	for(uint i = 0; i < nclauses; i++) {
		array_append(clauses,
		(cypher_astnode_t *)cypher_ast_foreach_get_clause(foreach_clause, i));
	}
	struct cypher_input_range range = {0};
	cypher_astnode_t *query_node =
		cypher_ast_query(NULL, 0, (cypher_astnode_t *const *)clauses, nclauses,
		clauses, nclauses, range);

	AST subquery_clauses_ast = {
		.root = query_node,
		.anot_ctx_collection = ast->anot_ctx_collection,
		.referenced_entities = ast->referenced_entities
	};

	// collect identifiers from body
	for(uint i = 0; i < nclauses; i++) {
		exp = cypher_ast_foreach_get_clause(foreach_clause, i);
		_collect_projected_identifier(exp, identifier_map);
	}
	// annotate named paths referring the outer scope
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);
	raxFreeWithCallback(identifier_map, array_free);

	// annotate named paths defined inside the body
	_annotate_projected_named_path(&subquery_clauses_ast);
	array_free(clauses);
	cypher_astnode_free(query_node);
}

static void _annotate_callsubquery_clause_projected_named_path
(
	AST *ast,
	const cypher_astnode_t *callsubquery_clause,
	uint scope_start,
	uint scope_end
) {
	rax *identifier_map = raxNew();

	const cypher_astnode_t *subquery = cypher_ast_call_subquery_get_query(
		callsubquery_clause);

	AST subquery_clauses_ast = {
		.root = subquery,
		.anot_ctx_collection = ast->anot_ctx_collection,
		.referenced_entities = ast->referenced_entities
	};

	// collect identifiers from importing WITH clauses (if exist)
	// annotate them only. the later references of imported paths with already
	// have the value in the record
	uint *union_indices = AST_GetClauseIndices(&subquery_clauses_ast,
		CYPHER_AST_UNION);
	uint n_union_clauses = array_len(union_indices);
	// handle first `UNION` branch
	const cypher_astnode_t *first_in_branch =
		cypher_ast_query_get_clause(subquery, 0);
	if(cypher_astnode_type(first_in_branch) == CYPHER_AST_WITH) {
		_collect_projected_identifier(first_in_branch, identifier_map);
	}
	// handle rest of `UNION` branches
	for(uint i = 0; i < n_union_clauses; i++) {
		first_in_branch =
			cypher_ast_query_get_clause(subquery, union_indices[i] + 1);
		if(cypher_astnode_type(first_in_branch) == CYPHER_AST_WITH) {
			_collect_projected_identifier(first_in_branch, identifier_map);
		}
	}
	array_free(union_indices);

	// annotate named paths referring the outer scope
	_annotate_relevant_projected_named_path_identifier(ast, identifier_map,
													   scope_start, scope_end);

	// annotate named paths defined inside the body
	_annotate_projected_named_path(&subquery_clauses_ast);
	raxFreeWithCallback(identifier_map, array_free);
}

static void _annotate_projected_named_path(AST *ast) {
	// TODO: find a better approach for PATH annotation.
	uint scope_end;
	uint scope_start = 0;
	uint clause_count = cypher_ast_query_nclauses(ast->root);
	for(uint i = 0; i < clause_count; i++) {
		scope_end = i;
		const cypher_astnode_t *child =
			cypher_ast_query_get_clause(ast->root, i);
		if(cypher_astnode_type(child) == CYPHER_AST_WITH) {
			_annotate_with_clause_projected_named_path(ast, child, scope_start,
													   scope_end);
			// update scope
			scope_start = scope_end;
		} else if(cypher_astnode_type(child) == CYPHER_AST_RETURN) {
			_annotate_return_clause_projected_named_path(ast, child,
														scope_start, scope_end);
			// update scope
			scope_start = scope_end;
		} else if(cypher_astnode_type(child) == CYPHER_AST_DELETE) {
			_annotate_delete_clause_projected_named_path(ast, child,
														scope_start, scope_end);
		} else if(cypher_astnode_type(child) == CYPHER_AST_UNWIND) {
			_annotate_unwind_clause_projected_named_path(ast, child,
														scope_start, scope_end);
		} else if(cypher_astnode_type(child) == CYPHER_AST_MATCH) {
			_annotate_match_clause_projected_named_path(ast, child,
														scope_start, scope_end);
		} else if(cypher_astnode_type(child) == CYPHER_AST_FOREACH) {
			_annotate_foreach_clause_projected_named_path(ast,
				child, scope_start, scope_end);
		} else if(cypher_astnode_type(child) == CYPHER_AST_CALL_SUBQUERY) {
			_annotate_callsubquery_clause_projected_named_path(ast, child,
				scope_start, scope_end);
		}
	}
}

void AST_AnnotateNamedPaths(AST *ast) {
	AST_AnnotationCtxCollection_SetNamedPathsCtx(ast->anot_ctx_collection,
											_AST_NewProjectNamedPathContext());
	_annotate_projected_named_path(ast);
}
