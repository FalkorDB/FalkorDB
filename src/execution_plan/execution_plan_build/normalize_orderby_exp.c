/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../errors/errors.h"
#include "../../util/rax_extensions.h"
#include "../../arithmetic/arithmetic_expression.h"

#include <string.h>

// checks if var is projected
static bool is_projected_var
(
	const char *var,         // variable in question
	const char **projected,  // projected aliases
	uint proj_count          // number of projected aliases
) {
	ASSERT (var       != NULL) ;
	ASSERT (projected != NULL) ;

	// see if var is a projected alias
	for (uint i = 0; i < proj_count; i++) {
		if (strcmp (var, projected[i]) == 0) {
			return true ;
		}
	}

	return false ;
}

// checks if `expr` is in `projections`
// return the first projected expression that's equals to `expr`
// or NULL if there's isn't such expression
static const AR_ExpNode *find_projection
(
	const AR_ExpNode **projections,  // projected expressions
	uint proj_count,                 // number of projected expressions
	const AR_ExpNode *expr           // expression to compare against
) {
	ASSERT (expr        != NULL) ;
	ASSERT (projections != NULL) ;	

	for (uint i = 0; i < proj_count; i++) {
		if (AR_EXP_Equals (expr, projections[i])) {
			return projections[i] ;
		}
	}

	return NULL ;
}

// collect all mentioned entities within `expr`
// caller is responsible for freeing the returned array and its elements
static char **collect_expr_vars
(
	const AR_ExpNode *expr  // expression to collect entities from
) {
		rax *entities = raxNew () ;
		AR_EXP_CollectEntities (expr, entities) ;

		char **vars = (char **)raxKeys (entities) ;
		raxFree (entities) ;

		ASSERT (vars != NULL) ;
		return vars ;
}

// collect projected aliases
// in addition to building an entity alias map
// e.g.
// WITH v.z = AS z, x AS y
// projections[0] = 'z'
// projections[1] = 'y'
// out_alias_map_keys[0] = 'x'
// out_alias_map_vals[0] = 'y'
static void build_alias_map
(
	const AR_ExpNode **projections,     // projected expressions
	uint proj_count,                    // number of expressions
	const char ***out_projected_names,  // [output] projected aliases
	const char ***out_alias_map_keys,   // [output] projected variables
	const char ***out_alias_map_vals,   // [output] projected variables aliases
	int *out_alias_map_size             // [output] size of out_alias_map_size
) {
	ASSERT (projections         != NULL) ;
	ASSERT (out_alias_map_keys  != NULL) ;
	ASSERT (out_alias_map_vals  != NULL) ;
	ASSERT (out_alias_map_size  != NULL) ;
	ASSERT (out_projected_names != NULL) ;

	*out_alias_map_vals = 0 ;
	const char **alias_map_keys  = rm_malloc (proj_count * sizeof (const char *)) ;
	const char **alias_map_vals  = rm_malloc (proj_count * sizeof (const char *)) ;
	const char **projected_names = rm_malloc (proj_count * sizeof (const char *)) ;

	for (uint i = 0; i < proj_count; i++) {
		const AR_ExpNode *exp = projections[i] ;
		projected_names[i] = exp->resolved_name ;

		if (AR_EXP_IsVariadic (exp)) {
			// WITH n AS m
			// n is aliased as m
			alias_map_keys[*out_alias_map_size] =
				exp->operand.variadic.entity_alias ;

			alias_map_vals[*out_alias_map_size] =
				exp->resolved_name ;

			*out_alias_map_size += 1 ;
		}
	}

	// set outputs
	*out_alias_map_keys  = alias_map_keys  ;
	*out_alias_map_vals  = alias_map_vals  ;
	*out_projected_names = projected_names ;
}

// normalize ORDER BY expressions
// if possible bring expressions to refer only to projected aliases
// transformation examples:
// 
// WITH a AS b, c.x AS X
// ORDER BY a.v, c.x
// 
// becomes:
//
// WITH a AS b, c.x AS X
// ORDER BY b.v, X
//
// in some cases this might not be possible
// e.g.
//
// MATCH (n), (m)
// WITH n.v AS V
// ORDER BY m.x + V
// 
// in such cases projected aliases would revert to their origin
//
// MATCH (n), (m)
// WITH n.v AS V
// ORDER BY m.x + n.v
void normalize_sort_exps
(
	AR_ExpNode ***projections,  // projected expressions
	AR_ExpNode **sort_exps,     // ORDER BY expressions to normalize
	bool aggregate              // true if this is an aggregation context
) {
    ASSERT (sort_exps   != NULL) ;
	ASSERT (projections != NULL && *projections != NULL) ;

	const AR_ExpNode **_projections = (const AR_ExpNode**)(*projections) ;
    int sort_count = array_len (sort_exps) ;
    int proj_count = array_len (_projections) ;

	ASSERT (sort_count > 0) ;
	ASSERT (proj_count > 0) ;

    // collect projected aliases and alias map
    int alias_count              = 0;
    const char **alias_keys      = NULL ;
    const char **alias_vals      = NULL ;
    const char **projected_names = NULL ;

    build_alias_map (_projections, proj_count, &projected_names, &alias_keys,
			&alias_vals, &alias_count) ;

	// process each ORDER BY expression
	for (int si = 0; si < sort_count; si++) {
		AR_ExpNode *sort_exp = sort_exps[si] ;

		//----------------------------------------------------------------------
		// check if any variables are not projected
		//----------------------------------------------------------------------

		char **vars = collect_expr_vars (sort_exp) ;
		int n_vars = array_len (vars) ;
		bool has_nonprojected_vars = false ;

		for (int vi = 0; vi < n_vars; vi++) {
			if (!is_projected_var (vars[vi], projected_names, proj_count)) {
				has_nonprojected_vars = true;
				break;
			}
		}

		// every alias in order-by expression is projected
		if (has_nonprojected_vars == false) {
			goto cleanup ;
		}

		// case A: entire order-by expression equals some projection
		// e.g.
		// WITH a.v AS V
		// ORDER BY a.v
		//
		// replace with projected alias
		//
		// WITH a.v AS V
		// ORDER BY V

		const AR_ExpNode *proj_eq =
			find_projection (_projections, proj_count, sort_exp) ;

		if (proj_eq != NULL) {
			// replace order-by expression with a new simpler one
			AR_EXP_Free (sort_exp) ;
			const char *alias_name = proj_eq->resolved_name ;
			sort_exps[si] = AR_EXP_NewVariableOperandNode (alias_name) ;
			sort_exps[si]->resolved_name = alias_name ;

			goto cleanup ;
		}

		// case B: remapping
		// e.g.
		//
		// WITH n AS m
		// ORDER BY n.v
		//
		// replace with alias
		//
		// WITH n AS m
		// ORDER BY m.v

		bool remap_all_possible = true ;

		// check whether every non-projected var is aliased
		for (int vi = 0; vi < n_vars; vi++) {
			const char *v = vars[vi] ;

			// skipped projected variables
			if (is_projected_var (v, projected_names, proj_count)) {
				continue ;
			}

			// non-projected, see if there's an alias
			bool has_alias = false ;
			for (int ak = 0; ak < alias_count; ak++) {
				if (strcmp (v, alias_keys[ak]) == 0) {
					has_alias = true ;
					break ;
				}
			}

			if (!has_alias) {
				remap_all_possible = false ;
				break ;
			}
		}

		if (remap_all_possible) {
			// perform remap
			// change each variable operand's entity_alias to alias_val

			// collect variabels operand nodes
			AR_ExpNode **var_ops = AR_EXP_CollectVariableOperands (sort_exp) ;

			for (int vo = 0; vo < array_len(var_ops); vo++) {
				AR_ExpNode *node = var_ops[vo] ;
				const char *entity_alias = node->operand.variadic.entity_alias ;

				// if entity_alias is projected, skip
				if (is_projected_var (entity_alias, projected_names, proj_count)) {
					continue ;
				}

				int ak = 0 ;
				for (; ak < alias_count; ak++) {
					if (strcmp (entity_alias, alias_keys[ak]) == 0) {
						// remap
						node->operand.variadic.entity_alias = alias_vals[ak] ;
						break ;
					}
				}
				ASSERT (ak < alias_count) ;  // remapping should be possible
			}

			array_free(var_ops) ;
			goto cleanup ;
		}

		// case C: last resort
		// replace projected aliases with their original expressions
		// e.g.
		//
		// MATCH (a), (b)
		// WITH b.v AS X
		// ORDER BY a.v + X
		//
		// rewritten as:
		//
		// MATCH (a), (b)
		// WITH b.v AS X, a.v + b.v
		// ORDER BY a.v, a.v + b.v
		if (aggregate) {
			// in aggregation, ORDER BY cannot reference unprojected variables
			// as it would introduce unintended grouping keys
			ErrorCtx_SetError ("ORDER BY cannot reference variables not "
					"projected or grouped in this scope") ;
			goto cleanup ;
		}

		AR_ExpNode **var_ops = AR_EXP_CollectVariableOperands (sort_exp) ;
		for (int vo = 0; vo < array_len(var_ops); vo++) {
			AR_ExpNode *vnode = var_ops[vo] ;
			const char *alias = vnode->operand.variadic.entity_alias ;

			// if alias names a projected expression
			// replace node with a clone of projected expression
			for (int pj = 0; pj < proj_count; pj++) {
				const AR_ExpNode *proj = _projections[pj] ;
				if (strcmp (alias, proj->resolved_name) == 0) {
					AR_EXP_Overwrite (vnode, proj) ;
					break;
				}
			}
		}
		array_free(var_ops);

		// add modified sort expression to projections
		assert (false && "I think we need to clone `sort_exp` as not to share expressions") ;
		array_append (_projections, sort_exp) ;

cleanup:
		// free the char ** returned by collect_expr_vars
		for (int vix = 0; vix < n_vars; vix++) {
			rm_free (vars[vix]) ;
		}
		array_free (vars) ;

    } // end for each sort expr

	rm_free (alias_keys) ;
	rm_free (alias_vals) ;
	rm_free (projected_names) ;

	*projections = (AR_ExpNode**)_projections ;
}

