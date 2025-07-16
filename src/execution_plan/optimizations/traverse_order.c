/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../arithmetic/algebraic_expression/utils.h"
#include "traverse_order_utils.h"

#include <stdlib.h>

// having chosen which algebraic expression will be evaluated first
// determine whether it is worthwhile to transpose it
// thus swap the source and destination
static bool _should_transpose_entry_point
(
	const QueryGraph *qg,
	AlgebraicExpression *ae,
	rax *filtered_entities,
	rax *bound_vars
) {
	// validate inputs
	ASSERT(qg                 !=  NULL);
	ASSERT(ae                 !=  NULL);
	ASSERT(bound_vars         !=  NULL);


	// consider src and dest as stand-alone expressions
	const char *src  = AlgebraicExpression_Src(ae);
	const char *dest = AlgebraicExpression_Dest(ae);

	ScoredExp scored_exp[2];
	AlgebraicExpression *exps[2];
	exps[0] = AlgebraicExpression_NewOperand(GrB_NULL, false, src, src, NULL,
											 NULL);
	exps[1] = AlgebraicExpression_NewOperand(GrB_NULL, false, dest, dest, NULL,
											 NULL);

	// compute src score
	TraverseOrder_ScoreExpressions(scored_exp, exps, 2, bound_vars,
								   filtered_entities, qg);
	int src_score = scored_exp[0].score;

	// transpose
	AlgebraicExpression *tmp = exps[0];
	exps[0] = exps[1];
	exps[1] = tmp;

	// compute dest score
	TraverseOrder_ScoreExpressions(scored_exp, exps, 2, bound_vars,
								   filtered_entities, qg);
	int dest_score = scored_exp[0].score;

	// transpose if top scored expression is 'dest_exp'
	bool transpose = dest_score > src_score;

	AlgebraicExpression_Free(exps[0]);
	AlgebraicExpression_Free(exps[1]);

	return transpose;
}

// transpose out-of-order expressions such that each expresson's source
// is resolved in the winning arrangement
static void _resolve_winning_sequence
(
	AlgebraicExpression **exps,
	uint exp_count
) {
	// skip opening expression
	for(uint i = 1; i < exp_count; i ++) {
		bool src_resolved = false;
		AlgebraicExpression *exp = exps[i];
		const char *src = AlgebraicExpression_Src(exp);

		// see if source is already resolved
		for(int j = i - 1; j >= 0; j--) {
			AlgebraicExpression *prev_exp = exps[j];
			if(!strcmp(AlgebraicExpression_Src(prev_exp), src) ||
			   !strcmp(AlgebraicExpression_Dest(prev_exp), src)) {
				src_resolved = true;
				break;
			}
		}
		if(!src_resolved) AlgebraicExpression_Transpose(exps + i);
		// TODO: when DEBUG is enabled, make sure transposed expression
		// source (prev dest) is resolved
	}
}

// construct a sorted list of valid expressions to consider, given a subset of
// expression already in use 'arrangement' these will not show up in the
// returned list.
// the elements are sorted by their score
static AlgebraicExpression **_valid_expressions
(
	const ScoredExp *exps,             // input list of expressions
	uint nexp,                         // number of expressions
	AlgebraicExpression **restricted,  // expressions already in use
	uint nrestricted                   // number of elements in restricted
) {
	// sorted array of valid expressions to return
	AlgebraicExpression **options = array_new(AlgebraicExpression *, 0);

	for(int i = nexp - 1; i >= 0; i--) {
		// see if current expression is a valid expression to use
		// a valid expression is one which isn't already in use
		// and either its source or destination have been encountered
		bool valid = true;
		AlgebraicExpression *exp = exps[i].exp;

		// see if expression already in use
		for(uint j = 0; j < nrestricted && valid; j++) {
			valid = (exp != restricted[j]);
		}
		if(!valid) continue;

		// see if either exp source or destination were already encountered
		const char *src  = AlgebraicExpression_Src(exp);
		const char *dest = AlgebraicExpression_Dest(exp);

		for(uint j = 0; j < nrestricted; j++) {
			valid = false;
			AlgebraicExpression *used = restricted[j];
			const char *used_src = AlgebraicExpression_Src(used);
			const char *used_dest  = AlgebraicExpression_Dest(used);

			if(strcmp(src, used_src)   == 0  ||
			   strcmp(src, used_dest)  == 0  ||
			   strcmp(dest, used_src)  == 0  ||
			   strcmp(dest, used_dest) == 0) {
				valid = true;
				break;
			}
		}

		if(valid) array_append(options, exp);
	}

	return options;
}

static bool _arrangement_set_expression
(
	AlgebraicExpression **arrangement,   // arrangement of expressions
	const ScoredExp *exps,              // input list of expressions
	uint nexp,                          // number of expressions
	AlgebraicExpression **options,      // posible expressions for position i
	uint i                              // index in arrangement to resolve
) {
	// Done.
	if(i == nexp) {
		array_free(options);
		return true;
	}

	bool position_set = false;  // did we manage to resolve position i..n
	AlgebraicExpression **follows = NULL;

	//--------------------------------------------------------------------------
	// Find the best possible expression to place at position i
	//--------------------------------------------------------------------------

	// as long as we didn't find an expression for position i
	// and there are options to go through
	while(!position_set && array_len(options) > 0) {
		// options are sorted by score
		AlgebraicExpression *exp = array_pop(options);

		// set current expression in arrangement
		arrangement[i] = exp;

		// compose a list of valid expressions for next position
		follows = _valid_expressions(exps, nexp, arrangement, i + 1);

		// position i set, recursively advance to next position
		position_set = _arrangement_set_expression(arrangement, exps, nexp,
												   follows, i + 1);
	}

	array_free(options);
	return position_set;
}

static void _order_expressions
(
	AlgebraicExpression **arrangement,   // arrangement of expressions
	const ScoredExp *exps,               // input list of expressions
	uint nexp                            // number of expressions
) {
	// collect all possible expression for first position in arrangement
	AlgebraicExpression **options = _valid_expressions(exps, nexp, NULL, 0);

	// construct arrangement
	bool res = _arrangement_set_expression(arrangement, exps, nexp, options, 0);
	ASSERT(res == true);
}

static int _score_cmp
(
	const ScoredExp *a,
	const ScoredExp *b
) {
	return b->score - a->score;
}

// given a set of algebraic expressions representing a graph traversal
// we pick the order in which the expressions will be evaluated
// taking into account filters and transposes
// 'exps' will be reordered
void orderExpressions
(
	const QueryGraph *qg,
	AlgebraicExpression **exps,
	uint *exp_count,
	const FT_FilterNode *ft,
	rax *bound_vars
) {
	// Validate inputs
	ASSERT(qg          != NULL);
	ASSERT(exps        != NULL);
	ASSERT(exp_count   != NULL);

	uint _exp_count = *exp_count;
	ScoredExp scored_exps[_exp_count];
	AlgebraicExpression *arrangement[_exp_count];

	// collect all filtered aliases
	rax *filtered_entities = NULL;
	if(ft) {
		filtered_entities = FilterTree_CollectModified(ft);
		// enrich filtered_entities with independent filtered entities frequency
		FilterTree_CollectIndependentEntities(ft, filtered_entities);
	}

	//--------------------------------------------------------------------------
	// score each expression and sort
	//--------------------------------------------------------------------------

	// associate each expression with a score
	TraverseOrder_ScoreExpressions(scored_exps, exps, _exp_count, bound_vars,
								   filtered_entities, qg);

	// sort scored_exps on score in descending order
	qsort(scored_exps, _exp_count, sizeof(ScoredExp),
			(int(*)(const void*, const void*))_score_cmp);

	//--------------------------------------------------------------------------
	// Find the highest-scoring valid arrangement
	//--------------------------------------------------------------------------

	_order_expressions(arrangement, scored_exps, _exp_count);

	// overwrite the original expressions array with the optimal arrangement
	memcpy(exps, arrangement, _exp_count * sizeof(AlgebraicExpression *));

	// transpose expressions as necessary so that the traversals will work in
	// the selected order
	_resolve_winning_sequence(exps, _exp_count);

	// transpose the winning expression if the destination node is a more
	// efficient starting point
	if(_should_transpose_entry_point(qg, exps[0], filtered_entities,
									 bound_vars)) {
		AlgebraicExpression_Transpose(exps);
	}

	// remove redundent operands from expressions
	// MATCH (a:A)-[:R]->(b:B), (a)-[:R]->(c:C), (a:A)-[:R]->(d:D)
	// will result in 2 expressions:
	// 1. A*R*B
	// 2. A*R*C
	// 3. A*R*D
	//
	// as we're dealing with the same 'a' node there's no need to multiply
	// expressions 2 and 3 by 'A'
	//
	// RemoveRedundentOperands will remove label operands from already resolved
	// nodes, this might reduce numder of expressions in 'exps'
	_AlgebraicExpression_RemoveRedundentOperands(exps, qg);
	*exp_count = array_len(exps);

	if(filtered_entities) {
		raxFree(filtered_entities);
	}
}

