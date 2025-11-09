/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../ops/op_filter.h"
#include "../../errors/errors.h"
#include "../ops/op_cartesian_product.h"
#include "../../util/rax_extensions.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"
#include "../execution_plan_build/execution_plan_construct.h"

#include <stdlib.h>

// this struct is an auxilary struct for sorting filters according to their
// referenced entities count
typedef struct {
	OpFilter *filter;   // filter operation
	rax *entities;      // contains the entities that the filter references
} FilterCtx;

static inline int _FilterCtx_cmp
(
	const FilterCtx *a,
	const FilterCtx *b
) {
	return raxSize(a->entities) - raxSize(b->entities);
}

// this optimization takes multiple branched cartesian product
// (with more than two branches)
// followed by filter(s) and try to apply the filter as soon possible by
// locating situations where a new Cartesian Product of smaller amount of
// streams can resolve the filter for a filter F executing on a dual-branched
// cartesian product output, the runtime complexity is at most f=n^2
// for a filter F' which execute on a dual-branched cartesian product output
// where one of its branches is F, the overall time complexity is at most
// f'=f*n = n^3
// in the general case, the runtime complaxity of filter that is executing over
// the output of a cartesian product which all of its children are nested
// cartesian product followed by a filter (as a result of this optimization)
// is at most n^x where x is the number of branchs of the original cartesian
// product consider MATCH (a), (b), (c) where a.x > b.x RETURN a, b, c
// prior to this optimization a, b and c will be combined via a cartesian
// product O(n^3) because we require a.v > b.v we can create a cartesian
// product between a and b, and re-position the filter after this new cartesian
// product, remove both a and b branches from the original cartesian product and
// place the filter operation is a new branch creating nested cartesian products
// operations and re-positioning the filter op will:
// 1. potentially reduce memory consumption (storing only f records instead n^x)
// in each phase
// 2. reduce the overall filter runtime by potentially order(s) of magnitude


// free FilterCtx
static inline void _FilterCtx_Free
(
	FilterCtx *ctx
) {
	raxFree(ctx->entities);
}

// collects all consecutive filters beneath given op
// sort them by the number of referenced entities
// the array is soreted in order to reposition the filter that require smaller
// cartiesian products first
static FilterCtx *_locate_filters_and_entities
(
	OpBase *cp
) {
	OpBase *parent = cp->parent;
	FilterCtx *filter_ctx_arr = array_new(FilterCtx, 0);

	while(parent != NULL && parent->type == OPType_FILTER) {
		OpFilter *filter_op = (OpFilter *)parent;

		// advance to the next op
		parent = parent->parent;

		// collect referenced entities
		rax *entities = FilterTree_CollectModified(filter_op->filterTree);

		// continue if entities count is less than two
		if(unlikely(raxSize(entities) <= 1)) {
			raxFree(entities);
			continue;
		}

		FilterCtx filter_ctx = {.filter = filter_op, .entities = entities};
		array_append(filter_ctx_arr, filter_ctx);
	}

	// sort by the number of referenced entities
	qsort(filter_ctx_arr, array_len(filter_ctx_arr), sizeof(FilterCtx),
			(int(*)(const void*, const void*))_FilterCtx_cmp);
	return filter_ctx_arr;
}

// finds all the cartesian product's children which solve
// a specific filter entities
// returns an array of branches resolving entities
// caller is responsibe for freeing the array
static OpBase **_find_entities_solving_branches
(
	rax *entities,  // entities to locate
	OpBase *cp      // cartesian product operation
) {
	// validations
	ASSERT(cp       != NULL);
	ASSERT(entities != NULL);

	// get an array of aliases to locate
	char **aliases = (char**)raxKeys(entities);
	int n = array_len(aliases);

	// expecting at least 2 entities
	ASSERT(n >= 2);

	// array of branches resolving aliases
	OpBase **resolving_branches = array_new(OpBase *, 1);

	// iterate through all children or until all the aliases are resolved
	for(int i = 0; i < cp->childCount && array_len(aliases) > 0; i++) {
		bool add_branch = false;
		OpBase *branch  = cp->children[i];

		// scan through the remaining aliases
		for(int j = 0; j < array_len(aliases); j++) {
			char *alias = aliases[j];
			// see if current branch resolves alias
			if(OpBase_Aware(branch, (const char**)&alias, 1)) {
				// branch resolves alias
				// remove it from the aliases array
				// and mark branch for output addition
				rm_free(alias);
				array_del_fast(aliases, j);
				add_branch = true;
				j--;  // compensate for the alias removal
			}
		}

		// add branch to output
		if(add_branch) {
			array_append(resolving_branches, branch);
		}
	}

	// all entities should have been resolved, error otherwise
	n = array_len(aliases);
	array_free_cb(aliases, rm_free);

	if(n != 0) {
		Error_InvalidFilterPlacement(entities);
		array_free(resolving_branches);
		return NULL;
	}

	return resolving_branches;
}

static void _optimize_cartesian_product
(
	ExecutionPlan *plan,
	OpBase *cp
) {
	// retrieve all filter operations following the cartesian product
	FilterCtx *filter_ctx_arr = _locate_filters_and_entities(cp);
	uint filter_count = array_len(filter_ctx_arr);

	for(uint i = 0; i < filter_count; i++) {
		// try to create a cartesian product, followed by the current filter
		OpFilter *filter_op = filter_ctx_arr[i].filter;
		OpBase **solving_branches =
			_find_entities_solving_branches(filter_ctx_arr[i].entities, cp);

		if(solving_branches == NULL) {
			// filter placement failed, return early
			array_free(filter_ctx_arr);
			return;
		}

		uint solving_branch_count = array_len(solving_branches);
		// in case this filter is solved by the entire cartesian product
		// it does not need to be repositioned
		if(solving_branch_count == cp->childCount) {
			array_free(solving_branches);
			continue;
		}

		// the filter needs to be repositioned
		ExecutionPlan_RemoveOp(plan, (OpBase *)filter_op);

		// this filter is solved by a single cartesian product child
		// and needs to be propagated up
		if(solving_branch_count == 1) {
			OpBase *solving_op = solving_branches[0];
			// single branch solving a filter that was after a cartesian product
			// the filter may be pushed directly onto the appropriate branch
			ExecutionPlan_PushBelow(solving_op, (OpBase *)filter_op);
			array_free(solving_branches);
			continue;
		}

		// need to create a new cartesian product and connect
		// the solving branches to the filter
		OpBase *new_cp = NewCartesianProductOp(cp->plan);
		ExecutionPlan_AddOp((OpBase *)filter_op, new_cp);

		// detach each solving branch from the original cp
		// and attach them as children for the new cp
		for(uint j = 0; j < solving_branch_count; j++) {
			OpBase *solving_branch = solving_branches[j];
			ExecutionPlan_DetachOp(solving_branch);
			ExecutionPlan_AddOp(new_cp, solving_branch);
		}
		array_free(solving_branches);

		ASSERT(cp->childCount > 0);
		ExecutionPlan_AddOp(cp, (OpBase *)filter_op);
	}

	// clean up
	for(uint i = 0; i < filter_count; i++) {
		_FilterCtx_Free(filter_ctx_arr + i);
	}

	array_free(filter_ctx_arr);
}

// optimize cartesian product operations by splitting them up
// into sub cartesian products allowing filters touching multiple branches
// to be applied earlier
void reduceCartesianProductStreamCount
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	// collect all cartesian product operations in plan
	OpBase **cps = ExecutionPlan_CollectOps(plan->root,
			OPType_CARTESIAN_PRODUCT);
	uint cp_count = array_len(cps);

	// try to optimize each cartesian product
	// by splitting it up into multiple sub cartesian products
	for(uint i = 0; i < cp_count; i++) {
		OpBase *cp = cps[i];

		// skip cartesian products with less then 3 branches
		if(cp->childCount > 2) {
			_optimize_cartesian_product(plan, cp);
		}
	}

	array_free(cps);
}

