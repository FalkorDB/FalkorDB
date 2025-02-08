/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../ops/op.h"
#include "../ops/op_filter.h"
#include "../execution_plan.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"

static inline void _PushDownPathFilters
(
	ExecutionPlan *plan,
	OpBase *path_filter_op
) {
	OpBase *relocate_to = path_filter_op;
	// find the earliest filter op in the path filter op's chain of parents
	while(relocate_to->parent && relocate_to->parent->type == OPType_FILTER) {
		relocate_to = relocate_to->parent;
	}
	// if the filter op is part of a chain of filter ops, migrate it
	// to be the topmost. This ensures that cheaper filters will be
	// applied first
	if(relocate_to != path_filter_op) {
		ExecutionPlan_RemoveOp(plan, path_filter_op);
		ExecutionPlan_PushBelow(relocate_to, path_filter_op);
	}
}

static void _ExecutionPlan_PlaceApplyOps(ExecutionPlan *plan) {
	OpBase **filter_ops = ExecutionPlan_CollectOps(plan->root, OPType_FILTER);
	uint filter_ops_count = array_len(filter_ops);
	for(uint i = 0; i < filter_ops_count; i++) {
		OpFilter *op = (OpFilter *)filter_ops[i];
		FT_FilterNode *node;
		if(FilterTree_ContainsFunc(op->filterTree, "path_filter", &node)) {
			// if the path filter op has other filter ops above it
			// migrate it to be the topmost.
			_PushDownPathFilters(plan, (OpBase *)op);
			// convert the filter op to an Apply operation
			ExecutionPlan_ReduceFilterToApply(plan, op);
		}
	}
	array_free(filter_ops);
}

// reposition a filter operation to the earliest position within the plan
// at which the filter can be evaluate
void ExecutionPlan_RePositionFilterOp
(
	ExecutionPlan *plan,        // plan
	OpBase *lower_bound,        // lower boundry
	const OpBase *upper_bound,  // upper boundry
	OpBase *filter              // filter
) {
	// validate inputs
	ASSERT(plan != NULL);
	ASSERT(filter->type == OPType_FILTER);

	// when placing filters, we should not recurse into certain operation's
	// subtrees that would cause logical errors
	// the cases we currently need to be concerned with are:
	// merge - the results which should only be filtered after the entity
	// is matched or created
	//
	// apply - which has an Optional child that should project results or NULL
	// before being filtered
	//
	// the family of SemiApply ops (including the Apply Multiplexers)
	// does not require this restriction since they are always exclusively
	// performing filtering

	OpBase *op = NULL; // operation after which filter will be located
	const FT_FilterNode *filter_tree = ((OpFilter *)filter)->filterTree;

	// collect all filtered entities
	// e.g. n.score > m.avg_score
	// will extract both 'n' and 'm' as these entities must be resolved
	// before the filter is applied
	rax *references = FilterTree_CollectModified(filter_tree);
	uint64_t references_count = raxSize(references);

	if(references_count > 0) {
		// scan execution plan, locate the earliest position where all
		// references been resolved
		op = ExecutionPlan_LocateReferencesExcludingOps(lower_bound,
				upper_bound, FILTER_RECURSE_BLACKLIST, BLACKLIST_OP_COUNT,
				references);
		if(!op) {
			// failed to resolve all filter references
			Error_InvalidFilterPlacement(references);
			OpBase_Free(filter);
			return;
		}
	} else {
		// the filter tree does not contain references
		// e.g.
		// WHERE 1=1
		// place the filter directly below the first projection if there is one
		// otherwise update the execution plan's root
		op = ExecutionPlan_LocateOpMatchingTypes(plan->root, PROJECT_OPS,
				PROJECT_OP_COUNT);
		op = (op == NULL) ? plan->root : op;
	}

	// in case this is a pre-existing filter
	// (this function is not called out from ExecutionPlan_PlaceFilterOps)
	if(filter->childCount > 0) {
		// if the located op is not the filter child, re position the filter
		if(op != filter->children[0]) {
			ExecutionPlan_RemoveOp(plan, (OpBase *)filter);
			ExecutionPlan_PushBelow(op, (OpBase *)filter);
		}
	} else if(op == NULL) {
		// no root was found, place filter at the root
		ExecutionPlan_UpdateRoot(plan, (OpBase *)filter);
		op = filter;
	} else {
		// this is a new filter
		ExecutionPlan_PushBelow(op, (OpBase *)filter);
	}

	// filter may have migrated a segment, update the filter segment
	// and check if the segment root needs to be updated
	// the filter should be associated with the op's segment
	filter->plan = op->plan;

	// re-set the segment root if needed
	if(op == op->plan->root) {
		ExecutionPlan *segment = (ExecutionPlan *)op->plan;
		segment->root = filter;
	}

	// clean up
	raxFree(references);
}

// place filter ops at the appropriate positions within the op tree
void ExecutionPlan_PlaceFilterOps
(
	ExecutionPlan *plan,           // plan 
	OpBase *root,                  // root
	const OpBase *recurse_limit,   // boundry
	FT_FilterNode *ft              // filter-tree to position
) {
	ASSERT(ft   != NULL);
	ASSERT(plan != NULL);
	ASSERT(root != NULL);

	//--------------------------------------------------------------------------
	// decompose filter tree
	//--------------------------------------------------------------------------

	// decompose filter tree into the smallest possible subtrees that do not
	// violate the rules of AND/OR combinations
	const FT_FilterNode **sub_trees = FilterTree_SubTrees(ft);

	// for each filter tree, find the earliest position in the op tree
	// after which the filter tree can be applied
	uint n = array_len(sub_trees);
	for(uint i = 0; i < n; i++) {
		// clone current sub-tree
		FT_FilterNode *tree = FilterTree_Clone(sub_trees[i]);

		// create a filter operation
		OpBase *op = NewFilterOp(plan, tree);

		// position filter op
		ExecutionPlan_RePositionFilterOp(plan, root, recurse_limit, op);
	}

	// all trees been positioned, clean up
	array_free(sub_trees);
	FilterTree_Free(ft);

	// build ops in the Apply family to appropriately process path filters
	_ExecutionPlan_PlaceApplyOps(plan);
}

