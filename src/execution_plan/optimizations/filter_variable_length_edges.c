/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../util/arr.h"
#include "../ops/op_filter.h"
#include "../ops/op_cond_var_len_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// the filterVariableLengthEdges optimization finds variable-length traversal
// ops in the op tree that are immediately followed by filter ops and
// if the filter op(s) apply predicates directly to the traversed edge
// migrates those predicates into the traversal itself
//
// this allows us to only expand paths on which the traversed edges pass the
// filter predicates, rather than applying them after discovering all paths

// returns true if the given filter operates exclusively on the traversed edge
// of a CondVarLenTraverse op
static bool _applicableFilter
(
	FT_FilterNode *ft,
	const char *src,
	const char *edge,
	const char *dest
) {
	bool applicable = false;

	// Collect all modified aliases in the FilterTree.
	rax *filtered = FilterTree_CollectModified(ft);

	// Look up the edge alias in the alias map.
	applicable = (raxFind(filtered, (unsigned char *)edge, strlen(edge)) != raxNotFound);
	if(applicable) {
		// reject filter trees that contain either the src or dest node
		// this avoids false positives on TOPATH expressions that collect all
		// aliases in a path and rejects filters between an edge property and
		// a traversal's src/dest property which will not be available during
		// traversals
		applicable =
			(raxFind(filtered, (unsigned char *)src, strlen(src)) == raxNotFound
			 &&
			 raxFind(filtered, (unsigned char *)dest, strlen(dest)) == raxNotFound);
	}

	raxFree(filtered);
	return applicable;
}

static void _filterVariableLengthEdges
(
	ExecutionPlan *plan,
	CondVarLenTraverse *traverse_op
) {
	ASSERT(plan        != NULL);
	ASSERT(traverse_op != NULL);

	// retrieve the aliases of the traversed source, destination, and edge
	OpBase *parent     = traverse_op->op.parent;
	const char *src    = AlgebraicExpression_Src(traverse_op->ae);
	const char *edge   = AlgebraicExpression_Edge(traverse_op->ae);
	const char *dest   = AlgebraicExpression_Dest(traverse_op->ae);
	OpFilter **filters = array_new(OpFilter *, 0);

	// collect applicable filters
	while(parent && parent->type == OPType_FILTER) {
		// track the next op to visit in case we free parent
		OpFilter *op_filter = (OpFilter *)parent;
		FT_FilterNode *ft = op_filter->filterTree;
		// check if the filter is applied to the traversed edge
		if(_applicableFilter(ft, src, edge, dest)) {
			array_append(filters, op_filter);
		}

		// advance
		parent = parent->parent;
	}

	uint n = array_len(filters);

	// concat filters using AND condition
	FT_FilterNode *root = NULL;
	for(uint i = 0; i < n; i++) {
		OpFilter *filter_op = filters[i];
		FT_FilterNode *ft = filter_op->filterTree;

		// remove filter operation from execution plan
		ExecutionPlan_RemoveOp(plan, (OpBase *)filter_op);
		// NULL-set the filter tree to avoid a double free
		filter_op->filterTree = NULL;
		OpBase_Free((OpBase *)filter_op);

		if(root == NULL) {
			root = ft;
		} else {
			FT_FilterNode * and = FilterTree_CreateConditionFilter(OP_AND);
			FilterTree_AppendLeftChild( and, root);
			FilterTree_AppendRightChild( and, ft);
			root = and;
		}

	}
	array_free(filters);

	// embed the filter tree in the variable-length traversal
	if(root != NULL) CondVarLenTraverseOp_SetFilter(traverse_op, root);
}

void filterVariableLengthEdges
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);
	OpBase **var_len_traverse_ops = NULL;

	// collect all variable-length traversals
	const OPType types[] = {OPType_CONDITIONAL_VAR_LEN_TRAVERSE,
							OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO
						   };

	var_len_traverse_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root,
																types, 2);

	uint count = array_len(var_len_traverse_ops);
	for(uint i = 0; i < count; i ++) {
		CondVarLenTraverse *op = (CondVarLenTraverse *)var_len_traverse_ops[i];
		_filterVariableLengthEdges(plan, op);
	}

	array_free(var_len_traverse_ops);
}

