/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/op_filter.h"
#include "../../filter_tree/filter_tree.h"
#include "../../ast/ast_build_filter_tree.h"

/* The reduce filters optimizer scans an execution plans for
 * consecutive filter operations, these can be reduced down into
 * a single filter operation by ANDing their filter trees
 * Reducing the overall number of operations is expected to produce
 * faster execution time. */

void _reduceFilter(OpBase *op) {
	OpBase *parent = op;
	OpFilter *filter = (OpFilter *)parent;
	FT_FilterNode *tree = filter->filterTree;
	OpBase *child = NULL;

	/* Filter operation is promised to have only one child. */
	while(parent->childCount == 1) {
		child = parent->children[0];
		if(child->type != OPType_FILTER) break;

		OpFilter *childFilter = (OpFilter *)child;

		/* Create a new root for the tree, merge trees using an AND. */
		FT_FilterNode *root = FilterTree_CreateConditionFilter(OP_AND);
		FilterTree_AppendLeftChild(root, tree);
		FilterTree_AppendRightChild(root, childFilter->filterTree);
		tree = root;

		// Proceed.
		parent = child;
	}

	// Did we performed a reduction?
	if(filter->filterTree != tree) {
		filter->filterTree = tree;
		// Remove intermidate filter ops.
		OpBase *intermidateChild = child->parent;
		while(intermidateChild != op) {
			parent = intermidateChild->parent;
			// Remove the filter tree pointer from the intermediate op, as it should not be freed
			((OpFilter *)intermidateChild)->filterTree = NULL;
			OpBase_Free(intermidateChild);
			intermidateChild = parent;
		}

		/* child is the first operation we encountered which is not of type filter.
		 * update child parent to reduced filter op
		 * update reduced filter op child. */
		child->parent = op;
		op->children[0] = child;
	}
}

void _reduceFilters(OpBase *op) {
	if(op == NULL) return;

	if(op->type == OPType_FILTER) {
		_reduceFilter(op);
	}

	for(int i = 0; i < op->childCount; i++) {
		_reduceFilters(op->children[i]);
	}
}

void reduceFilters(ExecutionPlan *plan) {
	_reduceFilters(plan->root);
}

