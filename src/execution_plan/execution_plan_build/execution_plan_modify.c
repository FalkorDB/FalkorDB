/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_plan_modify.h"
#include "../../RG.h"
#include "../execution_plan.h"
#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../../ast/ast_mock.h"
#include "../../util/rax_extensions.h"

// remove the operation old_child from its parent and replace it
// with the new child without reordering elements
static void _ExecutionPlan_ParentReplaceChild
(
	OpBase *parent,
	OpBase *old_child,
	OpBase *new_child
) {
	ASSERT(parent->childCount > 0);

	for(int i = 0; i < parent->childCount; i ++) {
		/* Scan the children array to find the op being replaced. */
		if(parent->children[i] != old_child) continue;
		/* Replace the original child with the new one. */
		parent->children[i] = new_child;
		new_child->parent = parent;
		return;
	}

	ASSERT(false && "failed to locate the operation to be replaced");
}

// removes node b from a and update child parent lists
// assuming B is a child of A
static void _OpBase_RemoveChild
(
	OpBase *parent,
	OpBase *child
) {
	// remove child from parent
	int i = 0;
	for(; i < parent->childCount; i++) {
		if(parent->children[i] == child) break;
	}

	ASSERT(i != parent->childCount);

	// update child count
	parent->childCount--;
	if(parent->childCount == 0) {
		rm_free(parent->children);
		parent->children = NULL;
	} else {
		// shift left children
		for(int j = i; j < parent->childCount; j++) {
			parent->children[j] = parent->children[j + 1];
		}
		parent->children = rm_realloc(parent->children, sizeof(OpBase *) * parent->childCount);
	}

	// remove parent from child
	child->parent = NULL;
}

inline void ExecutionPlan_AddOp
(
	OpBase *parent,
	OpBase *newOp
) {
	OpBase_AddChild(parent, newOp);
}

// adds child to be the ind'th child of parent
void ExecutionPlan_AddOpInd
(
	OpBase *parent,  // parent op
	OpBase *child,   // child op
	uint ind         // index of child
) {
	ASSERT(parent != NULL);
	ASSERT(child != NULL);

	OpBase *to_replace = parent->children[ind];
	_ExecutionPlan_ParentReplaceChild(parent, to_replace, child);
	OpBase_AddChild(parent, to_replace);
}

// introduce the new operation B between A and A's parent op
void ExecutionPlan_PushBelow
(
	OpBase *a,
	OpBase *b
) {
	// B belongs to A's plan.
	ExecutionPlan *plan = (ExecutionPlan *)a->plan;
	b->plan = plan;

	if(a->parent == NULL) {
		// A is the root operation.
		OpBase_AddChild(b, a);
		plan->root = b;
		return;
	}

	// disconnect A from its parent and replace it with B
	_ExecutionPlan_ParentReplaceChild(a->parent, a, b);

	// add A as a child of B
	OpBase_AddChild(b, a);
}

void ExecutionPlan_NewRoot
(
	OpBase *old_root,
	OpBase *new_root
) {
	// the new root should have no parent
	// but may have children if we've constructed a chain of traversals/scans
	ASSERT(!old_root->parent && !new_root->parent);

	// find the deepest child of the new root operation
	// currently, we can only follow the first child, since we don't call this function when
	// introducing Cartesian Products (the only multiple-stream operation at this stage.)
	// this may be inadequate later
	OpBase *tail = new_root;
	ASSERT(tail->childCount <= 1);

	while(tail->childCount > 0) tail = tail->children[0];

	// append the old root to the tail of the new root's chain
	OpBase_AddChild(tail, old_root);
}

inline void ExecutionPlan_UpdateRoot
(
	ExecutionPlan *plan,
	OpBase *new_root
) {
	if(plan->root) {
		ExecutionPlan_NewRoot(plan->root, new_root);
	}
	plan->root = new_root;
}

void ExecutionPlan_ReplaceOp
(
	ExecutionPlan *plan,
	OpBase *a,
	OpBase *b
) {
	// insert the new operation between the original and its parent
	ExecutionPlan_PushBelow(a, b);

	// delete the original operation
	ExecutionPlan_RemoveOp(plan, a);
}

void ExecutionPlan_RemoveOp
(
	ExecutionPlan *plan,
	OpBase *op
) {
	if(op->parent == NULL) {
		// removing execution plan root
		ASSERT(plan->root == op);
		ASSERT(op->childCount <= 1);

		if(op->childCount == 1) {
			// assign child as new root
			plan->root = op->children[0];

			// remove new root's parent pointer
			plan->root->parent = NULL;
		} else {
			// clear plan's root
			plan->root = NULL;
		}
	} else {
		OpBase *parent = op->parent;
		if(op->childCount > 0) {
			// in place replacement of the op first branch instead of op
			_ExecutionPlan_ParentReplaceChild(op->parent, op, op->children[0]);
			// add each of op's children as a child of op's parent
			for(int i = 1; i < op->childCount; i++) {
				OpBase_AddChild(parent, op->children[i]);
			}
		} else {
			// remove op from its parent
			_OpBase_RemoveChild(op->parent, op);
		}
	}

	// clear op
	op->parent     = NULL;
	op->childCount = 0;

	if(op->children != NULL) {
		rm_free(op->children);
		op->children = NULL;
	}
}

void ExecutionPlan_DetachOp
(
	OpBase *op
) {
	// operation has no parent
	if(op->parent == NULL) return;

	// remove op from its parent
	_OpBase_RemoveChild(op->parent, op);

	op->parent = NULL;
}

// for all ops in the given tree, associate the provided ExecutionPlan
// if qg is set, merge the query graphs of the temporary and main plans
void ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *root,         // root operation
	bool qg               // whether to merge QueryGraphs or not
) {
	if(!root) return;

	if(qg) {
		// If the temporary execution plan has added new QueryGraph entities,
		// migrate them to the master plan's QueryGraph.
		QueryGraph_MergeGraphs(plan->query_graph, root->plan->query_graph);
	}

	root->plan = plan;
	for(int i = 0; i < root->childCount; i ++) {
		ExecutionPlan_BindOpsToPlan(plan, root->children[i], qg);
	}
}

// binds all ops in `ops` to `plan`, other than ops of type `exclude_type`
void ExecutionPlan_MigrateOpsExcludeType
(
	OpBase *ops[],             // array of ops to bind
	OPType exclude_type,       // type of ops to exclude
	uint op_count,             // number of ops in the array
	const ExecutionPlan *plan  // plan to bind the ops to
) {
	for(uint i = 0; i < op_count; i++) {
		if(ops[i]->type != exclude_type) {
			OpBase_BindOpToPlan(ops[i], (ExecutionPlan *)plan);
		}
	}
}

