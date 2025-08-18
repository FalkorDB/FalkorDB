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
#include "execution_plan_awareness.h"
#include "../../util/rax_extensions.h"

static void _OpBase_AddChild
(
	OpBase *parent,
	OpBase *child
) {
	// add child to parent
	if(parent->children == NULL) {
		parent->children = rm_malloc(sizeof(OpBase *));
	} else {
		parent->children = rm_realloc(parent->children,
				sizeof(OpBase *) * (parent->childCount + 1));
	}
	parent->children[parent->childCount++] = child;

	// add parent to child
	child->parent = parent;

	ExecutionPlanAwareness_PropagateAwareness(child);
}

// remove the operation old_child from its parent and replace it
// with the new child without reordering elements
static void _ExecutionPlan_ParentReplaceChild
(
	OpBase *parent,
	OpBase *old_child,
	OpBase *new_child
) {
	ASSERT(parent->childCount > 0);

	for(int i = 0; i < parent->childCount; i++) {
		// scan the children array to find the op being replaced
		if(parent->children[i] != old_child) continue;

		ExecutionPlanAwareness_RemoveAwareness(new_child);
		ExecutionPlanAwareness_RemoveAwareness(old_child);

		// replace the original child with the new one
		parent->children[i] = new_child;
		new_child->parent = parent;
		old_child->parent = NULL;

		ExecutionPlanAwareness_PropagateAwareness(new_child);
		return;
	}

	ASSERT(false && "failed to locate the operation to be replaced");
}

// removes child from it's parent
static void _OpBase_RemoveChild
(
	OpBase *parent,
	OpBase *child
) {
	ASSERT(child  != NULL);
	ASSERT(parent != NULL);
	ASSERT(parent != child);
	ASSERT(child->parent == parent);

	//--------------------------------------------------------------------------
	// remove child from parent
	//--------------------------------------------------------------------------

	// locate child in parent's children array
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
		parent->children = rm_realloc(parent->children,
				sizeof(OpBase *) * parent->childCount);
	}

	// update parent awareness
	ExecutionPlanAwareness_RemoveAwareness(child);

	// remove parent from child
	child->parent = NULL;
}

// adds operation to execution plan as a child of parent
inline void ExecutionPlan_AddOp
(
	OpBase *parent,
	OpBase *newOp
) {
	_OpBase_AddChild(parent, newOp);
}

// adds child to be the i'th child of parent
void ExecutionPlan_AddOpInd
(
	OpBase *parent,  // parent op
	OpBase *child,   // child op
	uint idx         // index of child
) {
	ASSERT(child  != NULL);
	ASSERT(parent != NULL);
	ASSERT(OpBase_ChildCount(parent) > idx);

	OpBase *to_replace = parent->children[idx];

	// replace the original child with the new one
	parent->children[idx] = child;
	child->parent = parent;

	_OpBase_AddChild(parent, to_replace);
	ExecutionPlanAwareness_PropagateAwareness(child);
}

// introduce the new operation B between A and A's parent op
void ExecutionPlan_PushBelow
(
	OpBase *a,
	OpBase *b
) {
	ASSERT(a         != b);
	ASSERT(a         != NULL);
	ASSERT(b         != NULL);
	ASSERT(b->parent == NULL);

	// B belongs to A's plan
	ExecutionPlan *plan = (ExecutionPlan *)a->plan;
	b->plan = plan;

	if(a->parent == NULL) {
		// A is the root operation
		_OpBase_AddChild(b, a);
		plan->root = b;
		return;
	}

	// disconnect A from its parent and replace it with B
	_ExecutionPlan_ParentReplaceChild(a->parent, a, b);

	// add A as a child of B
	_OpBase_AddChild(b, a);
}

// update the root op of the execution plan
void ExecutionPlan_UpdateRoot
(
	ExecutionPlan *plan,  // plan set root of
	OpBase *new_root      // new root operation
) {
	ASSERT(plan             != NULL);
	ASSERT(new_root         != NULL);
	ASSERT(new_root->parent == NULL);

	if(plan->root) {
		ASSERT(new_root != plan->root);
		ASSERT(plan->root->parent == NULL);

		// find the deepest child of the new root operation
		// currently, we can only follow the first child
		// since we don't call this function when
		// introducing a multiple-stream operation at this stage
		// this may be inadequate later
		OpBase *tail = new_root;
		ASSERT(tail->childCount <= 1);
		while(tail->childCount > 0) tail = tail->children[0];

		// append the old root to the tail of the new root's chain
		_OpBase_AddChild(tail, plan->root);
	}

	plan->root = new_root;
}

// replace a with b
void ExecutionPlan_ReplaceOp
(
	ExecutionPlan *plan,  // plan
	OpBase *a,            // operation being replaced
	OpBase *b             // replacement operation
) {
	ASSERT(plan != NULL);
	ASSERT(a    != NULL);
	ASSERT(b    != NULL);
	ASSERT(a    != b);

	// insert the new operation between the original and its parent
	ExecutionPlan_PushBelow(a, b);

	// delete the original operation
	ExecutionPlan_RemoveOp(plan, a);
}


// removes operation from execution plan
void ExecutionPlan_RemoveOp
(
	ExecutionPlan *plan,
	OpBase *op
) {
	ASSERT(op   != NULL);
	ASSERT(plan != NULL);

	if(op->parent == NULL) {
		// removing execution plan root
		ASSERT(op->childCount <= 1);

		plan->root = NULL;

		if(OpBase_ChildCount(op) == 1) {
			// assign child as new root
			plan->root = op->children[0];

			// remove new root's parent pointer
			plan->root->parent = NULL;
		}
	} else {
		OpBase *parent = op->parent;
		if(op->childCount > 0) {
			// in place replacement of the op first branch instead of op
			_ExecutionPlan_ParentReplaceChild(op->parent, op, op->children[0]);

			// add each of op's children as a child of op's parent
			for(int i = 1; i < op->childCount; i++) {
				_OpBase_AddChild(parent, op->children[i]);
			}
		} else {
			// remove op from its parent
			_OpBase_RemoveChild(op->parent, op);
		}
	}

	// clear op
	rm_free(op->children);

	op->parent     = NULL;
	op->children   = NULL;
	op->childCount = 0;

	ExecutionPlanAwareness_SelfAware(op);
}

// detaches operation from its parent
void ExecutionPlan_DetachOp
(
	OpBase *op
) {
	// operation has no parent
	if(op->parent == NULL) return;

	// remove op from its parent
	_OpBase_RemoveChild(op->parent, op);
}

static void _ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *op            // current operation
) {
	if(!op) return;

	// no expecting op to migrate to its own plan
	ASSERT(op->plan != plan);

	// incase op is its own plan's root, nullify the plan's root
	// as its root is about to be migrated to a different plan
	if(op == op->plan->root) {
		((ExecutionPlan*)op->plan)->root = NULL;
	}

	op->plan = plan;
	for(int i = 0; i < op->childCount; i++) {
		OpBase *child = OpBase_GetChild(op, i);
		bool different_plans = (op->plan != child->plan);

		_ExecutionPlan_BindOpsToPlan(plan, child);

		if(different_plans) {
			ExecutionPlanAwareness_PropagateAwareness(child);
		}
	}
}

// for all ops in the given tree, associate the provided ExecutionPlan
// merge the query graphs of the temporary and main plans
void ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *root          // root operation
) {
	ASSERT(plan         != NULL);
	ASSERT(root         != NULL);
	ASSERT(root->plan   != plan);
	ASSERT(root->parent == NULL);

	// migrate QueryGraph entities to the master plan's QueryGraph
	QueryGraph_MergeGraphs(plan->query_graph, root->plan->query_graph);

	_ExecutionPlan_BindOpsToPlan(plan, root);
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

