/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "execution_plan_modify.h"
#include "../execution_plan.h"
#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../../ast/ast_mock.h"
#include "../../util/rax_extensions.h"

// remove entire branch
void ExecutionPlan_RemoveBranch
(
	OpBase *root  // root of branch to remove
) {
	ASSERT(root != NULL);
	ASSERT(root->plan != NULL);

	ExecutionPlan *plan = root->plan;

	if(plan->root == root) {
		ASSERT(root->parent == NULL);
		// root is already detached, set plan's root to NULL
		plan->root = NULL;
		return; 
	}

	OpBase_RemoveChild(root->parent, root, false);
}

// remove node from its parent
// parent inherits all children of node
void ExecutionPlan_RemoveOp
(
	OpBase *op  // op to remove
) {
	ASSERT(op       != NULL);
	ASSERT(op->plan != NULL);
	ASSERT(op->childCount <= 1);  // can't remove op with multiple children

	OpBase        *parent = op->parent;
	ExecutionPlan *plan   = op->plan;

	if(plan->root == op) {
		// root should have no parent
		ASSERT(parent == NULL);

		// removing execution plan's root
		if(op->childCount > 0) {
			ASSERT(op->childCount == 1);
			// assign child as new root
			OpBase *child = op->children[0];
			OpBase_RemoveChild(op, child, false);

			// set root
			plan->root = child;
		} else {
			// no children, no parent, no root
			plan->root = NULL;
		}
	} else {
		// removing an inner node
		OpBase_RemoveChild(parent, op, true);
	}

	//--------------------------------------------------------------------------
	// disassociate op from plan
	//--------------------------------------------------------------------------
	
	// remove op from plan's resolvers map
	ExecutionPlan_RemoveResolver(op->plan, op);

	op->plan = NULL;
}

// introduce the new operation B between A and A's parent op
void ExecutionPlan_PushBelow
(
	OpBase *a,
	OpBase *b
) {
	ASSERT(a != b);
	ASSERT(a != NULL);
	ASSERT(b != NULL);
	ASSERT(b->parent == NULL);
	ASSERT(b->childCount == 0);

	if(a->parent == NULL) {
		OpBase_AddChild(b, a);

		// update plan's root
		if(a->plan->root == a) a->plan->root = b;

		return;
	}

	// disconnect A from its parent and replace it with B
	// locate A in parent's children array
	int i;
	OpBase *parent = a->parent;
	bool found = OpBase_LocateChild(parent, a, &i);
	ASSERT(found == true);

	OpBase_AddChildAt(parent, b, i);

	// add A as a child of B
	OpBase_RemoveChild(parent, a, false);
	OpBase_AddChild(b, a);
}

// replace operation A with operation B
void ExecutionPlan_ReplaceOp
(
	OpBase *a,
	OpBase *b
) {
	ASSERT(a    != b);
	ASSERT(a    != NULL);
	ASSERT(b    != NULL);

	// insert the new operation between the original and its parent
	ExecutionPlan_PushBelow(a, b);

	// b inharets all of a's children
	OpBase_RemoveChild(b, a, true);

	a->plan = NULL;
}

// update the root op of the execution plan
void ExecutionPlan_UpdateRoot
(
	ExecutionPlan *plan,  // plan to update
	OpBase *new_root      // new root op
) {
	ASSERT(plan             != NULL);
	ASSERT(new_root         != NULL);
	ASSERT(new_root->parent == NULL);

	ASSERT(new_root->childCount == 0);

	if(plan->root != NULL) {
		// replace existing root
		ExecutionPlan_PushBelow(plan->root, new_root);
	} else {
		// no root, set new root
		plan->root = new_root;
	}
}

// for all ops in the given tree, associate the provided ExecutionPlan
// if qg is set, merge the query graphs of the temporary and main plans
static void _ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *root          // root operation
) {
	if(!root) return;

	root->plan = plan;
	for(int i = 0; i < root->childCount; i ++) {
		_ExecutionPlan_BindOpsToPlan(plan, root->children[i]);
	}
}

// for all ops in the given tree, associate the provided ExecutionPlan
// if qg is set, merge the query graphs of the temporary and main plans
void ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *root,         // root operation
	bool qg               // whether to merge QueryGraphs or not
) {
	ASSERT(plan != NULL);
	ASSERT(root != NULL);

	if(qg) {
		// if the temporary execution plan has added new QueryGraph entities
		// migrate them to the master plan's QueryGraph
		QueryGraph_MergeGraphs(plan->query_graph, root->plan->query_graph);
	}

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
	ASSERT(ops  != NULL);
	ASSERT(plan != NULL);

	for(uint i = 0; i < op_count; i++) {
		if(OpBase_Type(ops[i]) != exclude_type) {
			OpBase_BindOpToPlan(ops[i], (ExecutionPlan *)plan);
		}
	}
}

