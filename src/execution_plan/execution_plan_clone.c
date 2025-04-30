/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../RG.h"
#include "../util/hashmap.h"
#include "../query_ctx.h"
#include "execution_plan_clone.h"
#include "../util/rax_extensions.h"
#include "execution_plan_build/execution_plan_modify.h"

// clone plan's internals
// this includes:
//   record mapping
//   ast segment
//   query graph
static ExecutionPlan *_ClonePlanInternals
(
	const ExecutionPlan *plan  // plan to clone internals from
) {
	ASSERT(plan != NULL);

	ExecutionPlan *clone = ExecutionPlan_NewEmptyExecutionPlan();

	clone->record_map = raxClone(plan->record_map);

	if(plan->ast_segment) {
		clone->ast_segment = AST_ShallowCopy(plan->ast_segment);
	}

	if(plan->query_graph) {
		QueryGraph_ResolveUnknownRelIDs(plan->query_graph);
		clone->query_graph = QueryGraph_Clone(plan->query_graph);
	}

	return clone;
}

static OpBase *_CloneOpTree
(
	OpBase *op,
	hashmap old_to_new
) {
	ExecutionPlan *plan_segment;

	// see if op's plan had been cloned?
	ExecutionPlan **plan_segment_ptr = (ExecutionPlan **)hashmap_get_with_hash(
			old_to_new, NULL, (uint64_t)op->plan);

	if(plan_segment_ptr == NULL) {
		// clone segment
		plan_segment = _ClonePlanInternals(op->plan);
		hashmap_set_with_hash(old_to_new, (void *)&plan_segment, (uint64_t)op->plan);
	} else {
		plan_segment = *plan_segment_ptr;
	}

	// temporarily set the thread-local AST to be the one referenced by this
	// ExecutionPlan segment
	QueryCtx_SetAST(plan_segment->ast_segment);

	// clone the current operation
	OpBase *clone = OpBase_Clone(plan_segment, op);

	// make sure segment root is set
	if(plan_segment->root == NULL) {
		plan_segment->root = clone;
	}

	for(uint i = 0; i < op->childCount; i++) {
		// recursively visit and clone the op's children
		OpBase *child_op = _CloneOpTree(op->children[i], old_to_new);
		ExecutionPlan_AddOp(clone, child_op);
	}

	return clone;
}

static ExecutionPlan *_ExecutionPlan_Clone
(
	const ExecutionPlan *plan
) {
	// create mapping from old exec-plans to the new ones
	hashmap old_to_new = hashmap_new_with_redis_allocator(sizeof(void *), 0, 0,
			0, NULL, NULL, NULL, NULL);

	OpBase *root = _CloneOpTree(plan->root, old_to_new);

	hashmap_free(old_to_new);

	// the "master" execution plan is the one constructed with the root op
	ExecutionPlan *clone = (ExecutionPlan *)root->plan;

	return clone;
}

// clones an ExecutionPlan by recursively visiting its tree of operations
ExecutionPlan *ExecutionPlan_Clone
(
	const ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	// store the original AST pointer
	AST *master_ast = QueryCtx_GetAST();

	// verify that the plan is not prepared
	ASSERT(plan->prepared == false && "can not clone a prepared plan");

	ExecutionPlan *clone = _ExecutionPlan_Clone(plan);

	// restore the original AST pointer
	QueryCtx_SetAST(master_ast);
	return clone;
}

