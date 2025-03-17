/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../RG.h"
#include "../util/dict.h"
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
	dict* old_to_new
) {
	ExecutionPlan *plan_segment;

	// see if op's plan had been cloned?
	dictEntry *entry = HashTableFind(old_to_new, op->plan);
	if(entry == NULL) {
		// clone segment
		plan_segment = _ClonePlanInternals(op->plan);
		HashTableAdd(old_to_new, (void *)op->plan, (void *)plan_segment);
	} else {
		plan_segment = (ExecutionPlan *)HashTableGetVal(entry);
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
	dict *old_to_new = HashTableCreate(&def_dt);

	OpBase *root = _CloneOpTree(plan->root, old_to_new);

	HashTableRelease(old_to_new);

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

