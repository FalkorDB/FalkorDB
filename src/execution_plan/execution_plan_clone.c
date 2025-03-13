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

static ExecutionPlan *_ClonePlanInternals
(
	const ExecutionPlan *template
) {
	ExecutionPlan *clone = ExecutionPlan_NewEmptyExecutionPlan();

	clone->record_map = raxClone(template->record_map);
	if(template->ast_segment) clone->ast_segment = AST_ShallowCopy(template->ast_segment);
	if(template->query_graph) {
		QueryGraph_ResolveUnknownRelIDs(template->query_graph);
		clone->query_graph = QueryGraph_Clone(template->query_graph);
	}

	return clone;
}

static OpBase *_CloneOpTree
(
	OpBase *template_current,
	dict* old_to_new
) {
	ExecutionPlan *plan_segment;

	dictEntry *entry = HashTableFind(old_to_new, template_current->plan);
	if(entry == NULL) {
		plan_segment = _ClonePlanInternals(template_current->plan);
		HashTableAdd(old_to_new, (void *)template_current->plan, (void *)plan_segment);
	} else {
		plan_segment = (ExecutionPlan *)HashTableGetVal(entry);
	}

	// temporarily set the thread-local AST to be the one referenced by this
	// ExecutionPlan segment
	QueryCtx_SetAST(plan_segment->ast_segment);

	// clone the current operation
	OpBase *clone_current = OpBase_Clone(plan_segment, template_current);
	if(plan_segment->root == NULL) plan_segment->root = clone_current;

	for(uint i = 0; i < template_current->childCount; i++) {
		// recursively visit and clone the op's children
		OpBase *child_op = _CloneOpTree(template_current->children[i],
			old_to_new);
		ExecutionPlan_AddOp(clone_current, child_op);
	}

	return clone_current;
}

static ExecutionPlan *_ExecutionPlan_Clone
(
	const ExecutionPlan *template
) {
	// create mapping from old exec-plans to the new ones
	dict *old_to_new = HashTableCreate(&def_dt);

	OpBase *clone_root = _CloneOpTree(template->root, old_to_new);

	// the "master" execution plan is the one constructed with the root op
	ExecutionPlan *clone = (ExecutionPlan *)clone_root->plan;

	HashTableRelease(old_to_new);

	return clone;
}

// this function clones the input ExecutionPlan by recursively visiting its tree
// when an op is encountered that was constructed as part of a different
// ExecutionPlan segment, that segment and its internal members
// (FilterTree, record mapping, query graphs, and AST segment)
// are also cloned
ExecutionPlan *ExecutionPlan_Clone
(
	const ExecutionPlan *template
) {
	ASSERT(template != NULL);

	// store the original AST pointer
	AST *master_ast = QueryCtx_GetAST();

	// verify that the execution plan template is not prepared yet
	ASSERT(template->prepared == false && "Execution plan cloning should be only on templates");

	ExecutionPlan *clone = _ExecutionPlan_Clone(template);

	// restore the original AST pointer
	QueryCtx_SetAST(master_ast);
	return clone;
}

