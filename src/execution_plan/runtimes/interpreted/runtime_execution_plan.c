/*
 * Copyright 2018-2020 Redis Labs Ltd. and Contributors
 *
 * This file is available under the Redis Labs Source Available License Agreement
 */

#include "runtime_execution_plan.h"
#include "../../../RG.h"
#include "./ops/ops.h"
#include "../../ops/ops.h"
#include "../../../errors.h"
#include "../../../util/arr.h"
#include "../../../query_ctx.h"
#include "../../../util/rmalloc.h"
#include "execution_plan_build/runtime_execution_plan_modify.h"
#include "./../../optimizations/optimizer.h"
#include "../../../ast/ast_build_filter_tree.h"
#include "../../execution_plan_build/execution_plan_modify.h"
#include "../../execution_plan_build/execution_plan_construct.h"

#include <setjmp.h>

// Allocate a new ExecutionPlan segment.
inline RT_ExecutionPlan *RT_ExecutionPlan_NewEmptyExecutionPlan(void) {
	return rm_calloc(1, sizeof(RT_ExecutionPlan));
}

static RT_OpBase *_ExecutionPlan_FindLastWriter(RT_OpBase *root) {
	if(RT_OpBase_IsWriter(root)) return root;
	for(int i = root->childCount - 1; i >= 0; i--) {
		RT_OpBase *child = root->children[i];
		RT_OpBase *res = _ExecutionPlan_FindLastWriter(child);
		if(res) return res;
	}
	return NULL;
}

static RT_OpBase *_convert(const RT_ExecutionPlan *plan, const OpBase *op_desc) {
	RT_OpBase *result = NULL;
	if(plan->plan_desc != op_desc->plan) {
		RT_ExecutionPlan *new_plan = RT_ExecutionPlan_NewEmptyExecutionPlan();
		new_plan->plan_desc = op_desc->plan;
		new_plan->record_map = op_desc->plan->record_map;
		plan = new_plan;
	}
	result = RT_OpBase_New(plan, op_desc);
	
	for (int i = 0; i < op_desc->childCount; i++) {
		RT_OpBase *child = _convert(plan, op_desc->children[i]);
		RT_ExecutionPlan_AddOp(result, child);
	}

	return result;
}

RT_ExecutionPlan *RT_NewExecutionPlan(const ExecutionPlan *plan_desc) {
	RT_ExecutionPlan *plan = rm_malloc(sizeof(RT_ExecutionPlan));
	plan->prepared = false;
	plan->plan_desc = plan_desc;
	plan->record_map = plan_desc->record_map;
	plan->root = _convert(plan, plan_desc->root);
	plan->record_pool = NULL;
	return plan;
}

void RT_ExecutionPlan_PreparePlan(RT_ExecutionPlan *plan) {
	// Plan should be prepared only once.
	ASSERT(!plan->prepared);

	RT_ExecutionPlan_Init(plan);
	
	QueryCtx_SetLastWriter(_ExecutionPlan_FindLastWriter(plan->root));
	plan->prepared = true;
}

inline rax *RT_ExecutionPlan_GetMappings(const RT_ExecutionPlan *plan) {
	ASSERT(plan && plan->record_map);
	return plan->record_map;
}

Record RT_ExecutionPlan_BorrowRecord(RT_ExecutionPlan *plan) {
	rax *mapping = RT_ExecutionPlan_GetMappings(plan);
	ASSERT(plan->record_pool);

	// Get a Record from the pool and set its owner and mapping.
	Record r = ObjectPool_NewItem(plan->record_pool);
	r->owner = plan;
	r->mapping = mapping;
	return r;
}

void RT_ExecutionPlan_ReturnRecord(RT_ExecutionPlan *plan, Record r) {
	ASSERT(plan && r);
	ObjectPool_DeleteItem(plan->record_pool, r);
}

//------------------------------------------------------------------------------
// Execution plan initialization
//------------------------------------------------------------------------------

static inline void _ExecutionPlan_InitRecordPool(RT_ExecutionPlan *plan) {
	if(plan->record_pool) return;
	/* Initialize record pool.
	 * Determine Record size to inform ObjectPool allocation. */
	uint entries_count = raxSize(plan->record_map);
	uint rec_size = sizeof(_Record) + (sizeof(Entry) * entries_count);

	// Create a data block with initial capacity of 256 records.
	plan->record_pool = ObjectPool_New(256, rec_size, (fpDestructor)Record_FreeEntries);
}

static void _ExecutionPlanInit(RT_OpBase *root) {
	// If the ExecutionPlan associated with this op hasn't built a record pool yet, do so now.
	_ExecutionPlan_InitRecordPool((RT_ExecutionPlan *)root->plan);

	// Initialize the operation if necessary.
	if(root->init) root->init(root);

	// Continue initializing downstream operations.
	for(int i = 0; i < root->childCount; i++) {
		_ExecutionPlanInit(root->children[i]);
	}
}

void RT_ExecutionPlan_Init(RT_ExecutionPlan *plan) {
	_ExecutionPlanInit(plan->root);
}

ResultSet *RT_ExecutionPlan_Execute(RT_ExecutionPlan *plan) {
	ASSERT(plan->prepared)
	/* Set an exception-handling breakpoint to capture run-time errors.
	 * encountered_error will be set to 0 when setjmp is invoked, and will be nonzero if
	 * a downstream exception returns us to this breakpoint. */
	int encountered_error = SET_EXCEPTION_HANDLER();

	// Encountered a run-time error - return immediately.
	if(encountered_error) return QueryCtx_GetResultSet();

	RT_ExecutionPlan_Init(plan);

	Record r = NULL;
	// Execute the root operation and free the processed Record until the data stream is depleted.
	while((r = RT_OpBase_Consume(plan->root)) != NULL) RT_ExecutionPlan_ReturnRecord(r->owner, r);

	return QueryCtx_GetResultSet();
}

//------------------------------------------------------------------------------
// Execution plan draining
//------------------------------------------------------------------------------

// NOP operation consume routine for immediately terminating execution.
static Record deplete_consume(struct RT_OpBase *op) {
	return NULL;
}

// return true if execution plan been drained
// false otherwise
bool RT_ExecutionPlan_Drained(RT_ExecutionPlan *plan) {
	ASSERT(plan != NULL);
	ASSERT(plan->root != NULL);
	return (plan->root->consume == deplete_consume);
}

static void _ExecutionPlan_Drain(RT_OpBase *root) {
	root->consume = deplete_consume;
	for(int i = 0; i < root->childCount; i++) {
		_ExecutionPlan_Drain(root->children[i]);
	}
}

// Resets each operation consume function to simply return NULL
// this will cause the execution-plan to quickly deplete
void RT_ExecutionPlan_Drain(RT_ExecutionPlan *plan) {
	ASSERT(plan && plan->root);
	_ExecutionPlan_Drain(plan->root);
}

//------------------------------------------------------------------------------
// Execution plan ref count
//------------------------------------------------------------------------------

void RT_ExecutionPlan_IncreaseRefCount(RT_ExecutionPlan *plan) {
	ASSERT(plan);
	__atomic_fetch_add(&plan->ref_count, 1, __ATOMIC_RELAXED);
}

int RT_ExecutionPlan_DecRefCount(RT_ExecutionPlan *plan) {
	ASSERT(plan);
	return __atomic_sub_fetch(&plan->ref_count, 1, __ATOMIC_RELAXED);
}

//------------------------------------------------------------------------------
// Execution plan profiling
//------------------------------------------------------------------------------

static void _ExecutionPlan_InitProfiling(RT_OpBase *root) {
	root->profile = root->consume;
	root->consume = RT_OpBase_Profile;
	root->stats = rm_malloc(sizeof(OpStats));
	root->stats->profileExecTime = 0;
	root->stats->profileRecordCount = 0;

	if(root->childCount) {
		for(int i = 0; i < root->childCount; i++) {
			RT_OpBase *child = root->children[i];
			_ExecutionPlan_InitProfiling(child);
		}
	}
}

static void _ExecutionPlan_FinalizeProfiling(RT_OpBase *root) {
	if(root->childCount) {
		for(int i = 0; i < root->childCount; i++) {
			RT_OpBase *child = root->children[i];
			root->stats->profileExecTime -= child->stats->profileExecTime;
			_ExecutionPlan_FinalizeProfiling(child);
		}
	}
	root->stats->profileExecTime *= 1000;   // Milliseconds.
}

ResultSet *RT_ExecutionPlan_Profile(RT_ExecutionPlan *plan) {
	_ExecutionPlan_InitProfiling(plan->root);
	ResultSet *rs = RT_ExecutionPlan_Execute(plan);
	_ExecutionPlan_FinalizeProfiling(plan->root);
	return rs;
}

//------------------------------------------------------------------------------
// Execution plan free functions
//------------------------------------------------------------------------------

static void _ExecutionPlan_FreeInternals(RT_ExecutionPlan *plan) {
	if(plan == NULL) return;

	if(plan->record_map) raxFree(plan->record_map);
	if(plan->record_pool) ObjectPool_Free(plan->record_pool);
	rm_free(plan);
}

// Free an op tree and its associated ExecutionPlan segments.
static RT_ExecutionPlan *_ExecutionPlan_FreeOpTree(RT_OpBase *op) {
	if(op == NULL) return NULL;
	RT_ExecutionPlan *child_plan = NULL;
	RT_ExecutionPlan *prev_child_plan = NULL;
	// Store a reference to the current plan.
	RT_ExecutionPlan *current_plan = (RT_ExecutionPlan *)op->plan;
	for(uint i = 0; i < op->childCount; i ++) {
		child_plan = _ExecutionPlan_FreeOpTree(op->children[i]);
		// In most cases all children will share the same plan, but if they don't
		// (for an operation like UNION) then free the now-obsolete previous child plan.
		if(prev_child_plan != child_plan) {
			_ExecutionPlan_FreeInternals(prev_child_plan);
			prev_child_plan = child_plan;
		}
	}

	// Free this op.
	RT_OpBase_Free(op);

	// Free each ExecutionPlan segment once all ops associated with it have been freed.
	if(current_plan != child_plan) _ExecutionPlan_FreeInternals(child_plan);

	return current_plan;
}

void RT_ExecutionPlan_Free(RT_ExecutionPlan *plan) {
	if(plan == NULL) return;
	if(RT_ExecutionPlan_DecRefCount(plan) >= 0) return;

	// Free all ops and ExecutionPlan segments.
	_ExecutionPlan_FreeOpTree(plan->root);

	// Free the final ExecutionPlan segment.
	_ExecutionPlan_FreeInternals(plan);
}
