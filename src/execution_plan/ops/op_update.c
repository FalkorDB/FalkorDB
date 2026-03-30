/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_update.h"
#include "../../query_ctx.h"
#include "../../util/arr.h"
#include "../../schema/schema.h"
#include "shared/update_functions.h"
#include "../../util/rax_extensions.h"

// forward declarations
static Record UpdateConsume(OpBase *opBase);
static OpResult UpdateReset(OpBase *opBase);
static OpBase *UpdateClone(const ExecutionPlan *plan, const OpBase *opBase);
static void UpdateFree(OpBase *opBase);

static Record _handoff
(
	OpUpdate *op
) {
	if (op->rec_idx < array_len (op->records)) {
		return op->records [op->rec_idx++] ;
	} else {
		return NULL ;
	}
}

// create a new update operation
OpBase *NewUpdateOp
(
	const ExecutionPlan *plan,  // execution plan
	rax *update_exps            // update expressions
) {
	OpUpdate *op = rm_calloc (1, sizeof (OpUpdate)) ;

	op->gc             = QueryCtx_GetGraphCtx () ;
	op->records        = array_new (Record, 64) ;
	op->update_ctxs    = update_exps ;
	op->staged_updates = StagedUpdatesCtx_New () ;

	// set our op operations
	OpBase_Init ((OpBase *)op, OPType_UPDATE, "Update", NULL, UpdateConsume,
				UpdateReset, NULL, UpdateClone, UpdateFree, true, plan) ;

	// iterate over all update expressions
	// set the record index for every entity modified by this operation
	raxStart (&op->it, update_exps) ;
	raxSeek (&op->it, "^", NULL, 0) ;
	while (raxNext (&op->it)) {
		EntityUpdateDesc *desc = op->it.data ;
		ctx->record_idx = OpBase_Modifies ((OpBase *)op, desc->alias) ;
	}

	return (OpBase *)op ;
}

static Record UpdateConsume
(
	OpBase *opBase
) {
	OpUpdate *op = (OpUpdate *)opBase ;
	OpBase *child = op->op.children[0] ;

	// updates already performed
	if (array_len (op->records) > 0) {
		return _handoff (op) ;
	}

	Record r ;
	while ((r = OpBase_Consume (child))) {
		// evaluate update expressions
		raxSeek (&op->it, "^", NULL, 0) ;
		while (raxNext (&op->it)) {
			EntityUpdateDesc *desc = op->it.data ;
			EvalEntityUpdates (op->gc, op->staged_updates, r, desc, true) ;
		}

		array_append (op->records, r) ;
	}
	
	uint node_updates_count = HashTableElemCount (op->node_updates) ;
	uint edge_updates_count = HashTableElemCount (op->edge_updates) ;

	if (StagedUpdatesCtx_HasNodeUpdates (op->staged_updates) ||
		StagedUpdatesCtx_HasEdgeUpdates (op->staged_updates)) {
		// done reading; we're not going to call Consume any longer
		// there might be operations like "Index Scan" that need to free the
		// index R/W lock - as such
		// free all ExecutionPlan operations up the chain
		OpBase_PropagateReset (child) ;

		// lock everything
		QueryCtx_LockForCommit () ;

		// in cases such as:
		// MATCH (n) SET n:L
		// make sure L is of the right dimensions
		if (StagedUpdatesCtx_HasNodeUpdates (op->staged_updates)) {
			ensureMatrixDim (op->gc, op->update_ctxs) ;
		}

		CommitUpdates (op->gc, op->staged_updates) ;
	}

	StagedUpdatesCtx_Free (&op->staged_updates) ;

	// no one consumes our output, return NULL
	if (opBase->parent == NULL) {
		return NULL ;
	}

	return _handoff (op) ;
}

static OpBase *UpdateClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (opBase->type == OPType_UPDATE) ;
	OpUpdate *op = (OpUpdate *)opBase ;

	rax *update_ctxs =
		raxCloneWithCallback (op->update_ctxs,
				(void *(*)(void *))UpdateCtx_Clone) ;
	return NewUpdateOp (plan, update_ctxs) ;
}

static OpResult UpdateReset
(
	OpBase *ctx
) {
	OpUpdate *op = (OpUpdate *)ctx ;

	// re create staged updates context
	StagedUpdatesCtx_Free (&op->staged_updates) ;
	op->staged_updates = StagedUpdatesCtx_New () ;

	uint records_count = array_len (op->records) ;
	// records[0..op->record_idx] had been already emitted, skip them
	for (uint i = op->rec_idx; i < records_count; i++) {
		OpBase_DeleteRecord (op->records+i) ;
	}
	array_clear (op->records) ;
	op->rec_idx = 0 ;

	return OP_OK ;
}

static void UpdateFree
(
	OpBase *ctx
) {
	OpUpdate *op = (OpUpdate *)ctx ;

	if (op->staged_updates != NULL) {
		StagedUpdatesCtx_Free (&op->staged_updates) ;
	}

	// free each update context
	if (op->update_ctxs != NULL) {
		raxFreeWithCallback (op->update_ctxs, (void(*)(void *))UpdateCtx_Free) ;
		op->update_ctxs = NULL ;
	}

	if (op->records != NULL) {
		uint64_t n = array_len (op->records) ;
		for (uint64_t i = op->rec_idx; i < n; i++) {
			OpBase_DeleteRecord (op->records+i) ;
		}

		array_free (op->records) ;
		op->records = NULL ;
	}

	raxStop (&op->it) ;
}

