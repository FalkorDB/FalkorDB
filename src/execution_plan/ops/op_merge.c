/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_merge.h"
#include "op_merge_create.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"
#include "../../schema/schema.h"
#include "../../util/rax_extensions.h"
#include "../../arithmetic/arithmetic_expression.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult MergeInit(OpBase *opBase);
static Record MergeConsume(OpBase *opBase);
static OpBase *MergeClone(const ExecutionPlan *plan, const OpBase *opBase);
static void MergeFree(OpBase *opBase);

// fake hash function
// hash of key is simply key
static uint64_t _id_hash
(
	const void *key
) {
	return ((uint64_t)key);
}

// hashtable entry free callback
static void freeCallback
(
	dict *d,
	void *val
) {
	PendingUpdateCtx_Free((PendingUpdateCtx*)val);
}

// hashtable callbacks
static dictType _dt = { _id_hash, NULL, NULL, NULL, NULL, freeCallback, NULL,
	NULL, NULL, NULL};

//------------------------------------------------------------------------------
// ON MATCH / ON CREATE logic
//------------------------------------------------------------------------------

// apply a set of updates to the given records
static void _UpdateProperties
(
	dict *node_pending_updates,
	dict *edge_pending_updates,
	raxIterator updates,
	Record *records,
	uint record_count
) {
	ASSERT(record_count > 0);
	GraphContext *gc = QueryCtx_GetGraphCtx();

	for(uint i = 0; i < record_count; i++) {  // for each record to update
		Record r = records[i];
		// evaluate update expressions
		raxSeek(&updates, "^", NULL, 0);
		while(raxNext(&updates)) {
			EntityUpdateEvalCtx *ctx = updates.data;
			EvalEntityUpdates(gc, node_pending_updates, edge_pending_updates,
					r, ctx, true);
		}
	}
}

//------------------------------------------------------------------------------
// Merge logic
//------------------------------------------------------------------------------

static inline Record _pullFromStream
(
	OpBase *branch
) {
	return OpBase_Consume(branch);
}

static void _InitializeUpdates
(
	OpMerge *op,
	rax *updates,
	raxIterator *it
) {
	// if we have ON MATCH / ON CREATE directives
	// set the appropriate record IDs of entities to be updated
	raxStart(it, updates);
	raxSeek(it, "^", NULL, 0);
	// iterate over all expressions
	while(raxNext(it)) {
		EntityUpdateEvalCtx *ctx = it->data;
		// set the record index for every entity modified by this operation
		ctx->record_idx = OpBase_Modifies((OpBase *)op, ctx->alias);
	}
}

// free node and edge pending updates
static inline void _free_pending_updates
(
	OpMerge *op
) {
	if(op->node_pending_updates) {
		HashTableRelease(op->node_pending_updates);
		op->node_pending_updates = NULL;
	}

	if(op->edge_pending_updates) {
		HashTableRelease(op->edge_pending_updates);
		op->edge_pending_updates = NULL;
	}
}

OpBase *NewMergeOp
(
	const ExecutionPlan *plan,
	rax *on_match,
	rax *on_create
) {
	OpMerge *op = rm_calloc(1, sizeof(OpMerge));

	op->on_match  = on_match;
	op->on_create = on_create;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_MERGE, "Merge", MergeInit, MergeConsume,
			NULL, NULL, MergeClone, MergeFree, true, plan);

	if(op->on_match) _InitializeUpdates(op, op->on_match, &op->on_match_it);
	if(op->on_create) _InitializeUpdates(op, op->on_create, &op->on_create_it);

	return (OpBase *)op;
}

static OpResult MergeInit
(
	OpBase *opBase
) {
	// merge has 3 children
	// the first should resolve the Merge pattern's bound variables
	// the second should attempt to match the pattern
	// the last creates the pattern
	ASSERT(opBase->childCount == 3);

	OpMerge *op = (OpMerge *)opBase;

	op->bound_stream  = opBase->children[0];
	op->match_stream  = opBase->children[1];
	op->create_stream = opBase->children[2];

	return OP_OK;
}

static Record _handoff
(
	OpMerge *op
) {
	if(array_len(op->matched_records) > 0)
		return array_pop(op->matched_records);

	if(array_len(op->created_records) > 0)
		return array_pop(op->created_records);

	return NULL;
}

static Record MergeConsume
(
	OpBase *opBase
) {
	OpMerge *op = (OpMerge *)opBase;

	//--------------------------------------------------------------------------
	// handoff
	//--------------------------------------------------------------------------

	// return mode, all data was consumed
	if(op->matched_records) return _handoff(op);

	//--------------------------------------------------------------------------
	// consume bound stream
	//--------------------------------------------------------------------------

	op->matched_records = array_new(Record, 32);
	op->created_records = array_new(Record, 32);

	// eagerly deplete bound branch
	// on each record produced
	// try to match MERGE pattern
	// if there are no matches (pattern doesn't exists)
	// store the record for later creation
	// otherwise (pattern exists) collect every match for later emittion
	while((op->r = _pullFromStream(op->bound_stream))) {
		// reset stream before pulling
		OpBase_PropagateReset(op->match_stream);

		// try to match using current record
		Record match_record = _pullFromStream(op->match_stream);

		// no match
		if(match_record == NULL) {
			// set record aside for later creation
			array_append(op->created_records, op->r);
			continue;
		}

		// collect all matches
		array_append(op->matched_records, match_record);
		while((match_record = _pullFromStream(op->match_stream))) {
			array_append(op->matched_records, match_record);
		}

		// free current record
		OpBase_DeleteRecord(op->r);
	}

	// explicitly free the read streams in case either holds an index read lock
	OpBase_PropagateReset(op->bound_stream);
	OpBase_PropagateReset(op->match_stream);

	//--------------------------------------------------------------------------
	// compute updates and create
	//--------------------------------------------------------------------------

	op->node_pending_updates = HashTableCreate(&_dt);
	op->edge_pending_updates = HashTableCreate(&_dt);

	// if we are setting properties with ON MATCH, compute all pending updates
	int match_count = array_len(op->matched_records);
	if(op->on_match && match_count > 0) {
		_UpdateProperties(op->node_pending_updates, op->edge_pending_updates,
			op->on_match_it, op->matched_records, match_count);
	}

	int create_count = array_len(op->created_records);
	if(create_count > 0) {
		// commit all pending changes on the Create stream
		// 'MergeCreate_AddRecords' acquire write lock!
		// write lock is released further down
		MergeCreate_AddRecords((OpMergeCreate*)op->create_stream,
				&op->created_records);

		// MergeCreate_AddRecords modifies the input array
		// update its length
		create_count = array_len(op->created_records);

		// if we are setting properties with ON CREATE
		// compute all pending updates
		// TODO: note we're under lock at this point! is there a way
		// to compute these changes before locking ?
		if(op->on_create) {
			_UpdateProperties(op->node_pending_updates,
				op->edge_pending_updates, op->on_create_it, op->created_records,
				create_count);
		}
	}

	//--------------------------------------------------------------------------
	// update
	//--------------------------------------------------------------------------

	if(HashTableElemCount(op->node_pending_updates) > 0 ||
	   HashTableElemCount(op->edge_pending_updates) > 0) {
		GraphContext *gc = QueryCtx_GetGraphCtx();
		// lock everything
		QueryCtx_LockForCommit(); {
			CommitUpdates(gc, op->node_pending_updates, ENTITY_NODE);
			if(likely(!ErrorCtx_EncounteredError())) {
				CommitUpdates(gc, op->edge_pending_updates, ENTITY_EDGE);
			}
		}
	}

	//--------------------------------------------------------------------------
	// free updates
	//--------------------------------------------------------------------------

	HashTableEmpty(op->node_pending_updates, NULL);
	HashTableEmpty(op->edge_pending_updates, NULL);

	return _handoff(op);
}

static OpBase *MergeClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_MERGE);

	OpMerge *op    = (OpMerge *)opBase;
	rax *on_match  = NULL;
	rax *on_create = NULL;

	if(op->on_match) on_match = raxCloneWithCallback(op->on_match,
			(void *(*)(void *))UpdateCtx_Clone);

	if(op->on_create) on_create = raxCloneWithCallback(op->on_create,
			(void *(*)(void *))UpdateCtx_Clone);

	return NewMergeOp(plan, on_match, on_create);
}

static void MergeFree
(
	OpBase *opBase
) {
	OpMerge *op = (OpMerge *)opBase;

	if(op->matched_records) {
		uint n = array_len(op->matched_records);
		for(uint i = 0; i < n; i++) {
			OpBase_DeleteRecord(op->matched_records[i]);
		}
		array_free(op->matched_records);
		op->matched_records = NULL;
	}

	if(op->created_records) {
		uint n = array_len(op->created_records);
		for(uint i = 0; i < n; i++) {
			OpBase_DeleteRecord(op->created_records[i]);
		}
		array_free(op->created_records);
		op->created_records = NULL;
	}

	_free_pending_updates(op);

	if(op->on_match) {
		raxFreeWithCallback(op->on_match, (void(*)(void *))UpdateCtx_Free);
		op->on_match = NULL;
		raxStop(&op->on_match_it);
	}

	if(op->on_create) {
		raxFreeWithCallback(op->on_create, (void(*)(void *))UpdateCtx_Free);
		op->on_create = NULL;
		raxStop(&op->on_create_it);
	}
}

