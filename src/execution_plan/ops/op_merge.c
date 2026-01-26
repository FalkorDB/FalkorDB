/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_merge.h"
#include "../../RG.h"
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
	rax *blueprint,
	raxIterator updates,
	Record *records,
	uint record_count
) {
	ASSERT (record_count > 0) ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	// in cases such as:
	// MERGE (n) ON MATCH SET n:L
	// make sure L is of the right dimensions
	ensureMatrixDim (gc, blueprint) ;

	// for each record to update
	for (uint i = 0 ; i < record_count ; i++) {
		Record r = records[i] ;
		// evaluate update expressions
		raxSeek (&updates, "^", NULL, 0) ;
		while (raxNext (&updates)) {
			EntityUpdateEvalCtx *ctx = updates.data ;
			EvalEntityUpdates (gc, node_pending_updates, edge_pending_updates,
					r, ctx, true) ;
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
	// merge is an operator with two or three children
	// they will be created outside of here
	// as with other multi-stream operators
	// (see CartesianProduct and ValueHashJoin)
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
	// merge has 2 children if it is the first clause, and 3 otherwise
	// - if there are 3 children
	//   the first should resolve the Merge pattern's bound variables
	// - the next (first if there are 2 children, second otherwise)
	//   should attempt to match the pattern
	// - the last creates the pattern
	ASSERT(opBase->childCount == 2 || opBase->childCount == 3);
	OpMerge *op = (OpMerge *)opBase;
	if(opBase->childCount == 2) {
		// if we only have 2 streams, the first one is the bound variable stream
		// and the second is the match stream
		op->match_stream  = opBase->children[0];
		op->create_stream = opBase->children[1];

		ASSERT(OpBase_Type(op->create_stream) == OPType_MERGE_CREATE);
		return OP_OK;
	}

	// if we have 3 streams, the first is the bound variable stream
	// the second is the match stream, and the third is the create stream
	op->bound_variable_stream = opBase->children[0];
	op->match_stream          = opBase->children[1];
	op->create_stream         = opBase->children[2];

	ASSERT(OpBase_Type(op->create_stream) == OPType_MERGE_CREATE);
	ASSERT(op->bound_variable_stream != NULL &&
		   op->match_stream          != NULL &&
		   op->create_stream         != NULL);

	// find and store references to the:
	// Argument taps for the Match and Create streams
	// the Match stream is populated by an Argument tap
	// store a reference to it
	op->match_argument_tap =
		(OpArgument *)ExecutionPlan_LocateOp(op->match_stream, OPType_ARGUMENT);
	ASSERT(op->match_argument_tap != NULL);

	// if the create stream is populated by an Argument tap, store a reference to it.
	op->create_argument_tap =
		(OpArgument *)ExecutionPlan_LocateOp(op->create_stream, OPType_ARGUMENT);
	ASSERT(op->create_argument_tap != NULL);

	// set up an array to store records produced by the bound variable stream
	op->input_records = array_new(Record, 1);

	return OP_OK;
}

static Record _handoff
(
	OpMerge *op
) {
	if(op->output_rec_idx < array_len(op->output_records)) {
		return op->output_records[op->output_rec_idx++];
	} else {
		return NULL;
	}
}


// records which were scheduled for creation but resulted in duplications
// e.g.
// UNWIND [{a:1, b:1}, {a:1, b:2}] AS x
// MERGE (n {v:x.a})
// ON CREATE SET n.created = true
// ON MATCH  SET n.matched = true
//
// in this example the first record {a:1, b:1} will create the node 'n'
// the second record {a:1, b:2} will also be scheduled for creation but we'll
// detect it will create a duplication, and so for the {a:1, b:2} record
// we'll need to match the 'n' node
//
// this function matches duplicates and apply the ON MATCH directive if present
static void _processPostponedRecords
(
	OpMerge *op,
	uint match_count,  // number of records matched
	uint create_count  // number of records created

) {
	ASSERT(op != NULL);

	// run through the postponed records
	// match each one and add them to the output array
	int n = array_len(op->postponed_match);
	for(int i = 0; i < n; i++) {
		Record r = array_pop(op->postponed_match);

		// propagate record to the top of the Match stream
		Argument_AddRecord(op->match_argument_tap, r);

		// pull match stream
		r = _pullFromStream(op->match_stream);
		ASSERT(r != NULL);
		ASSERT(_pullFromStream(op->match_stream) == NULL);  // expecting a single record

		// add record to outputs
		array_append(op->output_records, r);
	}

	// if we are setting properties with ON MATCH, compute pending updates
	if(op->on_match && n > 0) {
		_UpdateProperties (op->node_pending_updates, op->edge_pending_updates,
				op->on_match, op->on_match_it,
				op->output_records + match_count + create_count, n) ;
	}
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
	if(op->output_records) {
		return _handoff(op);
	}

	//--------------------------------------------------------------------------
	// consume bound stream
	//--------------------------------------------------------------------------

	op->output_records  = array_new(Record, 32);
	op->postponed_match = array_new(Record, 0);

	// if we have a bound variable stream
	// pull from it and store records until depleted
	if(op->bound_variable_stream) {
		Record input_record;
		while((input_record = _pullFromStream(op->bound_variable_stream))) {
			array_append(op->input_records, input_record);
		}
	}

	//--------------------------------------------------------------------------
	// match pattern
	//--------------------------------------------------------------------------

	uint match_count         = 0;
	uint create_count        = 0;
	bool reading_matches     = true;
	bool must_create_records = false;

	// match mode: attempt to resolve the pattern for every record from
	// the bound variable stream, or once if we have no bound variables
	while(reading_matches) {
		Record lhs_record = NULL;
		if(op->input_records) {
			// if we had bound variables but have depleted our input records,
			// we're done pulling from the Match stream
			if(array_len(op->input_records) == 0) {
				break;
			}

			// pull a new input record
			lhs_record = array_pop(op->input_records);

			// propagate record to the top of the Match stream
			// (must clone the Record, as it will be freed in the Match stream)
			Argument_AddRecord(op->match_argument_tap, OpBase_CloneRecord(lhs_record));
		} else {
			// this loop only executes once if we don't have input records
			// resolving bound variables
			reading_matches = false;
		}

		Record rhs_record;
		bool should_create_pattern = true;
		// retrieve Records from the Match stream until it's depleted
		while((rhs_record = _pullFromStream(op->match_stream))) {
			// pattern was successfully matched
			should_create_pattern = false;
			array_append(op->output_records, rhs_record);
			match_count++;
		}

		if(should_create_pattern) {
			// transfer the unmatched record to the Create stream
			// we don't need to clone the record
			// as it won't be accessed again outside that stream
			// but we must make sure its elements are access-safe
			// as the input stream will be freed
			// before entities are created
			if(lhs_record) {
				Argument_AddRecord(op->create_argument_tap, lhs_record);
				lhs_record = NULL;
			}

			Record r = _pullFromStream(op->create_stream);
			if(r != NULL) {
				// duplicate detected, this record need to be matched
				// once we commit all of the changes
				array_append(op->postponed_match, r);
			} else {
				must_create_records = true;
			}
		}

		// free the LHS Record if we haven't transferred it to the Create stream
		if(lhs_record) {
			OpBase_DeleteRecord(&lhs_record);
		}
	}

	//--------------------------------------------------------------------------
	// compute updates and create
	//--------------------------------------------------------------------------

	// explicitly free the read streams in case either holds an index read lock
	if(op->bound_variable_stream) {
		OpBase_PropagateReset(op->bound_variable_stream);
	}
	OpBase_PropagateReset(op->match_stream);

	op->node_pending_updates = HashTableCreate(&_dt);
	op->edge_pending_updates = HashTableCreate(&_dt);

	// if we are setting properties with ON MATCH, compute all pending updates
	if(op->on_match && match_count > 0) {
		_UpdateProperties (op->node_pending_updates, op->edge_pending_updates,
			op->on_match, op->on_match_it, op->output_records, match_count) ;
	}

	if(must_create_records) {
		// commit all pending changes on the Create stream
		// 'MergeCreate_Commit' acquire write lock!
		// write lock is released further down
		MergeCreate_Commit(op->create_stream);

		// we only need to pull the created records if we're returning results
		// or performing updates on creation
		// pull all records from the Create stream
		Record created_record;
		while((created_record = _pullFromStream(op->create_stream))) {
			array_append(op->output_records, created_record);
			create_count++;
		}

		// if we are setting properties with ON CREATE
		// compute all pending updates
		// TODO: note we're under lock at this point! is there a way
		// to compute these changes before locking ?
		if(op->on_create) {
			_UpdateProperties (op->node_pending_updates,
				op->edge_pending_updates, op->on_create, op->on_create_it,
				op->output_records + match_count, create_count) ;
		}
	}

	// handle postpone records
	if(array_len(op->postponed_match) > 0) {
		// reset match stream, required as we've commited data to the graph
		OpBase_PropagateReset(op->match_stream);
		_processPostponedRecords(op, match_count, create_count);
		OpBase_PropagateReset(op->match_stream);
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

	// free input records
	if(op->input_records) {
		uint input_count = array_len(op->input_records);
		for(uint i = 0; i < input_count; i ++) {
			OpBase_DeleteRecord(op->input_records+i);
		}
		array_free(op->input_records);
		op->input_records = NULL;
	}

	// free postponed match records
	if(op->postponed_match != NULL) {
		uint n = array_len(op->postponed_match);
		for(uint i = 0; i < n; i ++) {
			OpBase_DeleteRecord(op->postponed_match + i);
		}
		array_free(op->postponed_match);
		op->postponed_match = NULL;
	}

	// free output records
	if(op->output_records != NULL) {
		uint output_count = array_len(op->output_records);
		// output_records[0..output_rec_idx] had been already emitted, skip them
		for(uint i = op->output_rec_idx; i < output_count; i ++) {
			OpBase_DeleteRecord(op->output_records+i);
		}
		array_free(op->output_records);
		op->output_records = NULL;
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

