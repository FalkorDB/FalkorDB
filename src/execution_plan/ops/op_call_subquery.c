/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_join.h"
#include "op_call_subquery.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// forward declarations
static void CallSubqueryFree(OpBase *opBase);
static OpResult CallSubqueryInit(OpBase *opBase);
static OpResult CallSubqueryReset(OpBase *opBase);
static Record CallSubqueryConsume(OpBase *opBase);
static Record CallSubqueryConsumeEager(OpBase *opBase);
static OpBase *CallSubqueryClone(const ExecutionPlan *plan,
	const OpBase *opBase);

// subquery input type
typedef enum {
	FEEDER_NONE,          // non-initialized
	FEEDER_ARGUMENT,      // arguments
	FEEDER_ARGUMENT_LIST  // argument lists
} FeederType;

// subquery input
typedef struct Feeder {
	union {
		OpArgument **arguments;        // array of argument ops
		ArgumentList **argumentLists;  // array of argument list ops
	};
	FeederType type;                   // type of input
} Feeder;

// call sub query operation type
typedef struct {
	OpBase op;
	bool first;         // is this the first call to consume
	bool is_eager;      // is the op eager
	bool is_returning;  // is the subquery returning or not
	OpBase *body;       // first op in the embedded execution-plan
	OpBase *lhs;        // op from which records are pulled
	Record r;           // current record consumed from lhs
	Record *records;    // records aggregated by the operation
	Feeder feeders;     // feeders to the body (Args/ArgLists)
} OpCallSubquery;

// find the deepest child of a root operation (feeder), and append it to the
// arguments/argumentLists array of the CallSubquery operation
static void _append_feeder
(
	OpCallSubquery *op,  // CallSubquery operation
	OpBase *branch       // root op of the branch
) {
	ASSERT(op     != NULL);
	ASSERT(branch != NULL);

	// get the deepest left child
	while(OpBase_ChildCount(branch) > 0) {
		branch = OpBase_GetChild(branch, 0);
	}

	if(op->is_eager) {
		ASSERT(OpBase_Type(branch) == OPType_ARGUMENT_LIST);
		array_append(op->feeders.argumentLists, (ArgumentList *)branch);
	} else {
		ASSERT(OpBase_Type(branch) == OPType_ARGUMENT);
		array_append(op->feeders.arguments, (OpArgument *)branch);
	}
}

// plant the input record(s) in the ArgumentList operation(s)
static void _plant_records_ArgumentLists
(
	OpCallSubquery *op  // CallSubquery operation
) {
	int n_branches = (int)array_len(op->feeders.argumentLists);

	for(int i = 0; i < n_branches - 1; i++) {
		Record *records_clone;
		array_clone_with_cb(records_clone, op->records,
			OpBase_CloneRecord);
		ArgumentList_AddRecordList(op->feeders.argumentLists[i],
			records_clone);
	}

	// [optimization]
	// if possible last branch takes ownership over the records array
	// this saves cloning the records
	if(op->is_returning) {
		// give the last branch the original records
		ArgumentList_AddRecordList(
			op->feeders.argumentLists[n_branches - 1], op->records);

		// responsibility for the records is passed to the argumentList op(s)
		op->records = NULL;
	} else {
		// give the last branch a clone of the original record(s)
		Record *records_clone;
		array_clone_with_cb(records_clone, op->records,
			OpBase_CloneRecord);
		ArgumentList_AddRecordList(
			op->feeders.argumentLists[n_branches - 1], records_clone);
	}
}

// plant the input record in the Argument operation(s)
static void _plant_records_Arguments
(
	OpCallSubquery *op  // CallSubquery operation
) {
	uint n_branches = array_len(op->feeders.arguments);
	for(uint i = 0; i < n_branches; i++) {
		Argument_AddRecord(op->feeders.arguments[i], OpBase_CloneRecord(op->r));
	}
}

// creates a new CallSubquery operation
OpBase *NewCallSubqueryOp
(
	const ExecutionPlan *plan,  // execution plan
	bool is_eager,              // is the subquery eager or not
	bool is_returning           // is the subquery returning or not
) {
	OpCallSubquery *op = rm_calloc(1, sizeof(OpCallSubquery));

	op->first        = true;
	op->is_eager     = is_eager;
	op->is_returning = is_returning;

	// set the consume function according to eagerness of the op
	fpConsume consumeFunc = is_eager ?
		CallSubqueryConsumeEager :
		CallSubqueryConsume;

	OpBase_Init((OpBase *)op, OPType_CALLSUBQUERY, "CallSubquery",
			CallSubqueryInit, consumeFunc, CallSubqueryReset, NULL,
			CallSubqueryClone, CallSubqueryFree, false, plan);

	return (OpBase *)op;
}

static OpResult CallSubqueryInit
(
	OpBase *opBase  // CallSubquery operation to initialize
) {
	OpCallSubquery *op = (OpCallSubquery *)opBase;

	// set the lhs (supplier) branch to be the first child, and rhs branch
	// (body) to be the second
	ASSERT(OpBase_ChildCount(opBase) <= 2);

	if(OpBase_ChildCount(opBase) == 2) {
		op->lhs  = OpBase_GetChild(opBase, 0);
		op->body = OpBase_GetChild(opBase, 1);
	} else {
		// no supplier, just sub-query
		op->lhs  = NULL;
		op->body = OpBase_GetChild(opBase, 0);
	}

	// search for the ArgumentList\Argument ops, depending if the op is eager
	if(op->is_eager) {
		op->feeders.type = FEEDER_ARGUMENT_LIST;
		op->feeders.argumentLists = array_new(ArgumentList *, 1);
	} else {
		op->feeders.type = FEEDER_ARGUMENT;
		op->feeders.arguments = array_new(OpArgument *, 1);
	}

	// in the case the subquery contains a `UNION` or `UNION ALL` clause, we
	// need to duplicate the input records to the multiple branches of the
	// Join op, that will be placed in one of the first two ops of the sub-plan
	// (first child or its child, according to whether there is an `ALL`)
	// "CALL {RETURN 1 AS num UNION RETURN 2 AS num} RETURN num"
	// "CALL {RETURN 1 AS num UNION ALL RETURN 2 AS num} RETURN num"
	//
	// search for a Join op

	OPType t         = OPType_JOIN;
	OPType blacklist = OPType_CALLSUBQUERY;  // do not search nested calls
	OpBase *op_join  = ExecutionPlan_LocateOpMatchingTypes(op->body, &t, 1,
			&blacklist, 1);

	//--------------------------------------------------------------------------
	// collect feeding points
	//--------------------------------------------------------------------------

	// found a join op
	if(op_join != NULL) {
		// how many branches are joined?
		uint n_branches = OpBase_ChildCount((OpBase *)op_join);

		// add a feeding point to each joined branch
		for(uint i = 0; i < n_branches; i++) {
			OpBase *branch = OpBase_GetChild((OpBase *)op_join, i);
			_append_feeder(op, branch);
		}
	} else {
		// no join, just a single feeding point
		OpBase *branch = op->body;
		_append_feeder(op, branch);
	}

	return OP_OK;
}

// passes a record to the parent op
// if the subquery is non-returning, yield input record(s)
// otherwise, yields a record produced by the subquery
static Record _handoff_eager
(
	OpCallSubquery *op  // CallSubquery operation
) {
	ASSERT(op->is_returning || op->records != NULL);

	if(!op->is_returning) {
		// subquery doesn't return anything, yield lhs records
		// NOTICE: the order of records reverses here
		return array_len(op->records) > 0 ? array_pop(op->records) : NULL;
	}

	// emit sub-query records
	return OpBase_Consume(op->body);
}

// eagerly consume and all records from the lhs (if exists)
// pass the records to the ArgumentList operation(s)
// if the subquery is returning return the consumed record(s) from the body
// otherwise return the input record(s)
static Record CallSubqueryConsumeEager
(
	OpBase *opBase  // operation
) {
	OpCallSubquery *op = (OpCallSubquery *)opBase;

	// if eager consumption has already occurred, don't consume again
	if(!op->first) {
		return _handoff_eager(op);
	}

	ASSERT(op->records == NULL);

	Record r;
	op->first   = false;
	op->records = array_new(Record, 1);

	// eagerly consume all records from lhs if exists or create an empty record
	// and place them \ it in op->records
	if(op->lhs) {
		// consume lhs records until depletion
		while((r = OpBase_Consume(op->lhs))) {
			array_append(op->records, r);
		}

		// propagate reset to lhs, to release RediSearch index locks (if any)
		OpBase_PropagateReset(op->lhs);
	} else {
		r = OpBase_CreateRecord((OpBase *)op);
		array_append(op->records, r);
	}

	// in case no records were produced by lhs we can quickly return
	if(unlikely(array_len(op->records) == 0)) {
		return NULL;
	}

	_plant_records_ArgumentLists(op);

	if(!op->is_returning) {
		// deplete body and discard records
		while((r = OpBase_Consume(op->body))) {
			OpBase_DeleteRecord(&r);
		}
	}

	return _handoff_eager(op);
}

// tries to consumes a record from the body, merge it with the current input
// record and return it
// if body is depleted for this record, tries to consume
// a record from the lhs, and repeat the process (if the lhs record is not NULL)
static Record _consume_and_merge
(
	OpCallSubquery *op  // call sub query op
) {
	// consume record from sub-query
	Record consumed;
	consumed = OpBase_Consume(op->body);

	// sub-query depleted
	if(consumed == NULL) {
		OpBase_PropagateReset(op->body);
		OpBase_DeleteRecord(&op->r);
		return NULL;
	}

	// merge consumed record into a clone of the received record
	Record clone = OpBase_CloneRecord(op->r);
	OpBase_MergeRecords(clone, &consumed);

	return clone;
}

// tries to consume a record from the body if successful
// returns the merged\unmerged record with the input record (op->r)
// according to the is_returning flag
// depletes child if is_returning is off (discard body records)
static Record _handoff
(
	OpCallSubquery *op
) {
	ASSERT(op->r != NULL);

	//--------------------------------------------------------------------------
	// returning subquery
	//--------------------------------------------------------------------------

	if(op->is_returning) {
		return _consume_and_merge(op);
	}

	//--------------------------------------------------------------------------
	// non-returning subquery
	//--------------------------------------------------------------------------

	Record consumed;
	// drain the body, deleting the subquery records and return current record
	while((consumed = OpBase_Consume(op->body))) {
		OpBase_DeleteRecord(&consumed);
	}

	OpBase_PropagateReset(op->body);
	Record r = op->r;
	op->r = NULL;
	return r;
}

// consumes a record from the lhs, plants it in the Argument\List op(s)
// and consumes a record from the body until depletion
// in case the subquery is returning, merges the input (lhs) record with the
// output record otherwise, the input record is passed as-is
// upon depletion of the body repeats the above
// depletion of lhs yields depletion of this operation
static Record CallSubqueryConsume
(
	OpBase *opBase  // operation
) {
	OpCallSubquery *op = (OpCallSubquery *)opBase;

	// if there are more records to consume from body, consume them before
	// consuming another record from lhs
emit:
	if(op->r != NULL) {
		Record r = _handoff(op);
		if(r != NULL) {
			return r;
		}
	}

	ASSERT(op->r == NULL);

	// consume from lhs if exists, otherwise create a dummy-record to pass to
	// the body (rhs)
	// the latter case will happen AT MOST once
	if(op->lhs) {
		op->r = OpBase_Consume(op->lhs);
	} else if(op->first) {
		// create an empty record
		op->r     = OpBase_CreateRecord((OpBase *)op);
		op->first = false;
	}

	// plant the record consumed at the Argument ops
	if(op->r != NULL) {
		_plant_records_Arguments(op);
		goto emit;
	}

	// no records - lhs depleted
	return NULL;
}

// frees CallSubquery stored records
static void _free_records
(
	OpCallSubquery *op  // call sub query op
) {
	if(op->records != NULL) {
		uint n_records = array_len(op->records);
		for(uint i = 0; i < n_records; i++) {
			OpBase_DeleteRecord(op->records+i);
		}
		array_free(op->records);
		op->records = NULL;
	}

	if(op->r != NULL) {
		OpBase_DeleteRecord(&op->r);
	}
}

// resets a CallSubquery operation
static OpResult CallSubqueryReset
(
	OpBase *opBase  // operation
) {
	OpCallSubquery *op = (OpCallSubquery *)opBase;

	_free_records(op);
	op->first = true;

	return OP_OK;
}

// clones a CallSubquery operation
static OpBase *CallSubqueryClone
(
	const ExecutionPlan *plan,  // plan
	const OpBase *opBase        // operation to clone
) {
	ASSERT(opBase->type == OPType_CALLSUBQUERY);
	OpCallSubquery *op = (OpCallSubquery *) opBase;

	return NewCallSubqueryOp(plan, op->is_eager, op->is_returning);
}

// frees a CallSubquery operation
static void CallSubqueryFree
(
	OpBase *op  // call sub query op
) {
	OpCallSubquery *_op = (OpCallSubquery *) op;

	// free op's stored records
	_free_records(_op);

	if(_op->feeders.type != FEEDER_NONE) {
		if(_op->feeders.type == FEEDER_ARGUMENT) {
			ASSERT(_op->feeders.arguments != NULL);
			array_free(_op->feeders.arguments);
			_op->feeders.arguments = NULL;
		} else if(_op->feeders.type == FEEDER_ARGUMENT_LIST) {
			ASSERT(_op->feeders.argumentLists != NULL);
			array_free(_op->feeders.argumentLists);
			_op->feeders.argumentLists = NULL;
		}
	}
}

