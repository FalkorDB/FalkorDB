/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_sort.h"
#include "op_project.h"
#include "op_aggregate.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/qsort.h"
#include "../../util/rmalloc.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult SortInit(OpBase *opBase);
static Record SortConsume(OpBase *opBase);
static OpResult SortReset(OpBase *opBase);
static OpBase *SortClone(const ExecutionPlan *plan, const OpBase *opBase);
static void SortFree(OpBase *opBase);

// function to compare two records on a subset of fields
// return value similar to strcmp
static int _record_cmp
(
	Record a,
	Record b,
	OpSort *op
) {
	ASSERT (a->owner == b->owner) ;

	uint comparison_count = array_len (op->sort_offsets) ;
	for(uint i = 0; i < comparison_count; i++) {
		SIValue aVal = Record_Get (a, op->sort_offsets[i]) ;
		SIValue bVal = Record_Get (b, op->sort_offsets[i]) ;

		int rel = SIValue_Compare (aVal, bVal, NULL) ;
		if (rel == 0) {
			continue ;  // elements are equal; try next ORDER BY element
		}

		rel *= op->directions[i] ; // flip value for descending order
		return rel ;
	}

	return 0 ;
}

static int _buffer_elem_cmp
(
	const Record *a,
	const Record *b,
	OpSort *op
) {
	return _record_cmp(*a, *b, op);
}

static void _accumulate
(
	OpSort *op,
	Record r
) {
	if(op->limit == UNLIMITED) {
		// not using a heap
		array_append (op->buffer, r) ;
		return ;
	}

	// add record to the heap if limit hasn't been reached
	if (Heap_count (op->heap) < op->limit) {
		Heap_offer (&op->heap, r) ;
	} else {
		// no room in the heap, see if we need to replace heap's head 
		if (_record_cmp (Heap_peek (op->heap), r, op) > 0) {
			// add record to heap
			Record replaced = Heap_poll (op->heap) ;
			OpBase_DeleteRecord (&replaced) ;
			Heap_offer (&op->heap, r) ;
		} else {
			// discard record
			OpBase_DeleteRecord (&r) ;
		}
	}
}

static inline Record _handoff
(
	OpSort *op
) {
	if(op->record_idx < array_len(op->buffer)) {
		return op->buffer[op->record_idx++];
	}
	return NULL;
}

static void _map_expressions
(
	OpSort *op,
	AR_ExpNode **exps
) {
	//--------------------------------------------------------------------------
	// compute expressions record index
	//--------------------------------------------------------------------------

	uint n = array_len (op->exps) ;
	op->to_eval        = array_new (AR_ExpNode *, n) ;  // expressions to eval
	op->sort_offsets   = array_new (uint, n) ;          // used for sorting
	op->record_offsets = array_new (uint, n) ;          // used for exp eval

	// process sort expressions
	for (uint i = 0; i < n; i++) {
		AR_ExpNode *exp = op->exps[i] ;
		const char *alias = exp->resolved_name ;

		bool mapped = OpBase_AliasMapping ((OpBase*)op, alias, NULL) ;
		if (!mapped) {
			// expression value is missing from record, make sure to evaluate
			OpBase_Modifies ((OpBase*)op, alias) ;
			array_append (op->to_eval, exp) ;
		}
	}
}

OpBase *NewSortOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps,
	int *directions
) {
	ASSERT (exps       != NULL) ;
	ASSERT (plan       != NULL) ;
	ASSERT (directions != NULL) ;

	OpSort *op = rm_calloc (1, sizeof (OpSort)) ;

	op->exps       = exps ;
	op->first      = true ;
	op->limit      = UNLIMITED ;
	op->directions = directions ;

	// set our Op operations
	OpBase_Init ((OpBase *)op, OPType_SORT, "Sort", SortInit, SortConsume,
			SortReset, NULL, SortClone, SortFree, false, plan) ;

	_map_expressions (op, op->exps) ;

	return (OpBase *)op ;
}

static OpResult SortInit
(
	OpBase *opBase
) {
	OpSort *op = (OpSort *)opBase;

	// set skip and limit if present in the execution-plan
	ExecutionPlan_ContainsSkip(opBase->parent, &op->skip);
	ExecutionPlan_ContainsLimit(opBase->parent, &op->limit);

	// if there is LIMIT value, l, set in the current clause,
	// the operation must return the top l records with respect to
	// the sorting criteria. In order to do so, it must collect the l records,
	// but if there is a SKIP value, s, set, it must collect l+s records,
	// sort them and return the top l
	if(op->limit != UNLIMITED) {
		op->limit += op->skip;
		// if a limit is specified, use heapsort to poll the top N
		op->heap = Heap_new((heap_cmp)_record_cmp, op);
	} else {
		// if all records are being sorted, use quicksort
		op->buffer = array_new(Record, 32);
	}

	//--------------------------------------------------------------------------
	// compute expressions record index
	//--------------------------------------------------------------------------

	uint n = array_len (op->exps) ;
	op->sort_offsets   = array_new (uint, n) ;          // used for sorting
	op->record_offsets = array_new (uint, n) ;          // used for exp eval

	// process sort expressions
	for (uint i = 0; i < n; i++) {
		int rec_idx ;
		AR_ExpNode *exp = op->exps[i] ;
		const char *alias = exp->resolved_name ;

		bool mapped = OpBase_AliasMapping ((OpBase*)op, alias, &rec_idx) ;
		ASSERT (mapped == true) ;

		array_append (op->sort_offsets, rec_idx) ;
		array_append (op->record_offsets, rec_idx) ;
	}

	return OP_OK;
}

static Record SortConsume
(
	OpBase *opBase
) {
	OpSort *op = (OpSort *)opBase ;

	if (!op->first) {
		return _handoff (op) ;
	}
	op->first = false ;

	//--------------------------------------------------------------------------
	// consume all records from child
	//--------------------------------------------------------------------------

	Record r ;
	OpBase *child = op->op.children[0] ;

	while ((r = OpBase_Consume (child))) {
		// evaluate sort expressions
		for (uint i = 0; i < array_len (op->to_eval); i++) {
			SIValue v = AR_EXP_Evaluate (op->to_eval[i], r) ;
			Record_Add (r, op->record_offsets[i], v) ;
		}

		_accumulate (op, r) ;
	}

	if(op->buffer) {
		sort_r(op->buffer, array_len(op->buffer), sizeof(Record),
				(heap_cmp)_buffer_elem_cmp, op);
	} else {
		// heap
		int records_count = Heap_count (op->heap) ;
		op->buffer = array_newlen (Record, records_count) ;

		for (int i = records_count-1; i >= 0 ; i--) {
			op->buffer[i] = Heap_poll (op->heap) ;
		}
	}

	// pass ordered records downward
	return _handoff(op);
}

// restart iterator
static OpResult SortReset
(
	OpBase *ctx
) {
	OpSort *op = (OpSort *)ctx;
	uint recordCount;

	if(op->heap) {
		recordCount = Heap_count(op->heap);
		for(uint i = 0; i < recordCount; i++) {
			Record r = (Record)Heap_poll(op->heap);
			OpBase_DeleteRecord(&r);
		}
	}

	if(op->buffer) {
		recordCount = array_len(op->buffer);
		for(uint i = op->record_idx; i < recordCount; i++) {
			Record r = op->buffer[i];
			OpBase_DeleteRecord(&r);
		}
		array_clear(op->buffer);
	}

	op->record_idx = 0;

	return OP_OK;
}

static OpBase *SortClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_SORT);
	OpSort *op = (OpSort *)opBase;
//	int *directions;
//	AR_ExpNode **exps;
//	array_clone(directions, op->directions);
//	array_clone_with_cb(exps, op->exps, AR_EXP_Clone);
//	return NewSortOp(plan, exps, directions);

	OpSort *clone = rm_calloc (1, sizeof (OpSort)) ;

	clone->first = true ;
	clone->limit = UNLIMITED ;

	array_clone (clone->directions, op->directions) ;
	array_clone (clone->sort_offsets, op->sort_offsets) ;
	array_clone (clone->record_offsets, op->record_offsets) ;

	array_clone_with_cb (clone->exps, op->exps, AR_EXP_Clone) ;
	array_clone_with_cb (clone->to_eval, op->to_eval, AR_EXP_Clone) ;

	// set our Op operations
	OpBase_Init ((OpBase *)clone, OPType_SORT, "Sort", SortInit, SortConsume,
			SortReset, NULL, SortClone, SortFree, false, plan) ;

	return (OpBase*)clone ;
}

void SortBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
) {
	OpSort *op = (OpSort *)opBase ;
	opBase->plan = plan ;

	// introduce the projected aliases to the plan record-mapping, 
	for (uint i = 0; i < array_len (op->to_eval); i++) {
		// the projected record will associate values with their resolved name
		// to ensure that space is allocated for each entry
		OpBase_Modifies ((OpBase *)op, op->to_eval[i]->resolved_name) ;
	}
}

// frees sort
static void SortFree
(
	OpBase *ctx
) {
	OpSort *op = (OpSort *)ctx;

	if(op->heap) {
		uint recordCount = Heap_count(op->heap);
		for(uint i = 0; i < recordCount; i++) {
			Record r = (Record)Heap_poll(op->heap);
			OpBase_DeleteRecord(&r);
		}
		Heap_free(op->heap);
		op->heap = NULL;
	}

	if(op->buffer) {
		uint recordCount = array_len(op->buffer);
		for(uint i = op->record_idx; i < recordCount; i++) {
			Record r = op->buffer[i];
			OpBase_DeleteRecord(&r);
		}
		array_free(op->buffer);
		op->buffer = NULL;
	}

	if(op->sort_offsets) {
		array_free(op->sort_offsets);
		op->sort_offsets = NULL;
	}

	if (op->record_offsets) {
		array_free (op->record_offsets) ;
		op->record_offsets = NULL ;
	}

	if(op->directions) {
		array_free(op->directions);
		op->directions = NULL;
	}

	if (op->to_eval) {
		uint exps_count = array_len(op->to_eval);
		for(uint i = 0; i < exps_count; i++) {
			AR_EXP_Free(op->to_eval[i]);
		}
		array_free (op->to_eval) ;
		op->to_eval = NULL ;
	}

	if (op->exps) {
	//	uint exps_count = array_len(op->exps);
	//	for(uint i = 0; i < exps_count; i++) {
	//		AR_EXP_Free(op->exps[i]);
	//	}
		array_free(op->exps);
		op->exps = NULL;
	}
}
