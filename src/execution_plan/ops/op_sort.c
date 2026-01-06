/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_sort.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/qsort.h"
#include "../../util/rmalloc.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult SortInit(OpBase *opBase);
static RecordBatch SortConsume(OpBase *opBase);
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

	for (uint i = 0 ; i < comparison_count ; i++) {
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
	return _record_cmp (*a, *b, op) ;
}

// accumulate batch
static void _accumulate
(
	OpSort *op,
	RecordBatch batch
) {
	ASSERT (op    != NULL) ;
	ASSERT (batch != NULL) ;
	ASSERT (RecordBatch_Size (batch) > 0) ;

	size_t batch_size = RecordBatch_Size (batch) ;

	if (op->limit == UNLIMITED) {
		// not using a heap
		// copy batch into buffer
		array_ensure_append (op->buffer, batch, batch_size, Record) ;
		RecordBatch_SetSize (batch, 0) ;
		op->buffer_len += batch_size ;
		return ;
	}

	//--------------------------------------------------------------------------
	// add records to heap
	//--------------------------------------------------------------------------

	uint rec_idx = 0 ;
	size_t heap_count = Heap_count (op->heap) ;

	if (unlikely (heap_count < op->limit)) {
		uint32_t additions = MIN (op->limit - heap_count, batch_size) ;

		// fill heap
		for (; rec_idx < additions ; rec_idx++) {
			Heap_offer (&op->heap, batch[batch_size - rec_idx - 1]) ;
		}

		batch_size -= additions ;
		RecordBatch_SetSize (batch, batch_size) ;

		// batch depleted
		if (unlikely (batch_size == 0)) {
			return ;
		}
	}

	rec_idx = 0 ;
	Record heap_head = Heap_peek (op->heap) ;

	for (; rec_idx < batch_size ; rec_idx++) {
		Record contender = batch[rec_idx] ;

		// if the contender is "better" than the head of the heap
		// (note: In a Max-Heap for ASC sort, head is the largest/worst value)
		if (_record_cmp (heap_head, contender, op) > 0) {
			// disconnect record from batch
			RecordBatch_RemoveRecord (batch, rec_idx) ;

			// replace the worst record with this better one
			Record replaced = Heap_replace_head (op->heap, contender) ;
			OpBase_DeleteRecord (&replaced) ;

			// the head has changed, we must peek again for the next comparison
			heap_head = Heap_peek (op->heap) ;

			rec_idx-- ;
			batch_size-- ;
		}
	}
}

static RecordBatch _handoff
(
	OpSort *op
) {
	size_t n = MIN (64, op->buffer_len) ;

	if (unlikely (n == 0)) {
		return NULL ;
	}

	// create a new empty batch
	RecordBatch batch = RecordBatch_New (n) ;

	// copy records from buffer to batch
	memcpy (batch, op->buffer + op->record_idx, sizeof (Record) * n) ;

	// update counters
	op->buffer_len -= n ;
	op->record_idx += n ;

	return batch ;
}

static void _map_expressions
(
	OpSort *op
) {
	//--------------------------------------------------------------------------
	// compute expressions record index
	//--------------------------------------------------------------------------

	uint n = array_len (op->exps) ;
	op->to_eval = array_new (AR_ExpNode *, n) ;  // expressions to eval

	// process sort expressions
	for (uint i = 0; i < n; i++) {
		AR_ExpNode *exp = op->exps[i] ;
		const char *alias = exp->resolved_name ;

		bool mapped = OpBase_AliasMapping ((OpBase*)op, alias, NULL) ;
		if (!mapped) {
			// expression value is missing from record, make sure to evaluate
			OpBase_Modifies ((OpBase*)op, alias) ;
			array_append (op->to_eval, AR_EXP_Clone (exp)) ;
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

	_map_expressions (op) ;

	return (OpBase *)op ;
}

static OpResult SortInit
(
	OpBase *opBase
) {
	OpSort *op = (OpSort *)opBase;

	// set skip and limit if present in the execution-plan
	ExecutionPlan_ContainsSkip (opBase->parent, &op->skip) ;
	ExecutionPlan_ContainsLimit (opBase->parent, &op->limit) ;

	// if there is LIMIT value, l, set in the current clause,
	// the operation must return the top l records with respect to
	// the sorting criteria. In order to do so, it must collect the l records,
	// but if there is a SKIP value, s, set, it must collect l+s records,
	// sort them and return the top l
	if (op->limit != UNLIMITED) {
		op->limit += op->skip ;
		// if a limit is specified, use heapsort to poll the top N
		op->heap = Heap_new ((heap_cmp)_record_cmp, op) ;
	} else {
		// if all records are being sorted, use quicksort
		op->buffer = array_new (Record, 32) ;
	}

	//--------------------------------------------------------------------------
	// compute expressions record index
	//--------------------------------------------------------------------------

	// process sort expressions
	uint n = array_len (op->exps) ;
	op->sort_offsets = array_new (uint, n) ;  // used for sorting
	for (uint i = 0; i < n; i++) {
		int rec_idx ;
		AR_ExpNode *exp = op->exps[i] ;
		const char *alias = exp->resolved_name ;

		bool mapped = OpBase_AliasMapping ((OpBase*)op, alias, &rec_idx) ;
		ASSERT (mapped == true) ;

		array_append (op->sort_offsets, rec_idx) ;
	}

	// process eval expressions
	n = array_len (op->to_eval) ;
	op->record_offsets = array_new (uint, n) ;  // used for exp eval
	for (uint i = 0; i < n; i++) {
		int rec_idx ;
		AR_ExpNode *exp = op->to_eval[i] ;
		const char *alias = exp->resolved_name ;

		bool mapped = OpBase_AliasMapping ((OpBase*)op, alias, &rec_idx) ;
		ASSERT (mapped == true) ;

		array_append (op->record_offsets, rec_idx) ;
	}

	return OP_OK;
}

static RecordBatch SortConsume
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

	RecordBatch batch ;
	OpBase *child = op->op.children[0]  ;
	SIValue vals[64] ;

	//--------------------------------------------------------------------------
	// pull & process batch
	//--------------------------------------------------------------------------

	size_t exp_count = array_len (op->to_eval) ;

	while ((batch = OpBase_Consume (child))) {
		size_t batch_size = RecordBatch_Size (batch) ;

		// evaluate sort expressions
		for (uint i = 0 ; i < exp_count ; i++) {
			uint rec_idx = op->record_offsets[i] ;
			AR_EXP_Evaluate_Batch (vals, op->to_eval[i], batch, batch_size) ;

			for (uint j = 0 ; j < batch_size ; j++) {
				Record_Add (batch[j], rec_idx, vals[j]) ;
			}
		}

		_accumulate (op, batch) ;
		RecordBatch_Free (&batch) ;
	}

	//--------------------------------------------------------------------------
	// sort records
	//--------------------------------------------------------------------------

	if (op->buffer == NULL) {
		// heap
		op->buffer_len = Heap_count (op->heap) ;
		op->buffer = (Record*)Heap_Items (op->heap) ;
	}

	sort_r (op->buffer, op->buffer_len, sizeof(Record),
			(heap_cmp)_buffer_elem_cmp, op) ;

	// pass ordered records downward
	return _handoff (op) ;
}

// restart iterator
static OpResult SortReset
(
	OpBase *ctx
) {
	OpSort *op = (OpSort *)ctx ;

	// heap yet to be dumped into buffer
	if (op->buffer == NULL) {
		op->buffer  = (Record*)Heap_Items (op->heap) ;
		op->buffer_len = Heap_count (op->heap) ;
		op->record_idx = 0 ;
	}

	// clear accumulated records
	for (uint i = op->record_idx ; i < op->buffer_len ; i++) {
		Record r = op->buffer[i] ;
		OpBase_DeleteRecord (&r) ;
	}

	if (op->heap != NULL) {
		Heap_clear (op->heap) ;
		op->buffer = NULL ;
	} else {
		array_clear (op->buffer) ;
	}

	op->first      = true ;
	op->buffer_len = 0 ;
	op->record_idx = 0;

	return OP_OK ;
}

static OpBase *SortClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (opBase->type == OPType_SORT) ;

	OpSort *op = (OpSort *)opBase ;
	OpSort *clone = rm_calloc (1, sizeof (OpSort)) ;

	clone->first = true ;
	clone->limit = UNLIMITED ;

	array_clone (clone->directions, op->directions) ;
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

	// heap yet to be dumped into buffer
	if (op->buffer == NULL && op->heap != NULL) {
		op->buffer  = (Record*)Heap_Items (op->heap) ;
		op->buffer_len = Heap_count (op->heap) ;
		op->record_idx = 0 ;
	}

	// clear accumulated records
	for (uint i = op->record_idx ; i < op->buffer_len ; i++) {
		Record r = op->buffer[i] ;
		OpBase_DeleteRecord (&r) ;
	}

	if (op->heap != NULL) {
		Heap_free (op->heap) ;
		op->heap = NULL ;
	} else {
		array_free (op->buffer) ;
		op->buffer = NULL ;
	}

	if (op->sort_offsets) {
		array_free (op->sort_offsets) ;
		op->sort_offsets = NULL ;
	}

	if (op->record_offsets) {
		array_free (op->record_offsets) ;
		op->record_offsets = NULL ;
	}

	if (op->directions) {
		array_free (op->directions) ;
		op->directions = NULL ;
	}

	if (op->to_eval) {
		uint n = array_len (op->to_eval) ;
		for (uint i = 0; i < n; i++) {
			AR_EXP_Free (op->to_eval[i]) ;
		}
		array_free (op->to_eval) ;
		op->to_eval = NULL ;
	}

	if (op->exps) {
		uint n = array_len (op->exps) ;
		for (uint i = 0; i < n; i++) {
			AR_EXP_Free (op->exps[i]) ;
		}
		array_free (op->exps) ;
		op->exps = NULL ;
	}
}

