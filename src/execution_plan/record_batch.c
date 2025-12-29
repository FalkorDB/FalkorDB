/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "record_batch.h"
#include "../util/arr.h"
#include "execution_plan.h"

// create a new RecordBatch
RecordBatch RecordBatch_New
(
	uint16_t cap  // batch capacity
) {
	return array_newlen (Record, cap) ;
}

// return number of records in batch
uint16_t RecordBatch_Size
(
	const RecordBatch batch  // record batch
) {
	ASSERT (batch != NULL) ;

	return array_len (batch) ;
}

// return batch capacity
uint16_t RecordBatch_Capacity
(
	const RecordBatch batch  // record batch
) {
	ASSERT (batch != NULL) ;
	return array_cap (batch) ;
}

// add a record to batch
void RecordBatch_AddRecord
(
	RecordBatch *batch,  // batch to extend
	Record r             // record to add
);

// remove record at position idx from the batch
void RecordBatch_RemoveRecord
(
	RecordBatch batch,  // batch to update
	uint16_t idx        // record position to remove
) {
	ASSERT (batch != NULL) ;

	uint32_t batch_size = RecordBatch_Size (batch) ;
	ASSERT (idx < batch_size) ;

	Record r = batch[idx] ;
	ExecutionPlan_ReturnRecord (r->owner, r) ;

	array_del_fast (batch, idx) ;
}

// remove the last n records from the batch
void RecordBatch_RemoveRecords
(
	RecordBatch batch,  // records batch
	uint32_t n          // number of records to remove
) {
	ASSERT (n     >  0) ;
	ASSERT (batch != NULL) ;

	uint32_t i = RecordBatch_Size (batch) ;
	ASSERT (n <= i) ;

	while (i > 0) {
		RecordBatch_RemoveRecord (batch, i-1) ;
		i-- ;
	}
}

// merge two batchs
// returns a merged batch
RecordBatch RecordBatch_Merge
(
	RecordBatch *A,  // first batch
	RecordBatch *B   // second batch
);

// free RecordBatch
void RecordBatch_Free
(
	RecordBatch *batch  // batch to free
) {
	ASSERT (batch != NULL && *batch != NULL) ;

	RecordBatch _batch = *batch ;

	uint32_t batch_size = RecordBatch_Size (_batch) ;
	for (uint32_t i = 0; i < batch_size; i++) {
		Record r = _batch[i] ;
		ExecutionPlan_ReturnRecord (r->owner, r) ;
	}

	array_free (_batch) ;
	*batch = NULL ;
}

