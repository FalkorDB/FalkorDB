/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "record.h"

// alias an array of records as RecordBatch
typedef Record *RecordBatch ;

// create a new RecordBatch
RecordBatch RecordBatch_New
(
	uint16_t cap  // batch capacity
);

// return number of records in batch
uint16_t RecordBatch_Size
(
	const RecordBatch batch  // record batch
);

// return batch capacity
uint16_t RecordBatch_Capacity
(
	const RecordBatch batch  // record batch
);

// update batch size
void RecordBatch_SetSize
(
	RecordBatch batch,  // batch
	uint16_t n          // new batch size
);

// add a record to batch
void RecordBatch_AddRecord
(
	RecordBatch *batch,  // batch to extend
	Record r             // record to add
);

// delete record at position idx from the batch
void RecordBatch_DeleteRecord
(
	RecordBatch batch,  // batch to update
	uint16_t idx        // record position to remove
);

// delete the first N records
// maintain records order within the batch
void RecordBatch_DeleteFirstN
(
	RecordBatch batch,  // batch
	uint16_t n          // number of records to delete
) ;

void RecordBatch_RemoveRecord
(
	RecordBatch batch,  // batch to update
	uint16_t idx        // record position
);

// deletes the last n records from the batch
void RecordBatch_DeleteRecords
(
	RecordBatch batch,  // records batch
	uint32_t n          // number of records to remove
);

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
);

