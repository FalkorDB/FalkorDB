/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_cartesian_product.h"
#include "RG.h"

// forward declarations
static OpResult CartesianProductInit(OpBase *opBase);
static Record CartesianProductConsume(OpBase *opBase);
static OpResult CartesianProductReset(OpBase *opBase);
static OpBase *CartesianProductClone(const ExecutionPlan *plan, const OpBase *opBase);
static void CartesianProductFree(OpBase *opBase);

OpBase *NewCartesianProductOp
(
	const ExecutionPlan *plan
) {
	CartesianProduct *op = rm_calloc (1, sizeof(CartesianProduct)) ;

	op->init = true;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_CARTESIAN_PRODUCT, "Cartesian Product",
			CartesianProductInit, CartesianProductConsume,
			CartesianProductReset, NULL, CartesianProductClone,
			CartesianProductFree, false, plan);
	return (OpBase *)op;
}

static OpResult CartesianProductInit
(
	OpBase *opBase
) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	op->records = rm_calloc(OpBase_ChildCount(opBase), sizeof(Record));
	return OP_OK;
}

// reset streams [0..streamIdx)
static void _ResetStreams
(
	CartesianProduct *cp,  // operation
	int streamIdx          // reset first 'streamIdx' streams
) {
	// reset each child stream
	for(int i = 0; i < streamIdx; i++) {
		OpBase_DeleteRecord(cp->records+i);
		OpBase_PropagateReset(cp->op.children[i]);
	}
}

static int _PullFromStreams
(
	CartesianProduct *op
) {
	for(int i = 1; i < op->op.childCount; i++) {
		OpBase *child = op->op.children[i];
		Record childRecord = OpBase_Consume(child);

		if(childRecord) {
			// free previous record
			OpBase_DeleteRecord(op->records+i);

			// set new record
			op->records[i] = childRecord;

			// managed to get new data
			// reset streams [0-i]
			_ResetStreams(op, i);

			// pull from resetted streams
			for(int j = 0; j < i; j++) {
				child = op->op.children[j];
				childRecord = OpBase_Consume(child);
				if(childRecord) {
					op->records[j] = childRecord;
				} else {
					return 0;
				}
			}
			// ready to continue
			return 1;
		}
	}

	// if we're here, then we didn't manged to get new data
	// last stream depleted
	return 0;
}

// generates a new record by combining streams
static Record _YieldRecord
(
	OpBase *opBase  // operation
) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	Record r = OpBase_CreateRecord(opBase);

	// clone entries from each stream into 'r'
	for(int i = 0; i < OpBase_ChildCount(opBase); i++) {
		Record_DuplicateEntries(r, op->records[i]);
	}

	return r;
}

static Record CartesianProductConsume
(
	OpBase *opBase
) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	OpBase *child;
	Record childRecord;

	// first call, pull from every stream
	if(op->init) {
		op->init = false;

		// pull from every stream
		for(int i = 0; i < op->op.childCount; i++) {
			child = op->op.children[i];
			childRecord = OpBase_Consume(child);

			// no data, depleted
			if(!childRecord) return NULL;

			// save record
			op->records[i] = childRecord;
		}

		return _YieldRecord(opBase);
	}

	// pull from first stream
	child = op->op.children[0];
	childRecord = OpBase_Consume(child);

	if(childRecord) {
		// update stream record
		OpBase_DeleteRecord(op->records);
		op->records[0] = childRecord;
	} else {
		// failed to get data from first stream
		// try pulling other streams for data
		if(!_PullFromStreams(op)) return NULL;
	}

	return _YieldRecord(opBase);
}

// free each cached stream record
static void _FreeStreamRecords
(
	OpBase *opBase
) {
	CartesianProduct *op = (CartesianProduct *)opBase;

	for(int i = 0; i < OpBase_ChildCount(opBase); i++) {
		if(op->records[i] != NULL) {
			OpBase_DeleteRecord(op->records+i);
		}
	}
}

static OpResult CartesianProductReset
(
	OpBase *opBase
) {
	CartesianProduct *op = (CartesianProduct *)opBase;

	op->init = true;
	if(op->records != NULL) {
		_FreeStreamRecords(opBase);
	}

	return OP_OK;
}

static OpBase *CartesianProductClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_CARTESIAN_PRODUCT);
	return NewCartesianProductOp(plan);
}

static void CartesianProductFree
(
	OpBase *opBase
) {
	CartesianProduct *op = (CartesianProduct *)opBase;

	if(op->records != NULL) {
		_FreeStreamRecords(opBase);
		rm_free(op->records);
	}
}

