/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "op_cartesian_product.h"
#include "RG.h"

/* Forward declarations. */
static OpResult CartesianProductInit(OpBase *opBase);
static Record CartesianProductConsume(OpBase *opBase);
static OpResult CartesianProductReset(OpBase *opBase);
static OpBase *CartesianProductClone(const ExecutionPlan *plan, const OpBase *opBase);
static void CartesianProductFree(OpBase *opBase);

OpBase *NewCartesianProductOp(const ExecutionPlan *plan) {
	CartesianProduct *op = rm_malloc(sizeof(CartesianProduct));
	op->init = true;
	op->r = NULL;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_CARTESIAN_PRODUCT, "Cartesian Product", CartesianProductInit,
				CartesianProductConsume, CartesianProductReset, NULL, CartesianProductClone, CartesianProductFree,
				false, plan);
	return (OpBase *)op;
}

static void _ResetStreams(CartesianProduct *cp, int streamIdx) {
	// Reset each child stream, Reset propagates upwards.
	for(int i = 0; i < streamIdx; i++) OpBase_PropagateReset(cp->op.children[i]);
}

static int _PullFromStreams(CartesianProduct *op) {
	for(int i = 1; i < op->op.childCount; i++) {
		OpBase *child = op->op.children[i];
		Record childRecord = OpBase_Consume(child);

		if(childRecord) {
			Record_TransferEntries(&op->r, childRecord, true);
			OpBase_DeleteRecord(childRecord);
			/* Managed to get new data
			 * Reset streams [0-i] */
			_ResetStreams(op, i);

			// Pull from resetted streams.
			for(int j = 0; j < i; j++) {
				child = op->op.children[j];
				childRecord = OpBase_Consume(child);
				if(childRecord) {
					Record_TransferEntries(&op->r, childRecord, true);
					OpBase_DeleteRecord(childRecord);
				} else {
					return 0;
				}
			}
			// Ready to continue.
			return 1;
		}
	}

	/* If we're here, then we didn't manged to get new data.
	 * Last stream depleted. */
	return 0;
}

static OpResult CartesianProductInit(OpBase *opBase) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	op->r = OpBase_CreateRecord((OpBase *)op);
	return OP_OK;
}

static Record CartesianProductConsume(OpBase *opBase) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	OpBase *child;
	Record childRecord;

	if(op->init) {
		op->init = false;

		for(int i = 0; i < op->op.childCount; i++) {
			child = op->op.children[i];
			childRecord = OpBase_Consume(child);
			if(!childRecord) return NULL;
			Record_TransferEntries(&op->r, childRecord, true);
			OpBase_DeleteRecord(childRecord);
		}
		return OpBase_CloneRecord(op->r);
	}

	// Pull from first stream.
	child = op->op.children[0];
	childRecord = OpBase_Consume(child);

	if(childRecord) {
		// Managed to get data from first stream.
		Record_TransferEntries(&op->r, childRecord, true);
		OpBase_DeleteRecord(childRecord);
	} else {
		// Failed to get data from first stream,
		// try pulling other streams for data.
		if(!_PullFromStreams(op)) return NULL;
	}

	// Pass down a clone of record.
	return OpBase_CloneRecord(op->r);
}

static OpResult CartesianProductReset(OpBase *opBase) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	op->init = true;
	return OP_OK;
}

static OpBase *CartesianProductClone(const ExecutionPlan *plan, const OpBase *opBase) {
	ASSERT(opBase->type == OPType_CARTESIAN_PRODUCT);
	return NewCartesianProductOp(plan);
}

static void CartesianProductFree(OpBase *opBase) {
	CartesianProduct *op = (CartesianProduct *)opBase;
	if(op->r) {
		OpBase_DeleteRecord(op->r);
		op->r = NULL;
	}
}

