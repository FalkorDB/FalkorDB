/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../query_ctx.h"
#include "op_node_by_id_seek.h"
#include "shared/print_functions.h"

// forward declarations
static OpResult NodeByIdSeekInit(OpBase *opBase);
static Record NodeByIdSeekConsume(OpBase *opBase);
static Record NodeByIdSeekDepleted(OpBase *opBase);
static Record NodeByIdSeekConsumeFromChild(OpBase *opBase);
static OpResult NodeByIdSeekReset(OpBase *opBase);
static OpBase *NodeByIdSeekClone(const ExecutionPlan *plan, const OpBase *opBase);
static void NodeByIdSeekFree(OpBase *opBase);

static inline void NodeByIdSeekToString
(
	const OpBase *ctx,
	sds *buf
) {
	ScanToString(ctx, buf, ((NodeByIdSeek *)ctx)->alias, NULL);
}

// create a new NodeByIdSeek operation
OpBase *NewNodeByIdSeekOp
(
	const ExecutionPlan *plan,  // execution plan
	const char *alias,          // node alias
	RangeExpression *ranges     // ID range expressions
) {
	NodeByIdSeek *op = rm_malloc(sizeof(NodeByIdSeek));

	op->g = QueryCtx_GetGraph();
	op->it           = NULL;
	op->ids          = NULL;
	op->alias        = alias;
	op->ranges       = ranges;
	op->child_record = NULL;

	OpBase_Init((OpBase *)op, OPType_NODE_BY_ID_SEEK, "NodeByIdSeek",
			NodeByIdSeekInit, NodeByIdSeekConsume, NodeByIdSeekReset,
			NodeByIdSeekToString, NodeByIdSeekClone, NodeByIdSeekFree, false,
			plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, alias);

	return (OpBase *)op;
}

static OpResult NodeByIdSeekInit
(
	OpBase *opBase
) {
	ASSERT(opBase->type == OPType_NODE_BY_ID_SEEK);

	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	// create empty ID range
	op->ids = roaring64_bitmap_create();
	op->it  = roaring64_iterator_create(op->ids);

	// operation is not a tap
	// update consume function
	if(opBase->childCount > 0) {
		OpBase_UpdateConsume(opBase, NodeByIdSeekConsumeFromChild);
		return OP_OK;
	}

	// operation is a tap
	// evaluate ID ranges
	if(!BitmapRange_FromRanges(op->ranges, op->ids, op->child_record, 0,
				Graph_UncompactedNodeCount(op->g))) {
		// failed to tighten range, update consume function to return NULL
		OpBase_UpdateConsume(opBase, NodeByIdSeekDepleted);
		return OP_OK;
	}

	// ID range set, reattach iterator
	roaring64_iterator_reinit(op->ids, op->it);

	return OP_OK;
}

// get the next node
// returns true on success, false when ID iterator deplete
static inline bool _SeekNextNode
(
	NodeByIdSeek *op,
	Node *n
) {
	ASSERT(n  != NULL);
	ASSERT(op != NULL);

	// as long as the ID iterator isn't depleted
	while(roaring64_iterator_has_value(op->it)) {
		// get current ID
		NodeID id = roaring64_iterator_value(op->it);

		// advance iterator
		roaring64_iterator_advance(op->it);

		// try to get node from graph
		if(Graph_GetNode(op->g, id, n)) {
			return true;
		}
	}

	// iterator depleted
	return false;
}

static Record NodeByIdSeekConsumeFromChild
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	// try to get a new record from child op
pull:
	if(op->child_record == NULL) {
		op->child_record = OpBase_Consume(OpBase_GetChild(opBase, 0));
		if(op->child_record == NULL) return NULL;  // child depleted

		// re-evealuate ID ranges
		if(!BitmapRange_FromRanges(op->ranges, op->ids, op->child_record, 0,
					Graph_UncompactedNodeCount(op->g))) {
			return NULL;
		}

		roaring64_iterator_reinit(op->ids, op->it);
	}

	ASSERT(op->child_record != NULL);

	Node n;
	if(!_SeekNextNode(op, &n)) { // failed to retrieve a node
		OpBase_DeleteRecord(&op->child_record); // free old record

		// try to pull a new record
		goto pull;
	}

	// clone the held Record, as it will be freed upstream
	Record r = OpBase_CloneRecord(op->child_record);

	// populate the Record with the actual node
	Record_AddNode(r, op->nodeRecIdx, n);

	return r;
}

static Record NodeByIdSeekConsume
(
	OpBase *opBase
) {
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	ASSERT(op      != NULL);
	ASSERT(op->it  != NULL);
	ASSERT(op->ids != NULL);

	Node n;
	if(!_SeekNextNode(op, &n)) return NULL; // failed to retrieve a node

	// create a new Record
	Record r = OpBase_CreateRecord(opBase);

	// populate the Record with the actual node
	Record_AddNode(r, op->nodeRecIdx, n);

	return r;
}

// depleted consume function
// returns NULL
static Record NodeByIdSeekDepleted
(
	OpBase *opBase
) {
	return NULL;
}

static OpResult NodeByIdSeekReset
(
	OpBase *ctx
) {
	NodeByIdSeek *op = (NodeByIdSeek *)ctx;

	if(op->it && op->child_record == 0) {
		// operation is a tap
		roaring64_iterator_reinit(op->ids, op->it);
	} else {
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(&op->child_record);
		}
	}

	return OP_OK;
}

static OpBase *NodeByIdSeekClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_NODE_BY_ID_SEEK);

	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	RangeExpression *ranges;
	array_clone_with_cb(ranges, op->ranges, RangeExpression_Clone);

	return NewNodeByIdSeekOp(plan, op->alias, ranges);
}

static void NodeByIdSeekFree
(
	OpBase *opBase
) {
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	if(op->child_record) {
		OpBase_DeleteRecord(&op->child_record);
	}

	if(op->ranges) {
		for(int i = 0; i < array_len(op->ranges); i++) {
			RangeExpression_Free(op->ranges + i);
		}
		array_free(op->ranges);
		op->ranges = NULL;
	}

	if(op->it) {
		roaring64_iterator_free(op->it);
		op->it = NULL;
	}

	if(op->ids) {
		roaring64_bitmap_free(op->ids);
		op->ids = NULL;
	}
}

