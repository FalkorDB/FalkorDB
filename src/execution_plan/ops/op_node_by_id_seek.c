/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_node_by_id_seek.h"
#include "RG.h"
#include "shared/print_functions.h"
#include "../../query_ctx.h"

/* Forward declarations. */
static OpResult NodeByIdSeekInit(OpBase *opBase);
static Record NodeByIdSeekConsume(OpBase *opBase);
static Record NodeByIdSeekNoOpConsume(OpBase *opBase);
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

OpBase *NewNodeByIdSeekOp
(
	const ExecutionPlan *plan,
	const char *alias,
	FilterExpression *filters
) {

	NodeByIdSeek *op = rm_malloc(sizeof(NodeByIdSeek));
	op->g = QueryCtx_GetGraph();
	op->it           = NULL;
	op->ids          = NULL;
	op->alias        = alias;
	op->filters      = filters;
	op->child_record = NULL;

	OpBase_Init((OpBase *)op, OPType_NODE_BY_ID_SEEK, "NodeByIdSeek", NodeByIdSeekInit,
				NodeByIdSeekConsume, NodeByIdSeekReset, NodeByIdSeekToString, NodeByIdSeekClone, NodeByIdSeekFree,
				false, plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, alias);

	return (OpBase *)op;
}

static OpResult NodeByIdSeekInit
(
	OpBase *opBase
) {
	ASSERT(opBase->type == OPType_NODE_BY_ID_SEEK);
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;
	op->ids = roaring64_bitmap_create();
	if(opBase->childCount > 0) {
		OpBase_UpdateConsume(opBase, NodeByIdSeekConsumeFromChild);
	} else {
		if(!FilterExpression_Resolve(op->g, op->filters, op->ids, op->child_record)) {
			OpBase_UpdateConsume(opBase, NodeByIdSeekNoOpConsume);
			return OP_OK;
		}
		op->it = roaring64_iterator_create(op->ids);
	}
	return OP_OK;
}

static inline bool _SeekNextNode
(
	NodeByIdSeek *op,
	Node *n
) {
	while(roaring64_iterator_has_value(op->it)) {
		NodeID id = roaring64_iterator_value(op->it);
		roaring64_iterator_advance(op->it);
		if(Graph_GetNode(op->g, id, n)) {
			return true;
		}
	}

	return false;
}

static Record NodeByIdSeekConsumeFromChild
(
	OpBase *opBase
) {
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	if(op->child_record == NULL) {
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) return NULL;
		if(!FilterExpression_Resolve(op->g, op->filters, op->ids, op->child_record)) {
			return NULL;
		}
		if(op->it == NULL) {
			op->it = roaring64_iterator_create(op->ids);
		} else {
			roaring64_iterator_reinit(op->ids, op->it);
		}
	}

	Node n;

	if(!_SeekNextNode(op, &n)) { // Failed to retrieve a node.
		OpBase_DeleteRecord(op->child_record); // Free old record.
		// Pull a new record from child.
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) return NULL; // Child depleted.

		// Reset iterator and evaluate again.
		if(!FilterExpression_Resolve(op->g, op->filters, op->ids, op->child_record)) {
			return NULL;
		}
		if(op->it == NULL) {
			op->it = roaring64_iterator_create(op->ids);
		} else {
			roaring64_iterator_reinit(op->ids, op->it);
		}
		if(!_SeekNextNode(op, &n)) return NULL; // Empty iterator; return immediately.
	}

	// Clone the held Record, as it will be freed upstream.
	Record r = OpBase_DeepCloneRecord(op->child_record);

	// Populate the Record with the actual node.
	Record_AddNode(r, op->nodeRecIdx, n);

	return r;
}

static Record NodeByIdSeekConsume
(
	OpBase *opBase
) {
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;

	Node n;
	if(!_SeekNextNode(op, &n)) return NULL; // Failed to retrieve a node.

	// Create a new Record.
	Record r = OpBase_CreateRecord(opBase);

	// Populate the Record with the actual node.
	Record_AddNode(r, op->nodeRecIdx, n);

	return r;
}

static Record NodeByIdSeekNoOpConsume
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
	if(op->it) {
		roaring64_iterator_reinit(op->ids, op->it);
	}
	return OP_OK;
}

static FilterExpression _cloneFilterExpression
(
	FilterExpression filter
) {
	return (FilterExpression){.op = filter.op, .id_exp = AR_EXP_Clone(filter.id_exp)};
}

static OpBase *NodeByIdSeekClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_NODE_BY_ID_SEEK);
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;
	FilterExpression *filters;
	array_clone_with_cb(filters, op->filters,_cloneFilterExpression);
	return NewNodeByIdSeekOp(plan, op->alias, filters);
}

static void NodeByIdSeekFree
(
	OpBase *opBase
) {
	NodeByIdSeek *op = (NodeByIdSeek *)opBase;
	if(op->child_record) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;
	}

	if(op->filters) {
		for(int i = 0; i < array_len(op->filters); i++) {
			AR_EXP_Free(op->filters[i].id_exp);
		}
		array_free(op->filters);
		op->filters = NULL;
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

