/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_node_by_label_scan.h"
#include "RG.h"
#include "shared/print_functions.h"
#include "../../ast/ast.h"
#include "../../query_ctx.h"

/* Forward declarations. */
static OpResult NodeByLabelScanInit(OpBase *opBase);
static Record NodeByLabelScanConsume(OpBase *opBase);
static Record NodeByLabelAndIDScanConsume(OpBase *opBase);
static Record NodeByLabelScanConsumeFromChild(OpBase *opBase);
static Record NodeByLabelAndIDScanConsumeFromChild(OpBase *opBase);
static Record NodeByLabelScanNoOp(OpBase *opBase);
static OpResult NodeByLabelScanReset(OpBase *opBase);
static OpBase *NodeByLabelScanClone(const ExecutionPlan *plan, const OpBase *opBase);
static void NodeByLabelScanFree(OpBase *opBase);

static inline void NodeByLabelScanToString
(
	const OpBase *ctx,
	sds *buf
) {
	NodeByLabelScan *op = (NodeByLabelScan *)ctx;
	ScanToString(ctx, buf, op->n->alias, op->n->label);
}

// update the label-id of a cached operation, as it may have not 
// been known when the plan was prepared.
static void _update_label_id
(
	NodeByLabelScan *op
) {
	if(op->n->label_id != GRAPH_UNKNOWN_LABEL) return;

	GraphContext *gc = QueryCtx_GetGraphCtx();
	Schema *s = GraphContext_GetSchema(gc, op->n->label, SCHEMA_NODE);
	if(s != NULL) op->n->label_id = Schema_GetID(s);
}

OpBase *NewNodeByLabelScanOp
(
	const ExecutionPlan *plan,
	NodeScanCtx *n
) {
	NodeByLabelScan *op = rm_calloc(sizeof(NodeByLabelScan), 1);
	op->g       = QueryCtx_GetGraph();
	op->n       = n;
	op->it      = NULL;
	op->ids     = NULL;
	op->filters = NULL;
	_update_label_id(op);

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_NODE_BY_LABEL_SCAN, "Node By Label Scan", NodeByLabelScanInit,
				NodeByLabelScanConsume, NodeByLabelScanReset, NodeByLabelScanToString, NodeByLabelScanClone,
				NodeByLabelScanFree, false, plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, n->alias);

	return (OpBase *)op;
}

void NodeByLabelScanOp_SetFilterID
(
	NodeByLabelScan *op,
	FilterExpression *filters
) {
	op->filters = filters;

	op->op.type = OPType_NODE_BY_LABEL_AND_ID_SCAN;
	op->op.name = "Node By Label and ID Scan";
}

static GrB_Info _ConstructIterator
(
	NodeByLabelScan *op
) {
	GrB_Info  info;
	NodeID    minId;
	NodeID    maxId;

	op->L = Graph_GetLabelMatrix(op->g, op->n->label_id);

	if(array_len(op->filters) > 0) {
		if(!FilterExpression_Resolve(op->g, op->filters, op->ids, op->child_record)) {
			return GrB_INVALID_INDEX;
		}

		if(roaring64_bitmap_get_cardinality(op->ids) == 0) {
			return GrB_DIMENSION_MISMATCH;
		}

		if(op->it) {
			roaring64_iterator_free(op->it);	
		}
		op->it = roaring64_iterator_create(op->ids);

		return GrB_SUCCESS;
	}

	info = RG_MatrixTupleIter_attach(&op->iter, op->L);
	ASSERT(info == GrB_SUCCESS);

	return info;
}

static OpResult NodeByLabelScanInit
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	bool has_filters = array_len(op->filters) > 0;

	OpBase_UpdateConsume(opBase, has_filters ? NodeByLabelAndIDScanConsume : NodeByLabelScanConsume); // default consume function

	if(has_filters) {
		op->ids = roaring64_bitmap_create();
	}

	// operation has children, consume from child
	if(opBase->childCount > 0) {
		OpBase_UpdateConsume(opBase, has_filters 
			? NodeByLabelAndIDScanConsumeFromChild
			: NodeByLabelScanConsumeFromChild);
		return OP_OK;
	}

	if(op->n->label_id == GRAPH_UNKNOWN_LABEL) {
		// missing schema, use the NOP consume function
		OpBase_UpdateConsume(opBase, NodeByLabelScanNoOp);
		return OP_OK;
	}

	// the iterator build may fail if the ID range does not match the matrix dimensions
	GrB_Info iterator_built = _ConstructIterator(op);
	if(iterator_built != GrB_SUCCESS) {
		// invalid range, use the NOP consume function
		OpBase_UpdateConsume(opBase, NodeByLabelScanNoOp);
		return OP_OK;
	}

	return OP_OK;
}

static inline void _UpdateRecord
(
	NodeByLabelScan *op,
	Record r,
	GrB_Index node_id
) {
	// Populate the Record with the graph entity data.
	Node n = GE_NEW_NODE();
	Graph_GetNode(op->g, node_id, &n);
	Record_AddNode(r, op->nodeRecIdx, n);
}

static Record NodeByLabelScanConsumeFromChild
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	// try to get new nodeID
	GrB_Index id;
	GrB_Info info = RG_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);
	while(info == GrB_NULL_POINTER || op->child_record == NULL || info == GxB_EXHAUSTED) {
		// try to get a new record
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(op->child_record);
		}

		op->child_record = OpBase_Consume(op->op.children[0]);

		if(op->child_record == NULL) {
			// depleted
			return NULL;
		}

		// got a record
		if(info == GrB_NULL_POINTER) {
			_update_label_id(op);
			if(_ConstructIterator(op) != GrB_SUCCESS) {
				continue;
			}
		} else {
			// iterator depleted - reset
			_ConstructIterator(op);
		}

		// try to get new NodeID
		info = RG_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);
	}

	// we've got a record and NodeID
	// clone the held Record, as it will be freed upstream
	Record r = OpBase_DeepCloneRecord(op->child_record);

	// populate the Record with the actual node
	_UpdateRecord(op, r, id);
	return r;
}

static Record NodeByLabelAndIDScanConsumeFromChild
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	// try to get new nodeID
	GrB_Info info;
	GrB_Index id;
	if(op->it == NULL) {
		info = GrB_NULL_POINTER;
	} else {
		bool is_valid = false;
		while(roaring64_iterator_has_value(op->it)) {
			id = roaring64_iterator_value(op->it);
			roaring64_iterator_advance(op->it);
			bool x;
			if(RG_Matrix_extractElement_BOOL(&x, op->L, id, id) == GrB_SUCCESS) {
				is_valid = true;
				break;
			}
		}
		info = is_valid ? GrB_SUCCESS : GxB_EXHAUSTED;
	}
	while(info == GrB_NULL_POINTER || op->child_record == NULL || info == GxB_EXHAUSTED) {
		// try to get a new record
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(op->child_record);
		}

		op->child_record = OpBase_Consume(op->op.children[0]);

		if(op->child_record == NULL) {
			// depleted
			return NULL;
		}

		// got a record
		if(info == GrB_NULL_POINTER) {
			_update_label_id(op);
			if(_ConstructIterator(op) != GrB_SUCCESS) {
				continue;
			}
		} else {
			// iterator depleted - reset
			_ConstructIterator(op);
		}

		// try to get new NodeID
		bool is_valid = false;
		while(roaring64_iterator_has_value(op->it)) {
			id = roaring64_iterator_value(op->it);
			roaring64_iterator_advance(op->it);
			bool x;
			if(RG_Matrix_extractElement_BOOL(&x, op->L, id, id) == GrB_SUCCESS) {
				is_valid = true;
				break;
			}
		}
		info = is_valid ? GrB_SUCCESS : GxB_EXHAUSTED;
	}

	// we've got a record and NodeID
	// clone the held Record, as it will be freed upstream
	Record r = OpBase_DeepCloneRecord(op->child_record);

	// populate the Record with the actual node
	_UpdateRecord(op, r, id);
	return r;
}

static Record NodeByLabelScanConsume
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	GrB_Index id;
	GrB_Info info = RG_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);
	if(info == GxB_EXHAUSTED) return NULL;

	ASSERT(info == GrB_SUCCESS);

	Record r = OpBase_CreateRecord((OpBase *)op);

	// Populate the Record with the actual node.
	_UpdateRecord(op, r, id);

	return r;
}

static Record NodeByLabelAndIDScanConsume
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	GrB_Index id;
	bool is_valid = false;
	while(roaring64_iterator_has_value(op->it)) {
		id = roaring64_iterator_value(op->it);
		roaring64_iterator_advance(op->it);
		bool x;
		if(RG_Matrix_extractElement_BOOL(&x, op->L, id, id) == GrB_SUCCESS) {
			is_valid = true;
			break;
		}
	}
	if(!is_valid) {
		return NULL;
	}

	Record r = OpBase_CreateRecord((OpBase *)op);

	// Populate the Record with the actual node.
	_UpdateRecord(op, r, id);

	return r;
}

// this function is invoked when the op has no children
// and no valid label is requested (either no label, or non existing label)
// the op simply needs to return NULL
static Record NodeByLabelScanNoOp
(
	OpBase *opBase
) {
	return NULL;
}

static OpResult NodeByLabelScanReset
(
	OpBase *ctx
) {
	NodeByLabelScan *op = (NodeByLabelScan *)ctx;

	if(op->child_record) {
		OpBase_DeleteRecord(op->child_record); // Free old record.
		op->child_record = NULL;
	}

	_ConstructIterator(op);
	return OP_OK;
}

static OpBase *NodeByLabelScanClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_NODE_BY_LABEL_SCAN);
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;
	return NewNodeByLabelScanOp(plan, NodeScanCtx_Clone(op->n));
}

static void NodeByLabelScanFree
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	GrB_Info info = RG_MatrixTupleIter_detach(&(op->iter));
	ASSERT(info == GrB_SUCCESS);

	if(op->child_record) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;
	}

	if(op->n != NULL) {
		NodeScanCtx_Free(op->n);
		op->n = NULL;
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
