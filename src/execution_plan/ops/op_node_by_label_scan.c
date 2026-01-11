/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "op_node_by_label_scan.h"
#include "shared/print_functions.h"
#include "../../ast/ast.h"
#include "../../query_ctx.h"

// forward declarations
static OpResult NodeByLabelScanInit(OpBase *opBase);
static Record NodeByLabelScanConsume(OpBase *opBase);
static Record NodeByLabelScanConsumeFromChild(OpBase *opBase);
static Record NodeByLabelAndIDScanConsume(OpBase *opBase);
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
// been known when the plan was prepared
static void _update_label_id
(
	NodeByLabelScan *op
) {
	if (op->n->label_id != GRAPH_UNKNOWN_LABEL) {
		return;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx();
	Schema *s = GraphContext_GetSchema(gc, op->n->label, SCHEMA_NODE);

	if (s != NULL) {
		op->n->label_id = Schema_GetID(s);
	}
}

OpBase *NewNodeByLabelScanOp
(
	const ExecutionPlan *plan,
	NodeScanCtx *n
) {
	NodeByLabelScan *op = rm_calloc (1, sizeof(NodeByLabelScan)) ;

	op->g = QueryCtx_GetGraph();
	op->n = n;

	_update_label_id(op);

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_NODE_BY_LABEL_SCAN, "Node By Label Scan",
			NodeByLabelScanInit, NodeByLabelScanConsume, NodeByLabelScanReset,
			NodeByLabelScanToString, NodeByLabelScanClone, NodeByLabelScanFree,
			false, plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, n->alias);

	return (OpBase *)op;
}

void NodeByLabelScanOp_SetIDRange
(
	NodeByLabelScan *op,
	RangeExpression *ranges  // ID range expressions
) {
	ASSERT(op         != NULL);
	ASSERT(ranges     != NULL);
	ASSERT(op->ranges == NULL);

	op->ranges  = ranges;
	op->op.type = OPType_NODE_BY_LABEL_AND_ID_SCAN;
	op->op.name = "Node By Label and ID Scan";

	// initialize IDs bitmap
	ASSERT(op->ids   == NULL);
	ASSERT(op->ID_it == NULL);

	op->ids   = roaring64_bitmap_create();
	op->ID_it = roaring64_iterator_create(op->ids);
}

// constructs either a range iterator or a matrix iterator
// depending on rather or not an ID range is provided
static bool _ConstructIterator
(
	NodeByLabelScan *op
) {
	GrB_Info info;
	NodeID   minId;
	NodeID   maxId;

	op->L = Graph_GetLabelMatrix(op->g, op->n->label_id);

	bool has_ranges = array_len(op->ranges) > 0;
	if(has_ranges) {
		// use range iterator
		if (!BitmapRange_FromRanges (op->ranges, &op->ids, op->child_record, 0,
				Graph_UncompactedNodeCount(op->g))) {
			return false;
		}

		if(roaring64_bitmap_get_cardinality(op->ids) == 0) {
			return false;
		}

		roaring64_iterator_reinit(op->ids, op->ID_it);

		return true;
	}

	// use matrix iterator
	info = Delta_MatrixTupleIter_attach(&op->iter, op->L);
	ASSERT(info == GrB_SUCCESS);

	return true;
}

static OpResult NodeByLabelScanInit
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	bool has_ranges = array_len(op->ranges) > 0;

	OpBase_UpdateConsume(opBase, has_ranges
		? NodeByLabelAndIDScanConsume 
		: NodeByLabelScanConsume); // default consume function

	// operation has children, consume from child
	if(OpBase_ChildCount(opBase) > 0) {
		OpBase_UpdateConsume(opBase, has_ranges 
			? NodeByLabelAndIDScanConsumeFromChild
			: NodeByLabelScanConsumeFromChild);
		return OP_OK;
	}

	if(op->n->label_id == GRAPH_UNKNOWN_LABEL) {
		// missing schema, use the NOP consume function
		OpBase_UpdateConsume(opBase, NodeByLabelScanNoOp);
		return OP_OK;
	}

	// iterator build may fail if ID range does not match the matrix dimensions
	if(!_ConstructIterator(op)) {
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
	// populate the Record with the graph entity data
	Node n = GE_NEW_NODE();
	Graph_GetNode(op->g, node_id, &n);
	Record_AddNode(r, op->nodeRecIdx, n);
}

//------------------------------------------------------------------------------
// consume functions
//------------------------------------------------------------------------------

// NOP consume function
// and no valid label is requested (either no label, or non existing label)
// or specified ID range is invalid e.g. ID(n) > 2 AND ID(n) < 1
// op simply needs to returns NULL
static Record NodeByLabelScanNoOp
(
	OpBase *opBase
) {
	return NULL;
}

// tap consume function
// no ID specified
// simply scan label matrix
static Record NodeByLabelScanConsume
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	GrB_Index id;
	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);
	if(info == GxB_EXHAUSTED) return NULL;

	ASSERT(info == GrB_SUCCESS);

	Record r = OpBase_CreateRecord((OpBase *)op);

	// populate the Record with the actual node
	_UpdateRecord(op, r, id);

	return r;
}

// tap consume function
// ID specified
// iterate over each specified ID and make sure the current ID is labeled as L
static Record NodeByLabelAndIDScanConsume
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	GrB_Index id;
	while(roaring64_iterator_has_value(op->ID_it)) {
		id = roaring64_iterator_value(op->ID_it);
		roaring64_iterator_advance(op->ID_it);
		if(Delta_Matrix_isStoredElement(op->L, id, id) == GrB_SUCCESS) {
			Record r = OpBase_CreateRecord((OpBase *)op);

			// Populate the Record with the actual node.
			_UpdateRecord(op, r, id);

			return r;
		}
	}

	return NULL;
}

// none tap consume function
// for each child record iterator over label matrix L
static Record NodeByLabelScanConsumeFromChild
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	// try to get new nodeID
	GrB_Index id;
	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);

	// iterator depleted, try to get a new record
	while(op->child_record == NULL ||
		  info == GrB_NULL_POINTER ||
		  info == GxB_EXHAUSTED) {

		// free current record
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(&op->child_record);
		}

		// try to get a new record
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) {
			// depleted
			return NULL;
		}

		// got a new record
		if(unlikely(op->n->label_id == GRAPH_UNKNOWN_LABEL)) {
			_update_label_id(op);
		}

		if(!_ConstructIterator(op)) {
			continue;
		}

		// try to get new NodeID
		info = Delta_MatrixTupleIter_next_BOOL(&op->iter, &id, NULL, NULL);
	}

	// we've got a record and NodeID
	// clone the held Record, as it will be freed upstream
	Record r = OpBase_CloneRecord(op->child_record);

	// populate the Record with the actual node
	_UpdateRecord(op, r, id);
	return r;
}

// none tap consume function
// for each child record iterator over ID range
// make sure node is labeled as L
static Record NodeByLabelAndIDScanConsumeFromChild
(
	OpBase *opBase
) {
	NodeByLabelScan *op = (NodeByLabelScan *)opBase;

	bool      emited;
	GrB_Index id;
	GrB_Info  info;

pull:
	// get next ID from range iterator
	emited = false;
	while(op->child_record != NULL && roaring64_iterator_has_value(op->ID_it)) {
		id = roaring64_iterator_value(op->ID_it);
		roaring64_iterator_advance(op->ID_it);

		// make sure ID is labeled as L
		if(Delta_Matrix_isStoredElement(op->L, id, id) == GrB_SUCCESS) {
			emited = true;
			break;
		}
	}

	if(!emited) {
		// try to get a new record
		// free old record
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(&op->child_record);
		}

		// ask child for a new record
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) {
			// depleted
			return NULL;
		}

		// got a record
		if(op->n->label_id == GRAPH_UNKNOWN_LABEL) {
			_update_label_id(op);
		}

		_ConstructIterator(op);

		goto pull;
	}

	// we've got a record and NodeID
	// clone the held Record, as it will be freed upstream
	Record r = OpBase_CloneRecord(op->child_record);

	// populate the Record with the actual node
	_UpdateRecord(op, r, id);
	return r;
}

static OpResult NodeByLabelScanReset
(
	OpBase *ctx
) {
	NodeByLabelScan *op = (NodeByLabelScan *)ctx;

	if(OpBase_ChildCount(ctx) > 0) {
		if(op->child_record != NULL) {
			OpBase_DeleteRecord(&op->child_record); // free old record
		}
	} else {
		_ConstructIterator(op);
	}

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

	GrB_Info info = Delta_MatrixTupleIter_detach(&(op->iter));
	ASSERT(info == GrB_SUCCESS);

	if(op->child_record) {
		OpBase_DeleteRecord(&op->child_record);
	}

	if(op->n != NULL) {
		NodeScanCtx_Free(op->n);
		op->n = NULL;
	}

	if(op->ranges) {
		for(int i = 0; i < array_len(op->ranges); i++) {
			RangeExpression_Free(op->ranges + i);
		}
		array_free(op->ranges);
		op->ranges = NULL;
	}

	if(op->ID_it) {
		roaring64_iterator_free(op->ID_it);
		op->ID_it = NULL;
	}

	if(op->ids) {
		roaring64_bitmap_free(op->ids);
		op->ids = NULL;
	}
}
