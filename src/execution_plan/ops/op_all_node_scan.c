/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "op_all_node_scan.h"
#include "../../query_ctx.h"
#include "shared/print_functions.h"

// forward declarations
static OpResult AllNodeScanInit(OpBase *opBase);
static RecordBatch AllNodeScanConsume(OpBase *opBase);
static Record AllNodeScanConsumeFromChild(OpBase *opBase);
static OpResult AllNodeScanReset(OpBase *opBase);
static OpBase *AllNodeScanClone(const ExecutionPlan *plan, const OpBase *opBase);
static void AllNodeScanFree(OpBase *opBase);

static inline void AllNodeScanToString(const OpBase *ctx, sds *buf) {
	ScanToString(ctx, buf, ((AllNodeScan *)ctx)->alias, NULL);
}

OpBase *NewAllNodeScanOp
(
	const ExecutionPlan *plan,
	const char *alias
) {
	AllNodeScan *op = rm_calloc (1, sizeof(AllNodeScan)) ;

	op->alias = alias;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_ALL_NODE_SCAN, "All Node Scan",
			AllNodeScanInit, AllNodeScanConsume, AllNodeScanReset,
			AllNodeScanToString, AllNodeScanClone, AllNodeScanFree, false,
			plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, alias);
	return (OpBase *)op;
}

static OpResult AllNodeScanInit
(
	OpBase *opBase
) {
	Graph       *g  = QueryCtx_GetGraph () ;
	AllNodeScan *op = (AllNodeScan *)opBase ;

	op->node_count = Graph_NodeCount (g) ;

	if (opBase->childCount > 0) {
		OpBase_UpdateConsume (opBase, AllNodeScanConsumeFromChild) ;
	} else {
		op->iter = Graph_ScanNodes (g) ;
	}

	return OP_OK ;
}

static Record AllNodeScanConsumeFromChild
(
	OpBase *opBase
) {
	AllNodeScan *op = (AllNodeScan *)opBase;

	if(op->child_record == NULL) {
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) {
			return NULL;
		} else {
			if(!op->iter) op->iter = Graph_ScanNodes(QueryCtx_GetGraph());
			else DataBlockIterator_Reset(op->iter);
		}
	}

	Node n = GE_NEW_NODE();
	n.attributes = DataBlockIterator_Next(op->iter, &n.id);
	if(n.attributes == NULL) {
		OpBase_DeleteRecord(&op->child_record); // Free old record.
		// Pull a new record from child.
		op->child_record = OpBase_Consume(op->op.children[0]);
		if(op->child_record == NULL) return NULL; // Child depleted.

		// Reset iterator and evaluate again.
		DataBlockIterator_Reset(op->iter);
		n.attributes = DataBlockIterator_Next(op->iter, &n.id);
		if(n.attributes == NULL) return NULL; // Iterator was empty; return immediately.
	}

	// Clone the held Record, as it will be freed upstream.
	Record r = OpBase_CloneRecord(op->child_record);

	// Populate the Record with the graph entity data.
	Record_AddNode(r, op->nodeRecIdx, n);

	return r;
}

static RecordBatch AllNodeScanConsume
(
	OpBase *opBase
) {
	AllNodeScan *op = (AllNodeScan *)opBase;

	uint16_t n = MIN (op->node_count - op->progress, 64) ;
	if (n == 0) {
		return NULL ;
	}

	RecordBatch batch = OpBase_CreateRecordBatch (opBase, n) ;

	// populate batch
	for (uint16_t i = 0; i < n; i++) {
		Record r = batch[i] ;
		Node *node = Record_GetSetNode (r, op->nodeRecIdx) ;

		node->attributes = DataBlockIterator_Next (op->iter, &node->id) ;
		ASSERT (node->attributes != NULL) ;
	}

	op->progress += n ;

	return batch ;
}

static void _AllNodeScan_FreeInternals
(
	AllNodeScan *op
) {
	if (op->child_record) {
		OpBase_DeleteRecord (&op->child_record) ;
		op->child_record = NULL ;
	}

	if (op->iter != NULL) {
		DataBlockIterator_Free (op->iter) ;
		op->iter = NULL ;
	}
}

static OpResult AllNodeScanReset
(
	OpBase *op
) {
	Graph       *g           = QueryCtx_GetGraph ();
	AllNodeScan *allNodeScan = (AllNodeScan *)op;

	_AllNodeScan_FreeInternals (allNodeScan) ;

	// reset iterator by recreating it
	// a simple DataBlockIterator_Reset is NOT good enough
	// as the iterator "end" position might be outdated
	allNodeScan->iter       = Graph_ScanNodes (g) ;
	allNodeScan->progress   = 0 ;
	allNodeScan->node_count = Graph_NodeCount (g) ;

	return OP_OK ;
}

static inline OpBase *AllNodeScanClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_ALL_NODE_SCAN);
	return NewAllNodeScanOp(plan, ((AllNodeScan *)opBase)->alias);
}

static void AllNodeScanFree
(
	OpBase *ctx
) {
	AllNodeScan *op = (AllNodeScan *)ctx;
	_AllNodeScan_FreeInternals(op);
}

