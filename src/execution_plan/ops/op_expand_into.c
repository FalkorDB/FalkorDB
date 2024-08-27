/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_expand_into.h"
#include "../../query_ctx.h"
#include "shared/print_functions.h"
#include "../execution_plan_build/execution_plan_util.h"

// default number of records to accumulate before traversing
#define BATCH_SIZE 16

// forward declarations
static OpResult ExpandIntoInit(OpBase *opBase);
static Record ExpandIntoConsume(OpBase *opBase);
static OpResult ExpandIntoReset(OpBase *opBase);
static OpBase *ExpandIntoClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ExpandIntoFree(OpBase *opBase);

// string representation of operation
static inline void ExpandIntoToString
(
	const OpBase *ctx,
	sds *buf
) {
	TraversalToString(ctx, buf, ((const OpExpandInto *)ctx)->ae);
}

// construct filter matrix F
// F[i,j] = 1 if row the ith record ID(src) = j
static void _populate_filter_matrix
(
	OpExpandInto *op
) {
	GrB_Matrix FM = RG_MATRIX_M(op->F);

	// clear filter matrix
	GrB_Matrix_clear(FM);

	for(uint i = 0; i < op->record_count; i++) {
		Record r = op->records[i];
		// update filter matrix F
		// set row i at position srcId
		// F[i, srcId] = true
		Node *n = Record_GetNode(r, op->srcNodeIdx);
		NodeID srcId = ENTITY_GET_ID(n);
		GrB_Matrix_setElement_BOOL(FM, true, i, srcId);
	}

	GrB_Matrix_wait(FM, GrB_MATERIALIZE);
}

// evaluate algebraic expression:
// appends filter matrix as the left most operand
// perform multiplications
// removed filter matrix from original expression
// clears filter matrix
static void _traverse
(
	OpExpandInto *op
) {
	// if op->F is null, this is the first time we are traversing
	if(op->F == NULL) {
		// create both filter matrix F and result matrix M
		size_t required_dim = Graph_RequiredMatrixDim(op->graph);
		RG_Matrix_new(&op->M, GrB_BOOL, op->record_cap, required_dim);
		RG_Matrix_new(&op->F, GrB_BOOL, op->record_cap, required_dim);

		// prepend the filter matrix to algebraic expression
		// as the leftmost operand
		AlgebraicExpression_MultiplyToTheLeft(&op->ae, op->F);
		AlgebraicExpression_Optimize(&op->ae);
	}

	// populate filter matrix
	_populate_filter_matrix(op);

	// evaluate expression
	AlgebraicExpression_Eval(op->ae, op->M);
}

OpBase *NewExpandIntoOp
(
	const ExecutionPlan *plan,
	Graph *g,
	AlgebraicExpression *ae
) {
	OpExpandInto *op = rm_malloc(sizeof(OpExpandInto));

	op->r               =  NULL;
	op->F               =  NULL;
	op->M               =  NULL;
	op->ae              =  ae;
	op->graph           =  g;
	op->records         =  NULL;
	op->edge_ctx        =  NULL;
	op->record_cap      =  BATCH_SIZE;
	op->record_count    =  0;
	op->single_operand  =  false;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_EXPAND_INTO, "Expand Into", ExpandIntoInit,
			ExpandIntoConsume, ExpandIntoReset, ExpandIntoToString,
			ExpandIntoClone, ExpandIntoFree, false, plan);

	// make sure that all entities are represented in record
	bool aware;
	UNUSED(aware);
 
	aware = OpBase_Aware((OpBase *)op, AlgebraicExpression_Src(ae), &op->srcNodeIdx);
	ASSERT(aware);
	aware = OpBase_Aware((OpBase *)op, AlgebraicExpression_Dest(ae), &op->destNodeIdx);
	ASSERT(aware);

	const char *edge = AlgebraicExpression_Edge(ae);
	if(edge) {
		// this operation will populate an edge in the record
		// prepare all necessary information for collecting matching edges
		uint edge_idx = OpBase_Modifies((OpBase *)op, edge);
		QGEdge *e = QueryGraph_GetEdgeByAlias(plan->query_graph, edge);
		op->edge_ctx = EdgeTraverseCtx_New(ae, e, edge_idx);
	}

	return (OpBase *)op;
}

static OpResult ExpandIntoInit
(
	OpBase *opBase
) {
	OpExpandInto *op = (OpExpandInto *)opBase;

	// in case this operation is restricted by a limit
	// set record_cap to the specified limit
	ExecutionPlan_ContainsLimit(opBase, &op->record_cap);

	// see if we can optimize by avoiding matrix multiplication
	// if the algebraic expression passed in is just a single operand
	// there's no need to compute F and perform F*X, we can simply inspect X
	if(op->ae->type == AL_OPERAND) {
		// if traversed expression is a single operand e.g. [R]
		// check if specified operand R exists
		GraphContext *gc = QueryCtx_GetGraphCtx();
		const char *label = AlgebraicExpression_Label(op->ae);
		if(label == NULL) {
			// matrix isn't associated with a label, use the adjacency matrix
			op->M = Graph_GetAdjacencyMatrix(op->graph, false);
		} else {
			// try to retrieve relationship matrix
			// it is OK if the relationship doesn't exists, in this case
			// we won't use this single operand optimization
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
			if(s != NULL) {
				// stream records as they enter
				op->M = Graph_GetRelationMatrix(op->graph, Schema_GetID(s), false);
			}
		}

		// if we've managed to set M, restrict record cap to 1
		// and note the optimization
		if(op->M) {
			op->record_cap      =  1;     // record buffer size will be set to 1
			op->single_operand  =  true;
		}
	}

	// allocate record buffer, limited to a maximum of BATCH_SIZE
	if(op->record_cap > BATCH_SIZE) op->record_cap = BATCH_SIZE;

	op->records = rm_calloc(op->record_cap, sizeof(Record));

	return OP_OK;
}

// emits a record returns NULL when depleted
static Record _handoff
(
	OpExpandInto *op
) {
	Record r;

	// if we're required to update an edge and have one queued
	// we can return early
	// otherwise, try to get a new pair of source and destination nodes
	if(op->edge_ctx != NULL && op->r != NULL) {
		emit_edge:
		if(EdgeTraverseCtx_SetEdge(op->edge_ctx, op->r)) {
			if(EdgeTraverseCtx_EdgeCount(op->edge_ctx) == 0) {
				// processing last edge, no need to clone op->r
				r = op->r;
				op->r = NULL;
				return r;
			} else {
				// multiple edges, clone op->r
				return OpBase_CloneRecord(op->r);
			}
		} else {
			// failed to produce edge, free record
			OpBase_DeleteRecord(&op->r);
		}
	}

	// find a record where both record's source and destination
	// nodes are connected M[i,j] is set
	while(op->record_count) {
		op->record_count--;
		r = op->records[op->record_count];

		bool x;
		uint row;

		// resolve row index
		if(op->single_operand) {
			// row idx = src node ID
			Node *srcNode = Record_GetNode(r, op->srcNodeIdx);
			row = ENTITY_GET_ID(srcNode);
		} else {
			// row idx = record idx
			row = op->record_count;
		}

		Node *destNode  =  Record_GetNode(r, op->destNodeIdx);
		NodeID col      =  ENTITY_GET_ID(destNode);
		// TODO: in the case of multiple operands ()-[:A]->()-[:B]->()
		// M is the result of F*A*B, in which case we can switch from
		// M being a RG_Matrix to a GrB_Matrix, making the extract element
		// operation a bit cheaper to compute
		GrB_Info res    =  RG_Matrix_extractElement_BOOL(&x, op->M, row, col);

		// src is not connected to dest, free the current record and continue
		if(res != GrB_SUCCESS) {
			OpBase_DeleteRecord(&r);
			continue;
		}

		// src is connected to dest
		// update the edge if necessary
		if(op->edge_ctx != NULL) {
			op->r = r;

			Node      *srcNode  =  Record_GetNode(r, op->srcNodeIdx);
			EntityID  row       =  ENTITY_GET_ID(srcNode);

			// collect all edges connecting the current pair of endpoints
			EdgeTraverseCtx_CollectEdges(op->edge_ctx, row, col);
			goto emit_edge;
		}

		return r;
	}

	// didn't manage to emit record
	return NULL;
}

// ExpandIntoConsume next operation
// returns OP_DEPLETED when no additional updates are available
static Record ExpandIntoConsume
(
	OpBase *opBase
) {
	Record r;
	OpExpandInto *op = (OpExpandInto *)opBase;
	OpBase *child = op->op.children[0];

	// as long as we don't have a record to emit
	while((r = _handoff(op)) == NULL) {
		// if we're here, we didn't manage to emit a record
		// clean up and try to get new data points

		// validate depleted
		ASSERT(op->r == NULL);
		ASSERT(op->record_count == 0);

		//----------------------------------------------------------------------
		// get data
		//----------------------------------------------------------------------

		// ask child operation for at most 'record_cap' records
		int i = 0;
		for(; i < op->record_cap; i++) {
			r = OpBase_Consume(child);
			// did not manage to get new data, break
			if(r == NULL) break;

			// check if both src and destination nodes are set
			if(!Record_GetNode(r, op->srcNodeIdx) ||
			   !Record_GetNode(r, op->destNodeIdx)) {
				// the child Record may not contain eithe
				// source or destination nodes in scenarios like a failed
				// OPTIONAL MATCH in this case, delete the Record and try again
				OpBase_DeleteRecord(&r);
				i--;
				continue;
			}

			// store received record
			op->records[i] = r;
		}
		op->record_count = i;

		// did not managed to produce data, depleted
		if(op->record_count == 0) return NULL;

		if(!op->single_operand) _traverse(op);
	}

	return r;
}

static OpResult ExpandIntoReset
(
	OpBase *ctx
) {
	OpExpandInto *op = (OpExpandInto *)ctx;

	if(op->r != NULL) {
		OpBase_DeleteRecord(&op->r);
	}

	for(uint i = 0; i < op->record_count; i++) {
		OpBase_DeleteRecord(op->records+i);
	}
	op->record_count = 0;

	if(op->edge_ctx != NULL) EdgeTraverseCtx_Reset(op->edge_ctx);

	return OP_OK;
}

static inline OpBase *ExpandIntoClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_EXPAND_INTO);

	OpExpandInto *op = (OpExpandInto *)opBase;

	return NewExpandIntoOp(plan, op->graph, AlgebraicExpression_Clone(op->ae));
}

// frees ExpandInto
static void ExpandIntoFree
(
	OpBase *ctx
) {
	OpExpandInto *op = (OpExpandInto *)ctx;

	if(op->F != NULL) {
		RG_Matrix_free(&op->F);
		op->F = NULL;
	}

	if(op->ae != NULL) {
		// M was allocated by us
		if(op->M != NULL && !op->single_operand) {
			RG_Matrix_free(&op->M);
			op->M = NULL;
		}

		AlgebraicExpression_Free(op->ae);
		op->ae = NULL;
	}

	if(op->edge_ctx != NULL) {
		EdgeTraverseCtx_Free(op->edge_ctx);
		op->edge_ctx = NULL;
	}

	if(op->records != NULL) {
		for(uint i = 0; i < op->record_count; i++) {
			OpBase_DeleteRecord(op->records+i);
		}
		rm_free(op->records);
		op->records = NULL;
	}

	if(op->r != NULL) {
		OpBase_DeleteRecord(&op->r);
	}
}
