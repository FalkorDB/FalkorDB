/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../query_ctx.h"
#include "shared/print_functions.h"
#include "op_conditional_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../../arithmetic/algebraic_expression/utils.h"

// default number of records to accumulate before traversing
#define BATCH_SIZE 16

// forward declarations
static OpResult CondTraverseInit(OpBase *opBase);
static Record CondTraverseConsume(OpBase *opBase);
static OpResult CondTraverseReset(OpBase *opBase);
static OpBase *CondTraverseClone(const ExecutionPlan *plan, const OpBase *opBase);
static void CondTraverseFree(OpBase *opBase);

static void CondTraverseToString
(
	const OpBase *ctx,
	sds *buf
) {
	TraversalToString(ctx, buf, ((const OpCondTraverse *)ctx)->ae);
}

static void _populate_filter_matrix
(
	OpCondTraverse *op
) {
	GrB_Matrix FM = Delta_Matrix_M(op->F);

	// clear filter matrix
	GrB_Matrix_clear(FM);

	// update filter matrix F, set row i at position srcId
	// F[i, srcId] = true
	for(uint i = 0; i < op->record_count; i++) {
		Record r = op->records[i];
		Node *n = Record_GetNode(r, op->srcNodeIdx);
		if (!op->optional || n != NULL) {
			NodeID srcId = ENTITY_GET_ID(n);
			GrB_Matrix_setElement_BOOL(FM, true, i, srcId);
		}
	}
}

// evaluate algebraic expression:
// prepends filter matrix as the left most operand
// perform multiplications
// set iterator over result matrix
// removed filter matrix from original expression
// clears filter matrix
void _traverse
(
	OpCondTraverse *op
) {
	// if op->F is null, this is the first time we are traversing
	if(op->F == NULL) {
		// create both filter and result matrices
		size_t required_dim = Graph_RequiredMatrixDim(op->graph);
		Delta_Matrix_new(&op->M, GrB_BOOL, op->record_cap, required_dim, false);
		Delta_Matrix_new(&op->F, GrB_BOOL, op->record_cap, required_dim, false);

		// prepend filter matrix to algebraic expression as the leftmost operand
		AlgebraicExpression_MultiplyToTheLeft(&op->ae, op->F);

		// optimize the expression tree
		AlgebraicExpression_Optimize(&op->ae);

		// partial_ae is true when
		// the algebraic expression contains the zero matrix
		op->partial_ae = AlgebraicExpression_ContainsMatrix(op->ae,
				Graph_GetZeroMatrix(QueryCtx_GetGraph()));
	}

	// populate filter matrix
	_populate_filter_matrix(op);

	// evaluate expression
	AlgebraicExpression_Eval(op->ae, op->M);

	Delta_MatrixTupleIter_attach(&op->iter, op->M);

	//--------------------------------------------------------------------------
	// collect records with neighborless src node
	//--------------------------------------------------------------------------

	if (op->optional) {
		// w[i] = true if row i of M is not empty
		GrB_OK (GrB_Matrix_reduce_Monoid (op->w, NULL, NULL,
					GxB_LOR_BOOL_MONOID, DELTA_MATRIX_M (op->M), NULL)) ;

		// e[i] = true row i of M is empty
		GrB_Index nrows = op->record_cap ;
		GrB_OK (GrB_Vector_assign_Scalar (op->e, op->w, NULL, op->s, GrB_ALL,
					nrows, GrB_DESC_RSC));

		// populate optional records
		GxB_Iterator it ;
		GrB_OK (GxB_Iterator_new (&it)) ;
		GrB_Info info = GxB_Vector_Iterator_attach (it, op->e, NULL) ;
		info = GxB_Vector_Iterator_seek (it, 0) ;

		while (info != GxB_EXHAUSTED) {
			GrB_Index i = GxB_Vector_Iterator_getIndex (it) ;

			// check for overflow
			if (i >= op->record_count) {
				break ;
			}

			// add record to optional_records
			Record r = op->records[i] ;
			Record_IncRefCount (r) ;
			array_append (op->optional_records, r) ;

			// move to the next entry in v
			info = GxB_Vector_Iterator_next (it) ;
		}

		GrB_OK (GrB_free (&it)) ;
	}
}

OpBase *NewCondTraverseOp
(
	const ExecutionPlan *plan,
	Graph *g,
	AlgebraicExpression *ae
) {
	OpCondTraverse *op = rm_calloc (1, sizeof(OpCondTraverse)) ;

	op->ae         = ae;
	op->graph      = g;
	op->record_cap = BATCH_SIZE;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_CONDITIONAL_TRAVERSE,
			"Conditional Traverse", CondTraverseInit, CondTraverseConsume,
			CondTraverseReset, CondTraverseToString, CondTraverseClone,
			CondTraverseFree, false, plan);

	bool aware = OpBase_AliasMapping((OpBase *)op, AlgebraicExpression_Src(ae),
			&op->srcNodeIdx);
	UNUSED(aware);
	ASSERT(aware == true);

	const char *dest = AlgebraicExpression_Dest(ae);
	op->destNodeIdx = OpBase_Modifies((OpBase *)op, dest);

	const char *edge = AlgebraicExpression_Edge(ae);
	if(edge) {
		// this operation will populate an edge in the Record
		// prepare all necessary information for collecting matching edges
		uint edge_idx = OpBase_Modifies((OpBase *)op, edge);
		QGEdge *e = QueryGraph_GetEdgeByAlias(plan->query_graph, edge);
		op->edge_ctx = EdgeTraverseCtx_New(ae, e, edge_idx);
	}

	return (OpBase *)op;
}

// make traversal optional
// OPTIONAL MATCH (a)-[]->(b)
// `a` will be passed onwards even if it doesn't have any neighbors
void CondTraverse_MakeOptional
(
	OpCondTraverse *op
) {
	ASSERT (op != NULL) ;
	ASSERT (op->optional == false) ;
	ASSERT (op->optional_records == NULL) ;

	op->optional = true ;
	op->op.name  = "Optional Conditional Traverse" ;
	op->op.type  = OPType_OPTIONAL_CONDITIONAL_TRAVERSE ;
	op->optional_records = array_new (Record, op->record_cap) ;

	// s = true
	GrB_OK (GxB_Scalar_new (&op->s, GrB_BOOL)) ;
	GrB_OK (GxB_Scalar_setElement_BOOL (op->s, true)) ;

	// allocate reduction & neighborless vectors
	GrB_Index nrows = op->record_cap ;
	GrB_OK (GrB_Vector_new (&op->w, GrB_BOOL, nrows)) ;
	GrB_OK (GrB_Vector_new (&op->e, GrB_BOOL, nrows)) ;
}

static OpResult CondTraverseInit
(
	OpBase *opBase
) {
	OpCondTraverse *op = (OpCondTraverse *)opBase;

	// in case this operation is restricted by a limit
	// set record_cap to the specified limit
	ExecutionPlan_ContainsLimit(opBase, &op->record_cap);

	// record_cap should not be greater than BATCH_SIZE
	if(op->record_cap > BATCH_SIZE) op->record_cap = BATCH_SIZE;

	op->records = rm_calloc(op->record_cap, sizeof(Record));

	return OP_OK;
}

// each call to CondTraverseConsume emits a Record containing the
// traversal's endpoints and, if required, an edge
// returns NULL once all traversals have been performed
static Record CondTraverseConsume
(
	OpBase *opBase
) {
	OpCondTraverse *op = (OpCondTraverse *)opBase ;
	OpBase *child = op->op.children[0] ;

	//--------------------------------------------------------------------------
	// emit edges
	//--------------------------------------------------------------------------

	if (op->r        != NULL &&
	    op->edge_ctx != NULL &&
	    EdgeTraverseCtx_SetEdge (op->edge_ctx, op->r)) {
		return OpBase_CloneRecord (op->r) ;
	}

	NodeID src_id  = INVALID_ENTITY_ID ;
	NodeID dest_id = INVALID_ENTITY_ID ;

	while (true) {
		GrB_Info info =
      Delta_MatrixTupleIter_next_BOOL (&op->iter, &src_id, &dest_id, NULL) ;

		// managed to get a tuple, break
		if (info == GrB_SUCCESS) {
			break ;
		}

		// if optional, emit nodes without neighbors
		if (op->optional && array_len (op->optional_records) > 0) {
			Record r = array_pop (op->optional_records) ;
			return r ;
		}

		//----------------------------------------------------------------------
		// run out of records, free old records
		//----------------------------------------------------------------------

		op->r = NULL ;

		for (uint i = 0; i < op->record_count; i++) {
			OpBase_DeleteRecord (op->records+i) ;
		}

		if (op->optional) {
			for (uint i = 0; i < array_len (op->optional_records); i++) {
				OpBase_DeleteRecord (op->optional_records+i) ;
			}
			array_clear (op->optional_records) ;
		}

		//----------------------------------------------------------------------
		// try to get new data
		//----------------------------------------------------------------------

		// ask child operations for data
		for (op->record_count = 0; op->record_count < op->record_cap; op->record_count++) {
			Record childRecord = OpBase_Consume (child) ;
			// if the Record is NULL, the child has been depleted
			if (childRecord == NULL) {
				break ;
			}

			if (!Record_GetNode (childRecord, op->srcNodeIdx) &&
				!op->optional) {
				// the child Record may not contain the source node in scenarios
				// like a failed OPTIONAL MATCH
				// in this case, delete the Record and try again
				OpBase_DeleteRecord (&childRecord) ;
				op->record_count-- ;
				continue ;
			}

			// store received record
			op->records[op->record_count] = childRecord ;
		}

		// no data
		if (op->record_count == 0) {
			return NULL ;
		}

		_traverse (op) ;
	}

	// get node from current column
	op->r = op->records[src_id] ;

	// populate the destination node and add it to the Record
	Node destNode = GE_NEW_NODE() ;
	Graph_GetNode (op->graph, dest_id, &destNode) ;
	Record_AddNode (op->r, op->destNodeIdx, destNode);

	if (op->edge_ctx) {
		Node *srcNode = Record_GetNode (op->r, op->srcNodeIdx) ;
		// collect all appropriate edges connecting the current pair of endpoints
		EdgeTraverseCtx_CollectEdges (op->edge_ctx, ENTITY_GET_ID (srcNode),
				ENTITY_GET_ID (&destNode)) ;
		// we're guaranteed to have at least one edge
		EdgeTraverseCtx_SetEdge (op->edge_ctx, op->r) ;
	}

	return OpBase_CloneRecord (op->r) ;
}

static OpResult CondTraverseReset
(
	OpBase *ctx
) {
	OpCondTraverse *op = (OpCondTraverse *)ctx ;

	// do not explicitly free op->r, as the same pointer is also held
	// in the op->records array and as such will be freed there
	op->r = NULL ;
	for (uint i = 0; i < op->record_count; i++) {
		OpBase_DeleteRecord (op->records+i) ;
	}
	op->record_count = 0 ;

	if (op->optional) {
		GrB_OK (GrB_Vector_clear (op->w)) ;
		GrB_OK (GrB_Vector_clear (op->e)) ;

		for (uint i = 0; i < array_len (op->optional_records); i++) {
			OpBase_DeleteRecord (op->optional_records+i) ;
		}
		array_clear (op->optional_records) ;
	}

	if(op->edge_ctx) EdgeTraverseCtx_Reset(op->edge_ctx);

	GrB_Info info = Delta_MatrixTupleIter_detach(&op->iter);
	ASSERT(info == GrB_SUCCESS);

	if(op->F != NULL) Delta_Matrix_clear(op->F);

	// in case algebraic expression has missing operands
	// i.e. has an operand which is the zero matrix
	// see if at this point in time the graph is aware of the missing operand
	// and if so replace the zero matrix operand with the actual matrix
	if(unlikely(op->partial_ae == true)) {
		_AlgebraicExpression_PopulateOperands(op->ae, QueryCtx_GetGraphCtx());
	}

	return OP_OK;
}

static inline OpBase *CondTraverseClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (opBase->type == OPType_CONDITIONAL_TRAVERSE ||
			opBase->type == OPType_OPTIONAL_CONDITIONAL_TRAVERSE) ;

	OpCondTraverse *op = (OpCondTraverse *)opBase ;

	OpBase *clone = NewCondTraverseOp (plan, QueryCtx_GetGraph(),
			AlgebraicExpression_Clone (op->ae)) ;

	if (op->optional) {
		CondTraverse_MakeOptional ((OpCondTraverse*)clone) ;
	}

	return clone ;
}

// frees CondTraverse
static void CondTraverseFree
(
	OpBase *ctx
) {
	OpCondTraverse *op = (OpCondTraverse *)ctx;

	GrB_Info info = Delta_MatrixTupleIter_detach(&op->iter);
	ASSERT(info == GrB_SUCCESS);

	if(op->F != NULL) {
		Delta_Matrix_free(&op->F);
		op->F = NULL;
	}

	if(op->M != NULL) {
		Delta_Matrix_free(&op->M);
		op->M = NULL;
	}

	if(op->ae) {
		AlgebraicExpression_Free(op->ae);
		op->ae = NULL;
	}

	if(op->edge_ctx) {
		EdgeTraverseCtx_Free(op->edge_ctx);
		op->edge_ctx = NULL;
	}

	if(op->records) {
		for(uint i = 0; i < op->record_count; i++) {
			OpBase_DeleteRecord(op->records+i);
		}
		rm_free(op->records);
		op->records = NULL;
	}

	if (op->optional) {
		for (uint i = 0; i < array_len (op->optional_records); i++) {
			OpBase_DeleteRecord (op->optional_records+i) ;
		}
		array_free (op->optional_records) ;
		op->optional_records = NULL ;

		GrB_free (&op->w) ;
		GrB_free (&op->e) ;
		GrB_free (&op->s) ;
	}
}

