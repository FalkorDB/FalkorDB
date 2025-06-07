/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_cond_var_len_traverse.h"
#include "shared/print_functions.h"
#include "../../util/arr.h"
#include "../../ast/ast.h"
#include "../../arithmetic/arithmetic_expression.h"
#include "../../graph/graphcontext.h"
#include "../../algorithms/all_paths.h"
#include "../../algorithms/all_neighbors.h"
#include "../../query_ctx.h"

// forward declarations
static OpResult CondVarLenTraverseInit(OpBase *opBase);
static OpResult CondVarLenTraverseReset(OpBase *opBase);
static Record CondVarLenTraverseConsume(OpBase *opBase);
static Record CondVarLenTraverseOptimizedConsume(OpBase *opBase);
static OpBase *CondVarLenTraverseClone(const ExecutionPlan *plan, const OpBase *opBase);
static void CondVarLenTraverseFree(OpBase *opBase);

static void _setupTraversedRelations
(
	CondVarLenTraverse *op
) {
	QGEdge *e = QueryGraph_GetEdgeByAlias(op->op.plan->query_graph,
			AlgebraicExpression_Edge(op->ae));

	ASSERT(e->minHops <= e->maxHops);
	op->minHops = e->minHops;
	op->maxHops = e->maxHops;

	uint reltype_count = QGEdge_RelationCount(e);
	if(reltype_count == 0) {
		op->edgeRelationCount = 1;
		op->edgeRelationTypes = array_new(int, 1);
		array_append(op->edgeRelationTypes, GRAPH_NO_RELATION);
	} else {
		GraphContext *gc = QueryCtx_GetGraphCtx();
		op->edgeRelationCount = 0;
		op->edgeRelationTypes = array_new(int, reltype_count);

		for(int i = 0; i < reltype_count; i++) {
			int rel_id = e->reltypeIDs[i];
			if(rel_id != GRAPH_UNKNOWN_RELATION) {
				array_append(op->edgeRelationTypes, rel_id);
			} else {
				const char *rel_type = e->reltypes[i];
				Schema *s = GraphContext_GetSchema(gc, rel_type, SCHEMA_EDGE);
				if(s) array_append(op->edgeRelationTypes, s->id);
			}
		}

		op->edgeRelationCount = array_len(op->edgeRelationTypes);
	}
}

// set the traversal direction to match the traversed edge
// and AlgebraicExpression form
static inline void _setTraverseDirection
(
	CondVarLenTraverse *op,
	const QGEdge *e
) {
	if(e->bidirectional) {
		op->traverseDir = GRAPH_EDGE_DIR_BOTH;
	} else {
		if(AlgebraicExpression_Transposed(op->ae)) {
			// traverse in the opposite direction, (dest)->(src) incoming edges
			op->traverseDir = GRAPH_EDGE_DIR_INCOMING;
		} else {
			op->traverseDir = GRAPH_EDGE_DIR_OUTGOING;
		}
	}
}

static inline void CondVarLenTraverseToString
(
	const OpBase *ctx,
	sds *buf
) {
	// TODO: tmp, improve TraversalToString
	CondVarLenTraverse *op = (CondVarLenTraverse *)ctx;
	AlgebraicExpression_Optimize(&op->ae);
	TraversalToString(ctx, buf, op->ae);
}

void CondVarLenTraverseOp_ExpandInto
(
	CondVarLenTraverse *op
) {
	// expand into doesn't performs any modifications
	array_clear(op->op.modifies);
	op->expandInto = true;
	op->op.type = OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO;
	op->op.name = "Conditional Variable Length Traverse (Expand Into)";
}

inline void CondVarLenTraverseOp_SetFilter
(
	CondVarLenTraverse *op,
	FT_FilterNode *ft
) {
	ASSERT(op     != NULL);
	ASSERT(ft     != NULL);
	ASSERT(op->ft == NULL);

	op->ft = ft;

	// adjust filter tree
	// replace each 'intermediate_path' function call argument from edge alias
	// to '@path'

	AR_ExpNode **fs = array_new(AR_ExpNode*, 0);
	FilterTree_CollectFunctionCalls(ft, "intermediate_path", &fs);

	size_t n = array_len(fs);
	if(n > 0) {
		op->pathIdx = OpBase_Modifies((OpBase *)op, "@path");

		for(int i = 0; i < n; i++) {
			AR_ExpNode *exp = fs[i];
			ASSERT(AR_EXP_childCount(exp) == 1);

			AR_ExpNode *arg = AR_EXP_getChild(exp, 0);

			if(arg->type         != AR_EXP_OPERAND ||
			   arg->operand.type != AR_EXP_VARIADIC) {
				ErrorCtx_SetError("intermediate_path must accept a single variable length edge alias");
				break;
			}

			// update operand name to '@path'
			arg->operand.variadic.entity_alias = "@path";
			arg->operand.variadic.entity_alias_idx = op->pathIdx;
		}
	}

	array_free(fs);
}

OpBase *NewCondVarLenTraverseOp
(
	const ExecutionPlan *plan,
	Graph *g,
	AlgebraicExpression *ae
) {
	ASSERT(g  != NULL);
	ASSERT(ae != NULL);

	CondVarLenTraverse *op = rm_calloc(1, sizeof(CondVarLenTraverse));

	op->g             = g;
	op->ae            = ae;
	op->pathIdx       = -1; // path is not referenced
	op->collect_paths = true;

	OpBase_Init((OpBase *)op, OPType_CONDITIONAL_VAR_LEN_TRAVERSE,
				"Conditional Variable Length Traverse", CondVarLenTraverseInit,
				CondVarLenTraverseConsume, CondVarLenTraverseReset,
				CondVarLenTraverseToString, CondVarLenTraverseClone,
				CondVarLenTraverseFree, false, plan);

	bool aware = OpBase_AliasMapping((OpBase *)op, AlgebraicExpression_Src(ae),
			&op->srcNodeIdx);
	ASSERT(aware);

	op->destNodeIdx = OpBase_Modifies((OpBase *)op,
			AlgebraicExpression_Dest(ae));

	// populate edge value in record only if it is referenced
	AST *ast = QueryCtx_GetAST();
	QGEdge *e = QueryGraph_GetEdgeByAlias(plan->query_graph,
			AlgebraicExpression_Edge(op->ae));

	op->edgesIdx = AST_AliasIsReferenced(ast, e->alias) ?
		OpBase_Modifies((OpBase *)op, e->alias) :
		-1;

	op->shortestPaths = QGEdge_IsShortestPath(e);

	_setTraverseDirection(op, e);

	return (OpBase *)op;
}

static OpResult CondVarLenTraverseInit
(
	OpBase *opBase
) {
	CondVarLenTraverse *op = (CondVarLenTraverse *)opBase;

	// check if variable length traversal doesn't require path construction
	// in which case we only care for reachable destination nodes
	// which is alot cheaper to compute
	//
	// consider:
	// MATCH (a)-[:L*2..4]->(b) RETURN b
	// we only care for destination nodes, the actual path is not of interest
	//
	// for this we require:
	// 1. no filters to be applied to pattern
	// 2. traversed edge isn't referenced
	// 3. traversal of a single relationship: R, RT
	// 4. traversal must be directed
	//
	// in which case we can use a faster consume function

	QGEdge *e = QueryGraph_GetEdgeByAlias(op->op.plan->query_graph,
			AlgebraicExpression_Edge(op->ae));
	uint reltype_count = QGEdge_RelationCount(e);

	bool multi_edge = true;
	if(reltype_count == 1) {
		int rel_id = QGEdge_RelationID(e, 0);
		if(rel_id != GRAPH_NO_RELATION && rel_id != GRAPH_UNKNOWN_RELATION) {
			multi_edge = Graph_RelationshipContainsMultiEdge(op->g, rel_id);
		}
	}

	if(op->ft          == NULL  && // no filter on path
	   op->edgesIdx    == -1    && // edge isn't required
	   op->expandInto  == false && // destination unknown
	   reltype_count   == 1     && // single relationship
	   multi_edge      == false && // no multi edge entries
	   op->traverseDir != GRAPH_EDGE_DIR_BOTH    // directed
	  ) {
		AlgebraicExpression_Optimize(&op->ae);
		ASSERT(op->ae->type == AL_OPERAND);
		op->collect_paths = false;
		OpBase_UpdateConsume(opBase, CondVarLenTraverseOptimizedConsume);
	}

	return OP_OK;
}

static Record CondVarLenTraverseOptimizedConsume
(
	OpBase *opBase
) {
	CondVarLenTraverse  *op     = (CondVarLenTraverse *)opBase;
	OpBase              *child  =  op->op.children[0];
	Node                dest    =  GE_NEW_NODE();
	EntityID            dest_id =  INVALID_ENTITY_ID;

	while((dest_id = AllNeighborsCtx_NextNeighbor(op->allNeighborsCtx)) ==
		  INVALID_ENTITY_ID) {
		Record childRecord = OpBase_Consume(child);
		if(!childRecord) return NULL;

		OpBase_DeleteRecord(&op->r);
		op->r = childRecord;

		Node *srcNode = Record_GetNode(op->r, op->srcNodeIdx);
		if(srcNode == NULL) {
			// the child Record may not contain the source node
			// in scenarios like a failed OPTIONAL MATCH
			// in this case, delete the Record and try again
			OpBase_DeleteRecord(&op->r);
			continue;
		}

		// create edge relation type array on first call to consume
		if(!op->edgeRelationTypes) {
			_setupTraversedRelations(op);
			// incase we don't have any relations to traverse
			// and minimal traversal is at least one hop
			// we can return quickly
			// consider: MATCH (S)-[:L*]->(M) RETURN M
			// where label L does not exists */
			if(op->edgeRelationCount == 0 && op->minHops > 0) return NULL;

			op->M = op->ae->operand.matrix;
		}

		if(op->allNeighborsCtx == NULL) {
			op->allNeighborsCtx = AllNeighborsCtx_New(srcNode->id, op->M,
					op->minHops, op->maxHops);
		} else {
			// in case ctx already allocated simply reset it
			AllNeighborsCtx_Reset(op->allNeighborsCtx, srcNode->id, op->M,
					op->minHops, op->maxHops);
		}
	}

	// could not produce destination node, return
	if(dest_id == INVALID_ENTITY_ID) return NULL;

	int res = Graph_GetNode(op->g, dest_id, &dest);
	UNUSED(res);
	ASSERT(res == true);

	//--------------------------------------------------------------------------
	// populate output record
	//--------------------------------------------------------------------------

	// add destination node to record
	Record r = OpBase_CloneRecord(op->r);
	Record_AddNode(r, op->destNodeIdx, dest);

	return r;
}

static Record CondVarLenTraverseConsume
(
	OpBase *opBase
) {
	CondVarLenTraverse  *op    = (CondVarLenTraverse *)opBase;
	Path                *p     = NULL;
	OpBase              *child = op->op.children[0];

	while(!(p = AllPathsCtx_NextPath(op->allPathsCtx))) {
		Record childRecord = OpBase_Consume(child);
		if(!childRecord) return NULL;

		OpBase_DeleteRecord(&op->r);
		op->r = childRecord;

		Node *srcNode = Record_GetNode(op->r, op->srcNodeIdx);
		if(srcNode == NULL) {
			// the child Record may not contain the source node in scenarios like
			// a failed OPTIONAL MATCH. In this case, delete the Record and try again
			OpBase_DeleteRecord(&op->r);
			continue;
		}

		// create edge relation type array on first call to consume
		if(!op->edgeRelationTypes) {
			_setupTraversedRelations(op);
			// incase we don't have any relations to traverse and
			// minimal traversal is at least one hop, we can return quickly
			// consider: MATCH (S)-[:L*]->(M) RETURN M
			// where label L does not exists
			if(op->edgeRelationCount == 0 && op->minHops > 0) return NULL;
		}

		Node *destNode = NULL;
		// the destination node is known in advance if we're performing an ExpandInto
		if(op->expandInto) destNode = Record_GetNode(op->r, op->destNodeIdx);

		AllPathsCtx_Free(op->allPathsCtx);
		op->allPathsCtx = AllPathsCtx_New(srcNode, destNode, op->g,
				op->edgeRelationTypes, op->edgeRelationCount, op->traverseDir,
				op->minHops, op->maxHops, op->r, op->ft, op->pathIdx,
				op->edgesIdx, op->shortestPaths);
	}

	//--------------------------------------------------------------------------
	// populate output record
	//--------------------------------------------------------------------------

	Record r = OpBase_CloneRecord(op->r);

	// add destination node to record
	if(!op->expandInto) Record_AddNode(r, op->destNodeIdx, Path_Head(p));

	// add new path to record
	if(op->edgesIdx >= 0) Record_AddScalar(r, op->edgesIdx, SI_Path(p));

	return r;
}

static OpResult CondVarLenTraverseReset
(
	OpBase *ctx
) {
	CondVarLenTraverse *op = (CondVarLenTraverse *)ctx;
	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}

	if(op->collect_paths) {
		if(op->allPathsCtx) {
			AllPathsCtx_Free(op->allPathsCtx);
			op->allPathsCtx = NULL;
		}
	} else {
		if(op->allNeighborsCtx) {
			AllNeighborsCtx_Free(op->allNeighborsCtx);
			op->allNeighborsCtx = NULL;
		}
	}

	return OP_OK;
}

static OpBase *CondVarLenTraverseClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_CONDITIONAL_VAR_LEN_TRAVERSE ||
		   opBase->type == OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO);

	// clone "conditional var length traversal" operation
	// handels the case where the operation had been modified into an
	// "conditional var len expand into"
	CondVarLenTraverse *op = (CondVarLenTraverse *) opBase;
	OpBase *clone = NewCondVarLenTraverseOp(plan, QueryCtx_GetGraph(),
			AlgebraicExpression_Clone(op->ae));

	// clone filter tree
	if(op->ft != NULL) {
		CondVarLenTraverseOp_SetFilter((CondVarLenTraverse*)clone,
				FilterTree_Clone(op->ft));
	}

	// switch to expand into
	if(opBase->type == OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO) {
		CondVarLenTraverseOp_ExpandInto((CondVarLenTraverse*) clone);
	}

	return clone;
}

static void CondVarLenTraverseFree
(
	OpBase *ctx
) {
	CondVarLenTraverse *op = (CondVarLenTraverse *)ctx;

	if(op->edgeRelationTypes) {
		array_free(op->edgeRelationTypes);
		op->edgeRelationTypes = NULL;
	}

	if(op->ae) {
		AlgebraicExpression_Free(op->ae);
		op->ae = NULL;
	}

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}

	if(op->collect_paths) {
		if(op->allPathsCtx) {
			AllPathsCtx_Free(op->allPathsCtx);
			op->allPathsCtx = NULL;
		}
	} else {
		if(op->allNeighborsCtx) {
			AllNeighborsCtx_Free(op->allNeighborsCtx);
			op->allNeighborsCtx = NULL;
		}
	}

	if(op->ft) {
		FilterTree_Free(op->ft);
		op->ft = NULL;
	}
}

