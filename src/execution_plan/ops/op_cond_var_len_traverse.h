/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../algorithms/algorithms.h"
#include "../../arithmetic/algebraic_expression.h"

/* OP Traverse */
typedef struct {
	OpBase op;
	Graph *g;
	Record r;
	RG_Matrix M;                           /* Traversed matrix if using the SimpleConsume routine. */
	int edgesIdx;                          /* Edges set by operation. */
	int srcNodeIdx;                        /* Node set by operation. */
	int destNodeIdx;                       /* Node set by operation. */
	bool expandInto;                       /* Both src and dest already resolved. */
	FT_FilterNode *ft;                     /* If not NULL, FilterTree applied to traversed edge. */
	bool shortestPaths;                    /* Only collect shortest paths. */
	unsigned int minHops;                  /* Maximum number of hops to perform. */
	unsigned int maxHops;                  /* Maximum number of hops to perform. */
	int edgeRelationCount;                 /* Length of edgeRelationTypes. */
	int *edgeRelationTypes;                /* Relation(s) we're traversing. */
	AlgebraicExpression *ae;               /* ArithmeticExpression describing op's traversal pattern. */
	union {
		AllPathsCtx *allPathsCtx;          /* Context for collecting all paths. */
		AllNeighborsCtx *allNeighborsCtx;  /* Context for collecting all neighbors . */
	};
	bool collect_paths;                    /* Whether we must populate the entire path. */
	GRAPH_EDGE_DIR traverseDir;            /* Traverse direction. */
} CondVarLenTraverse;

OpBase *NewCondVarLenTraverseOp(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae);

/* Transform operation from Conditional Variable Length Traverse
 * to Expand Into Conditional Variable Length Traverse */
void CondVarLenTraverseOp_ExpandInto(CondVarLenTraverse *op);

// Set the FilterTree pointer of a CondVarLenTraverse operation.
void CondVarLenTraverseOp_SetFilter(CondVarLenTraverse *op, FT_FilterNode *ft);

