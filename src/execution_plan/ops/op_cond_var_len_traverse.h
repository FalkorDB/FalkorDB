/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../algorithms/algorithms.h"
#include "../../arithmetic/algebraic_expression.h"

// OP Traverse
typedef struct {
	OpBase op;
	Graph *g;
	Record r;
	RG_Matrix M;              // traversed matrix if using the SimpleConsume routine
	int edgesIdx;             // edges set by operation
	int srcNodeIdx;           // node set by operation
	int destNodeIdx;          // node set by operation
	bool expandInto;          // both src and dest already resolved
	FT_FilterNode *ft;        // if not NULL, FilterTree applied to traversed edge
	bool shortestPaths;       // only collect shortest paths
	unsigned int minHops;     // maximum number of hops to perform
	unsigned int maxHops;     // maximum number of hops to perform
	int edgeRelationCount;    // length of edgeRelationTypes
	int *edgeRelationTypes;   // relation(s) we're traversing
	AlgebraicExpression *ae;  // arithmeticExpression describing op's traversal pattern
	union {
		AllPathsCtx *allPathsCtx;          // context for collecting all paths
		AllNeighborsCtx *allNeighborsCtx;  // context for collecting all neighbors
	};
	bool collect_paths;                    // whether we must populate the entire path
	GRAPH_EDGE_DIR traverseDir;            // traverse direction
} CondVarLenTraverse;

OpBase *NewCondVarLenTraverseOp
(
	ExecutionPlan *plan,
	Graph *g,
	AlgebraicExpression *ae
);

// transform operation from Conditional Variable Length Traverse
// to Expand Into Conditional Variable Length Traverse
void CondVarLenTraverseOp_ExpandInto
(
	CondVarLenTraverse *op
);

// set the FilterTree pointer of a CondVarLenTraverse operation
void CondVarLenTraverseOp_SetFilter
(
	CondVarLenTraverse *op,
	FT_FilterNode *ft
);

