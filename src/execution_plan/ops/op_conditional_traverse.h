/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "shared/traverse_functions.h"
#include "../../graph/delta_matrix/delta_matrix_iter.h"
#include "../../arithmetic/algebraic_expression.h"
#include "../../../deps/GraphBLAS/Include/GraphBLAS.h"

// op traverse
typedef struct {
	OpBase op;
	Graph *graph;
	AlgebraicExpression *ae;
	Delta_Matrix F;              // filter matrix
	Delta_Matrix M;              // algebraic expression result
	EdgeTraverseCtx *edge_ctx;   // edge collection data if the edge needs to be set
	Delta_MatrixTupleIter iter;  // iterator over M
	bool partial_ae;             // algebraic expression missing some operands
	int srcNodeIdx;              // source node index into record
	int destNodeIdx;             // destination node index into record
	uint64_t record_count;       // number of held records
	uint64_t record_cap;         // max number of records to process
	Record *records;             // array of records
	Record r;                    // currently selected record

	// optional conditional traversal fields
	bool optional;               // true if traversal is optional
	Record *optional_records;    // optional records
	GrB_Vector w;                // track src node with neighbors
	GrB_Vector e;                // track src nodes without neighbors
	GrB_Scalar s;                // GraphBLAS true scalar

} OpCondTraverse;

// creates a new Traverse operation
OpBase *NewCondTraverseOp
(
	const ExecutionPlan *plan,
	Graph *g,
	AlgebraicExpression *ae
);

// make traversal optional
// OPTIONAL MATCH (a)-[]->(b)
// `a` will be passed onwards even if it doesn't have any neighbors
void CondTraverse_MakeOptional
(
	OpCondTraverse *op
);

