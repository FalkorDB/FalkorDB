/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./optimizer.h"
#include "./optimizations.h"

// apply compile time optimizations
void Optimizer_CompileTimeOptimize
(
	ExecutionPlan *plan  // plan to optimize
) {
	// remove redundant SCAN operations
	reduceScans(plan);

	// tries to compact filter trees, and remove redundant filters
	compactFilters(plan);

	// scan optimizations order:
	// 1. remove redundant scans which checks for the same node
	// 2. try to use the indices
	//    given a label scan and an indexed property, apply index scan
	// 3. given a filter which checks id condition, and full or label scan
	//    reduce it to id scan or label with id scan
	//    note: due to the scan optimization order
	//          label scan will be replaced with index scan when possible
	//          so the id filter remains

	// migrate filters on variable-length edges into the traversal operations
	filterVariableLengthEdges(plan);

	// try to optimize cartesian product
	reduceCartesianProductStreamCount(plan);

	// try to match disjoint entities by applying a join
	applyJoin(plan);

	// reduce traversals where both src and dest nodes are already resolved
	// into an expand into operation
	reduceTraversal(plan);

	// try to reduce distinct if it follows aggregation
	reduceDistinct(plan);

	// try to reduce execution plan incase it perform node or edge counting
	reduceCount(plan);
}

// apply runtime optimizations
void Optimizer_RuntimeOptimize
(
	ExecutionPlan *plan  // plan to optimize
) {
	// when possible, replace label scan and filter ops with index scans
	utilizeIndices(plan);

	// scan label with least entities
	optimizeLabelScan(plan);

	// try to reduce SCAN + FILTER to a node seek operation
	seekByID(plan);

	// try to reduce a number of filters into a single filter op
	reduceFilters(plan);

	// let operations know about specified limit(s)
	applyLimit(plan);

	// let operations know about specified skip(s)
	applySkip(plan);
}

