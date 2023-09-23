/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./optimizer.h"
#include "./optimizations.h"

void optimizePlan(ExecutionPlan *plan) {
	// tries to compact filter trees, and remove redundant filters
	// [COMPILE TIME]
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

	// remove redundant SCAN operations
	// [COMPILE TIME]
	reduceScans(plan);

	// when possible, replace label scan and filter ops with index scans
	utilizeIndices(plan);

	// scan label with least entities
	optimizeLabelScan(plan);

	// try to reduce SCAN + FILTER to a node seek operation
	seekByID(plan);

	// migrate filters on variable-length edges into the traversal operations
	// [COMPILE TIME]
	filterVariableLengthEdges(plan);

	// try to optimize cartesian product
	// [COMPILE TIME]
	reduceCartesianProductStreamCount(plan);

	// try to match disjoint entities by applying a join
	// [COMPILE TIME]
	applyJoin(plan);

	// try to reduce a number of filters into a single filter op
	reduceFilters(plan);

	// reduce traversals where both src and dest nodes are already resolved
	// into an expand into operation
	reduceTraversal(plan);

	// try to reduce distinct if it follows aggregation
	// [COMPILE TIME]
	reduceDistinct(plan);

	// try to reduce execution plan incase it perform node or edge counting
	// [COMPILE TIME]
	reduceCount(plan);

	// let operations know about specified limit(s)
	applyLimit(plan);

	// let operations know about specified skip(s)
	// [COMPILE TIME]
	applySkip(plan);
}

