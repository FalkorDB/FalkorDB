/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "./optimizer.h"
#include "./optimizations.h"

void optimizePlan(ExecutionPlan *plan) {
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

	// remove redundant SCAN operations
	reduceScans(plan);

	// when possible, replace label scan and filter ops with index scans
	utilizeIndices(plan);

	// scan label with least entities
	optimizeLabelScan(plan);

	// try to reduce SCAN + FILTER to a node seek operation
	seekByID(plan);

	// migrate filters on variable-length edges into the traversal operations
	filterVariableLengthEdges(plan);

	// try to optimize cartesian product
	reduceCartesianProductStreamCount(plan);

	// try to match disjoint entities by applying a join
	applyJoin(plan);

	// try to reduce a number of filters into a single filter op
	reduceFilters(plan);

	// reduce traversals where both src and dest nodes are already resolved
	// into an expand into operation
	reduceTraversal(plan);

	// try to reduce distinct if it follows aggregation
	reduceDistinct(plan);

	// try to reduce execution plan incase it perform node or edge counting
	reduceCount(plan);

	// let operations know about specified limit(s)
	applyLimit(plan);

	// let operations know about specified skip(s)
	applySkip(plan);
}

