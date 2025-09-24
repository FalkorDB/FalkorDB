/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

/* Optional is an operation that manages the output of its child op tree rather
 * than producing or mutating data in itself.
 *
 * It attempts to pull a Record from its child op tree, which is typically
 * comprised of scan and traversal operations that model a MATCH pattern in a query.
 *
 * If the child successfully produces data, all of its Records are passed to Optional's
 * parent without modification.
 * If the child fails to produce any data, Optional will instead emit a single empty Record.
 *
 * This allows for the expression of complex patterns in which not all elements need to be resolved,
 * rather than the all-or-nothing matching logic traditionally used to resolve MATCH patterns.
 *
 * For example, the query:
 * MATCH (a:A) OPTIONAL MATCH (a)-[e]->(b) RETURN a, e, b
 * Produces the operation tree:
 *                   Results
 *                      |
 *                   Project
 *                      |
 *                    Apply
 *                   /     \
 *       LabelScan (a)    Optional
 *                           \
 *                          CondTraverse (a)-[e]->(b)
 *                              \
 *                             Argument (a)
 *
 * We want to retrieve all 'a' nodes regardless of whether they have an outgoing edge.
 * For each 'a' that does have at least one outgoing edge, all matching paths
 * will be resolved by Optional's CondTraverse child and returned as they would
 * be for a standard MATCH pattern.
 * For each 'a' that lacks an outgoing edge, the CondTraverse fails to produce any data,
 * and Optional instead emits one empty Record.
 * Optional's parent Apply op merges the empty Record with the left-hand Record
 * produced by the Label Scan, causing 'a' to be emitted unmodified with NULL entries
 * representing the elements of the failed traversal. */
typedef struct {
	OpBase op;
	bool emitted_record;     // True if this operation has returned at least one Record.
} Optional;

// creates a new Optional operation
OpBase *NewOptionalOp(const ExecutionPlan *plan);
