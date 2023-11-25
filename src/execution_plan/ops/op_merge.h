/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "op_argument.h"
#include "../execution_plan.h"
#include "shared/update_functions.h"
#include "../../resultset/resultset_statistics.h"

// the Merge operation accepts exactly one path in the query
// and attempts to match it
// if the path is not found, it will be created
// making new instances of every path element not bound in an earlier clause
// in the query
typedef struct {
	OpBase op;                  // base op
	Record r;                   // current record
	OpBase *match_stream;       // child stream that attempts to resolve the pattern
	OpBase *create_stream;      // child stream that will create the pattern if not found
	OpBase *bound_stream;       // child stream to resolve previously bound variables
	Record *matched_records;    // records which were matched
	Record *created_records;    // records which were created
	rax *on_match;              // updates to be performed on a successful match
	rax *on_create;             // updates to be performed on creation
	raxIterator on_match_it;    // iterator for traversing ON MATCH update contexts
	raxIterator on_create_it;   // iterator for traversing ON CREATE update contexts
	dict *node_pending_updates; // pending updates to apply, generated
	dict *edge_pending_updates; // pending updates to apply, generated
} OpMerge;

OpBase *NewMergeOp
(
	const ExecutionPlan *plan,
	rax *on_match,
	rax *on_create
);

