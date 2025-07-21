/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "op_argument.h"
#include "op_argument_list.h"

// The Call {} operation is used to embed a subquery in the
// execution-plan. It generally passes records from its first child (lhs), to
// the body (rhs) of the subquery via the Argument\ArgumentList operations,
// and then merges the records returned from the body with the records consumed
// from lhs, according to whether the subquery is returning or not (i.e,
// terminates with a `RETURN` clause). If there is no lhs, a dummy record
// is created and passed to the body.
// The Call {} operation is eager\non-eager according to whether its body
// is\isn't eager (non-eager -> Arguments, eager -> ArgumentLists).

typedef enum {
	FEEDER_NONE,          // non-initialized
	FEEDER_ARGUMENT,      // Arguments
	FEEDER_ARGUMENT_LIST  // ArgumentLists
} FeederType;

typedef struct Feeder{
	union {
		Argument **arguments;
		ArgumentList **argumentLists;
	};
	FeederType type;
} Feeder;

typedef struct {
	OpBase op;

	bool first;         // is this the first call to consume
	bool is_eager;      // is the op eager
	bool is_returning;  // is the subquery returning or not
	OpBase *body;       // first op in the embedded execution-plan
	OpBase *lhs;        // op from which records are pulled
	Record r;           // current record consumed from lhs
	Record *records;    // records aggregated by the operation
	Feeder feeders;     // feeders to the body (Args/ArgLists)
} OpCallSubquery;

// creates a new CallSubquery operation
OpBase *NewCallSubqueryOp
(
	const ExecutionPlan *plan,  // execution plan
	bool is_eager,              // if an updating clause lies in the body, eagerly consume the records
	bool is_returning           // is the subquery returning or unit
);
