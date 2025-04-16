/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// OpDistinct filters out non-unique records
// record uniqueness is determined by a set of expressions
// the operation computes hash based on the distinct expressions
// if the hash wasn't encountered the record will pass onward otherwise
// the record is dropped
//
// hash(record) = hash(record[expression]) for each distinct expression
//
// MATCH (n)
// RETURN DISTINCT n.first_name, n.last_name

#pragma once

#include "op.h"
#include "../../util/dict.h"
#include "../execution_plan.h"

typedef struct {
	OpBase op;
	dict *found;           // hashtable containing seen records
	int *offsets;          // offsets to expression values
	const char **aliases;  // expression aliases to distinct by
	uint offset_count;     // number of offsets

} OpDistinct;

// create a new distinct operation
OpBase *NewDistinctOp
(
	const ExecutionPlan *plan,  // execution plan
	const char **aliases,       // distinct aliases
	uint alias_count            // number of distinct expressions
);

