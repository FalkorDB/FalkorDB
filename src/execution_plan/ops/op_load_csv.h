/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// load CSV
typedef struct {
	OpBase op;            // op base must be the first field in this struct
	AR_ExpNode *exp;      // expression evaluated to CSV path
	char *path;           // CSV path
	char *alias;          // CSV row alias
	int recIdx;           // record index to populate with CSV row
	OpBase *child;        // child operation
	Record child_record;  // child record
} OpLoadCSV;

// creates a new load CSV operation
OpBase *NewLoadCSVOp
(
	const ExecutionPlan *plan,  // execution plan
	AR_ExpNode *exp,            // CSV URI path expression
	const char *alias           // CSV row alias
	bool with_headers           // CSV contains header row
);

