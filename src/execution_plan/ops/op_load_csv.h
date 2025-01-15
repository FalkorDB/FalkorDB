/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../csv_reader/csv_reader.h"
#include "../../curl_session/curl_session.h"

// load CSV
typedef struct {
	OpBase op;            // op base must be the first field in this struct
	CurlSession curl;     // curl session
	CSVReader reader;     // CSV reader
	AR_ExpNode *exp;      // expression evaluated to CSV URI
	SIValue uri;          // CSV URI
	char *alias;          // CSV row alias
	int recIdx;           // record index to populate with CSV row
	bool with_headers;    // CSV contains header row
	OpBase *child;        // child operation
	Record child_record;  // child record
} OpLoadCSV;

// creates a new load CSV operation
OpBase *NewLoadCSVOp
(
	const ExecutionPlan *plan,  // execution plan
	AR_ExpNode *exp,            // CSV URI expression
	const char *alias,          // CSV row alias
	bool with_headers           // CSV contains header row
);

