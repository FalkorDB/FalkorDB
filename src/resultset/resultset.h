/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "resultset_statistics.h"
#include "../redismodule.h"
#include "../execution_plan/record.h"
#include "rax.h"
#include "./formatters/resultset_formatters.h"

#define RESULTSET_OK 1
#define RESULTSET_FULL 0

typedef struct {
	RedisModuleCtx *ctx;            // redis context
	GraphContext *gc;               // context used for mapping attribute strings and IDs
	uint column_count;              // number of columns in result set
	const char **columns;           // field names for each column of results
	uint *columns_record_map;       // mapping between column name and record index
	DataBlock *cells;               // accumulated cells
	double timer[2];                // query runtime tracker
	ResultSetStatistics stats;      // result set statistics
	ResultSetFormatterType format;  // result set format; compact/verbose/nop
	ResultSetFormatter *formatter;  // result set data formatter
	SIAllocation cells_allocation;  // encountered values allocation
} ResultSet;

// map each column to a record index
// such that when resolving resultset row i column j we'll extract
// data from record at position columns_record_map[j]
void ResultSet_MapProjection
(
	ResultSet *set,  // resultset to init mappings for
	rax *mapping     // mapping
);

// create a new result set
ResultSet *NewResultSet
(
	RedisModuleCtx *ctx,
	ResultSetFormatterType format  // resultset format
);

// returns number of rows in result-set
uint64_t ResultSet_RowCount
(
	const ResultSet *set  // resultset to inquery
);

// add a new row to resultset
int ResultSet_AddRecord
(
	ResultSet *set,  // resultset to extend
	Record r         // record containing projected data
);

// update resultset index creation statistics
void ResultSet_IndexCreated
(
	ResultSet *set,  // resultset to update
	int status_code  // index creation status code
);

// update resultset index deleted statistics
void ResultSet_IndexDeleted
(
	ResultSet *set,  // resultset to update
	int status_code  // index deletion status code
);

// update resultset constraint creation statistics
void ResultSet_ConstraintCreated
(
	ResultSet *set,  // resultset to update
	int status_code  // constraint creation status code
);

// update resultset constraint deleted statistics
void ResultSet_ConstraintDeleted
(
	ResultSet *set,  // resultset to update
	int status_code  // constraint deletion status code
);

// update resultset cache execution statistics
void ResultSet_CachedExecution
(
	ResultSet *set  // resultset to update
);

// flush resultset to network
void ResultSet_Reply
(
	ResultSet *set  // resultset to reply with
);

// clear result set stats
void ResultSet_Clear
(
	ResultSet *set
);

// free resultset
void ResultSet_Free
(
	ResultSet *set  // resultset to free
);
