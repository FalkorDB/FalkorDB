/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/circular_buffer.h"

// query statistics
typedef struct QueryStats {
	uint64_t received;          // query received timestamp
	double wait_duration;       // waiting time
	double execution_duration;  // executing time
	double report_duration;     // reporting time
	bool parameterized;         // uses parameters
	bool utilized_cache;        // utilized cache
	bool write;    		        // write query
	bool timeout;    		    // timeout query
	char *params;               // query parameters
	char *query;                // query string
} LoggedQuery;

// forward declaration of opaque QueriesLog structure
typedef struct _QueriesLog *QueriesLog;

// create a new queries log structure
QueriesLog QueriesLog_New(void);

// add query to buffer
void QueriesLog_AddQuery
(
	QueriesLog log,             // queries log
	uint64_t received,          // query received timestamp
	double wait_duration,       // waiting time
	double execution_duration,  // executing time
	double report_duration,     // reporting time
	bool parameterized,         // uses parameters
	bool utilized_cache,        // utilized cache
	bool write,    		        // write query
	bool timeout,    		    // timeout query
	uint params_len,            // length of parameters
	const char *query           // query string
);

// returns number of queries in log
uint64_t QueriesLog_GetQueriesCount
(
	QueriesLog log  // queries log
);

// reset queries buffer
// returns queries buffer prior to reset
CircularBuffer QueriesLog_ResetQueries
(
	QueriesLog log  // queries log
);

// free a logged query
void LoggedQuery_Free
(
	LoggedQuery *q
) ;

// free the QueriesLog structure's content
void QueriesLog_Free
(
	QueriesLog log  // queries log
);

