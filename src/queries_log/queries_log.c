/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "queries_log.h"
#include "util/rmalloc.h"
#include "util/strutil.h"
#include "configuration/config.h"

#include <pthread.h>
#include <stdatomic.h>

#define QUERIES_LOG_STR_MAX_LEN   2048  // string max len

// holds query statistics per graph
typedef struct QueriesCounters {
    _Atomic uint64_t ro_succeeded_n;     // # read-only queries succeeded
    _Atomic uint64_t write_succeeded_n;  // # write queries succeeded
    _Atomic uint64_t ro_failed_n;        // # RO queries failed
    _Atomic uint64_t write_failed_n;     // # write queries failed
    _Atomic uint64_t ro_timedout_n;      // # RO queries timed out
    _Atomic uint64_t write_timedout_n;   // # write queries timed out
} QueriesCounters;

// QueriesLog
// maintains a log of queries
typedef struct _QueriesLog {
	CircularBuffer queries;    // buffer
	CircularBuffer swap;       // swap buffer
	QueriesCounters counters;  // counters with states
	pthread_rwlock_t rwlock;   // RWLock
} _QueriesLog;

// create a new queries log structure
QueriesLog QueriesLog_New(void) {
	QueriesLog log = rm_calloc(1, sizeof(struct _QueriesLog));

	// initialize read/write lock
	int res = pthread_rwlock_init(&log->rwlock, NULL);
	ASSERT(res == 0);

	// create circular buffer
	// read buffer capacity from configuration
	uint64_t cap;
	bool get = Config_Option_get(Config_CMD_INFO_MAX_QUERY_COUNT, &cap);
	ASSERT(get == true);

	size_t item_size = sizeof(LoggedQuery);
	log->swap    = CircularBuffer_New(item_size, cap);
	log->queries = CircularBuffer_New(item_size, cap);

	return log;
}

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
	bool write,                 // write query
	bool timeout,               // timeout query
	uint params_len,            // length of parameters
	const char *query           // query string
) {
	ASSERT (query != NULL) ;

	// cap query lenght
	const char *_query = query + params_len ;

	char *truncated_query;
	size_t n = strnlen (_query, QUERIES_LOG_STR_MAX_LEN) ;
	str_truncate (&truncated_query, _query, n, QUERIES_LOG_STR_MAX_LEN) ;

	// cap parameters lenght
	char *truncated_params = NULL ;
	if (params_len > 0) {
		str_truncate (&truncated_params, query, params_len,
				QUERIES_LOG_STR_MAX_LEN) ;
	}

	//--------------------------------------------------------------------------
	// add query stats to buffer
	//--------------------------------------------------------------------------

	LoggedQuery q = {
		. received           = received,
		. wait_duration      = wait_duration,
		. execution_duration = execution_duration,
		. report_duration    = report_duration,
		. parameterized      = parameterized,
		. utilized_cache     = utilized_cache,
		. write              = write,
		. timeout            = timeout,
		. params             = truncated_params,
		. query              = truncated_query
	} ;

	// try adding query to buffer
	if (!CircularBuffer_Add (log->queries, &q)) {
		LoggedQuery_Free (&q) ;
	}
}

// returns number of queries in log
uint64_t QueriesLog_GetQueriesCount
(
	QueriesLog log  // queries log
) {
	// acquire READ lock to buffer
	pthread_rwlock_rdlock(&log->rwlock);

	// there's no harm in returning a lower count than actual
	// in favour of performance
	uint64_t n = CircularBuffer_ItemCount(log->queries);

	// release lock
	pthread_rwlock_unlock(&log->rwlock);

	return n;
}

// reset queries buffer
// returns queries buffer prior to reset
CircularBuffer QueriesLog_ResetQueries
(
	QueriesLog log  // queries log
) {
	ASSERT(log != NULL);

	//--------------------------------------------------------------------------
	// swap buffers
	//--------------------------------------------------------------------------

	// acquire WRITE lock, waiting for all readers to finish
	int res = pthread_rwlock_wrlock(&log->rwlock);
	ASSERT(res == 0);

	CircularBuffer prev = log->queries;
	log->queries = log->swap;
	log->swap = prev;

	// release lock
	res = pthread_rwlock_unlock(&log->rwlock);
	ASSERT(res == 0);

	return prev;
}

// free a logged query
void LoggedQuery_Free
(
	LoggedQuery *q
) {
	ASSERT (q != NULL) ;

	free (q->query) ;

	if (q->params != NULL) {
		free (q->params) ;
	}
}

// free the QueriesLog structure's content
void QueriesLog_Free
(
	QueriesLog log  // queries log
) {
	ASSERT (log != NULL) ;

	CircularBuffer_Free (log->swap,
			(CircularBuffer_ItemFreeCB) LoggedQuery_Free) ;

	CircularBuffer_Free (log->queries,
			(CircularBuffer_ItemFreeCB) LoggedQuery_Free) ;

	pthread_rwlock_destroy (&log->rwlock) ;

	rm_free (log) ;
}

