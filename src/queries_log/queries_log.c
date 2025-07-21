/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "queries_log.h"
#include "util/rmalloc.h"
#include "configuration/config.h"

#include <pthread.h>
#include <stdatomic.h>

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
	uint32_t cap;
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
    QueriesLog log,               // queries log
	uint64_t received,            // query received timestamp
	double wait_duration,         // waiting time
	double execution_duration,    // executing time
	double report_duration,       // reporting time
	bool parameterized,           // uses parameters
	bool utilized_cache,          // utilized cache
	bool write,    	   	          // write query
	bool timeout,    		      // timeout query
	const char *query             // query string
) {
	// add query stats to buffer
	// acquire READ lock, multiple threads can be populating the circular buffer
	// simultaneously (the circular-buffer is lock-free)

	int res = pthread_rwlock_rdlock(&log->rwlock);
	ASSERT(res == 0);

	// get a slot within log's buffer
	void *slot = CircularBuffer_Reserve(log->queries);

	// dump query to slot
	LoggedQuery *q = (LoggedQuery*)(slot);

	if(q->query != NULL) {
		rm_free(q->query);
	}

	q->received           = received;
	q->wait_duration      = wait_duration;
	q->execution_duration = execution_duration;
	q->report_duration    = report_duration;
	q->parameterized      = parameterized;
	q->write              = write;
	q->timeout            = timeout;
	q->utilized_cache     = utilized_cache;
	q->query              = rm_strdup(query);

	res = pthread_rwlock_unlock(&log->rwlock);
	ASSERT(res == 0);
}

// returns number of queries in log
uint64_t QueriesLog_GetQueriesCount
(
	QueriesLog log  // queries log
) {
	// acquire READ lock to buffer
	pthread_rwlock_rdlock(&log->rwlock);

	// there's no harm in returning a lower count than actual
	// inf favour of performance
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

// free the QueriesLog structure's content
void QueriesLog_Free
(
	QueriesLog log  // queries log
) {
	ASSERT(log != NULL);

	LoggedQuery *q = NULL;
	CircularBuffer_ResetReader(log->queries);

	while((q = CircularBuffer_Read(log->queries, NULL)) != NULL) {
		// clean up
		if(q->query != NULL) {
			rm_free(q->query);
		}
	}

	CircularBuffer_Free(log->swap);
	CircularBuffer_Free(log->queries);

	pthread_rwlock_destroy(&log->rwlock);

	rm_free(log);
}

