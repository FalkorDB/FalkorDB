/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include <stdio.h>
#include <float.h>
#include <unistd.h>
#include <pthread.h>

#include "./slow_log.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/strutil.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pool.h"

#define SLOW_LOG_STR_MAX_LEN     2048  // string max len
#define SLOW_LOG_MIN_REQ_LATENCY 10    // 10ms

#define SLOW_LOG_SIZE 10

// slowlog item
typedef struct {
	char *cmd;          // command
	time_t time;        // item creation time
	char *query;        // query
	char *params;       // query parameters
	double latency;     // how much time query was processed
	XXH64_hash_t hash;  // item hash
} SlowLogItem ;

// slowlog, maintains N slowest queries
struct SlowLog {
	uint count;                         // number of items <= SLOW_LOG_SIZE
	double min_latency;                 // lowest stored latency
	pthread_mutex_t lock;               // lock
	SlowLogItem items[SLOW_LOG_SIZE] ;  // entries
} ;

// redis prints doubles with up to 17 digits of precision, which captures
// the inaccuracy of many floating-point numbers (such as 0.1)
// By using the %g format and a precision of 5 significant digits, we avoid many
// awkward representations like RETURN 0.1 emitting "0.10000000000000001",
// though we're still subject to many of the typical issues with floating-point error
static inline void _ReplyWithRoundedDouble
(
	RedisModuleCtx *ctx,
	double d
) {
	// get length required to print number
	int len = snprintf (NULL, 0, "%.5g", d) ;
	char str[len + 1] ;
	sprintf (str, "%.5g", d) ;
	// output string-formatted number
	RedisModule_ReplyWithStringBuffer (ctx, str, len) ;
}

// compute (cmd, query) hash
static inline XXH64_hash_t _compute_key
(
	const char *cmd,   // command
	const char *query  // query
) {
	ASSERT (cmd   != NULL) ;
	ASSERT (query != NULL) ;

	XXH64_state_t state ;
	XXH_errorcode res = XXH64_reset (&state, 0) ;
	ASSERT (res != XXH_ERROR) ;

	XXH64_update (&state, cmd, strlen (cmd)) ;
	XXH64_update (&state, query, strnlen (query, SLOW_LOG_STR_MAX_LEN)) ;

	return XXH64_digest (&state) ;
}

// create a new slowlog item
static void _SlowLogItem_New
(
	SlowLogItem *item,  // item to populate
	const char *cmd,    // command
	char **query,       // query
	char **params,      // params byte length
	double latency,     // query latency
	time_t t            // query receive time
) {
	ASSERT (cmd     != NULL) ;
	ASSERT (item    != NULL) ;
	ASSERT (params  != NULL) ;
	ASSERT (query   != NULL && *query != NULL) ;
	ASSERT (latency >= SLOW_LOG_MIN_REQ_LATENCY) ;

	char *_query  = *query  ;
	char *_params = *params ;

	// assert query and params are capped
	ASSERT (strlen (_query) <= SLOW_LOG_STR_MAX_LEN + 3) ;
	ASSERT (_params == NULL || strlen (_params) <= SLOW_LOG_STR_MAX_LEN + 3) ;

	item->cmd     = rm_strdup (cmd) ;
	item->time    = t ;
	item->hash    = _compute_key (item->cmd, _query) ;
	item->query   = _query ;
	item->params  = _params ;
	item->latency = latency ;

	// own query and params
	*query  = NULL ;
	*params = NULL ;
}

static void _SlowLog_Item_Free
(
	SlowLogItem *item
) {
	ASSERT (item        != NULL) ;
	ASSERT (item->cmd   != NULL) ;
	ASSERT (item->query != NULL) ;

	if (item->params != NULL) {
		free (item->params) ;
	}

	free    (item->query)  ;
	rm_free (item->cmd)    ;
}

static void _SlowLogItem_Update
(
	SlowLogItem *item,  // item to update
	char **params,      // params
	double latency,     // query latency
	time_t t            // query receive time
) {
	ASSERT (item        != NULL) ;
	ASSERT (params      != NULL) ;
	ASSERT (item->query != NULL) ;
	ASSERT (latency >= SLOW_LOG_MIN_REQ_LATENCY) ;

	char *_params = *params ;
	// assert params are capped
	ASSERT (_params == NULL || strlen (_params) <= SLOW_LOG_STR_MAX_LEN + 3) ;

	// free old params
	if (item->params != NULL) {
		free (item->params) ;
	}

	item->time    = t ;
	item->params  = _params ;
	item->latency = latency ;

	*params = NULL ;
}

// lookup entry
static bool _SlowLog_Contains
(
	SlowLog *slowlog,   // slowlog
	const char *cmd,    // command
	const char *query,  // query
	XXH64_hash_t *key,  // [output] item key
	SlowLogItem **item  // [output] located entry
) {

	*key = _compute_key (cmd, query) ;

	for (int i = 0; i < slowlog->count; i++) {
		if (slowlog->items[i].hash == *key) {
			*item = slowlog->items + i ;
			return true ;
		}
	}

	return false ;
}

// create a new slowlog
SlowLog *SlowLog_New (void) {
	SlowLog *slowlog = rm_calloc (1, sizeof (SlowLog)) ;

	int res = pthread_mutex_init (&slowlog->lock, NULL) ;
	ASSERT (res == 0) ;

	return slowlog ;
}

// introduce item to slow log
void SlowLog_Add
(
	SlowLog *slowlog,   // slowlog to add entry to
	const char *cmd,    // command
	const char *query,  // query being logged
	int params_len,     // params byte length
	double latency,     // command latency
	uint64_t time       // seconds since UNIX epoch
) {
	ASSERT (latency  >= 0) ;
	ASSERT (cmd      != NULL) ;
	ASSERT (query    != NULL) ;
	ASSERT (slowlog  != NULL) ;

	double min_latency = DBL_MAX ;

	// slow enough ?
	if (likely (latency < SLOW_LOG_MIN_REQ_LATENCY)) {
		return ;
	}

	bool add = (slowlog->min_latency < latency ||
				slowlog->count < SLOW_LOG_SIZE) ;

	if (likely (!add)) {
		return ;
	}

	//--------------------------------------------------------------------------
	// lock slowlog
	//--------------------------------------------------------------------------

	if (pthread_mutex_lock (&slowlog->lock) != 0) {
		// failed to lock, skip logging
		return ;
	}

	// recheck under lock
	add = (slowlog->count < SLOW_LOG_SIZE || slowlog->min_latency < latency) ;
	if (!add) {
		goto unlock ;
	}

	// check if we query is alreay in log?
	XXH64_hash_t key ;
	SlowLogItem *existing_item = NULL;

	//--------------------------------------------------------------------------
	// update existing item
	//--------------------------------------------------------------------------

	bool exists = _SlowLog_Contains (slowlog, cmd, query + params_len, &key,
			&existing_item) ;
	if (exists && latency <= existing_item->latency) {
			goto unlock ;
	}

	//--------------------------------------------------------------------------
	// truncate query & parameters
	//--------------------------------------------------------------------------

	char *truncated_query  = NULL ;
	char *truncated_params = NULL ;

	if (params_len > 0) {
		str_truncate (&truncated_params, query, params_len, SLOW_LOG_STR_MAX_LEN) ;
	}

	if (existing_item != NULL) {
		_SlowLogItem_Update (existing_item, &truncated_params, latency, time) ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// add a new item
	//--------------------------------------------------------------------------
	size_t n = strnlen (query + params_len, SLOW_LOG_STR_MAX_LEN) ;
	str_truncate (&truncated_query, query + params_len, n, SLOW_LOG_STR_MAX_LEN) ;

	if (slowlog->count < SLOW_LOG_SIZE) {
		_SlowLogItem_New (slowlog->items + slowlog->count, cmd,
				&truncated_query, &truncated_params, latency, time) ;

		slowlog->count++ ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// replace fastest item
	//--------------------------------------------------------------------------

	// locate fastest item
	int idx = -1 ;
	for (int i = 0; i < slowlog->count; i++) {
		if (slowlog->items[i].latency == slowlog->min_latency) {
			idx = i ;
			break ;
		}
	}
	ASSERT (idx != -1) ;

	// remove fastest
	_SlowLog_Item_Free (slowlog->items + idx) ;

	// add new
	_SlowLogItem_New (slowlog->items + idx, cmd, &truncated_query,
			&truncated_params, latency, time) ;

cleanup:
	// update min latency
	for (int i = 0; i < slowlog->count; i++) {
		min_latency = MIN (min_latency, slowlog->items[i].latency) ;
	}
	slowlog->min_latency = min_latency ;

unlock:
	pthread_mutex_unlock (&slowlog->lock) ;
}

// clear all entries from slowlog
void SlowLog_Clear
(
	SlowLog *slowlog  // slowlog to clear
) {
	ASSERT (slowlog != NULL) ;

	// lock slowlog
	if (pthread_mutex_lock (&slowlog->lock) != 0) {
		// failed to lock, skip logging
		return ;
	}

	// free items
	for(int i = 0; i < slowlog->count; i++) {
		_SlowLog_Item_Free (slowlog->items + i) ;
	}
	memset (slowlog->items, 0, sizeof (SlowLogItem) * SLOW_LOG_SIZE) ;

	slowlog->count       = 0 ;
	slowlog->min_latency = 0 ;

	// unlock
	pthread_mutex_unlock (&slowlog->lock) ;
}

// reports slowlog content
void SlowLog_Replay
(
	SlowLog *slowlog,    // slowlog
	RedisModuleCtx *ctx  // redis module context
) {
	// enter critical section
	bool locked = pthread_mutex_lock (&slowlog->lock) ;
	ASSERT (locked == 0) ;

	RedisModule_ReplyWithArray (ctx, slowlog->count) ;

	for (int i = 0; i < slowlog->count; i++) {
		SlowLogItem *item = slowlog->items + i ;

		RedisModule_ReplyWithArray (ctx, 5) ;

		RedisModule_ReplyWithDouble (ctx, item->time) ;

		RedisModule_ReplyWithStringBuffer (ctx, (const char *)item->cmd,
				strlen (item->cmd)) ;

		RedisModule_ReplyWithStringBuffer (ctx, (const char *)item->query,
				strlen (item->query)) ;

		if (item->params != NULL) {
			RedisModule_ReplyWithStringBuffer (ctx, (const char *)item->params,
					strlen (item->params)) ;
		} else {
			RedisModule_ReplyWithNull (ctx) ;
		}

		_ReplyWithRoundedDouble (ctx, item->latency) ;
	}

	// exit critical section
	pthread_mutex_unlock (&slowlog->lock) ;
}

// free slowlog
void SlowLog_Free
(
	SlowLog *slowlog  // slowlog to free
) {
	for (int i = 0; i < slowlog->count; i++) {
		_SlowLog_Item_Free (slowlog->items + i) ;
	}

	int res = pthread_mutex_destroy (&slowlog->lock) ;
	ASSERT (res == 0) ;

	rm_free (slowlog) ;
}

