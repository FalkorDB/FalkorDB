/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "globals.h"
#include "cron/cron.h"
#include "redismodule.h"
#include "../../util/rmalloc.h"
#include "graph/graphcontext.h"
#include "configuration/config.h"
#include "util/circular_buffer.h"
#include "stream_finished_queries.h"

// event fields count
#define FLD_COUNT 10

// field names
#define FLD_WRITE                    "Write"
#define FLD_TIMEOUT                  "Timeout"
#define FLD_NAME_QUERY               "Query"
#define FLD_NAME_QUERY_PARAMS        "Query parameters"
#define FLD_NAME_UTILIZED_CACHE      "Utilized cache"
#define FLD_NAME_WAIT_DURATION       "Wait duration"
#define FLD_NAME_TOTAL_DURATION      "Total duration"
#define FLD_NAME_RECEIVED_TIMESTAMP  "Received at"
#define FLD_NAME_REPORT_DURATION     "Report duration"
#define FLD_NAME_EXECUTION_DURATION  "Execution duration"

#define TELEMETRY_STREAM_TTL 1000 * 60 * 30  // stream TTL ms, 30 minutes

// event field:value pairs
static RedisModuleString *_event[FLD_COUNT * 2] = {0};

// initialize '_event' template
// this function should be called only once
static void _initEventTemplate
(
	RedisModuleCtx *ctx  // redis module context
) {
	ASSERT(ctx != NULL);
	ASSERT(_event[0] == NULL);

	//--------------------------------------------------------------------------
	// create field names
	//--------------------------------------------------------------------------

	_event[0] = RedisModule_CreateString(
					ctx,
					FLD_NAME_RECEIVED_TIMESTAMP,
					strlen(FLD_NAME_RECEIVED_TIMESTAMP)
				);

	_event[2]  = RedisModule_CreateString(
					ctx,
					FLD_NAME_QUERY,
					strlen(FLD_NAME_QUERY)
				 );

	_event[4]  = RedisModule_CreateString(
					ctx,
					FLD_NAME_QUERY_PARAMS,
					strlen(FLD_NAME_QUERY_PARAMS)
				 );

	_event[6]  = RedisModule_CreateString(
					ctx,
					FLD_NAME_TOTAL_DURATION,
					strlen(FLD_NAME_TOTAL_DURATION)
				 );

	_event[8] = RedisModule_CreateString(
					ctx,
					FLD_NAME_WAIT_DURATION,
					strlen(FLD_NAME_WAIT_DURATION)
				 );

	_event[10] = RedisModule_CreateString(
					ctx,
					FLD_NAME_EXECUTION_DURATION,
					strlen(FLD_NAME_EXECUTION_DURATION)
				 );

	_event[12] = RedisModule_CreateString(
					ctx,
					FLD_NAME_REPORT_DURATION,
					strlen(FLD_NAME_REPORT_DURATION)
				 );

	_event[14] = RedisModule_CreateString(
					ctx,
					FLD_NAME_UTILIZED_CACHE,
					strlen(FLD_NAME_UTILIZED_CACHE)
				 );

	_event[16] = RedisModule_CreateString(
					ctx,
					FLD_WRITE,
					strlen(FLD_WRITE)
				 );

	_event[18] = RedisModule_CreateString(
					ctx,
					FLD_TIMEOUT,
					strlen(FLD_TIMEOUT)
				 );
}

// populate event
// sets event values
static void _populateEvent
(
	RedisModuleCtx *ctx,  // redis module context
	const LoggedQuery *q  // query information
) {
	int l = 0;
	char buff[512] = {0};

	const double total_duration = q->wait_duration        +
									q->execution_duration +
									q->report_duration;

	// FLD_NAME_RECEIVED_TIMESTAMP
	_event[1] = RedisModule_CreateStringFromLongLong(ctx, q->received);

	// FLD_NAME_QUERY
	_event[3] = RedisModule_CreateString(ctx, q->query, strlen(q->query));

	// FLD_NAME_QUERY_PARAMS
	if (q->params != NULL) {
		_event[5] = RedisModule_CreateString (ctx, q->params, strlen (q->params)) ;
	} else {
		_event[5] = RedisModule_CreateString (ctx, "", 0) ;
	}

	// FLD_NAME_TOTAL_DURATION
	l = sprintf(buff, "%.6f", total_duration);
	_event[7] = RedisModule_CreateString(ctx, buff, l);

	// FLD_NAME_WAIT_DURATION
	l = sprintf(buff, "%.6f", q->wait_duration);
	_event[9] = RedisModule_CreateString(ctx, buff, l);

	// FLD_NAME_EXECUTION_DURATION
	l = sprintf(buff, "%.6f", q->execution_duration);
	_event[11] = RedisModule_CreateString(ctx, buff, l);

	// FLD_NAME_REPORT_DURATION
	l = sprintf(buff, "%.6f", q->report_duration);
	_event[13] = RedisModule_CreateString(ctx, buff, l);

	// FLD_NAME_UTILIZED_CACHE
	_event[15] = RedisModule_CreateStringFromLongLong(ctx, q->utilized_cache);

	// FLD_WRITE
	_event[17] = RedisModule_CreateStringFromLongLong(ctx, q->write);

	// FLD_TIMEOUT
	_event[19] = RedisModule_CreateStringFromLongLong(ctx, q->timeout);
}

// free event values
static void _clearEvent
(
	RedisModuleCtx *ctx  // redis module context
) {
	if(unlikely(_event[1] == NULL)) return;

	for(int i = 1; i < FLD_COUNT * 2; i += 2) {
		RedisModule_FreeString(ctx, _event[i]);
	}
}

// add queries to stream
static void _stream_queries
(
	RedisModuleCtx *ctx,    // redis module context
	RedisModuleKey *key,    // stream key
	CircularBuffer queries  // queries to stream
) {
	LoggedQuery q ;

	while (CircularBuffer_Read (queries, &q)) {
		_populateEvent (ctx, &q) ;

		RedisModule_StreamAdd (key, REDISMODULE_STREAM_ADD_AUTOID, NULL,
				_event, FLD_COUNT) ;

		// clean up
		LoggedQuery_Free (&q) ;
		_clearEvent (ctx) ;
	}
}

void *CronTask_newStreamFinishedQueries
(
	void *pdata  // task context
) {
	ASSERT(pdata != NULL);
	StreamFinishedQueryCtx *ctx = (StreamFinishedQueryCtx*)pdata;

	// create private data for next invocation
	StreamFinishedQueryCtx *new_ctx = rm_malloc(sizeof(StreamFinishedQueryCtx));

	// set next iteration graph index
	new_ctx->graph_idx = ctx->graph_idx;

	return new_ctx;
}

// cron task
// stream finished queries for each graph in the keyspace
bool CronTask_streamFinishedQueries
(
	void *pdata  // task context
) {
	StreamFinishedQueryCtx *ctx    = (StreamFinishedQueryCtx*)pdata;
	RedisModuleCtx         *rm_ctx = RedisModule_GetThreadSafeContext(NULL);

	// initialize stream event template
	if(unlikely(_event[0] == NULL)) {
		_initEventTemplate(rm_ctx);
	}

	// start stopwatch
	double deadline = 3;  // 3ms
	simple_timer_t stopwatch;
	simple_tic(stopwatch);

	uint64_t max_query_count = 0;  // determine max number of queries to collect
	Config_Option_get(Config_CMD_INFO_MAX_QUERY_COUNT, &max_query_count);

	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);

	// pick up from where we've left
	GraphIterator_Seek(&it, ctx->graph_idx);

	GraphContext *gc = NULL;

	// as long as we've got processing time
	bool gil_acquired = false;
	while(TIMER_GET_ELAPSED_MILLISECONDS(stopwatch) < deadline) {
		// get next graph to populate
		QueriesLog queries_log = NULL;
		while((gc = GraphIterator_Next(&it)) != NULL) {
			ctx->graph_idx++;  // prepare next iteration
			if(QueriesLog_GetQueriesCount(gc->queries_log) > 0) {
				queries_log = gc->queries_log;
				break;
			}
			GraphContext_DecreaseRefCount(gc);
		}

		// iterator depleted
		if((gc) == NULL) {
			break;
		}

		//----------------------------------------------------------------------
		// try to acquire GIL
		//----------------------------------------------------------------------

		while(!gil_acquired && TIMER_GET_ELAPSED_MILLISECONDS(stopwatch) < deadline) {
			gil_acquired =
				RedisModule_ThreadSafeContextTryLock(rm_ctx) == REDISMODULE_OK;
		}

		if(!gil_acquired) {
			GraphContext_DecreaseRefCount(gc);
			break;
		}

		CircularBuffer queries = QueriesLog_ResetQueries(queries_log);

		//----------------------------------------------------------------------
		// stream queries
		//----------------------------------------------------------------------

		RedisModuleString *keyname =
			(RedisModuleString*)GraphContext_GetTelemetryStreamName(gc);

		RedisModuleKey *key = RedisModule_OpenKey(rm_ctx, keyname,
			REDISMODULE_WRITE);

		// make sure key is of type stream
		int key_type = RedisModule_KeyType(key);
		if(key_type == REDISMODULE_KEYTYPE_STREAM ||
			key_type == REDISMODULE_KEYTYPE_EMPTY) {
			// add queries to stream
			_stream_queries (rm_ctx, key, queries) ;

			// cap stream
			RedisModule_StreamTrimByLength (key,
					REDISMODULE_STREAM_TRIM_APPROX, max_query_count) ;
		} else {
			// TODO: decide how to handle this...
		}

		// clean up
		RedisModule_CloseKey(key);

		GraphContext_DecreaseRefCount(gc);
	}

	if(gil_acquired) {
		RedisModule_ThreadSafeContextUnlock(rm_ctx);
	}

	RedisModule_FreeThreadSafeContext(rm_ctx);

	// set next iteration graph index
	ctx->graph_idx = (gc == NULL) ? 0 : ctx->graph_idx;

	return (gc != NULL);
}

