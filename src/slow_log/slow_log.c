/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <stdio.h>
#include <unistd.h>

#include "./slow_log.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pool.h"

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
	int len = snprintf(NULL, 0, "%.5g", d);
	char str[len + 1];
	sprintf(str, "%.5g", d);
	// output string-formatted number
	RedisModule_ReplyWithStringBuffer(ctx, str, len);
}

static SlowLogItem *_SlowLogItem_New
(
	const char *cmd,
	const char *query,
	double latency,
	time_t t
) {
	SlowLogItem *item = rm_malloc(sizeof(SlowLogItem));

	item->time    = t;
	item->cmd     = rm_strdup(cmd);
	item->query   = rm_strdup(query);
	item->latency = latency;

	return item;
}

static void _SlowLog_Item_Free
(
	SlowLogItem *item
) {
	ASSERT(item);

	rm_free(item->cmd);
	rm_free(item->query);
	rm_free(item);
}

// Compares two heap record nodes.
static int _slowlog_elem_compare
(
	const void *A,
	const void *B,
	void *udata
) {
	SlowLogItem *a = (SlowLogItem *)A;
	SlowLogItem *b = (SlowLogItem *)B;

	return b->latency - a->latency;
}

static inline size_t _compute_key
(
	char **s,
	const char *cmd,
	const char *query
) {
	return asprintf(s, "%s %s", cmd, query);
}

static size_t _SlowLogItem_ToString
(
	const SlowLogItem *item,
	char **s
) {
	ASSERT(item);
	return _compute_key(s, item->cmd, item->query);
}

static void _SlowLog_RemoveItemFromLookup
(
	SlowLog *slowlog,
	int t_id,
	const SlowLogItem *item
) {
	char *key;
	rax *lookup = slowlog->lookup[t_id];
	size_t key_len = _SlowLogItem_ToString(item, &key);
	int removed = raxRemove(lookup, (unsigned char *)key, key_len, NULL);
	ASSERT(removed == 1);
	free(key);
}

static bool _SlowLog_Contains
(
	const SlowLog *slowlog,
	int t_id,
	const char *cmd,
	const char *query,
	char **key,
	SlowLogItem **item
) {
	size_t len = _compute_key(key, cmd, query);
	rax *lookup = slowlog->lookup[t_id];
	*item = (SlowLogItem *)raxFind(lookup, (unsigned char *)*key, len);
	if(*item == raxNotFound) {
		*item = NULL;
	}

	return (*item != NULL);
}

SlowLog *SlowLog_New(void) {
	SlowLog *slowlog = rm_malloc(sizeof(SlowLog));

	// Redis main thread + worker threads
	int thread_count = ThreadPool_ThreadCount() + 1;

	slowlog->count    = thread_count;
	slowlog->locks    = rm_malloc(sizeof(pthread_mutex_t) * thread_count);
	slowlog->lookup   = rm_malloc(sizeof(rax*) * thread_count);
	slowlog->min_heap = rm_malloc(sizeof(heap_t*) * thread_count);

	for(int i = 0; i < thread_count; i++) {
		slowlog->lookup[i] = raxNew();
		slowlog->min_heap[i] = Heap_new(_slowlog_elem_compare, NULL);
		int res = pthread_mutex_init(slowlog->locks + i, NULL);
		ASSERT(res == 0);
	}

	return slowlog;
}

void SlowLog_Add
(
	SlowLog *slowlog,
	const char *cmd,
	const char *query,
	double latency,
	time_t *t
) {
	ASSERT(latency >= 0);
	ASSERT(cmd     != NULL);
	ASSERT(query   != NULL);
	ASSERT(slowlog != NULL);

	int res;
	UNUSED(res);
	char *key;
	time_t _time;
	SlowLogItem *existing_item;
	int t_id = ThreadPool_GetThreadID();
	rax *lookup = slowlog->lookup[t_id];
	heap_t *heap = slowlog->min_heap[t_id];
	pthread_mutex_t *lock = slowlog->locks + t_id;

	// initialise time
	(t) ? _time = *t : time(&_time);

	if(pthread_mutex_lock(lock) != 0) {
		// failed to lock, skip logging
		return;
	}

	{
		// in critical section
		bool exists = _SlowLog_Contains(slowlog, t_id, cmd, query, &key,
				&existing_item);
		size_t key_len = strlen(key);

		if(exists) {
			// a similar item already exists
			// see if we need to update its latency
			if(existing_item->latency < latency) {
				existing_item->time = _time;
				existing_item->latency = latency;
			}
			goto cleanup;
		}

		// similar item does not exist in the log
		// check if there's enough room to store item
		int introduce_item = 0;
		if(Heap_count(heap) < SLOW_LOG_SIZE) {
			introduce_item = 1;
		} else {
			// not enough room, see if item should be tracked
			SlowLogItem *top = Heap_peek(heap);
			if(top->latency < latency) {
				top = Heap_poll(heap);
				_SlowLog_RemoveItemFromLookup(slowlog, t_id, top);
				_SlowLog_Item_Free(top);
				introduce_item = 1;
			}
		}

		if(introduce_item) {
			SlowLogItem *item = _SlowLogItem_New(cmd, query, latency, _time);
			Heap_offer(slowlog->min_heap + t_id, item);
			raxInsert(lookup, (unsigned char *)key, key_len, item, NULL);
		}
	}   // end of critical section
cleanup:

	res = pthread_mutex_unlock(lock);
	ASSERT(res == 0);
	free(key);
}

// clear all entries from slowlog
void SlowLog_Clear
(
	SlowLog *slowlog  // slowlog to clear
) {
	ASSERT(slowlog != NULL);

	for(int t_id = 0; t_id < slowlog->count; t_id++) {
		raxIterator iter;
		rax *lookup = slowlog->lookup[t_id];
		heap_t *heap = slowlog->min_heap[t_id];

		// enter critical section
		if(pthread_mutex_lock(slowlog->locks + t_id) != 0) {
			// failed to lock, skip logging
			break;
		}

		//----------------------------------------------------------------------
		// free sloglog items
		//----------------------------------------------------------------------

		raxStart(&iter, lookup);
		raxSeek(&iter, "^", NULL, 0);
		while(raxNext(&iter)) {
			SlowLogItem *item = iter.data;
			_SlowLog_Item_Free(item);
		}

		// clear lookup and heap
		raxFree(lookup);
		slowlog->lookup[t_id] = raxNew();
		Heap_clear(heap);

		// end of critical section
		pthread_mutex_unlock(slowlog->locks + t_id);

		raxStop(&iter);
	}
}

void SlowLog_Replay
(
	const SlowLog *slowlog,
	RedisModuleCtx *ctx
) {
	int my_t_id = ThreadPool_GetThreadID();
	SlowLog *aggregated_slowlog = SlowLog_New();

	for(int t_id = 0; t_id < slowlog->count; t_id++) {
		// don't lock ourselves
		if(my_t_id != t_id) {
			if(pthread_mutex_lock(slowlog->locks + t_id) != 0) {
				// failed to lock, skip aggregating this thread slowlog entries
				continue;
			}
		}
		{
			// critical section
			rax *lookup = slowlog->lookup[t_id];
			raxIterator iter;
			raxStart(&iter, lookup);
			raxSeek(&iter, "^", NULL, 0);
			while(raxNext(&iter)) {
				SlowLogItem *item = iter.data;
				SlowLog_Add(aggregated_slowlog, item->cmd, item->query,
							item->latency, &item->time);
			}
			raxStop(&iter);
			// end of critical section
		}
		if(my_t_id != t_id) {
			pthread_mutex_unlock(slowlog->locks + t_id);
		}
	}

	heap_t *heap = aggregated_slowlog->min_heap[my_t_id];
	RedisModule_ReplyWithArray(ctx, Heap_count(heap));

	while(Heap_count(heap)) {
		SlowLogItem *item = Heap_poll(heap);
		RedisModule_ReplyWithArray(ctx, 4);
		RedisModule_ReplyWithDouble(ctx, item->time);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)item->cmd, strlen(item->cmd));
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)item->query, strlen(item->query));
		_ReplyWithRoundedDouble(ctx, item->latency);
	}

	SlowLog_Free(aggregated_slowlog);
}

void SlowLog_Free(SlowLog *slowlog) {
	for(int i = 0; i < slowlog->count; i++) {
		rax *lookup = slowlog->lookup[i];
		heap_t *heap = slowlog->min_heap[i];

		raxIterator iter;
		raxStart(&iter, lookup);
		raxSeek(&iter, "^", NULL, 0);
		while(raxNext(&iter)) {
			SlowLogItem *item = iter.data;
			_SlowLog_Item_Free(item);
		}
		raxStop(&iter);

		raxFree(lookup);
		Heap_free(heap);
		int res = pthread_mutex_destroy(slowlog->locks + i);
		ASSERT(res == 0);
	}

	rm_free(slowlog->locks);
	rm_free(slowlog->lookup);
	rm_free(slowlog->min_heap);
	rm_free(slowlog);
}

