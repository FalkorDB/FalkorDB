/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "redis_gil.h"
#include "thpool/pool.h"  // MAIN_THREAD_ID

#include <pthread.h>

// per-thread GIL acquisition depth; > 0 means this thread holds the GIL
static __thread int t_gil_depth = 0;

void GIL_Lock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
) {
	// only the outermost acquisition actually locks; nested calls just count
	if(t_gil_depth++ == 0) {
		RedisModule_ThreadSafeContextLock(ctx);
	}
}

int GIL_TryLock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
) {
	// already held by this thread - just count the nested acquisition
	if(t_gil_depth > 0) {
		t_gil_depth++;
		return REDISMODULE_OK;
	}

	// not held - attempt a non-blocking acquisition
	if(RedisModule_ThreadSafeContextTryLock(ctx) == REDISMODULE_OK) {
		t_gil_depth++;  // 0 -> 1
		return REDISMODULE_OK;
	}

	return REDISMODULE_ERR;
}

void GIL_Unlock
(
	RedisModuleCtx *ctx  // thread-safe context used to unlock the GIL
) {
	// guard against an unbalanced release
	if(t_gil_depth == 0) {
		return;
	}

	// release the GIL on the outermost unlock, but keep the depth counter
	// nonzero until RedisModule_ThreadSafeContextUnlock returns: Redis may run
	// post-execution jobs on this thread during the unlock while the mutex is
	// still held, and if such a job frees a graph its callback must see the GIL
	// as held (depth > 0) rather than try to re-lock it and deadlock
	if(t_gil_depth == 1) {
		RedisModule_ThreadSafeContextUnlock(ctx);
	}
	t_gil_depth--;
}

bool GIL_HeldByCurrentThread(void) {
	// the main thread holds the GIL implicitly (the event loop owns it);
	// background threads hold it only after acquiring via GIL_Lock / GIL_TryLock
	return ThreadIsMain() || t_gil_depth > 0;
}

RedisModuleCtx *GIL_AcquireContext
(
	bool *took_gil  // out: true if this call took the GIL
) {
	*took_gil = false;

	// the Redis module API may be unresolved (e.g. a standalone unit test that
	// frees a fake graph); there's no keyspace to mutate and nothing to lock, so
	// return a NULL context - GIL_ReleaseContext tolerates it
	if(RedisModule_GetThreadSafeContext == NULL) {
		return NULL;
	}

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);

	// take the GIL only when this thread doesn't already hold it
	if(!GIL_HeldByCurrentThread()) {
		GIL_Lock(ctx);
		*took_gil = true;
	}

	return ctx;
}

void GIL_ReleaseContext
(
	RedisModuleCtx *ctx,  // context returned by GIL_AcquireContext
	bool took_gil         // the flag GIL_AcquireContext reported
) {
	if(ctx == NULL) {
		return;  // nothing was acquired
	}
	if(took_gil) {
		GIL_Unlock(ctx);
	}
	RedisModule_FreeThreadSafeContext(ctx);
}
