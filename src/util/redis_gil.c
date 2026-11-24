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
	// only the outermost release unlocks; guard against an unbalanced release
	if(t_gil_depth > 0 && --t_gil_depth == 0) {
		RedisModule_ThreadSafeContextUnlock(ctx);
	}
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
	// take the GIL only when this thread doesn't already hold it
	bool need_lock = !GIL_HeldByCurrentThread() &&
		RedisModule_GetThreadSafeContext != NULL;

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
	if(need_lock) {
		GIL_Lock(ctx);
	}

	*took_gil = need_lock;
	return ctx;
}

void GIL_ReleaseContext
(
	RedisModuleCtx *ctx,  // context returned by GIL_AcquireContext
	bool took_gil         // the flag GIL_AcquireContext reported
) {
	if(took_gil) {
		GIL_Unlock(ctx);
	}
	RedisModule_FreeThreadSafeContext(ctx);
}
