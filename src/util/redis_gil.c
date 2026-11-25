/*
 * Copyright FalkorDB Ltd. 2024 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "redis_gil.h"
#include "../RG.h"          // ASSERT
#include "thpool/pool.h"    // ThreadIsMain

// per-thread GIL acquisition depth; > 0 means this thread took the GIL
static __thread int t_gil_depth = 0;

// the GIL must not be (re-)locked when the calling thread already holds it: the
// main thread holds it implicitly (the event loop owns it). the module API is
// unresolved in a standalone unit test - nothing to lock there either
static inline bool _gil_noop(void) {
	return RedisModule_ThreadSafeContextLock == NULL || ThreadIsMain();
}

void GIL_Lock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
) {
	if(_gil_noop()) {
		return;
	}

	// only the outermost acquisition on this thread takes the real lock
	if(t_gil_depth++ == 0) {
		RedisModule_ThreadSafeContextLock(ctx);
	}
}

int GIL_TryLock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
) {
	// already held (main thread) or nothing to lock (no module API)
	if(_gil_noop()) {
		return REDISMODULE_OK;
	}

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
	if(_gil_noop()) {
		return;
	}

	// reaching here at depth 0 is an unbalanced release - a bug
	ASSERT(t_gil_depth > 0);
	if(t_gil_depth == 0) {
		return;  // defensive: never underflow in release builds
	}

	// release the GIL on the outermost unlock, but keep the depth counter
	// nonzero until RedisModule_ThreadSafeContextUnlock returns: Redis may run
	// post-execution jobs on this thread during the unlock while the mutex is
	// still held, and if such a job frees a graph its GIL_Lock must see the GIL
	// as held (depth > 0) rather than re-lock it and deadlock
	if(t_gil_depth == 1) {
		RedisModule_ThreadSafeContextUnlock(ctx);
	}
	t_gil_depth--;
}
