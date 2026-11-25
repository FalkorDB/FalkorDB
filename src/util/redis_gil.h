/*
 * Copyright FalkorDB Ltd. 2024 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

// Reentrant, thread-aware wrappers around Redis's module GIL.
//
// Redis's GIL is a non-recursive mutex and the module API exposes no "do I
// already hold it?" query. These wrappers let code lock/unlock the GIL
// unconditionally: the call is a no-op when the calling thread already holds it.
// The redis main thread holds it implicitly (the event loop owns it); a
// background thread that already acquired it is tracked per thread, so only its
// first acquisition takes the real lock and nested acquisitions just count.
// Without this, a background thread that frees a graph while already holding the
// GIL (redis freeing a key under the GIL, or the enterprise offload's evict
// thread) would re-lock the non-recursive GIL and deadlock against itself.
//
// Calls are also a no-op when the Redis module API is unresolved (a standalone
// unit test with no keyspace). The ctx argument is the handle for the
// underlying Redis lock; it is dereferenced only when a real lock/unlock
// happens (a background thread's outermost acquisition).

// acquire the GIL, unless the calling thread already holds it
void GIL_Lock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
);

// try to acquire the GIL without blocking; returns REDISMODULE_OK when the
// calling thread holds it afterwards (including when it already held it) and
// REDISMODULE_ERR when it could not be taken. a successful call must be paired
// with a GIL_Unlock
int GIL_TryLock
(
	RedisModuleCtx *ctx  // thread-safe context used to lock the GIL
);

// release the GIL; only the outermost release on a background thread unlocks
void GIL_Unlock
(
	RedisModuleCtx *ctx  // thread-safe context used to unlock the GIL
);
