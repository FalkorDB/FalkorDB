/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

#include <stdbool.h>

// Reentrant helpers around Redis's module GIL that also track, per thread,
// whether the calling thread currently holds it.
//
// Redis's GIL is a non-recursive mutex and the module API exposes no "do I
// already hold it?" query. Off the main thread, code that frees a graph takes
// the GIL (keyspace mutation requires it) under the assumption that a non-main
// thread never holds it. That assumption breaks for a module background thread
// that locked the GIL itself (e.g. the enterprise offload's evict thread) and
// then triggers the graph free callback: re-locking the GIL deadlocks the
// thread against itself.
//
// Acquiring the GIL through these wrappers keeps an accurate per-thread
// ownership flag, so those free paths can take the GIL only when they don't
// already hold it. Both FalkorDB and the enterprise module acquire through the
// same wrappers (the latter via the shared API) so the flag is consistent
// across the .so boundary - the thread-local lives in FalkorDB and both call
// into it.

// acquire the GIL; nested acquisitions on the same thread lock only once
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

// release the GIL; only the outermost release actually unlocks
void GIL_Unlock
(
	RedisModuleCtx *ctx  // thread-safe context used to unlock the GIL
);

// returns true if the calling thread currently holds the GIL: the main thread
// holds it implicitly, a background thread holds it after GIL_Lock/GIL_TryLock.
// use this to decide whether a keyspace mutation needs to take the GIL first
bool GIL_HeldByCurrentThread(void);

// create a detached thread-safe context for a keyspace/RediSearch mutation and
// take the GIL on it, but only when the calling thread doesn't already hold it
// (the main thread holds it implicitly; a background thread may have locked it
// itself) - re-locking the non-recursive GIL would deadlock the thread against
// itself. reports in *took_gil whether this call took the GIL; pass it, with
// the returned context, to GIL_ReleaseContext
RedisModuleCtx *GIL_AcquireContext
(
	bool *took_gil  // out: true if this call took the GIL
);

// release a context obtained from GIL_AcquireContext, unlocking the GIL only
// when that call took it
void GIL_ReleaseContext
(
	RedisModuleCtx *ctx,  // context returned by GIL_AcquireContext
	bool took_gil         // the flag GIL_AcquireContext reported
);
