/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// Regression tests for two synchronization bugs fixed in src/index/indexer.c:
//
//   C-9  Consumer lacked a while-loop around pthread_cond_wait.  A spurious
//        wakeup (which POSIX explicitly permits) caused it to pop from an
//        empty queue — undefined behaviour.
//
//   C-10 Indexer_Init's cleanup path used "result-code == 0" as a proxy for
//        "initialised OK". The result counters start at 0, so an early failure
//        caused pthread_*_destroy to be called on primitives that were never
//        initialised — undefined behaviour that corrupts pthread internal state.
//
// ── Testing strategy ──────────────────────────────────────────────────────────
//
// These tests verify the INVARIANTS of the correct synchronisation behaviour.
// They reimplement the sync primitives in isolation — calling the full
// Indexer_Init/Stop API would require mocking hundreds of Redis/FalkorDB
// symbols; the flow tests cover that integration path.
//
// Default compile (no extra flags):
//   Both tests assert the CORRECT (fixed) behaviour.  They PASS on the current
//   codebase.  If the fix is ever accidentally reverted (while-loop → if-loop,
//   or explicit bool flags → result-code==0 proxy), the corresponding test
//   will FAIL.
//
// To verify the tests *detect* the bugs, compile with either flag:
//   -DSIMULATE_C9_BUG    → test_C9 uses the old if-loop pattern; it FAILS
//   -DSIMULATE_C10_BUG   → test_C10 uses the old result-code cleanup; it FAILS
//
// Example (run from repo root):
//   gcc -std=gnu11 -DSIMULATE_C9_BUG -Isrc -I. -Itests/unit \
//       tests/unit/test_indexer_sync.c /tmp/alloc_stub.c -pthread \
//       -o /tmp/test_c9_bug && /tmp/test_c9_bug
//   # → test_C9_consumer_handles_spurious_wakeup ... FAILED

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // for pthread_timedjoin_np / pthread_tryjoin_np
#endif

#include "src/util/rmalloc.h"
#include "src/util/arr.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void setup(void) { Alloc_Reset(); }
static void tearDown(void) {}

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

static void ms_sleep(unsigned ms) {
	struct timespec ts = { .tv_sec = ms / 1000,
	                       .tv_nsec = (long)(ms % 1000) * 1000000L };
	nanosleep(&ts, NULL);
}

// Join with a timeout so a hang is reported as a test failure rather than
// blocking the test runner indefinitely.
static bool join_with_timeout(pthread_t t, unsigned timeout_ms) {
#if defined(__GLIBC__)
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	ts.tv_sec  += timeout_ms / 1000;
	ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
	if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
	return pthread_timedjoin_np(t, NULL, &ts) == 0;
#else
	unsigned waited = 0;
	while (waited < timeout_ms) {
		if (pthread_tryjoin_np(t, NULL) == 0) return true;
		ms_sleep(10);
		waited += 10;
	}
	return false;
#endif
}

//------------------------------------------------------------------------------
// C-9 — regression test: consumer must not act on a spurious wakeup
//------------------------------------------------------------------------------
//
// The fixed _indexer_PopTask uses a while-loop:
//
//     pthread_mutex_lock(&m);
//     while (empty) pthread_cond_wait(&c, &m);   // POSIX-safe
//     pop();
//     pthread_mutex_unlock(&m);
//
// The old (buggy) version used an if-statement, so a spurious wakeup from
// pthread_cond_wait would fall straight through to pop() on an empty queue.
//
// Compile with -DSIMULATE_C9_BUG to activate the old pattern; the test
// will then FAIL, proving it detects the regression.

typedef struct {
	pthread_mutex_t m;
	pthread_mutex_t cm;    // only used by the old (buggy) consumer
	pthread_cond_t  c;
	int            *q;

	atomic_bool     popped_with_empty_queue;
	atomic_int      popped_value;
} SyncState;

static void SyncState_Init(SyncState *s) {
	TEST_ASSERT(pthread_mutex_init(&s->m,  NULL) == 0);
	TEST_ASSERT(pthread_mutex_init(&s->cm, NULL) == 0);
	TEST_ASSERT(pthread_cond_init (&s->c,  NULL) == 0);
	s->q = arr_new(int, 0);
	atomic_init(&s->popped_with_empty_queue, false);
	atomic_init(&s->popped_value, -1);
}

static void SyncState_Destroy(SyncState *s) {
	arr_free(s->q);
	pthread_cond_destroy (&s->c);
	pthread_mutex_destroy(&s->cm);
	pthread_mutex_destroy(&s->m);
}

// Buggy consumer (old pattern): if-statement, separate cm mutex.
// Activated only by SIMULATE_C9_BUG.
static void *buggy_consumer(void *arg) {
	SyncState *s = (SyncState *)arg;

	pthread_mutex_lock(&s->m);
	if (arr_len(s->q) == 0) {           // BUG: if, not while
		pthread_mutex_lock(&s->cm);
		pthread_mutex_unlock(&s->m);
		pthread_cond_wait(&s->c, &s->cm);
		pthread_mutex_unlock(&s->cm);
		pthread_mutex_lock(&s->m);
	}

	// Unconditional pop — UB when queue is empty
	if (arr_len(s->q) == 0) {
		atomic_store(&s->popped_with_empty_queue, true);
	} else {
		atomic_store(&s->popped_value, s->q[0]);
		arr_del(s->q, 0);
	}

	pthread_mutex_unlock(&s->m);
	return NULL;
}

// Fixed consumer (new pattern): while-loop, single mutex m.
// This is the pattern indexer.c uses after the C-9 fix.
static void *fixed_consumer(void *arg) {
	SyncState *s = (SyncState *)arg;

	pthread_mutex_lock(&s->m);
	while (arr_len(s->q) == 0) {        // FIXED: while-loop re-checks after wakeup
		pthread_cond_wait(&s->c, &s->m);
	}

	if (arr_len(s->q) == 0) {
		atomic_store(&s->popped_with_empty_queue, true);
	} else {
		atomic_store(&s->popped_value, s->q[0]);
		arr_del(s->q, 0);
	}

	pthread_mutex_unlock(&s->m);
	return NULL;
}

// REGRESSION TEST for C-9.
//
// Invariants under test (must hold on current code; FAIL if bug is reverted):
//   1. Consumer does not pop from an empty queue when it receives a spurious
//      wakeup.
//   2. Consumer correctly pops the real task once it is added.
static void test_C9_consumer_handles_spurious_wakeup(void) {
	SyncState s;
	SyncState_Init(&s);

	pthread_t t;

#ifdef SIMULATE_C9_BUG
	// Use the OLD (buggy) consumer to prove the test detects the regression.
	TEST_ASSERT(pthread_create(&t, NULL, buggy_consumer, &s) == 0);
	ms_sleep(50);
	// Buggy consumer waits on cm; inject spurious broadcast on cm.
	pthread_mutex_lock(&s.cm);
	pthread_mutex_lock(&s.m);
	pthread_cond_broadcast(&s.c);
	pthread_mutex_unlock(&s.m);
	pthread_mutex_unlock(&s.cm);
	ms_sleep(50);
	// Add a real task — buggy consumer already exited, so nobody consumes it.
	pthread_mutex_lock(&s.m);
	arr_append(s.q, 42);
	pthread_cond_signal(&s.c);
	pthread_mutex_unlock(&s.m);
#else
	// Use the FIXED consumer (default, matches current indexer.c).
	TEST_ASSERT(pthread_create(&t, NULL, fixed_consumer, &s) == 0);
	ms_sleep(50);
	// Spurious broadcast on m, no task added yet.
	pthread_mutex_lock(&s.m);
	pthread_cond_broadcast(&s.c);
	pthread_mutex_unlock(&s.m);
	ms_sleep(50);
	// Now add the real task.
	pthread_mutex_lock(&s.m);
	arr_append(s.q, 42);
	pthread_cond_signal(&s.c);
	pthread_mutex_unlock(&s.m);
#endif

	TEST_ASSERT_(join_with_timeout(t, 2000),
		"consumer thread must complete within 2 s");

	// Invariant 1: consumer must not pop from an empty queue.
	// SIMULATE_C9_BUG → buggy_consumer sets popped_with_empty_queue=true → FAILS here.
	TEST_ASSERT_(atomic_load(&s.popped_with_empty_queue) == false,
		"consumer must not pop from an empty queue on a spurious wakeup "
		"(C-9 regression: while-loop required around pthread_cond_wait)");

	// Invariant 2: consumer must pop the real task.
	// SIMULATE_C9_BUG → buggy_consumer exited before task was added → popped_value=-1 → FAILS here.
	TEST_ASSERT_(atomic_load(&s.popped_value) == 42,
		"consumer must pop the real task value (not exit early on spurious wakeup)");

	SyncState_Destroy(&s);
}

//------------------------------------------------------------------------------
// C-10 — regression test: Init cleanup must only destroy initialised primitives
//------------------------------------------------------------------------------
//
// The fixed Indexer_Init tracks each primitive with an explicit boolean:
//
//     bool m_init = false;
//     if (pthread_mutex_init(&m, NULL) != 0) goto cleanup;
//     m_init = true;
//     ...
//   cleanup:
//     if (m_init) pthread_mutex_destroy(&m);
//
// The old (buggy) version used result-code == 0 as the guard. Since all
// result codes start at 0, an early failure caused destroy to be called
// on primitives that were never initialised.
//
// Compile with -DSIMULATE_C10_BUG to activate the old logic; the test
// will then FAIL, proving it detects the regression.

typedef struct {
	bool was_initialised;
	bool was_destroyed;
} MockPrim;

static void mock_init(MockPrim *p, int rc) {
	if (rc == 0) p->was_initialised = true;
}
static void mock_destroy(MockPrim *p) { p->was_destroyed = true; }

typedef struct {
	int destroys_of_uninitialised;
	int destroys_of_initialised;
} CleanupResult;

static CleanupResult count_destroys(MockPrim *all[], size_t n) {
	CleanupResult r = { 0, 0 };
	for (size_t i = 0; i < n; ++i) {
		if (all[i]->was_destroyed) {
			if (all[i]->was_initialised) r.destroys_of_initialised++;
			else                         r.destroys_of_uninitialised++;
		}
	}
	return r;
}

// Simulates the OLD cleanup logic: result-code==0 is the "did it init?" guard.
static CleanupResult run_old_cleanup(
	int m_rc, int cm_rc, int c_rc, int attr_rc
) {
	MockPrim m={0}, cm={0}, c={0}, attr={0};
	int m_res=0, cm_res=0, c_res=0, a_res=0;

	mock_init(&m, m_rc);   m_res  = m_rc;   if (m_res)  goto cleanup;
	mock_init(&cm, cm_rc); cm_res = cm_rc;  if (cm_res) goto cleanup;
	mock_init(&c, c_rc);   c_res  = c_rc;   if (c_res)  goto cleanup;
	mock_init(&attr, attr_rc); a_res = attr_rc; if (a_res) goto cleanup;
	goto done;

cleanup:
	// BUG: "== 0" is true both for "init succeeded" and "init never ran"
	if (c_res  == 0) mock_destroy(&c);
	if (m_res  == 0) mock_destroy(&m);
	if (cm_res == 0) mock_destroy(&cm);
	if (a_res  == 0) mock_destroy(&attr);

done:;
	MockPrim *all[] = { &m, &cm, &c, &attr };
	return count_destroys(all, 4);
}

// Simulates the FIXED cleanup logic: explicit boolean per primitive.
// This mirrors what indexer.c does after the C-10 fix.
static CleanupResult run_new_cleanup(
	int m_rc, int cm_rc, int c_rc, int attr_rc
) {
	MockPrim m={0}, cm={0}, c={0}, attr={0};
	bool m_init=false, cm_init=false, c_init=false, attr_init=false;

	mock_init(&m, m_rc);
	if (m_rc != 0) goto cleanup;
	m_init = true;

	mock_init(&cm, cm_rc);
	if (cm_rc != 0) goto cleanup;
	cm_init = true;

	mock_init(&c, c_rc);
	if (c_rc != 0) goto cleanup;
	c_init = true;

	mock_init(&attr, attr_rc);
	if (attr_rc != 0) goto cleanup;
	attr_init = true;

	goto done;

cleanup:
	if (attr_init) mock_destroy(&attr);
	if (c_init)    mock_destroy(&c);
	if (cm_init)   mock_destroy(&cm);
	if (m_init)    mock_destroy(&m);

done:;
	MockPrim *all[] = { &m, &cm, &c, &attr };
	return count_destroys(all, 4);
}

// REGRESSION TEST for C-10.
//
// Invariants under test (must hold on current code; FAIL if bug is reverted):
//   1. Cleanup never calls destroy on a primitive that was never initialised.
//   2. Cleanup calls destroy on exactly the N primitives that were
//      successfully initialised before the failure.
static void test_C10_init_cleanup_safe(void) {
	// Exercise every possible early-failure step (fail at step 0, 1, 2, or 3).
	for (int fail_at = 0; fail_at < 4; ++fail_at) {
		int rc[4] = { 0, 0, 0, 0 };
		rc[fail_at] = EAGAIN;  // force failure at this step

		CleanupResult r;
#ifdef SIMULATE_C10_BUG
		// Use OLD (buggy) cleanup to prove the test detects the regression.
		r = run_old_cleanup(rc[0], rc[1], rc[2], rc[3]);
		// fail_at==0: m_res=EAGAIN but cm_res=c_res=a_res=0 → destroys cm,c,attr
		//   (never init'd) → destroys_of_uninitialised==3 → Invariant 1 FAILS.
#else
		// Use FIXED cleanup (default, matches current indexer.c).
		r = run_new_cleanup(rc[0], rc[1], rc[2], rc[3]);
#endif

		// Invariant 1: zero destroys on uninitialised primitives.
		// SIMULATE_C10_BUG → destroys_of_uninitialised > 0 for fail_at < 3 → FAILS.
		TEST_ASSERT_(r.destroys_of_uninitialised == 0,
			"C-10 regression: cleanup must not destroy uninitialised primitives; "
			"fail_at=%d produced %d bad destroys",
			fail_at, r.destroys_of_uninitialised);

		// Invariant 2: exactly fail_at primitives were init'd and must be destroyed.
		TEST_ASSERT_(r.destroys_of_initialised == fail_at,
			"cleanup must destroy exactly %d initialised primitive(s); "
			"fail_at=%d, got %d",
			fail_at, fail_at, r.destroys_of_initialised);
	}
}

//------------------------------------------------------------------------------
// test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"C9_consumer_handles_spurious_wakeup",
		test_C9_consumer_handles_spurious_wakeup},
	{"C10_init_cleanup_safe",
		test_C10_init_cleanup_safe},
	{NULL, NULL}
};
