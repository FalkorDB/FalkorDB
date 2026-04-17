/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// Regression tests for the two synchronization bugs that lived in
// src/index/indexer.c:
//
//   C-9  The producer signalled a condition variable guarded by a
//        *separate* mutex "cm", while the queue predicate was guarded by
//        "m". The consumer did NOT loop around pthread_cond_wait, so a
//        spurious wakeup (which POSIX explicitly allows) would cause it
//        to proceed and pop from an empty queue — undefined behaviour.
//
//   C-10 Indexer_Init's cleanup path used "result-code == 0" as a proxy
//        for "initialised OK". The counters were pre-set to 0, so an
//        early failure (e.g. pthread_mutex_init failing on the first
//        primitive) caused pthread_*_destroy to be called on primitives
//        that were never initialised — undefined behaviour that can
//        corrupt pthread-internal state for subsequent Init calls.
//
// These tests recreate *the patterns* used by the old code in isolation
// so that the bugs are observable without having to stand up the full
// Redis module context. Running them against the OLD pattern fails;
// running them against the FIXED pattern passes.

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

static void setup() { Alloc_Reset(); }
static void tearDown() {}

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

//------------------------------------------------------------------------------
// C-9 — spurious wakeup causes pop-from-empty in the OLD sync pattern
//------------------------------------------------------------------------------
//
// The old _indexer_PopTask did:
//
//     lock(m);
//     if (empty) {            // <-- IF, not WHILE
//         lock(cm);
//         unlock(m);
//         cond_wait(c, cm);   // spurious wakeups land here
//         unlock(cm);
//         lock(m);
//     }
//     pop();                  // <-- unconditional
//     unlock(m);
//
// With a spurious wakeup the consumer falls straight through to the
// unconditional pop, which operates on an empty queue.

typedef struct {
	pthread_mutex_t m;
	pthread_mutex_t cm;
	pthread_cond_t  c;
	int            *q;

	// results the consumer writes out for the test to inspect
	atomic_bool     woke_up;
	atomic_bool     popped_with_empty_queue;
	atomic_int      popped_value;
} SyncState;

static void SyncState_Init(SyncState *s) {
	TEST_ASSERT(pthread_mutex_init(&s->m,  NULL) == 0);
	TEST_ASSERT(pthread_mutex_init(&s->cm, NULL) == 0);
	TEST_ASSERT(pthread_cond_init (&s->c,  NULL) == 0);
	s->q = arr_new(int, 0);
	atomic_init(&s->woke_up,                 false);
	atomic_init(&s->popped_with_empty_queue, false);
	atomic_init(&s->popped_value,            -1);
}

static void SyncState_Destroy(SyncState *s) {
	arr_free(s->q);
	pthread_cond_destroy (&s->c);
	pthread_mutex_destroy(&s->cm);
	pthread_mutex_destroy(&s->m);
}

// OLD pattern — reproduces the bug
static void *old_consumer(void *arg) {
	SyncState *s = (SyncState *)arg;

	pthread_mutex_lock(&s->m);
	if (arr_len(s->q) == 0) {
		pthread_mutex_lock(&s->cm);
		pthread_mutex_unlock(&s->m);
		pthread_cond_wait(&s->c, &s->cm);
		pthread_mutex_unlock(&s->cm);
		pthread_mutex_lock(&s->m);
	}

	atomic_store(&s->woke_up, true);

	// *** the bug ***: pop with no re-check after cond_wait
	if (arr_len(s->q) == 0) {
		atomic_store(&s->popped_with_empty_queue, true);
	} else {
		atomic_store(&s->popped_value, s->q[0]);
		arr_del(s->q, 0);
	}

	pthread_mutex_unlock(&s->m);
	return NULL;
}

// FIXED pattern — same signal/wait semantics as indexer.c after the fix
static void *new_consumer(void *arg) {
	SyncState *s = (SyncState *)arg;

	pthread_mutex_lock(&s->m);
	while (arr_len(s->q) == 0) {
		pthread_cond_wait(&s->c, &s->m);
	}

	atomic_store(&s->woke_up, true);

	if (arr_len(s->q) == 0) {
		atomic_store(&s->popped_with_empty_queue, true);
	} else {
		atomic_store(&s->popped_value, s->q[0]);
		arr_del(s->q, 0);
	}

	pthread_mutex_unlock(&s->m);
	return NULL;
}

// Inject a spurious wakeup: broadcast on the condition variable without
// adding any task to the queue. POSIX allows this at any time; a real
// kernel may also deliver a spurious wakeup by itself.
static void inject_spurious_wakeup(SyncState *s) {
	// give the consumer time to actually enter cond_wait
	ms_sleep(50);

	// OLD code's producer signals with cm held — mimic that here, since
	// cm is the mutex the consumer parked on.
	pthread_mutex_lock(&s->cm);
	pthread_mutex_lock(&s->m);
	pthread_cond_broadcast(&s->c);
	pthread_mutex_unlock(&s->m);
	pthread_mutex_unlock(&s->cm);
}

// Wake the new-pattern consumer the "correct" way (only broadcast while
// holding m, same as a benign kernel spurious wakeup — no task added).
// With the fixed pattern the consumer re-checks and goes back to sleep,
// so we also push a real task afterwards so the thread can finish.
static void drive_new_consumer(SyncState *s) {
	ms_sleep(50);

	pthread_mutex_lock(&s->m);
	pthread_cond_broadcast(&s->c);   // spurious
	pthread_mutex_unlock(&s->m);

	ms_sleep(50);

	// real task
	pthread_mutex_lock(&s->m);
	arr_append(s->q, 42);
	pthread_cond_signal(&s->c);
	pthread_mutex_unlock(&s->m);
}

// Join with a timeout so a hang is reported as a test failure instead of
// hanging the test runner.
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

static void test_C9_old_pattern_spurious_wakeup_pops_empty(void) {
	// Demonstrates the bug the fix closes: the OLD pattern (no while-loop
	// around cond_wait) acts on a spurious wakeup and pops from an empty
	// queue. We flip the test so it's a positive regression check: the
	// OLD pattern MUST exhibit the bug here.

	SyncState s;
	SyncState_Init(&s);

	pthread_t t;
	TEST_ASSERT(pthread_create(&t, NULL, old_consumer, &s) == 0);

	inject_spurious_wakeup(&s);

	TEST_ASSERT(join_with_timeout(t, 2000));
	TEST_ASSERT(atomic_load(&s.woke_up) == true);

	// the point: OLD pattern popped from an empty queue
	TEST_ASSERT_(atomic_load(&s.popped_with_empty_queue) == true,
		"OLD pattern should pop from empty queue on spurious wakeup "
		"(this is exactly the bug the while-loop fix eliminates)");

	SyncState_Destroy(&s);
}

static void test_C9_new_pattern_survives_spurious_wakeup(void) {
	// Same scenario, run against the FIXED pattern. The consumer must
	// re-check the predicate, go back to sleep on the spurious wakeup,
	// then consume the real task without ever popping from an empty
	// queue.

	SyncState s;
	SyncState_Init(&s);

	pthread_t t;
	TEST_ASSERT(pthread_create(&t, NULL, new_consumer, &s) == 0);

	drive_new_consumer(&s);

	TEST_ASSERT_(join_with_timeout(t, 2000),
		"consumer must not hang on spurious wakeup with the fixed pattern");

	TEST_ASSERT(atomic_load(&s.woke_up) == true);
	TEST_ASSERT_(atomic_load(&s.popped_with_empty_queue) == false,
		"FIXED pattern must never act on a spurious wakeup");
	TEST_ASSERT(atomic_load(&s.popped_value) == 42);

	SyncState_Destroy(&s);
}

//------------------------------------------------------------------------------
// C-10 — cleanup of uninitialised pthread primitives on Init error path
//------------------------------------------------------------------------------
//
// The old cleanup logic:
//
//     int a_res=0, c_res=0, m_res=0, cm_res=0;
//     m_res  = pthread_mutex_init(&m,  NULL);  if (m_res)  goto cleanup;
//     cm_res = pthread_mutex_init(&cm, NULL);  if (cm_res) goto cleanup;
//     c_res  = pthread_cond_init (&c,  NULL);  if (c_res)  goto cleanup;
//     a_res  = pthread_attr_init (&attr);      if (a_res)  goto cleanup;
//     ...
//   cleanup:
//     if (c_res  == 0) pthread_cond_destroy (&c);   // BUG: "0" also means never-init
//     if (m_res  == 0) pthread_mutex_destroy(&m);
//     if (cm_res == 0) pthread_mutex_destroy(&cm);
//     if (a_res  == 0) pthread_attr_destroy (&attr);
//
// If mutex_init returns non-zero on the *first* call, c_res/cm_res/a_res
// are all still their initial 0, so cleanup destroys three primitives
// that were never created.
//
// To observe this deterministically we count destroy-calls against
// primitives we declared never-initialised, using counters.

typedef struct {
	int init_rc;          // return code to hand back from the mock init
	bool was_initialised; // flipped true iff mock init actually ran init
	bool was_destroyed;   // flipped true iff cleanup called destroy
} MockPrim;

static void mock_init(MockPrim *p, int rc) {
	p->init_rc = rc;
	if (rc == 0) p->was_initialised = true;
}
static void mock_destroy(MockPrim *p) { p->was_destroyed = true; }

typedef struct {
	int destroys_of_uninitialised;
	int destroys_of_initialised;
} CleanupResult;

// Mirrors the OLD Indexer_Init cleanup logic *exactly*, but operating on
// mock primitives so we can count mis-directed destroys.
static CleanupResult run_old_cleanup(
	int m_init_rc, int cm_init_rc, int c_init_rc, int attr_init_rc
) {
	MockPrim m={0}, cm={0}, c={0}, attr={0};

	int m_res=0, c_res=0, m_resc=0, a_res=0, cm_res=0;

	// init sequence in the original order
	mock_init(&m, m_init_rc);  m_res = m_init_rc;
	if (m_res != 0) goto cleanup;

	mock_init(&cm, cm_init_rc); cm_res = cm_init_rc;
	if (cm_res != 0) goto cleanup;

	mock_init(&c, c_init_rc);  c_res = c_init_rc;
	if (c_res != 0) goto cleanup;

	mock_init(&attr, attr_init_rc); a_res = attr_init_rc;
	if (a_res != 0) goto cleanup;

	// success — drop through with no cleanup (test only exercises failure paths)
	goto done;

cleanup:
	// BUG-RESIDENT cleanup, copied verbatim
	if (c_res  == 0) mock_destroy(&c);
	if (m_res  == 0) mock_destroy(&m);
	if (cm_res == 0) mock_destroy(&cm);
	if (a_res  == 0) mock_destroy(&attr);

done:;
	(void)m_resc; // unused, preserved to avoid unused-var warning on some compilers
	CleanupResult r = { 0, 0 };
	MockPrim *all[] = { &m, &cm, &c, &attr };
	for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); ++i) {
		if (all[i]->was_destroyed) {
			if (all[i]->was_initialised) r.destroys_of_initialised++;
			else                         r.destroys_of_uninitialised++;
		}
	}
	return r;
}

// Mirrors the FIXED Indexer_Init cleanup logic — tracks each primitive
// with an explicit "initialised" flag.
static CleanupResult run_new_cleanup(
	int m_init_rc, int cm_init_rc, int c_init_rc, int attr_init_rc
) {
	// NB: the real fix removed cm entirely; keep it here so both
	// functions take the same parameters and a single test table can
	// exercise them.
	MockPrim m={0}, cm={0}, c={0}, attr={0};

	bool m_init=false, cm_init=false, c_init=false, attr_init=false;

	mock_init(&m, m_init_rc);
	if (m_init_rc != 0) goto cleanup;
	m_init = true;

	mock_init(&cm, cm_init_rc);
	if (cm_init_rc != 0) goto cleanup;
	cm_init = true;

	mock_init(&c, c_init_rc);
	if (c_init_rc != 0) goto cleanup;
	c_init = true;

	mock_init(&attr, attr_init_rc);
	if (attr_init_rc != 0) goto cleanup;
	attr_init = true;

	goto done;

cleanup:
	if (attr_init) mock_destroy(&attr);
	if (c_init)    mock_destroy(&c);
	if (cm_init)   mock_destroy(&cm);
	if (m_init)    mock_destroy(&m);

done:;
	CleanupResult r = { 0, 0 };
	MockPrim *all[] = { &m, &cm, &c, &attr };
	for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); ++i) {
		if (all[i]->was_destroyed) {
			if (all[i]->was_initialised) r.destroys_of_initialised++;
			else                         r.destroys_of_uninitialised++;
		}
	}
	return r;
}

static void test_C10_old_init_destroys_uninitialised(void) {
	// Simulate mutex_init failing on the first call. The old cleanup
	// logic will wrongly destroy three primitives that were never
	// initialised.
	CleanupResult r = run_old_cleanup(/*m*/ EAGAIN, 0, 0, 0);

	TEST_ASSERT_(r.destroys_of_uninitialised == 3,
		"OLD cleanup should destroy 3 never-initialised primitives "
		"(cm, c, attr) when mutex_init fails first; got %d",
		r.destroys_of_uninitialised);
	TEST_ASSERT(r.destroys_of_initialised == 0);

	// Also: mutex_init fails on the *second* primitive — c and attr
	// were never initialised.
	r = run_old_cleanup(0, EAGAIN, 0, 0);
	TEST_ASSERT_(r.destroys_of_uninitialised == 2,
		"OLD cleanup should destroy 2 never-initialised primitives "
		"(c, attr) when cm init fails; got %d",
		r.destroys_of_uninitialised);
}

static void test_C10_new_init_only_destroys_initialised(void) {
	// Same failure scenarios against the fixed cleanup logic — no
	// primitive that was not initialised may be destroyed.
	for (int which = 0; which < 4; ++which) {
		int rc[4] = { 0, 0, 0, 0 };
		rc[which] = EAGAIN;
		CleanupResult r = run_new_cleanup(rc[0], rc[1], rc[2], rc[3]);
		TEST_ASSERT_(r.destroys_of_uninitialised == 0,
			"FIXED cleanup must never destroy an uninitialised primitive; "
			"failure-on-step=%d produced %d bad destroys",
			which, r.destroys_of_uninitialised);
		TEST_ASSERT_(r.destroys_of_initialised == which,
			"FIXED cleanup should destroy exactly the primitives that were "
			"successfully initialised; failure-on-step=%d expected %d got %d",
			which, which, r.destroys_of_initialised);
	}
}

//------------------------------------------------------------------------------
// test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"C9_old_pattern_spurious_wakeup_pops_empty",
		test_C9_old_pattern_spurious_wakeup_pops_empty},
	{"C9_new_pattern_survives_spurious_wakeup",
		test_C9_new_pattern_survives_spurious_wakeup},
	{"C10_old_init_destroys_uninitialised",
		test_C10_old_init_destroys_uninitialised},
	{"C10_new_init_only_destroys_initialised",
		test_C10_new_init_only_destroys_initialised},
	{NULL, NULL}
};
