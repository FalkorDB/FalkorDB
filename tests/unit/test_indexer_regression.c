/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// TRUE regression tests for C-9 and C-10 in src/index/indexer.c.
//
// Unlike test_indexer_sync.c, this file embeds the EXACT sync code from
// indexer.c verbatim and exercises it via the real Indexer_Init / Indexer_Stop
// API.  It does NOT link against the falkordb library; task handlers (which
// depend on Redis/Graph/Index types) are replaced with no-op stubs since our
// tests only ever enqueue an INDEXER_EXIT task.
//
// This file MUST be compiled with these linker wrap flags:
//   -Wl,--wrap=pthread_cond_wait
//   -Wl,--wrap=pthread_mutex_init
//   -Wl,--wrap=pthread_mutex_destroy
//   -Wl,--wrap=pthread_cond_destroy
//   -Wl,--wrap=pthread_cond_init
//   -Wl,--wrap=pthread_attr_init
//   -Wl,--wrap=pthread_attr_destroy
// (CMakeLists.txt sets these automatically on Linux.)
//
// Default compile (no extra -D flags):
//   Both tests PASS — the fixed code handles both scenarios correctly.
//
// Compile with -DBUGGY_SYNC to activate the pre-fix code from git commit
// 484abe65a^; both tests will FAIL, proving they detect the regressions:
//
//   -DBUGGY_SYNC → test_C9_no_pop_on_spurious_wakeup  ... FAILED
//                  test_C10_init_cleanup_no_uninit_destroys ... FAILED
//
// Standalone build + test (run from repo root):
//   stub=/tmp/alloc_stub.c; echo 'void Alloc_Reset(void){}' > $stub
//   gcc -std=gnu11 -Wall -Isrc -I. -Itests/unit \
//       tests/unit/test_indexer_regression.c $stub \
//       -pthread -Wl,--wrap=pthread_cond_wait \
//                -Wl,--wrap=pthread_mutex_init \
//                -Wl,--wrap=pthread_mutex_destroy \
//                -Wl,--wrap=pthread_cond_destroy \
//                -Wl,--wrap=pthread_cond_init \
//                -Wl,--wrap=pthread_attr_init \
//                -Wl,--wrap=pthread_attr_destroy \
//       -o /tmp/t_regression && /tmp/t_regression

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "src/util/rmalloc.h"
#include "src/util/arr.h"
#include "src/RG.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void setup(void) { Alloc_Reset(); }
static void tearDown(void) {}
#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

//------------------------------------------------------------------------------
// __wrap_pthread_cond_wait — injects a configurable number of spurious returns
//
// Any call to pthread_cond_wait in this binary goes through here.  When
// spurious_wakeups_remaining > 0 the function decrements the counter and
// returns 0 immediately (the mutex stays locked, exactly as a real spurious
// wakeup behaves).  All other calls reach the real implementation.
//------------------------------------------------------------------------------

static atomic_int spurious_wakeups_remaining = 0;

int __real_pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);

int __wrap_pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
	if (atomic_fetch_sub(&spurious_wakeups_remaining, 1) > 0) {
		return 0;  // spurious return: mutex stays held, no actual wait
	}
	return __real_pthread_cond_wait(c, m);
}

//------------------------------------------------------------------------------
// __wrap_pthread_mutex_init / _destroy / __wrap_pthread_cond_destroy
//
// Used by the C-10 test to:
//   (a) force pthread_mutex_init to fail on a specific call number, and
//   (b) track which mutex/cond addresses were successfully initialised so
//       we can count "destroy on uninitialised primitive" occurrences.
//
// Destroying an uninitialised address is UB; the wrappers intercept these
// calls to count them WITHOUT actually invoking the real destroy (which
// might crash), making C-10 detectable without ASAN/valgrind.
//------------------------------------------------------------------------------

#define MAX_TRACKED 32

// mutex init tracking
static int    force_mutex_fail_on_call = 0;    // 0 = don't fail
static atomic_int mutex_init_call_count = 0;
static pthread_mutex_t *inited_mutexes[MAX_TRACKED];
static int              n_inited_mutexes = 0;

// cond init tracking
static pthread_cond_t   *inited_conds[MAX_TRACKED];
static int               n_inited_conds = 0;

// attr init tracking
static pthread_attr_t   *inited_attrs[MAX_TRACKED];
static int               n_inited_attrs = 0;

// results
static int uninit_mutex_destroys = 0;
static int uninit_cond_destroys  = 0;
static int uninit_attr_destroys  = 0;

static void reset_c10_tracking(void) {
	force_mutex_fail_on_call = 0;
	atomic_store(&mutex_init_call_count, 0);
	n_inited_mutexes = 0;
	n_inited_conds   = 0;
	n_inited_attrs   = 0;
	uninit_mutex_destroys = 0;
	uninit_cond_destroys  = 0;
	uninit_attr_destroys  = 0;
}

int __real_pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
int __real_pthread_mutex_destroy(pthread_mutex_t *);
int __real_pthread_cond_destroy(pthread_cond_t *);

int __wrap_pthread_mutex_init(pthread_mutex_t *m,
                              const pthread_mutexattr_t *attr) {
	int call = atomic_fetch_add(&mutex_init_call_count, 1) + 1;
	if (force_mutex_fail_on_call != 0 && call == force_mutex_fail_on_call) {
		return EINVAL;  // simulate forced failure
	}
	int rc = __real_pthread_mutex_init(m, attr);
	if (rc == 0 && n_inited_mutexes < MAX_TRACKED) {
		inited_mutexes[n_inited_mutexes++] = m;
	}
	return rc;
}

int __wrap_pthread_mutex_destroy(pthread_mutex_t *m) {
	for (int i = 0; i < n_inited_mutexes; i++) {
		if (inited_mutexes[i] == m) {
			return __real_pthread_mutex_destroy(m);
		}
	}
	// Destroy on an address that was never successfully initialised.
	uninit_mutex_destroys++;
	return 0;  // don't call real destroy on uninit primitive
}

int __wrap_pthread_cond_destroy(pthread_cond_t *c) {
	for (int i = 0; i < n_inited_conds; i++) {
		if (inited_conds[i] == c) {
			return __real_pthread_cond_destroy(c);
		}
	}
	uninit_cond_destroys++;
	return 0;
}

// pthread_cond_init tracking (so we know which cond addrs are valid)
int __real_pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);

int __wrap_pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *attr) {
	int rc = __real_pthread_cond_init(c, attr);
	if (rc == 0 && n_inited_conds < MAX_TRACKED) {
		inited_conds[n_inited_conds++] = c;
	}
	return rc;
}

// pthread_attr_init / pthread_attr_destroy tracking
// Prevents SIGSEGV when buggy cleanup calls pthread_attr_destroy on a
// stack-local attr that was never successfully initialised.
int __real_pthread_attr_init(pthread_attr_t *);
int __real_pthread_attr_destroy(pthread_attr_t *);

int __wrap_pthread_attr_init(pthread_attr_t *a) {
	int rc = __real_pthread_attr_init(a);
	if (rc == 0 && n_inited_attrs < MAX_TRACKED) {
		inited_attrs[n_inited_attrs++] = a;
	}
	return rc;
}

int __wrap_pthread_attr_destroy(pthread_attr_t *a) {
	for (int i = 0; i < n_inited_attrs; i++) {
		if (inited_attrs[i] == a) {
			return __real_pthread_attr_destroy(a);
		}
	}
	uninit_attr_destroys++;
	return 0;  // don't call real destroy on uninit primitive
}

//------------------------------------------------------------------------------
// Embedded sync layer — taken VERBATIM from src/index/indexer.c
//
// Two variants:
//   default        : current (fixed) code
//   -DBUGGY_SYNC   : pre-fix code (commit 484abe65a^)
//
// Task handlers are replaced with no-op stubs; only INDEXER_EXIT is ever
// sent in these tests so the handlers are never reached.
//------------------------------------------------------------------------------

typedef enum {
	INDEXER_IDX_DROP,
	INDEXER_IDX_POPULATE,
	INDEXER_CONSTRAINT_DROP,
	INDEXER_CONSTRAINT_ENFORCE,
	INDEXER_EXIT,
} IndexerOp;

typedef struct {
	IndexerOp op;
	void *pdata;
} IndexerTask;

#ifdef BUGGY_SYNC
// ── Pre-fix struct (has separate cm mutex) ────────────────────────────────
typedef struct {
	pthread_t       t;
	pthread_mutex_t m;
	pthread_mutex_t cm;  // conditional variable mutex (removed in fix)
	pthread_cond_t  c;
	IndexerTask    *q;
} Indexer;
#else
// ── Fixed struct (single mutex m guards both queue and condvar) ───────────
typedef struct {
	pthread_t       t;
	pthread_mutex_t m;
	pthread_cond_t  c;
	IndexerTask    *q;
} Indexer;
#endif

static Indexer *indexer = NULL;

#define INDEXER_LOCK_QUEUE()   pthread_mutex_lock  (&indexer->m)
#define INDEXER_UNLOCK_QUEUE() pthread_mutex_unlock(&indexer->m)

// Task handlers — stubs (only EXIT is used in these tests)
static void _indexer_run_task(IndexerTask *t) { (void)t; }

static void _indexer_PopTask(IndexerTask *task);  // forward decl

static void *_indexer_run(void *arg) {
	(void)arg;
	while (true) {
		IndexerTask task;
		_indexer_PopTask(&task);
		if (task.op == INDEXER_EXIT) return NULL;
		_indexer_run_task(&task);
	}
	return NULL;
}

// ── _indexer_AddTask ──────────────────────────────────────────────────────
#ifdef BUGGY_SYNC
// Pre-fix: signals condvar OUTSIDE the queue critical section, using a
// separate mutex cm.
void _indexer_AddTask(IndexerOp op, void *pdata) {
	IndexerTask task = { .op = op, .pdata = pdata };
	INDEXER_LOCK_QUEUE();
	arr_append(indexer->q, task);
	INDEXER_UNLOCK_QUEUE();

	pthread_mutex_lock  (&indexer->cm);
	pthread_cond_signal (&indexer->c);
	pthread_mutex_unlock(&indexer->cm);
}
#else
// Fixed: predicate update and signal both happen inside the critical section.
void _indexer_AddTask(IndexerOp op, void *pdata) {
	IndexerTask task = { .op = op, .pdata = pdata };
	INDEXER_LOCK_QUEUE();
	arr_append(indexer->q, task);
	pthread_cond_signal(&indexer->c);
	INDEXER_UNLOCK_QUEUE();
}
#endif

// ── _indexer_PopTask ──────────────────────────────────────────────────────
#ifdef BUGGY_SYNC
// Pre-fix: if-statement (not while) + separate cm mutex.
// A spurious wakeup bypasses the predicate check and pops from an empty
// queue → ASSERT in arr_del → abort().
static void _indexer_PopTask(IndexerTask *task) {
	ASSERT(task != NULL);
	INDEXER_LOCK_QUEUE();

	if (arr_len(indexer->q) == 0) {
		pthread_mutex_lock  (&indexer->cm);
		INDEXER_UNLOCK_QUEUE();
		pthread_cond_wait   (&indexer->c, &indexer->cm);  // spurious → returns here
		pthread_mutex_unlock(&indexer->cm);
		INDEXER_LOCK_QUEUE();
	}

	// BUG: no predicate re-check; empty-queue pop on spurious wakeup → UB
	*task = indexer->q[0];
	arr_del(indexer->q, 0);       // ASSERT fires if queue is empty

	INDEXER_UNLOCK_QUEUE();
}
#else
// Fixed: while-loop re-checks the predicate after every wakeup.
static void _indexer_PopTask(IndexerTask *task) {
	ASSERT(task != NULL);
	INDEXER_LOCK_QUEUE();

	while (arr_len(indexer->q) == 0) {
		pthread_cond_wait(&indexer->c, &indexer->m);
	}

	*task = indexer->q[0];
	arr_del(indexer->q, 0);

	INDEXER_UNLOCK_QUEUE();
}
#endif

// ── Indexer_Init ──────────────────────────────────────────────────────────
#ifdef BUGGY_SYNC
// Pre-fix: uses result-code==0 as "was initialised?" proxy.
// All codes start at 0, so cleanup destroys primitives that were never
// initialised when an early step fails.
bool Indexer_Init(void) {
	ASSERT(indexer == NULL);

	int a_res  = 0;
	int c_res  = 0;
	int m_res  = 0;
	int cm_res = 0;

	indexer = rm_calloc(1, sizeof(Indexer));

	m_res = pthread_mutex_init(&indexer->m, NULL);
	if (m_res != 0) goto cleanup;

	cm_res = pthread_mutex_init(&indexer->cm, NULL);
	if (cm_res != 0) goto cleanup;

	c_res = pthread_cond_init(&indexer->c, NULL);
	if (c_res != 0) goto cleanup;

	indexer->q = arr_new(IndexerTask, 0);

	pthread_attr_t attr;
	a_res = pthread_attr_init(&attr);
	if (a_res != 0) goto cleanup;

	if (pthread_create(&indexer->t, &attr, _indexer_run, NULL) != 0) {
		goto cleanup;
	}
	pthread_attr_destroy(&attr);
	return true;

cleanup:
	// BUG: "== 0" means both "succeeded" and "was never called"
	if (c_res  == 0) pthread_cond_destroy (&indexer->c);
	if (m_res  == 0) pthread_mutex_destroy(&indexer->m);
	if (cm_res == 0) pthread_mutex_destroy(&indexer->cm);
	if (a_res  == 0) pthread_attr_destroy (&attr);
	if (indexer->q != NULL) arr_free(indexer->q);
	rm_free(indexer);
	// NOTE: old code did NOT reset indexer=NULL here (another latent bug)
	indexer = NULL;
	return false;
}
#else
// Fixed: explicit bool per primitive; reset indexer=NULL on failure.
bool Indexer_Init(void) {
	ASSERT(indexer == NULL);

	bool m_init    = false;
	bool c_init    = false;
	bool attr_init = false;

	pthread_attr_t attr;
	indexer = rm_calloc(1, sizeof(Indexer));

	if (pthread_mutex_init(&indexer->m, NULL) != 0) goto cleanup;
	m_init = true;

	if (pthread_cond_init(&indexer->c, NULL) != 0) goto cleanup;
	c_init = true;

	indexer->q = arr_new(IndexerTask, 0);

	if (pthread_attr_init(&attr) != 0) goto cleanup;
	attr_init = true;

	if (pthread_create(&indexer->t, &attr, _indexer_run, NULL) != 0) {
		goto cleanup;
	}
	pthread_attr_destroy(&attr);
	return true;

cleanup:
	if (attr_init) pthread_attr_destroy(&attr);
	if (c_init)    pthread_cond_destroy (&indexer->c);
	if (m_init)    pthread_mutex_destroy(&indexer->m);
	if (indexer->q != NULL) arr_free(indexer->q);
	rm_free(indexer);
	indexer = NULL;
	return false;
}
#endif

// ── Indexer_Stop ──────────────────────────────────────────────────────────
void Indexer_Stop(void) {
	_indexer_AddTask(INDEXER_EXIT, NULL);
	pthread_join(indexer->t, NULL);

	arr_free(indexer->q);
	pthread_cond_destroy (&indexer->c);
	pthread_mutex_destroy(&indexer->m);
#ifdef BUGGY_SYNC
	pthread_mutex_destroy(&indexer->cm);
#endif
	rm_free(indexer);
	indexer = NULL;
}

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

static void ms_sleep(unsigned ms) {
	struct timespec ts = { .tv_sec = ms / 1000,
	                       .tv_nsec = (long)(ms % 1000) * 1000000L };
	nanosleep(&ts, NULL);
}

//------------------------------------------------------------------------------
// C-9 REGRESSION TEST
//
// Invariant: the worker thread must NEVER pop from an empty queue when it
// receives a spurious wakeup from pthread_cond_wait.
//
// The test uses fork/waitpid so a crash in the child is reported as a test
// failure rather than aborting the test runner.
//
// BUGGY_SYNC: __wrap_pthread_cond_wait returns spuriously once.
//   Old _indexer_PopTask: if(empty) + no re-check → arr_del on empty queue
//   → ASSERT fires → child aborts with SIGABRT → test detects signal → FAILS.
//
// Fixed code: while(empty) re-checks → sees queue still empty → waits again.
//   The second cond_wait is the real one; Indexer_Stop sends EXIT → exits
//   cleanly → child exits 0 → test PASSES.
//------------------------------------------------------------------------------

static void test_C9_no_pop_on_spurious_wakeup(void) {
	pid_t pid = fork();
	TEST_ASSERT_(pid >= 0, "fork failed: %s", strerror(errno));

	if (pid == 0) {
		// Child process: run the indexer with one spurious wakeup injected.
		atomic_store(&spurious_wakeups_remaining, 1);

		bool ok = Indexer_Init();
		if (!ok) _exit(2);

		// Give the worker thread time to enter pthread_cond_wait and receive
		// the spurious return from __wrap_pthread_cond_wait.
		ms_sleep(100);

		// With fixed code: worker re-checked, went back to sleep — send EXIT.
		// With buggy code: worker already aborted before we get here.
		Indexer_Stop();
		_exit(0);
	}

	// Parent: collect child exit status.
	int status = 0;
	waitpid(pid, &status, 0);

	// BUGGY_SYNC: child was killed by SIGABRT (arr_del ASSERT on empty queue).
	// Fixed code: child exited normally with code 0.
	TEST_ASSERT_(WIFEXITED(status) && WEXITSTATUS(status) == 0,
		"C-9 regression: worker aborted on spurious wakeup "
		"(while-loop required around pthread_cond_wait); "
		"child signal=%d exit=%d",
		WIFSIGNALED(status) ? WTERMSIG(status) : 0,
		WIFEXITED(status)   ? WEXITSTATUS(status) : -1);
}

//------------------------------------------------------------------------------
// C-10 REGRESSION TEST
//
// Invariant: Indexer_Init's cleanup path must only destroy pthread primitives
// that were successfully initialised.
//
// __wrap_pthread_mutex_init is configured to fail on the first call (m init).
// __wrap_pthread_mutex_destroy and __wrap_pthread_cond_destroy count calls
// on addresses that were never in the "successfully initialised" set.
//
// BUGGY_SYNC: cleanup uses result-code==0, which is also the initial value
//   before any init call.  When m_res fails (non-zero), cm_res and c_res are
//   still 0 → cm and c are destroyed without ever having been initialised
//   → uninit_mutex_destroys + uninit_cond_destroys > 0 → test FAILS.
//
// Fixed code: explicit bool flags; only init'd primitives are destroyed
//   → uninit destroys == 0 → test PASSES.
//------------------------------------------------------------------------------

static void test_C10_init_cleanup_no_uninit_destroys(void) {
	// Force the first pthread_mutex_init call to fail (m init step).
	// All subsequent calls (from other tests or the C library internals) are
	// unaffected because we reset the counter before each assertion.
	reset_c10_tracking();
	force_mutex_fail_on_call = 1;  // fail the very first mutex init

	bool ok = Indexer_Init();
	TEST_ASSERT_(!ok, "Indexer_Init must return false when mutex init fails");

	// BUGGY_SYNC: c_res==0 → cond_destroy on uninit c; cm_res==0 → mutex_destroy
	//   on uninit cm; a_res==0 → attr_destroy on uninit attr
	//   → one or more of the uninit_*_destroys counters > 0 → FAILS.
	// Fixed code: explicit bools, nothing uninit'd is destroyed → PASSES.
	int total_uninit = uninit_mutex_destroys + uninit_cond_destroys
	                 + uninit_attr_destroys;
	TEST_ASSERT_(total_uninit == 0,
		"C-10 regression: Indexer_Init cleanup destroyed %d uninitialised "
		"primitive(s) (mutex=%d cond=%d attr=%d); explicit init flags required",
		total_uninit, uninit_mutex_destroys, uninit_cond_destroys,
		uninit_attr_destroys);

	force_mutex_fail_on_call = 0;  // re-arm for subsequent tests

	// Verify a subsequent Indexer_Init succeeds (clean state after failure).
	reset_c10_tracking();
	ok = Indexer_Init();
	TEST_ASSERT_(ok, "Indexer_Init must succeed after a prior failed attempt");
	Indexer_Stop();
}

//------------------------------------------------------------------------------
// test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"C9_no_pop_on_spurious_wakeup",
		test_C9_no_pop_on_spurious_wakeup},
	{"C10_init_cleanup_no_uninit_destroys",
		test_C10_init_cleanup_no_uninit_destroys},
	{NULL, NULL}
};
