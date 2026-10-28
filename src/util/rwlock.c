/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "rwlock.h"

#include <time.h>
#include <errno.h>

// portable wrapper: try to acquire a write lock with timeout
int rwlock_timedwrlock
(
	pthread_rwlock_t *lock,
	int64_t timeout_ms
) {
	// negative: block until acquired
	if (timeout_ms < 0) {
		return pthread_rwlock_wrlock (lock) ;
	}

	// zero: non-blocking attempt;
	if (timeout_ms == 0) {
		return pthread_rwlock_trywrlock (lock) ;
	}

	// positive: compute absolute timeout
	struct timespec ts ;
	clock_gettime (CLOCK_REALTIME, &ts) ;
	ts.tv_sec  += timeout_ms / 1000 ;
	ts.tv_nsec += (timeout_ms % 1000) * 1000000 ;
	if (ts.tv_nsec >= 1000000000L) {
		ts.tv_sec++ ;
		ts.tv_nsec -= 1000000000L ;
	}

#if defined(__APPLE__)
	// macOS does not implement pthread_rwlock_timedwrlock
	const long sleep_ns = 1000000; // 1ms backoff

	while (1) {
		// check the deadline before every attempt so a sleep that crossed it
		// can't acquire the lock and report success after the timeout
		struct timespec now ;
		clock_gettime (CLOCK_REALTIME, &now) ;
		long remaining_ns = (long)(ts.tv_sec - now.tv_sec) * 1000000000L +
			(ts.tv_nsec - now.tv_nsec) ;
		if (remaining_ns <= 0) {
			return ETIMEDOUT ;
		}

		int rc = pthread_rwlock_trywrlock (lock) ;
		if (rc == 0) {
			return 0 ; // acquired
		}

		if (rc != EBUSY) {
			return rc ; // propagate fatal errors
		}

		// cap the backoff to the time left so we never sleep past the deadline
		struct timespec sleep_ts =
			{0, remaining_ns < sleep_ns ? remaining_ns : sleep_ns} ;
		nanosleep (&sleep_ts, NULL) ; // brief pause before retry
	}
#else
	// Linux / BSD: native support
	return pthread_rwlock_timedwrlock (lock, &ts) ;
#endif
}

// portable wrapper: try to acquire a read lock with timeout
int rwlock_timedrdlock
(
	pthread_rwlock_t *lock,
	int64_t timeout_ms
) {
	// negative: block until acquired
	if (timeout_ms < 0) {
		return pthread_rwlock_rdlock (lock) ;
	}

	// zero: non-blocking attempt
	if (timeout_ms == 0) {
		return pthread_rwlock_tryrdlock (lock) ;
	}

	// positive: compute absolute timeout
	struct timespec ts ;
	clock_gettime (CLOCK_REALTIME, &ts) ;
	ts.tv_sec  += timeout_ms / 1000 ;
	ts.tv_nsec += (timeout_ms % 1000) * 1000000 ;
	if (ts.tv_nsec >= 1000000000L) {
		ts.tv_sec++ ;
		ts.tv_nsec -= 1000000000L ;
	}

#if defined(__APPLE__)
	// macOS does not implement pthread_rwlock_timedrdlock
	const long sleep_ns = 1000000; // 1ms backoff

	while (1) {
		// check the deadline before every attempt so a sleep that crossed it
		// can't acquire the lock and report success after the timeout
		struct timespec now ;
		clock_gettime (CLOCK_REALTIME, &now) ;
		long remaining_ns = (long)(ts.tv_sec - now.tv_sec) * 1000000000L +
			(ts.tv_nsec - now.tv_nsec) ;
		if (remaining_ns <= 0) {
			return ETIMEDOUT ;
		}

		int rc = pthread_rwlock_tryrdlock (lock) ;
		if (rc == 0) {
			return 0 ; // acquired
		}

		if (rc != EBUSY) {
			return rc ; // propagate fatal errors
		}

		// cap the backoff to the time left so we never sleep past the deadline
		struct timespec sleep_ts =
			{0, remaining_ns < sleep_ns ? remaining_ns : sleep_ns} ;
		nanosleep (&sleep_ts, NULL) ; // brief pause before retry
	}
#else
	// Linux / BSD: native support
	return pthread_rwlock_timedrdlock (lock, &ts) ;
#endif
}

