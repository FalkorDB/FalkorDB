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
	int timeout_ms
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
	const int sleep_ns = 1000000; // 1ms backoff
	struct timespec sleep_ts = {0, sleep_ns} ;

	while (1) {
		int rc = pthread_rwlock_trywrlock (lock) ;
		if (rc == 0) {
			return 0 ; // acquired
		}

		if (rc != EBUSY) {
			return rc ; // propagate fatal errors
		}

		// check timeout
		struct timespec now ;
		clock_gettime (CLOCK_REALTIME, &now) ;
		if ((now.tv_sec > ts.tv_sec) ||
			(now.tv_sec == ts.tv_sec && now.tv_nsec >= ts.tv_nsec)) {
			return ETIMEDOUT ;
		}

		nanosleep (&sleep_ts, NULL) ; // brief pause before retry
	}
#else
	// Linux / BSD: native support
	return pthread_rwlock_timedwrlock (lock, &ts) ;
#endif
}

