/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <pthread.h>

int rwlock_timedwrlock
(
	pthread_rwlock_t *lock,
	int64_t timeout_ms
) ;

int rwlock_timedrdlock
(
	pthread_rwlock_t *lock,
	int64_t timeout_ms
) ;

