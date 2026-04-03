/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <pthread.h>

int rwlock_timedwrlock
(
	pthread_rwlock_t *lock,
	int timeout_ms
) ;

