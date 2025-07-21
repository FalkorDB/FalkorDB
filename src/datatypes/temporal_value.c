/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include <time.h>
#include "temporal_value.h"

int64_t TemporalValue_NewTimestamp() {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return 1000 * ts.tv_sec + ts.tv_nsec / 1000000;;
}
