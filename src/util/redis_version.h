/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <sys/types.h>

typedef struct {
	uint major;
	uint minor;
	uint patch;
} Redis_Version;

// Returns a Version struct with redis version;
Redis_Version RG_GetRedisVersion();

// Checks if the current redis server version is compliant (grater or equal) to a given version.
bool Redis_Version_GreaterOrEqual(uint major, uint minor, uint patch);
