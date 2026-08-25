/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>

#include "../serializers_include.h"

void RdbSaveGraph
(
	RedisModuleIO *rdb,
	void *value
);

// return the graph encoding version RdbSaveGraph_latest writes
// the graph-offloading module records this in every dump so the matching
// decoder can be selected on load (see Graph_GetDecoder)
uint32_t Graph_EncodingVersion(void);

