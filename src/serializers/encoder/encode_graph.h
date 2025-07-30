/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../serializers_include.h"

void RdbSaveGraph
(
	RedisModuleIO *rdb,
	void *value
);

