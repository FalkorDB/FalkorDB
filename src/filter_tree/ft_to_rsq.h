/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "filter_tree.h"
#include "../../deps/RediSearch/src/redisearch_api.h"

// construct a RediSearch query node from filter tree
RSQNode *FilterTreeToQueryNode(FT_FilterNode **none_converted_filters,
		const FT_FilterNode *tree, RSIndex *idx);

