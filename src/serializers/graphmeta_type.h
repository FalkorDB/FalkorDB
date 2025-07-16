/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "serializers_include.h"

/* The keys from the "graphmeta" type are virtual keys that created upon RDB save
 * and deleted upon RDB or replication process ending.
 * Each for those keys RDB value represent an independent part of the graph that can be encoded/decoded as is.
 * This allows saving and replicating large graphs with small memory overhead */

int GraphMetaType_Register(RedisModuleCtx *ctx);
