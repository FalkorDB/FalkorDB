/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#define GRAPH_ENCODING_LATEST_V  20  // latest RDB encoding version
#define GRAPH_DECODE_MIN_V       14  // min backward compatible version
                                     // (14 = FalkorDB v4.0; versions 10-13 are
                                     // RedisGraph 2.8-2.12 and are no longer
                                     // decodable)
