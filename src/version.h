/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#ifndef REDISGRAPH_MODULE_VERSION

#define REDISGRAPH_VERSION_MAJOR 99
#define REDISGRAPH_VERSION_MINOR 99
#define REDISGRAPH_VERSION_PATCH 99

#define REDISGRAPH_SEMANTIC_VERSION(major, minor, patch) \
  (major * 10000 + minor * 100 + patch)

#define REDISGRAPH_MODULE_VERSION REDISGRAPH_SEMANTIC_VERSION(REDISGRAPH_VERSION_MAJOR, REDISGRAPH_VERSION_MINOR, REDISGRAPH_VERSION_PATCH)

#endif
